from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from cbct_n2n.config import load_config
from cbct_n2n.data.dataset import PairedSliceDataset
from cbct_n2n.data.manifest import load_manifest
from cbct_n2n.models import UNet2D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.trainer import fit, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="Train full-slice 2D UNet Noise2Noise model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg, train_cfg = config["data"], config["training"]
    seed_everything(int(data_cfg["seed"]))
    manifest = load_manifest(args.manifest or data_cfg["manifest"])
    max_slices = data_cfg.get("max_slices_per_case")
    train_data = PairedSliceDataset(manifest, "train", augment=bool(data_cfg.get("augment", True)), max_slices_per_case=max_slices)
    val_data = PairedSliceDataset(manifest, "val", augment=False, max_slices_per_case=max_slices)
    loader_args = {"batch_size": int(train_cfg["batch_size"]), "num_workers": int(train_cfg["num_workers"]), "pin_memory": True}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    model = UNet2D(**config["model"])
    requested = str(train_cfg.get("device", "cuda"))
    device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    start_epoch = 0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = int(state["epoch"]) + 1
    run_dir = args.run_dir or Path(train_cfg["run_dir"])
    metrics = fit(model, train_loader, val_loader, optimizer, scheduler, device, run_dir, config, start_epoch=start_epoch)
    print(f"Finished on {device}: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
