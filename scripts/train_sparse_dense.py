from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from cbct_n2n.config import load_config
from cbct_n2n.data.n2n_dataset import N2NDataset
from cbct_n2n.models import UNet2D, UNet2D_DS
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.trainer import fit, seed_everything


def build_model(arch: str, model_cfg: dict) -> "torch.nn.Module":
    """Construct the requested 2D architecture from the ``model`` config block.

    ``arch`` is the dispatch selector and is dropped before forwarding to the
    constructor (neither model accepts an ``arch`` kwarg).  Default ``unet2d``
    preserves the historical behaviour of this script.
    """
    cfg = {k: v for k, v in model_cfg.items() if k != "arch"}
    if arch == "unet2d":
        return UNet2D(**cfg)
    if arch == "unet2d_ds":
        return UNet2D_DS(**cfg)
    raise ValueError(f"Unknown model.arch {arch!r}; expected one of: unet2d, unet2d_ds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train sparse-angle to full-view 2D UNet from RAW volumes.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Task YAML config; no dataset-specific config is assumed.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-weights", type=Path,
                        help="Load model weights only (cold optimizer/scheduler, start from epoch 0). "
                             "Use for fine-tune-style plans whose supervision target is derived from a "
                             "previous converged run (e.g. teacher-denoised targets).")
    parser.add_argument("--arch", choices=("unet2d", "unet2d_ds"), default=None,
                        help="Override model.arch from the config (default: unet2d).")
    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg, train_cfg = config["data"], config["training"]
    seed_everything(int(data_cfg.get("seed", 42)))
    # N2NDataset: explicit mapping directions take precedence; a legacy
    # supervision_mode is translated when no mappings are given (default
    # full_view keeps the historical RawPairedSliceDataset behaviour).
    mappings = data_cfg.get("mappings")
    residuals = data_cfg.get("residuals")
    supervision_mode = str(data_cfg.get("supervision_mode", "full_view"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_data = N2NDataset(manifest, args.manifest, "train",
                            mappings=mappings, residuals=residuals, supervision_mode=supervision_mode,
                            augment=bool(data_cfg.get("augment", True)))
    val_data = N2NDataset(manifest, args.manifest, "val",
                          mappings=mappings, residuals=residuals, supervision_mode=supervision_mode,
                          augment=False)
    loader_args = {"batch_size": int(train_cfg["batch_size"]), "num_workers": int(train_cfg.get("num_workers", 0)), "pin_memory": True}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    arch = args.arch or str(config.get("model", {}).get("arch", "unet2d"))
    model = build_model(arch, config["model"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[arch] {arch} | params={n_params:,}", flush=True)
    requested = str(train_cfg.get("device", "cuda"))
    device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    start_epoch = 0
    if args.init_weights:
        if args.resume:
            parser.error("--init-weights and --resume are mutually exclusive")
        state = load_checkpoint(args.init_weights, model, map_location=device)
        print(f"[init-weights] loaded model weights only from {args.init_weights} "
              f"(teacher epoch {state.get('epoch')}, val_l1 {state.get('metrics', {}).get('val_loss', float('nan')):.6f}); "
              "optimizer/scheduler cold-started, training from epoch 0")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = int(state["epoch"]) + 1
    run_dir = args.run_dir or Path(train_cfg["run_dir"])
    metrics = fit(model, train_loader, val_loader, optimizer, scheduler, device, run_dir, config, start_epoch=start_epoch)
    print(f"Finished on {device}: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
