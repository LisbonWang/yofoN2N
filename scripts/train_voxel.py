"""Train a volumetric 3D-input / 3D-output model on RAW reconstructions.

This is the 3D counterpart to ``scripts/train_sparse_dense.py``.  It
deliberately does not touch the 2D script: the two entry points can be
maintained independently and a config flag (``model.arch``) routes
between them.

Behavioural differences vs. the 2D script:

* Model: ``UNet3D`` (depthwise-separable, default ``base_channels=8``,
  ``depth=3``).  Pooling is fixed: every stage downsamples D/H/W by 2 and
  the decoder mirrors it back, so the model maps ``[B, C, D, H, W]`` to a
  same-D-length ``[B, C, D, H, W]`` volume.  The 5D prediction is matched
  directly against the 5D target that :class:`N2N3DDataset` returns, so
  the existing :func:`cbct_n2n.training.trainer.fit` L1-loss path works
  unchanged (no centre-slice head).  Falls back to ``UNet2D`` if the
  config explicitly sets ``model.arch: unet2d``.
* Dataset: :class:`N2N3DDataset` returns a 5D source and a 5D target over
  the same D-slice window.  ``voxel_depth`` is read from the config
  (``data.voxel_depth``, default 16); it must satisfy
  ``>= 2 ** model.depth`` for the fixed 2x pooling.  ``stride`` (default 4)
  is the step between consecutive window starts along D.
* All other machinery -- manifest parsing, mappings, residual tags,
  checkpoints, ``fit()``, scheduler, AMP -- is reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader

from cbct_n2n.config import load_config
from cbct_n2n.data.n2n3d_dataset import N2N3DDataset
from cbct_n2n.data.sampler import GroupedShuffleSampler
from cbct_n2n.models.unet3d import UNet3D
from cbct_n2n.models.unet import UNet2D
from cbct_n2n.models.unet2d_ds import UNet2D_DS
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.trainer import fit, seed_everything


def _worker_init_fn(worker_id: int) -> None:
    """Pin each DataLoader worker to a single torch/OMP thread.

    Without this every worker spawns its own full thread pool
    (``num_workers`` x threads) and oversubscribes the CPU, which makes
    loading slower.  Data loading here (memmap read + augment) is
    single-threaded work, so one thread per worker is the right setting.
    """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def build_model(arch: str, model_cfg: dict) -> nn.Module:
    """Construct the requested architecture from the ``model`` config block.

    The ``arch`` key is the dispatch selector and is dropped before forwarding
    to the constructor.  UNet3D maps ``[B, C, D, H, W]`` -> ``[B, C, D, H, W]``
    (same D length), so no extra head is needed.
    """
    cfg = {k: v for k, v in model_cfg.items() if k != "arch"}
    if arch == "unet3d":
        return UNet3D(**cfg)
    if arch == "unet2d":
        return UNet2D(**cfg)
    if arch == "unet2d_ds":
        return UNet2D_DS(**cfg)
    raise ValueError(f"Unknown model.arch {arch!r}; expected one of: unet2d, unet2d_ds, unet3d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a 3D-input / 3D-output model on RAW volumes.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Task YAML config; no dataset-specific config is assumed.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-weights", type=Path,
                        help="Load model weights only (cold optimizer/scheduler).")
    parser.add_argument("--arch", choices=("unet3d", "unet2d", "unet2d_ds"), default=None,
                        help="Override model.arch from the config.")
    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg, train_cfg = config["data"], config["training"]
    seed_everything(int(data_cfg.get("seed", 42)))

    arch = args.arch or config.get("model", {}).get("arch", "unet3d")

    mappings = data_cfg.get("mappings")
    residuals = data_cfg.get("residuals")
    supervision_mode = str(data_cfg.get("supervision_mode", "full_view"))
    voxel_depth = int(data_cfg.get("voxel_depth", 16))
    stride = int(data_cfg.get("stride", 4))
    patch_size = data_cfg.get("patch_size")
    if patch_size is not None:
        patch_size = int(patch_size)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    common_kwargs = dict(
        mappings=mappings, residuals=residuals, supervision_mode=supervision_mode,
    )
    train_data = N2N3DDataset(manifest, args.manifest, "train", voxel_depth=voxel_depth,
                              stride=stride, patch_size=patch_size,
                              augment=bool(data_cfg.get("augment", True)),
                              **common_kwargs)
    val_data = N2N3DDataset(manifest, args.manifest, "val", voxel_depth=voxel_depth,
                            stride=stride, patch_size=patch_size, augment=False,
                            **common_kwargs)
    num_workers = int(train_cfg.get("num_workers", 0))
    loader_args = {"batch_size": int(train_cfg["batch_size"]),
                   "num_workers": num_workers,
                   "pin_memory": True}
    if num_workers > 0:
        loader_args["persistent_workers"] = True
        loader_args["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
        loader_args["worker_init_fn"] = _worker_init_fn
    # Two-level shuffle: shuffle studies first, then samples within each
    # study, so consecutive batches stay inside one study (no raw-file
    # jumping within a local window of time).
    train_sampler = GroupedShuffleSampler(train_data.group_ids)
    train_loader = DataLoader(train_data, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)

    model = build_model(arch, config["model"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[arch] {arch} | params={n_params:,} | voxel_depth={voxel_depth} | "
          f"stride={stride} | patch_size={patch_size} | data shape: source=5D target=5D", flush=True)
    requested = str(train_cfg.get("device", "cuda"))
    device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=float(train_cfg["learning_rate"]),
                                   weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    start_epoch = 0
    if args.init_weights:
        if args.resume:
            parser.error("--init-weights and --resume are mutually exclusive")
        state = load_checkpoint(args.init_weights, model, map_location=device)
        print(f"[init-weights] loaded weights from {args.init_weights} "
              f"(epoch {state.get('epoch')}); optimizer/scheduler cold-started", flush=True)
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = int(state["epoch"]) + 1

    run_dir = args.run_dir or Path(train_cfg["run_dir"])
    metrics = fit(model, train_loader, val_loader, optimizer, scheduler, device, run_dir, config,
                  start_epoch=start_epoch)
    print(f"Finished on {device}: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
