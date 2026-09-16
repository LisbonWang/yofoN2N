"""Denoise a single headerless float32 RAW CBCT volume with a trained model.

Supports both the 2D depthwise-separable UNet (``unet2d_ds``) and the 3D UNet
(``unet3d``).  The volume is fed directly to the network in native attenuation
space (no normalization), matching the training pipeline.  H/W is padded to a
multiple of 32 internally; the 3D path slides a non-overlapping
``voxel_depth``-slice window along z (mirroring training-time placement).

Examples:
  python scripts/denoise_single_raw.py \
    --input vol.raw --output vol_dn_2d.raw \
    --shape 520 680 680 --arch unet2d_ds \
    --checkpoint runs/1609v2_n2n_2d_ds_h2f_dn_2d_r1/best.pt

  python scripts/denoise_single_raw.py \
    --input vol.raw --output vol_dn_3d.raw \
    --shape 520 680 680 --arch unet3d \
    --checkpoint runs/1609v2_n2n_3d_h2f_v2_dn_3d_r1/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cbct_n2n.models import UNet2D_DS, UNet3D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import (
    infer_volume,
    infer_volume_3d,
    infer_volume_3d_tiled,
    write_raw_volume,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Denoise a single float32 RAW CBCT volume.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("Z", "Y", "X"), required=True)
    parser.add_argument("--arch", choices=("unet2d_ds", "unet3d"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--voxel-depth", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tiled", action="store_true",
                        help="Use overlapping-tile 3D inference (H/W + D tiles with ramp fusion).")
    parser.add_argument("--patch-hw", type=int, default=256,
                        help="H/W tile size for --tiled (default 256).")
    parser.add_argument("--overlap-d", type=int, default=16,
                        help="D-axis tile overlap for --tiled (default 16 slices).")
    parser.add_argument("--overlap-hw", type=int, default=16,
                        help="H/W tile overlap for --tiled (default 16 pixels).")
    args = parser.parse_args()

    shape = tuple(args.shape)
    expected = int(np.prod(shape)) * np.dtype("<f4").itemsize
    if args.input.stat().st_size != expected:
        raise ValueError(
            f"Input size {args.input.stat().st_size} != {expected} bytes for shape {shape} float32 RAW."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(f"[warn] CUDA unavailable, falling back to CPU", flush=True)

    model_kwargs = dict(
        in_channels=1, out_channels=1,
        base_channels=args.base_channels, depth=args.depth, residual=True,
    )
    if args.arch == "unet2d_ds":
        model = UNet2D_DS(**model_kwargs).to(device)
    else:
        model = UNet3D(**model_kwargs).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[infer] arch={args.arch} params={n_params:,} device={device} "
          f"input={shape} checkpoint={args.checkpoint}", flush=True)

    source = np.memmap(args.input, dtype="<f4", mode="r", shape=shape)
    with torch.no_grad():
        if args.arch == "unet2d_ds":
            prediction = infer_volume(model, np.asarray(source), device, args.batch_size)
        elif args.tiled:
            prediction = infer_volume_3d_tiled(
                model, np.asarray(source), args.voxel_depth, args.patch_hw,
                args.overlap_d, args.overlap_hw, device, args.batch_size,
            )
        else:
            prediction = infer_volume_3d(model, np.asarray(source), args.voxel_depth,
                                         device, args.batch_size)

    write_raw_volume(args.output, prediction)
    print(f"[done] output={args.output} shape={prediction.shape} "
          f"bytes={args.output.stat().st_size} "
          f"min={float(prediction.min()):.5f} max={float(prediction.max()):.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
