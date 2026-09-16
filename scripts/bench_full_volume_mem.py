"""Measure peak GPU memory of a *whole-volume* UNet3D forward pass.

The current pipeline feeds the model 16-slice D-windows; this benchmark
answers "what if we feed the entire 360-slice volume at once?".  Peak
activation memory is dominated by the finest encoder/decoder level
(full H x W, `base_channels` wide), so it scales ~linearly with D.  We
measure a few D values here and extrapolate to D=360.

Usage:
  python scripts/bench_full_volume_mem.py --base 16 --depth 4 \
      --d-list 16,32,64,96 --dtype fp32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cbct_n2n.models.unet3d import UNet3D  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--d-list", type=str, default="16,32,64,96")
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    model = UNet3D(base_channels=args.base, depth=args.depth, residual=True).to(dev)
    if args.dtype == "fp16":
        model = model.half()
    model.eval()
    dt = torch.float16 if args.dtype == "fp16" else torch.float32
    n_params = sum(p.numel() for p in model.parameters())

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                "arch": f"UNet3D base{args.base}/depth{args.depth}",
                "params": n_params,
                "dtype": args.dtype,
                "h": args.h,
                "w": args.w,
            }
        ),
        flush=True,
    )

    for d in [int(v) for v in args.d_list.split(",")]:
        rec: dict[str, object] = {"d": d}
        try:
            x = torch.randn(1, 1, d, args.h, args.w, device=dev, dtype=dt)
            with torch.no_grad():
                for _ in range(args.warmup):
                    model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(args.repeat):
                    y = model(x)
            torch.cuda.synchronize()
            rec["sec_per_forward"] = round((time.perf_counter() - t0) / args.repeat, 3)
            rec["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
            assert y.shape == x.shape
            del x, y
        except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
            rec["oom"] = True
            rec["error"] = str(exc).splitlines()[0][:120]
        torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
