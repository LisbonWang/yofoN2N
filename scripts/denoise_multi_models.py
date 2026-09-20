"""Denoise one RAW volume with several checkpoints, suffixing the outputs by model.

Why this exists
---------------
Judging which recursive round to ship means looking at the *same* input volume
denoised by every candidate model.  Running ``denoise_single_raw.py`` once per
model would re-read the volume N times, leave N unrelated commands in the
history, and give no guarantee that all models saw identical input.  This
driver reads the volume once, runs every checkpoint on the identical in-memory
array through the deployment inference path (overlapping-tile 3D + trapezoid
fusion), and writes the results side by side as::

    <stem>.<model>.raw

e.g. ``full.bhc.raw`` -> ``full.bhc.r0.raw``, ``full.bhc.r1b.raw``, ...

so a viewer can flip between them by suffix alone.

It also attaches the numbers that drive the delivery decision: the
density-banded Immerkaer sigma of every output, keyed on one chosen model with
the same convention as ``eval_rounds_banded.py`` (``--band-key``/``--lo``/
``--hi``), plus the per-model global sigma and reduction versus the input.

Examples
--------
    python scripts/denoise_multi_models.py \\
        --input outputs/judge/full.bhc.raw --shape 360 640 640 \\
        --config configs/air3_mode2_1609_unet3d_h2f_128b_r4.yaml \\
        --model r0=outputs/judge/checkpoints/r0.best.pt \\
        --model r1b=outputs/judge/checkpoints/r1b.best.pt \\
        --band-key r1b --summary outputs/judge/summary.json

Existing ``<stem>.<model>.raw`` files are skipped unless ``--force`` is given,
so an interrupted run can simply be repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cbct_n2n.config import load_config
from cbct_n2n.models import UNet2D, UNet2D_DS, UNet3D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import (
    infer_volume,
    infer_volume_3d,
    infer_volume_3d_tiled,
    write_raw_volume,
)
from cbct_n2n.training.metrics import band_noise_std_immerkaer, estimate_noise_std_immerkaer

DTYPE = np.dtype("<f4")
TISSUE_FLOOR = 0.01  # below this mu a voxel is air and is excluded from every band
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def build_model(arch: str, model_cfg: dict) -> "torch.nn.Module":
    cfg = {k: v for k, v in model_cfg.items() if k != "arch"}
    if arch == "unet2d":
        return UNet2D(**cfg)
    if arch == "unet2d_ds":
        return UNet2D_DS(**cfg)
    if arch == "unet3d":
        return UNet3D(**cfg)
    raise ValueError(f"unknown model.arch {arch!r}; expected unet2d | unet2d_ds | unet3d")


def parse_model_spec(spec: str) -> tuple[str, Path]:
    """Split ``NAME=PATH`` and reject names that cannot be a filename suffix."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--model expects NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    name, path = name.strip(), path.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--model has an empty name: {spec!r}")
    if not SAFE_NAME.match(name):
        raise argparse.ArgumentTypeError(
            f"--model name {name!r} must match {SAFE_NAME.pattern} (it becomes a filename suffix)"
        )
    if not path:
        raise argparse.ArgumentTypeError(f"--model has an empty path: {spec!r}")
    return name, Path(path)


def output_path_for(input_path: Path, name: str, outdir: Path | None = None) -> Path:
    """``full.bhc.raw`` + ``r1b`` -> ``<dir>/full.bhc.r1b.raw``.

    ``Path.stem`` of ``full.bhc.raw`` is ``full.bhc``, so multi-dot domains keep
    their full logical name and only the model tag is appended.
    """
    directory = outdir if outdir is not None else input_path.parent
    return directory / f"{input_path.stem}.{name}.raw"


def expected_bytes(shape: tuple[int, int, int]) -> int:
    return int(np.prod(shape)) * DTYPE.itemsize


def file_md5(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def band_masks(key_volume: np.ndarray, lo: float, hi: float) -> dict[str, np.ndarray]:
    return {
        "low": (key_volume > TISSUE_FLOOR) & (key_volume <= lo),
        "mid": (key_volume > lo) & (key_volume < hi),
        "high": key_volume >= hi,
    }


def severity_stats(volume: np.ndarray) -> dict[str, float]:
    return {
        "min": float(volume.min()),
        "max": float(volume.max()),
        "mean": float(volume.mean()),
        "std": float(volume.std()),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Denoise one RAW volume with several checkpoints (suffixed outputs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Headerless float32 RAW volume.")
    p.add_argument("--shape", type=int, nargs=3, metavar=("Z", "Y", "X"), required=True)
    p.add_argument("--config", type=Path, required=True,
                   help="Config supplying the model block and voxel_depth.")
    p.add_argument("--model", type=parse_model_spec, action="append", required=True,
                   metavar="NAME=PATH", help="Repeatable; order defines the report order.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Where to write outputs (default: alongside --input).")
    p.add_argument("--arch", default=None, choices=("unet2d", "unet2d_ds", "unet3d"),
                   help="Override model.arch from the config.")
    p.add_argument("--tiled", action="store_true", default=True,
                   help="Overlapping-tile 3D inference (default).")
    p.add_argument("--no-tiled", dest="tiled", action="store_false")
    p.add_argument("--voxel-depth", type=int, default=None,
                   help="Tile depth (default: config data.voxel_depth).")
    p.add_argument("--patch-hw", type=int, default=256)
    p.add_argument("--overlap-d", type=int, default=16)
    p.add_argument("--overlap-hw", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--band-key", default=None,
                   help="Model whose output keys the density bands (default: 2nd model).")
    p.add_argument("--lo", type=float, default=0.2, help="Low/mid band edge in mu (default 0.2).")
    p.add_argument("--hi", type=float, default=0.4, help="Mid/high band edge in mu (default 0.4).")
    p.add_argument("--max-slices", type=int, default=24,
                   help="Slices sampled per banded/global sigma (default 24).")
    p.add_argument("--summary", type=Path, default=None, help="Write a JSON summary here.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if <stem>.<model>.raw already exists.")
    args = p.parse_args()

    shape = tuple(int(v) for v in args.shape)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    actual = args.input.stat().st_size
    expected = expected_bytes(shape)
    if actual != expected:
        raise SystemExit(
            f"input size {actual} != {expected} bytes for shape {shape} float32 RAW"
        )

    names = [n for n, _ in args.model]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate --model names in {names}; outputs would collide")

    config = load_config(args.config)
    arch = args.arch or str(config.get("model", {}).get("arch", "unet3d"))
    voxel_depth = int(args.voxel_depth or config["data"].get("voxel_depth", 16))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[warn] {args.device} unavailable, falling back to CPU", flush=True)

    band_key = args.band_key or (names[1] if len(names) > 1 else names[0])
    if band_key not in names:
        raise SystemExit(f"--band-key {band_key!r} is not one of {names}")

    outdir = args.outdir if args.outdir is not None else args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] arch={arch} voxel_depth={voxel_depth} tiled={args.tiled} "
          f"patch_hw={args.patch_hw} overlap={args.overlap_d}/{args.overlap_hw} "
          f"batch={args.batch_size} device={device}", flush=True)
    print(f"[io] input={args.input} shape={shape} ({actual / 1024**2:.0f} MiB) "
          f"outdir={outdir} band_key={band_key} lo={args.lo} hi={args.hi}", flush=True)

    source = np.asarray(np.memmap(args.input, dtype=DTYPE, mode="r", shape=shape))
    full = source

    results: dict[str, dict[str, Any]] = {}
    outputs: dict[str, np.ndarray] = {}
    loaded: dict[str, "torch.nn.Module"] = {}

    for name, ckpt in args.model:
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found for {name}: {ckpt}")
        out_path = output_path_for(args.input, name, outdir)
        entry: dict[str, Any] = {
            "checkpoint": str(ckpt),
            "checkpoint_md5": file_md5(ckpt),
            "output": str(out_path),
        }

        if out_path.exists() and not args.force:
            size_ok = out_path.stat().st_size == expected
            entry["status"] = "cached" if size_ok else "cached-size-mismatch"
            print(f"[skip] {name}: {out_path.name} exists ({entry['status']})", flush=True)
            outputs[name] = np.asarray(np.memmap(out_path, dtype=DTYPE, mode="r", shape=shape))
        else:
            model = build_model(arch, config["model"]).to(device)
            load_checkpoint(str(ckpt), model, map_location=device)
            model.eval()
            loaded[name] = model
            t0 = time.time()
            with torch.no_grad():
                if arch == "unet3d":
                    if args.tiled:
                        prediction = infer_volume_3d_tiled(
                            model, full, voxel_depth, args.patch_hw,
                            args.overlap_d, args.overlap_hw, device, args.batch_size,
                        )
                    else:
                        prediction = infer_volume_3d(
                            model, full, voxel_depth, device, args.batch_size)
                else:
                    prediction = infer_volume(model, full, device, args.batch_size)
            elapsed = time.time() - t0
            if tuple(prediction.shape) != shape:
                raise SystemExit(
                    f"{name}: prediction shape {prediction.shape} != requested {shape}"
                )
            write_raw_volume(out_path, prediction)
            written = out_path.stat().st_size
            if written != expected:
                raise SystemExit(f"{name}: wrote {written} bytes, expected {expected}")
            entry["status"] = "done"
            entry["seconds"] = round(elapsed, 1)
            entry["bytes"] = written
            outputs[name] = np.asarray(prediction)
            print(f"[done] {name} -> {out_path.name} in {elapsed:.1f}s "
                  f"(min={prediction.min():.5f} max={prediction.max():.5f})", flush=True)

        results[name] = entry

    key_volume = outputs[band_key]
    masks = band_masks(key_volume, args.lo, args.hi)
    band_n_voxels = {b: int(np.count_nonzero(m[1:-1, 1:-1])) for b, m in masks.items()}

    full_global = estimate_noise_std_immerkaer(full, max_slices=args.max_slices)
    # Tissue mask is fixed by the *input*, so every model is scored on the same voxels.
    # accumulate in float64 but only over the selected voxels -- casting whole volumes to
    # float64 would cost ~1.2 GiB each for no benefit.
    tissue = np.asarray(full) > TISSUE_FLOOR
    tissue_sum = float(np.sum(np.asarray(full)[tissue], dtype=np.float64))
    tissue_n = int(np.count_nonzero(tissue))

    entries = list(outputs.items()) + [("full", full)]
    for name, volume in entries:
        banded = {
            band: band_noise_std_immerkaer(volume, mask, max_slices=args.max_slices)
            for band, mask in masks.items()
        }
        global_sigma = estimate_noise_std_immerkaer(volume, max_slices=args.max_slices)
        row = results.setdefault(name, {"status": "input"})
        row["global_sigma"] = global_sigma
        row["bands"] = banded
        row["band_differentiation_high_over_low"] = (
            float(banded["high"] / banded["low"]) if banded["low"] else float("nan")
        )
        row["sigma_reduction_vs_input_pct"] = (
            float(100.0 * (1.0 - global_sigma / full_global)) if full_global else float("nan")
        )
        if tissue_n:
            vol_mean = float(np.sum(np.asarray(volume)[tissue], dtype=np.float64)) / tissue_n
            row["mean_offset_on_tissue"] = vol_mean - tissue_sum / tissue_n
        else:
            row["mean_offset_on_tissue"] = float("nan")
        row.update({f"volume_{k}": v for k, v in severity_stats(volume).items()})

    print("\n[bands] mean Immerkaer sigma (lo=%.2f hi=%.2f keyed on %s, tissue>%.3f)"
          % (args.lo, args.hi, band_key, TISSUE_FLOOR), flush=True)
    header = "band".ljust(8) + "".join(n.rjust(11) for n in names) + "input".rjust(11)
    print(header, flush=True)
    for band in ("low", "mid", "high"):
        line = band.ljust(8) + "".join(f"{results[n]['bands'][band]:.5f}".rjust(11) for n in names)
        line += f"{results['full']['bands'][band]:.5f}".rjust(11)
        print(line, flush=True)
    print("global".ljust(8) + "".join(f"{results[n]['global_sigma']:.5f}".rjust(11) for n in names)
          + f"{full_global:.5f}".rjust(11), flush=True)
    print("diff".ljust(8) + "".join(
        f"{results[n]['band_differentiation_high_over_low']:.4f}".rjust(11) for n in names)
        + f"{results['full']['band_differentiation_high_over_low']:.4f}".rjust(11), flush=True)
    print("cut%".ljust(8) + "".join(
        f"{results[n]['sigma_reduction_vs_input_pct']:.1f}".rjust(11) for n in names)
        + "0.0".rjust(11), flush=True)

    if args.summary is not None:
        payload = {
            "input": {
                "path": str(args.input),
                "md5": file_md5(args.input),
                "shape": list(shape),
                "bytes": actual,
                "dtype": str(DTYPE),
            },
            "config": str(args.config),
            "arch": arch,
            "voxel_depth": voxel_depth,
            "tiled": {
                "enabled": bool(args.tiled), "patch_hw": args.patch_hw,
                "overlap_d": args.overlap_d, "overlap_hw": args.overlap_hw,
                "batch_size": args.batch_size,
            },
            "device": str(device),
            "bands": {
                "key": band_key, "lo": args.lo, "hi": args.hi,
                "tissue_floor": TISSUE_FLOOR, "n_voxels": band_n_voxels,
            },
            "models": names,
            "results": {n: results[n] for n in names},
            "input_stats": results["full"],
            "note": (
                "Immerkaer sigma measures high-frequency energy; it cannot tell noise "
                "from texture. 'diff' = sigma(high band)/sigma(low band): <1 means the "
                "model is gentler on soft tissue than on bone (density-adaptive), "
                "1.0 means bone and soft tissue are smoothed equally hard."
            ),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[summary] {args.summary}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
