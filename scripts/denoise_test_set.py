"""Copy a test-set slice of volumes locally and denoise them with a trained model.

Two clearly separated output folders under ``--out-root``:

* ``inputs/<manifest-relative-path>``   -- verbatim copies of the test volumes
  that are fed to the model (so the evaluated input is preserved next to the
  result);
* ``denoised/<manifest-relative-path>`` -- the model prediction, same shape and
  dtype (``<f4``), named by appending ``_denoised`` before the extension
  (``full.raw`` -> ``full_denoised.raw``).

Both phases are idempotent: a copy is skipped when the destination already has
the expected byte size, and inference is skipped when the output already exists
with the expected size.  Re-running therefore resumes instead of redoing work.

Volumes are processed exactly as in the training pipeline: native attenuation
space (no normalization), H/W padded to a multiple of 32 internally, and the 3D
UNet applied through overlapping H/W + D tiles with trapezoid (ramp) fusion.

Example:
  python scripts/denoise_test_set.py \\
    --manifest /path/to/dataset/manifest.json \\
    --root /path/to/dataset \\
    --out-root outputs/test-set \\
    --checkpoint runs/EXPERIMENT/best.pt \\
    --domains full.raw,full.bhc.raw,full.marfi.bhc.raw \\
    --voxel-depth 128 --patch-hw 256 --overlap-d 16 --overlap-hw 16
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cbct_n2n.models import UNet3D
from cbct_n2n.training.checkpoint import load_checkpoint
from cbct_n2n.training.inference import infer_volume_3d_tiled, write_raw_volume
from cbct_n2n.training.metrics import estimate_noise_std_immerkaer

DTYPE = np.dtype("<f4")
DENOISED_SUFFIX = "_denoised"


def expected_bytes(shape: tuple[int, int, int]) -> int:
    return int(np.prod(shape)) * DTYPE.itemsize


def denoised_name(name: str) -> str:
    """``full.raw`` -> ``full_denoised.raw``; keeps multi-suffix names intact."""
    path = Path(name)
    return f"{path.stem}{DENOISED_SUFFIX}{path.suffix}"


def copy_input(source: Path, destination: Path, shape: tuple[int, int, int]) -> str:
    """Copy one volume, verifying the declared shape.  Returns a status word."""
    size = expected_bytes(shape)
    if not source.exists():
        raise FileNotFoundError(f"missing source volume: {source}")
    actual = source.stat().st_size
    if actual != size:
        raise ValueError(
            f"{source}: {actual} bytes != {size} for shape {tuple(shape)} float32; "
            "manifest and on-disk volume disagree"
        )
    if destination.exists() and destination.stat().st_size == size:
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size != size:
        raise ValueError(f"{destination}: copy is incomplete")
    return "copied"


def volume_stats(volume: np.ndarray) -> dict[str, float]:
    data = np.asarray(volume, dtype=np.float32)
    step = max(1, data.shape[0] // 32)
    sigma = estimate_noise_std_immerkaer(np.asarray(data[::step], dtype=np.float32))
    return {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "sigma_immerkaer": float(sigma),
        "finite": bool(np.isfinite(data).all()),
    }


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = UNet3D(
        in_channels=1, out_channels=1,
        base_channels=args.base_channels, depth=args.depth, residual=True,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="split manifest listing the test records")
    parser.add_argument("--root", type=Path, required=True,
                        help="root the manifest's relative output paths resolve against")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--domains", default="full.raw,full.bhc.raw,full.marfi.bhc.raw",
                        help="comma-separated output filenames to denoise")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--voxel-depth", type=int, default=128)
    parser.add_argument("--patch-hw", type=int, default=256)
    parser.add_argument("--overlap-d", type=int, default=16)
    parser.add_argument("--overlap-hw", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true",
                        help="re-run inference over outputs that already exist")
    parser.add_argument("--copy-only", action="store_true")
    args = parser.parse_args()

    domains = [name.strip() for name in args.domains.split(",") if name.strip()]
    if not domains:
        raise ValueError("--domains must list at least one output filename")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if not records:
        raise ValueError(f"{args.manifest} has no records")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warn] CUDA unavailable, falling back to CPU", flush=True)

    inputs_root = args.out_root / "inputs"
    denoised_root = args.out_root / "denoised"

    jobs: list[dict] = []
    for record in records:
        shape = tuple(int(v) for v in record["shape"])
        for domain in domains:
            relative = record["outputs"].get(domain)
            if relative is None:
                raise KeyError(
                    f"{record['key']} has no output entry for {domain!r}; "
                    f"available: {sorted(record['outputs'])}"
                )
            jobs.append({
                "key": record["key"],
                "group": record["group"],
                "case_id": record["case_id"],
                "study_uid": record["study_uid"],
                "domain": domain,
                "shape": shape,
                "source": args.root / relative,
                "input": inputs_root / relative,
                "output": denoised_root / Path(relative).with_name(denoised_name(domain)),
            })

    print(f"[plan] records={len(records)} domains={domains} jobs={len(jobs)} "
          f"device={device} checkpoint={args.checkpoint}", flush=True)
    print(f"[plan] inputs ->  {inputs_root}", flush=True)
    print(f"[plan] denoised -> {denoised_root}", flush=True)

    # ---- phase 1: stage the inputs -------------------------------------
    print("\n[phase 1] staging inputs", flush=True)
    staged = {"copied": 0, "cached": 0}
    for index, job in enumerate(jobs, start=1):
        status = copy_input(job["source"], job["input"], job["shape"])
        staged[status] += 1
        print(f"  [{index}/{len(jobs)}] {status:6s} {job['group']:12s} "
              f"{job['domain']:22s} {job['input'].name}", flush=True)
    total_bytes = sum(expected_bytes(job["shape"]) for job in jobs)
    print(f"[phase 1] done: {staged['copied']} copied, {staged['cached']} already present, "
          f"{total_bytes / 2**30:.2f} GiB staged", flush=True)

    if args.copy_only:
        return 0

    # ---- phase 2: denoise ---------------------------------------------
    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[phase 2] denoising | params={n_params:,} | tiles "
          f"{args.voxel_depth}x{args.patch_hw}x{args.patch_hw} "
          f"overlap d={args.overlap_d} hw={args.overlap_hw} batch={args.batch_size}",
          flush=True)

    summary: list[dict] = []
    for index, job in enumerate(jobs, start=1):
        shape = job["shape"]
        size = expected_bytes(shape)
        label = f"{job['group']} {job['case_id']} {job['domain']}"
        if job["output"].exists() and job["output"].stat().st_size == size and not args.force:
            existing = np.memmap(job["output"], dtype=DTYPE, mode="r", shape=shape)
            entry = {"key": job["key"], "group": job["group"], "case_id": job["case_id"],
                     "study_uid": job["study_uid"], "domain": job["domain"],
                     "shape": list(shape), "status": "cached",
                     "output": str(job["output"]), "input_path": str(job["input"]),
                     "bytes": size,
                     "denoised": volume_stats(existing)}
            summary.append(entry)
            print(f"  [{index}/{len(jobs)}] cached  {label}", flush=True)
            continue

        started = time.time()
        source = np.memmap(job["input"], dtype=DTYPE, mode="r", shape=shape)
        prediction = infer_volume_3d_tiled(
            model, np.asarray(source), args.voxel_depth, args.patch_hw,
            args.overlap_d, args.overlap_hw, device, args.batch_size,
        )
        write_raw_volume(job["output"], prediction)
        elapsed = time.time() - started
        written = job["output"].stat().st_size
        if written != size:
            raise ValueError(
                f"{job['output']}: wrote {written} bytes, expected {size} "
                f"for shape {tuple(shape)}"
            )
        before, after = volume_stats(source), volume_stats(prediction)
        drop = (1.0 - after["sigma_immerkaer"] / before["sigma_immerkaer"]) * 100.0
        entry = {"key": job["key"], "group": job["group"], "case_id": job["case_id"],
                 "study_uid": job["study_uid"], "domain": job["domain"],
                 "shape": list(shape), "status": "done",
                 "output": str(job["output"]), "input_path": str(job["input"]),
                 "bytes": size,
                 "seconds": round(elapsed, 1),
                 "input": before, "denoised": after,
                 "sigma_drop_pct": float(drop), "tissue_shift_rel_pct": float(
                     100.0 * (after["mean"] - before["mean"]) / before["mean"]
                 )}
        summary.append(entry)
        print(f"  [{index}/{len(jobs)}] done    {label} | {elapsed:.1f}s | "
              f"sigma {before['sigma_immerkaer']:.5f} -> {after['sigma_immerkaer']:.5f} "
              f"({drop:+.1f}%)", flush=True)

    report = {
        "manifest": str(args.manifest),
        "root": str(args.root),
        "out_root": str(args.out_root),
        "checkpoint": str(args.checkpoint),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "domains": domains,
        "tiled": {"voxel_depth": args.voxel_depth, "patch_hw": args.patch_hw,
                  "overlap_d": args.overlap_d, "overlap_hw": args.overlap_hw,
                  "batch_size": args.batch_size},
        "records": summary,
    }
    report_path = args.out_root / "denoise_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {len(summary)} volumes | summary -> {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
