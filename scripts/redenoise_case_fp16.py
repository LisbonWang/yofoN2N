"""Historical AIReduceNoise2 inference utility; example paths are run-specific.

Re-denoise a single case directory with an fp16 3D UNet, reporting pure-GPU time.

Reads ``full.raw`` from the case dir, runs whole-volume fp16 inference,
overwrites ``full_denoised.raw`` + ``full_metrics.json``, and reports the
**pure-GPU forward time**: every window is staged on the GPU once, then warmup
+ repeated timed forwards are measured with ``torch.cuda.synchronize()``.  CPU
window assembly, H2D/D2H copies and disk I/O are excluded.

Example:
  python scripts/redenoise_case_fp16.py \\
    --config configs/1609v2_unet3d_h2f.yaml \\
    --manifest Z:/raid0/yyc/AIReduceNoise2/splits_1609v2.json \\
    --checkpoint outputs/eval_checkpoints/best_1609v2_h2f.pt \\
    --case-dir outputs/eval_1609v2_h2f_best/Mode2_1609__<study_uid>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

import train_voxel  # noqa: E402
from cbct_n2n.config import load_config  # noqa: E402
from cbct_n2n.data.dataset import pad_to_multiple, unpad  # noqa: E402
from cbct_n2n.training.checkpoint import load_checkpoint  # noqa: E402
from cbct_n2n.training.inference import evaluate_volume, write_raw_volume  # noqa: E402


def _parse_case_dir(case_dir: Path) -> tuple[str, str]:
    """``{group}__{study_uid}`` -> (group, study_uid)."""
    group, study_uid = case_dir.name.split("__", 1)
    return group, study_uid


def _build_windows(source: np.ndarray, voxel_depth: int):
    """Pad H/W to a multiple of 32 and split into non-overlapping D-slice
    windows (strided by voxel_depth; the last window is zero-padded).

    Returns (windows [nw, D, Hp, Wp] float32, z_starts, padding, padded).
    """
    n = int(source.shape[0])
    padded_slices: list[np.ndarray] = []
    padding = None
    for i in range(n):
        p, pad = pad_to_multiple(source[i].astype(np.float32))
        padded_slices.append(p)
        if padding is None:
            padding = pad
    padded = np.stack(padded_slices, axis=0)  # [n, Hp, Wp]
    Hp, Wp = padded.shape[-2], padded.shape[-1]
    z_starts = list(range(0, n, voxel_depth))
    windows = []
    for z_start in z_starts:
        win = np.zeros((voxel_depth, Hp, Wp), dtype=np.float32)
        z_end = min(z_start + voxel_depth, n)
        win[:z_end - z_start] = padded[z_start:z_end]
        windows.append(win)
    return np.stack(windows, axis=0), z_starts, padding, padded


@torch.no_grad()
def _forward_batches(model, all_windows: torch.Tensor, batch_size: int) -> None:
    for start in range(0, all_windows.shape[0], batch_size):
        model(all_windows[start:start + batch_size])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    voxel_depth = int(config["data"].get("voxel_depth", 16))
    data_range = float(config["data"]["metric_data_range"])
    arch = config.get("model", {}).get("arch", "unet3d")
    device = torch.device(args.device)

    case_dir = Path(args.case_dir)
    group, study_uid = _parse_case_dir(case_dir)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    base = Path(args.manifest).parent
    record = next(
        (r for r in manifest["records"]
         if str(r.get("study_uid")) == study_uid and r.get("group") == group),
        None,
    )
    if record is None:
        print(f"[error] manifest has no study_uid={study_uid} group={group}", flush=True)
        return 1
    shape = tuple(int(d) for d in record["shape"])
    dtype = np.dtype(record.get("dtype") or "<f4")

    # --- read full.raw ---
    full_local = case_dir / "full.raw"
    full_path = full_local if full_local.exists() else base / record["outputs"]["full.raw"]
    full = np.asarray(np.memmap(full_path, dtype=dtype, mode="r", shape=shape))
    n_slices = int(full.shape[0])
    print(f"[read] {full_path.name} shape={full.shape}", flush=True)

    # --- load model (fp16 quantised) ---
    model = train_voxel.build_model(arch, config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval().half()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] arch={arch} params={n_params:,} fp16 device={device}", flush=True)

    # --- build windows on CPU (excluded from pure-GPU timing) ---
    t0 = time.perf_counter()
    windows, z_starts, padding, padded = _build_windows(full, voxel_depth)
    n_windows = int(windows.shape[0])
    t_asm = time.perf_counter() - t0
    print(f"[asm] {n_windows} windows {windows.shape} {t_asm:.2f}s (CPU, excluded)", flush=True)

    # --- stage all windows on GPU once (H2D copy excluded from timing) ---
    all_windows = torch.from_numpy(windows).unsqueeze(1).to(device).half()  # [nw,1,D,Hp,Wp]
    torch.cuda.synchronize()

    # --- warmup ---
    for _ in range(args.warmup):
        _forward_batches(model, all_windows, args.batch_size)
    torch.cuda.synchronize()

    # --- pure-GPU timed forwards ---
    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        _forward_batches(model, all_windows, args.batch_size)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    pure_min = min(times)
    pure_avg = sum(times) / len(times)
    print(f"[pure-gpu] min={pure_min * 1000:.1f}ms  avg={pure_avg * 1000:.1f}ms  "
          f"(n_windows={n_windows} batch={args.batch_size} warmup={args.warmup} repeat={args.repeat})",
          flush=True)

    # --- real inference (reuse the staged windows) ---
    t0 = time.perf_counter()
    preds = []
    for start in range(0, n_windows, args.batch_size):
        p = model(all_windows[start:start + args.batch_size]).detach().float().cpu().numpy()[:, 0]
        preds.append(p)
    all_pred = np.concatenate(preds, axis=0)  # [nw, D, Hp, Wp]
    out = np.zeros_like(padded)  # [n, Hp, Wp]
    for k, z_start in enumerate(z_starts):
        z_end = min(z_start + voxel_depth, n_slices)
        out[z_start:z_end] = all_pred[k][:z_end - z_start]
    pred = np.stack([unpad(s, padding) for s in out]).astype(np.float32)
    t_infer_e2e = time.perf_counter() - t0

    # --- metrics vs full + write outputs ---
    metrics = evaluate_volume(pred, full, data_range)
    t0 = time.perf_counter()
    write_raw_volume(case_dir / "full_denoised.raw", pred)
    t_write = time.perf_counter() - t0

    entry = {
        "key": record["key"],
        "study_uid": study_uid,
        "group": group,
        "case_id": record.get("case_id"),
        "source_kind": "full",
        "reference": "full",
        "n_slices": n_slices,
        "fp16": True,
        "metrics": metrics,
        "timing": {
            "pure_gpu_min_s": round(pure_min, 4),
            "pure_gpu_avg_s": round(pure_avg, 4),
            "windows": n_windows,
            "batch_size": args.batch_size,
            "infer_e2e_s": round(t_infer_e2e, 3),
            "write_s": round(t_write, 3),
        },
    }
    (case_dir / "full_metrics.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")

    print(f"[done] {group}__{study_uid}", flush=True)
    print(f"  pure-GPU forward: min={pure_min * 1000:.1f}ms  avg={pure_avg * 1000:.1f}ms", flush=True)
    print(f"  infer_e2e={t_infer_e2e:.2f}s  write={t_write:.2f}s", flush=True)
    print(f"  mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} psnr={metrics['psnr']:.4f} ssim={metrics['ssim']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
