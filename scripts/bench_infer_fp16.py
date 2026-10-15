"""Benchmark fp32 vs fp16 inference for the 3D UNet on a full 360-slice volume.

Measures, on the training server GPU:
  * pure-GPU forward time for a whole volume (18 sliding windows, batch=2)
  * end-to-end pipeline time (CPU window assembly + GPU forward + reassembly,
    no disk I/O)
  * fp32 vs fp16 numerical error and peak activation memory.

Historical benchmark command from the AIR3 server experiment (GPU selection
was local to that run; replace paths and device for a new machine):
  /opt/anaconda3/envs/pytorch201/bin/python scripts/bench_infer_fp16.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from cbct_n2n.models.unet3d import UNet3D  # noqa: E402


def build(batch_size: int, device: torch.device) -> tuple[int, int]:
    base, depth = 16, 4
    voxel_depth = 16
    n_slices = 360
    n_windows = (n_slices + voxel_depth - 1) // voxel_depth
    return voxel_depth, n_windows


def main() -> None:
    device = torch.device("cuda:0")
    base, depth = 16, 4
    voxel_depth, n_windows = build(2, device)
    H = W = 640
    batch_size = 2

    print(
        f"[bench] base={base} depth={depth} voxel_depth={voxel_depth} "
        f"n_windows={n_windows} batch={batch_size} H=W={H}",
        flush=True,
    )

    # synthetic input: [n_windows, 1, D, H, W]
    x = torch.randn(n_windows, 1, voxel_depth, H, W, device=device)

    def run_forward(model, xin, warmup=5, repeat=20):
        model.eval()
        for _ in range(warmup):
            for i in range(0, n_windows, batch_size):
                with torch.no_grad():
                    model(xin[i : i + batch_size])
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            for i in range(0, n_windows, batch_size):
                with torch.no_grad():
                    model(xin[i : i + batch_size])
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return min(ts), sum(ts) / len(ts)

    # --- fp32 ---
    m32 = UNet3D(base_channels=base, depth=depth, residual=True).to(device)
    n_params = sum(p.numel() for p in m32.parameters())
    t32_min, t32_avg = run_forward(m32, x)

    # --- fp16 (pure) ---
    # Same weights as fp32 (only downcast), so the error is a true fp16 quantisation
    # error rather than the difference between two independently-initialised models.
    m16 = UNet3D(base_channels=base, depth=depth, residual=True).to(device).half()
    m16.load_state_dict({k: v.half() for k, v in m32.state_dict().items()})
    x16 = x.half()
    t16_min, t16_avg = run_forward(m16, x16)

    # --- numerical error ---
    with torch.no_grad():
        y32 = m32(x[:batch_size])
        y16 = m16(x16[:batch_size]).float()
    rel_err = (y16 - y32).abs().mean().item() / y32.abs().mean().clamp(min=1e-6).item()

    # --- peak activation memory (per batch) ---
    def peak(model, xin):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(xin[:batch_size])
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e9

    peak32 = peak(m32, x)
    peak16 = peak(m16, x16)

    # --- end-to-end pipeline (memory-resident volume, no disk I/O) ---
    def end_to_end(model, fp16: bool, repeat: int = 5) -> float:
        torch.manual_seed(0)
        vol = np.random.rand(360, H, W).astype(np.float32)  # ~590 MB
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            padded = np.stack([vol[i] for i in range(360)])  # 640 already 32-aligned
            windows = []
            for z_start in range(0, 360, voxel_depth):
                win = np.zeros((voxel_depth, H, W), dtype=np.float32)
                z_end = min(z_start + voxel_depth, 360)
                win[:z_end - z_start] = padded[z_start:z_end]
                windows.append(win)
            preds = []
            for start in range(0, len(windows), batch_size):
                b = torch.from_numpy(np.stack(windows[start : start + batch_size])).unsqueeze(1).to(device)
                if fp16:
                    b = b.half()
                with torch.no_grad():
                    p = model(b).float().cpu().numpy()[:, 0]
                preds.append(p)
            all_pred = np.concatenate(preds, axis=0)
            out = np.zeros_like(padded)
            for k, z_start in enumerate(range(0, 360, voxel_depth)):
                z_end = min(z_start + voxel_depth, 360)
                out[z_start:z_end] = all_pred[k][:z_end - z_start]
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times)

    e2e32 = end_to_end(m32, False)
    e2e16 = end_to_end(m16, True)

    result = {
        "device": f"cuda:0 ({torch.cuda.get_device_name(0)})",
        "arch": f"UNet3D base{base}/depth{depth}",
        "params": n_params,
        "n_slices": 360,
        "n_windows": n_windows,
        "batch_size": batch_size,
        "fp32_forward_min_s": round(t32_min, 4),
        "fp32_forward_avg_s": round(t32_avg, 4),
        "fp16_forward_min_s": round(t16_min, 4),
        "fp16_forward_avg_s": round(t16_avg, 4),
        "forward_speedup": round(t32_avg / t16_avg, 3),
        "rel_err_fp16": rel_err,
        "peak_act_gb_fp32": round(peak32, 3),
        "peak_act_gb_fp16": round(peak16, 3),
        "e2e_fp32_s": round(e2e32, 4),
        "e2e_fp16_s": round(e2e16, 4),
        "e2e_speedup": round(e2e32 / e2e16, 3),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
