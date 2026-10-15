"""Benchmark a TensorRT engine on a full CBCT volume via overlapping-tile inference.

The ``best_fp16.engine`` was built for a static shape ``1 x 1 x 128 x 256 x 256``
(FP16 weights/activations, FP32 I/O — see ``build_trt_engine.py``).  A whole
volume (e.g. ``360 x 640 x 640``) cannot fit the engine in one shot, so it is
covered by the same overlapping-tile scheme used at training/inference time:

    * D   tile 128, overlap 16  -> stride 112
    * H/W tile 256, overlap 16  -> stride 240

Each tile is a single batch-1 engine call; adjacent tiles are fused with a
3D-separable trapezoid window.  This script reports the tile grid, the pure GPU
kernel time (CUDA events) and the end-to-end wall-clock time (including host<->
device copies and CPU accumulation) for one complete volume.

Usage:
    python scripts/bench_trt_full_volume.py \
        --engine runs/1609v2_n2n_3d_h2f_128/best_fp16.engine \
        --input /path/to/dataset/PDL/Mode2_1609/<case>/<study>/full.raw \
        --shape 360 640 640
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _trapezoid_window(size: int, ramp: int) -> np.ndarray:
    w = np.ones(size, dtype=np.float32)
    r = min(max(int(ramp), 0), size // 2)
    for i in range(r):
        v = (i + 0.5) / r
        w[i] = v
        w[size - 1 - i] = v
    return w


def _tile_starts(total: int, tile: int, stride: int) -> list[int]:
    if total <= tile:
        return [0]
    starts: list[int] = []
    s = 0
    while s + tile < total:
        starts.append(s)
        s += stride
    starts.append(total - tile)
    return starts


def tile_grid(shape: tuple[int, int, int], voxel_depth: int, patch_hw: int,
              overlap_d: int, overlap_hw: int) -> tuple[int, int, int]:
    n, H, W = shape
    Np = max(n, voxel_depth)
    Hp = int(np.ceil(max(H, patch_hw) / 32) * 32)
    Wp = int(np.ceil(max(W, patch_hw) / 32) * 32)
    d = _tile_starts(Np, voxel_depth, voxel_depth - overlap_d)
    h = _tile_starts(Hp, patch_hw, patch_hw - overlap_hw)
    w = _tile_starts(Wp, patch_hw, patch_hw - overlap_hw)
    return len(d), len(h), len(w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--shape", type=int, nargs=3, metavar=("Z", "Y", "X"), required=True)
    ap.add_argument("--voxel-depth", type=int, default=128)
    ap.add_argument("--patch-hw", type=int, default=256)
    ap.add_argument("--overlap-d", type=int, default=16)
    ap.add_argument("--overlap-hw", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    import tensorrt as trt

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    shape = tuple(args.shape)
    n, H, W = shape
    expected = int(np.prod(shape)) * 4
    if args.input.stat().st_size != expected:
        raise SystemExit(f"input size {args.input.stat().st_size} != {expected} bytes for shape {shape}")

    # --- load engine ---
    with open(args.engine, "rb") as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = eng.create_execution_context()
    print(f"[engine] {args.engine} ({len(engine_bytes) / 1e6:.1f} MB)", flush=True)
    for i in range(eng.num_io_tensors):
        name = eng.get_tensor_name(i)
        mode = eng.get_tensor_mode(name)
        dtypes = eng.get_tensor_dtype(name)
        shape_i = eng.get_tensor_shape(name)
        print(f"[engine]   {'IN ' if mode == trt.TensorIOMode.INPUT else 'OUT'} {name} {dtypes} {shape_i}", flush=True)

    # --- tile layout ---
    voxel_depth, patch_hw, overlap_d, overlap_hw = args.voxel_depth, args.patch_hw, args.overlap_d, args.overlap_hw
    Np = max(n, voxel_depth)
    Hp = int(np.ceil(max(H, patch_hw) / 32) * 32)
    Wp = int(np.ceil(max(W, patch_hw) / 32) * 32)
    d_starts = _tile_starts(Np, voxel_depth, voxel_depth - overlap_d)
    h_starts = _tile_starts(Hp, patch_hw, patch_hw - overlap_hw)
    w_starts = _tile_starts(Wp, patch_hw, patch_hw - overlap_hw)
    tiles = [(d0, h0, w0) for d0 in d_starts for h0 in h_starts for w0 in w_starts]
    ntiles = len(tiles)
    print(f"[tile] volume {shape} -> padded {Np}x{Hp}x{Wp}", flush=True)
    print(f"[tile] grid D={len(d_starts)} x H={len(h_starts)} x W={len(w_starts)} = {ntiles} tiles "
          f"(tile {voxel_depth}x{patch_hw}x{patch_hw})", flush=True)

    # --- load volume + pad ---
    source = np.memmap(args.input, dtype="<f4", mode="r", shape=shape)
    padded = np.asarray(source).astype(np.float32)
    if (Np, Hp, Wp) != (n, H, W):
        padded = np.pad(padded, ((0, Np - n), (0, Hp - H), (0, Wp - W)), mode="constant")

    # trapezoid fusion weights
    w_d = _trapezoid_window(voxel_depth, overlap_d)
    w_h = _trapezoid_window(patch_hw, overlap_hw)
    w_w = _trapezoid_window(patch_hw, overlap_hw)
    tile_weight = (w_d[:, None, None] * w_h[None, :, None] * w_w[None, None, :]).astype(np.float32)

    out = np.zeros((Np, Hp, Wp), dtype=np.float32)
    weight = np.zeros((Np, Hp, Wp), dtype=np.float32)

    # --- pre-allocate device buffers + CUDA events ---
    dev_in = torch.zeros((1, 1, voxel_depth, patch_hw, patch_hw), dtype=torch.float32, device="cuda")
    dev_out = torch.empty_like(dev_in)
    ctx.set_tensor_address("input", dev_in.data_ptr())
    ctx.set_tensor_address("output", dev_out.data_ptr())
    torch_stream = torch.cuda.Stream()  # non-default stream: avoids TRT's default-stream sync
    trt_stream = torch_stream.cuda_stream  # raw CUstream handle for TRT
    default_stream = torch.cuda.current_stream()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    def run_tile(slab: np.ndarray) -> np.ndarray:
        with torch.cuda.stream(torch_stream):
            dev_in.copy_(torch.from_numpy(slab).view(1, 1, voxel_depth, patch_hw, patch_hw), non_blocking=True)
            start_ev.record(torch_stream)
            ctx.execute_async_v3(trt_stream)
            end_ev.record(torch_stream)
        return dev_out

    # warmup
    for _ in range(args.warmup):
        run_tile(np.zeros((voxel_depth, patch_hw, patch_hw), dtype=np.float32))
    torch.cuda.synchronize()

    # --- end-to-end timed pass ---
    gpu_ms_total = 0.0
    t0 = time.perf_counter()
    for (d0, h0, w0) in tiles:
        slab = padded[d0:d0 + voxel_depth, h0:h0 + patch_hw, w0:w0 + patch_hw]
        dev_out_t = run_tile(slab)
        default_stream.wait_stream(torch_stream)
        pred = dev_out_t.cpu().numpy()[0, 0]  # [D, Ht, Wt]
        start_ev.synchronize()
        end_ev.synchronize()
        gpu_ms_total += start_ev.elapsed_time(end_ev)
        out[d0:d0 + voxel_depth, h0:h0 + patch_hw, w0:w0 + patch_hw] += pred * tile_weight
        weight[d0:d0 + voxel_depth, h0:h0 + patch_hw, w0:w0 + patch_hw] += tile_weight
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    fused = out / np.maximum(weight, 1e-8)
    fused = fused[:n, :H, :W]

    gpu_ms_total = float(gpu_ms_total)
    per_tile_wall_ms = wall_s * 1000 / ntiles
    per_tile_gpu_ms = gpu_ms_total / ntiles
    overhead_ms = per_tile_wall_ms - per_tile_gpu_ms

    print(f"\n===== full-volume timing (engine fp16, {ntiles} tiles) =====", flush=True)
    print(f"total wall-clock        : {wall_s * 1000:8.1f} ms   ({wall_s:.3f} s)", flush=True)
    print(f"  pure GPU kernel sum   : {gpu_ms_total:8.1f} ms", flush=True)
    print(f"  memcpy + accumulate   : {wall_s * 1000 - gpu_ms_total:8.1f} ms", flush=True)
    print(f"per-tile wall           : {per_tile_wall_ms:8.2f} ms", flush=True)
    print(f"per-tile GPU kernel     : {per_tile_gpu_ms:8.2f} ms", flush=True)
    print(f"per-tile host overhead  : {overhead_ms:8.2f} ms", flush=True)
    print(f"output                  : min={float(fused.min()):.5f} max={float(fused.max()):.5f} "
          f"shape={fused.shape}", flush=True)

    # projection for other common clinical sizes
    print(f"\n===== projected time for other volume sizes (per-tile wall {per_tile_wall_ms:.2f} ms) =====", flush=True)
    for name, vol_shape in [("training 360x640x640", (360, 640, 640)),
                            ("clinical 520x680x680", (520, 680, 680)),
                            ("PCG      700x1000x1000", (700, 1000, 1000))]:
        nd, nh, nw = tile_grid(vol_shape, voxel_depth, patch_hw, overlap_d, overlap_hw)
        t = nd * nh * nw * per_tile_wall_ms
        print(f"  {name:24s} {vol_shape[0]:4d}x{vol_shape[1]:4d}x{vol_shape[2]:4d} -> "
              f"{nd}x{nh}x{nw}={nd * nh * nw:4d} tiles -> {t / 1000:7.2f} s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
