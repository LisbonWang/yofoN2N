"""Benchmark a TensorRT engine on a whole CBCT volume that stays resident on the GPU.

``bench_trt_full_volume.py`` times the classical deployment shape: the volume lives in
host memory, each tile is copied up, executed, copied back, and the trapezoid fusion is
accumulated on the CPU.  That is the right number when the volume is streamed from disk
-- but it mixes CPU work into the measurement and it ignores the device footprint of the
volume itself.

This script answers the other question: *how much device memory and how much pure GPU
time does one whole volume cost, assuming the volume is already on the GPU and never
leaves it?*  Concretely:

* the whole input volume is one ``float32`` device tensor (``Z x Y x X``);
* the fused output and the fusion-weight accumulator are device tensors of the same size;
* every tile is sliced in place on the device, pushed through the engine, and the
  trapezoid-weighted accumulation happens on the device;
* nothing in the timed loop touches the host, so the number is not diluted by copies.

The footprint is reported as a decomposition (CUDA context / engine weights + execution
context / volume buffers / activation pool), taken from ``cudaMemGetInfo`` so that memory
allocated by TensorRT outside the PyTorch caching allocator is counted.

Usage::

    python scripts/bench_trt_volume_resident.py \
      --engine outputs/trt_bench/unet3d_256x256x128_fp16_io32.engine \
      --shape 360 640 640 --repeats 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_trt_full_volume import (  # noqa: E402
    _tile_starts,
    _trapezoid_window,
)

MIB = 1024.0 * 1024.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--input", type=Path, default=None,
                        help="Optional .raw to read; random data is used otherwise.")
    parser.add_argument("--voxel-depth", type=int, default=128)
    parser.add_argument("--patch-hw", type=int, default=256)
    parser.add_argument("--overlap-d", type=int, default=16)
    parser.add_argument("--overlap-hw", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None, help="Optional sidecar report path.")
    parser.add_argument("--save", type=Path, default=None,
                        help="Write the fused volume here (float32 .raw) to cross-check numerics.")
    parser.add_argument("--cleanup-every-repeat", action="store_true",
                        help="Zero the accumulators between repeats (timed).")
    return parser.parse_args(argv)


def _mib(nbytes: float) -> float:
    return float(nbytes) / MIB


def _device_used() -> int:
    free, total = torch.cuda.mem_get_info()
    return int(total - free)


def _padded(shape: tuple[int, int, int], voxel_depth: int, patch_hw: int):
    n, h, w = shape
    return (max(n, voxel_depth),
            int(np.ceil(max(h, patch_hw) / 32) * 32),
            int(np.ceil(max(w, patch_hw) / 32) * 32))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import tensorrt as trt

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    torch.cuda.init()

    shape = tuple(int(v) for v in args.shape)
    n, height, width = shape
    voxel_depth, patch_hw = args.voxel_depth, args.patch_hw
    overlap_d, overlap_hw = args.overlap_d, args.overlap_hw

    device_name = torch.cuda.get_device_name(0)
    _, total_bytes = torch.cuda.mem_get_info()
    used_at_start = _device_used()
    print(f"[gpu] {device_name}  total {_mib(total_bytes):.0f} MiB  "
          f"already in use {_mib(used_at_start):.0f} MiB", flush=True)

    # ------------------------------------------------------------------ engine
    engine_bytes = args.engine.read_bytes()
    log = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(log)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    if engine is None or context is None:
        raise SystemExit("failed to deserialize the engine / create a context")

    io_dtype = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        io_dtype[name] = (engine.get_tensor_mode(name), engine.get_tensor_dtype(name),
                          tuple(engine.get_tensor_shape(name)))
        print(f"[engine] {name}: dtype={io_dtype[name][1]} shape={io_dtype[name][2]}", flush=True)
    weights_and_context = _device_used() - used_at_start
    activation_pool = int(getattr(engine, "device_memory_size_v2", 0))

    engine_shape = tuple(io_dtype["input"][2])          # (1, 1, D, H, W)
    if engine_shape[0] != 1 or engine_shape[1] != 1:
        raise SystemExit(f"unexpected engine input shape {engine_shape}")
    tile_d, tile_h, tile_w = engine_shape[2], engine_shape[3], engine_shape[4]

    # ------------------------------------------------------------------- layout
    nz, ny, nx = _padded(shape, voxel_depth, patch_hw)
    d_starts = _tile_starts(nz, tile_d, tile_d - overlap_d)
    h_starts = _tile_starts(ny, tile_h, tile_h - overlap_hw)
    w_starts = _tile_starts(nx, tile_w, tile_w - overlap_hw)
    tiles = [(d0, h0, w0) for d0 in d_starts for h0 in h_starts for w0 in w_starts]
    ntiles = len(tiles)
    print(f"[tile] grid D={len(d_starts)} x H={len(h_starts)} x W={len(w_starts)} = "
          f"{ntiles} tiles of {tile_d}x{tile_h}x{tile_w}", flush=True)

    # ------------------------------------------------- device-resident buffers
    mark = _device_used()
    volume = torch.empty((nz, ny, nx), dtype=torch.float32, device="cuda")
    if args.input is not None:
        expected = int(np.prod(shape)) * 4
        if args.input.stat().st_size != expected:
            raise SystemExit(f"{args.input} is {args.input.stat().st_size} B, expected {expected} B")
        host = np.memmap(args.input, dtype="<f4", mode="r", shape=shape)
        volume[:n, :height, :width].copy_(torch.from_numpy(np.asarray(host)))
        volume[n:, :, :] = 0.0
        volume[:, height:, :] = 0.0
        volume[:, :, width:] = 0.0
    else:
        volume.normal_()

    accumulator = torch.zeros_like(volume)
    weight_sum = torch.zeros_like(volume)

    tile_in = torch.empty((1, 1, tile_d, tile_h, tile_w), dtype=torch.float32, device="cuda")
    tile_out = torch.empty_like(tile_in)

    ramps = (_trapezoid_window(tile_d, overlap_d),
             _trapezoid_window(tile_h, overlap_hw),
             _trapezoid_window(tile_w, overlap_hw))
    tile_weight = torch.from_numpy(
        (ramps[0][:, None, None] * ramps[1][None, :, None] * ramps[2][None, None, :]).astype(np.float32)
    ).to("cuda")

    context.set_tensor_address("input", tile_in.data_ptr())
    context.set_tensor_address("output", tile_out.data_ptr())

    volume_bytes = volume.numel() * volume.element_size()
    buffers = _device_used() - mark
    print(f"[mem] volume {shape} fp32 = {_mib(volume_bytes):.1f} MiB, "
          f"x3 (in + fused + weights) = {_mib(volume_bytes * 3):.1f} MiB", flush=True)

    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def run_once(timed: bool) -> float:
        if args.cleanup_every_repeat:
            accumulator.zero_()
            weight_sum.zero_()
        with torch.cuda.stream(stream):
            if timed:
                start.record(stream)
            for (d0, h0, w0) in tiles:
                tile_in[0, 0].copy_(volume[d0:d0 + tile_d, h0:h0 + tile_h, w0:w0 + tile_w])
                context.execute_async_v3(stream.cuda_stream)
                accumulator[d0:d0 + tile_d, h0:h0 + tile_h, w0:w0 + tile_w] += \
                    tile_out[0, 0] * tile_weight
                weight_sum[d0:d0 + tile_d, h0:h0 + tile_h, w0:w0 + tile_w] += tile_weight
            if timed:
                end.record(stream)
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)) if timed else 0.0

    for _ in range(args.warmup):
        run_once(timed=False)

    peak_used = _device_used()
    peak_reserved = torch.cuda.max_memory_reserved()

    per_repeat_ms: list[float] = []
    walls: list[float] = []
    for _ in range(args.repeats):
        wall0 = time.perf_counter()
        per_repeat_ms.append(run_once(timed=True))
        walls.append((time.perf_counter() - wall0) * 1000.0)

    fused = accumulator / torch.clamp(weight_sum, min=1e-8)
    fused = fused[:n, :height, :width]
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_bytes(fused.detach().cpu().numpy().astype("<f4").tobytes())
        print(f"[save] {args.save} ({args.save.stat().st_size / MIB:.1f} MiB)")

    ordered = sorted(per_repeat_ms)
    median_ms = ordered[len(ordered) // 2]
    print(f"\n===== device-resident whole-volume run ({shape}, {ntiles} tiles) =====")
    print(f"input dtype                    : fp32 (as given)")
    print(f"engine tile                    : {tile_d}x{tile_h}x{tile_w}")
    print(f"per-repeat GPU time median     : {median_ms:9.1f} ms   "
          f"(min {ordered[0]:.1f} / max {ordered[-1]:.1f})")
    print(f"per-tile GPU time              : {median_ms / ntiles:9.2f} ms")
    print(f"whole-volume wall per repeat   : {sorted(walls)[len(walls) // 2]:9.1f} ms")
    print(f"\n--- device memory (cudaMemGetInfo deltas) ---")
    print(f"engine weights + exec context  : {_mib(weights_and_context):9.1f} MiB")
    print(f"activation pool (engine)       : {_mib(activation_pool):9.1f} MiB")
    print(f"volume buffers (in+out+wgt)    : {_mib(buffers):9.1f} MiB")
    print(f"torch peak reserved            : {_mib(peak_reserved):9.1f} MiB")
    print(f"total device delta at peak     : {_mib(peak_used - used_at_start):9.1f} MiB")
    print(f"output                         : min={float(fused.min()):.5f} "
          f"max={float(fused.max()):.5f} mean={float(fused.mean()):.5f}")

    payload = {
        "device": device_name,
        "shape": list(shape),
        "input_dtype": "float32",
        "engine": str(args.engine),
        "engine_bytes": len(engine_bytes),
        "tile": [tile_d, tile_h, tile_w],
        "tile_grid": [len(d_starts), len(h_starts), len(w_starts)],
        "ntiles": ntiles,
        "repeats": args.repeats,
        "per_repeat_gpu_ms": per_repeat_ms,
        "per_repeat_wall_ms": walls,
        "median_gpu_ms": median_ms,
        "per_tile_gpu_ms": median_ms / ntiles,
        "mem_mib": {
            "already_in_use_at_start": _mib(used_at_start),
            "engine_weights_and_context": _mib(weights_and_context),
            "activation_pool": _mib(activation_pool),
            "volume_buffers": _mib(buffers),
            "torch_peak_reserved": _mib(peak_reserved),
            "total_delta_at_peak": _mib(peak_used - used_at_start),
        },
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
