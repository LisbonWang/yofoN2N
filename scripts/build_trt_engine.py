"""Build a TensorRT engine from an ONNX denoiser and verify it against torch.

Usage:
    python scripts/build_trt_engine.py --onnx runs/xxx/best.onnx --fp16 \
        --input-shape 1x1x128x256x256 --output runs/xxx/best_fp16.engine \
        --checkpoint runs/xxx/best.pt

The ONNX is produced by ``export_onnx.py`` with dynamic batch/depth/height/width
axes, so an optimization profile pins the concrete shape used at inference
(1 x 1 x 128 x 256 x 256 for the 128-window UNet3D).  ``--fp16`` enables FP16
precision (16-bit weights/activations) — the practical quantization for this
GroupNorm-heavy denoiser family (int8 is a poor fit; see quantize_onnx.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_shape(text: str) -> tuple[int, ...]:
    dims = tuple(int(v) for v in text.lower().split("x"))
    if len(dims) != 5 or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError(f"shape must be NxCxDxHxW, got {text!r}")
    return dims


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool,
                 opt_shape: tuple[int, ...], workspace_gb: int = 4):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()  # TRT 10+: explicit batch is the default
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[parser] {parser.get_error(i)}", flush=True)
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    # TRT 11.x removed all precision builder flags (FP16/INT8/...). Precision now
    # comes from the ONNX graph itself: pass an FP16-quantized ONNX (e.g. from
    # `export_onnx.py --fp16`) to build a 16-bit engine, or an FP32 ONNX for FP32.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", opt_shape, opt_shape, opt_shape)  # static shape
    config.add_optimization_profile(profile)

    print(f"[trt] building engine (fp16={fp16}, shape={opt_shape}) ...", flush=True)
    t0 = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("engine build returned None")
    print(f"[trt] build done in {time.perf_counter() - t0:.1f}s", flush=True)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = bytes(engine)  # IHostMemory -> bytes
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[trt] saved {engine_path} ({len(serialized) / 1e6:.1f} MB)", flush=True)
    return serialized


def trt_infer(engine_bytes: bytes, x_np: np.ndarray, warmup: int = 5, iters: int = 20):
    import tensorrt as trt
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = eng.create_execution_context()
    x = torch.from_numpy(x_np).cuda()
    y = torch.empty_like(x)
    ctx.set_tensor_address("input", x.data_ptr())
    ctx.set_tensor_address("output", y.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(warmup):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    return y.cpu().numpy(), elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None, help="default: <onnx stem>_fp16.engine")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--input-shape", type=_parse_shape, default=(1, 1, 128, 256, 256),
                    metavar="NxCxDxHxW")
    ap.add_argument("--workspace-gb", type=int, default=4)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="torch .pt for reference comparison (verify mode)")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if not args.onnx.is_file():
        ap.error(f"onnx not found: {args.onnx}")
    output = args.output or args.onnx.with_name(args.onnx.stem + ("_fp16" if args.fp16 else "_fp32") + ".engine")

    engine = build_engine(args.onnx, output, args.fp16, args.input_shape, args.workspace_gb)

    if args.no_verify:
        return 0
    if args.checkpoint is None:
        print("[verify] SKIP (no --checkpoint provided)", flush=True)
        return 0

    # --- verify TRT engine vs torch fp32 on real-domain random-ish input ---
    from cbct_n2n.models import UNet3D
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mcfg = ck["config"]["model"]
    model = UNet3D(**{k: v for k, v in mcfg.items() if k != "arch"})
    model.load_state_dict(ck["model"])
    model.eval().cuda()

    n, c, d, h, w = args.input_shape
    rng = np.random.default_rng(0)
    # μ-domain-like magnitudes (bulk ~0.01, tail up to ~1) instead of unit gaussian
    x = (rng.standard_normal((n, c, d, h, w)) * 0.05 + 0.01).astype(np.float32)
    x[..., : h // 8, : w // 8] = 0.0  # air pocket to stress GroupNorm low-variance path

    with torch.no_grad():
        ref = model(torch.from_numpy(x).cuda()).cpu().numpy()
    got, per_iter = trt_infer(engine, x)
    diff = np.abs(got - ref)
    print(f"[verify] latency {per_iter * 1000:.2f} ms/iter (shape {args.input_shape})", flush=True)
    print(f"[verify] max abs diff = {diff.max():.3e}  rel = {diff.max() / max(float(np.abs(ref).max()), 1e-12):.3e}",
          flush=True)
    print(f"[verify] mean abs diff = {diff.mean():.3e}  median = {np.median(diff):.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
