"""Benchmark UNet3D on a 256x256x128 volume: ONNX -> FP16 -> TensorRT.

Answers, on this workstation (RTX 5060 Ti 16GB, TensorRT 11.3 strongly-typed):

  * ONNX fp32 / fp16 file sizes and which one the engine is built from
  * TensorRT engine build wall time and serialized engine size
  * device memory footprint at runtime (engine weights + per-context
    activation pool + I/O buffers)
  * pure-GPU inference latency, input/output kept in device memory with
    no host<->device copies, timed with CUDA events

Stages are run in separate processes so the memory accounting of each is
clean:

    python scripts/_bench_trt_256x256x128.py --stage torch     # PyTorch eager reference
    python scripts/_bench_trt_256x256x128.py --stage build     # ONNX export + TRT build
    python scripts/_bench_trt_256x256x128.py --stage bench     # TRT runtime latency/memory
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

SHAPE = (1, 1, 128, 256, 256)  # B, C, D, H, W  (production patch, no tiling)
BASE_CHANNELS = 16
DEPTH = 4
WORKSPACE_BYTES = 4 << 30  # 4 GiB tactic workspace cap (1 GiB was too tight for level-0)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "outputs", "trt_bench")
CKPT = os.path.join(ROOT, "runs", "air3_mode2_1609_unet3d_h2f_128_r0", "best.pt")

ONNX_FP32 = os.path.join(OUT_DIR, "unet3d_256x256x128_fp32.onnx")
ONNX_FP16 = os.path.join(OUT_DIR, "unet3d_256x256x128_fp16.onnx")
ONNX_FP16_IO32 = os.path.join(OUT_DIR, "unet3d_256x256x128_fp16_io32.onnx")
ENGINE_FP16 = os.path.join(OUT_DIR, "unet3d_256x256x128_fp16.engine")
ENGINE_FP16_IO32 = os.path.join(OUT_DIR, "unet3d_256x256x128_fp16_io32.engine")
REPORT = os.path.join(OUT_DIR, "trt_bench_report.json")

ENGINES = {"fp16": ENGINE_FP16, "fp16io32": ENGINE_FP16_IO32}
ONNX_CANDIDATES = {"fp16": ONNX_FP16, "fp16io32": ONNX_FP16_IO32}

sys.path.insert(0, os.path.join(ROOT, "src"))


def _build_model():
    import torch

    from cbct_n2n.models.unet3d import UNet3D

    model = UNet3D(in_channels=1, out_channels=1, base_channels=BASE_CHANNELS, depth=DEPTH, residual=True)
    if os.path.exists(CKPT):
        ckpt = torch.load(CKPT, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        print(f"[info] loaded weights from {os.path.relpath(CKPT, ROOT)}")
    else:
        print("[warn] checkpoint missing -> random weights (memory/latency unaffected)")
    return model.eval()


def _mem_used_mib():
    import torch

    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def _time_cuda(fn, warmup, iters, stream=None):
    import contextlib

    import torch

    ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with ctx:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    times.sort()
    return {
        "n": iters,
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": times[0],
        "p99_ms": times[min(len(times) - 1, int(round(0.99 * (len(times) - 1))))],
        "max_ms": times[-1],
    }


# --------------------------------------------------------------------------
# stage: torch eager reference
# --------------------------------------------------------------------------
def stage_torch(iters: int) -> dict:
    import torch

    torch.cuda.empty_cache()
    base = _mem_used_mib()
    model = _build_model().cuda()
    n_params = model.count_parameters()
    x = torch.randn(*SHAPE, device="cuda", dtype=torch.float32)

    report = {"n_params": n_params, "shape": list(SHAPE)}

    with torch.no_grad():
        # fp32: no TF32, honest fp32 reference
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        fp32 = _time_cuda(lambda: model(x), 5, iters)
        del model
        torch.cuda.empty_cache()

        model16 = _build_model().cuda().half()
        x16 = x.half()
        fp16 = _time_cuda(lambda: model16(x16), 5, iters)
        peak16 = torch.cuda.max_memory_allocated() / 1024**2

    report["torch_fp32"] = fp32
    report["torch_fp16"] = fp16
    report["torch_fp16_peak_alloc_MiB"] = peak16
    report["mem_used_delta_MiB"] = _mem_used_mib() - base
    return report


# --------------------------------------------------------------------------
# stage: ONNX export + fp16 conversion + TensorRT build
# --------------------------------------------------------------------------
def _export_onnx_fp32() -> float:
    import torch

    os.makedirs(OUT_DIR, exist_ok=True)
    model = _build_model()
    dummy = torch.randn(*SHAPE, dtype=torch.float32)
    kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        with torch.no_grad():
            torch.onnx.export(model, dummy, ONNX_FP32, dynamo=False, **kwargs)
    except TypeError:
        # older/newer torch without the `dynamo` switch
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, dummy, ONNX_FP32, **kwargs)
    return os.path.getsize(ONNX_FP32) / 1024**2


def _convert_onnx_fp16(dst: str, keep_io_types: bool) -> float:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(ONNX_FP32)
    kw = dict(keep_io_types=keep_io_types, disable_shape_infer=False)
    try:
        model16 = convert_float_to_float16(model, verbose=False, **kw)
    except TypeError:
        model16 = convert_float_to_float16(model, **kw)
    onnx.save(model16, dst)
    return os.path.getsize(dst) / 1024**2


def _count_fp32_nodes(onnx_path: str) -> dict:
    """Which ops the fp16 converter deliberately left in fp32 (they cost speed)."""
    import onnx

    model = onnx.load(onnx_path)
    fp32_ops: dict[str, int] = {}
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT:
            fp32_ops["<initializer>"] = fp32_ops.get("<initializer>", 0) + 1
    for node in model.graph.node:
        for out in node.output:
            vi = [v for v in model.graph.value_info if v.name == out]
            if vi and vi[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                fp32_ops[node.op_type] = fp32_ops.get(node.op_type, 0) + 1
    return fp32_ops


def _build_engine(onnx_path: str, engine_path: str, workspace: int, fp16: bool) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            for i in range(parser.num_errors):
                print("  [onnx-parse-error]", parser.get_error(i))
            raise RuntimeError(f"failed to parse {onnx_path}")

    n_layers = network.num_layers
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    if fp16:
        # strongly-typed TRT 11: precision comes from the ONNX dtypes, we only
        # make sure nothing is silently upcast back to fp32.
        config.set_flag(trt.BuilderFlag.DIRECT_IO)
    else:
        config.set_flag(trt.BuilderFlag.TF32)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("engine build returned None")

    blob = bytes(serialized)
    with open(engine_path, "wb") as fh:
        fh.write(blob)

    # second build in-process to show the timing/compilation cache effect
    t0 = time.perf_counter()
    serialized2 = builder.build_serialized_network(network, config)
    rebuild_s = time.perf_counter() - t0
    if serialized2 is not None:
        blob = bytes(serialized2)
        with open(engine_path, "wb") as fh:
            fh.write(blob)

    return {
        "onnx": os.path.basename(onnx_path),
        "network_layers": n_layers,
        "workspace_MiB": workspace / 1024**2,
        "build_s": build_s,
        "rebuild_s_with_cache": rebuild_s,
        "engine_MiB": len(blob) / 1024**2,
        "engine_bytes": len(blob),
    }


def stage_build(iters: int) -> dict:
    import torch

    torch.cuda.init()
    base = _mem_used_mib()
    rep: dict = {"ckpt": os.path.relpath(CKPT, ROOT), "shape": list(SHAPE)}

    rep["onnx_fp32_MiB"] = _export_onnx_fp32()
    try:
        rep["onnx_fp16_MiB"] = _convert_onnx_fp16(ONNX_FP16, keep_io_types=False)
        rep["onnx_fp16_io32_MiB"] = _convert_onnx_fp16(ONNX_FP16_IO32, keep_io_types=True)
        rep["fp32_nodes_left_in_fp16_graph"] = _count_fp32_nodes(ONNX_FP16)
        fp16_ok = True
    except Exception as exc:  # pragma: no cover - environment dependent
        rep["onnx_fp16_error"] = f"{type(exc).__name__}: {exc}"
        fp16_ok = False

    if fp16_ok:
        rep["engine_fp16"] = _build_engine(ONNX_FP16, ENGINE_FP16, WORKSPACE_BYTES, fp16=True)
        rep["engine_fp16io32"] = _build_engine(ONNX_FP16_IO32, ENGINE_FP16_IO32, WORKSPACE_BYTES, fp16=True)
        rep["mem_used_delta_during_build_MiB"] = _mem_used_mib() - base

    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump({"build": rep}, fh, indent=2)
    return rep


# --------------------------------------------------------------------------
# stage: TensorRT runtime benchmark
# --------------------------------------------------------------------------
def stage_bench(iters: int, engine_key: str = "fp16", engine_file: str | None = None) -> dict:
    import tensorrt as trt
    import torch

    engine_path = engine_file or ENGINES[engine_key]
    torch.cuda.empty_cache()
    base = _mem_used_mib()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as fh:
        engine = runtime.deserialize_cuda_engine(fh.read())
    if engine is None:
        raise RuntimeError("failed to deserialize engine")

    rep: dict = {
        "engine_key": engine_key,
        "engine_path": engine_path,
        "engine_MiB": os.path.getsize(engine_path) / 1024**2,
        "num_io_tensors": engine.num_io_tensors,
    }
    try:
        rep["engine_device_memory_MiB"] = engine.device_memory_size_v2 / 1024**2
    except Exception:
        rep["engine_device_memory_MiB"] = None
    try:
        rep["streamable_weights_MiB"] = engine.streamable_weights_size / 1024**2
    except Exception:
        pass

    mem_after_engine = _mem_used_mib() - base
    context = engine.create_execution_context()
    mem_after_context = _mem_used_mib() - base

    # ---- device-resident I/O: no H2D/D2H anywhere in the timed path ----
    import numpy as np

    io = {}
    io_dtypes = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        np_dtype = trt.nptype(engine.get_tensor_dtype(name))
        torch_dtype = torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype
        mode = engine.get_tensor_mode(name)
        t = torch.empty(shape, dtype=torch_dtype, device="cuda")
        io[name] = {"shape": list(shape), "mode": str(mode), "torch_dtype": str(t.dtype), "nbytes": t.numel() * t.element_size()}
        io_dtypes[name] = torch_dtype
        context.set_tensor_address(name, t.data_ptr())
        if str(mode).startswith("TensorIOMode.IN"):
            t.normal_()
            io[name]["filled"] = "normal_(0,1)"
        else:
            io[name]["filled"] = "uninitialized"

    stream = torch.cuda.Stream()
    stream_ptr = stream.cuda_stream

    def run():
        context.execute_async_v3(stream_ptr)

    # events are recorded on the *same* stream as the work, otherwise the
    # measurement collapses to event overhead
    timing = _time_cuda(run, 20, iters, stream=stream)
    mem_total = _mem_used_mib() - base

    # ---- correctness: engine output vs PyTorch reference on the same input ----
    ref_note = None
    try:
        torch.manual_seed(0)
        xin = torch.empty(tuple(io["input"]["shape"]), dtype=io_dtypes["input"], device="cuda").normal_()
        yout = torch.empty(tuple(io["output"]["shape"]), dtype=io_dtypes["output"], device="cuda")
        for name, meta in io.items():
            ptr = xin.data_ptr() if meta["mode"].endswith("INPUT") else yout.data_ptr()
            context.set_tensor_address(name, ptr)
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_ptr)
        torch.cuda.synchronize()

        with torch.no_grad():
            model32 = _build_model().cuda()
            ref32 = model32(xin.float())
            del model32
            torch.cuda.empty_cache()
            model16 = _build_model().cuda().half()
            ref16 = model16(xin.half()).float()
            del model16
        t = yout.float()
        ref_note = {
            "ref_fp32_absmean": ref32.abs().mean().item(),
            "trt_vs_torch_fp32_maxabs": (t - ref32).abs().max().item(),
            "trt_vs_torch_fp32_meanabs": (t - ref32).abs().mean().item(),
            "trt_vs_torch_fp16_maxabs": (t - ref16).abs().max().item(),
            "torch_fp16_vs_torch_fp32_maxabs": (ref16 - ref32).abs().max().item(),
        }
        del ref16, ref32, t
        torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover
        ref_note = {"error": f"{type(exc).__name__}: {exc}"}

    rep["io_tensors"] = io
    rep["buffers_MiB"] = sum(v["nbytes"] for v in io.values()) / 1024**2
    rep["latency"] = timing
    rep["fps"] = 1000.0 / timing["mean_ms"]
    rep["mem_after_engine_MiB"] = mem_after_engine
    rep["mem_after_context_MiB"] = mem_after_context
    rep["mem_total_runtime_MiB"] = mem_total
    rep["correctness"] = ref_note

    prev = {}
    if os.path.exists(REPORT):
        with open(REPORT, encoding="utf-8") as fh:
            prev = json.load(fh)
    prev.setdefault("bench", {})[engine_key] = rep
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, indent=2)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["torch", "build", "bench"], required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="fp16")
    ap.add_argument("--engine-file", default=None, help="bench an existing .engine instead of the built ones")
    args = ap.parse_args()

    if args.stage == "bench":
        rep = stage_bench(args.iters, args.engine, args.engine_file)
    else:
        rep = {"torch": stage_torch, "build": stage_build}[args.stage](args.iters)
    print(f"=== stage {args.stage} ({args.engine}) ===")
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
