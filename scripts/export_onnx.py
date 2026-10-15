"""Export a trained UNet2D checkpoint to ONNX with dynamic input size.

Usage:
    python scripts/export_onnx.py --checkpoint /path/to/runs/experiment/best.pt
    python scripts/export_onnx.py --checkpoint /path/to/runs/experiment/best.pt --input-size 512x512 --opset 17
    python scripts/export_onnx.py --checkpoint /path/to/runs/experiment/best.pt --fp16

The model architecture is rebuilt from the checkpoint's ``config["model"]`` section,
so no manual hyper-parameter entry is needed.  Input is dynamic on batch/height/width
(the network is fully convolutional); the ``--input-size`` only fixes the dummy shape
used for tracing and shape inference.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbct_n2n.models import UNet2D, UNet2D_DS, UNet3D  # noqa: E402


def load_model(checkpoint: str | Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("config", {}).get("model", {})
    if not model_cfg:
        raise ValueError(f"checkpoint {checkpoint} has no config['model']; cannot rebuild architecture")
    arch = model_cfg.get("arch", "unet2d")
    model_cls = {"unet2d": UNet2D, "unet2d_ds": UNet2D_DS, "unet3d": UNet3D}.get(arch)
    if model_cls is None:
        raise ValueError(f"unknown arch {arch!r}; expected unet2d, unet2d_ds or unet3d")
    model = model_cls(**{k: v for k, v in model_cfg.items() if k != "arch"})
    state = ckpt["model"]
    # strip DataParallel / DDP prefixes if present
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def parse_size(text: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(v) for v in text.lower().split("x"))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"input size must be HxW or DxHxW, got {text!r}")
    if len(dims) not in (2, 3):
        raise argparse.ArgumentTypeError(f"input size must be HxW (2D) or DxHxW (3D), got {text!r}")
    if any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError(f"input size must be positive, got {text!r}")
    return dims


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="checkpoint .pt file")
    parser.add_argument("--output", type=Path, default=None,
                        help="output .onnx path (default: <checkpoint dir>/<stem>.onnx)")
    parser.add_argument("--input-size", type=parse_size, default=(256, 256), metavar="HxW",
                        help="dummy input HxW used for tracing (default: 256x256; runtime size is dynamic)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")
    parser.add_argument("--fp16", action="store_true",
                        help="convert exported model weights to FP16 (halves file size, "
                             "nearly lossless accuracy; I/O stays FP32)")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip onnxruntime numeric validation")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    output = args.output or args.checkpoint.with_suffix(".onnx")
    print(f"[export] checkpoint : {args.checkpoint}", flush=True)

    model, ckpt = load_model(args.checkpoint)
    arch = ckpt.get("config", {}).get("model", {}).get("arch", "unet2d")
    print(f"[export] model arch  : {arch} from config['model']", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] parameters : {n_params:,}  (epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('metrics', {}).get('val_loss', 'n/a')})", flush=True)

    if arch == "unet3d":
        if len(args.input_size) != 3:
            parser.error(f"--input-size for unet3d must be DxHxW, got {'x'.join(map(str, args.input_size))}")
        d, h, w = args.input_size
        input_shape = (1, 1, d, h, w)
        dynamic_axes = {"input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
                        "output": {0: "batch", 2: "depth", 3: "height", 4: "width"}}
        size_label = f"{d}x{h}x{w}"
    else:
        if len(args.input_size) != 2:
            parser.error(f"--input-size for {arch} must be HxW, got {'x'.join(map(str, args.input_size))}")
        h, w = args.input_size
        input_shape = (1, 1, h, w)
        dynamic_axes = {"input": {0: "batch", 2: "height", 3: "width"},
                        "output": {0: "batch", 2: "height", 3: "width"}}
        size_label = f"{h}x{w}"
    dummy = torch.randn(*input_shape)
    try:
        # legacy exporter: most compatible path for GroupNorm + dynamic axes,
        # avoids the dynamo exporter's dependency on the optional onnxscript package
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as exc:
        print(f"[export] FAILED: {exc}", flush=True)
        return 1

    import onnx
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    if args.fp16:
        try:
            from onnxconverter_common import float16 as to_fp16
        except ImportError as exc:
            print(f"[export] --fp16 requires onnxconverter-common: {exc}", flush=True)
            return 1
        onnx_model = to_fp16.convert_float_to_float16(onnx_model, keep_io_types=True)
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(output))
        print(f"[export] precision  : weights converted to FP16 (I/O stays FP32)", flush=True)

    print(f"[export] saved       : {output} ({output.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"[export] ir_version  : {onnx_model.ir_version}, opset {onnx_model.opset_import[0].version}", flush=True)
    print(f"[export] inputs      : {[(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input]}", flush=True)
    print(f"[export] outputs     : {[(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in onnx_model.graph.output]}", flush=True)

    if not args.no_validate:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            print(f"[validate] SKIP (onnxruntime not installed: {exc})", flush=True)
            return 0
        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).normal(size=input_shape).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        t0 = time.perf_counter()
        actual = sess.run(None, {"input": x})[0]
        elapsed = (time.perf_counter() - t0) * 1000
        diff = np.abs(actual - expected)
        print(f"[validate] onnxruntime OK  (cpu {elapsed:.1f} ms for {size_label})", flush=True)
        print(f"[validate] max abs diff   : {diff.max():.3e}  (rel {float(diff.max()) / max(float(np.abs(expected).max()), 1e-12):.3e})", flush=True)
        ok = diff.max() < 1e-4 * max(float(np.abs(expected).max()), 1.0)
        print(f"[validate] {'PASS' if ok else 'FAIL'} (torch vs onnxruntime)", flush=True)
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
