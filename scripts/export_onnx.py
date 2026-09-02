"""Export a trained UNet2D checkpoint to ONNX with dynamic input size.

Usage:
    python scripts/export_onnx.py --checkpoint outputs/demo/best.pt
    python scripts/export_onnx.py --checkpoint outputs/demo/best.pt --input-size 512x512 --opset 17
    python scripts/export_onnx.py --checkpoint outputs/demo/best.pt --fp16   # half-size ONNX

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

from cbct_n2n.models import UNet2D  # noqa: E402


def load_model(checkpoint: str | Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("config", {}).get("model", {})
    if not model_cfg:
        raise ValueError(f"checkpoint {checkpoint} has no config['model']; cannot rebuild architecture")
    model = UNet2D(**model_cfg)
    state = ckpt["model"]
    # strip DataParallel / DDP prefixes if present
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def parse_size(text: str) -> tuple[int, int]:
    try:
        h, w = (int(v) for v in text.lower().split("x"))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"input size must be HxW, got {text!r}")
    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError(f"input size must be positive, got {text!r}")
    return h, w


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/demo/best.pt"),
                        help="checkpoint .pt file (default: outputs/demo/best.pt)")
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
    print(f"[export] model arch  : UNet2D from config['model']", flush=True)

    model, ckpt = load_model(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] parameters : {n_params:,}  (epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('metrics', {}).get('val_loss', 'n/a')})", flush=True)

    h, w = args.input_size
    dummy = torch.randn(1, 1, h, w)
    try:
        # legacy exporter: most compatible path for GroupNorm + dynamic axes,
        # avoids the dynamo exporter's dependency on the optional onnxscript package
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                          "output": {0: "batch", 2: "height", 3: "width"}},
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
        x = np.random.default_rng(0).normal(size=(1, 1, h, w)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        t0 = time.perf_counter()
        actual = sess.run(None, {"input": x})[0]
        elapsed = (time.perf_counter() - t0) * 1000
        diff = np.abs(actual - expected)
        print(f"[validate] onnxruntime OK  (cpu {elapsed:.1f} ms for {h}x{w})", flush=True)
        print(f"[validate] max abs diff   : {diff.max():.3e}  (rel {float(diff.max()) / max(float(np.abs(expected).max()), 1e-12):.3e})", flush=True)
        ok = diff.max() < 1e-4 * max(float(np.abs(expected).max()), 1.0)
        print(f"[validate] {'PASS' if ok else 'FAIL'} (torch vs onnxruntime)", flush=True)
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
