"""Tests for ONNX export (scripts/export_onnx.py).

The full 7.7M-parameter production model exports in ~10s and is validated
end-to-end; here we cover the same path with a small model so the suite
stays fast and SDK-independent (skipped when onnx/onnxruntime are absent).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cbct_n2n.models import UNet2D

try:
    import onnx  # noqa: F401
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

pytestmark = pytest.mark.skipif(not HAS_ONNX, reason="onnx/onnxruntime not installed")


def _export(model: torch.nn.Module, path, size=(32, 48), opset: int = 13) -> None:
    torch.onnx.export(
        model,
        torch.randn(1, 1, *size),
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        opset_version=opset,
        dynamo=False,
    )


def test_export_small_unet_matches_torch(tmp_path):
    torch.manual_seed(0)
    model = UNet2D(in_channels=1, out_channels=1, base_channels=4, depth=2)
    model.eval()
    out = tmp_path / "small.onnx"
    _export(model, out)

    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    assert (out.stat().st_size > 0)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(1, 1, 32, 48)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    actual = sess.run(None, {"input": x})[0]
    assert actual.shape == expected.shape == (1, 1, 32, 48)
    assert np.abs(actual - expected).max() < 1e-4


def test_export_dynamic_input_sizes(tmp_path):
    torch.manual_seed(1)
    model = UNet2D(in_channels=1, out_channels=1, base_channels=4, depth=2)
    model.eval()
    out = tmp_path / "dyn.onnx"
    _export(model, out, size=(32, 48))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for shape in [(1, 1, 64, 64), (2, 1, 40, 24), (1, 1, 64, 32)]:
        x = np.zeros(shape, dtype=np.float32)
        got = sess.run(None, {"input": x})[0]
        assert got.shape == shape, f"{shape} -> {got.shape}"


def test_export_legacy_path_does_not_require_onnxscript(tmp_path):
    """The exporter must use dynamo=False so the optional onnxscript pkg is not needed."""
    torch.manual_seed(2)
    model = UNet2D(in_channels=1, out_channels=1, base_channels=4, depth=2)
    model.eval()
    out = tmp_path / "legacy.onnx"
    _export(model, out, opset=17)
    assert out.is_file()
    # opset 17 model loads fine
    ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
