import numpy as np
import torch

from cbct_n2n.training.inference import (
    infer_volume,
    infer_volume_3d,
    resolve_input_kinds,
    write_raw_volume,
)


def test_identity_inference_restores_shape_and_intensity():
    volume = np.linspace(-2, 4.5, 2 * 33 * 35, dtype=np.float32).reshape(2, 33, 35)
    output = infer_volume(torch.nn.Identity(), volume, torch.device("cpu"), batch_size=2)
    assert output.shape == volume.shape
    np.testing.assert_allclose(output, volume, atol=1e-5)


def test_infer_volume_3d_identity_restores_shape_and_intensity():
    # depth not divisible by voxel_depth (zero-padded tail) + non-32-aligned H/W
    volume = np.linspace(-3, 5.0, 5 * 33 * 35, dtype=np.float32).reshape(5, 33, 35)
    output = infer_volume_3d(torch.nn.Identity(), volume, voxel_depth=2,
                             device=torch.device("cpu"), batch_size=2)
    assert output.shape == volume.shape
    np.testing.assert_allclose(output, volume, atol=1e-5)


def test_resolve_input_kinds_expands_both_and_preserves_single_kind():
    assert resolve_input_kinds("both") == ("even", "odd")
    assert resolve_input_kinds("even") == ("even",)
    assert resolve_input_kinds("odd") == ("odd",)


def test_write_raw_volume_writes_little_endian_float32_without_header(tmp_path):
    volume = np.array([[[1.25, -2.5]]], dtype=np.float64)
    destination = tmp_path / "volume.raw"

    write_raw_volume(destination, volume)

    assert destination.read_bytes() == volume.astype("<f4").tobytes()
