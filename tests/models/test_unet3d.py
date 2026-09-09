import pytest
import torch

from cbct_n2n.models.unet3d import (
    DSConvBlock3d,
    DepthwiseSeparableConv3d,
    UNet3D,
)


def _make_volume(batch: int = 1, channels: int = 1, depth: int = 16,
                 height: int = 64, width: int = 64) -> torch.Tensor:
    return torch.randn(batch, channels, depth, height, width)


@pytest.mark.parametrize("shape", [
    (1, 1, 16, 64, 64),
    (2, 1, 8, 96, 64),
    (1, 1, 32, 32, 32),
])
def test_unet3d_preserves_shape_and_backpropagates(shape):
    model = UNet3D(base_channels=8, depth=3)
    source = torch.randn(shape, requires_grad=True)
    output = model(source)
    assert output.shape == shape
    output.mean().backward()
    assert source.grad is not None


def test_unet3d_default_architecture_matches_base8_depth3_spec():
    # spec: base_channels=8, depth=3 -> encoder features [8, 16, 32], bottleneck 64
    model = UNet3D()
    assert model.in_channels == 1
    assert model.out_channels == 1
    assert model.residual is True
    param_count = model.count_parameters()
    # base8/depth3 should stay well under 200k parameters to be a "light" 3D model.
    assert param_count < 200_000, f"UNet3D base8/depth3 has {param_count:,} params, expected <200k"
    assert param_count > 1_000, f"UNet3D base8/depth3 has only {param_count:,} params, looks broken"


def test_unet3d_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        UNet3D(base_channels=0)
    with pytest.raises(ValueError):
        UNet3D(depth=1)


def test_unet3d_rejects_non_5d_input():
    model = UNet3D(base_channels=8, depth=3)
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 32, 32))  # 4D, slice-shaped


def test_unet3d_residual_off_changes_output_distribution():
    torch.manual_seed(0)
    with_residual = UNet3D(base_channels=8, depth=3, residual=True)
    torch.manual_seed(0)
    without_residual = UNet3D(base_channels=8, depth=3, residual=False)
    # Both initialised from the same RNG seed -> same conv weights; only the
    # global skip path differs (head projection + Identity).
    x = _make_volume()
    out_with = with_residual(x)
    out_without = without_residual(x)
    # Residual path adds the input back, so output magnitudes differ.
    assert not torch.allclose(out_with, out_without)


def test_depthwise_separable_conv_uses_groups_equal_to_in_channels():
    conv = DepthwiseSeparableConv3d(in_channels=12, out_channels=24)
    assert conv.depthwise.groups == 12
    assert conv.pointwise.kernel_size == (1, 1, 1)
    assert conv.pointwise.in_channels == 12
    assert conv.pointwise.out_channels == 24


def test_depthwise_separable_conv_preserves_shape_when_padding_compensates():
    conv = DepthwiseSeparableConv3d(in_channels=8, out_channels=16)
    x = torch.randn(2, 8, 12, 24, 24)
    assert conv(x).shape == (2, 16, 12, 24, 24)


def test_dsconv_block_local_residual_aligns_channels_when_they_differ():
    block = DSConvBlock3d(in_channels=4, out_channels=8)
    # When channels differ, the skip path must be a 1x1x1 conv (not Identity).
    assert isinstance(block.skip, torch.nn.Conv3d)
    assert block.skip.kernel_size == (1, 1, 1)
    assert block.skip.in_channels == 4 and block.skip.out_channels == 8


def test_dsconv_block_uses_identity_skip_when_channels_match():
    block = DSConvBlock3d(in_channels=8, out_channels=8)
    assert isinstance(block.skip, torch.nn.Identity)
    x = torch.randn(1, 8, 16, 16, 16)
    assert block(x).shape == x.shape


def test_unet3d_handles_non_power_of_two_dims():
    # Fixed 2x pooling + _align should preserve shape for non-power-of-two
    # D/H/W as long as D >= 2**depth (9 >= 8 for depth=3).
    model = UNet3D(base_channels=8, depth=3)
    x = torch.randn(1, 1, 9, 47, 53)
    out = model(x)
    assert out.shape == x.shape


# ---- fixed 2x pooling / same-D-length output ------------------------------


def test_unet3d_output_depth_equals_input_depth():
    """The model is full-volume: output D length matches input D length."""
    model = UNet3D(base_channels=8, depth=3)
    for depth in (9, 11, 16, 20, 32):
        x = torch.randn(1, 1, depth, 32, 32)
        out = model(x)
        assert out.shape[2] == depth, f"input D={depth}, output D={out.shape[2]}"


def test_unet3d_fixed_pooling_requires_minimum_depth():
    """D < 2**depth can't survive depth rounds of halving -> max_pool3d errors."""
    model = UNet3D(base_channels=8, depth=3)
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 1, 4, 32, 32))  # 4 < 8 = 2**3


def test_unet3d_no_voxel_depth_parameter():
    """voxel_depth is gone: UNet3D no longer accepts it (fixed pooling)."""
    with pytest.raises(TypeError):
        UNet3D(voxel_depth=16)
