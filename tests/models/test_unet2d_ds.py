import pytest
import torch

from cbct_n2n.models.unet2d_ds import (
    DSConvBlock2d,
    DepthwiseSeparableConv2d,
    UNet2D_DS,
)
from cbct_n2n.models import UNet2D


@pytest.mark.parametrize("shape", [
    (1, 1, 64, 64),
    (2, 1, 64, 96),
    (1, 1, 47, 53),
])
def test_unet2d_ds_preserves_shape_and_backpropagates(shape):
    model = UNet2D_DS(base_channels=8, depth=3)
    source = torch.randn(shape, requires_grad=True)
    output = model(source)
    assert output.shape == shape
    output.mean().backward()
    assert source.grad is not None


def test_unet2d_ds_default_architecture_matches_base16_depth4_spec():
    model = UNet2D_DS()
    assert model.in_channels == 1
    assert model.out_channels == 1
    assert model.residual is True
    n = model.count_parameters()
    # base16/depth4 depthwise-separable ~= 0.47M params (benchmarked).
    assert 300_000 < n < 700_000, f"expected ~0.47M params, got {n:,}"


def test_unet2d_ds_is_lighter_than_standard_conv_at_same_config():
    ds = UNet2D_DS(base_channels=16, depth=4).count_parameters()
    std = sum(p.numel() for p in UNet2D(base_channels=16, depth=4).parameters())
    # Depthwise-separable should be a small fraction of the standard conv UNet.
    assert ds < std / 3, f"DS params {ds:,} not < 1/3 of standard {std:,}"


def test_unet2d_ds_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        UNet2D_DS(base_channels=0)
    with pytest.raises(ValueError):
        UNet2D_DS(depth=1)


def test_unet2d_ds_rejects_non_4d_input():
    model = UNet2D_DS(base_channels=8, depth=3)
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 5, 32, 32))  # 5D, volume-shaped


def test_unet2d_ds_residual_off_changes_output_distribution():
    torch.manual_seed(0)
    with_residual = UNet2D_DS(base_channels=8, depth=3, residual=True)
    torch.manual_seed(0)
    without_residual = UNet2D_DS(base_channels=8, depth=3, residual=False)
    x = torch.randn(1, 1, 32, 32)
    assert not torch.allclose(with_residual(x), without_residual(x))


def test_depthwise_separable_conv_uses_groups_equal_to_in_channels():
    conv = DepthwiseSeparableConv2d(in_channels=12, out_channels=24)
    assert conv.depthwise.groups == 12
    assert conv.pointwise.kernel_size == (1, 1)
    assert conv.pointwise.in_channels == 12
    assert conv.pointwise.out_channels == 24


def test_depthwise_separable_conv_preserves_shape_when_padding_compensates():
    conv = DepthwiseSeparableConv2d(in_channels=8, out_channels=16)
    x = torch.randn(2, 8, 24, 24)
    assert conv(x).shape == (2, 16, 24, 24)


def test_dsconv_block_local_residual_aligns_channels_when_they_differ():
    block = DSConvBlock2d(in_channels=4, out_channels=8)
    assert isinstance(block.skip, torch.nn.Conv2d)
    assert block.skip.kernel_size == (1, 1)
    assert block.skip.in_channels == 4 and block.skip.out_channels == 8


def test_dsconv_block_uses_identity_skip_when_channels_match():
    block = DSConvBlock2d(in_channels=8, out_channels=8)
    assert isinstance(block.skip, torch.nn.Identity)
    x = torch.randn(1, 8, 16, 16)
    assert block(x).shape == x.shape


def test_unet2d_ds_handles_non_power_of_two_spatial_dims():
    model = UNet2D_DS(base_channels=16, depth=4)
    x = torch.randn(1, 1, 47, 53)
    out = model(x)
    assert out.shape == x.shape


def test_unet2d_ds_accepts_base_channels_not_divisible_by_eight():
    model = UNet2D_DS(base_channels=10, depth=2)
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 1, 32, 32)
