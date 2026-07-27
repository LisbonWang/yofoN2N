import pytest
import torch

from cbct_n2n.models import UNet2D


@pytest.mark.parametrize("shape", [(1, 1, 64, 64), (2, 1, 64, 96)])
def test_unet_preserves_shape_and_backpropagates(shape):
    model = UNet2D(base_channels=8, depth=3)
    source = torch.randn(shape, requires_grad=True)
    output = model(source)
    assert output.shape == source.shape
    output.mean().backward()
    assert source.grad is not None


def test_unet_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        UNet2D(base_channels=0)
    with pytest.raises(ValueError):
        UNet2D(depth=1)


def test_unet_accepts_base_channels_not_divisible_by_eight():
    model = UNet2D(base_channels=10, depth=2)
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 1, 32, 32)
