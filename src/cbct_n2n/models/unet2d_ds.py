"""2D residual UNet with depthwise-separable convolutions.

Drop-in alternative to :class:`cbct_n2n.models.UNet2D` that trades the
standard double-conv block for a depthwise-separable one, exactly the way
:class:`cbct_n2n.models.unet3d.UNet3D` does in 3D.  Keeping the 2D and 3D
models on the same depthwise-separable footing makes the two architectures
directly comparable: any remaining FLOPs/parameter gap then reflects the
extra slice axis alone, not the conv flavour.

The block is a 1:1 port of ``UNet3D``'s ``DSConvBlock3d`` to 2D:

* ``DepthwiseSeparableConv2d``: per-channel 3x3 spatial filter
  (``groups = in_channels``) followed by a 1x1 channel mixer.
* ``DSConvBlock2d``: two depthwise-separable convs plus a **block-local**
  residual (1x1 projection when channels change).
* ``UNet2D_DS``: standard encoder / bottleneck / decoder layout with the
  **global** residual ``unet(source) + source`` on top, mirroring
  :class:`UNet2D` so it can be dropped into the same config verbatim.

The default ``base_channels=16, depth=4`` matches the benchmarked config
(``~0.47M`` params / ``~6.9 G`` MACs at 640x640 vs. ``~1.94M`` / ``~21.4 G``
for the standard-conv ``UNet2D``).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _norm_groups(out_channels: int) -> int:
    """Mirror UNet2D's GroupNorm heuristic: at most 8 groups, divisible by C."""
    groups = min(8, out_channels)
    while out_channels % groups:
        groups -= 1
    return max(groups, 1)


class DepthwiseSeparableConv2d(nn.Module):
    """One depthwise-separable 2D convolution + norm + SiLU.

    * depthwise: ``Conv2d(in, in, 3, padding=1, groups=in, bias=False)``
    * pointwise: ``Conv2d(in, out, 1, bias=False)``
    * norm: ``GroupNorm`` with the same heuristic as :class:`UNet2D`
    * act: ``SiLU(inplace=True)``
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("channel counts must be positive")
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(_norm_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class DSConvBlock2d(nn.Module):
    """Two depthwise-separable 3x3 convs with a local residual.

    The block learns ``f(x)`` and returns ``x + f(x)``; when ``in != out`` a
    1x1 projection aligns the residual path.  This is the depthwise-separable
    analogue of the double-conv block used by :class:`UNet2D` plus a
    per-block skip, matching :class:`cbct_n2n.models.unet3d.DSConvBlock3d`.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class UNet2D_DS(nn.Module):
    """2D residual UNet with depthwise-separable conv blocks.

    Parameters
    ----------
    in_channels, out_channels:
        Number of channels on the input/target slice (typically 1).
    base_channels:
        Width of the first encoder stage.  Each subsequent encoder stage
        doubles the channel count; the bottleneck doubles once more.
        Default ``16`` keeps the model light; ``base16/depth4`` ~= 0.47M
        parameters.
    depth:
        Number of encoder/decoder stages (>=2).  Default ``4`` -> encoder
        features ``[16, 32, 64, 128]`` with a ``[256]`` bottleneck.
    residual:
        If ``True``, the network learns the delta and the final output is
        ``unet(source) + source`` (channel-aligned with a 1x1 conv when
        ``in_channels != out_channels``).  Mirrors :class:`UNet2D`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1 or base_channels < 1:
            raise ValueError("channel counts must be positive")
        if depth < 2:
            raise ValueError("depth must be at least 2")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        features = [base_channels * 2 ** i for i in range(depth)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for channels in features:
            self.encoders.append(DSConvBlock2d(previous, channels))
            previous = channels

        self.bottleneck = DSConvBlock2d(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = features[-1] * 2
        for channels in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(previous, channels, 2, stride=2))
            # decoder input = upconv(c) concatenated with the matching skip (c) -> 2c
            self.decoders.append(DSConvBlock2d(channels * 2, channels))
            previous = channels

        self.head = nn.Conv2d(features[0], out_channels, 1)
        self.global_skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if residual and in_channels != out_channels
            else nn.Identity()
        )

    @staticmethod
    def _align(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Crop / pad ``source`` to match ``reference``'s spatial dims."""
        dh = reference.shape[-2] - source.shape[-2]
        dw = reference.shape[-1] - source.shape[-1]
        if dh or dw:
            source = F.pad(source, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return source[..., : reference.shape[-2], : reference.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"UNet2D_DS expects a 4D tensor [B, C, H, W]; got shape {tuple(x.shape)}")
        source = x
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = self._align(upconv(x), skip)
            x = decoder(torch.cat((skip, x), dim=1))
        result = self.head(x)
        if self.residual:
            return result + self.global_skip(source)
        return result

    def count_parameters(self) -> int:
        """Convenience: total trainable parameters (handy for smoke checks)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["UNet2D_DS", "DSConvBlock2d", "DepthwiseSeparableConv2d"]
