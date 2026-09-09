"""3D residual UNet with depthwise-separable convolutions.

Drop-in alternative to :class:`cbct_n2n.models.UNet2D` for true volumetric
processing.  The architecture mirrors ``UNet2D`` so configs that already
describe a 2D model (in_channels / out_channels / base_channels / depth /
residual) can be reused verbatim.  Two design choices differ on purpose:

* Every conv block is **depthwise-separable**: a per-channel 3x3x3 spatial
  filter (groups = in_channels) followed by a 1x1x1 channel mixer.  This
  cuts parameters and FLOPs roughly 8x versus a vanilla Conv3d at the same
  channel count, which matters for 3D where parameter counts balloon.
* Both a **block-local** residual (input -> block output, with 1x1x1
  projection when channels change) and the standard **global** residual
  (input -> final UNet output) are kept.  This matches the N2N-style
  ``UNet2D`` behaviour where the network learns the noise / delta and the
  clean target is ``denoised + source``.

Pooling is **fixed**: every encoder stage downsamples by 2 along all three
axes with ``max_pool3d(kernel_size=2)`` and the decoder upsamples with a
matching ``ConvTranspose3d(kernel_size=2, stride=2)`` -- exactly the
UNet2D/UNet2D_DS schedule lifted to 3D.  The D (depth / slice) axis is
therefore treated identically to H and W: the output volume has the **same
D length as the input** (full-volume denoising, no centre-slice collapse).

Consequence of fixed 2x pooling: the input depth must be large enough to
survive ``depth`` rounds of halving, i.e. ``D >= 2 ** depth`` (the default
``D=22`` comfortably satisfies ``base_channels=8, depth=3``).  Non-power-of-
two lengths are still accepted -- ``_align`` absorbs the off-by-one that
``ConvTranspose3d`` introduces -- as long as the minimum length holds.
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


class DepthwiseSeparableConv3d(nn.Module):
    """One depthwise-separable 3D convolution + norm + SiLU.

    * depthwise: ``Conv3d(in, in, 3, padding=1, groups=in, bias=False)``
    * pointwise: ``Conv3d(in, out, 1, bias=False)``
    * norm: ``GroupNorm`` with the same heuristic as :class:`UNet2D`
    * act: ``SiLU(inplace=True)``
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("channel counts must be positive")
        # depthwise requires groups == in_channels; valid for any in_channels.
        self.depthwise = nn.Conv3d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(_norm_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class DSConvBlock3d(nn.Module):
    """Two depthwise-separable 3x3x3 convs with a local residual.

    The block learns ``f(x)`` and returns ``x + f(x)``; when ``in != out`` a
    1x1x1 projection aligns the residual path.  This is the 3D analogue of
    the double-conv block used by :class:`UNet2D` plus a per-block skip.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv3d(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv3d(out_channels, out_channels)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class UNet3D(nn.Module):
    """3D residual UNet with depthwise-separable conv blocks and fixed 2x pooling.

    Parameters
    ----------
    in_channels, out_channels:
        Number of channels on the input/target volume (typically 1).
    base_channels:
        Width of the first encoder stage.  Each subsequent encoder stage
        doubles the channel count; the bottleneck doubles once more.
        Default ``8`` keeps parameters small for volumetric inputs.
    depth:
        Number of encoder/decoder stages (>=2).  Default ``3`` -> encoder
        features ``[8, 16, 32]`` with a ``[64]`` bottleneck.  Every stage
        pools by 2 along D/H/W, so the input depth must satisfy
        ``D >= 2 ** depth``.
    residual:
        If ``True``, the network learns the delta and the final output is
        ``unet(source) + source`` (channel-aligned with a 1x1x1 conv when
        ``in_channels != out_channels``).  Mirrors :class:`UNet2D`.

    Input / output are both ``[B, C, D, H, W]`` with the **same** D length:
    the model denoises the whole input volume, not a single centre slice.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
        depth: int = 3,
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
            self.encoders.append(DSConvBlock3d(previous, channels))
            previous = channels

        self.bottleneck = DSConvBlock3d(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = features[-1] * 2
        for channels in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose3d(previous, channels, kernel_size=2, stride=2)
            )
            # decoder input = upconv(c) concatenated with the matching skip (c) -> 2c
            self.decoders.append(DSConvBlock3d(channels * 2, channels))
            previous = channels

        self.head = nn.Conv3d(features[0], out_channels, 1)
        # global skip: project input to output channels when they differ.
        self.global_skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if residual and in_channels != out_channels
            else nn.Identity()
        )

    @staticmethod
    def _align(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Crop / pad ``source`` to match ``reference``'s spatial dims.

        Mirrors :meth:`UNet2D._align` but extends to 3 axes.  The decoder
        uses this to absorb off-by-one size mismatches introduced by
        ``ConvTranspose3d`` when the input volume isn't an exact power of
        two along every axis.
        """
        dd = reference.shape[-3] - source.shape[-3]
        dh = reference.shape[-2] - source.shape[-2]
        dw = reference.shape[-1] - source.shape[-1]
        if dd or dh or dw:
            source = F.pad(
                source,
                (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2, dd // 2, dd - dd // 2),
            )
        return source[..., : reference.shape[-3], : reference.shape[-2], : reference.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"UNet3D expects a 5D tensor [B, C, D, H, W]; got shape {tuple(x.shape)}")
        source = x
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
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


__all__ = ["UNet3D", "DSConvBlock3d", "DepthwiseSeparableConv3d"]
