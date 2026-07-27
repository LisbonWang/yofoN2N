from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, depth: int = 4, residual: bool = True):
        super().__init__()
        if in_channels < 1 or out_channels < 1 or base_channels < 1:
            raise ValueError("channel counts must be positive")
        if depth < 2:
            raise ValueError("depth must be at least 2")
        self.residual = residual
        features = [base_channels * 2**i for i in range(depth)]
        self.encoders = nn.ModuleList()
        previous = in_channels
        for channels in features:
            self.encoders.append(ConvBlock(previous, channels))
            previous = channels
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = features[-1] * 2
        for channels in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(previous, channels, 2, stride=2))
            self.decoders.append(ConvBlock(channels * 2, channels))
            previous = channels
        self.output = nn.Conv2d(features[0], out_channels, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if residual and in_channels != out_channels else nn.Identity()

    @staticmethod
    def _align(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        dh, dw = reference.shape[-2] - source.shape[-2], reference.shape[-1] - source.shape[-1]
        if dh or dw:
            source = F.pad(source, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return source[..., : reference.shape[-2], : reference.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = x
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = self._align(upconv(x), skip)
            x = decoder(torch.cat((skip, x), dim=1))
        result = self.output(x)
        return result + self.skip(source) if self.residual else result
