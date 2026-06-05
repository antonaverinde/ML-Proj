"""Models used by the autoencoder + UNet anomaly pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """Small bottleneck autoencoder for clean thermal-feature reconstruction."""

    def __init__(self, in_channels: int, base_channels: int = 32, latent_channels: int = 128):
        super().__init__()
        self.enc1 = _conv_block(in_channels, base_channels)
        self.enc2 = _conv_block(base_channels, base_channels * 2)
        self.bottleneck = _conv_block(base_channels * 2, latent_channels)

        self.up2 = nn.ConvTranspose2d(latent_channels, base_channels * 2, 2, stride=2)
        self.dec2 = _conv_block(base_channels * 2, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = _conv_block(base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_in, w_in = x.shape[2], x.shape[3]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        z = self.bottleneck(self.pool(e2))

        u2 = self.up2(z)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        d1 = self.dec1(u1)
        y = self.out(d1)
        if y.shape[2:] != (h_in, w_in):
            y = F.interpolate(y, size=(h_in, w_in), mode="bilinear", align_corners=False)
        return y
