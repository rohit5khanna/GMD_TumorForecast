"""One-shot predictor model."""

from __future__ import annotations

import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class OneShotPredictor(nn.Module):
    """
    Input:  [B, 2, H, W, D]  -> baseline + delta_t_map
    Output: [B, 1, H, W, D]  -> logits for future tumor mask
    """

    def __init__(self, in_ch: int=2, base_ch: int=16):
        super().__init__()

        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.pool1 = nn.MaxPool3D(2)

        self.enc2 = ConvBlock3D(in_ch, base_ch*2)
        self.pool2 = nn.MaxPool3D(2)

        self.bootleneck = ConvBlock3D(base_ch*2, base_ch*4)

        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_ch*2, base_ch)

        self.head = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bootleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)
        return logits


if __name__ == "__main__":
    model = OneShotPredictor(in_channels=2, base_channels=8)
    x = torch.randn(2, 2, 64, 64, 64)
    y = model(x)
    print("input:", x.shape)
    print("logits:", y.shape)








        
