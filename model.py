"""One-shot predictor model(s)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OneShotPredictor(nn.Module):
    """
    Input:  [B, 2, H, W, D]  -> baseline + delta_t_map
    Output: [B, 1, H, W, D]  -> logits for future tumor mask
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        self.head = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)
        if return_features:
            return logits, {"dec1": d1, "bottleneck": b}
        return logits


class ResidualRefinePredictor(nn.Module):
    """
    Two-stage one-shot predictor:
    1) coarse logits from baseline U-Net
    2) residual refinement logits from (dec1 features + baseline mask + coarse probs)
    Final logits = coarse + residual.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        self.coarse_model = OneShotPredictor(in_channels=in_channels, base_channels=base_channels)
        self.refine_block = ConvBlock3D(base_channels + 2, base_channels)
        self.refine_head = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coarse_logits, coarse_feats = self.coarse_model(x, return_features=True)
        coarse_probs = torch.sigmoid(coarse_logits)
        baseline = x[:, :1]

        refine_in = torch.cat([coarse_feats["dec1"], baseline, coarse_probs], dim=1)
        refine_feat = self.refine_block(refine_in)
        refine_logits = self.refine_head(refine_feat)

        logits = coarse_logits + refine_logits
        if return_features:
            out_feats = dict(coarse_feats)
            out_feats["coarse_logits"] = coarse_logits
            out_feats["refine_feat"] = refine_feat
            return logits, out_feats
        return logits


def build_model(cfg: dict, in_channels: int = 2) -> nn.Module:
    model_type = str(cfg.get("model_type", "unet_baseline")).lower()
    base_channels = int(cfg.get("base_channels", 16))

    if model_type in {"unet_baseline", "baseline", "oneshot"}:
        return OneShotPredictor(in_channels=in_channels, base_channels=base_channels)
    if model_type in {"unet_residual_refine", "residual_refine", "refine"}:
        return ResidualRefinePredictor(in_channels=in_channels, base_channels=base_channels)
    raise ValueError(
        "Unknown model_type. Supported: "
        "{'unet_baseline','unet_residual_refine'}"
    )


if __name__ == "__main__":
    model = ResidualRefinePredictor(in_channels=2, base_channels=8)
    x = torch.randn(2, 2, 64, 64, 64)
    y = model(x)
    print("input:", x.shape)
    print("logits:", y.shape)
        
