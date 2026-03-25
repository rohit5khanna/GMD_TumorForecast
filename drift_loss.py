"""Simplified drifting loss (PyTorch) for MVP-1."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def pooled_feature_map(volume: torch.Tensor, pool_size: int = 4) -> torch.Tensor:
    """
    Convert a 3D volume into compact feature vectors.

    Args:
        volume: [B, C, H, W, D]
        pool_size: adaptive pooling target size per dimension

    Returns:
        features: [B, C * pool_size^3]
    """
    pooled = F.adaptive_avg_pool3d(volume, output_size=(pool_size, pool_size, pool_size))
    return pooled.flatten(start_dim=1)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
    tau = max(float(temperature), 1e-8)
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist_sq / tau)


def compute_drifting_field(
    x_gen: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Compute simplified drifting field:
      V = V_pos - V_neg
    where:
      V_pos attracts generated features to real (target) features
      V_neg repels generated features from generated reference features
    """
    k_pos = _rbf_kernel(x_gen, y_pos, temperature)
    k_neg = _rbf_kernel(x_gen, y_neg, temperature)

    # Softmax-normalized affinities.
    w_pos = F.softmax(k_pos / max(float(temperature), 1e-8), dim=1)
    w_neg = F.softmax(k_neg / max(float(temperature), 1e-8), dim=1)

    mu_pos = w_pos @ y_pos
    mu_neg = w_neg @ y_neg

    v_pos = mu_pos - x_gen
    v_neg = mu_neg - x_gen
    return v_pos - v_neg


def drifting_loss_from_logits(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    temperature: float = 0.1,
    pool_size: int = 4,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Build drifting loss from model output logits and target masks.

    Notes:
    - This is intentionally lightweight for MVP-1.
    - With batch size 1, repulsion collapses; attraction still provides signal.
    """
    probs = torch.sigmoid(logits)

    # Feature space for drifting objective.
    feat_gen = pooled_feature_map(probs, pool_size=pool_size)
    feat_pos = pooled_feature_map(target_mask.float(), pool_size=pool_size).detach()
    feat_neg = feat_gen.detach()

    v = compute_drifting_field(
        x_gen=feat_gen,
        y_pos=feat_pos,
        y_neg=feat_neg,
        temperature=temperature,
    )

    # Fixed-point style objective: ||x - stopgrad(x + V)||^2 = ||V||^2
    goal = feat_gen + v.detach()
    loss = F.mse_loss(feat_gen, goal)

    stats = {
        "drift_field_l2": float(v.pow(2).mean().item()),
    }
    return loss, stats
