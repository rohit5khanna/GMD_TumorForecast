"""Extended drifting loss utilities for incremental MVP ablations."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F


class DriftFeatureBank:
    """Simple FIFO memory bank for stronger negatives across batches."""

    def __init__(self, max_items: int = 0):
        self.max_items = max(0, int(max_items))
        self._storage: torch.Tensor | None = None

    def add(self, feats: torch.Tensor) -> None:
        if self.max_items <= 0:
            return
        feats = feats.detach()
        if self._storage is None:
            self._storage = feats[-self.max_items :].clone()
            return
        bank = self._storage.to(feats.device)
        bank = torch.cat([bank, feats], dim=0)[-self.max_items :]
        self._storage = bank.detach()

    def sample(self, n_items: int) -> torch.Tensor | None:
        if self._storage is None or self._storage.shape[0] == 0:
            return None
        bank = self._storage
        n = min(int(n_items), int(bank.shape[0]))
        if n <= 0:
            return None
        if n == bank.shape[0]:
            return bank
        idx = torch.randperm(bank.shape[0], device=bank.device)[:n]
        return bank[idx]


def _normalize_pool_scales(pool_scales: int | Iterable[int] | None, fallback: int = 4) -> list[int]:
    if pool_scales is None:
        return [max(1, int(fallback))]
    if isinstance(pool_scales, int):
        return [max(1, int(pool_scales))]
    out = [max(1, int(s)) for s in pool_scales]
    return out if out else [max(1, int(fallback))]


def pooled_feature_map_multiscale(
    volume: torch.Tensor,
    pool_scales: int | Iterable[int] | None = 4,
) -> torch.Tensor:
    """
    Convert a 3D volume into a compact multiscale feature vector.
    Returns [B, C * sum(scale^3)].
    """
    feats = []
    for s in _normalize_pool_scales(pool_scales):
        pooled = F.adaptive_avg_pool3d(volume, output_size=(s, s, s))
        feats.append(pooled.flatten(start_dim=1))
    return torch.cat(feats, dim=1)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
    tau = max(float(temperature), 1e-8)
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist_sq / tau)


def compute_drifting_field(
    x_gen: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute drifting field:
      V = pos_weight * V_pos - neg_weight * V_neg
    """
    tau = max(float(temperature), 1e-8)
    k_pos = _rbf_kernel(x_gen, y_pos, tau)
    k_neg = _rbf_kernel(x_gen, y_neg, tau)

    w_pos = F.softmax(k_pos / tau, dim=1)
    w_neg = F.softmax(k_neg / tau, dim=1)

    mu_pos = w_pos @ y_pos
    mu_neg = w_neg @ y_neg

    v_pos = mu_pos - x_gen
    v_neg = mu_neg - x_gen
    return float(pos_weight) * v_pos - float(neg_weight) * v_neg


def _boundary_weight_map(mask: torch.Tensor, boundary_gamma: float) -> torch.Tensor:
    if float(boundary_gamma) <= 0.0:
        return torch.ones_like(mask)
    # Morphological gradient approximation.
    dil = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool3d(-mask, kernel_size=3, stride=1, padding=1)
    boundary = (dil - ero).clamp(min=0.0, max=1.0)
    return 1.0 + float(boundary_gamma) * boundary


def _delta_t_scale(
    delta_t_map: torch.Tensor | None,
    batch_size: int,
    delta_t_beta: float,
    delta_t_center: float,
    device: torch.device,
) -> torch.Tensor:
    if delta_t_map is None or float(delta_t_beta) == 0.0:
        return torch.ones(batch_size, device=device)
    dt = delta_t_map.float().mean(dim=(1, 2, 3, 4))
    scale = 1.0 + float(delta_t_beta) * (dt - float(delta_t_center))
    return torch.clamp(scale, min=0.1)


def drifting_loss_from_logits(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    temperature: float = 0.1,
    pool_size: int = 4,
    pool_scales: int | Iterable[int] | None = None,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    boundary_gamma: float = 0.0,
    delta_t_map: torch.Tensor | None = None,
    delta_t_beta: float = 0.0,
    delta_t_center: float = 0.6,
    feature_bank: DriftFeatureBank | None = None,
    pred_feature_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Build drifting loss with optional extensions:
    - stronger negatives via memory bank
    - multiscale pooled features
    - separate pos/neg weighting
    - boundary emphasis
    - delta_t-aware scaling
    - latent-feature source (pred_feature_map) instead of mask probabilities
    """
    if pool_scales is None:
        pool_scales = [pool_size]
    else:
        pool_scales = _normalize_pool_scales(pool_scales, fallback=pool_size)

    if pred_feature_map is None:
        pred_volume = torch.sigmoid(logits)
    else:
        pred_volume = pred_feature_map

    target_mask = target_mask.float()
    bsz = int(target_mask.shape[0])
    device = target_mask.device

    # Boundary-aware feature emphasis.
    w_boundary = _boundary_weight_map(target_mask, boundary_gamma=boundary_gamma)
    pred_weighted = pred_volume * w_boundary
    target_weighted = target_mask * w_boundary
    if target_weighted.shape[1] != pred_weighted.shape[1]:
        target_weighted = target_weighted.repeat(1, pred_weighted.shape[1], 1, 1, 1)

    feat_gen = pooled_feature_map_multiscale(pred_weighted, pool_scales=pool_scales)
    x_old = feat_gen.detach()
    feat_pos = pooled_feature_map_multiscale(target_weighted, pool_scales=pool_scales).detach()

    # In-batch negatives.
    if x_old.shape[0] > 1:
        feat_neg = torch.roll(x_old, shifts=1, dims=0)
    else:
        feat_neg = x_old

    # Optional memory-bank negatives (stronger than pure in-batch for small B).
    if feature_bank is not None:
        bank_neg = feature_bank.sample(n_items=int(x_old.shape[0]))
        if bank_neg is not None:
            bank_neg = bank_neg.to(x_old.device)
            feat_neg = torch.cat([feat_neg, bank_neg], dim=0)

    v = compute_drifting_field(
        x_gen=x_old,
        y_pos=feat_pos,
        y_neg=feat_neg,
        temperature=temperature,
        pos_weight=pos_weight,
        neg_weight=neg_weight,
    )

    goal = x_old + v.detach()
    per_sample = (feat_gen - goal).pow(2).mean(dim=1)

    # Optional delta_t-aware weighting.
    dt_scale = _delta_t_scale(
        delta_t_map=delta_t_map,
        batch_size=bsz,
        delta_t_beta=delta_t_beta,
        delta_t_center=delta_t_center,
        device=device,
    )
    loss = (per_sample * dt_scale).mean()

    if feature_bank is not None:
        feature_bank.add(x_old)

    stats = {
        "drift_field_l2": float(v.pow(2).mean().item()),
        "dt_scale_mean": float(dt_scale.mean().item()),
    }
    return loss, stats
