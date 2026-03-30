"""Extended drifting loss utilities for incremental MVP ablations."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
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


def _tokenize_volume(
    volume: torch.Tensor,
    patch_size: int = 4,
    stride: int | None = None,
) -> torch.Tensor:
    p = max(1, int(patch_size))
    s = p if stride is None else max(1, int(stride))
    pooled = F.avg_pool3d(volume, kernel_size=p, stride=s)
    b, c, h, w, d = pooled.shape
    return pooled.permute(0, 2, 3, 4, 1).reshape(b, h * w * d, c)


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


def local_token_drift_loss(
    pred_feature_map: torch.Tensor,
    target_mask: torch.Tensor,
    temperature: float = 0.1,
    patch_size: int = 4,
    token_stride: int | None = None,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    boundary_gamma: float = 0.0,
    delta_t_map: torch.Tensor | None = None,
    delta_t_beta: float = 0.0,
    delta_t_center: float = 0.6,
    feature_bank: DriftFeatureBank | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Local-token drift loss for Experiment G.
    - Works on local pooled tokens from latent feature maps.
    - Attraction/repulsion is computed in token feature space.
    """
    target_mask = target_mask.float()
    bsz = int(target_mask.shape[0])
    device = target_mask.device

    w_boundary = _boundary_weight_map(target_mask, boundary_gamma=boundary_gamma)
    pred_weighted = pred_feature_map * w_boundary
    target_weighted = target_mask * w_boundary
    if target_weighted.shape[1] != pred_weighted.shape[1]:
        target_weighted = target_weighted.repeat(1, pred_weighted.shape[1], 1, 1, 1)

    tok_gen = _tokenize_volume(pred_weighted, patch_size=patch_size, stride=token_stride)
    tok_pos = _tokenize_volume(target_weighted, patch_size=patch_size, stride=token_stride)
    tok_mask = _tokenize_volume(target_mask, patch_size=patch_size, stride=token_stride).squeeze(-1)

    # Normalize token vectors for stability.
    tok_gen = F.normalize(tok_gen, dim=-1, eps=1e-6)
    tok_pos = F.normalize(tok_pos, dim=-1, eps=1e-6)

    b, n_tokens, c = tok_gen.shape
    feat_gen = tok_gen.reshape(b * n_tokens, c)
    x_old = feat_gen.detach()
    feat_pos = tok_pos.reshape(b * n_tokens, c).detach()

    if x_old.shape[0] > 1:
        feat_neg = torch.roll(x_old, shifts=1, dims=0)
    else:
        feat_neg = x_old

    if feature_bank is not None:
        bank_neg = feature_bank.sample(n_items=int(x_old.shape[0]))
        if bank_neg is not None and bank_neg.shape[1] == x_old.shape[1]:
            feat_neg = torch.cat([feat_neg, bank_neg.to(x_old.device)], dim=0)

    v = compute_drifting_field(
        x_gen=x_old,
        y_pos=feat_pos,
        y_neg=feat_neg,
        temperature=temperature,
        pos_weight=pos_weight,
        neg_weight=neg_weight,
    )

    goal = x_old + v.detach()
    per_token = (feat_gen - goal).pow(2).mean(dim=1)

    # Focus drift slightly more on tumor-containing local regions.
    token_weights = 1.0 + tok_mask.reshape(-1)

    dt_scale = _delta_t_scale(
        delta_t_map=delta_t_map,
        batch_size=bsz,
        delta_t_beta=delta_t_beta,
        delta_t_center=delta_t_center,
        device=device,
    )
    dt_scale_token = dt_scale.repeat_interleave(n_tokens)
    loss = (per_token * token_weights * dt_scale_token).mean()

    if feature_bank is not None:
        feature_bank.add(x_old)

    stats = {
        "local_token_field_l2": float(v.pow(2).mean().item()),
        "local_token_count": float(n_tokens),
    }
    return loss, stats


def _build_gaussian_component_windows(
    shape: tuple[int, int, int],
    centers: list[tuple[float, float, float]],
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    if len(centers) == 0:
        return torch.zeros((0, *shape), device=device, dtype=torch.float32)

    h, w, d = shape
    yy, xx, zz = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        torch.arange(d, device=device, dtype=torch.float32),
        indexing="ij",
    )
    sigma2 = max(float(sigma), 1e-3) ** 2

    windows = []
    for cy, cx, cz in centers:
        dist2 = (yy - float(cy)) ** 2 + (xx - float(cx)) ** 2 + (zz - float(cz)) ** 2
        g = torch.exp(-dist2 / (2.0 * sigma2))
        g = g / (g.sum() + 1e-8)
        windows.append(g)
    return torch.stack(windows, dim=0)


def _extract_component_centers(
    target_3d: np.ndarray,
    max_components: int,
    min_voxels: int,
) -> list[tuple[float, float, float]]:
    try:
        from scipy import ndimage as ndi  # type: ignore
    except Exception:
        ndi = None

    mask = (target_3d > 0.5).astype(np.uint8)
    if mask.sum() == 0:
        return []

    centers: list[tuple[float, float, float]] = []
    if ndi is None:
        coords = np.argwhere(mask > 0)
        c = coords.mean(axis=0)
        return [(float(c[0]), float(c[1]), float(c[2]))]

    labels, num = ndi.label(mask)
    if num <= 0:
        coords = np.argwhere(mask > 0)
        c = coords.mean(axis=0)
        return [(float(c[0]), float(c[1]), float(c[2]))]

    comp_sizes = []
    for i in range(1, num + 1):
        size = int((labels == i).sum())
        if size >= int(min_voxels):
            comp_sizes.append((i, size))
    if len(comp_sizes) == 0:
        coords = np.argwhere(mask > 0)
        c = coords.mean(axis=0)
        return [(float(c[0]), float(c[1]), float(c[2]))]

    comp_sizes.sort(key=lambda x: x[1], reverse=True)
    keep = comp_sizes[: max(1, int(max_components))]
    for i, _ in keep:
        com = ndi.center_of_mass(mask, labels, i)
        centers.append((float(com[0]), float(com[1]), float(com[2])))
    return centers


def component_aware_loss_from_logits(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    max_components: int = 3,
    component_sigma: float = 3.0,
    min_component_voxels: int = 10,
    off_target_weight: float = 0.25,
    delta_t_map: torch.Tensor | None = None,
    delta_t_beta: float = 0.0,
    delta_t_center: float = 0.6,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Component-aware loss for multifocal growth:
    - build Gaussian windows at target connected-component centers
    - match predicted mass in each component window
    - penalize off-component excess mass
    """
    probs = torch.sigmoid(logits)
    target = target_mask.float()
    bsz = int(target.shape[0])
    device = target.device

    per_sample_losses = []
    n_components_total = 0.0
    for b in range(bsz):
        pred_b = probs[b, 0]   # [H, W, D]
        targ_b = target[b, 0]  # [H, W, D]
        targ_np = targ_b.detach().cpu().numpy()

        centers = _extract_component_centers(
            target_3d=targ_np,
            max_components=max_components,
            min_voxels=min_component_voxels,
        )
        windows = _build_gaussian_component_windows(
            shape=tuple(targ_b.shape),
            centers=centers,
            sigma=component_sigma,
            device=device,
        )

        n_components_total += float(max(1, windows.shape[0]))
        if windows.shape[0] == 0:
            # Fallback to global mean absolute difference.
            per_sample_losses.append(torch.mean(torch.abs(pred_b - targ_b)))
            continue

        # Mass matching per target component.
        pred_mass = (windows * pred_b.unsqueeze(0)).sum(dim=(1, 2, 3))
        targ_mass = (windows * targ_b.unsqueeze(0)).sum(dim=(1, 2, 3))
        loss_mass = torch.mean(torch.abs(pred_mass - targ_mass))

        # Off-component control.
        union = windows.max(dim=0).values.clamp(0.0, 1.0)
        pred_off = (pred_b * (1.0 - union)).mean()
        targ_off = (targ_b * (1.0 - union)).mean()
        loss_off = torch.abs(pred_off - targ_off)

        # Coverage consistency in component windows.
        pred_cov = (pred_b * union).sum() / (pred_b.sum() + 1e-8)
        targ_cov = (targ_b * union).sum() / (targ_b.sum() + 1e-8)
        loss_cov = torch.abs(pred_cov - targ_cov)

        per_sample_losses.append(loss_mass + float(off_target_weight) * loss_off + 0.5 * loss_cov)

    losses = torch.stack(per_sample_losses, dim=0)

    dt_scale = _delta_t_scale(
        delta_t_map=delta_t_map,
        batch_size=bsz,
        delta_t_beta=delta_t_beta,
        delta_t_center=delta_t_center,
        device=device,
    )
    loss = (losses * dt_scale).mean()

    stats = {
        "component_count_mean": float(n_components_total / max(bsz, 1)),
        "component_dt_scale_mean": float(dt_scale.mean().item()),
    }
    return loss, stats


def _compute_target_sdf_numpy(mask_np: np.ndarray) -> np.ndarray:
    """
    Signed distance field from binary mask.
    Positive inside tumor, negative outside.
    """
    try:
        from scipy import ndimage as ndi  # type: ignore
    except Exception:
        ndi = None

    mask = (mask_np > 0.5).astype(np.uint8)
    if ndi is None:
        # Fallback: if SciPy is unavailable, return centered occupancy proxy.
        return (2.0 * mask.astype(np.float32) - 1.0).astype(np.float32)

    dist_in = ndi.distance_transform_edt(mask)
    dist_out = ndi.distance_transform_edt(1 - mask)
    sdf = dist_in - dist_out
    return sdf.astype(np.float32)


def sdf_boundary_drift_loss_from_logits(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    sdf_band_width: float = 4.0,
    sdf_clip_value: float = 10.0,
    sdf_logit_scale: float = 3.0,
    delta_t_map: torch.Tensor | None = None,
    delta_t_beta: float = 0.0,
    delta_t_center: float = 0.6,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    SDF/boundary drift loss:
    - Builds target SDF from ground truth mask.
    - Compares target signed field to logit-based signed prediction.
    - Weights errors near the boundary band more strongly.
    """
    target = target_mask.float()
    bsz = int(target.shape[0])
    device = target.device

    # Differentiable signed proxy from logits.
    pred_signed = torch.tanh(logits / max(float(sdf_logit_scale), 1e-6))

    sdf_targets = []
    for b in range(bsz):
        sdf_np = _compute_target_sdf_numpy(target[b, 0].detach().cpu().numpy())
        sdf_targets.append(torch.from_numpy(sdf_np))
    target_sdf = torch.stack(sdf_targets, dim=0).unsqueeze(1).to(device=device, dtype=pred_signed.dtype)

    clip_val = max(float(sdf_clip_value), 1e-6)
    target_signed = torch.clamp(target_sdf, min=-clip_val, max=clip_val) / clip_val

    # Boundary-focused weighting: larger weight near sdf=0.
    band = max(float(sdf_band_width), 1e-6)
    boundary_weight = torch.exp(-torch.abs(target_sdf) / band)
    boundary_weight = 0.5 + boundary_weight

    per_sample = (boundary_weight * (pred_signed - target_signed).pow(2)).mean(dim=(1, 2, 3, 4))

    dt_scale = _delta_t_scale(
        delta_t_map=delta_t_map,
        batch_size=bsz,
        delta_t_beta=delta_t_beta,
        delta_t_center=delta_t_center,
        device=device,
    )
    loss = (per_sample * dt_scale).mean()

    stats = {
        "sdf_boundary_weight_mean": float(boundary_weight.mean().item()),
        "sdf_dt_scale_mean": float(dt_scale.mean().item()),
    }
    return loss, stats
