"""MVP-0 training entrypoint."""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dataset import make_dataloaders
from drift_loss import (
    DriftFeatureBank,
    component_aware_loss_from_logits,
    drifting_loss_from_logits,
    local_token_drift_loss,
    sdf_boundary_drift_loss_from_logits,
)
from model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: Dict) -> torch.device:
    want = str(cfg.get("device", "auto")).lower()
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    target = target.float()

    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    denom = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()

    inter = (probs * target).sum(dim=(1, 2, 3, 4))
    denom = probs.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def run_val(model: torch.nn.Module, val_loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss, total_dice, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss_bce = F.binary_cross_entropy_with_logits(logits, yb)
            loss_dice = soft_dice_loss(logits, yb)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            dice = dice_from_logits(logits, yb)

            total_loss += float(loss.item())
            total_dice += float(dice.item())
            n_batches += 1

    if n_batches == 0:
        return {"val_loss": 0.0, "val_dice": 0.0}

    return {"val_loss": total_loss / n_batches, "val_dice": total_dice / n_batches}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MVP model.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg)
    print(f"[INFO] Device: {device}")

    train_loader, val_loader = make_dataloaders(cfg)

    model = build_model(cfg, in_channels=2).to(device)
    model_type = str(cfg.get("model_type", "unet_baseline"))
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Model: {model_type} | params={n_params_m:.2f}M")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    use_drift_loss = bool(cfg.get("use_drift_loss", False))
    lambda_drift = float(cfg.get("lambda_drift", 0.1))
    drift_lambda_warmup_epochs = int(cfg.get("drift_lambda_warmup_epochs", 0))
    drift_temperature = float(cfg.get("drift_temperature", 0.1))
    drift_feature_pool = int(cfg.get("drift_feature_pool", 4))
    drift_pool_scales = cfg.get("drift_pool_scales", [drift_feature_pool])
    drift_pos_weight = float(cfg.get("drift_pos_weight", 1.0))
    drift_neg_weight = float(cfg.get("drift_neg_weight", 1.0))
    drift_boundary_gamma = float(cfg.get("drift_boundary_gamma", 0.0))
    drift_delta_t_beta = float(cfg.get("drift_delta_t_beta", 0.0))
    drift_delta_t_center = float(cfg.get("drift_delta_t_center", 0.6))
    drift_use_memory_bank = bool(cfg.get("drift_use_memory_bank", False))
    drift_neg_bank_size = int(cfg.get("drift_neg_bank_size", 0))
    drift_feature_source = str(cfg.get("drift_feature_source", "probs")).lower()
    if drift_feature_source not in {"probs", "latent"}:
        raise ValueError("drift_feature_source must be one of {'probs', 'latent'}")

    use_local_token_drift = bool(cfg.get("use_local_token_drift", False))
    lambda_local_drift = float(cfg.get("lambda_local_drift", 0.0))
    local_lambda_warmup_epochs = int(cfg.get("local_lambda_warmup_epochs", drift_lambda_warmup_epochs))
    token_patch_size = int(cfg.get("token_patch_size", 4))
    token_stride = cfg.get("token_stride", None)
    token_stride = int(token_stride) if token_stride is not None else None
    token_temperature = float(cfg.get("token_temperature", drift_temperature))
    token_pos_weight = float(cfg.get("token_pos_weight", drift_pos_weight))
    token_neg_weight = float(cfg.get("token_neg_weight", drift_neg_weight))
    token_boundary_gamma = float(cfg.get("token_boundary_gamma", drift_boundary_gamma))
    token_delta_t_beta = float(cfg.get("token_delta_t_beta", drift_delta_t_beta))
    token_delta_t_center = float(cfg.get("token_delta_t_center", drift_delta_t_center))
    token_use_memory_bank = bool(cfg.get("token_use_memory_bank", drift_use_memory_bank))
    token_neg_bank_size = int(cfg.get("token_neg_bank_size", drift_neg_bank_size))
    token_feature_source = str(cfg.get("token_feature_source", "latent")).lower()
    token_feature_key = str(cfg.get("token_feature_key", "dec1"))
    if token_feature_source not in {"probs", "latent"}:
        raise ValueError("token_feature_source must be one of {'probs', 'latent'}")

    use_component_drift = bool(cfg.get("use_component_drift", False))
    lambda_component_drift = float(cfg.get("lambda_component_drift", 0.0))
    component_lambda_warmup_epochs = int(
        cfg.get("component_lambda_warmup_epochs", drift_lambda_warmup_epochs)
    )
    component_max_components = int(cfg.get("component_max_components", 3))
    component_sigma = float(cfg.get("component_sigma", 3.0))
    component_min_voxels = int(cfg.get("component_min_voxels", 10))
    component_off_target_weight = float(cfg.get("component_off_target_weight", 0.25))
    component_delta_t_beta = float(cfg.get("component_delta_t_beta", drift_delta_t_beta))
    component_delta_t_center = float(cfg.get("component_delta_t_center", drift_delta_t_center))

    use_sdf_boundary_drift = bool(cfg.get("use_sdf_boundary_drift", False))
    lambda_sdf_drift = float(cfg.get("lambda_sdf_drift", 0.0))
    sdf_lambda_warmup_epochs = int(cfg.get("sdf_lambda_warmup_epochs", drift_lambda_warmup_epochs))
    sdf_band_width = float(cfg.get("sdf_band_width", 4.0))
    sdf_clip_value = float(cfg.get("sdf_clip_value", 10.0))
    sdf_logit_scale = float(cfg.get("sdf_logit_scale", 3.0))
    sdf_delta_t_beta = float(cfg.get("sdf_delta_t_beta", drift_delta_t_beta))
    sdf_delta_t_center = float(cfg.get("sdf_delta_t_center", drift_delta_t_center))

    global_feature_bank = DriftFeatureBank(max_items=drift_neg_bank_size) if drift_use_memory_bank else None
    local_feature_bank = DriftFeatureBank(max_items=token_neg_bank_size) if token_use_memory_bank else None

    print(
        "[INFO] Drifting loss: "
        f"{'ON' if use_drift_loss else 'OFF'} | "
        f"lambda={lambda_drift:.4f} | "
        f"temperature={drift_temperature:.4f} | "
        f"pool={drift_feature_pool} | "
        f"scales={drift_pool_scales} | "
        f"src={drift_feature_source}"
    )
    print(
        "[INFO] Local token drift: "
        f"{'ON' if use_local_token_drift else 'OFF'} | "
        f"lambda_local={lambda_local_drift:.4f} | "
        f"patch={token_patch_size} | "
        f"stride={token_stride if token_stride is not None else token_patch_size} | "
        f"src={token_feature_source}:{token_feature_key}"
    )
    print(
        "[INFO] Component drift: "
        f"{'ON' if use_component_drift else 'OFF'} | "
        f"lambda_comp={lambda_component_drift:.4f} | "
        f"max_comp={component_max_components} | "
        f"sigma={component_sigma:.2f}"
    )
    print(
        "[INFO] SDF boundary drift: "
        f"{'ON' if use_sdf_boundary_drift else 'OFF'} | "
        f"lambda_sdf={lambda_sdf_drift:.4f} | "
        f"band={sdf_band_width:.2f} | "
        f"clip={sdf_clip_value:.2f}"
    )

    best_val_dice = -1.0
    epochs = int(cfg["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_dice, n_batches = 0.0, 0.0, 0
        running_seg_loss, running_drift_loss, running_local_drift_loss, running_comp_drift_loss, running_sdf_drift_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        running_drift_field_l2, running_dt_scale, running_local_field_l2 = 0.0, 0.0, 0.0
        running_comp_count, running_sdf_weight = 0.0, 0.0

        if drift_lambda_warmup_epochs > 0:
            warm = min(1.0, float(epoch) / float(drift_lambda_warmup_epochs))
            lambda_drift_epoch = lambda_drift * warm
        else:
            lambda_drift_epoch = lambda_drift
        if local_lambda_warmup_epochs > 0:
            warm_local = min(1.0, float(epoch) / float(local_lambda_warmup_epochs))
            lambda_local_epoch = lambda_local_drift * warm_local
        else:
            lambda_local_epoch = lambda_local_drift
        if component_lambda_warmup_epochs > 0:
            warm_comp = min(1.0, float(epoch) / float(component_lambda_warmup_epochs))
            lambda_comp_epoch = lambda_component_drift * warm_comp
        else:
            lambda_comp_epoch = lambda_component_drift
        if sdf_lambda_warmup_epochs > 0:
            warm_sdf = min(1.0, float(epoch) / float(sdf_lambda_warmup_epochs))
            lambda_sdf_epoch = lambda_sdf_drift * warm_sdf
        else:
            lambda_sdf_epoch = lambda_sdf_drift

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            need_latent_features = (
                (use_drift_loss and drift_feature_source == "latent")
                or (use_local_token_drift and token_feature_source == "latent")
            )
            if need_latent_features:
                logits, feat_dict = model(xb, return_features=True)
                drift_pred_feature = feat_dict[token_feature_key] if drift_feature_source == "latent" else None
                local_pred_feature = feat_dict[token_feature_key] if token_feature_source == "latent" else None
            else:
                logits = model(xb)
                drift_pred_feature = None
                local_pred_feature = None

            loss_bce = F.binary_cross_entropy_with_logits(logits, yb)
            loss_dice = soft_dice_loss(logits, yb)
            seg_loss = 0.5 * loss_bce + 0.5 * loss_dice

            if use_drift_loss:
                drift_loss, drift_stats = drifting_loss_from_logits(
                    logits=logits,
                    target_mask=yb,
                    temperature=drift_temperature,
                    pool_size=drift_feature_pool,
                    pool_scales=drift_pool_scales,
                    pos_weight=drift_pos_weight,
                    neg_weight=drift_neg_weight,
                    boundary_gamma=drift_boundary_gamma,
                    delta_t_map=xb[:, 1:2],
                    delta_t_beta=drift_delta_t_beta,
                    delta_t_center=drift_delta_t_center,
                    feature_bank=global_feature_bank,
                    pred_feature_map=drift_pred_feature,
                )
            else:
                drift_loss = torch.zeros((), device=device)
                drift_stats = {"drift_field_l2": 0.0, "dt_scale_mean": 1.0}

            if use_local_token_drift:
                if token_feature_source == "probs":
                    local_pred = torch.sigmoid(logits)
                else:
                    local_pred = local_pred_feature
                local_drift_loss, local_stats = local_token_drift_loss(
                    pred_feature_map=local_pred,
                    target_mask=yb,
                    temperature=token_temperature,
                    patch_size=token_patch_size,
                    token_stride=token_stride,
                    pos_weight=token_pos_weight,
                    neg_weight=token_neg_weight,
                    boundary_gamma=token_boundary_gamma,
                    delta_t_map=xb[:, 1:2],
                    delta_t_beta=token_delta_t_beta,
                    delta_t_center=token_delta_t_center,
                    feature_bank=local_feature_bank,
                )
            else:
                local_drift_loss = torch.zeros((), device=device)
                local_stats = {"local_token_field_l2": 0.0}

            if use_component_drift:
                component_drift_loss, component_stats = component_aware_loss_from_logits(
                    logits=logits,
                    target_mask=yb,
                    max_components=component_max_components,
                    component_sigma=component_sigma,
                    min_component_voxels=component_min_voxels,
                    off_target_weight=component_off_target_weight,
                    delta_t_map=xb[:, 1:2],
                    delta_t_beta=component_delta_t_beta,
                    delta_t_center=component_delta_t_center,
                )
            else:
                component_drift_loss = torch.zeros((), device=device)
                component_stats = {"component_count_mean": 0.0}

            if use_sdf_boundary_drift:
                sdf_drift_loss, sdf_stats = sdf_boundary_drift_loss_from_logits(
                    logits=logits,
                    target_mask=yb,
                    sdf_band_width=sdf_band_width,
                    sdf_clip_value=sdf_clip_value,
                    sdf_logit_scale=sdf_logit_scale,
                    delta_t_map=xb[:, 1:2],
                    delta_t_beta=sdf_delta_t_beta,
                    delta_t_center=sdf_delta_t_center,
                )
            else:
                sdf_drift_loss = torch.zeros((), device=device)
                sdf_stats = {"sdf_boundary_weight_mean": 0.0}

            loss = (
                seg_loss
                + lambda_drift_epoch * drift_loss
                + lambda_local_epoch * local_drift_loss
                + lambda_comp_epoch * component_drift_loss
                + lambda_sdf_epoch * sdf_drift_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dice = dice_from_logits(logits, yb)

            running_loss += float(loss.item())
            running_seg_loss += float(seg_loss.item())
            running_drift_loss += float(drift_loss.item())
            running_local_drift_loss += float(local_drift_loss.item())
            running_comp_drift_loss += float(component_drift_loss.item())
            running_sdf_drift_loss += float(sdf_drift_loss.item())
            running_drift_field_l2 += float(drift_stats.get("drift_field_l2", 0.0))
            running_dt_scale += float(drift_stats.get("dt_scale_mean", 1.0))
            running_local_field_l2 += float(local_stats.get("local_token_field_l2", 0.0))
            running_comp_count += float(component_stats.get("component_count_mean", 0.0))
            running_sdf_weight += float(sdf_stats.get("sdf_boundary_weight_mean", 0.0))
            running_dice += float(dice.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_seg_loss = running_seg_loss / max(n_batches, 1)
        train_drift_loss = running_drift_loss / max(n_batches, 1)
        train_local_drift_loss = running_local_drift_loss / max(n_batches, 1)
        train_comp_drift_loss = running_comp_drift_loss / max(n_batches, 1)
        train_sdf_drift_loss = running_sdf_drift_loss / max(n_batches, 1)
        train_drift_field_l2 = running_drift_field_l2 / max(n_batches, 1)
        train_local_field_l2 = running_local_field_l2 / max(n_batches, 1)
        train_comp_count = running_comp_count / max(n_batches, 1)
        train_sdf_weight = running_sdf_weight / max(n_batches, 1)
        train_dt_scale = running_dt_scale / max(n_batches, 1)
        train_dice = running_dice / max(n_batches, 1)

        val_metrics = run_val(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        val_dice = val_metrics["val_dice"]

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"train_seg={train_seg_loss:.4f} "
            f"train_drift={train_drift_loss:.4f} "
            f"train_local={train_local_drift_loss:.4f} "
            f"train_comp={train_comp_drift_loss:.4f} "
            f"train_sdf={train_sdf_drift_loss:.4f} "
            f"drift_l2={train_drift_field_l2:.4f} "
            f"local_l2={train_local_field_l2:.4f} "
            f"comp_n={train_comp_count:.2f} "
            f"sdf_w={train_sdf_weight:.3f} "
            f"dt_scale={train_dt_scale:.3f} "
            f"lambda_t={lambda_drift_epoch:.4f} "
            f"lambda_local={lambda_local_epoch:.4f} "
            f"lambda_comp={lambda_comp_epoch:.4f} "
            f"lambda_sdf={lambda_sdf_epoch:.4f} "
            f"train_dice={train_dice:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        # Save latest
        latest_path = os.path.join(cfg["output_dir"], "model_latest.pt")
        torch.save({"model_state_dict": model.state_dict(), "cfg": cfg, "epoch": epoch}, latest_path)

        # Save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = os.path.join(cfg["output_dir"], "model_best.pt")
            torch.save({"model_state_dict": model.state_dict(), "cfg": cfg, "epoch": epoch}, best_path)

    print(f"[DONE] Best val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
