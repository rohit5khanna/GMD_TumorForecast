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
from drift_loss import DriftFeatureBank, drifting_loss_from_logits
from model import OneShotPredictor


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

    model = OneShotPredictor(in_channels=2, base_channels=16).to(device)
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

    feature_bank = DriftFeatureBank(max_items=drift_neg_bank_size) if drift_use_memory_bank else None

    print(
        "[INFO] Drifting loss: "
        f"{'ON' if use_drift_loss else 'OFF'} | "
        f"lambda={lambda_drift:.4f} | "
        f"temperature={drift_temperature:.4f} | "
        f"pool={drift_feature_pool} | "
        f"scales={drift_pool_scales} | "
        f"src={drift_feature_source}"
    )

    best_val_dice = -1.0
    epochs = int(cfg["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_dice, n_batches = 0.0, 0.0, 0
        running_seg_loss, running_drift_loss = 0.0, 0.0
        running_drift_field_l2, running_dt_scale = 0.0, 0.0

        if drift_lambda_warmup_epochs > 0:
            warm = min(1.0, float(epoch) / float(drift_lambda_warmup_epochs))
            lambda_drift_epoch = lambda_drift * warm
        else:
            lambda_drift_epoch = lambda_drift

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            if use_drift_loss and drift_feature_source == "latent":
                logits, feat_dict = model(xb, return_features=True)
                drift_pred_feature = feat_dict["dec1"]
            else:
                logits = model(xb)
                drift_pred_feature = None

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
                    feature_bank=feature_bank,
                    pred_feature_map=drift_pred_feature,
                )
            else:
                drift_loss = torch.zeros((), device=device)
                drift_stats = {"drift_field_l2": 0.0, "dt_scale_mean": 1.0}

            loss = seg_loss + lambda_drift_epoch * drift_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dice = dice_from_logits(logits, yb)

            running_loss += float(loss.item())
            running_seg_loss += float(seg_loss.item())
            running_drift_loss += float(drift_loss.item())
            running_drift_field_l2 += float(drift_stats.get("drift_field_l2", 0.0))
            running_dt_scale += float(drift_stats.get("dt_scale_mean", 1.0))
            running_dice += float(dice.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_seg_loss = running_seg_loss / max(n_batches, 1)
        train_drift_loss = running_drift_loss / max(n_batches, 1)
        train_drift_field_l2 = running_drift_field_l2 / max(n_batches, 1)
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
            f"drift_l2={train_drift_field_l2:.4f} "
            f"dt_scale={train_dt_scale:.3f} "
            f"lambda_t={lambda_drift_epoch:.4f} "
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
