"""MVP-0 training entrypoint."""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dataset import make_dataloaders
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
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg)
    print(f"[INFO] Device: {device}")

    train_loader, val_loader = make_dataloaders(cfg)

    model = OneShotPredictor(in_channels=2, base_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))

    best_val_dice = -1.0
    epochs = int(cfg["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_dice, n_batches = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            loss_bce = F.binary_cross_entropy_with_logits(logits, yb)
            loss_dice = soft_dice_loss(logits, yb)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dice = dice_from_logits(logits, yb)

            running_loss += float(loss.item())
            running_dice += float(dice.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_dice = running_dice / max(n_batches, 1)

        val_metrics = run_val(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        val_dice = val_metrics["val_dice"]

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
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
