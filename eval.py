"""MVP-0 evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import torch
import torch.nn.functional as F
import yaml

from dataset import make_dataloaders
from model import build_model


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
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def evaluate_split(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0
    total_infer_time = 0.0
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            loss_bce = F.binary_cross_entropy_with_logits(logits, y_batch)
            dice = dice_from_logits(logits, y_batch)

            batch_size = int(x_batch.shape[0])
            total_loss += float(loss_bce.item())
            total_dice += float(dice.item())
            total_samples += batch_size
            total_infer_time += dt
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_dice = total_dice / max(n_batches, 1)
    sec_per_sample = total_infer_time / max(total_samples, 1)

    return {
        "bce_loss": avg_loss,
        "dice": avg_dice,
        "num_samples": total_samples,
        "total_infer_sec": total_infer_time,
        "sec_per_sample": sec_per_sample,
        "samples_per_sec": (1.0 / sec_per_sample) if sec_per_sample > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MVP-0 checkpoint.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/model_best.pt")
    parser.add_argument("--split", type=str, choices=["val", "train"], default="val")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading checkpoint: {args.ckpt}")

    train_loader, val_loader = make_dataloaders(cfg)
    loader = val_loader if args.split == "val" else train_loader

    use_image_channel = bool(cfg.get("use_image_channel", False))
    input_channels = int(cfg.get("input_channels", 3 if use_image_channel else 2))
    model = build_model(cfg, in_channels=input_channels).to(device)
    model_type = str(cfg.get("model_type", "unet_baseline"))
    print(f"[INFO] Model: {model_type}")
    print(f"[INFO] Inputs: channels={input_channels} | use_image_channel={use_image_channel}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_split(model, loader, device)
    metrics["split"] = args.split
    metrics["checkpoint"] = args.ckpt

    print(
        f"[RESULT] split={args.split} "
        f"dice={metrics['dice']:.4f} "
        f"bce={metrics['bce_loss']:.4f} "
        f"sec_per_sample={metrics['sec_per_sample']:.4f}"
    )

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
