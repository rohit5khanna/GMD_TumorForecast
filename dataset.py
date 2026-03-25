"""Synthetic dataset for one-shot conditional tumor growth forecasting"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class SyntheticTumorDataset(Dataset):
    """
    Returns:
      x: [2, H, W, D] float32
         channel 0 = baseline mask
         channel 1 = constant delta_t conditioning map
      y: [1, H, W, D] float32 (future mask)
    """

    def __init__(
        self, 
        num_samples: int,
        volume_shape: Tuple[int, int, int] = (64, 64, 64),
        seed: int = 42,
    ) -> None:
        self.num_samples = int(num_samples)
        self.shape = tuple(volume_shape)
        self.seed = int(seed)

        # Precompute coordinate grid for fast ellipsoid drawing
        self.xx, self.yy, self.zz = np.indices(self.shape)

    def __len__(self) -> int:
        return self.num_samples

    def _draw_ellipsoid(
        self,
        mask: np.ndarray,
        center: Tuple[int, int, int],
        radii: Tuple[int, int, int],
    ) -> None:
        cx, cy, cz = center
        rx, ry, rz = max(1, radii[0]), max(1, radii[1]), max(1, radii[2])

        eq = (
            ((self.xx - cx) / rx) ** 2
            + ((self.yy - cy) / ry) ** 2
            + ((self.zz - cz) / rz) ** 2
        ) <= 1.0
        mask[eq] = 1.0

    def _make_baseline(self, rng: np.random.Generator) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=np.float32)
        n_components = int(rng.integers(1, 3))  # 1-2 blobs

        for _ in range(n_components):
            center = (
                int(rng.integers(12, self.shape[0] - 12)),
                int(rng.integers(12, self.shape[1] - 12)),
                int(rng.integers(12, self.shape[2] - 12)),
            )
            radii = (
                int(rng.integers(3, 8)),
                int(rng.integers(3, 8)),
                int(rng.integers(3, 8)),
            )
            self._draw_ellipsoid(mask, center, radii)

        return mask

    def _single_growth_step(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = mask.copy()
        coords = np.argwhere(mask > 0.5)
        if len(coords) == 0:
            return out

        n_seeds = int(rng.integers(1, 4))
        pick_idx = rng.integers(0, len(coords), size=n_seeds)

        for idx in pick_idx:
            cx, cy, cz = coords[idx]
            radii = (
                int(rng.integers(2, 6)),
                int(rng.integers(2, 6)),
                int(rng.integers(2, 6)),
            )
            self._draw_ellipsoid(out, (int(cx), int(cy), int(cz)), radii)

        # Mild random shift to imitate anisotropic-looking drift.
        if rng.random() < 0.35:
            shift = (
                int(rng.integers(-2, 3)),
                int(rng.integers(-2, 3)),
                int(rng.integers(-2, 3)),
            )
            shifted = np.roll(out, shift=shift, axis=(0, 1, 2))
            out = np.maximum(out, shifted)

        # Optional small satellite focus.
        if rng.random() < 0.2:
            anchor = coords[int(rng.integers(0, len(coords)))]
            offset = rng.integers(-10, 11, size=3)
            c = np.clip(anchor + offset, 4, np.array(self.shape) - 5)
            self._draw_ellipsoid(
                out,
                (int(c[0]), int(c[1]), int(c[2])),
                (int(rng.integers(1, 3)), int(rng.integers(1, 3)), int(rng.integers(1, 3))),
            )

        return out

    def _make_pair(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
        baseline = self._make_baseline(rng)

        # Conditional scalar: larger delta_t means more growth steps.
        delta_t = float(rng.uniform(0.2, 1.0))
        n_steps = int(2 + np.round(delta_t * 4))  # 2..6 steps

        future = baseline.copy()
        for _ in range(n_steps):
            future = self._single_growth_step(future, rng)

        future = (future > 0.5).astype(np.float32)
        baseline = (baseline > 0.5).astype(np.float32)

        return baseline, future, delta_t

    def __getitem__(self, idx: int):
        # Deterministic sample generation per index.
        rng = np.random.default_rng(self.seed + idx * 9973)

        baseline, future, delta_t = self._make_pair(rng)

        cond_map = np.full(self.shape, delta_t, dtype=np.float32)
        x = np.stack([baseline, cond_map], axis=0)  # [2, H, W, D]
        y = future[None, ...]  # [1, H, W, D]

        return torch.from_numpy(x), torch.from_numpy(y)


def make_dataloaders(cfg: Dict):
    train_ds = SyntheticTumorDataset(
        num_samples=cfg["train_samples"],
        volume_shape=tuple(cfg["volume_shape"]),
        seed=int(cfg.get("seed", 42)),
    )
    val_ds = SyntheticTumorDataset(
        num_samples=cfg["val_samples"],
        volume_shape=tuple(cfg["volume_shape"]),
        seed=int(cfg.get("seed", 42)) + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    cfg = {
        "seed": 42,
        "train_samples": 4,
        "val_samples": 2,
        "batch_size": 2,
        "volume_shape": [64, 64, 64],
    }
    train_loader, _ = make_dataloaders(cfg)
    xb, yb = next(iter(train_loader))
    print("x shape:", xb.shape)  # [B, 2, H, W, D]
    print("y shape:", yb.shape)  # [B, 1, H, W, D]


# Backward-compatible alias.
SyntheticTumor = SyntheticTumorDataset
