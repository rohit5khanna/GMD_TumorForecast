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
        params: Dict | None = None,
    ) -> None:
        self.num_samples = int(num_samples)
        self.shape = tuple(volume_shape)
        self.seed = int(seed)
        self.params = {
            "baseline_components_min": 1,
            "baseline_components_max": 2,
            "baseline_radius_min": 3,
            "baseline_radius_max": 7,
            "center_margin": 12,
            "growth_seed_min": 1,
            "growth_seed_max": 3,
            "growth_radius_min": 2,
            "growth_radius_max": 5,
            "shift_prob": 0.35,
            "shift_voxel_max": 2,
            "satellite_prob": 0.20,
            "satellite_offset_max": 10,
            "satellite_radius_min": 1,
            "satellite_radius_max": 2,
            "delta_t_min": 0.2,
            "delta_t_max": 1.0,
            "growth_steps_min": 2,
            "growth_steps_max": 6,
        }
        if params:
            self.params.update(params)

        # Precompute coordinate grid for fast ellipsoid drawing
        self.xx, self.yy, self.zz = np.indices(self.shape)

    def __len__(self) -> int:
        return self.num_samples

    def _sample_int_inclusive(
        self,
        rng: np.random.Generator,
        min_key: str,
        max_key: str,
    ) -> int:
        low = int(self.params[min_key])
        high = int(self.params[max_key])
        if high < low:
            high = low
        return int(rng.integers(low, high + 1))

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
        n_components = self._sample_int_inclusive(
            rng, "baseline_components_min", "baseline_components_max"
        )
        center_margin = int(self.params["center_margin"])
        max_margin = max(2, min(s // 3 for s in self.shape))
        center_margin = min(center_margin, max_margin)

        for _ in range(n_components):
            center = (
                int(rng.integers(center_margin, self.shape[0] - center_margin)),
                int(rng.integers(center_margin, self.shape[1] - center_margin)),
                int(rng.integers(center_margin, self.shape[2] - center_margin)),
            )
            radii = (
                self._sample_int_inclusive(rng, "baseline_radius_min", "baseline_radius_max"),
                self._sample_int_inclusive(rng, "baseline_radius_min", "baseline_radius_max"),
                self._sample_int_inclusive(rng, "baseline_radius_min", "baseline_radius_max"),
            )
            self._draw_ellipsoid(mask, center, radii)

        return mask

    def _single_growth_step(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = mask.copy()
        coords = np.argwhere(mask > 0.5)
        if len(coords) == 0:
            return out

        n_seeds = self._sample_int_inclusive(rng, "growth_seed_min", "growth_seed_max")
        pick_idx = rng.integers(0, len(coords), size=n_seeds)

        for idx in pick_idx:
            cx, cy, cz = coords[idx]
            radii = (
                self._sample_int_inclusive(rng, "growth_radius_min", "growth_radius_max"),
                self._sample_int_inclusive(rng, "growth_radius_min", "growth_radius_max"),
                self._sample_int_inclusive(rng, "growth_radius_min", "growth_radius_max"),
            )
            self._draw_ellipsoid(out, (int(cx), int(cy), int(cz)), radii)

        # Mild random shift to imitate anisotropic-looking drift.
        if rng.random() < float(self.params["shift_prob"]):
            shift_max = max(1, int(self.params["shift_voxel_max"]))
            shift = (
                int(rng.integers(-shift_max, shift_max + 1)),
                int(rng.integers(-shift_max, shift_max + 1)),
                int(rng.integers(-shift_max, shift_max + 1)),
            )
            shifted = np.roll(out, shift=shift, axis=(0, 1, 2))
            out = np.maximum(out, shifted)

        # Optional small satellite focus.
        if rng.random() < float(self.params["satellite_prob"]):
            anchor = coords[int(rng.integers(0, len(coords)))]
            off = max(1, int(self.params["satellite_offset_max"]))
            offset = rng.integers(-off, off + 1, size=3)
            c = np.clip(anchor + offset, 4, np.array(self.shape) - 5)
            self._draw_ellipsoid(
                out,
                (int(c[0]), int(c[1]), int(c[2])),
                (
                    self._sample_int_inclusive(rng, "satellite_radius_min", "satellite_radius_max"),
                    self._sample_int_inclusive(rng, "satellite_radius_min", "satellite_radius_max"),
                    self._sample_int_inclusive(rng, "satellite_radius_min", "satellite_radius_max"),
                ),
            )

        return out

    def _make_pair(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
        baseline = self._make_baseline(rng)

        # Conditional scalar: larger delta_t means more growth steps.
        dt_min = float(self.params["delta_t_min"])
        dt_max = float(self.params["delta_t_max"])
        if dt_max < dt_min:
            dt_max = dt_min
        delta_t = float(rng.uniform(dt_min, dt_max))

        gs_min = int(self.params["growth_steps_min"])
        gs_max = int(self.params["growth_steps_max"])
        if gs_max < gs_min:
            gs_max = gs_min
        if dt_max == dt_min:
            alpha = 0.0
        else:
            alpha = (delta_t - dt_min) / (dt_max - dt_min)
        n_steps = int(round(gs_min + alpha * (gs_max - gs_min)))
        n_steps = max(gs_min, min(gs_max, n_steps))

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
    dataset_params = dict(cfg.get("dataset_params", {}))

    train_ds = SyntheticTumorDataset(
        num_samples=cfg["train_samples"],
        volume_shape=tuple(cfg["volume_shape"]),
        seed=int(cfg.get("seed", 42)),
        params=dataset_params,
    )
    val_ds = SyntheticTumorDataset(
        num_samples=cfg["val_samples"],
        volume_shape=tuple(cfg["volume_shape"]),
        seed=int(cfg.get("seed", 42)) + 1,
        params=dataset_params,
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
