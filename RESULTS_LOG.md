# MVP-0 Results Log

## Iteration 0: One-Shot Baseline (No Drifting Loss, No PDE)
- Date: March 25, 2026
- Setup: Synthetic conditional forecasting, one-shot CNN baseline, CPU (Colab)

### Seed Sweep
| seed | dice | sec_per_sample |
|---:|---:|---:|
| 42 | 0.6640415092 | 0.2502081825 |
| 43 | 0.6840255260 | 0.2627061185 |
| 44 | 0.5278459216 | 0.2521438345 |

### Aggregate
- `mean_dice`: `0.6253043190`
- `std_dice`: `0.0693947387`
- `mean_sec_per_sample`: `0.2550193785`

### Notes
- Baseline is learning meaningful signal at one-shot inference speed.
- Seed variance exists (especially seed 44), which motivates adding drifting regularization in the next iteration.

---

## Iteration 1: One-Shot + Simplified Drifting Loss (No PDE)
- Date: March 25, 2026
- Setup: Synthetic conditional forecasting, drifting loss enabled (`lambda_drift=0.1`, `temperature=0.1`, `pool=4`), CPU (Colab)
- Run type: single-seed pilot (seed 42 config)

### Result
| split | dice | bce_loss | sec_per_sample | samples_per_sec |
|---|---:|---:|---:|---:|
| val | 0.5278459216 | 0.5862929523 | 0.2464569298 | 4.0575040867 |

### Comparison vs Iteration 0
- Iteration 0 single-seed best (seed 42): Dice `0.6640415092`
- Iteration 1 single-seed: Dice `0.5278459216`
- Absolute delta: `-0.1361955876`
- Inference speed: effectively unchanged (~0.25 sec/sample)

### Notes
- First drifting-loss setting underperformed the baseline on Dice.
- This does not invalidate the approach; it indicates the current drift regularization weight/formulation is too aggressive or mismatched for tiny-data training.
