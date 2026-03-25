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
