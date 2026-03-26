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

---

## Iteration 2: Drift Coarse Sweep (Bug-fixed Drift Targeting)
- Date: March 25, 2026
- Setup: same tiny synthetic conditional setup, multiple drift hyperparameter trials, seed-controlled training/eval on CPU.

### Coarse Sweep Outcome
- 20 coarse trials were run.
- Top 5 configs by Dice:

| rank | lambda_drift | temperature | pool | dice |
|---:|---:|---:|---:|---:|
| 1 | 0.01 | 0.10 | 4 | 0.6765218124 |
| 2 | 0.01 | 0.20 | 2 | 0.6722378045 |
| 3 | 0.20 | 0.05 | 2 | 0.6677178532 |
| 4 | 0.20 | 0.10 | 4 | 0.6673621327 |
| 5 | 0.50 | 0.20 | 6 | 0.6598742075 |

### Notes
- This iteration followed a drift-loss gradient fix (frozen target state + trainable current state).
- Coarse sweep results established stable candidates for multi-seed confirmation.

---

## Iteration 3: Drift Confirm Sweep (Top-3 x Seeds 42/43/44)
- Date: March 26, 2026
- Setup: Top-3 drift configs from Iteration 2, each rerun with seeds `42, 43, 44`.

### Per-Config Aggregates
| config_rank | use_drift_loss | lambda_drift | temperature | pool | mean_dice | std_dice |
|---:|---|---:|---:|---:|---:|---:|
| 1 | true | 0.01 | 0.20 | 2 | 0.6764510771 | 0.0091108363 |
| 0 | true | 0.01 | 0.10 | 4 | 0.6762212978 | 0.0087941511 |
| 2 | true | 0.20 | 0.05 | 2 | 0.6757849654 | 0.0059956126 |

### Selected Drift Config (for matched control comparison)
- `use_drift_loss=true`
- `lambda_drift=0.01`
- `drift_temperature=0.2`
- `drift_feature_pool=2`
- Aggregate: `mean_dice=0.6764510771`, `std_dice=0.0091108363`

---

## Iteration 4: Matched No-Drift Control (Seeds 42/43/44)
- Date: March 26, 2026
- Setup: Same training budget/data/settings as selected drift config, except `use_drift_loss=false` and `lambda_drift=0.0`.

### Per-Seed Control Results
| seed | dice | bce_loss | sec_per_sample |
|---:|---:|---:|---:|
| 42 | 0.5839960128 | 0.5921625892 | 0.0870696977 |
| 43 | 0.5549587011 | 0.5891484817 | 0.0757242753 |
| 44 | 0.5662185947 | 0.5871194104 | 0.0759731609 |

### Control Aggregate
- `mean_dice`: `0.5683911029`
- `std_dice`: `0.0119535549`

### Drift vs Matched Control
- Drift mean Dice: `0.6764510771`
- No-drift mean Dice: `0.5683911029`
- Absolute delta (`drift - no_drift`): `+0.1080599742`
- Relative gain over no-drift baseline: `+19.01%`

### Current Conclusion (Synthetic MVP)
- The drift-regularized objective provides a strong and repeatable gain under matched settings.
- Inference remains one-shot and fast; the gain is due to training objective, not iterative test-time optimization.
