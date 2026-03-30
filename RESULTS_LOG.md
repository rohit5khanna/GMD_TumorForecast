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

---

## Iteration 5: MVP-1 Hard-Synthetic Baseline (Initial Settings)
- Date: March 26, 2026
- Setup: Harder synthetic generation regime, CPU/GPU Colab run, `epochs=10`.
- Drift config: `lambda_drift=0.01`, `temperature=0.2`, `pool=2`.

### Single-Seed Smoke (seed 42)
| variant | dice | bce_loss | sec_per_sample |
|---|---:|---:|---:|
| drift | 0.5645870220 | 0.4637282357 | 0.3281 |
| no-drift | 0.5658592191 | 0.4669078544 | 0.2673 |

### 3-Seed Matched Comparison (Initial Hard Config)
| variant | mean_dice | std_dice | n |
|---|---:|---:|---:|
| drift | 0.5380797624 | 0.0215629546 | 3 |
| no-drift | 0.5386029230 | 0.0220036685 | 3 |

- Delta (`drift - no_drift`): `-0.0005231606` (effectively neutral)

### Notes
- On hard synthetic, the original drift setting did not improve Dice.
- This motivated a small retune under matched conditions.

---

## Iteration 6: MVP-1 Hard-Synthetic Retune (Seed 42)
- Date: March 26, 2026
- Setup: GPU runtime, matched data regime, `batch_size=2`, `epochs=12`, grid over drift hyperparameters.

### Retune Trials
| trial | lambda_drift | temperature | dice | bce_loss | sec_per_sample |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.05 | 0.1 | 0.6258903444 | 0.4142772138 | 0.0136006657 |
| 1 | 0.05 | 0.2 | 0.6294468164 | 0.4084808558 | 0.0140792258 |
| 2 | 0.10 | 0.1 | 0.6289154887 | 0.4086705625 | 0.0146412360 |
| 3 | 0.10 | 0.2 | 0.6326019883 | 0.4034049958 | 0.0134116809 |
| 4 | 0.20 | 0.1 | 0.6378787696 | 0.3897754490 | 0.0194680911 |
| 5 | 0.20 | 0.2 | 0.6372174203 | 0.3921767890 | 0.0140503070 |

### Selected Config for Confirm
- `lambda_drift=0.2`
- `drift_temperature=0.1`
- `drift_feature_pool=2`
- `batch_size=2`
- `epochs=12`

---

## Iteration 7: MVP-1 Tuned Hard-Synthetic Confirm (3 Seeds, Matched Control)
- Date: March 26, 2026
- Setup: Tuned config from Iteration 6, matched no-drift control, seeds `42, 43, 44`.

### Per-Seed Results
| seed | drift_dice | nodrift_dice |
|---:|---:|---:|
| 42 | 0.6379722297 | 0.6248903215 |
| 43 | 0.6047135562 | 0.6061816037 |
| 44 | 0.6715537488 | 0.6690127134 |

### Aggregate Summary
| variant | mean_dice | std_dice | mean_bce | mean_sec_per_sample | n |
|---|---:|---:|---:|---:|---:|
| drift | 0.6380798449 | 0.0272875005 | 0.4243817170 | 0.0135437654 | 3 |
| no-drift | 0.6333615462 | 0.0263408216 | 0.4331462363 | 0.0147896976 | 3 |

- Dice delta (`drift - no_drift`): `+0.0047182987`
- Relative Dice gain: `+0.74%`

### Current MVP-1 Interpretation
- Tuned drift provides a small positive gain on the harder synthetic regime.
- Effect size is much smaller than MVP-0 easy-synthetic gains, indicating regime sensitivity.

---

## Iteration 8: Planned Incremental Ablation Roadmap (Mods 1-7)
- Date: March 26, 2026
- Objective: test the first 7 drift modifications incrementally (not all at once) for clean attribution.

### Stage Order (Locked)
| stage | modifications enabled |
|---|---|
| A | #1 stronger negatives (memory bank) + #4 lambda warmup |
| B | Stage A + #3 separate attraction/repulsion weights |
| C | Stage B + #2 multiscale drift features |
| D | Stage C + #5 boundary-aware drift emphasis |
| E | Stage D + #7 time-aware drift scaling |
| F | Stage E + #6 latent-feature drift source |

### Evaluation Policy
1. One-seed screen for each stage.
2. Run 3-seed confirm only if stage improves over current best.
3. Keep matched no-drift control unchanged for fair deltas.

### Status
- Completed: Stage A through Stage F runs are logged in Iterations 9-14 below.

---

## Iteration 9: Stage A Confirm (Mods #1 + #4)
- Date: March 26, 2026
- Setup: Stage A = stronger negatives (memory bank) + lambda warmup, seeds `42, 43, 44`.

### Aggregate Summary
| variant | mean_dice | std_dice | mean_bce | mean_sec_per_sample | n |
|---|---:|---:|---:|---:|---:|
| stageA_drift | 0.6346362849 | 0.0280411552 | 0.4274779131 | 0.0141928587 | 3 |
| stageA_nodrift | 0.6337653587 | 0.0265588378 | 0.4323370179 | 0.0153874444 | 3 |

- Dice delta (`drift - no_drift`): `+0.0008709262`

### Interpretation
- Stage A is effectively neutral (gain is very small vs seed variance).

---

## Iteration 10: Stage B Confirm (Stage A + Mod #3)
- Date: March 26, 2026
- Setup: Added separate attraction/repulsion weights with `pos_weight=1.2`, `neg_weight=0.8`, seeds `42, 43, 44`.

### Aggregate Summary
| variant | mean_dice | std_dice | mean_bce | mean_sec_per_sample | n |
|---|---:|---:|---:|---:|---:|
| stageB_drift | 0.6357185006 | 0.0275541027 | 0.4265960058 | 0.0153262412 | 3 |
| stageB_nodrift | 0.6333197842 | 0.0259367227 | 0.4335639666 | 0.0157830102 | 3 |

- Dice delta (`drift - no_drift`): `+0.0023987164`

### Interpretation
- Stage B is a modest improvement over Stage A, but still small overall.

---

## Iteration 11: Stage C Screen (Stage B + Mod #2)
- Date: March 26, 2026
- Setup: One-seed screen (seed 42), multiscale pooled features.

### Trials (seed 42)
| trial | pool_scales | dice | bce_loss | sec_per_sample |
|---:|---|---:|---:|---:|
| 0 | [2, 4] | 0.6292674065 | 0.4039867640 | 0.0138124347 |
| 1 | [2, 6] | 0.6301793039 | 0.4042026639 | 0.0143542097 |
| 2 | [2, 4, 8] | 0.6305581570 | 0.4011632502 | 0.0192243318 |

### Interpretation
- Stage C did not improve over Stage B screen and increased runtime in the best variant.
- Stage C was not promoted to 3-seed confirm.

---

## Iteration 12: Stage D Confirm (Stage B + Mod #5)
- Date: March 26, 2026
- Setup: Boundary-aware drift enabled with `boundary_gamma=2.0`, seeds `42, 43, 44`.

### Aggregate Summary
| variant | mean_dice | std_dice | mean_bce | mean_sec_per_sample | n |
|---|---:|---:|---:|---:|---:|
| stageD_drift | 0.6359375656 | 0.0253504385 | 0.4306476593 | 0.0163082939 | 3 |
| stageD_nodrift | 0.6335843553 | 0.0265542224 | 0.4323927214 | 0.0145855418 | 3 |

- Dice delta (`drift - no_drift`): `+0.0023532103`

### Interpretation
- Similar gain to Stage B, but slower; not a clear upgrade over simpler Stage B.

---

## Iteration 13: Stage E Confirm (Stage B + Mod #7)
- Date: March 26, 2026
- Setup: Time-aware drift scaling with `delta_t_beta=1.5`, `delta_t_center=0.6`, seeds `42, 43, 44`.

### Aggregate Summary
| variant | mean_dice | std_dice | mean_bce | mean_sec_per_sample | n |
|---|---:|---:|---:|---:|---:|
| stageE_drift | 0.6371155560 | 0.0269811778 | 0.4249438087 | 0.0161233642 | 3 |
| stageE_nodrift | 0.6332782507 | 0.0260515714 | 0.4323130161 | 0.0144320983 | 3 |

- Dice delta (`drift - no_drift`): `+0.0038373053`

### Interpretation
- Best of staged modifications so far, but still below the tuned hard baseline gain from Iteration 7.

---

## Iteration 14: Stage F Screen (Stage E + Mod #6)
- Date: March 26, 2026
- Setup: One-seed screen (seed 42), compare drift feature source `probs` vs `latent`.

### Screen Results (seed 42)
| variant | drift_feature_source | dice | bce_loss | sec_per_sample |
|---|---|---:|---:|---:|
| probs_ref | probs | 0.6366147161 | 0.3989271522 | 0.0187395729 |
| latent_try | latent | 0.6397397399 | 0.3780892551 | 0.0135357846 |

- Delta (`latent - probs`): `+0.0031250238`

### Interpretation
- Positive but below the pre-set promotion trigger (`+0.005` on seed screen).
- Stage F was not promoted to 3-seed confirm.

---

## Synthetic Modification Sweep: Final Takeaway
- Incremental mods A-F were implemented and tested.
- Gains on hard synthetic are real but modest (`~+0.001` to `~+0.004` Dice deltas).
- Most complex staged variants did not clearly outperform the simpler tuned hard baseline from Iteration 7 (`+0.0047182987`).

---

## Iteration 15: Experiment G (Local Token Drift) - Code Integration
- Date: March 30, 2026
- Objective: add local token-level drift (decoder feature tokens) to better capture scattered/irregular morphology.

### Implementation Status
- Added optional local token drift loss path in training.
- Added Experiment G configs (drift + matched no-drift).
- Next action: seed-42 screen followed by 3-seed confirm if promoted.

---

## Iteration 16: Experiment H (Component-Aware Drift) - Code Integration
- Date: March 30, 2026
- Objective: add component-aware mass/coverage loss to better model multifocal/satellite behavior.

### Implementation Status
- Added optional component-aware drift loss path in training.
- Added Experiment H configs (drift + matched no-drift).
- Next action: seed-42 screen followed by 3-seed confirm if promoted.

---

## Iteration 17: Experiment I (SDF Boundary Drift) - Code Integration
- Date: March 30, 2026
- Objective: add SDF-boundary loss to better align irregular tumor fronts and boundary geometry.

### Implementation Status
- Added optional SDF-boundary drift loss path in training.
- Added Experiment I configs (drift + matched no-drift).
- Next action: seed-42 screen followed by 3-seed confirm if promoted.
