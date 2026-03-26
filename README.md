# MVP-0 (No PDE): One-Shot Conditional Tumor Forecasting

This folder is the minimal PyTorch scaffold for Phase 1/2.

## Goal
- Train a one-shot conditional model on synthetic tumor data.
- Predict future tumor mask from early tumor state.
- Report Dice + inference time.
- Run matched drift vs no-drift ablations.

## File Map
- `dataset.py`: synthetic data generation + dataloaders
- `model.py`: one-shot 3D predictor model
- `drift_loss.py`: simplified drifting loss (MVP-1, PyTorch)
- `train.py`: training loop
- `eval.py`: evaluation loop (Dice/runtime)
- `infer.py`: inference entrypoint (reserved for next step)
- `config.yaml`: experiment config
- `config_mvp1_hard.yaml`: harder synthetic benchmark config (drift on)
- `config_mvp1_hard_nodrift.yaml`: matched harder benchmark control (drift off)
- `config_mvp1_stageA.yaml`: incremental Stage A config (#1 + #4 enabled)
- `config_mvp1_stageA_nodrift.yaml`: matched Stage A no-drift control
- `MVP1_PROTOCOL.md`: exact multi-seed Colab protocol for matched hard-synthetic runs
- `RESULTS_LOG.md`: experiment history and aggregate metrics
- `ARCHITECTURE_DIAGRAMS.md`: shareable Mermaid diagrams of the pipeline

## MVP-1 Quick Start (Hard Synthetic, Matched Ablation)
Drift model:
```bash
python train.py --config config_mvp1_hard.yaml
python eval.py --config config_mvp1_hard.yaml --ckpt outputs/mvp1_hard_drift/model_best.pt --split val
```

Matched no-drift control:
```bash
python train.py --config config_mvp1_hard_nodrift.yaml
python eval.py --config config_mvp1_hard_nodrift.yaml --ckpt outputs/mvp1_hard_nodrift/model_best.pt --split val
```

Stage A (stronger negatives + warmup):
```bash
python train.py --config config_mvp1_stageA.yaml
python eval.py --config config_mvp1_stageA.yaml --ckpt outputs/mvp1_stageA/model_best.pt --split val

python train.py --config config_mvp1_stageA_nodrift.yaml
python eval.py --config config_mvp1_stageA_nodrift.yaml --ckpt outputs/mvp1_stageA_nodrift/model_best.pt --split val
```

## Drift Ablation Knobs (Incremental Mods 1-7)
- `drift_use_memory_bank`, `drift_neg_bank_size` : stronger negatives (#1)
- `drift_lambda_warmup_epochs` : lambda warmup (#4)
- `drift_pos_weight`, `drift_neg_weight` : separate attraction/repulsion weights (#3)
- `drift_pool_scales` : multiscale drift features (#2)
- `drift_boundary_gamma` : boundary-aware emphasis (#5)
- `drift_delta_t_beta`, `drift_delta_t_center` : time-aware drift scaling (#7)
- `drift_feature_source` (`probs` or `latent`) : latent feature drift option (#6)
