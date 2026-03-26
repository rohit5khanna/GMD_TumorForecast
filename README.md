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
