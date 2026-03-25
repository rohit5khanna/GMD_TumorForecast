# MVP-0 (No PDE): One-Shot Conditional Tumor Forecasting

This folder is the minimal PyTorch scaffold for Phase 1.

## Goal
- Train a one-shot conditional model on synthetic tumor data.
- Predict future tumor mask from early tumor state.
- Report Dice + inference time.

## File Map
- `src/data/synthetic_dataset.py`: synthetic data generation + dataset class
- `src/models/oneshot_predictor.py`: minimal one-shot model
- `src/train.py`: training loop
- `src/evaluate.py`: evaluation loop (Dice/runtime)
- `src/infer.py`: quick inference + save predictions
- `src/utils/metrics.py`: Dice and helpers
- `src/utils/visualize.py`: qualitative plots
- `configs/mvp0_synth.yaml`: experiment config
- `scripts/run_*.sh`: convenience launch scripts
