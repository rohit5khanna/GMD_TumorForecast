# MVP-0 (No PDE): One-Shot Conditional Tumor Forecasting

This folder is the minimal PyTorch scaffold for Phase 1/2.

## Goal
- Train a one-shot conditional model on synthetic tumor data.
- Predict future tumor mask from early tumor state.
- Report Dice + inference time.

## File Map
- `dataset.py`: synthetic data generation + dataloaders
- `model.py`: one-shot 3D predictor model
- `drift_loss.py`: simplified drifting loss (MVP-1, PyTorch)
- `train.py`: training loop
- `eval.py`: evaluation loop (Dice/runtime)
- `infer.py`: inference entrypoint (reserved for next step)
- `config.yaml`: experiment config
- `RESULTS_LOG.md`: experiment history and aggregate metrics
- `ARCHITECTURE_DIAGRAMS.md`: shareable Mermaid diagrams of the pipeline
