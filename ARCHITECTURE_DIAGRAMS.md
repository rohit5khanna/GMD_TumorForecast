# Architecture Diagrams

Date: March 25, 2026  
Project: GMD tumor forecasting MVP (PyTorch)

## 1) Training Pipeline (MVP-1 with Optional Drifting Loss)

```mermaid
flowchart LR
    A["SyntheticTumorDataset"] --> B["Input x: [baseline_mask, delta_t_map]"]
    A --> C["Target y: future_mask"]

    B --> D["OneShotPredictor (3D U-Net style)"]
    D --> E["Logits"]
    E --> F["Sigmoid -> Predicted future mask"]

    E --> G["Segmentation Loss: BCEWithLogits + SoftDice"]

    F --> H["Pooled generated features"]
    C --> I["Pooled target features"]
    H --> J["Drifting field V = V_pos - V_neg"]
    I --> J
    J --> K["Drift Loss: MSE(f_gen, stopgrad(f_gen + V))"]

    G --> L["Total Loss: L = L_seg + lambda_drift * L_drift"]
    K --> L
    L --> M["Backprop + Adam update"]

    D -. save best .-> N["outputs/model_best.pt"]
```

## 2) Inference Path (One-Shot)

```mermaid
flowchart LR
    A["baseline_mask + delta_t_map"] --> B["OneShotPredictor"]
    B --> C["Sigmoid + threshold"]
    C --> D["Future tumor mask (single forward pass)"]
```

## 3) Notes
- `use_drift_loss: false` gives MVP-0 behavior (segmentation-only baseline).
- `use_drift_loss: true` enables MVP-1 behavior (segmentation + drifting regularization).
- Both variants keep one-shot inference at test time.
