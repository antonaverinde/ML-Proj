# ML-Project

A modular PyTorch pipeline for training hybrid XGBoost+CNN segmentation models on laser pulse thermographic datasets.

---

## Repository Structure

```
ML-Project/
└── NNsTorchV2/              # Main pipeline package
    ├── core/                # Shared data, loss, and callback utilities
    └── HybridTrainV2/       # K-fold hybrid training pipeline
```

---

## NNsTorchV2

Self-contained pipeline for hybrid XGBoost+CNN segmentation of laser pulse thermographic data. Refactored from a flat layout into a two-layer package: `core/` (shared utilities) and `HybridTrainV2/` (training logic).

**Quick import:**
```python
from NNsTorchV2 import HybridTrainingManager, build_hybrid_model, get_loss_function
```

---

## `core/` — Shared Utilities

Pure utilities with no training logic. Used by `HybridTrainV2` and can be used standalone.

| File | Purpose |
|---|---|
| `config_paths.py` | Environment-aware path configuration (GPU, Thermo10, Linux, Windows) |
| `data_discovery.py` | Scan dataset directories; discover PCA/PPT/ICA/Raw `.npz` files and `MaskV2` masks |
| `data_loading.py` | Load and aggregate feature arrays for one (sample, location) |
| `data_pipeline.py` | Eager data pipeline — loads all data into memory upfront |
| `data_pipeline_lazy.py` | Lazy data pipeline — loads per-batch for lower memory footprint |
| `patch_extraction.py` | Sliding-window patch extraction |
| `full_img_padding.py` | Padded full-image patch extraction |
| `losses.py` | Loss function registry (see table below) |
| `lovasz_loss.py` | Lovász-Hinge loss implementation |
| `callbacks.py` | `SafeModelCheckpoint`, `MemoryCleanupCallback`, `LearningRateLogger`, and more |

### Available Loss Functions

| Key | Class | Notes |
|---|---|---|
| `'dice'` | `DiceLoss` | Standard soft Dice |
| `'log_dice'` | `LogDiceLoss` | `-log(1 - DiceLoss)` |
| `'soft_iou'` | `SoftIoULoss` | Soft intersection-over-union |
| `'weighted'` | `WeightedBCE` | Weighted BCE with `pos_w` |
| `'focal'` | `FocalLoss` | Focal loss with `alpha`, `gamma`, `pos_weight` |
| `'tversky'` | `TverskyLoss` | alpha=0.7 FP / beta=0.3 FN penalty |
| `'combined'` | `CombinedLoss` | `alpha*WeightedBCE + beta*Dice` |
| `'combined2'` | `CombinedLoss2` | `Focal + dice_weight*Dice` |
| `'lovasz'` | `LovaszHingeLoss` | Lovász-Hinge (operates on logits) |
| _(default)_ | `nn.BCELoss` | Plain BCE |

All losses operate on logits (sigmoid applied internally), except `nn.BCELoss`.

---

## `HybridTrainV2/` — Training Pipeline

K-fold hybrid XGBoost+CNN training with MLflow logging. Organised around a strategy pattern — adding a new hybrid mode requires only a new `ForwardStrategy` subclass.

### Hybrid Modes

| Mode | CNN Input | XGB Role | Fusion |
|---|---|---|---|
| `prob_only` | XGB probability map (1 ch) | Produces input | CNN refines XGB output |
| `prob_feat` | Raw features + XGB prob (C+1 ch) | Provides prior | Concatenated at input |
| `parallel` | Raw features only (C ch) | Independent | CNN + XGB probabilities combined at inference |
| `nn_only` | Raw features only (C ch) | Not used | Pure CNN baseline |

### CNN Architectures

| Class | Description |
|---|---|
| `RefinementCNN` | 4-layer Conv-BN-ReLU + 1×1 head. Kaiming init with class-imbalance prior bias. |
| `RefinementCNNSE` | `RefinementCNN` with Squeeze-and-Excitation channel attention after each block |
| `FusionWeight` | Learnable scalar for blending CNN and XGB probabilities in `parallel` mode |
| `build_hybrid_model` | Factory function — instantiates the correct CNN and optional `FusionWeight` |

### Quick Start

```python
from NNsTorchV2 import HybridTrainingManager

manager = HybridTrainingManager(
    model_name     = "my_run",
    sys            = "GPU",            # "GPU" | "Thermo10" | "Linux" | "Windows"
    mode           = "prob_only",      # hybrid mode
    xgb_model_path = "/path/to/xgb.pkl",
    power_mode     = "4kw_both",
    subfolder_name = "my_dataset",
    patch_size     = (128, 128),
    initial_lr     = 1e-3,
    loss_name      = "combined",
    n_filters      = 32,
    n_splits       = 5,
)
manager.run()
```

### Notebooks

| Notebook | Purpose |
|---|---|
| `HybridTraining.ipynb` | Standard entry point — configure mode/loss/data and run k-fold training |
| `ConHybridTraining.ipynb` | Warm-start fine-tuning — loads `parallel`-mode checkpoints and switches to `nn_only` mode |

### Warm-Start Fine-Tuning

Used in `ConHybridTraining.ipynb` to fine-tune a `parallel`-mode checkpoint in `nn_only` mode using a two-phase strategy:

| Phase | Layers Trained | Trigger |
|---|---|---|
| 1 — Head recalibration | `head.*` only (1×1 conv) | Start, when `head_freeze_epochs > 0` |
| 2 — Full fine-tune | All layers unfrozen, CosineAnnealingLR | At epoch == `head_freeze_epochs` |

---

## Environment Paths

Configured via `core/config_paths.py`:

| Key | Description |
|---|---|
| `GPU` | HPC cluster (Linux) |
| `Thermo10` | Thermo10 workstation (Linux) |
| `Linux` | Local Linux machine |
| `Windows` | Windows workstation (network paths) |

---

## Dependencies

- Python 3.8+
- PyTorch
- XGBoost
- MLflow
- NumPy
