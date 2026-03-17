# NNsTorchV2 — Structure Reference

Self-contained, modular PyTorch pipeline for hybrid XGBoost+CNN segmentation of laser pulse thermographic data.
Refactored from `NNsTorch` (flat layout) into a two-layer package: `core/` (shared utilities) + `HybridTrainV2/` (training logic).

---

## Top-level package

```
NNsTorchV2/
├── __init__.py              # Re-exports: set_load_path, get_loss_function,
│                            #             MemoryCleanupCallback,
│                            #             HybridTrainingManager, build_hybrid_model
├── STRUCTURE.md             # This file
├── core/                    # Shared data/loss/callback utilities
└── HybridTrainV2/           # K-fold hybrid training pipeline
```

**Quick import:**
```python
from NNsTorchV2 import HybridTrainingManager, build_hybrid_model, get_loss_function
```

---

## `core/` — Shared Utilities

Pure utilities with no training logic. Used by `HybridTrainV2` and can be used standalone.

| File | Purpose |
|---|---|
| `config_paths.py` | `set_load_path(sys)` / `set_base_path(sys)` — environment-aware path config (GPU, Thermo10, Linux, Windows) |
| `data_discovery.py` | Scan dataset directories; discover `PCA/PPT/ICA/Raw` `.npz` files + `MaskV2` masks; validate `.npz` integrity |
| `data_loading.py` | `load_and_aggregate_location()` — load and stack feature arrays for one (sample, location) |
| `data_pipeline.py` | Eager data pipeline (loads all data into memory upfront) |
| `data_pipeline_lazy.py` | Lazy data pipeline (loads per-batch; lower memory footprint) |
| `patch_extraction.py` | `extract_patches_from_image()` — sliding-window patch extraction |
| `full_img_padding.py` | `extract_full_padding_patch()` — padded full-image patch extraction |
| `losses.py` | Loss function registry — see table below |
| `lovasz_loss.py` | Lovász-Hinge loss implementation (third-party, segmentation-optimised) |
| `callbacks.py` | `SafeModelCheckpoint`, `MemoryCleanupCallback`, `DebugOpenFilesCallback`, `LearningRateLogger` |

### Data regimes (`data_discovery.py`)

Two regimes controlled by `data_regime` parameter:

- **`postprocessed`** — expects `PCA_a=…_width=…_{loc}.npz`, `PPT_…`, `ICA_…` files + `MaskV2_{loc}.npy`
- **`raw`** — expects `Raw_{loc}.npz` + `MaskV2_{loc}.npy`

### Available loss functions (`losses.py`)

| Key (string) | Class | Notes |
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

All losses operate on **logits** (sigmoid applied internally), except `nn.BCELoss`.

---

## `HybridTrainV2/` — Training Pipeline

K-fold hybrid XGBoost+CNN training with MLflow logging. Organised around a strategy pattern so adding a new mode requires only a new Strategy subclass.

```
HybridTrainV2/
├── __init__.py              # Re-exports everything below
├── hybrid_manager.py        # HybridTrainingManager (main entry point)
├── HybridTraining.ipynb     # Main training notebook (standard k-fold run)
├── ConHybridTraining.ipynb  # Continued training notebook (warm-start fine-tuning)
└── components/
    ├── hybrid_models.py     # CNN architectures + FusionWeight
    ├── hybrid_utils.py      # Dataset + DataLoader factory
    ├── forward_strategies.py# Mode-specific forward/loss routing
    ├── epoch_runner.py      # Stateless train_epoch / validate / validate_final
    ├── threshold_tuner.py   # IoU-maximising threshold search
    ├── infrastructure.py    # TrainingInfrastructure (dirs, MLflow, data discovery)
    └── warm_start.py        # Two-phase warm-start: head-freeze → full fine-tune
```

### Hybrid modes

| Mode | CNN input | XGB role | Fusion |
|---|---|---|---|
| `prob_only` | XGB probability map (1 ch) | Produces input | None — CNN refines XGB output |
| `prob_feat` | Raw features + XGB prob (C+1 ch) | Provides prior | Concatenated at input |
| `parallel` | Raw features only (C ch) | Independent | CNN prob + XGB prob combined at inference (mean/max) |
| `nn_only` | Raw features only (C ch) | Not used | Pure CNN baseline |

### CNN architectures (`hybrid_models.py`)

| Class | Description |
|---|---|
| `RefinementCNN` | 4-layer Conv-BN-ReLU + 1×1 head. Shared across all modes (only `in_channels` differs). Kaiming init; class-imbalance prior bias `≈ -2.2`. |
| `SEBlock` | Squeeze-and-Excitation channel attention (reduction=8) |
| `RefinementCNNSE` | `RefinementCNN` + SE attention after each conv block |
| `FusionWeight` | Learnable scalar `w` for blending CNN and XGB probabilities in `parallel` mode |
| `build_hybrid_model` | Factory: instantiates correct CNN + optional `FusionWeight` given mode and input channels |

### `HybridTrainingManager` — key constructor parameters

```python
HybridTrainingManager(
    model_name    : str,               # MLflow run name / checkpoint prefix
    sys           : str,               # Environment key ("GPU", "Thermo10", ...)
    mode          : str,               # One of VALID_MODES above
    xgb_model_path: str | None,        # Path to pre-trained XGBoost .pkl (None for nn_only)
    power_mode    : str = '4kw_both',  # Data subfolder
    subfolder_name: str,               # Sub-path within load_path
    patch_size    : tuple = (128,128),
    initial_lr    : float = 1e-3,
    optimizer_name: str  = 'adam',     # adam | adamw | nadam | rmsprop
    loss_name     : str,               # See loss table above
    n_filters     : int  = 32,
    n_splits      : int  = 5,          # K-fold splits
    ...
)
manager.run()                          # Executes full k-fold training loop
```

### Component responsibilities

| Component | Key symbols | Responsibility |
|---|---|---|
| `infrastructure.py` | `TrainingInfrastructure` | Creates output dirs, starts MLflow run, calls `discover_samples`, infers input shape |
| `forward_strategies.py` | `make_strategy(mode)` | Returns the correct `BaseForwardStrategy` subclass for the mode |
| `epoch_runner.py` | `train_epoch`, `validate`, `validate_final` | Stateless loops over DataLoader; compute loss + IoU/F1 metrics |
| `threshold_tuner.py` | `find_best_threshold` | Grid search over threshold values to maximise validation IoU |
| `warm_start.py` | `setup_warmstart`, `maybe_transition_phase2`, `unfreeze_all` | Load checkpoint to resume; manage two-phase head-freeze → full fine-tune transitions |

### Warm-start two-phase strategy (`warm_start.py`)

Used by `ConHybridTraining.ipynb` to fine-tune a `parallel`-mode checkpoint in `nn_only` mode.

| Phase | Layers trained | LR | Trigger |
|---|---|---|---|
| 1 — Head recalibration | `head.*` only (1×1 conv) | `head_lr` | `setup_warmstart()` when `head_freeze_epochs > 0` |
| 2 — Full fine-tune | all layers unfrozen | `initial_lr` + CosineAnnealingLR | `maybe_transition_phase2()` at epoch == `head_freeze_epochs` |

**Why**: the CNN trained in `parallel` mode optimised under a blended logit objective (fusion weight `w ≈ 0.48`). Its feature detectors are strong but the output bias is miscalibrated for standalone use. Phase 1 recalibrates the head; Phase 2 fine-tunes all layers at a reduced LR to preserve the learned features.

**API:**
```python
# Per-fold checkpoint paths: {fold_int: '/path/to/fold1_ep03.pt'} or 'best'
phase1_active, phase1_opt = setup_warmstart(
    model, warmstart_ckpt_paths, fold, device,
    head_freeze_epochs, head_lr, optimizer_name, weight_decay)

# Called inside epoch loop:
phase1_active, optimizer, scheduler = maybe_transition_phase2(
    epoch, head_freeze_epochs, phase1_active, model,
    get_optimizer_fn, remaining_epochs, initial_lr)
```

### Notebooks

| Notebook | Purpose |
|---|---|
| `HybridTraining.ipynb` | Standard entry point — configure mode/loss/data, call `manager.run_kfold()`. Includes optional inference check cell. Saves checkpoints to `checkpoints/<model_name>/<timestamp>/fold{N}_best.pt`. |
| `ConHybridTraining.ipynb` | Warm-start fine-tuning — loads `parallel`-mode checkpoints, switches to `nn_only` mode, runs two-phase training via `warmstart_ckpt_paths` + `head_freeze_epochs`/`head_lr` args to `run_kfold()`. |

---

## Dependency graph

```
HybridTrainingManager
    └── TrainingInfrastructure   ← core.data_discovery, core.config_paths
    └── make_strategy(mode)      ← forward_strategies → hybrid_models
    └── train_epoch / validate   ← epoch_runner → HybridPatchDataset (hybrid_utils)
    └── find_best_threshold      ← threshold_tuner
    └── get_loss_function        ← core.losses
    └── MemoryCleanupCallback    ← core.callbacks
```

---

## Environment paths (`config_paths.py`)

| Key | Load path (data) | Base path (models/logs) |
|---|---|---|
| `GPU` | `/home/aaverin/RZ-Dienste/hpc-user/aaverin/2025/…` | `/home/aaverin/RZ-Dienste/hpc-user/aaverin/Python/Pulse/KIprojV2_Claude` |
| `Thermo10` | `/home/aaverin/RZ-Dienste/Thermo_Daten-MX2/2025/…` | `/home/aaverin/RZ-Dienste/Thermo-MX1/Mitarbeiter/…/KerasKIv2` |
| `Linux` | `/mnt/daten-mx2/2025/…` | `/mnt/thermo/Mitarbeiter/…/KerasKIv2` |
| `Windows` | `\\Gfs01\g71\Thermo_Daten-MX2\2025\…` | `\\gfs03\G33a\Thermo-MX1\Mitarbeiter\…\KerasKIv2` |
