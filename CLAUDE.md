# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning project focused on laser pulse analysis using TensorFlow/Keras. The project implements various neural network architectures (2D CNN, 3D U-Net, Informer, WaveNet) for image segmentation and analysis of laser pulse measurement data from thermographic and visual cameras.

## System Environment Setup

Always run Python, notebook, training, and test commands for this repository inside the Conda environment `pytorch_env`:
```bash
conda activate pytorch_env
```
For non-interactive checks, use `conda run -n pytorch_env ...`.

The codebase is designed to work across different systems with path configurations:
- `"GPU"`: HPC GPU cluster environment (current)
- `"Thermo10"`: Thermo workstation
- `"Linux"`: Linux workstation with mounted drives
- `"Windows"`: Windows environment with network drives

Network storage paths:
- Primary data: `\\Gfs01\g71\Thermo_Daten-MX2\...` (thermal IR data)
- Secondary storage: `\\gfs03\G33a\Thermo-MX1\...` (visual/test images)

Use `set_load_path(sys)` and `set_base_path(sys)` functions in scripts to configure paths for your environment.

## Architecture Overview

### Core Training Pipeline
- **AutoNNmerge.py**: Main training orchestration script with automatic model training, patching, k-fold cross-validation, and hyperparameter management
- **DataLoad.py**: Data loading utilities with path management and data preprocessing functions

### Model Architectures (`ModelsPy/`)
- **CNN2d.py**: Simple 2D CNN models (various configurations: 3x32, 5x32, 64x2)
- **Unet3D.py**: 3D U-Net implementation with custom ConvBlock3D layers and mixed precision support
- **informer.py / informer_claude.py**: Transformer-based models for time series analysis
- **WaveNetand3DUnet.py**: Hybrid WaveNet + 3D U-Net architecture

### Image Processing Pipeline (`BinaryMasks/`)

The `BinaryMasks` subdirectory contains preprocessing tools for creating training masks by aligning thermal and visual camera data:

**Key Workflows:**
1. **Image Alignment** - Align thermal IR images with visual camera reference images using homography transformations
2. **Geometric Correction** - Apply cylindrical coordinate transformations to correct barrel distortion
3. **Mask Generation** - Create binary masks from aligned overlays for model training

**Core Scripts:**
- **1ThermVisOverlay.py**: Interactive point selection for homography-based image alignment between thermal and visual images. Includes zoom/pan functionality for precise point matching.
- **NotMainMaskAppl.py**: Geometric transformations (rotation, cylindrical projection) to correct barrel surface distortion. Includes both outer and inner surface correction modes.
- **NotMainPhotConv.py**: Historical/experimental image processing techniques (SIFT/ORB feature matching, blob detection, template matching, perspective warping).
- **1OverlayMath.py**: Mask preprocessing - thresholding, morphological operations (erosion/dilation), connected component filtering.

**Data Flow:**
1. Load thermal IR (`meas2_{n}Img.npy`) and visual reference images
2. Manually select correspondence points using `InteractivePointSelector` class
3. Compute homography matrix and warp visual mask to thermal coordinate space
4. Apply morphological operations to clean up mask
5. Save aligned masks as `meas2_{n}MaskV*.npy` for training

**Important Classes:**
- `InteractivePointSelector` (1ThermVisOverlay.py:9): Interactive point selection with zoom, pan, and reference image display

**Key Functions:**
- `pixelrenum()` (NotMainMaskAppl.py:194, 280): Applies cylindrical coordinate transformation for barrel distortion correction
- `fill_black_pixels()` / `fill_zeros_with_neighbor_mean()`: Various implementations for filling gaps created by geometric transformations

### Training Features
- K-fold cross-validation support
- Mixed precision training (`mixed_float16`)
- Custom callbacks and learning rate scheduling
- Dice loss implementation for segmentation tasks
- Automatic model checkpointing and logging

## Data Structure

**Thermal IR Data:**
- Location: `\\Gfs01\g71\Thermo_Daten-MX2\2025\2025-02-12_av_hi-zika_velox_barrelpart_LDM500_puls\meas03_pos{n}_500ms_500W_100Hz_10avg_500us/`
- Raw thermal images: `meas2_{n}Img.npy`
- Masks: `meas2_{n}Mask.npy`, `meas2_{n}MaskV*.npy` (various versions)
- Binary masks use values: 0 (background), 1 or 255 (foreground)

**Visual Camera Data:**
- Location: `\\gfs03\G33a\Thermo-MX1\Mitarbeiter\Averin\Python\Sample4Fasseteil\Laser\Pulse\KerasKI\PreProcess\`
- H5 files: `VLXT-55C.I/meas1_.h5` (4D: height × width × frame × channel)
- Reference images: `TestImg/Image*.jpg`
- Processed overlays: `TestImg/ImageOver_*.jpg`

**Model Training Data:**
- NPZ format with keys like `highlighted_cleaned_image`
- Masks in `.npy` format (binary or normalized)

## Directory Structure

- `models/`: Saved trained models organized by architecture and version
- `logs/`: TensorBoard logs and training logs
- `ModelsPy/`: Neural network architecture definitions
- `Forpaper/`: Research/paper related scripts
- `BinaryMasks/`: Image preprocessing and mask generation tools

## Running Scripts

### Training
```bash
python AutoNNmerge.py
```

### Testing GPU Setup
```bash
python test.py
```

### Model Validation
```bash
python validation_temp.py
```

### Mask Generation (Interactive)
Scripts in `BinaryMasks/` are designed to run interactively in Jupyter/IPython cells (marked with `# %%`). They require manual point selection via OpenCV windows.

## Key Configuration Notes

- CUDA device selection via `os.environ["CUDA_VISIBLE_DEVICES"]`
- Mixed precision training enabled globally in Unet3D
- TensorBoard integration for training monitoring
- Models saved in HDF5 format with custom serialization support
- Interactive OpenCV windows require X11 forwarding or local display

## Model Versioning

Models are organized with descriptive names indicating:
- Architecture (2DCNN, Unet3D, WaveNet+3DUnet)
- Parameters (layer counts, filter sizes)
- Training specifics (Dice loss, dropout rates, learning rates)
- Version numbers (v1, v2, etc.)

Mask versions (V1, V2, V3, VUpd1) indicate different preprocessing or dilation/erosion iterations.

---

## NNsTorch / HybridTrainV2 Pipeline (PyTorch)

Active pipeline for hybrid XGBoost + CNN segmentation. Entry point: `NNsTorch/HybridTrainV2/HybridTraining.ipynb`.

### Modes

| Mode | CNN input | Loss signal | Notes |
|---|---|---|---|
| `prob_only` | XGB prob (1 ch) | CNN logit | Best results so far (IoU ~0.88) |
| `prob_feat` | features + XGB prob (C+1 ch) | CNN logit | |
| `parallel` | features (C ch) | combined logit (w·CNN + (1-w)·XGB) | IoU ~0.47–0.55 after bug fixes |
| `nn_only` | features (C ch) | CNN logit | Standalone CNN; cold-start IoU ~0.18 |

### Key Files

- `hybrid_manager.py` — `HybridTrainingManager`: data loading, k-fold loop, checkpointing, MLflow
- `hybrid_models.py` — CNN architectures: `RefinementCNNSE` (cnn_se), `RefinementCNNSkip`, `RefinementMLP`, `TCNEncoder`, `BidirectionalWaveNetEncoder`; `FusionWeight` for parallel mode
- `hybrid_utils.py` — `HybridPatchDataset`, `create_hybrid_dataloader`
- `ConHybridTraining.ipynb` — warm-start fine-tuning: loads parallel checkpoint, fine-tunes in nn_only mode

### Architecture: RefinementCNNSE (cnn_se)
```
block1: Conv-BN-ReLU(in→32) + SEBlock
block2: Conv-BN-ReLU(32→32) + SEBlock
block3: Conv-BN-ReLU(32→16) + SEBlock
head:   Conv2d(16→1, 1×1)   ← output bias init to -2.2 (class-imbalance prior)
```

### Known Issues Fixed (2026-03-05)

1. **`FusionWeight` int tensor**: `torch.tensor(init_logit)` with integer `INIT_W=0` creates int64 → no gradients. Fix: `torch.tensor(float(init_logit))`.
2. **Parallel mode double-sigmoid + gradient kill**: old code did `sigmoid(cnn_logit)` → combine in prob space → `logit(combined)` → criterion. Saturates sigmoid, kills gradient to `logit_w`. Fix: combine in logit space directly: `w * cnn_logit + (1-w) * xgb_logit`.
3. **`_validate` tracking wrong signal**: used CNN-only logit for parallel mode → val_loss diverged from train_loss. Fix: `_validate` and `_validate_final` now use combined logit for parallel mode.
4. **Missing `import torch.nn.functional as F`** in `hybrid_models.py` — needed by `TCNBlock`.

### Tversky Loss Guidance
- Sparse masks (~5–10% foreground): use `alpha=0.3, beta=0.7` (penalise FN more than FP)
- Default `alpha=0.65, beta=0.35` drives model to predict all-negative on sparse masks

### Warm-Start Fine-Tuning (`run_kfold` params)
```python
manager.run_kfold(
    model_fn,
    warmstart_ckpt_paths = {1: 'path/fold1_ep03.pt', ...},  # load from parallel run
    head_freeze_epochs   = 5,    # phase 1: head-only at head_lr
    head_lr              = 1e-3, # phase 2: full model at initial_lr (1e-4)
)
```
Checkpoint format: `{'epoch', 'fold', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'fusion_state_dict'}`.

### Git Workflow
- Working branch: `claude-work` — push all Claude-authored changes here
- Main branch: `main`
