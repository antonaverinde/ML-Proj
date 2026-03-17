"""
Training infrastructure — directories, MLflow, sample discovery, input-shape.

Isolated here so that hybrid_manager.py only handles the training loop logic.
"""

import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("[WARNING] MLflow not installed — logging disabled")

from ...core.config_paths import set_load_path
from ...core.data_discovery import discover_samples, discover_data_files_for_location
from ...core.data_loading import calculate_total_channels


class TrainingInfrastructure:
    """Handles directories, MLflow experiment, sample discovery, and input-shape."""

    def __init__(
        self,
        model_name: str,
        sys: str,
        mode: str,
        subfolder_name: str,
        power_mode: str,
        patch_size: list,
        patch_mode: str,
        mask_type: str,
        dirs: List[int],
        data_regime: str,
        ppt_phases: Union[str, int],
        ppt_amps: int,
        mlflow_uri: str,
        base_path: str,
    ):
        self.model_name   = model_name
        self.mode         = mode
        self.power_mode   = power_mode
        self.patch_size   = patch_size
        self.patch_mode   = patch_mode
        self.mask_type    = mask_type
        self.dirs         = dirs
        self.data_regime  = data_regime
        self.ppt_phases   = ppt_phases
        self.ppt_amps     = ppt_amps
        self.mlflow_uri   = mlflow_uri
        self.base_path    = base_path
        self.load_path    = os.path.join(set_load_path(sys), subfolder_name)

        self.versioned_name: str = ''
        self.ckpt_dir: str       = ''
        self.model_save_loc: str = ''

    # ── Directories ───────────────────────────────────────────────────────────

    def setup_directories(self) -> Tuple[str, str, str]:
        """Create versioned model/checkpoint dirs.  Returns (versioned_name, ckpt_dir, model_save_loc)."""
        model_dir_base = os.path.join(self.base_path, 'models', self.model_name)
        version_num, model_dir = 0, model_dir_base
        while os.path.exists(model_dir) and os.listdir(model_dir):
            version_num += 1
            model_dir = f"{model_dir_base}_{version_num}"
        versioned_name = (self.model_name if version_num == 0
                          else f"{self.model_name}_{version_num}")
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.versioned_name  = versioned_name
        self.ckpt_dir        = os.path.join(self.base_path, 'checkpoints', versioned_name, ts)
        self.model_save_loc  = os.path.join(self.base_path, 'models', versioned_name, ts + '.pt')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        print(f"Output directory: {versioned_name}")
        return versioned_name, self.ckpt_dir, self.model_save_loc

    # ── MLflow ────────────────────────────────────────────────────────────────

    def setup_mlflow(self) -> None:
        """Configure MLflow tracking URI and experiment."""
        if not HAS_MLFLOW:
            return
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment('HybridTrainV2')
        print(f"MLflow tracking: {self.mlflow_uri}")

    def start_run(self, run_name: str) -> None:
        """Start an MLflow run."""
        if HAS_MLFLOW:
            mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        """End the current MLflow run."""
        if HAS_MLFLOW:
            mlflow.end_run()

    def log_params(self, params: dict) -> None:
        """Log a dict of hyperparameters to MLflow."""
        if HAS_MLFLOW:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log a dict of metrics to MLflow at the given step."""
        if HAS_MLFLOW:
            mlflow.log_metrics(metrics, step=step)

    # ── Sample discovery ──────────────────────────────────────────────────────

    def discover_samples(self) -> List[Tuple[str, int]]:
        """Discover all (sample_name, location_idx) pairs in the dataset."""
        return discover_samples(
            self.load_path, self.power_mode,
            mask_type=self.mask_type, dirs=self.dirs,
            data_regime=self.data_regime)

    # ── Input shape ───────────────────────────────────────────────────────────

    def determine_input_shape(
        self, samples: List[Tuple[str, int]]
    ) -> Tuple[int, Tuple[int, int, int], list]:
        """Compute CNN input channels and spatial patch size from samples.

        For full_padding: auto-computes min(H, W) and updates self.patch_size.
        Returns (n_raw_ch, input_shape (C, H, W), patch_size [H, W]).
        """
        if not samples:
            raise ValueError("No valid samples found!")

        sample, loc = samples[0]
        sample_dir = os.path.join(self.load_path, self.power_mode, sample)
        fi = discover_data_files_for_location(
            sample_dir, loc, self.mask_type, self.data_regime)
        n_raw_ch = calculate_total_channels(
            fi, ppt_phases=self.ppt_phases,
            ppt_amps=self.ppt_amps, data_regime=self.data_regime)

        # CNN input channels depend on mode
        if self.mode == 'prob_only':
            n_ch = 1
        elif self.mode == 'prob_feat':
            n_ch = n_raw_ch + 1
        else:  # 'parallel' or 'nn_only'
            n_ch = n_raw_ch

        # Spatial size
        if self.patch_mode == 'full_padding':
            min_h = min_w = float('inf')
            for s, l in samples:
                sd  = os.path.join(self.load_path, self.power_mode, s)
                fi2 = discover_data_files_for_location(sd, l, self.mask_type, self.data_regime)
                mask = np.load(fi2['mask'])
                min_h = min(min_h, mask.shape[0])
                min_w = min(min_w, mask.shape[1])
            self.patch_size = [int(min_h), int(min_w)]
            H, W = int(min_h), int(min_w)
            print(f"Auto patch_size (min H/W): {self.patch_size}")
        else:
            H, W = self.patch_size[0], self.patch_size[1]

        input_shape = (n_ch, H, W)
        print(f"n_raw_ch={n_raw_ch}  CNN input: {input_shape}  (mode={self.mode})")
        return n_raw_ch, input_shape, self.patch_size
