"""
HybridTrainingManager V2 — identical training logic, H5 data source.

Switch from V1 to V2 with one import line:
    from NNsTorchV2.HybridTrainV2.hybrid_manager_V2 import HybridTrainingManager

Also set subfolder_name='Taris/Data_ML_V1_h5' in the constructor.
"""

import gc
import os
from collections import Counter
from sched import scheduler
from typing import Callable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, NAdam, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR, LambdaLR
)
from sklearn.model_selection import StratifiedKFold

try:
    import joblib
except ImportError:
    joblib = None

from ..core.data_discovery_V2 import discover_data_files_for_location
from ..core.data_loading_V2 import load_and_aggregate_location
from ..core.losses import get_loss_function
from ..core.callbacks import MemoryCleanupCallback

from .components.hybrid_utils_V2 import create_hybrid_dataloader
from .components.hybrid_models import FusionWeight
from .components.forward_strategies import make_strategy
from .components.epoch_runner import train_epoch, validate
from .components.threshold_tuner import find_best_threshold
from .components.infrastructure_V2 import TrainingInfrastructure
from .components.warm_start import setup_warmstart, maybe_transition_phase2

VALID_MODES = ('prob_only', 'prob_feat', 'parallel', 'nn_only')


class HybridTrainingManager:
    """K-fold training manager for Hybrid XGBoost+CNN models with H5 data source."""

    def __init__(
        self,
        model_name: str,
        sys: str,
        mode: str,
        xgb_model_path: Optional[str] = None,
        power_mode: str = '4kw_both',
        subfolder_name: str = 'Taris/Data_ML_V1_h5',   # V2: changed from 'Taris/Data_ML_V3'
        patch_size: tuple = (128, 128),
        initial_lr: float = 1e-3,
        drop: float = 0.5,
        epochs_drop: int = 10,
        optimizer_name: str = 'adam',
        scheduler_name: str = 'cosine',
        loss_name: str = 'tversky',
        alpha: float = 0.65,
        beta: float = 0.35,
        mask_type: str = 'alternative',
        restore_best: bool = True,
        dirs: Optional[List[int]] = None,
        rotate_img: bool = True,
        ppt_phases: Union[str, int] = 'all',
        ppt_amps: int = 6,
        invert_mask: bool = True,
        apply_jitter: bool = True,
        mlflow_uri: str = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:////tmp/mlflow_experiments/mlflow.db'),
        augment: bool = True,
        min_positive_ratio: float = 0.05,
        patch_mode: str = 'full_padding',
        pos_w: float = 1.0,
        comp_weights: bool = False,
        data_regime: str = 'postprocessed',
        weight_decay: float = 1e-4,
        save_chckpnt: Optional[str] = None,
        combine: str = 'mean',
        fusion_freeze_epochs: int = 5,
        init_w: float = 0.0,
        dataset_n: str = "V1",
        max_locations: Optional[int] = None,
        min_mask_area: int = 0,
        warmup_epochs: int = 3,
    ):
        assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}, got {mode!r}"
        if mode != 'nn_only':
            assert xgb_model_path is not None, \
                "xgb_model_path is required for all modes except 'nn_only'"
            assert joblib is not None, "joblib is required (pip install joblib)"

        self.model_name          = model_name
        self.mode                = mode
        self.combine             = combine
        self.initial_lr          = initial_lr
        self.drop                = drop
        self.epochs_drop         = epochs_drop
        self.optimizer_name      = optimizer_name.lower()
        self.scheduler_name      = scheduler_name.lower()
        self.loss_name           = loss_name.lower()
        self.alpha               = alpha
        self.beta                = beta
        self.mask_type           = mask_type
        self.restore_best        = restore_best
        self.rotate_img          = rotate_img
        self.ppt_phases          = ppt_phases
        self.ppt_amps            = ppt_amps
        self.invert_mask         = invert_mask
        self.dirs                = dirs if dirs is not None else []
        self.apply_jitter        = apply_jitter
        self.augment             = augment
        self.min_positive_ratio  = min_positive_ratio
        self.patch_mode          = patch_mode
        self.pos_w               = pos_w
        self.neg_w               = 1.0
        self.comp_weights        = comp_weights
        self.data_regime         = data_regime
        self.save_chckpnt        = save_chckpnt
        self.weight_decay        = weight_decay
        self.init_w              = init_w
        self.fusion_freeze_epochs = fusion_freeze_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_n = dataset_n
        self.max_locations = max_locations
        self.min_mask_area = min_mask_area
        self.warmup_epochs = warmup_epochs  # For cosine_warmup scheduler
        print(f"Device: {self.device}  |  mode: {mode}  |  patch_mode: {patch_mode}")

        if mode != 'nn_only':
            self.xgb_global = joblib.load(xgb_model_path)
            print(f"Loaded XGBoost: {os.path.basename(xgb_model_path)}")
        else:
            self.xgb_global = None

        self.infra = TrainingInfrastructure(
            model_name=model_name, sys=sys, mode=mode,
            subfolder_name=subfolder_name, power_mode=power_mode,
            patch_size=list(patch_size), patch_mode=patch_mode,
            mask_type=mask_type, dirs=self.dirs,
            data_regime=data_regime, ppt_phases=ppt_phases, ppt_amps=ppt_amps,
            mlflow_uri=mlflow_uri,
            base_path=os.path.dirname(os.path.abspath(__file__)),
            max_locations=self.max_locations,
        )
        self.infra.setup_directories()
        self.infra.setup_mlflow()

        self.versioned_name  = self.infra.versioned_name
        self.ckpt_dir        = self.infra.ckpt_dir
        self.model_save_loc  = self.infra.model_save_loc
        self.load_path       = self.infra.load_path
        self.power_mode      = power_mode

        self.all_samples = self.infra.discover_samples()
        print(f"Discovered {len(self.all_samples)} samples")

        self.n_raw_ch, self.input_shape, self.patch_size = \
            self.infra.determine_input_shape(self.all_samples)

        if self.comp_weights:
            self.pos_w, self.neg_w = self.compute_weights()

    def compute_weights(self) -> Tuple[float, float]:
        """Compute positive/negative pixel class weights from all samples."""
        pos = neg = 0
        for sample_name, location_name in self.all_samples:
            fi = discover_data_files_for_location(
                self.load_path, self.power_mode, sample_name, location_name,
                self.mask_type, data_regime=self.data_regime)
            _, mask = load_and_aggregate_location(
                fi, invert_mask=False, mask_only=True,
                data_regime=self.data_regime, min_mask_area=self.min_mask_area)
            pos += (mask > 0).sum()
            neg += (mask == 0).sum()
        pw = neg / (pos + 1e-7)
        print(f"Computed weights: pos_w={pw:.4f}, neg_w=1.0")
        return pw, 1.0

    def get_loss(self) -> nn.Module:
        return get_loss_function(self.loss_name, self.pos_w, self.neg_w,
                                 self.alpha, self.beta)

    def get_optimizer(
        self, model: nn.Module, fusion: Optional[FusionWeight] = None
    ) -> torch.optim.Optimizer:
        opt_cls = {'adam': Adam, 'adamw': AdamW,
                   'rmsprop': RMSprop, 'nadam': NAdam}.get(self.optimizer_name, Adam)
        if fusion is not None:
            params = [
                {'params': model.parameters(),  'lr': self.initial_lr},
                {'params': fusion.parameters(), 'lr': self.initial_lr / 10},
            ]
        else:
            params = [{'params': model.parameters(), 'lr': self.initial_lr}]
        return opt_cls(params, weight_decay=self.weight_decay)

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer, T_max: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build a learning-rate scheduler by name."""
        name = self.scheduler_name
        if name == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=T_max,
                                     eta_min=self.initial_lr * 0.001)
        if name == 'step':
            return StepLR(optimizer, step_size=max(1, T_max // 3), gamma=0.5)
        if name == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        if name == 'onecycle':
            return OneCycleLR(optimizer, max_lr=self.initial_lr,
                              epochs=T_max, steps_per_epoch=1)
        if name == 'none':
            return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        if name == 'cosine_warmup':
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1,     # 1e-3 → 1e-2 over 3 epochs
                end_factor=1.0, 
                total_iters=self.warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max-self.warmup_epochs,
                eta_min=1e-5           # end at 1e-5, not 1e-4
            )
            return  torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[self.warmup_epochs]
)

        raise ValueError(f"Unknown scheduler_name={name!r}. "
                         "Choose: cosine | step | plateau | onecycle | none")

    def save_fold_split(self, train_samples, val_samples, fold: int) -> None:
        fold_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'fold_splits', self.model_name)
        os.makedirs(fold_dir, exist_ok=True)
        np.savez(os.path.join(fold_dir, f'fold_{fold}.npz'),
                 train_samples=np.array(train_samples, dtype=object),
                 val_samples=np.array(val_samples, dtype=object))

    def save_checkpoint(
        self, model, optimizer, scheduler, epoch: int, fold: int,
        fusion: Optional[FusionWeight] = None, is_best: bool = False
    ) -> None:
        ckpt = dict(epoch=epoch, fold=fold,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict())
        if fusion is not None:
            ckpt['fusion_state_dict'] = fusion.state_dict()
        torch.save(ckpt, os.path.join(self.ckpt_dir, f'fold{fold}_ep{epoch:02d}.pt'))
        if is_best:
            torch.save(ckpt, os.path.join(self.ckpt_dir, f'fold{fold}_best.pt'))

    def run_kfold(
        self,
        model_fn: Callable[[], nn.Module],
        n_splits: int = 3,
        batch_size: int = 4,
        epochs: int = 20,
        patience: int = 10,
        random_state: int = 42,
        num_workers: int = 4,
        warmstart_ckpt_paths: Optional[dict] = None,
        head_freeze_epochs: int = 0,
        head_lr: float = 1e-3,
    ) -> Tuple[List, np.ndarray]:
        """Run stratified k-fold cross-validation with MLflow logging."""

        def get_class(r: float) -> int:
            return 0 if r < 0.1 else (1 if r < 0.15 else 2)

        all_samples = np.array(self.all_samples, dtype=object)
        classes = []
        for name, loc_name in all_samples:
            fi = discover_data_files_for_location(
                self.load_path, self.power_mode, name, loc_name,
                mask_type=self.mask_type, data_regime=self.data_regime)
            _, mask = load_and_aggregate_location(
                fi, invert_mask=self.invert_mask, mask_only=True,
                data_regime=self.data_regime,
                min_mask_area=self.min_mask_area)
            classes.append(get_class(mask.sum() / mask.size))

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)
        fold_metrics = []

        self.infra.start_run(self.versioned_name+"_"+str(self.mode)+"_"+str(self.patch_mode)+"_data"+str(self.dataset_n))
        self.infra.log_params(dict(
            model_name=self.model_name, mode=self.mode,
            combine=self.combine, n_splits=n_splits,
            batch_size=batch_size, epochs=epochs, patience=patience,
            initial_lr=self.initial_lr, loss=self.loss_name,
            alpha=self.alpha, beta=self.beta,
            mask_type=self.mask_type, data_regime=self.data_regime,
            n_samples=len(all_samples), patch_mode=self.patch_mode,
            patch_size=str(self.patch_size), dirs=str(self.dirs),
            max_locations=self.max_locations, min_mask_area=self.min_mask_area,
        ))

        try:
            for f, (tr_idx, va_idx) in enumerate(kf.split(all_samples, classes)):
                print(f"\n{'='*60}\nFold {f+1}/{n_splits}\n{'='*60}")
                train_samples = [all_samples[i] for i in tr_idx]
                val_samples   = [all_samples[i] for i in va_idx]
                print(f"TRAIN: {len(train_samples)} — "
                      f"{Counter(classes[i] for i in tr_idx)}")
                print(f"VAL:   {len(val_samples)} — "
                      f"{Counter(classes[i] for i in va_idx)}")
                self.save_fold_split(train_samples, val_samples, f + 1)

                loader_kw = dict(
                    xgb_model=self.xgb_global,
                    load_path=self.load_path, power_mode=self.power_mode,
                    patch_size=self.patch_size, batch_size=batch_size,
                    mask_type=self.mask_type, num_workers=num_workers,
                    ppt_phases=self.ppt_phases, ppt_amps=self.ppt_amps,
                    invert_mask=self.invert_mask, patch_mode=self.patch_mode,
                    data_regime=self.data_regime,
                    min_mask_area=self.min_mask_area)
                train_loader = create_hybrid_dataloader(
                    train_samples, augment=self.augment, shuffle=True,
                    apply_jitter=self.apply_jitter,
                    min_positive_ratio=self.min_positive_ratio, **loader_kw)
                val_loader = create_hybrid_dataloader(
                    val_samples, augment=False, shuffle=False,
                    apply_jitter=False, min_positive_ratio=0.0, **loader_kw)

                model = model_fn().to(self.device)

                _phase1_active, _phase1_opt = setup_warmstart(
                    model, warmstart_ckpt_paths, f + 1, self.device,
                    head_freeze_epochs, head_lr,
                    self.optimizer_name, self.weight_decay)

                if self.mode == 'parallel':
                    fusion = FusionWeight(init_logit=self.init_w).to(self.device)
                    if self.fusion_freeze_epochs > 0:
                        fusion.logit_w.requires_grad_(False)
                        print(f"  Fusion weight frozen for first "
                              f"{self.fusion_freeze_epochs} epochs")
                    optimizer = self.get_optimizer(model, fusion)
                elif _phase1_active:
                    fusion = None
                    optimizer = _phase1_opt
                else:
                    fusion = None
                    optimizer = self.get_optimizer(model)

                strategy  = make_strategy(self.mode, fusion=fusion, combine=self.combine)
                scheduler = self.get_scheduler(optimizer, epochs)
                criterion = self.get_loss()
                memory_cb = MemoryCleanupCallback()

                best_loss, no_improve = float('inf'), 0
                best_state, best_fusion_state = None, None

                for epoch in range(epochs):
                    _phase1_active, new_opt, new_sched = maybe_transition_phase2(
                        epoch, head_freeze_epochs, _phase1_active, model,
                        self.get_optimizer, epochs - head_freeze_epochs, self.initial_lr,
                        self.get_scheduler)
                    if new_opt is not None:
                        optimizer, scheduler = new_opt, new_sched

                    if (fusion is not None
                            and epoch == self.fusion_freeze_epochs
                            and not fusion.logit_w.requires_grad):
                        fusion.logit_w.requires_grad_(True)
                        print(f"  Epoch {epoch+1}: fusion weight unfrozen "
                              f"(lr={self.initial_lr/10:.2e})")

                    tr = train_epoch(model, train_loader, criterion, optimizer,
                                     strategy, self.device)
                    va = validate(model, val_loader, criterion, strategy, self.device)
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(va[0])
                    else:
                        scheduler.step()
                    lr = optimizer.param_groups[0]['lr']

                    step = f * epochs + epoch
                    metrics_step = {
                        f'fold{f+1}/tr_loss': tr[0], f'fold{f+1}/tr_iou': tr[4],
                        f'fold{f+1}/va_loss': va[0], f'fold{f+1}/va_iou': va[4],
                        f'fold{f+1}/va_prec': va[2], f'fold{f+1}/va_rec': va[3],
                        f'fold{f+1}/lr': lr,
                    }
                    if fusion is not None:
                        metrics_step[f'fold{f+1}/fusion_w'] = fusion.weight()
                    self.infra.log_metrics(metrics_step, step=step)

                    is_best = va[0] < best_loss
                    if is_best:
                        best_loss  = va[0]
                        best_state = {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}
                        if fusion is not None:
                            best_fusion_state = {k: v.cpu().clone()
                                                 for k, v in fusion.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1

                    if self.save_chckpnt is not None:
                        self.save_checkpoint(model, optimizer, scheduler,
                                             epoch, f + 1, fusion, is_best)
                    memory_cb.on_epoch_end(epoch, model, optimizer)

                    print(f"  Ep {epoch+1:3d}: tr_loss={tr[0]:.4f} "
                          f"va_loss={va[0]:.4f} prec={va[2]:.4f} "
                          f"rec={va[3]:.4f} iou={va[4]:.4f}")

                    if no_improve >= patience:
                        print(f"  Early stop at epoch {epoch+1}")
                        break

                if self.restore_best and best_state:
                    model.load_state_dict(best_state)
                    if fusion is not None and best_fusion_state is not None:
                        fusion.load_state_dict(best_fusion_state)

                print("  Threshold tuning...")
                best_thr = find_best_threshold(model, val_loader, strategy, self.device)

                va_final = validate(
                    model, val_loader, criterion, strategy, self.device, best_thr)
                if self.mode == 'parallel':
                    fw = fusion.weight() if fusion is not None else 0.5
                    mode_tag = f"combined(w_cnn={fw:.3f})"
                else:
                    mode_tag = 'CNN'
                print(f"  Fold {f+1} final [{mode_tag}] (thr={best_thr:.2f}): "
                      f"loss={va_final[0]:.4f} acc={va_final[1]:.4f} "
                      f"prec={va_final[2]:.4f} rec={va_final[3]:.4f} "
                      f"iou={va_final[4]:.4f}")

                self.infra.log_metrics({
                    f'fold{f+1}/best_thr':   best_thr,
                    f'fold{f+1}/final_iou':  va_final[4],
                    f'fold{f+1}/final_prec': va_final[2],
                    f'fold{f+1}/final_rec':  va_final[3],
                }, step=f)

                fold_metrics.append(va_final)

                del model, optimizer, train_loader, val_loader
                gc.collect()
                torch.cuda.empty_cache()

            avg = np.mean(fold_metrics, axis=0)
            self.infra.log_metrics(dict(avg_loss=avg[0], avg_acc=avg[1],
                                        avg_prec=avg[2], avg_rec=avg[3],
                                        avg_iou=avg[4]), step=n_splits)
            print(f"\n{'='*60}")
            print(f"Average: loss={avg[0]:.4f} acc={avg[1]:.4f} "
                  f"prec={avg[2]:.4f} rec={avg[3]:.4f} iou={avg[4]:.4f}")

        finally:
            self.infra.end_run()

        return fold_metrics, avg
