"""Training manager for clean autoencoder reconstruction + full-image UNet."""

from __future__ import annotations

import gc
import os
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from torch.optim import Adam, AdamW, NAdam, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, ReduceLROnPlateau, SequentialLR, StepLR
from torch.utils.data import DataLoader

from ...core.data_discovery_V2 import discover_data_files_for_location
from ...core.data_loading_V2 import load_and_aggregate_location
from ...core.losses import get_loss_function
from ..components.hybrid_models import SimplerUNet
from ..components.infrastructure_V2 import TrainingInfrastructure
from .gan_unet_data import CleanPatchDataset, DifferenceFullImageDataset, H5DataConfig
from .gan_unet_models import ConvAutoencoder


Sample = Tuple[str, str]
METRIC_NAMES = ("loss", "acc", "prec", "rec", "f1", "iou")


def _compute_metrics(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> List[float]:
    pred = (prob > threshold).float()
    target = target.float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    acc = (pred == target).float().mean().item()
    prec = (tp / (tp + fp + 1e-6)).item()
    rec = (tp / (tp + fn + 1e-6)).item()
    f1 = (2.0 * tp / (2.0 * tp + fp + fn + 1e-6)).item()
    iou = (tp / (tp + fp + fn + 1e-6)).item()
    return [acc, prec, rec, f1, iou]


def _full_image_collate(batch):
    """Collate full images; batch size >1 requires identical H/W."""
    diffs, masks = zip(*batch)
    shapes = {tuple(x.shape) for x in diffs}
    if len(shapes) > 1:
        raise ValueError("Full-image batches need equal shapes. Use batch_size=1 for mixed-size locations.")
    return torch.stack(diffs, dim=0), torch.stack(masks, dim=0)


class GANUNetTrainingManager:
    """Fold trainer for autoencoder reconstruction followed by UNet segmentation."""

    def __init__(
        self,
        model_name: str,
        sys: str,
        power_mode: str = "4kw_both",
        subfolder_name: str = "Taris/Data_ML_V1_h5",
        ae_patch_size: Tuple[int, int] = (128, 128),
        ae_max_positive_ratio: float = 0.01,
        ae_base_channels: int = 32,
        ae_latent_channels: int = 128,
        ae_noise_std: float = 0.0,
        unet_dropout_rate: float = 0.0,
        mask_type: str = "alternative",
        dirs: Optional[List[int]] = None,
        ppt_phases="all",
        ppt_amps: int = 6,
        invert_mask: bool = False,
        data_regime: str = "postprocessed",
        min_mask_area: int = 0,
        max_locations: Optional[int] = None,
        mlflow_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlflow_experiments/mlflow.db"),
        loss_name: str = "soft_iou",
        alpha: float = 0.8,
        beta: float = 0.3,
        pos_w: float = 1.0,
        neg_w: float = 1.0,
        optimizer_name: str = "adamw",
        scheduler_name: str = "cosine",
        ae_scheduler_name: Optional[str] = None,
        unet_scheduler_name: Optional[str] = None,
        ae_loss_name: str = "l1_mse",
        warmup_epochs: int = 3,
        lr: float = 1e-3,
        ae_lr: Optional[float] = None,
        weight_decay: float = 1e-4,
        restore_best: bool = True,
    ):
        self.model_name = model_name
        self.power_mode = power_mode
        self.subfolder_name = subfolder_name
        self.ae_patch_size = tuple(ae_patch_size)
        self.ae_max_positive_ratio = ae_max_positive_ratio
        self.ae_base_channels = ae_base_channels
        self.ae_latent_channels = ae_latent_channels
        self.ae_noise_std = float(ae_noise_std)
        self.unet_dropout_rate = unet_dropout_rate
        self.mask_type = mask_type
        self.dirs = dirs if dirs is not None else []
        self.ppt_phases = ppt_phases
        self.ppt_amps = ppt_amps
        self.invert_mask = invert_mask
        self.data_regime = data_regime
        self.min_mask_area = min_mask_area
        self.max_locations = max_locations
        self.loss_name = loss_name
        self.alpha = alpha
        self.beta = beta
        self.pos_w = pos_w
        self.neg_w = neg_w
        self.optimizer_name = optimizer_name.lower()
        self.scheduler_name = scheduler_name.lower()
        self.ae_scheduler_name = (ae_scheduler_name or scheduler_name).lower()
        self.unet_scheduler_name = (unet_scheduler_name or scheduler_name).lower()
        self.ae_loss_name = ae_loss_name.lower()
        self.warmup_epochs = int(warmup_epochs)
        self.lr = lr
        self.ae_lr = ae_lr if ae_lr is not None else lr
        self.weight_decay = weight_decay
        self.restore_best = restore_best
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.infra = TrainingInfrastructure(
            model_name=model_name,
            sys=sys,
            mode="nn_only",
            subfolder_name=subfolder_name,
            power_mode=power_mode,
            patch_size=list(ae_patch_size),
            patch_mode="patches",
            mask_type=mask_type,
            dirs=self.dirs,
            data_regime=data_regime,
            ppt_phases=ppt_phases,
            ppt_amps=ppt_amps,
            mlflow_uri=mlflow_uri,
            base_path=os.path.dirname(os.path.abspath(__file__)),
            max_locations=max_locations,
        )
        self.infra.setup_directories()
        self.infra.setup_mlflow()

        self.versioned_name = self.infra.versioned_name
        self.ckpt_dir = self.infra.ckpt_dir
        self.load_path = self.infra.load_path
        self.all_samples = self.infra.discover_samples()
        self.n_raw_ch, self.input_shape, _ = self.infra.determine_input_shape(self.all_samples)
        self.data_config = H5DataConfig(
            load_path=self.load_path,
            power_mode=self.power_mode,
            mask_type=self.mask_type,
            ppt_phases=self.ppt_phases,
            ppt_amps=self.ppt_amps,
            invert_mask=self.invert_mask,
            data_regime=self.data_regime,
            min_mask_area=self.min_mask_area,
        )
        print(f"Device: {self.device} | samples={len(self.all_samples)} | channels={self.n_raw_ch}")

    def _optimizer(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
        opt_cls = {"adam": Adam, "adamw": AdamW, "rmsprop": RMSprop, "nadam": NAdam}.get(
            self.optimizer_name, AdamW
        )
        return opt_cls(model.parameters(), lr=lr, weight_decay=self.weight_decay)

    def _scheduler(self, optimizer: torch.optim.Optimizer, epochs: int, name: Optional[str] = None):
        name = (name or self.scheduler_name).lower()
        if name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
        if name == "cosine_warmup":
            warmup_epochs = min(max(1, self.warmup_epochs), max(1, epochs))
            if warmup_epochs >= epochs:
                return LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, epochs))
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
            return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        if name == "step":
            return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
        if name == "plateau":
            return ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        if name == "none":
            return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        raise ValueError(f"Unknown scheduler_name={name!r}. Choose: cosine | cosine_warmup | step | plateau | none")

    def _ae_loss(self, recon: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1 = F.l1_loss(recon, target)
        mse = F.mse_loss(recon, target)
        if self.ae_loss_name == "l1_mse":
            loss = l1 + 0.25 * mse
        elif self.ae_loss_name == "l1":
            loss = l1
        elif self.ae_loss_name == "mse":
            loss = mse
        elif self.ae_loss_name == "smooth_l1":
            loss = F.smooth_l1_loss(recon, target)
        else:
            raise ValueError(f"Unknown ae_loss_name={self.ae_loss_name!r}. Choose: l1_mse | l1 | mse | smooth_l1")
        return loss, {"l1": float(l1.detach().item()), "mse": float(mse.detach().item())}

    def _mask_ratio(self, sample: Sample) -> float:
        sample_name, location_name = sample
        fi = discover_data_files_for_location(
            self.load_path,
            self.power_mode,
            sample_name,
            location_name,
            mask_type=self.mask_type,
            data_regime=self.data_regime,
        )
        _, mask = load_and_aggregate_location(
            fi,
            invert_mask=self.invert_mask,
            mask_only=True,
            data_regime=self.data_regime,
            min_mask_area=self.min_mask_area,
        )
        return float((mask > 0).mean())

    @staticmethod
    def _ratio_class(ratio: float) -> int:
        return 0 if ratio < 0.1 else (1 if ratio < 0.15 else 2)

    def _splitter(self, samples: np.ndarray, classes: Sequence[int], n_splits: int, random_state: int):
        counts = Counter(classes)
        if counts and min(counts.values()) >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(samples, classes)
        print(f"Stratified split not possible for class counts {counts}; using KFold.")
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(samples)

    def save_fold_split(
        self,
        train_samples: Sequence[Sample],
        val_samples: Sequence[Sample],
        fold: int,
        split_dir: Optional[str] = None,
    ) -> str:
        fold_dir = split_dir or os.path.join(self.ckpt_dir, "fold_splits")
        os.makedirs(fold_dir, exist_ok=True)
        path = os.path.join(fold_dir, f"fold_{fold}.npz")
        np.savez(
            path,
            train_samples=np.array(train_samples, dtype=object),
            val_samples=np.array(val_samples, dtype=object),
        )
        self.infra.log_artifact(path, artifact_path="fold_splits")
        return path

    @staticmethod
    def load_fold_split(split_dir: str, fold: int) -> Tuple[List[Sample], List[Sample]]:
        path = os.path.join(split_dir, f"fold_{fold}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fold split file: {path}")
        data = np.load(path, allow_pickle=True)
        train_samples = [tuple(x) for x in data["train_samples"]]
        val_samples = [tuple(x) for x in data["val_samples"]]
        return train_samples, val_samples

    def load_autoencoder_checkpoint(self, checkpoint_dir: str, fold: int) -> nn.Module:
        path = os.path.join(checkpoint_dir, f"fold{fold}_autoencoder_best.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing autoencoder checkpoint: {path}")
        model = ConvAutoencoder(
            self.n_raw_ch,
            base_channels=self.ae_base_channels,
            latent_channels=self.ae_latent_channels,
        ).to(self.device)
        ckpt = torch.load(path, map_location=self.device)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Autoencoder checkpoint is incompatible with the current bottleneck-only AE. "
                "Retrain the AE, or use a checkpoint created after skip connections were removed."
            ) from exc
        model.eval()
        return model

    def train_autoencoder(
        self,
        train_samples: Sequence[Sample],
        val_samples: Sequence[Sample],
        fold: int,
        epochs: int,
        batch_size: int,
        num_workers: int,
        patience: int,
        augment: bool,
        rot_angle: float,
        log_prefix: Optional[str] = None,
    ) -> Tuple[nn.Module, List[Dict[str, float]]]:
        train_ds = CleanPatchDataset(
            train_samples,
            self.data_config,
            patch_size=self.ae_patch_size,
            max_positive_ratio=self.ae_max_positive_ratio,
            augment=augment,
            rot_angle=rot_angle,
            noise_std=self.ae_noise_std,
        )
        try:
            val_ds = CleanPatchDataset(
                val_samples,
                self.data_config,
                patch_size=self.ae_patch_size,
                max_positive_ratio=self.ae_max_positive_ratio,
                augment=False,
                noise_std=0.0,
            )
        except ValueError:
            val_ds = None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            if val_ds is not None
            else None
        )
        model = ConvAutoencoder(
            self.n_raw_ch,
            base_channels=self.ae_base_channels,
            latent_channels=self.ae_latent_channels,
        ).to(self.device)
        optimizer = self._optimizer(model, self.ae_lr)
        scheduler = self._scheduler(optimizer, epochs, self.ae_scheduler_name)
        best_loss = float("inf")
        best_state = None
        no_improve = 0
        history = []

        print(f"  AE clean patches: train={len(train_ds)} val={len(val_ds) if val_ds else 0}")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_l1 = 0.0
            train_mse = 0.0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    ae_input, target = batch
                    ae_input = ae_input.to(self.device)
                    target = target.to(self.device)
                else:
                    ae_input = batch.to(self.device)
                    target = ae_input
                optimizer.zero_grad()
                recon = model(ae_input)
                loss, components = self._ae_loss(recon, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                train_l1 += components["l1"]
                train_mse += components["mse"]
            train_loss /= max(1, len(train_loader))
            train_l1 /= max(1, len(train_loader))
            train_mse /= max(1, len(train_loader))

            val_loss = train_loss
            val_l1 = train_l1
            val_mse = train_mse
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                val_l1 = 0.0
                val_mse = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        if isinstance(batch, (tuple, list)):
                            ae_input, target = batch
                            ae_input = ae_input.to(self.device)
                            target = target.to(self.device)
                        else:
                            ae_input = batch.to(self.device)
                            target = ae_input
                        recon = model(ae_input)
                        loss, components = self._ae_loss(recon, target)
                        val_loss += loss.item()
                        val_l1 += components["l1"]
                        val_mse += components["mse"]
                val_loss /= max(1, len(val_loader))
                val_l1 /= max(1, len(val_loader))
                val_mse /= max(1, len(val_loader))

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                "train_loss": train_loss,
                "train_l1": train_l1,
                "train_mse": train_mse,
                "val_loss": val_loss,
                "val_l1": val_l1,
                "val_mse": val_mse,
                "lr": lr,
            }
            history.append(epoch_metrics)
            if log_prefix is not None:
                self.infra.log_metrics({f"{log_prefix}/{k}": v for k, v in epoch_metrics.items()}, step=epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                ckpt_path = os.path.join(self.ckpt_dir, f"fold{fold}_autoencoder_best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "fold": fold,
                        "model_state_dict": best_state,
                        "val_loss": best_loss,
                        "ae_loss_name": self.ae_loss_name,
                        "ae_patch_size": self.ae_patch_size,
                        "ae_noise_std": self.ae_noise_std,
                        "ae_skip_connections": False,
                        "n_raw_ch": self.n_raw_ch,
                    },
                    ckpt_path,
                )
            else:
                no_improve += 1

            print(
                f"    AE Ep {epoch + 1:3d}: train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} lr={lr:.2e}"
            )
            if no_improve >= patience:
                print(f"    AE early stop at epoch {epoch + 1}")
                break

        if self.restore_best and best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    def train_unet(
        self,
        autoencoder: nn.Module,
        train_samples: Sequence[Sample],
        val_samples: Sequence[Sample],
        fold: int,
        epochs: int,
        batch_size: int,
        num_workers: int,
        patience: int,
        model_fn: Optional[Callable[[], nn.Module]] = None,
        model_label: str = "unet",
        log_prefix: Optional[str] = None,
    ) -> Dict[str, object]:
        train_ds = DifferenceFullImageDataset(
            train_samples, self.data_config, autoencoder, self.ae_patch_size, self.device
        )
        val_ds = DifferenceFullImageDataset(
            val_samples, self.data_config, autoencoder, self.ae_patch_size, self.device
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_full_image_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_full_image_collate,
        )
        model = (model_fn() if model_fn is not None else SimplerUNet(self.n_raw_ch, self.unet_dropout_rate)).to(self.device)
        optimizer = self._optimizer(model, self.lr)
        scheduler = self._scheduler(optimizer, epochs, self.unet_scheduler_name)
        criterion = get_loss_function(self.loss_name, self.pos_w, self.neg_w, self.alpha, self.beta)
        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            train_metrics = self._run_unet_epoch(model, train_loader, criterion, optimizer)
            val_metrics = self._validate_unet(model, val_loader, criterion)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics[0])
            else:
                scheduler.step()

            if val_metrics[0] < best_loss:
                best_loss = val_metrics[0]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                ckpt_path = os.path.join(self.ckpt_dir, f"fold{fold}_{model_label}_best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "fold": fold,
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_loss": best_loss,
                        "loss_name": self.loss_name,
                        "model_label": model_label,
                        "n_raw_ch": self.n_raw_ch,
                    },
                    ckpt_path,
                )
                if model_label != "unet":
                    torch.save(
                        {
                            "epoch": epoch,
                            "fold": fold,
                            "model_state_dict": best_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "val_loss": best_loss,
                            "loss_name": self.loss_name,
                            "model_label": model_label,
                            "n_raw_ch": self.n_raw_ch,
                        },
                        os.path.join(self.ckpt_dir, f"fold{fold}_unet_best.pt"),
                    )
            else:
                no_improve += 1

            lr = optimizer.param_groups[0]["lr"]
            if log_prefix is not None:
                metrics = {}
                for prefix, values in (("train", train_metrics), ("val", val_metrics)):
                    metrics.update({f"{log_prefix}/{prefix}_{name}": value for name, value in zip(METRIC_NAMES, values)})
                metrics[f"{log_prefix}/lr"] = lr
                self.infra.log_metrics(metrics, step=epoch)

            print(
                f"    UNet Ep {epoch + 1:3d}: tr_loss={train_metrics[0]:.4f} "
                f"va_loss={val_metrics[0]:.4f} va_prec={val_metrics[2]:.4f} "
                f"va_rec={val_metrics[3]:.4f} va_f1={val_metrics[4]:.4f} "
                f"va_iou={val_metrics[5]:.4f} lr={lr:.2e}"
            )
            if no_improve >= patience:
                print(f"    UNet early stop at epoch {epoch + 1}")
                break

        if self.restore_best and best_state is not None:
            model.load_state_dict(best_state)
        best_thr = self._find_best_threshold(model, val_loader)
        train_final_metrics = self._validate_unet(model, train_loader, criterion, threshold=best_thr)
        final_metrics = self._validate_unet(model, val_loader, criterion, threshold=best_thr)
        if log_prefix is not None:
            final_log = {f"{log_prefix}/best_thr": best_thr}
            for prefix, values in (("final_train", train_final_metrics), ("final_val", final_metrics)):
                final_log.update({f"{log_prefix}/{prefix}_{name}": value for name, value in zip(METRIC_NAMES, values)})
            self.infra.log_metrics(final_log, step=epochs)
        print(
            f"    UNet final fold {fold} thr={best_thr:.2f}: loss={final_metrics[0]:.4f} "
            f"acc={final_metrics[1]:.4f} prec={final_metrics[2]:.4f} "
            f"rec={final_metrics[3]:.4f} f1={final_metrics[4]:.4f} iou={final_metrics[5]:.4f}"
        )
        return {"train": train_final_metrics, "val": final_metrics, "best_thr": best_thr, "model": model}

    def _run_unet_epoch(self, model, loader, criterion, optimizer) -> List[float]:
        model.train()
        totals = [0.0] * 6
        for diff, mask in loader:
            diff = diff.to(self.device)
            mask = mask.to(self.device)
            optimizer.zero_grad()
            logit = model(diff).squeeze(1)
            loss = criterion(logit, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                prob = torch.sigmoid(logit)
                acc, prec, rec, f1, iou = _compute_metrics(prob, mask)
            for i, value in enumerate([loss.item(), acc, prec, rec, f1, iou]):
                totals[i] += value
        return [v / max(1, len(loader)) for v in totals]

    @torch.no_grad()
    def _validate_unet(self, model, loader, criterion, threshold: float = 0.5) -> List[float]:
        model.eval()
        totals = [0.0] * 6
        for diff, mask in loader:
            diff = diff.to(self.device)
            mask = mask.to(self.device)
            logit = model(diff).squeeze(1)
            loss = criterion(logit, mask)
            prob = torch.sigmoid(logit)
            acc, prec, rec, f1, iou = _compute_metrics(prob, mask, threshold)
            for i, value in enumerate([loss.item(), acc, prec, rec, f1, iou]):
                totals[i] += value
        return [v / max(1, len(loader)) for v in totals]

    @torch.no_grad()
    def _find_best_threshold(self, model, loader) -> float:
        best_thr, best_iou = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            metrics = self._validate_unet(model, loader, nn.BCEWithLogitsLoss(), threshold=float(thr))
            if metrics[5] > best_iou:
                best_thr, best_iou = float(thr), metrics[5]
        return best_thr

    def _common_params(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "power_mode": self.power_mode,
            "subfolder_name": self.subfolder_name,
            "mask_type": self.mask_type,
            "data_regime": self.data_regime,
            "dirs": str(self.dirs),
            "ppt_phases": self.ppt_phases,
            "ppt_amps": self.ppt_amps,
            "invert_mask": self.invert_mask,
            "min_mask_area": self.min_mask_area,
            "max_locations": self.max_locations,
            "n_samples": len(self.all_samples),
            "n_raw_ch": self.n_raw_ch,
            "optimizer_name": self.optimizer_name,
            "weight_decay": self.weight_decay,
            "restore_best": self.restore_best,
            "ckpt_dir": self.ckpt_dir,
        }

    def _split_samples(self, n_splits: int, random_state: int):
        samples = np.array(self.all_samples, dtype=object)
        ratios = [self._mask_ratio(tuple(sample)) for sample in samples]
        classes = [self._ratio_class(ratio) for ratio in ratios]
        return samples, classes, self._splitter(samples, classes, n_splits, random_state)

    def run_autoencoder_kfold(
        self,
        n_splits: int = 3,
        ae_epochs: int = 20,
        ae_batch_size: int = 16,
        ae_patience: int = 8,
        num_workers: int = 0,
        random_state: int = 42,
        augment_ae: bool = True,
        ae_rot_angle: float = 0.0,
    ) -> List[Dict[str, object]]:
        samples, classes, splits = self._split_samples(n_splits, random_state)
        results = []

        self.infra.start_run(f"{self.versioned_name}_autoencoder")
        self.infra.log_params(
            {
                **self._common_params(),
                "stage": "autoencoder",
                "n_splits": n_splits,
                "ae_epochs": ae_epochs,
                "ae_batch_size": ae_batch_size,
                "ae_patience": ae_patience,
                "ae_lr": self.ae_lr,
                "ae_loss_name": self.ae_loss_name,
                "ae_scheduler_name": self.ae_scheduler_name,
                "ae_patch_size": str(self.ae_patch_size),
                "ae_max_positive_ratio": self.ae_max_positive_ratio,
                "ae_base_channels": self.ae_base_channels,
                "ae_latent_channels": self.ae_latent_channels,
                "ae_noise_std": self.ae_noise_std,
                "ae_skip_connections": False,
                "augment_ae": augment_ae,
                "ae_rot_angle": ae_rot_angle,
                "random_state": random_state,
                "num_workers": num_workers,
            }
        )

        try:
            for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
                print(f"\n{'=' * 60}\nAE Fold {fold}/{n_splits}\n{'=' * 60}")
                train_samples = [tuple(samples[i]) for i in tr_idx]
                val_samples = [tuple(samples[i]) for i in va_idx]
                print(f"TRAIN: {len(train_samples)} - {Counter(classes[i] for i in tr_idx)}")
                print(f"VAL:   {len(val_samples)} - {Counter(classes[i] for i in va_idx)}")
                split_path = self.save_fold_split(train_samples, val_samples, fold)

                ae_model, ae_history = self.train_autoencoder(
                    train_samples,
                    val_samples,
                    fold=fold,
                    epochs=ae_epochs,
                    batch_size=ae_batch_size,
                    num_workers=num_workers,
                    patience=ae_patience,
                    augment=augment_ae,
                    rot_angle=ae_rot_angle,
                    log_prefix=f"fold{fold}/ae",
                )
                best_loss = min((m["val_loss"] for m in ae_history), default=np.nan)
                ckpt_path = os.path.join(self.ckpt_dir, f"fold{fold}_autoencoder_best.pt")
                self.infra.log_metrics(
                    {
                        f"fold{fold}/ae_best_loss": best_loss,
                        f"fold{fold}/ae_train_samples": len(train_samples),
                        f"fold{fold}/ae_val_samples": len(val_samples),
                    },
                    step=fold,
                )
                self.infra.log_artifact(ckpt_path, artifact_path="checkpoints")
                results.append(
                    {
                        "fold": fold,
                        "best_loss": best_loss,
                        "checkpoint": ckpt_path,
                        "split": split_path,
                        "train_samples": len(train_samples),
                        "val_samples": len(val_samples),
                    }
                )

                del ae_model
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            self.infra.end_run()

        return results

    def run_unet_kfold(
        self,
        ae_checkpoint_dir: str,
        n_splits: int = 3,
        unet_epochs: int = 20,
        unet_batch_size: int = 1,
        unet_patience: int = 10,
        num_workers: int = 0,
        model_fn: Optional[Callable[[], nn.Module]] = None,
        model_label: str = "unet",
        split_dir: Optional[str] = None,
    ) -> Tuple[List[List[float]], np.ndarray]:
        split_dir = split_dir or os.path.join(ae_checkpoint_dir, "fold_splits")
        fold_metrics = []

        self.infra.start_run(f"{self.versioned_name}_{model_label}_from_ae")
        self.infra.log_params(
            {
                **self._common_params(),
                "stage": "unet",
                "ae_checkpoint_dir": ae_checkpoint_dir,
                "split_dir": split_dir,
                "n_splits": n_splits,
                "unet_epochs": unet_epochs,
                "unet_batch_size": unet_batch_size,
                "unet_patience": unet_patience,
                "unet_lr": self.lr,
                "unet_dropout_rate": self.unet_dropout_rate,
                "unet_model": model_label,
                "unet_loss_name": self.loss_name,
                "alpha": self.alpha,
                "beta": self.beta,
                "pos_w": self.pos_w,
                "neg_w": self.neg_w,
                "unet_scheduler_name": self.unet_scheduler_name,
                "warmup_epochs": self.warmup_epochs,
                "num_workers": num_workers,
            }
        )

        try:
            for fold in range(1, n_splits + 1):
                print(f"\n{'=' * 60}\nUNet Fold {fold}/{n_splits} ({model_label})\n{'=' * 60}")
                train_samples, val_samples = self.load_fold_split(split_dir, fold)
                print(f"TRAIN: {len(train_samples)}")
                print(f"VAL:   {len(val_samples)}")
                ae_model = self.load_autoencoder_checkpoint(ae_checkpoint_dir, fold)
                result = self.train_unet(
                    ae_model,
                    train_samples,
                    val_samples,
                    fold=fold,
                    epochs=unet_epochs,
                    batch_size=unet_batch_size,
                    num_workers=num_workers,
                    patience=unet_patience,
                    model_fn=model_fn,
                    model_label=model_label,
                    log_prefix=f"fold{fold}/unet",
                )
                val_metrics = result["val"]
                fold_metrics.append(val_metrics)
                fold_log = {f"fold{fold}/unet_{name}": value for name, value in zip(METRIC_NAMES, val_metrics)}
                fold_log[f"fold{fold}/unet_best_thr"] = result["best_thr"]
                self.infra.log_metrics(fold_log, step=fold)
                ckpt_path = os.path.join(self.ckpt_dir, f"fold{fold}_{model_label}_best.pt")
                if os.path.exists(ckpt_path):
                    self.infra.log_artifact(ckpt_path, artifact_path="checkpoints")

                del ae_model, result
                gc.collect()
                torch.cuda.empty_cache()

            avg = np.mean(fold_metrics, axis=0)
            self.infra.log_metrics({f"avg_{name}": value for name, value in zip(METRIC_NAMES, avg)}, step=n_splits)
        finally:
            self.infra.end_run()

        return fold_metrics, avg

    def run_kfold(
        self,
        n_splits: int = 3,
        ae_epochs: int = 20,
        unet_epochs: int = 20,
        ae_batch_size: int = 16,
        unet_batch_size: int = 1,
        ae_patience: int = 8,
        unet_patience: int = 10,
        num_workers: int = 0,
        random_state: int = 42,
        augment_ae: bool = True,
        ae_rot_angle: float = 0.0,
        model_fn: Optional[Callable[[], nn.Module]] = None,
    ) -> Tuple[List[List[float]], np.ndarray]:
        self.run_autoencoder_kfold(
            n_splits=n_splits,
            ae_epochs=ae_epochs,
            ae_batch_size=ae_batch_size,
            ae_patience=ae_patience,
            num_workers=num_workers,
            random_state=random_state,
            augment_ae=augment_ae,
            ae_rot_angle=ae_rot_angle,
        )
        return self.run_unet_kfold(
            ae_checkpoint_dir=self.ckpt_dir,
            n_splits=n_splits,
            unet_epochs=unet_epochs,
            unet_batch_size=unet_batch_size,
            unet_patience=unet_patience,
            num_workers=num_workers,
            model_fn=model_fn,
            model_label="unet",
        )
