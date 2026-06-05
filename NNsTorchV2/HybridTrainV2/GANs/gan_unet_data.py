"""Data and reconstruction utilities for the autoencoder + UNet pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode

from ...core.data_discovery_V2 import discover_data_files_for_location
from ...core.data_loading_V2 import load_and_aggregate_location

Sample = Tuple[str, str]


@dataclass(frozen=True)
class H5DataConfig:
    load_path: str
    power_mode: str
    mask_type: str = "alternative"
    ppt_phases: Union[str, int] = "all"
    ppt_amps: int = 6
    invert_mask: bool = False
    data_regime: str = "postprocessed"
    min_mask_area: int = 0


_spatial_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
])


def load_location(sample: Sample, config: H5DataConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Load one V2 H5 location as `(data, mask)`."""
    sample_name, location_name = sample
    fi = discover_data_files_for_location(
        config.load_path,
        config.power_mode,
        sample_name,
        location_name,
        mask_type=config.mask_type,
        data_regime=config.data_regime,
    )
    if fi is None:
        raise FileNotFoundError(f"No H5 data found for sample={sample_name!r}, location={location_name!r}")
    data, mask = load_and_aggregate_location(
        fi,
        ppt_phases=config.ppt_phases,
        ppt_amps=config.ppt_amps,
        invert_mask=config.invert_mask,
        data_regime=config.data_regime,
        min_mask_area=config.min_mask_area,
    )
    return data.astype(np.float32), mask.astype(np.float32)


def iter_patch_slices(image_shape: Tuple[int, int], patch_size: Tuple[int, int]) -> Iterable[Tuple[slice, slice]]:
    """Yield deterministic full-coverage patch slices for reconstruction/stitching."""
    h, w = image_shape
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    if patch_h > h or patch_w > w:
        raise ValueError(f"patch_size={patch_size} is larger than image shape {(h, w)}")

    n_h = max(1, int(np.ceil(h / patch_h)))
    n_w = max(1, int(np.ceil(w / patch_w)))
    y_positions = np.linspace(0, h - patch_h, n_h)
    x_positions = np.linspace(0, w - patch_w, n_w)

    for y in y_positions:
        for x in x_positions:
            top = int(round(y))
            left = int(round(x))
            yield slice(top, top + patch_h), slice(left, left + patch_w)


class CleanPatchDataset(Dataset):
    """Patches with defect-positive ratio below `max_positive_ratio` for AE training."""

    def __init__(
        self,
        samples: Sequence[Sample],
        data_config: H5DataConfig,
        patch_size: Tuple[int, int] = (128, 128),
        max_positive_ratio: float = 0.01,
        augment: bool = True,
        rot_angle: float = 0.0,
        noise_std: float = 0.0,
    ):
        self.patch_size = tuple(patch_size)
        self.max_positive_ratio = float(max_positive_ratio)
        self.transform = _spatial_transform if augment else None
        self.max_rot_angle = float(rot_angle) if augment else 0.0
        self.noise_std = float(noise_std)
        self.patches: List[np.ndarray] = []
        self.patch_ratios: List[float] = []

        for sample in samples:
            data, mask = load_location(sample, data_config)
            for y_slice, x_slice in iter_patch_slices(mask.shape, self.patch_size):
                patch_mask = mask[y_slice, x_slice]
                pos_ratio = float((patch_mask > 0).mean())
                if pos_ratio < self.max_positive_ratio:
                    self.patches.append(data[y_slice, x_slice, :].copy())
                    self.patch_ratios.append(pos_ratio)

        if not self.patches:
            raise ValueError(
                "No clean patches found. Increase max_positive_ratio, reduce patch_size, "
                "or include more samples/locations."
            )

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        data = self.patches[idx]
        data_t = torch.from_numpy(data.transpose(2, 0, 1).copy()).float()
        if self.transform:
            data_t = self.transform(data_t)
            if self.max_rot_angle > 0:
                angle = (torch.rand(1).item() * 2 - 1) * self.max_rot_angle
                data_t = TF.rotate(data_t, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        if self.noise_std > 0:
            noisy_t = data_t + self.noise_std * torch.randn_like(data_t)
            return noisy_t, data_t
        return data_t


@torch.no_grad()
def reconstruct_full_image(
    autoencoder: torch.nn.Module,
    data: np.ndarray,
    patch_size: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """Reconstruct a full `(H, W, C)` location by averaging reconstructed patches."""
    autoencoder.eval()
    h, w, c = data.shape
    recon_sum = np.zeros((h, w, c), dtype=np.float32)
    recon_count = np.zeros((h, w, 1), dtype=np.float32)

    for y_slice, x_slice in iter_patch_slices((h, w), patch_size):
        patch = data[y_slice, x_slice, :]
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(device)
        recon_t = autoencoder(patch_t).squeeze(0).cpu()
        recon_patch = recon_t.numpy().transpose(1, 2, 0).astype(np.float32)
        recon_sum[y_slice, x_slice, :] += recon_patch
        recon_count[y_slice, x_slice, :] += 1.0

    if np.any(recon_count == 0):
        raise RuntimeError("Patch stitching left uncovered pixels; check patch grid logic.")
    return recon_sum / recon_count


class DifferenceFullImageDataset(Dataset):
    """Full-location UNet dataset using `abs(original - autoencoder(original))`."""

    def __init__(
        self,
        samples: Sequence[Sample],
        data_config: H5DataConfig,
        autoencoder: torch.nn.Module,
        patch_size: Tuple[int, int],
        device: torch.device,
        cache_in_memory: bool = True,
    ):
        self.samples = list(samples)
        self.data_config = data_config
        self.autoencoder = autoencoder
        self.patch_size = tuple(patch_size)
        self.device = device
        self.cache = []

        if cache_in_memory:
            for sample in self.samples:
                self.cache.append(self._build_item(sample))
            self.autoencoder = None

    def _build_item(self, sample: Sample) -> Tuple[torch.Tensor, torch.Tensor]:
        data, mask = load_location(sample, self.data_config)
        reconstruction = reconstruct_full_image(self.autoencoder, data, self.patch_size, self.device)
        diff = np.abs(data - reconstruction).astype(np.float32)
        diff_t = torch.from_numpy(diff.transpose(2, 0, 1).copy()).float()
        mask_t = torch.from_numpy(mask.copy()).float()
        return diff_t, mask_t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache:
            return self.cache[idx]
        return self._build_item(self.samples[idx])
