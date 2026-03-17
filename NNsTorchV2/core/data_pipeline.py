"""
PyTorch data pipeline for patch-based training.
"""

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .full_img_padding import extract_full_padding_patch
from .data_discovery import discover_data_files_for_location
from .data_loading import load_and_aggregate_location, calculate_total_channels
from .patch_extraction import extract_patches_from_image
#import torchvision.transforms as T
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    #T.RandomRotation(degrees=30),
    # T.RandomAffine(
    #     degrees=0,
    #     translate=(0.05, 0.05),
    #     scale=(0.95, 1.05)
    # ),
])
mask_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    #T.RandomRotation(degrees=30, interpolation=InterpolationMode.NEAREST),
    #T.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05), interpolation='nearest')
])

class PatchDataset(Dataset):
    """PyTorch Dataset for patch-based training."""

    def __init__(
        self,
        sample_indices: List[Tuple[str, int]],
        load_path: str,
        power_mode: str,
        patch_size: tuple = [128, 128],
        augment: bool = True,
        mask_type: str = 'normal',
        rotate_img: bool = True,
        ppt_phases: Union[str, int] = 'all',
        ppt_amps: int = 6,
        invert_mask: bool = False,
        apply_jitter: bool = True,
        min_positive_ratio: float = 0.05,
        patch_mode: str = "patches",
        data_regime: str = 'postprocessed'
    ):
        self.sample_indices = sample_indices
        self.load_path = load_path
        self.power_mode = power_mode
        self.patch_size = patch_size
        self.augment = augment
        self.mask_type = mask_type
        self.rotate_img = rotate_img
        self.ppt_phases = ppt_phases
        self.ppt_amps = ppt_amps
        self.invert_mask = invert_mask
        self.apply_jitter = apply_jitter
        self.transform = train_transform if augment else None
        self.mask_transform = mask_transform if augment else None
        self.min_positive_ratio = min_positive_ratio
        self.data_regime = data_regime
        # Pre-extract all patches
        self.patches = []

        for sample_name, location_idx in sample_indices:
            data, mask = self._load_sample(sample_name, location_idx)
            if patch_mode=="patches":
                patches = extract_patches_from_image(data, mask, patch_size, apply_jitter=apply_jitter, min_positive_ratio=min_positive_ratio)
                self.patches.extend(patches)
            elif patch_mode=="full":
                self.patches.append((data, mask))
            elif patch_mode == "full_padding":
                # Extract centered patch with jitter from full image
                patch_data, patch_mask = extract_full_padding_patch(
                    data, mask, 
                    patch_size=patch_size,
                    apply_jitter=apply_jitter
                    # You can make this a parameter
                )
                self.patches.append((patch_data, patch_mask))
            else:
                raise ValueError(f"Unknown patch_mode: {patch_mode}")

    def _load_sample(self, sample_name: str, location_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sample_dir = os.path.join(self.load_path, self.power_mode, sample_name)
        files_info = discover_data_files_for_location(
            sample_dir, location_idx, mask_type=self.mask_type,
            data_regime=self.data_regime
        )

        if files_info is None:
            raise ValueError(f"Cannot find data for {sample_name}, location {location_idx}")

        data, mask = load_and_aggregate_location(
            files_info, ppt_phases=self.ppt_phases, ppt_amps=self.ppt_amps,
            invert_mask=self.invert_mask, data_regime=self.data_regime
        )
        return data.astype(np.float32), mask.astype(np.uint8)

    def __len__(self):
        return len(self.patches)
        

    def __getitem__(self, idx):
        patch_data, patch_mask = self.patches[idx]
        patch_data = torch.from_numpy(patch_data.transpose(2, 0, 1).copy())
        patch_mask = torch.from_numpy(patch_mask.copy()).float()
        if patch_mask.ndim == 2:
            patch_mask = patch_mask.unsqueeze(0)  # (1,H,W)

        if self.transform:
            # patch_data, patch_mask = augment_patch(patch_data, patch_mask, rotate_img=self.rotate_img)
            seed = torch.randint(0, 10_000, (1,)).item()
            torch.manual_seed(seed)
            patch_data = self.transform(patch_data)
            torch.manual_seed(seed)
            patch_mask = self.mask_transform(patch_mask)
        # Convert to torch tensors (C, H, W format for PyTorch)
        #patch_data = torch.from_numpy(patch_data.transpose(2, 0, 1).copy())
        #patch_mask = torch.from_numpy(patch_mask.copy()).float()
        if patch_mask.shape[0] == 1:
            patch_mask = patch_mask.squeeze(0)
        return patch_data, patch_mask


def create_patch_dataloader(
    sample_indices: List[Tuple[str, int]],
    load_path: str,
    power_mode: str,
    patch_size: tuple = [ 128, 128  ],
    batch_size: int = 12,
    augment: bool = True,
    mask_type: str = 'normal',
    shuffle: bool = True,
    num_workers: int = 4,
    rotate_img: bool = True,
    ppt_phases: Union[str, int] = 'all',
    ppt_amps: int = 6,
    invert_mask: bool = False,
    apply_jitter: bool = True,
    min_positive_ratio: float = 0.05,
    patch_mode: str = "patches",
    data_regime: str = 'postprocessed'
) -> DataLoader:
    """Create a DataLoader for patch-based training."""
    dataset = PatchDataset(
        sample_indices=sample_indices,
        load_path=load_path,
        power_mode=power_mode,
        patch_size=patch_size,
        augment=augment,
        mask_type=mask_type,
        rotate_img=rotate_img,
        ppt_phases=ppt_phases,
        ppt_amps=ppt_amps,
        invert_mask=invert_mask,
        apply_jitter=apply_jitter,
        min_positive_ratio=min_positive_ratio,
        patch_mode=patch_mode,
        data_regime=data_regime
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_input_shape(
    load_path: str,
    power_mode: str,
    sample_indices: List[Tuple[str, int]],
    patch_size: tuple = [128, 128],
    mask_type: str = 'normal',
    ppt_phases: Union[str, int] = 'all',
    ppt_amps: int = 6,
    data_regime: str = 'postprocessed',
    patch_mode: str = "patches",
) -> Tuple[int, int, int]:
    """Get the input shape (C, H, W) for PyTorch model."""
    sample_name, location_idx = sample_indices[0]
    sample_dir = os.path.join(load_path, power_mode, sample_name)
    files_info = discover_data_files_for_location(
        sample_dir, location_idx, mask_type=mask_type, data_regime=data_regime
    )

    total_channels = calculate_total_channels(
        files_info, ppt_phases=ppt_phases, ppt_amps=ppt_amps, data_regime=data_regime
    )
    H,W= patch_size[0], patch_size[1]
    return (total_channels, H, W)
