"""
V2 Hybrid dataset and XGBoost utilities for HybridTrainV2.

Identical to hybrid_utils.py but reads from HDF5 files via data_discovery_V2 / data_loading_V2.
Switch pipeline with one import line change in the notebook.
"""

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

from ...core.data_discovery_V2 import discover_data_files_for_location
from ...core.data_loading_V2 import load_and_aggregate_location
from ...core.patch_extraction import extract_patches_from_image
from ...core.full_img_padding import extract_full_padding_patch


_spatial_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
])


class HybridPatchDataset(Dataset):
    """
    V2 Patch dataset: reads from HDF5 via data_discovery_V2 / data_loading_V2.
    API identical to V1 HybridPatchDataset; location is str (not int) in V2.
    """

    def __init__(
        self,
        sample_indices: List[Tuple[str, str]],   # (sample_name, location_name) both str
        xgb_model,
        load_path: str,          # h5_base_dir in V2 (directory with 3 H5 files)
        power_mode: str,
        patch_size: tuple = (128, 128),
        augment: bool = True,
        mask_type: str = 'alternative',
        ppt_phases: Union[str, int] = 'all',
        ppt_amps: int = 6,
        invert_mask: bool = True,
        apply_jitter: bool = True,
        min_positive_ratio: float = 0.05,
        patch_mode: str = 'full_padding',
        data_regime: str = 'postprocessed',
        min_mask_area: int = 0,
    ):
        self.transform = _spatial_transform if augment else None
        self.patches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for sample_name, location_name in sample_indices:
            fi = discover_data_files_for_location(
                load_path, power_mode, sample_name, location_name,
                mask_type=mask_type, data_regime=data_regime)
            if fi is None:
                continue

            data, mask = load_and_aggregate_location(
                fi, ppt_phases=ppt_phases, ppt_amps=ppt_amps,
                invert_mask=invert_mask, data_regime=data_regime,
                min_mask_area=min_mask_area)
            data = data.astype(np.float32)
            mask = mask.astype(np.float32)

            H, W, C = data.shape
            if xgb_model is not None:
                xgb_prob = (xgb_model.predict_proba(data.reshape(-1, C))[:, 1]
                            .reshape(H, W).astype(np.float32))
            else:
                xgb_prob = np.zeros((H, W), dtype=np.float32)

            data_xgb = np.concatenate(
                [data, xgb_prob[:, :, np.newaxis]], axis=2)

            if patch_mode == 'patches':
                for d_patch, m_patch in extract_patches_from_image(
                        data_xgb, mask, patch_size,
                        apply_jitter=apply_jitter,
                        min_positive_ratio=min_positive_ratio):
                    self.patches.append((
                        d_patch[:, :, :-1],
                        d_patch[:, :, -1],
                        m_patch))

            elif patch_mode == 'full_padding':
                d_patch, m_patch = extract_full_padding_patch(
                    data_xgb, mask,
                    patch_size=patch_size, apply_jitter=apply_jitter)
                self.patches.append((
                    d_patch[:, :, :-1],
                    d_patch[:, :, -1],
                    m_patch))

            elif patch_mode == 'full':
                self.patches.append((data, xgb_prob, mask))

            else:
                raise ValueError(f"Unknown patch_mode: {patch_mode!r}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        data, xgb_prob, mask = self.patches[idx]

        data_t = torch.from_numpy(data.transpose(2, 0, 1).copy())
        xgb_t  = torch.from_numpy(xgb_prob[np.newaxis].copy())
        mask_t = torch.from_numpy(mask.copy()).float().unsqueeze(0)

        if self.transform:
            seed = torch.randint(0, 10_000, (1,)).item()
            torch.manual_seed(seed); data_t = self.transform(data_t)
            torch.manual_seed(seed); xgb_t  = self.transform(xgb_t)
            torch.manual_seed(seed); mask_t = self.transform(mask_t)

        return data_t, xgb_t, mask_t.squeeze(0)


def create_hybrid_dataloader(
    sample_indices: List[Tuple[str, str]],
    xgb_model,
    load_path: str,
    power_mode: str,
    patch_size: tuple = (128, 128),
    batch_size: int = 4,
    augment: bool = True,
    mask_type: str = 'alternative',
    shuffle: bool = True,
    num_workers: int = 4,
    ppt_phases: Union[str, int] = 'all',
    ppt_amps: int = 6,
    invert_mask: bool = True,
    apply_jitter: bool = True,
    min_positive_ratio: float = 0.05,
    patch_mode: str = 'full_padding',
    data_regime: str = 'postprocessed',
    min_mask_area: int = 0,
) -> DataLoader:
    """Create a DataLoader backed by HybridPatchDataset (V2)."""
    dataset = HybridPatchDataset(
        sample_indices=sample_indices,
        xgb_model=xgb_model,
        load_path=load_path,
        power_mode=power_mode,
        patch_size=patch_size,
        augment=augment,
        mask_type=mask_type,
        ppt_phases=ppt_phases,
        ppt_amps=ppt_amps,
        invert_mask=invert_mask,
        apply_jitter=apply_jitter,
        min_positive_ratio=min_positive_ratio,
        patch_mode=patch_mode,
        data_regime=data_regime,
        min_mask_area=min_mask_area,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
