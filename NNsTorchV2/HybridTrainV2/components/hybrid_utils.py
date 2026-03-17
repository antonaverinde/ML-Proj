"""
Hybrid dataset and XGBoost utilities for HybridTrainV2.

HybridPatchDataset is a close adaptation of NNsTorch.data_pipeline.PatchDataset.
Key difference: XGBoost is run on the full image first, its probability map is
appended as a temporary extra channel, then existing patch extractors
(extract_patches_from_image / extract_full_padding_patch) run unchanged.
The extra channel is split off before storing, so __getitem__ always returns
    (data   [C, H, W],  raw features)
    (xgb_t  [1, H, W],  XGB probability)
    (mask_t [H, W])

Zero changes are needed to any original NNsTorch file.
"""

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

from ...core.data_discovery import discover_data_files_for_location
from ...core.data_loading import load_and_aggregate_location
from ...core.patch_extraction import extract_patches_from_image
from ...core.full_img_padding import extract_full_padding_patch


# ── Transforms (same as data_pipeline.py) ────────────────────────────────────

_spatial_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5), #can add rotate but afraid it will harm channel wise representation as it uses interpolation.
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class HybridPatchDataset(Dataset):
    """
    Patch dataset with XGBoost probability channel.

    Adapted from NNsTorch.data_pipeline.PatchDataset.  XGBoost is applied once
    per full image; its output is temporarily appended as channel C+1 so that
    existing patch extractors guarantee identical spatial crops for both data
    and XGB prob.

    Supports all patch_modes:
        'patches'      — grid-based patch extraction (extract_patches_from_image)
        'full_padding' — single center crop with jitter (extract_full_padding_patch)
        'full'         — full image (no crop, variable size — use batch_size=1)

    Returns per __getitem__:
        data_t  : torch.float32  (C, H, W)  — raw feature channels
        xgb_t   : torch.float32  (1, H, W)  — XGB probability
        mask_t  : torch.float32  (H, W)     — binary ground truth
    """

    def __init__(
        self,
        sample_indices: List[Tuple[str, int]],
        xgb_model,
        load_path: str,
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
    ):
        self.transform = _spatial_transform if augment else None
        self.patches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for sample_name, location_idx in sample_indices:
            sample_dir = os.path.join(load_path, power_mode, sample_name)
            fi = discover_data_files_for_location(
                sample_dir, location_idx,
                mask_type=mask_type, data_regime=data_regime)
            if fi is None:
                continue

            data, mask = load_and_aggregate_location(
                fi, ppt_phases=ppt_phases, ppt_amps=ppt_amps,
                invert_mask=invert_mask, data_regime=data_regime)
            data = data.astype(np.float32)     # (H, W, C)
            mask = mask.astype(np.float32)     # (H, W)

            # XGB probability map — run once on the full image (None → zeros for nn_only)
            H, W, C = data.shape
            if xgb_model is not None:
                xgb_prob = (xgb_model.predict_proba(data.reshape(-1, C))[:, 1]
                            .reshape(H, W).astype(np.float32))
            else:
                xgb_prob = np.zeros((H, W), dtype=np.float32)

            # Append XGB as last channel so extractors apply the same crop to both
            data_xgb = np.concatenate(
                [data, xgb_prob[:, :, np.newaxis]], axis=2)   # (H, W, C+1)

            if patch_mode == 'patches':
                for d_patch, m_patch in extract_patches_from_image(
                        data_xgb, mask, patch_size,
                        apply_jitter=apply_jitter,
                        min_positive_ratio=min_positive_ratio):
                    self.patches.append((
                        d_patch[:, :, :-1],   # (H_p, W_p, C)
                        d_patch[:, :, -1],    # (H_p, W_p)
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

        data_t = torch.from_numpy(data.transpose(2, 0, 1).copy())   # (C, H, W)
        xgb_t  = torch.from_numpy(xgb_prob[np.newaxis].copy())      # (1, H, W)
        mask_t = torch.from_numpy(mask.copy()).float().unsqueeze(0)  # (1, H, W)

        if self.transform:
            # Same seed → identical spatial transform on all three
            seed = torch.randint(0, 10_000, (1,)).item()
            torch.manual_seed(seed); data_t = self.transform(data_t)
            torch.manual_seed(seed); xgb_t  = self.transform(xgb_t)
            torch.manual_seed(seed); mask_t = self.transform(mask_t)

        return data_t, xgb_t, mask_t.squeeze(0)


def create_hybrid_dataloader(
    sample_indices: List[Tuple[str, int]],
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
) -> DataLoader:
    """Create a DataLoader backed by HybridPatchDataset."""
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
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
