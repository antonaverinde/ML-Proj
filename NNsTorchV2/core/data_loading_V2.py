"""
V2 data loading — reads from HDF5 files using files_info from data_discovery_V2.

Key difference from V1: data is stored in H5 component=N/data datasets, not NPZ arrays.
Opens each H5 file only once per load call.
"""
import re
from typing import Dict, Any, List, Tuple, Union

import h5py
import numpy as np


def _sort_component_keys(keys) -> List[str]:
    """Sort 'component=N' strings numerically (not lexicographically)."""
    return sorted(keys, key=lambda k: int(re.match(r'component=(\d+)', k).group(1)))


def _read_components(h5file, group_path: str) -> np.ndarray:
    """Read all component=N/data datasets from group_path; stack to (H, W, N)."""
    grp = h5file[group_path]
    comp_keys = _sort_component_keys(
        [k for k in grp.keys() if k.startswith('component=')])
    arrays = [grp[k]['data'][...].astype(np.float32) for k in comp_keys]
    return np.stack(arrays, axis=2)


def load_and_aggregate_location(
    files_info: Dict[str, Any],
    ppt_phases: Union[str, int] = 'all',
    ppt_amps: int = 6,
    invert_mask: bool = False,
    mask_only: bool = False,
    data_regime: str = 'postprocessed',
    min_mask_area: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mask + feature data for one location from H5 files.

    Returns:
        data: (H, W, C) float32
        mask: (H, W) float32
    """
    with h5py.File(files_info['mask_h5'], 'r') as f:
        mask = f[files_info['mask_key']][...].astype(np.float32)
    if min_mask_area > 0:
        from scipy import ndimage
        labeled, _ = ndimage.label(mask > 0)
        sizes = np.bincount(labeled.ravel())
        keep = sizes >= min_mask_area
        keep[0] = False  # background (label 0) must always map to 0
        mask = keep[labeled].astype(mask.dtype)
    if invert_mask:
        mask = 1.0 - mask
    if mask_only:
        return mask, mask

    if data_regime == 'raw':
        with h5py.File(files_info['raw_h5'], 'r') as f:
            return f[files_info['raw_key']][...].astype(np.float32), mask

    # postprocessed: open features H5 once
    parts = []
    base = files_info['base_key']
    with h5py.File(files_info['features_h5'], 'r') as f:
        for a, w, rel in files_info['PCA']:
            parts.append(_read_components(f, f'{base}/{rel}'))

        for a, w, rel in files_info['PPT']:
            phase = _read_components(f, f'{base}/{rel}/Phase')
            amp   = _read_components(f, f'{base}/{rel}/Amp')
            n_ph  = phase.shape[2] if ppt_phases == 'all' else int(ppt_phases)
            parts.append(phase[:, :, :n_ph])
            parts.append(amp[:, :, :ppt_amps])

        for a, w, rel in files_info.get('ICA', []):
            parts.append(_read_components(f, f'{base}/{rel}'))

    return np.concatenate(parts, axis=2), mask


def calculate_total_channels(
    files_info: Dict[str, Any],
    ppt_phases: Union[str, int] = 'all',
    ppt_amps: int = 6,
    data_regime: str = 'postprocessed'
) -> int:
    """Calculate total feature channels without loading all data."""
    if data_regime == 'raw':
        with h5py.File(files_info['raw_h5'], 'r') as f:
            return f[files_info['raw_key']].shape[2]

    total = 0
    base = files_info['base_key']
    with h5py.File(files_info['features_h5'], 'r') as f:
        for a, w, rel in files_info['PCA']:
            grp = f[f'{base}/{rel}']
            total += len([k for k in grp if k.startswith('component=')])

        for a, w, rel in files_info['PPT']:
            ph_grp = f[f'{base}/{rel}/Phase']
            am_grp = f[f'{base}/{rel}/Amp']
            n_ph   = len([k for k in ph_grp if k.startswith('component=')])
            n_am   = len([k for k in am_grp if k.startswith('component=')])
            total += (n_ph if ppt_phases == 'all' else min(int(ppt_phases), n_ph))
            total += min(ppt_amps, n_am)

        for a, w, rel in files_info.get('ICA', []):
            grp = f[f'{base}/{rel}']
            total += len([k for k in grp if k.startswith('component=')])

    return total
