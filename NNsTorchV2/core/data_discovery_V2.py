"""
V2 data discovery — reads from consolidated HDF5 files instead of per-sample NPZ/NPY files.

H5 files in h5_base_dir:
  dataset_masks_v1.h5    — binary masks
  dataset_raw_v1.h5      — raw temperature data
  dataset_features_v1.h5 — PCA/ICA/PPT components

Key format for mask key: /{power_mode}/{sample}/{location}/BinMask_0/data
Key format for feature:  /{power_mode}/{sample}/{location}/PCA|ICA|PPT/a=N_width=M/...
mask_type parameter is accepted but IGNORED (only BinMask_0 exists).
"""
import h5py, os, re
from typing import Dict, List, Optional, Any, Tuple


def discover_data_files_for_location(
    h5_base_dir: str,       # directory containing the 3 H5 files
    power_mode: str,        # e.g. '4kw_both'
    sample_name: str,       # e.g. 's0'
    location_name: str,     # e.g. 'bottom_left'  ← str not int (V2 difference)
    mask_type: str = 'normal',    # accepted but IGNORED (only BinMask_0)
    data_regime: str = 'postprocessed'
) -> Optional[Dict[str, Any]]:
    """
    Return files_info dict for one (power_mode, sample_name, location_name) triple.
    Returns None if required data is missing.

    postprocessed return format:
    {
        'mask_h5':     '/path/dataset_masks_v1.h5',
        'mask_key':    '/4kw_both/s0/bottom_left/BinMask_0/data',
        'features_h5': '/path/dataset_features_v1.h5',
        'base_key':    '/4kw_both/s0/bottom_left',
        'PCA': [(a, width, 'PCA/a=10_width=110'), ...],
        'PPT': [(a, width, 'PPT/a=0_width=110'), ...],
        'ICA': [(a, width, 'ICA/a=0_width=280')],
        'data_regime': 'postprocessed',
        'shape': (H, W),    # cached so infrastructure avoids re-opening H5
    }

    raw return format:
    {
        'mask_h5':  '/path/dataset_masks_v1.h5',
        'mask_key': '/4kw_both/s0/bottom_left/BinMask_0/data',
        'raw_h5':   '/path/dataset_raw_v1.h5',
        'raw_key':  '/4kw_both/s0/bottom_left/temperature/data',
        'data_regime': 'raw',
        'shape': (H, W),
    }
    """
    masks_path    = os.path.join(h5_base_dir, 'dataset_masks_v1.h5')
    features_path = os.path.join(h5_base_dir, 'dataset_features_v1.h5')
    raw_path      = os.path.join(h5_base_dir, 'dataset_raw_v1.h5')

    base_key = f'/{power_mode}/{sample_name}/{location_name}'
    mask_key = f'{base_key}/BinMask_0/data'

    with h5py.File(masks_path, 'r') as f:
        if mask_key not in f:
            return None
        H, W = f[mask_key].shape

    if data_regime == 'raw':
        raw_key = f'{base_key}/temperature/data'
        with h5py.File(raw_path, 'r') as f:
            if raw_key not in f:
                return None
        return {'mask_h5': masks_path, 'mask_key': mask_key,
                'raw_h5': raw_path, 'raw_key': raw_key,
                'data_regime': 'raw', 'shape': (H, W)}

    # postprocessed: enumerate feature groups
    pca, ppt, ica = [], [], []
    with h5py.File(features_path, 'r') as f:
        if base_key not in f:
            return None
        grp = f[base_key]
        for feat_type, container in [('PCA', pca), ('PPT', ppt), ('ICA', ica)]:
            if feat_type in grp:
                for variant in grp[feat_type]:
                    m = re.match(r'a=(\d+)_width=(\d+)', variant)
                    if m:
                        container.append((int(m.group(1)), int(m.group(2)),
                                          f'{feat_type}/{variant}'))
    pca.sort(); ppt.sort(); ica.sort()

    if not pca or not ppt:
        return None

    return {'mask_h5': masks_path, 'mask_key': mask_key,
            'features_h5': features_path, 'base_key': base_key,
            'PCA': pca, 'PPT': ppt, 'ICA': ica,
            'data_regime': 'postprocessed', 'shape': (H, W)}


def discover_samples(
    h5_base_dir: str,
    power_mode: str,
    dirs: List[int],
    mask_type: str = 'normal',
    data_regime: str = 'postprocessed',
    max_locations: Optional[int] = None,
) -> List[Tuple[str, str]]:   # (sample_name, location_name) — both strings
    """
    Discover all available (sample_name, location_name) pairs.
    dirs: list of sample indices to include (e.g. [0,1,2]); empty list = all.
    Handles multi-digit sample names (s10, s11, ...).
    """
    masks_path = os.path.join(h5_base_dir, 'dataset_masks_v1.h5')
    samples = []
    with h5py.File(masks_path, 'r') as f:
        if power_mode not in f:
            return []
        pm = f[power_mode]
        all_sample_keys = sorted(pm.keys())
        if dirs:
            all_sample_keys = [s for s in all_sample_keys
                               if s.startswith('s') and s[1:].isdigit()
                               and int(s[1:]) in dirs]
        for sname in all_sample_keys:
            for loc_name in sorted(pm[sname].keys()):
                fi = discover_data_files_for_location(
                    h5_base_dir, power_mode, sname, loc_name,
                    mask_type=mask_type, data_regime=data_regime)
                if fi is not None:
                    samples.append((sname, loc_name))

    if max_locations is not None:
        unique_locs = []
        for _, loc in samples:
            if loc not in unique_locs:
                unique_locs.append(loc)      # preserve discovery order, deduplicate
        keep = set(unique_locs[:max_locations])
        samples = [(s, l) for s, l in samples if l in keep]

    return samples
