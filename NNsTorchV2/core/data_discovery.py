"""
Data file discovery and validation utilities.

This module provides functions to discover PCA/PPT/ICA/Raw data files and masks,
extract parameters from filenames, and validate data file integrity.
Supports two data regimes: 'postprocessed' (PCA+PPT+ICA) and 'raw' (Raw only).
"""

import os
import re
from glob import glob
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


def extract_file_parameters(filename: str) -> Optional[Dict[str, Any]]:
    """
    Extract parameters from PCA/PPT/ICA/Raw filename.
    Patterns:
        {PCA|PPT|ICA}_a={a_value}_width={width_value}_{location}.npz
        Raw_{location}.npz
    Args:
        filename: Name of the file to parse
    Returns:
        dict with keys: 'type', 'a', 'width', 'location' (for PCA/PPT/ICA)
        or dict with keys: 'type', 'location' (for Raw)
        or None if pattern doesn't match
    """
    pattern = r'(PCA|PPT|ICA)_a=([0-9]+)_width=([0-9]+)_([0-9]+)\.npz'
    match = re.match(pattern, filename)

    if match:
        return {
            'type': match.group(1),
            'a': int(match.group(2)),
            'width': int(match.group(3)),
            'location': int(match.group(4))
        }

    # Raw file pattern: Raw_{location}.npz
    raw_match = re.match(r'Raw_([0-9]+)\.npz', filename)
    if raw_match:
        return {
            'type': 'Raw',
            'location': int(raw_match.group(1))
        }

    return None


def discover_data_files_for_location(
    sample_dir: str,
    location_idx: int,
    mask_type: str = 'normal',
    data_regime: str = 'postprocessed'
) -> Optional[Dict[str, Any]]:
    """
    Dynamically discover data files for a given location.

    Args:
        sample_dir: path to sample directory
        location_idx: location index
        mask_type: 'normal' or 'alternative'
        data_regime: 'postprocessed' (PCA+PPT+ICA) or 'raw' (Raw only)

    Returns:
        dict with structure depending on data_regime:
        postprocessed: {'mask':..., 'PCA':[(a,width,path),...], 'PPT':[...], 'ICA':[...]}
        raw: {'mask':..., 'Raw':[path,...]}
        or None if required files don't exist
    """
    # Determine mask file
    if mask_type == 'normal':
        mask_file = os.path.join(sample_dir, f'MaskV2_{location_idx}.npy')
    elif mask_type == 'alternative':
        mask_file = os.path.join(sample_dir, f'MaskV2_2sDiff_{location_idx}.npy')
    else:
        mask_file = os.path.join(sample_dir, f'{mask_type}_{location_idx}.npy')

    if not os.path.exists(mask_file):
        return None

    # Find all data files for this location
    pca_files = []
    ppt_files = []
    ica_files = []
    raw_files = []

    for filename in os.listdir(sample_dir):
        params = extract_file_parameters(filename)
        if params and params['location'] == location_idx:
            filepath = os.path.join(sample_dir, filename)

            if params['type'] == 'PCA':
                pca_files.append((params['a'], params['width'], filepath))
            elif params['type'] == 'PPT':
                ppt_files.append((params['a'], params['width'], filepath))
            elif params['type'] == 'ICA':
                ica_files.append((params['a'], params['width'], filepath))
            elif params['type'] == 'Raw':
                raw_files.append(filepath)

    # Sort by (a, width) for consistency
    pca_files.sort()
    ppt_files.sort()
    ica_files.sort()

    if data_regime == 'raw':
        if not raw_files:
            return None
        return {
            'mask': mask_file,
            'Raw': raw_files,
        }

    # postprocessed regime
    if not pca_files or not ppt_files:
        return None

    return {
        'mask': mask_file,
        'PCA': pca_files,
        'PPT': ppt_files,
        'ICA': ica_files,
    }


def validate_npz_file(filepath: str) -> bool:
    """
    Check if an npz file is valid and can be loaded.

    Args:
        filepath: Path to the npz file

    Returns:
        True if file is valid, False otherwise
    """
    try:
        with np.load(filepath) as data:
            _ = list(data.keys())
        return True
    except Exception as e:
        print(f"[WARNING] Invalid or corrupted file: {filepath} - {e}")
        return False


def discover_samples(
    load_path: str,
    power_mode: str,
    dirs: List[int],
    mask_type: str = 'normal',
    data_regime: str = 'postprocessed',
    max_locations: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Discover all available (sample, location) pairs in the dataset.

    Args:
        load_path: base data path
        power_mode: power mode to search
        dirs: list of directory indices to filter (empty list means all)
        mask_type: 'normal' or 'alternative'
        data_regime: 'postprocessed' (PCA+PPT+ICA) or 'raw' (Raw only)

    Returns:
        list of (sample_name, location_idx) tuples
    """
    power_dir = os.path.join(load_path, power_mode)
    samples = []

    if dirs:
        sample_dirs = sorted(
            d for d in os.listdir(power_dir)
            if os.path.isdir(os.path.join(power_dir, d))
            and d.startswith('s')
            and int(d[-1]) in dirs
        )
    else:
        sample_dirs = sorted([
            d for d in os.listdir(power_dir)
            if os.path.isdir(os.path.join(power_dir, d)) and d.startswith('s')
        ])

    for sample_name in sample_dirs:
        sample_dir = os.path.join(power_dir, sample_name)

        # Find all mask files
        if mask_type == 'normal':
            mask_pattern = 'MaskV2_[0-9]*.npy'
        elif mask_type == 'alternative':
            mask_pattern = 'MaskV2_2sDiff_[0-9]*.npy'
        else:
            mask_pattern = f'{mask_type}_[0-9]*.npy'

        mask_files = sorted(glob(os.path.join(sample_dir, mask_pattern)))

        # Filter mask files based on type
        if mask_type != 'alternative':
            # Exclude files with 'Diff' in the name
            mask_files = [f for f in mask_files if 'Diff' not in os.path.basename(f)]

        for mask_file in mask_files:
            basename = os.path.basename(mask_file)

            # Extract location index using regex for more robustness
            if mask_type == 'normal':
                match = re.match(r'MaskV2_(\d+)\.npy', basename)
            elif mask_type == 'alternative':
                match = re.match(r'MaskV2_2sDiff_(\d+)\.npy', basename)
            else:
                match = re.match(rf'{mask_type}_(\d+)\.npy', basename)

            if not match:
                continue

            location_idx = int(match.group(1))

            # Discover files for this location
            files_info = discover_data_files_for_location(
                sample_dir, location_idx, mask_type=mask_type,
                data_regime=data_regime
            )

            if files_info is None:
                continue

            # Validate all npz files
            all_valid = True

            if data_regime == 'raw':
                for filepath in files_info['Raw']:
                    if not validate_npz_file(filepath):
                        all_valid = False
                        break
            else:
                for _, _, filepath in files_info['PCA']:
                    if not validate_npz_file(filepath):
                        all_valid = False
                        break

                if all_valid:
                    for _, _, filepath in files_info['PPT']:
                        if not validate_npz_file(filepath):
                            all_valid = False
                            break

                if all_valid:
                    for _, _, filepath in files_info.get('ICA', []):
                        if not validate_npz_file(filepath):
                            all_valid = False
                            break

            if all_valid:
                samples.append((sample_name, location_idx))
            else:
                print(f"[WARNING] Skipping {sample_name}, location {location_idx} due to corrupted files")

    if max_locations is not None:
        unique_locs = []
        for _, loc in samples:
            if loc not in unique_locs:
                unique_locs.append(loc)      # preserve discovery order, deduplicate
        keep = set(unique_locs[:max_locations])
        samples = [(s, l) for s, l in samples if l in keep]

    return samples
