"""
Data loading and aggregation utilities.

This module handles loading PCA/PPT/ICA/Raw data files and aggregating them
into unified feature arrays for model training.
Supports 'postprocessed' (PCA+PPT+ICA) and 'raw' (Raw only) data regimes.
"""

from typing import Dict, Any, Tuple, Union

import numpy as np


def load_and_aggregate_location(
    files_info: Dict[str, Any],
    ppt_phases: Union[str, int] = 'all',
    ppt_amps: int = 6,
    invert_mask: bool = False,
    mask_only: bool = False,
    data_regime: str = 'postprocessed'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and aggregate all data for a single location with dynamic file discovery.

    Args:
        files_info: dict from discover_data_files_for_location
        ppt_phases: 'all' or int (number of phases to use)
        ppt_amps: int (number of amplitudes to use)
        invert_mask: if True, inverts the mask (0->1, 1->0)
        mask_only: if True, returns (mask, mask) without loading data
        data_regime: 'postprocessed' (PCA+PPT+ICA) or 'raw' (Raw only)

    Returns:
        data: array of shape (H, W, C)
        mask: array of shape (H, W)
    """
    # Load mask
    mask = np.load(files_info['mask'])
    H, W = mask.shape

    if invert_mask:
        # Invert mask: 0 becomes 1, 1 becomes 0
        mask = 1 - mask
    if mask_only:
        return mask, mask

    if data_regime == 'raw':
        # Load Raw data
        raw_parts = []
        for filepath in files_info['Raw']:
            raw_data = np.load(filepath)['Raw']
            raw_parts.append(raw_data.astype(np.float32))
        data = np.concatenate(raw_parts, axis=2) if len(raw_parts) > 1 else raw_parts[0]
        return data, mask

    # postprocessed regime: PCA + PPT + ICA
    total_channels = 0

    # Count PCA channels
    for _, _, filepath in files_info['PCA']:
        test_data = np.load(filepath)['converted_data']
        total_channels += test_data.shape[2]

    # Count PPT channels
    for _, _, filepath in files_info['PPT']:
        test_data = np.load(filepath)
        n_phases = len(test_data['Phase'][0, 0, :]) if ppt_phases == 'all' else ppt_phases
        total_channels += n_phases + ppt_amps

    # Count ICA channels
    for _, _, filepath in files_info.get('ICA', []):
        test_data = np.load(filepath)['ICA_data']
        total_channels += test_data.shape[2]

    # Initialize data array
    data = np.zeros((H, W, total_channels), dtype=np.float32)
    channel_idx = 0

    # Load PCA files
    for a, width, filepath in files_info['PCA']:
        pca_data = np.load(filepath)['converted_data']
        n_components = pca_data.shape[2]
        data[:, :, channel_idx:channel_idx + n_components] = pca_data.astype(np.float32)
        channel_idx += n_components

    # Load PPT files
    for a, width, filepath in files_info['PPT']:
        ppt_data = np.load(filepath)

        # Add phases
        ppt_phase = ppt_data['Phase']
        if ppt_phases == 'all':
            n_phases = ppt_phase.shape[2]
        else:
            n_phases = ppt_phases
        data[:, :, channel_idx:channel_idx + n_phases] = ppt_phase[:, :, :n_phases].astype(np.float32)
        channel_idx += n_phases

        # Add amplitudes
        ppt_amp = ppt_data['Amp'][:, :, :ppt_amps]
        data[:, :, channel_idx:channel_idx + ppt_amps] = ppt_amp.astype(np.float32)
        channel_idx += ppt_amps

    # Load ICA files (same format as PCA)
    for a, width, filepath in files_info.get('ICA', []):
        ica_data = np.load(filepath)['ICA_data']
        n_components = ica_data.shape[2]
        data[:, :, channel_idx:channel_idx + n_components] = ica_data.astype(np.float32)
        channel_idx += n_components

    return data, mask


def calculate_total_channels(
    files_info: Dict[str, Any],
    ppt_phases: Union[str, int] = 'all',
    ppt_amps: int = 6,
    data_regime: str = 'postprocessed'
) -> int:
    """
    Calculate total number of channels for a given files_info.

    Args:
        files_info: dict from discover_data_files_for_location
        ppt_phases: 'all' or int (number of phases to use)
        ppt_amps: int (number of amplitudes to use)
        data_regime: 'postprocessed' or 'raw'

    Returns:
        Total number of channels
    """
    if data_regime == 'raw':
        total = 0
        for filepath in files_info['Raw']:
            raw_data = np.load(filepath)['Raw']
            total += raw_data.shape[2]
        return total

    # postprocessed regime
    total_channels = 0

    # Count PCA channels
    for _, _, filepath in files_info['PCA']:
        test_data = np.load(filepath)['converted_data']
        total_channels += test_data.shape[2]

    # Count PPT channels
    for _, _, filepath in files_info['PPT']:
        test_data = np.load(filepath)
        n_phases = len(test_data['Phase'][0, 0, :]) if ppt_phases == 'all' else ppt_phases
        total_channels += n_phases + ppt_amps

    # Count ICA channels
    for _, _, filepath in files_info.get('ICA', []):
        test_data = np.load(filepath)['ICA_data']
        total_channels += test_data.shape[2]

    return total_channels
