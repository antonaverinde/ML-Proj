import math
from typing import List, Tuple

import numpy as np


def calculate_patch_grid(image_size: int, patch_size: int) -> Tuple[int, float, float, float]:
    """
    Calculate patch extraction grid based on overlap strategy.

    Args:
        image_size: Size of the image along one dimension
        patch_size: Size of patches to extract

    Returns:
        n_patches: number of patches along this dimension
        shift_per_gap: overlap amount per gap
        ratio: ratio of overlap to patch_size
        max_jitter: maximum jitter to apply (0 if no jitter)
    """
    n_patches = math.ceil(image_size / patch_size)

    if n_patches == 1:
        return 1, 0.0, 0.0, 0.0

    deficit = n_patches * patch_size - image_size
    shift_per_gap = deficit / (n_patches - 1)
    ratio = shift_per_gap / patch_size

    if ratio > 0.25:
        max_jitter = (shift_per_gap / 2)
        n_patches -= 1
    else:
        max_jitter = 0.0

    return n_patches, shift_per_gap, ratio, max_jitter


def get_base_positions(
    n_patches: int,
    image_size: int,
    patch_size: int,
    max_jitter: float
) -> List[float]:
    """
    Calculate base positions for patch extraction.

    Args:
        n_patches: Number of patches to extract
        image_size: Size of the image
        patch_size: Size of each patch
        max_jitter: Maximum jitter value

    Returns:
        List of base positions
    """
    if n_patches == 1:
        return [0.0]

    base_first = max_jitter
    base_last = image_size - patch_size - max_jitter

    positions = []
    for i in range(n_patches):
        pos = base_first + i * (base_last - base_first) / (n_patches - 1)
        positions.append(pos)
    return positions


def extract_patches_from_image(
    data: np.ndarray,
    mask: np.ndarray,
    patch_size: tuple,
    apply_jitter: bool = False,
    min_positive_ratio: float = 0.05,
    masks_only=False  # NEW PARAMETER
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract patches from a single image using the calculated grid strategy with automatic jitter.

    Args:
        data: array of shape (H, W, C)
        mask: array of shape (H, W)
        patch_size: target patch size (assumed square)
        apply_jitter: whether to apply random jitter (for training)
        min_positive_ratio: minimum ratio of positive pixels to keep patch (0.0 to 1.0)
                           0.0 = keep all patches
                           0.05 = keep patches with ≥5% positive pixels
                           0.10 = keep patches with ≥10% positive pixels

    Returns:
        patches: list of (patch_data, patch_mask) tuples
    """
    if masks_only:
        H, W = mask.shape
    else:
        H, W, C = data.shape

    n_h, shift_per_gap_h, ratio_h, max_jitter_h = calculate_patch_grid(H, patch_size[0])
    n_w, shift_per_gap_w, ratio_w, max_jitter_w = calculate_patch_grid(W, patch_size[1])

    base_positions_h = get_base_positions(n_h, H, patch_size[0], max_jitter_h)
    base_positions_w = get_base_positions(n_w, W, patch_size[1], max_jitter_w)

    patches = []
    total_patches = 0
    filtered_patches = 0

    for i in range(n_h):
        for j in range(n_w):
            y = base_positions_h[i]
            x = base_positions_w[j]

            # Apply jitter if enabled
            if apply_jitter:
                if max_jitter_h > 0:
                    y_jitter = np.random.uniform(-max_jitter_h, max_jitter_h)
                    y += y_jitter
                if max_jitter_w > 0:
                    x_jitter = np.random.uniform(-max_jitter_w, max_jitter_w)
                    x += x_jitter

            # Convert to integer coordinates
            y = int(round(y))
            x = int(round(x))
            if not masks_only:
                patch_data = data[y:y + patch_size[0], x:x + patch_size[1], :]
            patch_mask = mask[y:y + patch_size[0], x:x + patch_size[1]]

            total_patches += 1

            # Filter based on positive pixel ratio
            if min_positive_ratio > 0.0:
                pos_ratio = patch_mask.sum() / (patch_size[0] * patch_size[1])
                if pos_ratio < min_positive_ratio:
                    filtered_patches += 1
                    continue  # Skip this patch
            if masks_only:
                patches.append(patch_mask)
            else:
                patches.append((patch_data, patch_mask))

    # Optional: print filtering statistics (can be removed if too verbose)
    if min_positive_ratio > 0.0 and total_patches > 0:
        kept_ratio = (total_patches - filtered_patches) / total_patches
        print(f"  Patch filtering: {total_patches - filtered_patches}/{total_patches} kept "
              f"({kept_ratio*100:.1f}%), filtered {filtered_patches} patches "
              f"with <{min_positive_ratio*100:.1f}% positive pixels")

    return patches
def augment_patch(
    data: np.ndarray,
    mask: np.ndarray,
    rotate_img: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentations to a patch.

    Args:
        data: Patch data of shape (H, W, C)
        mask: Patch mask of shape (H, W)
        rotate_img: Whether to apply random 90-degree rotations

    Returns:
        Augmented (data, mask) tuple
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        data = np.fliplr(data)
        mask = np.fliplr(mask)

    # Random vertical flip
    if np.random.rand() > 0.5:
        data = np.flipud(data)
        mask = np.flipud(mask)

    # Random rotation with fixed 90-degree angles
    if rotate_img:
        # Randomly select from 0, 90, 180, 270 degrees
        angle = np.random.choice([0, 90, 180, 270])

        if angle > 0:
            # Number of 90-degree rotations
            k = angle // 90

            # Rotate data using rot90 (faster for 90-degree multiples)
            data = np.rot90(data, k=k, axes=(0, 1))

            # Rotate mask
            mask = np.rot90(mask, k=k, axes=(0, 1))

    return data, mask
