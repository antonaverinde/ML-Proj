import numpy as np
from typing import Tuple

import numpy as np
from typing import Tuple

def extract_full_padding_patch(
    data: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int],
    apply_jitter: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a patch of given size from the original image, centered with optional random jitter.
    The patch_size must be smaller than or equal to the original image dimensions.
    
    Args:
        data: Input image of shape (H, W, C)
        mask: Input mask of shape (H, W)
        patch_size: Target patch size as (height, width) - must be <= original image size
        apply_jitter: Whether to apply random jitter to the center position
    
    Returns:
        Tuple of (patch_data, patch_mask) with shapes (patch_h, patch_w, C) and (patch_h, patch_w)
    """
    h, w = data.shape[:2]
    patch_h, patch_w = patch_size
    if patch_h > h or patch_w > w:
        raise ValueError(
            f"Patch size {patch_size} cannot be larger than image size ({h}, {w}). "
            f"Use patch_mode='full' or reduce patch_size."
        )
    center_y = h // 2
    center_x = w // 2
    max_jitter_y = center_y - patch_h // 2  
    max_jitter_x = center_x - patch_w // 2  
    max_jitter_y = min(max_jitter_y, h - center_y - (patch_h - patch_h // 2))
    max_jitter_x = min(max_jitter_x, w - center_x - (patch_w - patch_w // 2))
   
    if apply_jitter:
        # Random displacement from 0 to max_jitter in each direction (positive or negative)
        jitter_y = np.random.randint(-max_jitter_y, max_jitter_y + 1) if max_jitter_y > 0 else 0
        jitter_x = np.random.randint(-max_jitter_x, max_jitter_x + 1) if max_jitter_x > 0 else 0
    else:
        jitter_y = 0
        jitter_x = 0
    
    # Calculate top-left corner of the patch
    # Start from center, subtract half patch size, add jitter
    top = center_y - patch_h // 2 + jitter_y
    left = center_x - patch_w // 2 + jitter_x
    
    # Extract the patch (no padding needed)
    patch_data = data[top:top + patch_h, left:left + patch_w]
    patch_mask = mask[top:top + patch_h, left:left + patch_w]
    
    return patch_data, patch_mask
