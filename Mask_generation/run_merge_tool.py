"""
Simple script to run the interactive image merge tool.

This script merges two images (Front and Back) using correspondence points.
The tool automatically:
- Derives the Back image path from the Front image path
- Creates the output directory if needed
- Preserves defect regions (black pixels) from the Back image
- Saves the transformation matrix for later use

Usage:
------
Simply run this script and follow the interactive prompts:
1. Select correspondence points on the Front image
2. Select matching points on the Back image (in the same order)
3. Preview the merged result
4. Press 'S' to save or ESC to cancel
"""

from ImageMergeTool import ImageMergeTool
from config import BINARY_MASKS_PATH


# ========== Configuration ==========
# Path to the front image (back image path will be auto-derived)
image1_path = str(BINARY_MASKS_PATH / "Barrel_Images2_croped_Transformed_V2" / "S2_Front_Transformed.jpg")

# Optional: Specify back image path explicitly (if None, auto-derived from image1_path)
image2_path = None  # Will become S2_Back_Transformed.jpg

# Optional: Specify save directory (if None, auto-derived as "Merged" folder)
save_dir = None  # Will become Barrel_Images2_croped_Merged

# Flip image2 horizontally? (usually True for back images)
flip_image2 = True
# ===================================


if __name__ == "__main__":
    # Create and run the merge tool
    tool = ImageMergeTool(
        image1_path=image1_path,
        image2_path=image2_path,
        save_dir=save_dir,
        flip_image2=flip_image2
    )

    tool.run()

    # Results are accessible if needed
    if tool.merged_image is not None:
        print("\n" + "="*60)
        print("RESULTS AVAILABLE")
        print("="*60)
        print("Access in code via:")
        print("  - tool.merged_image (numpy array)")
        print("  - tool.homography_matrix (3x3 matrix)")
        print("  - tool.aligned_image2 (warped back image)")
        print("="*60)
