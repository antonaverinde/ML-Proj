"""
Interactive Image Rotation and Displacement Tool

This tool allows you to interactively select two points on an image to define
a new coordinate system axis. The image is then rotated and displaced so that
the selected axis aligns with the vertical axis of the image.

Features:
---------
- Interactive point selection with visual feedback
- Zoom in/out with Ctrl + Mouse Wheel
- Pan image with Right Mouse Button
- Undo point selection with 'U' key
- Live line preview between points
- Separate viewer for transformed image with zoom/pan
- Preserves original image resolution

Controls:
---------
Selection Window:
  - Left Click: Select point (1st and 2nd)
  - Right Drag: Pan the image
  - Ctrl + Wheel: Zoom in/out
  - U: Undo last point
  - C: Calculate transformation
  - S: Save and exit
  - ESC: Exit without saving

Transformed Image Window:
  - Right Drag: Pan the image
  - Ctrl + Wheel: Zoom in/out

Workflow:
---------
1. Run this script
2. Use Ctrl+Wheel to zoom and Right-drag to pan for better view
3. Click on the 1st point on the y-axis of the new coordinate system
4. Click on the 2nd point on the y-axis of the new coordinate system
5. Press 'U' if you want to undo the last point
6. Press 'C' to calculate and preview the transformation
7. Press 'S' to save the transformed image and rotation matrix
8. Press ESC to exit

Output:
-------
- Transformed image: [basename]_transformed.jpg
- Rotation matrix: [basename]_rotation_matrix.npy

For integration in larger pipelines, you can access:
- tool.transformed_image (numpy array)
- tool.rotation_matrix (2x3 affine transformation matrix)
"""

from ImageRotationTool import ImageRotationTool
from config import BINARY_MASKS_PATH


# ========== Configuration ==========
image_path = str(BINARY_MASKS_PATH / "Barrel_Images2_Masked" / "S2_Back.jpg")

# Optional: change save directory (default is Barrel_Images2_croped_NewAxes)
save_dir = str(BINARY_MASKS_PATH / "Barrel_Images2_croped_NewAxes")

# Optional: change fill value for empty regions (default is 0=black)
# Examples: 0=black, 255=white, 128=gray
fill_value = 0
# ===================================


if __name__ == "__main__":
    # Create and run tool
    tool = ImageRotationTool(image_path, save_dir=save_dir, fill_value=fill_value)
    tool.run()

    # Access results if needed
    if tool.transformed_image is not None:
        print("\nTransformation complete!")
        print(f"Results available in: tool.transformed_image, tool.rotation_matrix")
