import sys
import os
from CylinderFlattener_V2 import test_transform
from config import BINARY_MASKS_PATH

if __name__ == '__main__':
    # Test 3: Real image with both transforms
    real_image_path = str(BINARY_MASKS_PATH / "Barrel_Images2_croped_NewAxes" / "S2_Front_transformed.jpg")

    if os.path.exists(real_image_path):
        print("\n")
        test_transform(
            image_source=real_image_path,
            f=3000,  # 3000
            R=20,  # 10, 20
            transform_type='inverse',
            save_output=True,
            save_transform_matrix=True  # Save transformation coordinate mapping
        )
    else:
        print(f"Error: Image not found at {real_image_path}")