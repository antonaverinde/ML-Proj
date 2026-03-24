"""
ImprovedMaskTool - Apply homography transformation to masks and save results

This tool:
1. Loads grayscale IR image from a location folder
2. Loads corresponding masks (MaskV2 and Mask_2sDif)
3. Loads homography matrix
4. Applies transformation to masks
5. Saves transformed masks and visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class ImprovedMaskTool:
    """Tool for transforming and saving masks with homography"""

    def __init__(self, sample_number, location_folder_path, mask_base_path):
        """
        Initialize the tool.

        Parameters:
        -----------
        sample_number : int
            Sample number (e.g., 1, 2, 3)
        location_folder_path : str
            Path to specific location folder containing merged_image.jpg and homography_matrix.npy
        mask_base_path : str
            Base path to mask directory (Barrel_Images2_croped_Merged_V2_Mask)
        """
        self.sample_number = sample_number
        self.location_folder = Path(location_folder_path)
        self.mask_base_path = Path(mask_base_path)

        # Define file paths
        self.image_path = self.location_folder / "merged_image.jpg"
        self.homography_path = self.location_folder / "homography_matrix.npy"
        self.mask_path = self.mask_base_path / f"S{sample_number}_Mask.npy"
        self.mask_2sdif_path = self.mask_base_path / f"S{sample_number}_Mask_2sDif.npy"

        # Output paths
        self.output_mask_path = self.location_folder / "MaskV2.npy"
        self.output_mask_2sdif_path = self.location_folder / "MaskV2_2sDiff.npy"
        self.output_plot_path = self.location_folder / "mask_comparison.png"

    def validate_inputs(self):
        """Check if all required input files exist"""
        missing_files = []

        if not self.image_path.exists():
            missing_files.append(f"Image: {self.image_path}")
        if not self.homography_path.exists():
            missing_files.append(f"Homography: {self.homography_path}")
        if not self.mask_path.exists():
            missing_files.append(f"Mask: {self.mask_path}")

        if missing_files:
            raise FileNotFoundError(f"Missing required files:\n" + "\n".join(missing_files))

        return True

    def load_data(self):
        """Load all required data files"""
        print(f"  Loading image: {self.image_path.name}")
        self.img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)

        print(f"  Loading mask: {self.mask_path.name}")
        self.mask = np.load(str(self.mask_path))

        # Try to load 2sDif mask
        self.mask_2sdif = None
        if self.mask_2sdif_path.exists():
            print(f"  Loading 2sDif mask: {self.mask_2sdif_path.name}")
            self.mask_2sdif = np.load(str(self.mask_2sdif_path))
        else:
            print(f"  2sDif mask not found (optional)")

        print(f"  Loading homography matrix: {self.homography_path.name}")
        self.H = np.load(str(self.homography_path))

        print(f"    Image shape: {self.img.shape}")
        print(f"    Mask shape: {self.mask.shape}")

    def transform_masks(self):
        """Apply homography transformation to masks"""
        print("  Applying homography transformation...")

        # Transform main mask
        mask_uint8 = (self.mask * 255).astype(np.uint8)
        transformed_mask = cv2.warpPerspective(
            mask_uint8,
            self.H,
            (self.img.shape[1], self.img.shape[0]),
            flags=cv2.INTER_NEAREST
        )
        self.transformed_mask = (transformed_mask > 127).astype(np.uint8)

        # Transform 2sDif mask if available
        self.transformed_mask_2sdif = None
        if self.mask_2sdif is not None:
            mask_2sdif_uint8 = (self.mask_2sdif * 255).astype(np.uint8)
            transformed_mask_2sdif = cv2.warpPerspective(
                mask_2sdif_uint8,
                self.H,
                (self.img.shape[1], self.img.shape[0]),
                flags=cv2.INTER_NEAREST
            )
            self.transformed_mask_2sdif = (transformed_mask_2sdif > 127).astype(np.uint8)

    def save_masks(self):
        """Save transformed masks to disk"""
        print("  Saving transformed masks...")

        # Save main mask
        np.save(str(self.output_mask_path), self.transformed_mask)
        print(f"    Saved: {self.output_mask_path.name}")

        # Save 2sDif mask if available
        if self.transformed_mask_2sdif is not None:
            np.save(str(self.output_mask_2sdif_path), self.transformed_mask_2sdif)
            print(f"    Saved: {self.output_mask_2sdif_path.name}")

    def create_visualization(self):
        """Create and save matplotlib visualization"""
        print("  Creating visualization...")

        num_cols = 3 if self.transformed_mask_2sdif is not None else 2
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

        # Grayscale image
        axes[0].imshow(self.img, cmap='gray')
        axes[0].set_title('Grayscale Image')
        axes[0].axis('off')

        # Transformed mask
        axes[1].imshow(self.transformed_mask, cmap='gray')
        axes[1].set_title('Transformed Mask')
        axes[1].axis('off')

        # Transformed 2sDif mask if available
        if self.transformed_mask_2sdif is not None:
            axes[2].imshow(self.transformed_mask_2sdif, cmap='gray')
            axes[2].set_title('Transformed Mask 2sDif')
            axes[2].axis('off')

        location_name = self.location_folder.name
        plt.suptitle(f'Sample S{self.sample_number} - {location_name}', fontsize=14)
        plt.tight_layout()

        # Save figure
        plt.savefig(str(self.output_plot_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {self.output_plot_path.name}")

    def process(self):
        """
        Main processing pipeline.

        Returns:
        --------
        bool : True if successful, False otherwise
        """
        try:
            print(f"\nProcessing: {self.location_folder.name}")

            # Validate inputs
            self.validate_inputs()

            # Load data
            self.load_data()

            # Transform masks
            self.transform_masks()

            # Save results
            self.save_masks()

            # Create visualization
            self.create_visualization()

            print("  ✓ Processing complete")
            return True

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def process_single_location(sample_number, location_folder_path, mask_base_path):
    """
    Convenience function to process a single location.

    Parameters:
    -----------
    sample_number : int
        Sample number
    location_folder_path : str
        Path to location folder
    mask_base_path : str
        Base path to mask directory

    Returns:
    --------
    bool : Success status
    """
    tool = ImprovedMaskTool(sample_number, location_folder_path, mask_base_path)
    return tool.process()
