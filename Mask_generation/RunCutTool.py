"""
Regional Crop Tool for IR Images

This module provides functionality to interactively select and crop regions
from merged IR images and apply the crop to all associated data files.

Classes:
--------
RegionCutTool : Main class for interactive region cropping
"""

import cv2
import numpy as np
import os
import glob
import shutil

#from run_cut_regions import SUBPATH


class RegionCutTool:
    """
    Interactive tool for selecting crop regions and applying them to IR data.

    Workflow:
    1. Displays merged image, Amp0, and Amp1 side by side
    2. User selects rectangular crop region on merged image
    3. On save (press 's'), crops all npy/npz/jpg files in location folders
    4. Saves cropped data to 'cutted_V1' subfolder

    Parameters:
    -----------
    location_folders : list
        List of folder paths for this location (e.g., different power/side modes)
    location_name : str
        Name of the location (e.g., 'bottom_left')
    npz_filename : str, optional
        Name of NPZ file to load (default: "PPT_a=0_width=280.npz")
    """

    def __init__(self, location_folders, location_name, npz_filename="PPT_a=0_width=280.npz",SUBPATH=""):
        """
        Initialize the Region Cut Tool.

        Parameters:
        -----------
        location_folders : list
            List of folder paths for this location
        location_name : str
            Name of the location
        npz_filename : str, optional
            Name of NPZ file to load
        """
        self.location_folders = location_folders
        self.location_name = location_name
        self.npz_filename = npz_filename
        self.crop_coords = None  # Will store (x1, y1, x2, y2)
        self.merged_image = None
        self.merged_image_path = None
        self.amp0_image = None
        self.amp1_image = None

        # Rectangle selection state
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.SUBPATH=SUBPATH
        # Display settings
        self.display_scale = 1.5  # Scale factor for display

    def _normalize_to_uint8(self, data, percentile_clip=True, low_percentile=5, high_percentile=95):
        """
        Normalize float data to 0-255 uint8 range with optional percentile clipping.

        Parameters:
        -----------
        data : numpy array
            Input data to normalize
        percentile_clip : bool
            If True, use percentiles instead of min/max to clip outliers
        low_percentile : float
            Lower percentile for clipping
        high_percentile : float
            Upper percentile for clipping
        """
        # Handle NaN and inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if percentile_clip:
            # Use percentiles to clip outliers and improve contrast
            data_min = np.percentile(data, low_percentile)
            data_max = np.percentile(data, high_percentile)
        else:
            data_min = data.min()
            data_max = data.max()

        if data_max - data_min > 1e-8:
            # Clip to remove outliers and normalize to 0-255
            data_clipped = np.clip(data, data_min, data_max)
            normalized = ((data_clipped - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(data, dtype=np.uint8)

        return normalized

    def _enhance_contrast(self, image):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement.

        Parameters:
        -----------
        image : numpy array (uint8)
            Grayscale image to enhance

        Returns:
        --------
        Enhanced grayscale image
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def find_merged_image(self):
        """
        Find the merged_image.jpg in one of the location folders.

        Returns:
        --------
        str : Path to merged_image.jpg or None if not found
        """
        for folder in self.location_folders:
            # Look for merged_image.jpg or similar
            possible_names = [
                "merged_image.jpg",
                "Merged_image.jpg",
                "merged.jpg",
                "Merged.jpg"
            ]

            for name in possible_names:
                img_path = os.path.join(folder, name)
                if os.path.exists(img_path):
                    return img_path

        # If not found, look for any jpg file as fallback
        for folder in self.location_folders:
            jpg_files = glob.glob(os.path.join(folder, "*.jpg"))
            if jpg_files:
                return jpg_files[0]

        return None

    def find_and_load_npz(self):
        """
        Find and load NPZ file containing Amp data.

        Returns:
        --------
        tuple : (amp0_normalized, amp1_normalized) or (None, None) if not found
        """
        for folder in self.location_folders:
            if self.SUBPATH!="":
                base_folder = os.path.dirname(folder)
                npz_path = os.path.join(base_folder, self.npz_filename)
            else:
                npz_path = os.path.join(folder, self.npz_filename)
            if os.path.exists(npz_path):
                print(f"  Loading NPZ: {npz_path}")

                # Load NPZ and extract amplitude
                npz_data = np.load(npz_path)
                if 'Amp' not in npz_data:
                    print(f"  Warning: 'Amp' not found in NPZ file")
                    continue

                amp_data = npz_data['Amp']
                if amp_data.ndim != 3:
                    print(f"  Warning: Expected 3D amplitude data, got shape: {amp_data.shape}")
                    continue

                # Extract Amp[:,:,0] and Amp[:,:,1]
                amp0_raw = amp_data[:, :, 0]
                amp0_norm = self._normalize_to_uint8(amp0_raw, percentile_clip=True,
                                                      low_percentile=5, high_percentile=95)
                amp0_norm = self._enhance_contrast(amp0_norm)

                if amp_data.shape[2] > 1:
                    amp1_raw = amp_data[:, :, 1]
                    amp1_norm = self._normalize_to_uint8(amp1_raw, percentile_clip=True,
                                                          low_percentile=5, high_percentile=95)
                    amp1_norm = self._enhance_contrast(amp1_norm)
                else:
                    amp1_norm = amp0_norm.copy()

                print(f"  ✓ Loaded Amp0: {amp0_norm.shape}, Amp1: {amp1_norm.shape}")
                return amp0_norm, amp1_norm

        return None, None

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for interactive rectangle selection on merged image.
        Coordinates are mapped to the merged image section only.

        Parameters:
        -----------
        event : int
            OpenCV mouse event type
        x, y : int
            Mouse cursor coordinates
        flags : int
            OpenCV event flags
        param : any
            Additional parameters (unused)
        """
        # Only allow selection on the merged image (first third of display)
        h_scaled = int(self.merged_image.shape[0] * self.display_scale)
        w_scaled = int(self.merged_image.shape[1] * self.display_scale)

        # Check if click is in merged image area (first third)
        if x > w_scaled:
            return  # Ignore clicks outside merged image area

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing rectangle
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            # Update rectangle while dragging
            if self.drawing:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing rectangle
            self.drawing = False
            self.end_point = (x, y)

            # Store coordinates in scaled display space
            x1_disp, x2_disp = sorted([self.start_point[0], self.end_point[0]])
            y1_disp, y2_disp = sorted([self.start_point[1], self.end_point[1]])

            # Convert to original image coordinates
            x1 = int(x1_disp / self.display_scale)
            y1 = int(y1_disp / self.display_scale)
            x2 = int(x2_disp / self.display_scale)
            y2 = int(y2_disp / self.display_scale)

            # Clamp to image boundaries
            x1 = max(0, min(x1, self.merged_image.shape[1]))
            y1 = max(0, min(y1, self.merged_image.shape[0]))
            x2 = max(0, min(x2, self.merged_image.shape[1]))
            y2 = max(0, min(y2, self.merged_image.shape[0]))

            self.crop_coords = (x1, y1, x2, y2)

            print(f"\nSelected region (original coords): ({x1}, {y1}) to ({x2}, {y2})")
            print(f"Width: {x2-x1}, Height: {y2-y1}")

    def select_crop_region(self):
        """
        Display merged image, Amp0, and Amp1 side by side for crop region selection.

        Returns:
        --------
        bool : True if region selected successfully, False if cancelled
        """
        # Find and load merged image
        self.merged_image_path = self.find_merged_image()

        if self.merged_image_path is None:
            print(f"Error: No merged image found in location '{self.location_name}'")
            return False

        print(f"\nLoading merged image: {self.merged_image_path}")
        self.merged_image = cv2.imread(self.merged_image_path, cv2.IMREAD_GRAYSCALE)

        if self.merged_image is None:
            print(f"Error: Could not load image from {self.merged_image_path}")
            return False

        # Load NPZ data for Amp0 and Amp1
        self.amp0_image, self.amp1_image = self.find_and_load_npz()

        if self.amp0_image is None or self.amp1_image is None:
            print(f"Warning: Could not load Amp data, displaying merged image only")
            self.amp0_image = np.zeros_like(self.merged_image)
            self.amp1_image = np.zeros_like(self.merged_image)

        # Create window and set mouse callback
        window_name = f"Select Crop Region - {self.location_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "="*70)
        print(f"CROP REGION SELECTION - {self.location_name}")
        print("="*70)
        print("Instructions:")
        print("  - Click and drag on MERGED IMAGE (left panel) to select crop region")
        print("  - Press 'S' to save and apply crop to all files")
        print("  - Press 'R' to reset selection")
        print("  - Press 'ESC' to cancel and skip this location")
        print("="*70)

        while True:
            # Scale images for display
            h, w = self.merged_image.shape
            h_scaled = int(h * self.display_scale)
            w_scaled = int(w * self.display_scale)

            # Resize all images
            merged_display = cv2.resize(self.merged_image, (w_scaled, h_scaled))
            amp0_display = cv2.resize(self.amp0_image, (w_scaled, h_scaled))
            amp1_display = cv2.resize(self.amp1_image, (w_scaled, h_scaled))

            # Convert to BGR for display
            merged_bgr = cv2.cvtColor(merged_display, cv2.COLOR_GRAY2BGR)
            amp0_colored = cv2.applyColorMap(amp0_display, cv2.COLORMAP_TURBO)
            amp1_colored = cv2.applyColorMap(amp1_display, cv2.COLORMAP_TURBO)

            # Stack horizontally: Merged | Amp0 | Amp1
            composite = np.hstack([merged_bgr, amp0_colored, amp1_colored])

            # Add labels
            cv2.putText(composite, "Merged Image (SELECT HERE)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(composite, "Amp0 (5-95)", (w_scaled + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(composite, "Amp1 (5-95)", (2*w_scaled + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw current rectangle on merged image section
            if self.start_point and self.end_point:
                cv2.rectangle(composite, self.start_point, self.end_point,
                            (0, 255, 0), 3)

                # Show dimensions
                if self.crop_coords:
                    x1, y1, x2, y2 = self.crop_coords
                    # Display in scaled coordinates
                    x1_disp = int(x1 * self.display_scale)
                    y1_disp = int(y1 * self.display_scale)
                    text = f"W:{x2-x1} H:{y2-y1}"
                    cv2.putText(composite, text, (x1_disp, max(20, y1_disp-10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Add instructions at bottom
            h_comp = composite.shape[0]
            cv2.putText(composite, "S=Save | R=Reset | ESC=Cancel", (10, h_comp - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Display composite image
            cv2.imshow(window_name, composite)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                # Save and exit
                if self.crop_coords is not None:
                    print("\n✓ Region selected, saving cropped data...")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("\nWarning: No region selected! Please select a region first.")

            elif key == ord('r') or key == ord('R'):
                # Reset selection
                self.start_point = None
                self.end_point = None
                self.crop_coords = None
                print("\n↻ Selection reset")

            elif key == 27:  # ESC
                # Cancel
                print("\n✗ Cancelled by user (ESC)")
                cv2.destroyAllWindows()
                return False

    def crop_and_save_files(self):
        """
        Crop all npy, npz, and jpg files in location folders and save to cutted_V1.
        Also saves crop coordinates for later use.

        Returns:
        --------
        bool : True if successful, False otherwise
        """
        if self.crop_coords is None:
            print("Error: No crop coordinates defined!")
            return False

        x1, y1, x2, y2 = self.crop_coords

        print("\n" + "="*70)
        print(f"CROPPING AND SAVING FILES")
        print("="*70)
        print(f"Crop region: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"Processing {len(self.location_folders)} folders...")

        success_count = 0

        for folder_path in self.location_folders:
            folder_name = os.path.basename(folder_path)
            print(f"\n► Processing: {folder_name}")

            # Create cutted_V1 subfolder
            output_folder = os.path.join(folder_path, "cutted_V1")
            os.makedirs(output_folder, exist_ok=True)
            print(f"  Output folder: {output_folder}")

            # Save crop coordinates for later use
            crop_coords_path = os.path.join(output_folder, "crop_coordinates.npy")
            np.save(crop_coords_path, np.array([x1, y1, x2, y2], dtype=np.int32))
            print(f"  ✓ crop_coordinates.npy: [{x1}, {y1}, {x2}, {y2}]")

            # Find all relevant files
            npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
            npz_files = glob.glob(os.path.join(folder_path, "*.npz"))
            jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))

            all_files = npy_files + npz_files + jpg_files

            if not all_files:
                print(f"  Warning: No files found to crop!")
                continue

            print(f"  Found {len(all_files)} files to process:")
            print(f"    - {len(npy_files)} .npy files")
            print(f"    - {len(npz_files)} .npz files")
            print(f"    - {len(jpg_files)} .jpg files")

            # Process each file
            for file_path in all_files:
                try:
                    filename = os.path.basename(file_path)
                    output_path = os.path.join(output_folder, filename)

                    if file_path.endswith('.npy'):
                        # Skip homography matrix - we save crop_coordinates.npy instead
                        if filename == 'homography_matrix.npy':
                            print(f"  ↪ {filename}: Skipped (crop_coordinates.npy saved instead)")
                            continue

                        # Crop npy file
                        data = np.load(file_path)
                        # Assuming data is (H, W) or (H, W, C)
                        if data.ndim >= 2:
                            cropped = data[y1:y2, x1:x2]
                            np.save(output_path, cropped)
                            print(f"  ✓ {filename}: {data.shape} → {cropped.shape}")
                        else:
                            # 1D array - copy as-is
                            np.save(output_path, data)
                            print(f"  ↪ {filename}: {data.shape} (not cropped)")

                    elif file_path.endswith('.npz'):
                        # Crop npz file (crop all arrays inside)
                        data = np.load(file_path)
                        cropped_dict = {}

                        for key in data.files:
                            arr = data[key]
                            # Crop first two dimensions (H, W)
                            if arr.ndim >= 2:
                                cropped_dict[key] = arr[y1:y2, x1:x2]
                                print(f"  ✓ {filename}[{key}]: {arr.shape} → {cropped_dict[key].shape}")
                            else:
                                # Keep 1D arrays as-is
                                cropped_dict[key] = arr
                                print(f"  ↪ {filename}[{key}]: {arr.shape} (not cropped)")

                        np.savez(output_path, **cropped_dict)

                    elif file_path.endswith('.jpg'):
                        # Crop jpg file
                        img = cv2.imread(file_path)
                        if img is not None:
                            cropped = img[y1:y2, x1:x2]
                            cv2.imwrite(output_path, cropped)
                            print(f"  ✓ {filename}: {img.shape} → {cropped.shape}")
                        else:
                            print(f"  ✗ {filename}: Could not load image")

                except Exception as e:
                    print(f"  ✗ {filename}: Error - {e}")
                    continue

            success_count += 1

        print("\n" + "="*70)
        print(f"✓ Successfully processed {success_count}/{len(self.location_folders)} folders")
        print("="*70)

        return success_count > 0

    def run(self):
        """
        Main execution method: select region and save cropped data.

        Returns:
        --------
        bool : True if successfully saved, False if cancelled
        """
        # Step 1: Select crop region
        if not self.select_crop_region():
            return False

        # Step 2: Crop and save all files
        if not self.crop_and_save_files():
            return False

        return True


def test_tool():
    """Test function for development"""
    # Example usage with placeholder paths
    test_folders = [
        "/path/to/your/test/folder1",
        "/path/to/your/test/folder2",
    ]

    tool = RegionCutTool(test_folders, "test_location")
    success = tool.run()

    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test cancelled or failed")


if __name__ == "__main__":
    test_tool()
