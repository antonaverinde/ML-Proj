"""
Interactive Image Merging Tool with Correspondence Point Selection

This tool allows you to select correspondence points on two images to merge them
using homography transformation. The tool preserves defect regions (black pixels)
from the second image.

Features:
---------
- Interactive point selection with visual feedback
- Zoom in/out with Ctrl + Mouse Wheel
- Pan image with Right Mouse Button
- Undo point selection with 'U' key
- Numbered points for easy tracking
- Automatic defect mask preservation
- Preview merged result before saving
- Saves transformation matrix for later use

Controls:
---------
Selection Windows:
  - Left Click: Select correspondence point
  - Right Drag: Pan the image
  - Ctrl + Wheel: Zoom in/out
  - U: Undo last point
  - Q: Finish point selection

Preview Window:
  - S: Save merged image and transformation matrix
  - ESC: Exit without saving

Workflow:
---------
1. Select correspondence points on Image 1 (Front)
2. Select matching points on Image 2 (Back) in the same order
3. Preview the merged result
4. Press 'S' to save or ESC to exit

Output:
-------
- Merged image: [basename]_Merged.jpg
- Transformation matrix: [basename]_homography_matrix.npy
- Defect regions from Image 2 are preserved as black pixels
"""

import cv2
import numpy as np
import os


class ImageMergeTool:
    """
    Interactive tool for merging two images using correspondence points.

    Parameters:
    -----------
    image1_path : str
        Path to the first image (Front)
    image2_path : str, optional
        Path to the second image (Back). If None, derived from image1_path
        by replacing "Front" with "Back"
    save_dir : str, optional
        Directory to save merged images. If None, derived from image1_path
    flip_image2 : bool, optional
        Whether to flip image2 horizontally (default: True)

    Usage:
    ------
    tool = ImageMergeTool(image1_path)
    tool.run()
    # Access results: tool.merged_image, tool.homography_matrix
    """

    def __init__(self, image1_path, image2_path=None, save_dir=None, flip_image2=True):
        self.image1_path = image1_path
        self.flip_image2 = flip_image2

        # Derive image2_path if not provided
        if image2_path is None:
            self.image2_path = image1_path.replace("Front", "Back")
        else:
            self.image2_path = image2_path

        # Derive save directory if not provided
        if save_dir is None:
            # Replace "Transformed" with "Merged" in path
            path_parts = os.path.dirname(image1_path)
            self.save_dir = path_parts.replace("Transformed", "Merged")
        else:
            self.save_dir = save_dir

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Load images
        self.image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        self.image2 = cv2.imread(self.image2_path, cv2.IMREAD_GRAYSCALE)

        if self.image1 is None:
            raise FileNotFoundError(f"Image 1 not found at: {image1_path}")
        if self.image2 is None:
            raise FileNotFoundError(f"Image 2 not found at: {self.image2_path}")

        # Flip image2 if requested
        if self.flip_image2:
            self.image2 = cv2.flip(self.image2, 1)

        # Create defect mask from image2 (1 where pixels are NOT black, 0 where black)
        self.defect_mask = (self.image2 != 0).astype(np.uint8)

        # State for image1 selection
        self.points1 = []
        self.zoom1 = 1.0
        self.pan1 = [0, 0]
        self.is_panning1 = False
        self.pan_start1 = None
        self.mouse_pos1 = None

        # State for image2 selection
        self.points2 = []
        self.zoom2 = 1.0
        self.pan2 = [0, 0]
        self.is_panning2 = False
        self.pan_start2 = None
        self.mouse_pos2 = None

        # Results
        self.homography_matrix = None
        self.merged_image = None
        self.aligned_image2 = None

        # Display settings
        self.canvas_size = (800, 800)

    def _get_display_coordinates(self, x, y, zoom, offset):
        """Convert screen coordinates to image coordinates"""
        img_x = int((x - offset[0]) / zoom)
        img_y = int((y - offset[1]) / zoom)
        return img_x, img_y

    def _get_screen_coordinates(self, img_x, img_y, zoom, offset):
        """Convert image coordinates to screen coordinates"""
        x = int(img_x * zoom + offset[0])
        y = int(img_y * zoom + offset[1])
        return x, y

    def _update_display(self, image, points, zoom, pan, window_name, mouse_pos=None, reference_points=None):
        """Update the display window with current state"""
        h, w = image.shape[:2]
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas_h, canvas_w = self.canvas_size
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        x_start = int(max(0, -pan[0]))
        y_start = int(max(0, -pan[1]))
        x_end = int(min(new_w, canvas_w - pan[0]))
        y_end = int(min(new_h, canvas_h - pan[1]))

        paste_x = int(max(0, pan[0]))
        paste_y = int(max(0, pan[1]))

        if x_end > x_start and y_end > y_start:
            paste_w = int(x_end - x_start)
            paste_h = int(y_end - y_start)
            canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = zoomed[y_start:y_end, x_start:x_end]

        # Convert to BGR for colored annotations
        display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Draw selected points with numbers
        for i, (img_x, img_y) in enumerate(points):
            screen_x, screen_y = self._get_screen_coordinates(img_x, img_y, zoom, pan)
            if 0 <= screen_x < canvas_w and 0 <= screen_y < canvas_h:
                cv2.circle(display, (screen_x, screen_y), 5, (0, 255, 0), -1)
                cv2.putText(display, f"{i+1}", (screen_x + 10, screen_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw reference points if provided (for image2 window)
        if reference_points is not None:
            for i, (img_x, img_y) in enumerate(reference_points):
                screen_x, screen_y = self._get_screen_coordinates(img_x, img_y, zoom, pan)
                if 0 <= screen_x < canvas_w and 0 <= screen_y < canvas_h:
                    cv2.circle(display, (screen_x, screen_y), 8, (255, 0, 0), 2)
                    cv2.putText(display, f"{i+1}", (screen_x + 10, screen_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Instructions
        instructions = [
            "Left Click: Select point",
            "Right Drag: Pan",
            "Ctrl+Wheel: Zoom",
            "U: Undo last point",
            "Q: Finish selection"
        ]
        y_pos = 20
        for inst in instructions:
            cv2.putText(display, inst, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 0), 1, cv2.LINE_AA)
            y_pos += 15

        # Zoom level
        cv2.putText(display, f"Zoom: {zoom:.2f}x", (10, canvas_h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Point count
        cv2.putText(display, f"Points: {len(points)}", (10, canvas_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)

    def _mouse_callback_image1(self, event, x, y, flags, param):
        """Mouse callback for image1 window"""
        self.mouse_pos1 = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning1 = True
            self.pan_start1 = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning1 = False
            self.pan_start1 = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning1 and self.pan_start1 is not None:
                dx = x - self.pan_start1[0]
                dy = y - self.pan_start1[1]
                self.pan1[0] += dx
                self.pan1[1] += dy
                self.pan_start1 = (x, y)
                self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Front")
        elif event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self._get_display_coordinates(x, y, self.zoom1, self.pan1)
            h, w = self.image1.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.points1.append((img_x, img_y))
                print(f"Image 1 - Point {len(self.points1)}: ({img_x}, {img_y})")
                self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Front")
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom1
            if flags > 0:
                self.zoom1 *= 1.1
            else:
                self.zoom1 /= 1.1
            self.zoom1 = max(0.1, min(self.zoom1, 10.0))

            zoom_ratio = self.zoom1 / old_zoom
            self.pan1[0] = x - (x - self.pan1[0]) * zoom_ratio
            self.pan1[1] = y - (y - self.pan1[1]) * zoom_ratio
            self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Front")

    def _mouse_callback_image2(self, event, x, y, flags, param):
        """Mouse callback for image2 window"""
        self.mouse_pos2 = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning2 = True
            self.pan_start2 = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning2 = False
            self.pan_start2 = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning2 and self.pan_start2 is not None:
                dx = x - self.pan_start2[0]
                dy = y - self.pan_start2[1]
                self.pan2[0] += dx
                self.pan2[1] += dy
                self.pan_start2 = (x, y)
                self._update_display(self.image2, self.points2, self.zoom2, self.pan2,
                                   "Image 2 - Back", reference_points=self.points1)
        elif event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self._get_display_coordinates(x, y, self.zoom2, self.pan2)
            h, w = self.image2.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.points2.append((img_x, img_y))
                print(f"Image 2 - Point {len(self.points2)}: ({img_x}, {img_y})")
                self._update_display(self.image2, self.points2, self.zoom2, self.pan2,
                                   "Image 2 - Back", reference_points=self.points1)
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom2
            if flags > 0:
                self.zoom2 *= 1.1
            else:
                self.zoom2 /= 1.1
            self.zoom2 = max(0.1, min(self.zoom2, 10.0))

            zoom_ratio = self.zoom2 / old_zoom
            self.pan2[0] = x - (x - self.pan2[0]) * zoom_ratio
            self.pan2[1] = y - (y - self.pan2[1]) * zoom_ratio
            self._update_display(self.image2, self.points2, self.zoom2, self.pan2,
                               "Image 2 - Back", reference_points=self.points1)

    def _select_points_image1(self):
        """Interactive point selection for image1"""
        print("\n" + "="*60)
        print("SELECT CORRESPONDENCE POINTS ON IMAGE 1 (FRONT)")
        print("="*60)

        cv2.namedWindow("Image 1 - Front", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image 1 - Front", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("Image 1 - Front", self._mouse_callback_image1)
        self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Front")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                if len(self.points1) < 4:
                    print(f"Warning: Only {len(self.points1)} points selected. Need at least 4 for homography.")
                break
            elif key == ord('u') or key == ord('U'):
                if len(self.points1) > 0:
                    removed = self.points1.pop()
                    print(f"Undone point {len(self.points1)+1}: {removed}")
                    self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Front")

        cv2.destroyWindow("Image 1 - Front")
        print(f"Selected {len(self.points1)} points on Image 1")

    def _select_points_image2(self):
        """Interactive point selection for image2"""
        print("\n" + "="*60)
        print("SELECT MATCHING POINTS ON IMAGE 2 (BACK)")
        print("="*60)
        print("Blue circles show reference points from Image 1")

        cv2.namedWindow("Image 2 - Back", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image 2 - Back", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("Image 2 - Back", self._mouse_callback_image2)
        self._update_display(self.image2, self.points2, self.zoom2, self.pan2,
                           "Image 2 - Back", reference_points=self.points1)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                if len(self.points2) != len(self.points1):
                    print(f"Warning: {len(self.points2)} points on Image 2, but {len(self.points1)} on Image 1")
                break
            elif key == ord('u') or key == ord('U'):
                if len(self.points2) > 0:
                    removed = self.points2.pop()
                    print(f"Undone point {len(self.points2)+1}: {removed}")
                    self._update_display(self.image2, self.points2, self.zoom2, self.pan2,
                                       "Image 2 - Back", reference_points=self.points1)

        cv2.destroyWindow("Image 2 - Back")
        print(f"Selected {len(self.points2)} points on Image 2")

    def _compute_merge(self):
        """Compute homography and merge images"""
        if len(self.points1) < 4 or len(self.points2) < 4:
            raise ValueError(f"Need at least 4 points on each image. Got {len(self.points1)} and {len(self.points2)}")

        if len(self.points1) != len(self.points2):
            raise ValueError(f"Point count mismatch: {len(self.points1)} vs {len(self.points2)}")

        points1_array = np.array(self.points1, dtype=np.float32)
        points2_array = np.array(self.points2, dtype=np.float32)

        # Compute homography
        print("\nComputing homography...")
        # self.homography_matrix, status = cv2.findHomography(points2_array, points1_array, cv2.RANSAC)
        self.homography_matrix, status = cv2.findHomography(points2_array, points1_array, 0)
        print("Using bare homography without RANSAC")
        # self.homography_matrix, status = cv2.findHomography(points2_array, points1_array, cv2.LMEDS)
        # self.homography_matrix, status = cv2.findHomography(points2_array, points1_array,
        # cv2.RANSAC, ransacReprojThreshold=5.0)
        if self.homography_matrix is None:
            raise ValueError("Failed to compute homography matrix")

        # Warp image2 to align with image1
        print("Warping Image 2...")
        h, w = self.image1.shape[:2]
        self.aligned_image2 = cv2.warpPerspective(self.image2, self.homography_matrix, (w, h))

        # Warp the defect mask as well
        print("Warping defect mask...")
        aligned_defect_mask = cv2.warpPerspective(self.defect_mask, self.homography_matrix, (w, h))

        # Create a mask to identify valid warped regions (vs empty regions from transformation)
        # Warp a full white mask to see which regions are inside the transformation bounds
        valid_region_mask = np.ones_like(self.image2, dtype=np.uint8) * 255
        aligned_valid_region = cv2.warpPerspective(valid_region_mask, self.homography_matrix, (w, h))
        # Regions with value > threshold are inside the warped bounds
        inside_warp_bounds = (aligned_valid_region > 10).astype(bool)

        # Merge images with gray background for empty regions
        print("Merging images...")

        # Start with gray canvas
        self.merged_image = np.full((h, w), 125, dtype=np.uint8)

        # Create masks for valid data regions
        # Image1 always has data everywhere (assuming it's the base image)
        mask1 = (self.image1 > 0).astype(bool)
        # Image2 has data where it was warped to (non-zero after warping)
        mask2 = (self.aligned_image2 > 0).astype(bool)

        # Create blended image (average of both images)
        blended = cv2.addWeighted(self.image1, 0.5, self.aligned_image2, 0.5, 0)

        # Where both images have data: use blended
        both_mask = mask1 & mask2
        self.merged_image[both_mask] = blended[both_mask]

        # Where only image1 has data: use image1
        only1_mask = mask1 & ~mask2 & inside_warp_bounds
        self.merged_image[only1_mask] = self.image1[only1_mask]

        # Where only image2 has data: use image2
        only2_mask = ~mask1 & mask2
        self.merged_image[only2_mask] = self.aligned_image2[only2_mask]

        # Empty regions (outside warp bounds) already filled with 125 from initialization

        # Apply defect mask: restore black pixels from original image2 position
        # Only where we're inside the warped bounds (not in empty edge regions)
        defect_pixels = (aligned_defect_mask == 0) & inside_warp_bounds
        self.merged_image[defect_pixels] = 0

        print("Merge complete!")
        print(f"Preserved {np.sum(aligned_defect_mask == 0)} defect pixels")

    def _preview_result(self):
        """Show preview of merged result"""
        print("\n" + "="*60)
        print("PREVIEW MERGED RESULT")
        print("="*60)
        print("Press 'S' to save, ESC to exit without saving")

        # Create side-by-side comparison
        h, w = self.image1.shape[:2]

        # Resize for display if too large
        max_display_width = 1800
        if w * 3 > max_display_width:
            scale = max_display_width / (w * 3)
            display_w = int(w * scale)
            display_h = int(h * scale)
            img1_display = cv2.resize(self.image1, (display_w, display_h))
            img2_display = cv2.resize(self.aligned_image2, (display_w, display_h))
            merged_display = cv2.resize(self.merged_image, (display_w, display_h))
        else:
            img1_display = self.image1
            img2_display = self.aligned_image2
            merged_display = self.merged_image

        # Stack horizontally
        comparison = np.hstack([img1_display, img2_display, merged_display])

        # Convert to BGR for display
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)

        # Add labels
        h_comp, w_comp = comparison_bgr.shape[:2]
        w_third = w_comp // 3
        cv2.putText(comparison_bgr, "Image 1 (Front)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(comparison_bgr, "Image 2 (Back Aligned)", (w_third + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(comparison_bgr, "Merged Result", (2*w_third + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.namedWindow("Merge Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Merge Preview", comparison_bgr)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Exiting without saving...")
                cv2.destroyAllWindows()
                return False
            elif key == ord('s') or key == ord('S'):
                cv2.destroyAllWindows()
                return True

    def save(self, output_path=None):
        """Save merged image and transformation matrix"""
        if self.merged_image is None:
            raise ValueError("No merged image to save. Run the tool first.")

        # Generate output path if not provided
        if output_path is None:
            basename = os.path.splitext(os.path.basename(self.image1_path))[0]
            # Remove "_Front_Transformed" and replace with "_Merged"
            basename = basename.replace("_Front_Transformed", "").replace("_Front", "")
            output_path = os.path.join(self.save_dir, f"{basename}_Merged.jpg")
            matrix_path = os.path.join(self.save_dir, f"{basename}_homography_matrix.npy")
        else:
            matrix_path = output_path.replace(".jpg", "_homography_matrix.npy")

        # Save merged image
        cv2.imwrite(output_path, self.merged_image)
        print(f"Merged image saved to: {output_path}")

        # Save homography matrix
        np.save(matrix_path, self.homography_matrix)
        print(f"Homography matrix saved to: {matrix_path}")

        return output_path, matrix_path

    def run(self):
        """Run the interactive merging tool"""
        print("="*60)
        print("INTERACTIVE IMAGE MERGE TOOL")
        print("="*60)
        print(f"Image 1 (Front): {self.image1_path}")
        print(f"Image 2 (Back):  {self.image2_path}")
        print(f"Image 2 flipped: {self.flip_image2}")
        print(f"Save directory:  {self.save_dir}")
        print("="*60)

        # Step 1: Select points on image1
        self._select_points_image1()

        if len(self.points1) < 4:
            print("Error: Need at least 4 points. Exiting.")
            return

        # Step 2: Select corresponding points on image2
        self._select_points_image2()

        if len(self.points2) < 4:
            print("Error: Need at least 4 points. Exiting.")
            return

        # Step 3: Compute merge
        try:
            self._compute_merge()
        except Exception as e:
            print(f"Error during merge: {e}")
            return

        # Step 4: Preview and save
        if self._preview_result():
            self.save()
            print("\n" + "="*60)
            print("MERGE COMPLETE!")
            print("="*60)
        else:
            print("Merge cancelled by user")
