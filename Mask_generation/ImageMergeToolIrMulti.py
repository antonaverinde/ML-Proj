"""
Interactive Image Merging Tool for IR Data with Scrollable Image Lists

Enhanced version of ImageMergeToolIr that allows scrolling through multiple
Phase and Amplitude images during correspondence point selection.

Key Enhancements:
-----------------
- Scrollable image lists (N/P keys, 1-9 for direct jump)
- Loads Amp[0,1] from PPT_a=0_width=280.npz (NEW!)
- Loads Phase[1,2,3,4,5] from PPT_a=0_width=280.npz
- Loads Phase[1,2] from PPT_a=0_width=110.npz
- Total: 9 IR images available
- Viridis colormap for better visualization
- Adjustable histogram-based intensity thresholding
- Correspondence points persist across all images in the list
- Visible reference window shown during step 1 (NEW!)
- Option to rotate reference image by 180 degrees (NEW!)

Controls:
---------
Step 1 - IR Image Selection:
  Two windows: IR Images (select points) + Visible Reference (view only)

  - Left Click: Select correspondence point (on IR image window)
  - Right Drag: Pan the image (both windows)
  - Ctrl + Wheel: Zoom in/out (both windows)
  - N: Next image in list
  - P: Previous image in list
  - 1-9: Jump to specific image
  - U: Undo last point
  - Q: Finish and proceed to step 2
  - ESC: Cancel and exit

Step 2 - Visible Image Selection:
  Two windows: Visible Image (select points) + IR Reference (view only)

  - Left Click: Select matching point (on visible image window)
  - Right Drag: Pan the image (both windows)
  - Ctrl + Wheel: Zoom in/out (both windows)
  - N: Next IR reference image
  - P: Previous IR reference image
  - 1-9: Jump to specific IR reference image
  - U: Undo last point
  - B: Go back to step 1
  - Q: Finish and show preview
  - ESC: Cancel and exit
"""

import cv2
import numpy as np
import os
import glob


def apply_histogram_threshold(image, nbins=50, low_factor=0.25, high_factor=0.30):
    """
    Apply histogram-based intensity thresholding to enhance image contrast.

    Parameters:
    -----------
    image : numpy.ndarray (uint8)
        Input grayscale image
    nbins : int
        Number of histogram bins
    low_factor : float
        Threshold factor for left side (0.0-1.0)
    high_factor : float
        Threshold factor for right side (0.0-1.0)

    Returns:
    --------
    numpy.ndarray (uint8) : Thresholded and normalized image
    """
    # Compute histogram
    hist, bin_edges = np.histogram(image.flatten(), bins=nbins, range=(0, 256))

    # Find maximum bin count
    max_count = hist.max()

    # Find left threshold: first bin where count >= low_factor * max_count
    low_threshold = 0
    for i in range(len(hist)):
        if hist[i] >= low_factor * max_count:
            low_threshold = int(bin_edges[i])
            break

    # Find right threshold: last bin where count >= high_factor * max_count
    high_threshold = 255
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] >= high_factor * max_count:
            high_threshold = int(bin_edges[i + 1])
            break

    # Ensure valid range
    if high_threshold <= low_threshold:
        high_threshold = low_threshold + 1

    # Clip and normalize to 0-255
    clipped = np.clip(image, low_threshold, high_threshold)
    if high_threshold > low_threshold:
        normalized = ((clipped - low_threshold) / (high_threshold - low_threshold) * 255).astype(np.uint8)
    else:
        normalized = image

    return normalized


class ImageMergeToolIrMulti:
    """
    Interactive tool for merging visible images with IR data using scrollable image lists.

    Parameters:
    -----------
    image1_path : str
        Path to the visible image (e.g., S{i}_Merged.jpg)
    npz_path_280 : str
        Path to PPT_a=0_width=280.npz file
    npz_path_110 : str
        Path to PPT_a=0_width=110.npz file
    flip_image1 : bool, optional
        Whether to flip image1 horizontally (default: False)
    rotate_reference : bool, optional
        Whether to rotate reference image by 180 degrees (default: False)
    use_colormap : bool, optional
        Whether to apply viridis colormap to IR images (default: True)
    histogram_threshold_low : float, optional
        Low factor for histogram thresholding (default: 0.25)
    histogram_threshold_high : float, optional
        High factor for histogram thresholding (default: 0.30)
    histogram_bins : int, optional
        Number of histogram bins (default: 50)

    Images Loaded:
    --------------
    - Amp[0, 1] from PPT_a=0_width=280.npz (2 images)
    - Phase[1, 2, 3, 4, 5] from PPT_a=0_width=280.npz (5 images)
    - Phase[1, 2] from PPT_a=0_width=110.npz (2 images)
    Total: 9 IR images
    """

    def __init__(self, image1_path, npz_path_280, npz_path_110, flip_image1=False,
                 rotate_reference=False, use_colormap=True, histogram_threshold_low=0.25,
                 histogram_threshold_high=0.30, histogram_bins=50, homography_method=0):
        self.image1_path = image1_path
        self.npz_path_280 = npz_path_280
        self.npz_path_110 = npz_path_110
        self.flip_image1 = flip_image1
        self.rotate_reference = rotate_reference
        self.use_colormap = use_colormap
        self.histogram_threshold_low = histogram_threshold_low
        self.histogram_threshold_high = histogram_threshold_high
        self.histogram_bins = histogram_bins
        self.homography_method = homography_method

        # Derive save directory from npz_path_280 - save to Masks_V3 subfolder
        location_folder = os.path.dirname(npz_path_280)
        self.save_dir = os.path.join(location_folder, "Masks_V3")

        # Load image1 (visible image / reference)
        self.image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        if self.image1 is None:
            raise FileNotFoundError(f"Image 1 not found at: {image1_path}")

        # Flip image1 if requested
        if self.flip_image1:
            self.image1 = cv2.flip(self.image1, 1)

        # Rotate image1 by 180 degrees if requested
        if self.rotate_reference:
            self.image1 = cv2.rotate(self.image1, cv2.ROTATE_180)
            print(f"Rotated reference image by 180 degrees")

        # Load all IR images into list
        print("Loading IR images...")
        self.ir_image_list = []
        self.ir_image_names = []

        # Load Amplitude[0,1] from PPT_a=0_width=280.npz (FIRST)
        if os.path.exists(npz_path_280):
            self._load_images_from_npz(npz_path_280, 'Amp', [0, 1], "280_Amp")
        else:
            print(f"Warning: {npz_path_280} not found")

        # Load Phase[1,2,3,4,5] from PPT_a=0_width=280.npz
        if os.path.exists(npz_path_280):
            self._load_images_from_npz(npz_path_280, 'Phase', [1, 2, 3, 4, 5], "280_Phase")
        else:
            print(f"Warning: {npz_path_280} not found")

        # Load Phase[1,2] from PPT_a=0_width=110.npz
        if os.path.exists(npz_path_110):
            self._load_images_from_npz(npz_path_110, 'Phase', [1, 2], "110_Phase")
        else:
            print(f"Warning: {npz_path_110} not found")

        if len(self.ir_image_list) == 0:
            raise ValueError("No IR images loaded!")

        print(f"Loaded {len(self.ir_image_list)} IR images total")

        # Use first image for defect mask
        self.image2 = self.ir_image_list[0]
        self.defect_mask = (self.image2 != 0).astype(np.uint8)

        # Current image indices for step 1 and step 2
        self.current_ir_idx = 0

        # State for image1 selection
        self.points1 = []
        self.zoom1 = 1.0
        self.pan1 = [0, 0]
        self.is_panning1 = False
        self.pan_start1 = None

        # State for IR image selection (step 1)
        self.points2 = []
        self.zoom_ir = 1.0
        self.pan_ir = [0, 0]
        self.is_panning_ir = False
        self.pan_start_ir = None

        # State for IR reference during image1 selection (step 2)
        self.current_ir_ref_idx = 0
        self.zoom_ir_ref = 1.0
        self.pan_ir_ref = [0, 0]
        self.is_panning_ir_ref = False
        self.pan_start_ir_ref = None

        # State for visible reference during IR selection (step 1)
        self.zoom_visible_ref = 1.0
        self.pan_visible_ref = [0, 0]
        self.is_panning_visible_ref = False
        self.pan_start_visible_ref = None

        # Results
        self.homography_matrix = None
        self.merged_image = None
        self.binary_mask = None
        self.aligned_image1 = None

        # Display settings
        self.canvas_size = (800, 800)

    def _load_images_from_npz(self, npz_path, data_key, indices, name_prefix):
        """Load specific channels from NPZ file and add to image list."""
        npz_data = np.load(npz_path)
        if data_key not in npz_data:
            print(f"Warning: '{data_key}' not found in {npz_path}")
            return

        data = npz_data[data_key]
        if data.ndim != 3:
            print(f"Warning: Expected 3D data, got shape: {data.shape}")
            return

        for idx in indices:
            if idx >= data.shape[2]:
                print(f"Warning: Index {idx} out of range for shape {data.shape}")
                continue

            # Extract channel
            channel = data[:, :, idx]

            # Normalize to uint8 with percentile clipping
            channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
            p_min = np.percentile(channel, 2)
            p_max = np.percentile(channel, 98)

            if p_max - p_min > 1e-8:
                channel_clipped = np.clip(channel, p_min, p_max)
                channel_norm = ((channel_clipped - p_min) / (p_max - p_min) * 255).astype(np.uint8)
            else:
                channel_norm = np.zeros_like(channel, dtype=np.uint8)

            # Apply histogram-based thresholding
            channel_norm = apply_histogram_threshold(
                channel_norm,
                nbins=self.histogram_bins,
                low_factor=self.histogram_threshold_low,
                high_factor=self.histogram_threshold_high
            )

            self.ir_image_list.append(channel_norm)
            self.ir_image_names.append(f"{name_prefix}[{idx}]")
            print(f"  Loaded {name_prefix}[{idx}]: shape={channel_norm.shape}")

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

    def _update_display(self, image, points, zoom, pan, window_name,
                       current_idx=None, total_images=None, image_name="",
                       show_navigation=True):
        """Update the display window with current state."""
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

        # Apply colormap for IR images
        if self.use_colormap:
            display = cv2.applyColorMap(canvas, cv2.COLORMAP_VIRIDIS)
        else:
            display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Draw selected points with numbers
        for i, (img_x, img_y) in enumerate(points):
            screen_x, screen_y = self._get_screen_coordinates(img_x, img_y, zoom, pan)
            if 0 <= screen_x < canvas_w and 0 <= screen_y < canvas_h:
                cv2.circle(display, (screen_x, screen_y), 5, (0, 255, 0), -1)
                cv2.putText(display, f"{i+1}", (screen_x + 10, screen_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Instructions
        y_pos = 20
        if current_idx is not None and total_images is not None:
            cv2.putText(display, f"Image {current_idx+1}/{total_images}: {image_name}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            y_pos += 18

        instructions = [
            "Left Click: Select point",
            "Right Drag: Pan | Ctrl+Wheel: Zoom",
        ]

        if show_navigation:
            instructions.append("N: Next | P: Prev | 1-9: Jump")

        instructions.extend([
            "U: Undo | Q: Finish"
        ])

        for inst in instructions:
            cv2.putText(display, inst, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 15

        # Zoom level and point count
        cv2.putText(display, f"Zoom: {zoom:.2f}x | Points: {len(points)}",
                   (10, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)

    def _mouse_callback_ir(self, event, x, y, flags, param):
        """Mouse callback for IR image window (step 1)"""
        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning_ir = True
            self.pan_start_ir = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning_ir = False
            self.pan_start_ir = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning_ir and self.pan_start_ir is not None:
                dx = x - self.pan_start_ir[0]
                dy = y - self.pan_start_ir[1]
                self.pan_ir[0] += dx
                self.pan_ir[1] += dy
                self.pan_start_ir = (x, y)
                self._update_display(self.ir_image_list[self.current_ir_idx], self.points2,
                                   self.zoom_ir, self.pan_ir, "IR Images (Select Points)",
                                   self.current_ir_idx, len(self.ir_image_list),
                                   self.ir_image_names[self.current_ir_idx])
        elif event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self._get_display_coordinates(x, y, self.zoom_ir, self.pan_ir)
            h, w = self.ir_image_list[self.current_ir_idx].shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.points2.append((img_x, img_y))
                print(f"IR Image [{self.ir_image_names[self.current_ir_idx]}] - Point {len(self.points2)}: ({img_x}, {img_y})")
                self._update_display(self.ir_image_list[self.current_ir_idx], self.points2,
                                   self.zoom_ir, self.pan_ir, "IR Images (Select Points)",
                                   self.current_ir_idx, len(self.ir_image_list),
                                   self.ir_image_names[self.current_ir_idx])
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom_ir
            if flags > 0:
                self.zoom_ir *= 1.1
            else:
                self.zoom_ir /= 1.1
            self.zoom_ir = max(0.1, min(self.zoom_ir, 10.0))

            zoom_ratio = self.zoom_ir / old_zoom
            self.pan_ir[0] = x - (x - self.pan_ir[0]) * zoom_ratio
            self.pan_ir[1] = y - (y - self.pan_ir[1]) * zoom_ratio
            self._update_display(self.ir_image_list[self.current_ir_idx], self.points2,
                               self.zoom_ir, self.pan_ir, "IR Images (Select Points)",
                               self.current_ir_idx, len(self.ir_image_list),
                               self.ir_image_names[self.current_ir_idx])

    def _mouse_callback_image1(self, event, x, y, flags, param):
        """Mouse callback for image1 window (step 2)"""
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
                self._update_display(self.image1, self.points1, self.zoom1, self.pan1,
                                   "Image 1 - Visible (Select Points)", show_navigation=False)
        elif event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self._get_display_coordinates(x, y, self.zoom1, self.pan1)
            h, w = self.image1.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.points1.append((img_x, img_y))
                print(f"Image 1 - Point {len(self.points1)}: ({img_x}, {img_y})")
                self._update_display(self.image1, self.points1, self.zoom1, self.pan1,
                                   "Image 1 - Visible (Select Points)", show_navigation=False)
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
            self._update_display(self.image1, self.points1, self.zoom1, self.pan1,
                               "Image 1 - Visible (Select Points)", show_navigation=False)

    def _mouse_callback_ir_ref(self, event, x, y, flags, param):
        """Mouse callback for IR reference window (step 2) - view only"""
        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning_ir_ref = True
            self.pan_start_ir_ref = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning_ir_ref = False
            self.pan_start_ir_ref = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning_ir_ref and self.pan_start_ir_ref is not None:
                dx = x - self.pan_start_ir_ref[0]
                dy = y - self.pan_start_ir_ref[1]
                self.pan_ir_ref[0] += dx
                self.pan_ir_ref[1] += dy
                self.pan_start_ir_ref = (x, y)
                self._update_display(self.ir_image_list[self.current_ir_ref_idx], self.points2,
                                   self.zoom_ir_ref, self.pan_ir_ref, "IR Reference",
                                   self.current_ir_ref_idx, len(self.ir_image_list),
                                   self.ir_image_names[self.current_ir_ref_idx])
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom_ir_ref
            if flags > 0:
                self.zoom_ir_ref *= 1.1
            else:
                self.zoom_ir_ref /= 1.1
            self.zoom_ir_ref = max(0.1, min(self.zoom_ir_ref, 10.0))

            zoom_ratio = self.zoom_ir_ref / old_zoom
            self.pan_ir_ref[0] = x - (x - self.pan_ir_ref[0]) * zoom_ratio
            self.pan_ir_ref[1] = y - (y - self.pan_ir_ref[1]) * zoom_ratio
            self._update_display(self.ir_image_list[self.current_ir_ref_idx], self.points2,
                               self.zoom_ir_ref, self.pan_ir_ref, "IR Reference",
                               self.current_ir_ref_idx, len(self.ir_image_list),
                               self.ir_image_names[self.current_ir_ref_idx])

    def _mouse_callback_visible_ref(self, event, x, y, flags, param):
        """Mouse callback for visible reference window (step 1) - view only"""
        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning_visible_ref = True
            self.pan_start_visible_ref = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning_visible_ref = False
            self.pan_start_visible_ref = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning_visible_ref and self.pan_start_visible_ref is not None:
                dx = x - self.pan_start_visible_ref[0]
                dy = y - self.pan_start_visible_ref[1]
                self.pan_visible_ref[0] += dx
                self.pan_visible_ref[1] += dy
                self.pan_start_visible_ref = (x, y)
                # Display visible reference (grayscale, no colormap)
                self._update_visible_reference()
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom_visible_ref
            if flags > 0:
                self.zoom_visible_ref *= 1.1
            else:
                self.zoom_visible_ref /= 1.1
            self.zoom_visible_ref = max(0.1, min(self.zoom_visible_ref, 10.0))

            zoom_ratio = self.zoom_visible_ref / old_zoom
            self.pan_visible_ref[0] = x - (x - self.pan_visible_ref[0]) * zoom_ratio
            self.pan_visible_ref[1] = y - (y - self.pan_visible_ref[1]) * zoom_ratio
            self._update_visible_reference()

    def _update_visible_reference(self):
        """Update the visible reference window (grayscale, no colormap)."""
        h, w = self.image1.shape[:2]
        new_w = int(w * self.zoom_visible_ref)
        new_h = int(h * self.zoom_visible_ref)
        zoomed = cv2.resize(self.image1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas_h, canvas_w = self.canvas_size
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        x_start = int(max(0, -self.pan_visible_ref[0]))
        y_start = int(max(0, -self.pan_visible_ref[1]))
        x_end = int(min(new_w, canvas_w - self.pan_visible_ref[0]))
        y_end = int(min(new_h, canvas_h - self.pan_visible_ref[1]))

        paste_x = int(max(0, self.pan_visible_ref[0]))
        paste_y = int(max(0, self.pan_visible_ref[1]))

        if x_end > x_start and y_end > y_start:
            paste_w = int(x_end - x_start)
            paste_h = int(y_end - y_start)
            canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = zoomed[y_start:y_end, x_start:x_end]

        # Convert to BGR for display (grayscale, no colormap)
        display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Add instructions
        instructions = [
            "Visible Reference (View Only)",
            "Right Drag: Pan",
            "Ctrl+Wheel: Zoom"
        ]
        y_pos = 20
        for inst in instructions:
            cv2.putText(display, inst, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 15

        # Zoom level
        cv2.putText(display, f"Zoom: {self.zoom_visible_ref:.2f}x",
                   (10, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Visible Reference", display)

    def _navigate_ir_image(self, new_idx):
        """Navigate to a different IR image"""
        if 0 <= new_idx < len(self.ir_image_list):
            self.current_ir_idx = new_idx
            self._update_display(self.ir_image_list[self.current_ir_idx], self.points2,
                               self.zoom_ir, self.pan_ir, "IR Images (Select Points)",
                               self.current_ir_idx, len(self.ir_image_list),
                               self.ir_image_names[self.current_ir_idx])

    def _navigate_ir_ref_image(self, new_idx):
        """Navigate to a different IR reference image"""
        if 0 <= new_idx < len(self.ir_image_list):
            self.current_ir_ref_idx = new_idx
            self._update_display(self.ir_image_list[self.current_ir_ref_idx], self.points2,
                               self.zoom_ir_ref, self.pan_ir_ref, "IR Reference",
                               self.current_ir_ref_idx, len(self.ir_image_list),
                               self.ir_image_names[self.current_ir_ref_idx])

    def _select_points_ir(self):
        """Interactive point selection on scrollable IR images (step 1)."""
        print("\n" + "="*60)
        print("SELECT CORRESPONDENCE POINTS ON IR IMAGES")
        print("="*60)
        print(f"Loaded {len(self.ir_image_list)} IR images")
        print("Two windows: IR Images (select points) + Visible Reference (view only)")
        print("Use N/P to scroll through IR images, points persist across all images")
        print("Press 'Q' when done, ESC to cancel")

        # Open IR Images window (for point selection)
        cv2.namedWindow("IR Images (Select Points)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("IR Images (Select Points)", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("IR Images (Select Points)", self._mouse_callback_ir)

        # Open Visible Reference window (view only)
        cv2.namedWindow("Visible Reference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Visible Reference", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("Visible Reference", self._mouse_callback_visible_ref)

        # Initial display
        self._update_display(self.ir_image_list[self.current_ir_idx], self.points2,
                           self.zoom_ir, self.pan_ir, "IR Images (Select Points)",
                           self.current_ir_idx, len(self.ir_image_list),
                           self.ir_image_names[self.current_ir_idx])
        self._update_visible_reference()

        result = 'continue'
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("ESC pressed - Cancelling...")
                result = 'cancel'
                break
            elif key == ord('q') or key == ord('Q'):
                if len(self.points2) < 4:
                    print(f"Warning: Only {len(self.points2)} points selected. Need at least 4.")
                result = 'continue'
                break
            elif key == ord('n') or key == ord('N'):
                # Next image
                self._navigate_ir_image(self.current_ir_idx + 1)
            elif key == ord('p') or key == ord('P'):
                # Previous image
                self._navigate_ir_image(self.current_ir_idx - 1)
            elif ord('1') <= key <= ord('9'):
                # Direct jump
                target_idx = key - ord('1')
                self._navigate_ir_image(target_idx)
            elif key == ord('u') or key == ord('U'):
                if len(self.points2) > 0:
                    removed = self.points2.pop()
                    print(f"Undone point {len(self.points2)+1}: {removed}")
                    self._update_display(self.ir_image_list[self.current_ir_idx], self.points2,
                                       self.zoom_ir, self.pan_ir, "IR Images (Select Points)",
                                       self.current_ir_idx, len(self.ir_image_list),
                                       self.ir_image_names[self.current_ir_idx])

        cv2.destroyWindow("IR Images (Select Points)")
        cv2.destroyWindow("Visible Reference")
        print(f"Selected {len(self.points2)} points on IR images")
        return result

    def _select_points_image1(self):
        """Interactive point selection on image1 with scrollable IR reference (step 2)."""
        print("\n" + "="*60)
        print("SELECT MATCHING POINTS ON VISIBLE IMAGE")
        print("="*60)
        print("IR reference window shows selected points - use N/P to scroll")
        print("Press 'B' to go back, 'Q' when done, ESC to cancel")

        # Open Image 1 window
        cv2.namedWindow("Image 1 - Visible (Select Points)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image 1 - Visible (Select Points)", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("Image 1 - Visible (Select Points)", self._mouse_callback_image1)

        # Open IR reference window
        cv2.namedWindow("IR Reference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("IR Reference", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("IR Reference", self._mouse_callback_ir_ref)

        # Initial display
        self._update_display(self.image1, self.points1, self.zoom1, self.pan1,
                           "Image 1 - Visible (Select Points)", show_navigation=False)
        self._update_display(self.ir_image_list[self.current_ir_ref_idx], self.points2,
                           self.zoom_ir_ref, self.pan_ir_ref, "IR Reference",
                           self.current_ir_ref_idx, len(self.ir_image_list),
                           self.ir_image_names[self.current_ir_ref_idx])

        result = 'continue'
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("ESC pressed - Cancelling...")
                result = 'cancel'
                break
            elif key == ord('q') or key == ord('Q'):
                if len(self.points1) != len(self.points2):
                    print(f"Warning: {len(self.points1)} points on Image 1, but {len(self.points2)} on IR")
                result = 'continue'
                break
            elif key == ord('b') or key == ord('B'):
                print("Going back to IR selection...")
                result = 'back'
                break
            elif key == ord('n') or key == ord('N'):
                # Next IR reference image
                self._navigate_ir_ref_image(self.current_ir_ref_idx + 1)
            elif key == ord('p') or key == ord('P'):
                # Previous IR reference image
                self._navigate_ir_ref_image(self.current_ir_ref_idx - 1)
            elif ord('1') <= key <= ord('9'):
                # Direct jump in IR reference
                target_idx = key - ord('1')
                self._navigate_ir_ref_image(target_idx)
            elif key == ord('u') or key == ord('U'):
                if len(self.points1) > 0:
                    removed = self.points1.pop()
                    print(f"Undone point {len(self.points1)+1}: {removed}")
                    self._update_display(self.image1, self.points1, self.zoom1, self.pan1,
                                       "Image 1 - Visible (Select Points)", show_navigation=False)

        cv2.destroyWindow("Image 1 - Visible (Select Points)")
        cv2.destroyWindow("IR Reference")
        print(f"Selected {len(self.points1)} points on Image 1")
        return result

    def _compute_merge(self):
        """Compute homography and merge images at IR resolution"""
        if len(self.points1) < 4 or len(self.points2) < 4:
            raise ValueError(f"Need at least 4 points on each image. Got {len(self.points1)} and {len(self.points2)}")

        if len(self.points1) != len(self.points2):
            raise ValueError(f"Point count mismatch: {len(self.points1)} vs {len(self.points2)}")

        points1_array = np.array(self.points1, dtype=np.float32)
        points2_array = np.array(self.points2, dtype=np.float32)

        # Compute homography using specified method
        print("\nComputing homography...")
        if self.homography_method == 0:
            # Standard method - use all points (least squares)
            print("Using standard method (all points, least squares)")
            self.homography_matrix, status = cv2.findHomography(points1_array, points2_array, 0)
        elif self.homography_method == cv2.RANSAC:
            # RANSAC method
            print("Using RANSAC method (robust, outlier rejection)")
            self.homography_matrix, status = cv2.findHomography(points1_array, points2_array, cv2.RANSAC, 5.0)
        elif self.homography_method == cv2.LMEDS:
            # LMEDS method
            print("Using LMEDS method (robust, least median)")
            self.homography_matrix, status = cv2.findHomography(points1_array, points2_array, cv2.LMEDS)
        elif self.homography_method == cv2.RHO:
            # RHO method
            print("Using RHO method (robust, PROSAC-based)")
            self.homography_matrix, status = cv2.findHomography(points1_array, points2_array, cv2.RHO)
        else:
            # Fallback to standard
            print(f"Unknown method {self.homography_method}, using standard method")
            self.homography_matrix, status = cv2.findHomography(points1_array, points2_array, 0)

        if self.homography_matrix is None:
            raise ValueError("Failed to compute homography matrix")

        # Use first IR image as reference for merge
        h, w = self.ir_image_list[0].shape[:2]
        self.aligned_image1 = cv2.warpPerspective(self.image1, self.homography_matrix, (w, h))

        # Create merged image (simple average for preview)
        self.merged_image = cv2.addWeighted(self.aligned_image1, 0.5, self.ir_image_list[0], 0.5, 0)

        # Create binary mask
        self.binary_mask = (self.merged_image == 0).astype(np.uint8)

        print("Merge complete!")

    def _preview_result(self):
        """Show preview of merged result."""
        print("\n" + "="*60)
        print("PREVIEW MERGED RESULT")
        print("="*60)
        print("Press 'S' to save, 'R' to retry, ESC to cancel")

        # Get Phase[2] from 280.npz (index 3: Amp[0], Amp[1], Phase[1], Phase[2])
        phase2_280 = self.ir_image_list[3]

        # Create merged image of Phase[2] and aligned VIS
        phase2_vis_merged = cv2.addWeighted(self.aligned_image1, 0.5, phase2_280, 0.5, 0)

        # Create comparison with 4 images
        phase2_display = cv2.resize(phase2_280, (300, 300))
        ir_display = cv2.resize(self.ir_image_list[0], (300, 300))
        vis_display = cv2.resize(self.aligned_image1, (300, 300))
        phase2_merged_display = cv2.resize(phase2_vis_merged, (300, 300))

        # Convert to BGR for display
        phase2_colored = cv2.applyColorMap(phase2_display, cv2.COLORMAP_VIRIDIS)
        ir_colored = cv2.applyColorMap(ir_display, cv2.COLORMAP_VIRIDIS)
        vis_bgr = cv2.cvtColor(vis_display, cv2.COLOR_GRAY2BGR)
        phase2_merged_bgr = cv2.cvtColor(phase2_merged_display, cv2.COLOR_GRAY2BGR)

        # Stack 4 images horizontally
        comparison = np.hstack([phase2_colored, ir_colored, vis_bgr, phase2_merged_bgr])

        # Add labels
        cv2.putText(comparison, "Phase[2] (280)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(comparison, "IR Reference", (310, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(comparison, "VIS (Aligned)", (610, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(comparison, "Merged P[2]+VIS", (910, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.namedWindow("Merge Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Merge Preview", comparison)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return 'cancel'
            elif key == ord('s') or key == ord('S'):
                cv2.destroyAllWindows()
                return 'save'
            elif key == ord('r') or key == ord('R'):
                cv2.destroyAllWindows()
                return 'retry'

    def save(self):
        """Save merged image, binary mask, and transformation matrix to Masks_V3 folder"""
        if self.merged_image is None:
            raise ValueError("No merged image to save. Run the tool first.")

        # Create Masks_V3 directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"\nSaving to: {self.save_dir}")

        merged_path = os.path.join(self.save_dir, "merged_image.jpg")
        cv2.imwrite(merged_path, self.merged_image)
        print(f"  [OK] Merged image: merged_image.jpg")

        mask_path = os.path.join(self.save_dir, "binary_mask.npy")
        np.save(mask_path, self.binary_mask)
        print(f"  [OK] Binary mask: binary_mask.npy")

        matrix_path = os.path.join(self.save_dir, "homography_matrix.npy")
        np.save(matrix_path, self.homography_matrix)
        print(f"  [OK] Homography matrix: homography_matrix.npy")

        return merged_path, mask_path, matrix_path

    def apply_transformation_to_siblings(self):
        """
        Apply the computed transformation to sibling NPZ files in the same location.
        Saves to Masks_V3 subfolder in each sibling location.
        """
        if self.homography_matrix is None:
            raise ValueError("No transformation matrix available. Run the tool first.")

        # Get current folder name from npz_path_280
        current_folder = os.path.dirname(self.npz_path_280)
        current_folder_name = os.path.basename(current_folder)
        parent_dir = os.path.dirname(current_folder)

        # Parse folder name to extract location (e.g., "pos3_top_right")
        # Expected format: prefix_power_side_location
        parts = current_folder_name.split('_')

        # Find power indicator (e.g., "4kw", "2kw")
        power_idx = -1
        for i, part in enumerate(parts):
            if 'kw' in part.lower():
                power_idx = i
                break

        if power_idx == -1:
            print(f"Warning: Could not parse location from folder name: {current_folder_name}")
            return []

        # Position part is everything after power and side
        position_parts = parts[power_idx + 2:]
        position_suffix = '_'.join(position_parts)

        # Prefix is everything before power
        prefix_parts = parts[:power_idx]
        prefix = '_'.join(prefix_parts)

        # Define sibling patterns (all three: 4kw_both, 2kw_right, 2kw_left)
        sibling_patterns = [
            f"{prefix}_4kw_both_{position_suffix}",
            f"{prefix}_2kw_right_{position_suffix}",
            f"{prefix}_2kw_left_{position_suffix}"
        ]

        print(f"\n" + "="*70)
        print("APPLYING TRANSFORMATION TO SIBLING LOCATIONS")
        print("="*70)
        print(f"  Source location: {current_folder_name}")
        print(f"  Position suffix: {position_suffix}")

        results = []
        for sibling_pattern in sibling_patterns:
            sibling_folder = os.path.join(parent_dir, sibling_pattern)

            if not os.path.exists(sibling_folder):
                print(f"  [SKIP] Folder not found: {sibling_pattern}")
                continue

            # Check if this is the current folder (already saved)
            if sibling_folder == current_folder:
                print(f"  [SKIP] Current folder: {sibling_pattern}")
                continue

            print(f"\n  Processing: {sibling_pattern}")

            # Load NPZ file from sibling
            npz_file_280 = os.path.join(sibling_folder, "PPT_a=0_width=280.npz")
            if not os.path.exists(npz_file_280):
                print(f"    [WARN] NPZ file not found")
                continue

            # Load and process first IR image (for size reference)
            npz_data = np.load(npz_file_280)
            if 'Amp' in npz_data:
                ir_data = npz_data['Amp'][:, :, 0]
            elif 'Phase' in npz_data:
                ir_data = npz_data['Phase'][:, :, 0]
            else:
                print(f"    [WARN] No Amp or Phase data found")
                continue

            # Warp visible image using saved homography
            h, w = ir_data.shape[:2]
            aligned_image1 = cv2.warpPerspective(self.image1, self.homography_matrix, (w, h))

            # Create simple merged image
            merged_image = cv2.addWeighted(aligned_image1, 0.5,
                                          (ir_data * 255 / ir_data.max()).astype(np.uint8), 0.5, 0)
            binary_mask = (merged_image == 0).astype(np.uint8)

            # Save to Masks_V3 subfolder
            save_folder = os.path.join(sibling_folder, "Masks_V3")
            os.makedirs(save_folder, exist_ok=True)

            merged_path = os.path.join(save_folder, "merged_image.jpg")
            mask_path = os.path.join(save_folder, "binary_mask.npy")
            matrix_path = os.path.join(save_folder, "homography_matrix.npy")

            cv2.imwrite(merged_path, merged_image)
            np.save(mask_path, binary_mask)
            np.save(matrix_path, self.homography_matrix)

            print(f"    [OK] Saved to: {save_folder}")

            results.append({
                'folder': sibling_folder,
                'merged_path': merged_path,
                'mask_path': mask_path,
                'matrix_path': matrix_path
            })

        print(f"\n  Total processed: {len(results)} sibling locations")
        print("="*70)

        return results

    def run(self):
        """Run the interactive merging tool with scrollable images."""
        print("="*60)
        print("IR IMAGE MERGE TOOL - MULTI IMAGE VERSION")
        print("="*60)
        print(f"Image 1 (Visible): {self.image1_path}")
        print(f"IR Images: {len(self.ir_image_list)} loaded")
        print(f"Colormap: {'Viridis' if self.use_colormap else 'Grayscale'}")
        print("="*60)

        while True:
            # Step 1: Select points on scrollable IR images
            self.points2 = []
            self.current_ir_idx = 0
            result = self._select_points_ir()

            if result == 'cancel':
                print("Merge cancelled by user")
                return False

            if len(self.points2) < 4:
                print("Error: Need at least 4 points. Exiting.")
                return False

            # Step 2: Select corresponding points on image1
            step2_result = None
            while True:
                self.points1 = []
                self.current_ir_ref_idx = 0
                step2_result = self._select_points_image1()

                if step2_result == 'cancel':
                    print("Merge cancelled by user")
                    return False

                if step2_result == 'back':
                    break  # Go back to step 1

                if len(self.points1) < 4:
                    print("Error: Need at least 4 points. Exiting.")
                    return False

                # Step 3: Compute merge
                try:
                    self._compute_merge()
                except Exception as e:
                    print(f"Error during merge: {e}")
                    import traceback
                    traceback.print_exc()
                    return False

                # Step 4: Preview
                preview_result = self._preview_result()

                if preview_result == 'save':
                    # Save to current location
                    self.save()

                    # Apply to sibling locations
                    print("\n" + "="*60)
                    print("APPLYING TO SIBLING LOCATIONS")
                    print("="*60)
                    sibling_results = self.apply_transformation_to_siblings()

                    print("\n" + "="*60)
                    print("MERGE COMPLETE!")
                    print("="*60)
                    print(f"Saved to {len(sibling_results) + 1} locations total")
                    return True
                elif preview_result == 'retry':
                    # Go back to step 1
                    print("\nRetrying from step 1...")
                    break  # Break inner loop, continue outer loop
                else:  # cancel
                    print("Merge cancelled by user")
                    return False

            # Check if we should continue outer loop
            if step2_result == 'back' or preview_result == 'retry':
                continue  # Go back to step 1
            else:
                # Should not reach here, but safety break
                break
