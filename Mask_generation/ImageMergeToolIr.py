"""
Interactive Image Merging Tool for IR Data (NPZ + Image)

This tool merges visible images with infrared amplitude data from NPZ files
using correspondence point selection and homography transformation.

Key Features:
-------------
- Loads NPZ files and extracts Amp[:,:,0] amplitude data
- Advanced contrast enhancement:
  * Percentile-based normalization (2-98%) to remove outliers
  * CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
  * Turbo colormap for better feature discrimination
- Dual reference display: shows Amp[:,:,1] alongside visible image for verification
- Merges at IR resolution (warps visible image to IR dimensions)
- Preserves black regions (0 values) from IR data
- Saves merged image, binary mask, and transformation matrix
- Auto-applies transformation to sibling datasets (other power/side modes)

Controls:
---------
Step 1 - Amp0/Amp1 Selection:
  - Left Click: Select correspondence point (in either window)
  - Right Drag: Pan the image
  - Ctrl + Wheel: Zoom in/out
  - U: Undo last point
  - R: Retry - restart from beginning
  - Q: Finish and proceed to step 2
  - ESC: Cancel and exit

Step 2 - Image1 Selection:
  - Left Click: Select matching point
  - Right Drag: Pan the image
  - Ctrl + Wheel: Zoom in/out
  - U: Undo last point
  - B: Go back to step 1
  - R: Retry - restart from step 1
  - Q: Finish and show preview
  - ESC: Cancel and exit

Preview Window (4 images):
  - S: Save merged image, mask, and transformation matrix
  - R: Retry - restart from step 1
  - ESC: Cancel and exit

Workflow:
---------
1. Select correspondence points on Amp[:,:,0] or Amp[:,:,1] with contrast enhancement
   - Amp0 13-87 window: Click here to select points
   - Amp1 5-95 window: Click here to select points
   - Points can be selected from either window simultaneously
2. Select matching points on Image 1 (Visible) in the same order
   - Three windows open: Image 1 (click here), Amp0 5-95 (view only), Amp1 5-95 (view only)
   - Reference windows show points from step 1 for verification
3. Preview the merged result at IR resolution (4 images shown)
4. Press 'S' to save, 'R' to retry, or ESC to exit
5. Transformation is automatically applied to sibling datasets

Output:
-------
- Merged image: [folder]/merged_image.jpg
- Binary mask: [folder]/binary_mask.npy
- Transformation matrix: [folder]/homography_matrix.npy
"""

import cv2
import numpy as np
import os
import glob


class ImageMergeToolIr:
    """
    Interactive tool for merging visible images with IR amplitude data.

    Parameters:
    -----------
    image1_path : str
        Path to the visible image (e.g., S{i}_Merged.jpg)
    npz_path : str
        Path to the NPZ file containing amplitude data
    flip_image1 : bool, optional
        Whether to flip image1 horizontally (default: False)

    Usage:
    ------
    tool = ImageMergeToolIr(image1_path, npz_path)
    tool.run()
    # Step 1: Select points on Amp0 13-87 OR Amp1 5-95 (both windows clickable)
    # Step 2: Select matching points on visible image (references: Amp0 5-95, Amp1 5-95)
    # Results: tool.merged_image, tool.homography_matrix, tool.binary_mask
    """

    def __init__(self, image1_path, npz_path, flip_image1=False):
        self.image1_path = image1_path
        self.npz_path = npz_path
        self.flip_image1 = flip_image1

        # Derive save directory from npz_path
        self.save_dir = os.path.dirname(npz_path)

        # Load image1 (visible image)
        self.image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        if self.image1 is None:
            raise FileNotFoundError(f"Image 1 not found at: {image1_path}")

        # Flip image1 if requested
        if self.flip_image1:
            self.image1 = cv2.flip(self.image1, 1)

        # Load NPZ and extract amplitude
        npz_data = np.load(npz_path)
        if 'Amp' not in npz_data:
            raise KeyError(f"'Amp' not found in NPZ file: {npz_path}")

        amp_data = npz_data['Amp']
        if amp_data.ndim != 3:
            raise ValueError(f"Expected 3D amplitude data, got shape: {amp_data.shape}")

        # Extract first channel Amp[:,:,0]
        self.image2_raw = amp_data[:, :, 0]

        # Create amp0 with percentile 13-87 for point selection
        self.amp0_1387 = self._normalize_to_uint8(self.image2_raw, percentile_clip=True,
                                                   low_percentile=13, high_percentile=87)
        self.amp0_1387 = self._enhance_contrast(self.amp0_1387)

        # Create amp0 with percentile 5-95 for reference
        self.amp0_595 = self._normalize_to_uint8(self.image2_raw, percentile_clip=True,
                                                  low_percentile=5, high_percentile=95)
        self.amp0_595 = self._enhance_contrast(self.amp0_595)

        # For backward compatibility
        self.image2 = self.amp0_1387

        # Extract second channel Amp[:,:,1] for point selection
        if amp_data.shape[2] > 1:
            self.image2_channel1_raw = amp_data[:, :, 1]
            # Create amp1 with percentile 5-95 for point selection
            self.amp1_595 = self._normalize_to_uint8(self.image2_channel1_raw,
                                                      percentile_clip=True,
                                                      low_percentile=5, high_percentile=95)
            self.amp1_595 = self._enhance_contrast(self.amp1_595)
            self.image2_channel1 = self.amp1_595
        else:
            self.amp1_595 = self.amp0_595.copy()
            self.image2_channel1 = self.amp1_595

        # Create defect mask (0 where IR data is black, 1 elsewhere)
        self.defect_mask = (self.image2 != 0).astype(np.uint8)

        # State for image1 selection
        self.points1 = []
        self.zoom1 = 1.0
        self.pan1 = [0, 0]
        self.is_panning1 = False
        self.pan_start1 = None
        self.mouse_pos1 = None

        # State for amp0 selection (image2)
        self.points2 = []
        self.zoom2 = 1.0
        self.pan2 = [0, 0]
        self.is_panning2 = False
        self.pan_start2 = None
        self.mouse_pos2 = None

        # State for amp1 selection (shares points2 with amp0)
        self.zoom_amp1_select = 1.0
        self.pan_amp1_select = [0, 0]
        self.is_panning_amp1_select = False
        self.pan_start_amp1_select = None
        self.mouse_pos_amp1_select = None

        # State for amp0 5-95 reference display during image1 selection
        self.zoom_amp0_ref2 = 1.0
        self.pan_amp0_ref2 = [0, 0]
        self.is_panning_amp0_ref2 = False
        self.pan_start_amp0_ref2 = None
        self.mouse_pos_amp0_ref2 = None

        # State for amp1 5-97 reference display during image1 selection
        self.zoom_amp1_ref = 1.0
        self.pan_amp1_ref = [0, 0]
        self.is_panning_amp1_ref = False
        self.pan_start_amp1_ref = None
        self.mouse_pos_amp1_ref = None

        # Results
        self.homography_matrix = None
        self.merged_image = None
        self.binary_mask = None
        self.aligned_image1 = None

        # Display settings
        self.canvas_size = (800, 800)

    def _normalize_to_uint8(self, data, percentile_clip=True, low_percentile=2, high_percentile=98):
        """
        Normalize float data to 0-255 uint8 range with optional percentile clipping.

        Parameters:
        -----------
        data : numpy array
            Input data to normalize
        percentile_clip : bool
            If True, use percentiles instead of min/max to clip outliers
        low_percentile : float
            Lower percentile for clipping (default: 2)
        high_percentile : float
            Upper percentile for clipping (default: 98)
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

    def _update_display(self, image, points, zoom, pan, window_name, mouse_pos=None, reference_points=None, use_colormap=False):
        """
        Update the display window with current state.

        Parameters:
        -----------
        use_colormap : bool
            If True, apply colormap to IR amplitude images for better visualization
        """
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
        # Apply colormap for IR images to enhance visibility
        if use_colormap:
            display = cv2.applyColorMap(canvas, cv2.COLORMAP_TURBO)
        else:
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
                self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Visible",
                                   reference_points=None)
        elif event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self._get_display_coordinates(x, y, self.zoom1, self.pan1)
            h, w = self.image1.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.points1.append((img_x, img_y))
                print(f"Image 1 - Point {len(self.points1)}: ({img_x}, {img_y})")
                self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Visible",
                                   reference_points=None)
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
            self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Visible",
                               reference_points=None)

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
                self._update_display(self.amp0_1387, self.points2, self.zoom2, self.pan2,
                                   "Amp0 13-87 (Select Points)", use_colormap=True)
        elif event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self._get_display_coordinates(x, y, self.zoom2, self.pan2)
            h, w = self.amp0_1387.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.points2.append((img_x, img_y))
                print(f"Amp0 13-87 - Point {len(self.points2)}: ({img_x}, {img_y})")
                self._update_display(self.amp0_1387, self.points2, self.zoom2, self.pan2,
                                   "Amp0 13-87 (Select Points)", use_colormap=True)
                # Update amp1 window too to show new point
                self._update_display(self.amp1_595, self.points2, self.zoom_amp1_select, self.pan_amp1_select,
                                   "Amp1 5-95 (Select Points)", use_colormap=True)
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
            self._update_display(self.amp0_1387, self.points2, self.zoom2, self.pan2,
                               "Amp0 13-87 (Select Points)", use_colormap=True)

    def _mouse_callback_amp1_select(self, event, x, y, flags, param):
        """Mouse callback for amp1 5-95 selection window - accepts clicks to add points to points2"""
        self.mouse_pos_amp1_select = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning_amp1_select = True
            self.pan_start_amp1_select = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning_amp1_select = False
            self.pan_start_amp1_select = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning_amp1_select and self.pan_start_amp1_select is not None:
                dx = x - self.pan_start_amp1_select[0]
                dy = y - self.pan_start_amp1_select[1]
                self.pan_amp1_select[0] += dx
                self.pan_amp1_select[1] += dy
                self.pan_start_amp1_select = (x, y)
                self._update_display(self.amp1_595, self.points2, self.zoom_amp1_select, self.pan_amp1_select,
                                   "Amp1 5-95 (Select Points)", use_colormap=True)
                # Update amp0 window too to show new point
                self._update_display(self.amp0_1387, self.points2, self.zoom2, self.pan2,
                                   "Amp0 13-87 (Select Points)", use_colormap=True)
        elif event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self._get_display_coordinates(x, y, self.zoom_amp1_select, self.pan_amp1_select)
            h, w = self.amp1_595.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.points2.append((img_x, img_y))
                print(f"Amp1 5-95 - Point {len(self.points2)}: ({img_x}, {img_y})")
                self._update_display(self.amp1_595, self.points2, self.zoom_amp1_select, self.pan_amp1_select,
                                   "Amp1 5-95 (Select Points)", use_colormap=True)
                # Update amp0 window too to show new point
                self._update_display(self.amp0_1387, self.points2, self.zoom2, self.pan2,
                                   "Amp0 13-87 (Select Points)", use_colormap=True)
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom_amp1_select
            if flags > 0:
                self.zoom_amp1_select *= 1.1
            else:
                self.zoom_amp1_select /= 1.1
            self.zoom_amp1_select = max(0.1, min(self.zoom_amp1_select, 10.0))

            zoom_ratio = self.zoom_amp1_select / old_zoom
            self.pan_amp1_select[0] = x - (x - self.pan_amp1_select[0]) * zoom_ratio
            self.pan_amp1_select[1] = y - (y - self.pan_amp1_select[1]) * zoom_ratio
            self._update_display(self.amp1_595, self.points2, self.zoom_amp1_select, self.pan_amp1_select,
                               "Amp1 5-95 (Select Points)", use_colormap=True)

    def _mouse_callback_reference(self, event, x, y, flags, param):
        """Mouse callback for reference channel window (Amp[:,:,1]) - view only with pan/zoom"""
        self.mouse_pos_ref = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning_ref = True
            self.pan_start_ref = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning_ref = False
            self.pan_start_ref = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning_ref and self.pan_start_ref is not None:
                dx = x - self.pan_start_ref[0]
                dy = y - self.pan_start_ref[1]
                self.pan_ref[0] += dx
                self.pan_ref[1] += dy
                self.pan_start_ref = (x, y)
                self._update_display(self.image2_channel1, [], self.zoom_ref, self.pan_ref,
                                   "Reference - IR Amp[:,:,1]", reference_points=self.points2, use_colormap=True)
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom_ref
            if flags > 0:
                self.zoom_ref *= 1.1
            else:
                self.zoom_ref /= 1.1
            self.zoom_ref = max(0.1, min(self.zoom_ref, 10.0))

            zoom_ratio = self.zoom_ref / old_zoom
            self.pan_ref[0] = x - (x - self.pan_ref[0]) * zoom_ratio
            self.pan_ref[1] = y - (y - self.pan_ref[1]) * zoom_ratio
            self._update_display(self.image2_channel1, [], self.zoom_ref, self.pan_ref,
                               "Reference - IR Amp[:,:,1]", reference_points=self.points2, use_colormap=True)

    def _mouse_callback_amp0_ref(self, event, x, y, flags, param):
        """Mouse callback for amp0 5-95 reference window during amp0 15-87 selection - view only"""
        self.mouse_pos_amp0_ref = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning_amp0_ref = True
            self.pan_start_amp0_ref = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning_amp0_ref = False
            self.pan_start_amp0_ref = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning_amp0_ref and self.pan_start_amp0_ref is not None:
                dx = x - self.pan_start_amp0_ref[0]
                dy = y - self.pan_start_amp0_ref[1]
                self.pan_amp0_ref[0] += dx
                self.pan_amp0_ref[1] += dy
                self.pan_start_amp0_ref = (x, y)
                self._update_display(self.amp0_595, [], self.zoom_amp0_ref, self.pan_amp0_ref,
                                   "Reference - Amp0 5-95", use_colormap=True)
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom_amp0_ref
            if flags > 0:
                self.zoom_amp0_ref *= 1.1
            else:
                self.zoom_amp0_ref /= 1.1
            self.zoom_amp0_ref = max(0.1, min(self.zoom_amp0_ref, 10.0))

            zoom_ratio = self.zoom_amp0_ref / old_zoom
            self.pan_amp0_ref[0] = x - (x - self.pan_amp0_ref[0]) * zoom_ratio
            self.pan_amp0_ref[1] = y - (y - self.pan_amp0_ref[1]) * zoom_ratio
            self._update_display(self.amp0_595, [], self.zoom_amp0_ref, self.pan_amp0_ref,
                               "Reference - Amp0 5-95", use_colormap=True)

    def _mouse_callback_amp0_ref2(self, event, x, y, flags, param):
        """Mouse callback for amp0 5-95 reference window during image1 selection - view only"""
        self.mouse_pos_amp0_ref2 = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning_amp0_ref2 = True
            self.pan_start_amp0_ref2 = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning_amp0_ref2 = False
            self.pan_start_amp0_ref2 = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning_amp0_ref2 and self.pan_start_amp0_ref2 is not None:
                dx = x - self.pan_start_amp0_ref2[0]
                dy = y - self.pan_start_amp0_ref2[1]
                self.pan_amp0_ref2[0] += dx
                self.pan_amp0_ref2[1] += dy
                self.pan_start_amp0_ref2 = (x, y)
                self._update_display(self.amp0_595, [], self.zoom_amp0_ref2, self.pan_amp0_ref2,
                                   "Reference - Amp0 5-95", reference_points=self.points2, use_colormap=True)
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom_amp0_ref2
            if flags > 0:
                self.zoom_amp0_ref2 *= 1.1
            else:
                self.zoom_amp0_ref2 /= 1.1
            self.zoom_amp0_ref2 = max(0.1, min(self.zoom_amp0_ref2, 10.0))

            zoom_ratio = self.zoom_amp0_ref2 / old_zoom
            self.pan_amp0_ref2[0] = x - (x - self.pan_amp0_ref2[0]) * zoom_ratio
            self.pan_amp0_ref2[1] = y - (y - self.pan_amp0_ref2[1]) * zoom_ratio
            self._update_display(self.amp0_595, [], self.zoom_amp0_ref2, self.pan_amp0_ref2,
                               "Reference - Amp0 5-95", reference_points=self.points2, use_colormap=True)

    def _mouse_callback_amp1_ref(self, event, x, y, flags, param):
        """Mouse callback for amp1 5-95 reference window during image1 selection - view only"""
        self.mouse_pos_amp1_ref = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning_amp1_ref = True
            self.pan_start_amp1_ref = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning_amp1_ref = False
            self.pan_start_amp1_ref = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning_amp1_ref and self.pan_start_amp1_ref is not None:
                dx = x - self.pan_start_amp1_ref[0]
                dy = y - self.pan_start_amp1_ref[1]
                self.pan_amp1_ref[0] += dx
                self.pan_amp1_ref[1] += dy
                self.pan_start_amp1_ref = (x, y)
                self._update_display(self.amp1_595, [], self.zoom_amp1_ref, self.pan_amp1_ref,
                                   "Reference - Amp1 5-95", reference_points=self.points2, use_colormap=True)
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom_amp1_ref
            if flags > 0:
                self.zoom_amp1_ref *= 1.1
            else:
                self.zoom_amp1_ref /= 1.1
            self.zoom_amp1_ref = max(0.1, min(self.zoom_amp1_ref, 10.0))

            zoom_ratio = self.zoom_amp1_ref / old_zoom
            self.pan_amp1_ref[0] = x - (x - self.pan_amp1_ref[0]) * zoom_ratio
            self.pan_amp1_ref[1] = y - (y - self.pan_amp1_ref[1]) * zoom_ratio
            self._update_display(self.amp1_595, [], self.zoom_amp1_ref, self.pan_amp1_ref,
                               "Reference - Amp1 5-95", reference_points=self.points2, use_colormap=True)

    def _select_points_image1(self):
        """
        Interactive point selection for image1.

        Returns:
        --------
        str : 'continue' to proceed, 'back' to return to amp0 selection, 'retry' to restart from step 1
        """
        print("\n" + "="*60)
        print("SELECT MATCHING POINTS ON IMAGE 1 (VISIBLE)")
        print("="*60)
        print("Reference windows show Amp0 5-95 and Amp1 5-95 for verification")
        print("Press 'B' to go back, 'R' to retry from step 1, ESC to cancel")

        # Open Image 1 window (visible image)
        cv2.namedWindow("Image 1 - Visible", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image 1 - Visible", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("Image 1 - Visible", self._mouse_callback_image1)

        # Open reference window (Amp0 5-95)
        cv2.namedWindow("Reference - Amp0 5-95", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reference - Amp0 5-95", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("Reference - Amp0 5-95", self._mouse_callback_amp0_ref2)

        # Open reference window (Amp1 5-95)
        cv2.namedWindow("Reference - Amp1 5-95", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reference - Amp1 5-95", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("Reference - Amp1 5-95", self._mouse_callback_amp1_ref)

        # Initial display (no reference points on image1 - user requested)
        self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Visible",
                           reference_points=None, use_colormap=False)
        self._update_display(self.amp0_595, [], self.zoom_amp0_ref2, self.pan_amp0_ref2,
                           "Reference - Amp0 5-95", reference_points=self.points2, use_colormap=True)
        self._update_display(self.amp1_595, [], self.zoom_amp1_ref, self.pan_amp1_ref,
                           "Reference - Amp1 5-95", reference_points=self.points2, use_colormap=True)

        result = 'continue'
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key
                print("ESC pressed - Cancelling...")
                result = 'cancel'
                break
            elif key == ord('q') or key == ord('Q'):
                if len(self.points1) != len(self.points2):
                    print(f"Warning: {len(self.points1)} points on Image 1, but {len(self.points2)} on Image 2")
                result = 'continue'
                break
            elif key == ord('b') or key == ord('B'):
                print("Going back to Amp0/Amp1 selection...")
                result = 'back'
                break
            elif key == ord('r') or key == ord('R'):
                print("Retrying from step 1...")
                result = 'retry'
                break
            elif key == ord('u') or key == ord('U'):
                if len(self.points1) > 0:
                    removed = self.points1.pop()
                    print(f"Undone point {len(self.points1)+1}: {removed}")
                    self._update_display(self.image1, self.points1, self.zoom1, self.pan1, "Image 1 - Visible",
                                       reference_points=None, use_colormap=False)

        cv2.destroyWindow("Image 1 - Visible")
        cv2.destroyWindow("Reference - Amp0 5-95")
        cv2.destroyWindow("Reference - Amp1 5-95")
        print(f"Selected {len(self.points1)} points on Image 1")
        return result

    def _select_points_image2(self):
        """
        Interactive point selection for image2 (amp0 and amp1 simultaneously).

        Returns:
        --------
        str : 'continue' to proceed, 'retry' to restart from beginning
        """
        print("\n" + "="*60)
        print("SELECT CORRESPONDENCE POINTS ON AMP0 13-87 OR AMP1 5-95")
        print("="*60)
        print("You can click on EITHER window to select points")
        print("Points are shared between both windows")
        print("Press 'R' to retry from the beginning, 'Q' when done, ESC to cancel")

        # Open Amp0 13-87 window (for point selection)
        cv2.namedWindow("Amp0 13-87 (Select Points)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Amp0 13-87 (Select Points)", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("Amp0 13-87 (Select Points)", self._mouse_callback_image2)

        # Open Amp1 5-95 window (also for point selection)
        cv2.namedWindow("Amp1 5-95 (Select Points)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Amp1 5-95 (Select Points)", self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback("Amp1 5-95 (Select Points)", self._mouse_callback_amp1_select)

        # Initial display
        self._update_display(self.amp0_1387, self.points2, self.zoom2, self.pan2,
                           "Amp0 13-87 (Select Points)", use_colormap=True)
        self._update_display(self.amp1_595, self.points2, self.zoom_amp1_select, self.pan_amp1_select,
                           "Amp1 5-95 (Select Points)", use_colormap=True)

        result = 'continue'
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key
                print("ESC pressed - Cancelling...")
                result = 'cancel'
                break
            elif key == ord('q') or key == ord('Q'):
                if len(self.points2) < 4:
                    print(f"Warning: Only {len(self.points2)} points selected. Need at least 4 for homography.")
                result = 'continue'
                break
            elif key == ord('r') or key == ord('R'):
                print("Retrying from the beginning...")
                result = 'retry'
                break
            elif key == ord('u') or key == ord('U'):
                if len(self.points2) > 0:
                    removed = self.points2.pop()
                    print(f"Undone point {len(self.points2)+1}: {removed}")
                    self._update_display(self.amp0_1387, self.points2, self.zoom2, self.pan2,
                                       "Amp0 13-87 (Select Points)", use_colormap=True)
                    self._update_display(self.amp1_595, self.points2, self.zoom_amp1_select, self.pan_amp1_select,
                                       "Amp1 5-95 (Select Points)", use_colormap=True)

        cv2.destroyWindow("Amp0 13-87 (Select Points)")
        cv2.destroyWindow("Amp1 5-95 (Select Points)")
        print(f"Selected {len(self.points2)} points on Amp0/Amp1")
        return result

    def _compute_merge(self):
        """Compute homography and merge images at IR resolution"""
        if len(self.points1) < 4 or len(self.points2) < 4:
            raise ValueError(f"Need at least 4 points on each image. Got {len(self.points1)} and {len(self.points2)}")

        if len(self.points1) != len(self.points2):
            raise ValueError(f"Point count mismatch: {len(self.points1)} vs {len(self.points2)}")

        points1_array = np.array(self.points1, dtype=np.float32)
        points2_array = np.array(self.points2, dtype=np.float32)

        # Compute homography: warp image1 to image2 coordinates
        print("\nComputing homography...")
        # points1 = visible image, points2 = IR image
        # We want to map: visible → IR, so findHomography(visible, IR)
        self.homography_matrix, status = cv2.findHomography(points1_array, points2_array, 0)
        print("Using LMEDS method (robust to outliers)")

        if self.homography_matrix is None:
            raise ValueError("Failed to compute homography matrix")

        # Warp image1 to align with image2 (at IR resolution)
        print("Warping Image 1 to IR resolution...")
        h, w = self.image2.shape[:2]
        self.aligned_image1 = cv2.warpPerspective(self.image1, self.homography_matrix, (w, h))

        # Create and warp binary mask for image1 black pixels (to avoid interpolation artifacts)
        mask_image1_black = (self.image1 == 0).astype(np.uint8)
        warped_mask_black = cv2.warpPerspective(mask_image1_black, self.homography_matrix, (w, h),
                                                flags=cv2.INTER_NEAREST)  # No interpolation for mask

        # Merge images
        print("Merging images...")

        # Start with gray canvas
        self.merged_image = np.full((h, w), 125, dtype=np.uint8)

        # Create masks for valid data regions
        mask1 = (self.aligned_image1 > 0).astype(bool)
        mask2 = (self.image2 > 0).astype(bool)

        # Create blended image (average of both images)
        blended = cv2.addWeighted(self.aligned_image1, 0.5, self.image2, 0.5, 0)

        # Where both images have data: use blended
        both_mask = mask1 & mask2
        self.merged_image[both_mask] = blended[both_mask]

        # Where only image2 has data: use image2 (prioritize IR data)
        only2_mask = ~mask1 & mask2
        self.merged_image[only2_mask] = self.image2[only2_mask]

        # Where only image1 has data: use image1
        only1_mask = mask1 & ~mask2
        self.merged_image[only1_mask] = self.aligned_image1[only1_mask]

        # Preserve black pixels from IR data (defects)
        black_pixels_ir = (self.image2 == 0)
        self.merged_image[black_pixels_ir] = 0

        # Preserve black pixels from visible image (image1)
        # IMPORTANT: image1's 0 values must remain 0 after transformation
        # Use the warped mask (not aligned_image1) to avoid interpolation artifacts
        black_pixels_img1 = (warped_mask_black > 0)
        self.merged_image[black_pixels_img1] = 0

        # Create binary mask: 1 where merged image is 0, 0 elsewhere
        self.binary_mask = (self.merged_image == 0).astype(np.uint8)

        print("Merge complete!")
        print(f"Merged image shape: {self.merged_image.shape}")
        print(f"Black pixels from IR: {np.sum(black_pixels_ir)}")
        print(f"Black pixels from image1: {np.sum(black_pixels_img1)}")
        print(f"Total black pixels in merged: {np.sum(self.binary_mask)}")

    def _preview_result(self):
        """
        Show preview of merged result with 4 images.

        Returns:
        --------
        str : 'save' to save and continue, 'retry' to restart point selection, 'cancel' to exit
        """
        print("\n" + "="*60)
        print("PREVIEW MERGED RESULT")
        print("="*60)
        print("Press 'S' to save, 'R' to retry point selection, ESC to exit")

        # Create side-by-side comparison with 4 images
        h, w = self.image2.shape[:2]

        # Resize for display if too large (4 images now)
        max_display_width = 2400
        if w * 4 > max_display_width:
            scale = max_display_width / (w * 4)
            display_w = int(w * scale)
            display_h = int(h * scale)
            vis_display = cv2.resize(self.aligned_image1, (display_w, display_h))
            amp0_display = cv2.resize(self.amp0_1387, (display_w, display_h))
            amp1_display = cv2.resize(self.amp1_595, (display_w, display_h))
            merged_display = cv2.resize(self.merged_image, (display_w, display_h))
        else:
            vis_display = self.aligned_image1
            amp0_display = self.amp0_1387
            amp1_display = self.amp1_595
            merged_display = self.merged_image

        # Convert grayscale to BGR for stacking
        vis_bgr = cv2.cvtColor(vis_display, cv2.COLOR_GRAY2BGR)
        merged_bgr = cv2.cvtColor(merged_display, cv2.COLOR_GRAY2BGR)

        # Convert amp images with colormap to BGR
        amp0_colored = cv2.applyColorMap(amp0_display, cv2.COLORMAP_TURBO)
        amp1_colored = cv2.applyColorMap(amp1_display, cv2.COLORMAP_TURBO)

        # Stack horizontally: Visible | Amp0 | Amp1 | Merged
        comparison = np.hstack([vis_bgr, amp0_colored, amp1_colored, merged_bgr])

        # Add labels
        h_comp, w_comp = comparison.shape[:2]
        w_quarter = w_comp // 4

        cv2.putText(comparison, "Visible (Aligned)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(comparison, "Amp0 (13-87)", (w_quarter + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(comparison, "Amp1 (5-95)", (2*w_quarter + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(comparison, "Merged Result", (3*w_quarter + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Add instructions at bottom
        cv2.putText(comparison, "S=Save | R=Retry | ESC=Cancel", (10, h_comp - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.namedWindow("Merge Preview (4 Images)", cv2.WINDOW_NORMAL)
        cv2.imshow("Merge Preview (4 Images)", comparison)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Cancelled without saving...")
                cv2.destroyAllWindows()
                return 'cancel'
            elif key == ord('s') or key == ord('S'):
                cv2.destroyAllWindows()
                return 'save'
            elif key == ord('r') or key == ord('R'):
                print("Retrying point selection...")
                cv2.destroyAllWindows()
                return 'retry'

    def save(self):
        """Save merged image, binary mask, and transformation matrix"""
        if self.merged_image is None:
            raise ValueError("No merged image to save. Run the tool first.")

        # Save merged image
        merged_path = os.path.join(self.save_dir, "merged_image.jpg")
        cv2.imwrite(merged_path, self.merged_image)
        print(f"Merged image saved to: {merged_path}")

        # Save binary mask
        mask_path = os.path.join(self.save_dir, "binary_mask.npy")
        np.save(mask_path, self.binary_mask)
        print(f"Binary mask saved to: {mask_path}")

        # Save homography matrix
        matrix_path = os.path.join(self.save_dir, "homography_matrix.npy")
        np.save(matrix_path, self.homography_matrix)
        print(f"Homography matrix saved to: {matrix_path}")

        return merged_path, mask_path, matrix_path

    def apply_transformation_to_siblings(self):
        """
        Apply the computed transformation to sibling NPZ files in the same location.
        Finds other folders with the same location suffix but different power/side modes.
        """
        if self.homography_matrix is None:
            raise ValueError("No transformation matrix available. Run the tool first.")

        # Get current folder name
        current_folder = os.path.basename(self.save_dir)
        parent_dir = os.path.dirname(self.save_dir)

        # Parse folder name to extract location (last two parts: e.g., "bottom_left")
        parts = current_folder.rsplit('_', 2)
        if len(parts) < 3:
            print(f"Warning: Could not parse location from folder name: {current_folder}")
            return []

        location_suffix = f"{parts[1]}_{parts[2]}"  # e.g., "bottom_left"

        # Find all sibling folders with same location
        pattern = os.path.join(parent_dir, f"*_{location_suffix}")
        sibling_folders = [f for f in glob.glob(pattern) if os.path.isdir(f) and f != self.save_dir]

        print(f"\nFound {len(sibling_folders)} sibling folders with location '{location_suffix}'")

        results = []
        for sibling_folder in sibling_folders:
            # Find NPZ file in sibling folder
            npz_pattern = os.path.join(sibling_folder, "PPT_a=0_width=280.npz")
            npz_files = glob.glob(npz_pattern)

            if not npz_files:
                print(f"Warning: No NPZ file found in {sibling_folder}")
                continue

            npz_file = npz_files[0]
            print(f"\nProcessing sibling: {os.path.basename(sibling_folder)}")

            # Load NPZ and extract amplitude
            npz_data = np.load(npz_file)
            if 'Amp' not in npz_data:
                print(f"Warning: 'Amp' not found in {npz_file}")
                continue

            amp_data = npz_data['Amp'][:, :, 0]
            image2_norm = self._normalize_to_uint8(amp_data, percentile_clip=True,
                                                     low_percentile=2, high_percentile=98)
            image2_norm = self._enhance_contrast(image2_norm)

            # Warp image1 to align with this sibling's IR data
            h, w = image2_norm.shape[:2]
            aligned_image1 = cv2.warpPerspective(self.image1, self.homography_matrix, (w, h))

            # Create and warp binary mask for image1 black pixels (to avoid interpolation artifacts)
            mask_image1_black = (self.image1 == 0).astype(np.uint8)
            warped_mask_black = cv2.warpPerspective(mask_image1_black, self.homography_matrix, (w, h),
                                                    flags=cv2.INTER_NEAREST)  # No interpolation for mask

            # Merge images
            merged_image = np.full((h, w), 125, dtype=np.uint8)

            mask1 = (aligned_image1 > 0).astype(bool)
            mask2 = (image2_norm > 0).astype(bool)

            blended = cv2.addWeighted(aligned_image1, 0.5, image2_norm, 0.5, 0)

            both_mask = mask1 & mask2
            merged_image[both_mask] = blended[both_mask]

            only2_mask = ~mask1 & mask2
            merged_image[only2_mask] = image2_norm[only2_mask]

            only1_mask = mask1 & ~mask2
            merged_image[only1_mask] = aligned_image1[only1_mask]

            # Preserve black pixels from IR data
            black_pixels_ir = (image2_norm == 0)
            merged_image[black_pixels_ir] = 0

            # Preserve black pixels from visible image (image1)
            # Use the warped mask to avoid interpolation artifacts
            black_pixels_img1 = (warped_mask_black > 0)
            merged_image[black_pixels_img1] = 0

            # Create binary mask
            binary_mask = (merged_image == 0).astype(np.uint8)

            # Save results
            merged_path = os.path.join(sibling_folder, "merged_image.jpg")
            mask_path = os.path.join(sibling_folder, "binary_mask.npy")
            matrix_path = os.path.join(sibling_folder, "homography_matrix.npy")

            cv2.imwrite(merged_path, merged_image)
            np.save(mask_path, binary_mask)
            np.save(matrix_path, self.homography_matrix)

            print(f"  Saved merged image: {merged_path}")
            print(f"  Saved binary mask: {mask_path}")
            print(f"  Saved transformation matrix: {matrix_path}")

            results.append({
                'folder': sibling_folder,
                'merged_path': merged_path,
                'mask_path': mask_path,
                'matrix_path': matrix_path
            })

        return results

    def run(self):
        """
        Run the interactive merging tool with navigation support.

        Navigation:
        - From step 2 (image1 selection): Press 'B' to go back to step 1
        - From preview: Press 'R' to retry from step 1
        """
        print("="*60)
        print("INTERACTIVE IR IMAGE MERGE TOOL")
        print("="*60)
        print(f"Image 1 (Visible): {self.image1_path}")
        print(f"Image 2 (IR NPZ):  {self.npz_path}")
        print(f"Image 1 flipped:   {self.flip_image1}")
        print(f"Save directory:    {self.save_dir}")
        print("="*60)
        print("\nNavigation: Press 'B' in step 2 to go back, 'R' in preview to retry")
        print("="*60)

        # Main loop for navigation
        while True:
            # Step 1: Select points on amp0/amp1 (IR amplitude)
            self.points2 = []  # Reset points2
            result = self._select_points_image2()

            if result == 'cancel':
                # ESC pressed - exit
                print("Merge cancelled by user (ESC)")
                return False

            if result == 'retry':
                # Retry from beginning
                print("\n" + "="*60)
                print("RETRYING FROM BEGINNING")
                print("="*60)
                continue  # Restart outer loop

            if len(self.points2) < 4:
                print("Error: Need at least 4 points. Exiting.")
                return False

            # Inner loop for step 2 navigation
            while True:
                # Step 2: Select corresponding points on image1 (visible)
                self.points1 = []  # Reset points1
                result = self._select_points_image1()

                if result == 'cancel':
                    # ESC pressed - exit
                    print("Merge cancelled by user (ESC)")
                    return False

                if result == 'back':
                    # Go back to step 1
                    print("\n" + "="*60)
                    print("RETURNING TO STEP 1")
                    print("="*60)
                    break  # Break inner loop, continue outer loop

                if result == 'retry':
                    # Retry from step 1
                    print("\n" + "="*60)
                    print("RETRYING FROM STEP 1")
                    print("="*60)
                    break  # Break inner loop, continue outer loop

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

                # Step 4: Preview and decide
                preview_result = self._preview_result()

                if preview_result == 'save':
                    # Save and finish
                    self.save()

                    # Apply to sibling folders
                    print("\n" + "="*60)
                    print("APPLYING TRANSFORMATION TO SIBLING DATASETS")
                    print("="*60)
                    sibling_results = self.apply_transformation_to_siblings()

                    print("\n" + "="*60)
                    print("MERGE COMPLETE!")
                    print("="*60)
                    print(f"Processed 1 + {len(sibling_results)} datasets at this location")
                    return True

                elif preview_result == 'retry':
                    # Retry from step 1
                    print("\n" + "="*60)
                    print("RETRYING FROM STEP 1")
                    print("="*60)
                    break  # Break inner loop, continue outer loop

                else:  # 'cancel'
                    print("Merge cancelled by user")
                    return False

            # Check if we should continue outer loop
            if result == 'back' or result == 'retry' or preview_result == 'retry':
                continue  # Go back to step 1
            else:
                break  # Should not reach here, but safety exit
