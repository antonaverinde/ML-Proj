"""Simple wrapper for rotate_displace functionality"""
import cv2
import numpy as np
import os


class ImageRotationTool:
    """
    Simple class wrapper for interactive image rotation and displacement.

    Parameters:
    -----------
    image_path : str
        Path to the image file
    save_dir : str
        Directory to save transformed images (default: Barrel_Images2_croped_NewAxes)
    fill_value : int
        Value to fill empty regions after transformation (default: 0)
        - 0 = black
        - 255 = white
        - any value 0-255

    Usage:
        tool = ImageRotationTool(image_path)
        tool.run()
        # Access results: tool.transformed_image, tool.rotation_matrix
    """

    def __init__(self, image_path, save_dir=None, fill_value=0):
        self.image_path = image_path

        # Set default save_dir if not provided
        if save_dir is None:
            try:
                from config import BINARY_MASKS_PATH
                save_dir = str(BINARY_MASKS_PATH / "Barrel_Images2_croped_NewAxes")
            except ImportError:
                # Fallback if config not available
                save_dir = "Barrel_Images2_croped_NewAxes"

        self.save_dir = save_dir
        self.fill_value = fill_value  # Value to fill empty regions (0=black, 255=white, etc.)

        # Results - accessible after run()
        self.transformed_image = None
        self.rotation_matrix = None

        # Internal state
        self.selected_points = []
        self.original_image = None
        self.image_copy = None
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.is_panning = False
        self.pan_start = None
        self.mouse_pos = None
        self.transform_calculated = False

        # Transformed window state
        self.transformed_zoom = 1.0
        self.transformed_pan = [0, 0]
        self.transformed_panning = False
        self.transformed_pan_start = None

    def _get_display_coordinates(self, x, y, zoom, offset):
        img_x = int((x - offset[0]) / zoom)
        img_y = int((y - offset[1]) / zoom)
        return img_x, img_y

    def _get_screen_coordinates(self, img_x, img_y, zoom, offset):
        x = int(img_x * zoom + offset[0])
        y = int(img_y * zoom + offset[1])
        return x, y

    def _update_display(self):
        h, w = self.original_image.shape[:2]
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)
        zoomed = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas_h, canvas_w = 800, 800
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        x_start = int(max(0, -self.pan_offset[0]))
        y_start = int(max(0, -self.pan_offset[1]))
        x_end = int(min(new_w, canvas_w - self.pan_offset[0]))
        y_end = int(min(new_h, canvas_h - self.pan_offset[1]))

        paste_x = int(max(0, self.pan_offset[0]))
        paste_y = int(max(0, self.pan_offset[1]))

        if x_end > x_start and y_end > y_start:
            paste_w = int(x_end - x_start)
            paste_h = int(y_end - y_start)
            canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = zoomed[y_start:y_end, x_start:x_end]

        self.image_copy = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for i, (img_x, img_y) in enumerate(self.selected_points):
            screen_x, screen_y = self._get_screen_coordinates(img_x, img_y, self.zoom_level, self.pan_offset)
            if 0 <= screen_x < canvas_w and 0 <= screen_y < canvas_h:
                cv2.circle(self.image_copy, (screen_x, screen_y), 5, (0, 255, 0), -1)
                label = "1st" if i == 0 else "2nd"
                cv2.putText(self.image_copy, label, (screen_x + 10, screen_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(self.selected_points) == 1 and self.mouse_pos is not None:
            screen_x1, screen_y1 = self._get_screen_coordinates(self.selected_points[0][0], self.selected_points[0][1],
                                                           self.zoom_level, self.pan_offset)
            if 0 <= self.mouse_pos[0] < canvas_w and 0 <= self.mouse_pos[1] < canvas_h:
                cv2.line(self.image_copy, (screen_x1, screen_y1), self.mouse_pos, (255, 0, 0), 2)
        elif len(self.selected_points) == 2:
            screen_x1, screen_y1 = self._get_screen_coordinates(self.selected_points[0][0], self.selected_points[0][1],
                                                           self.zoom_level, self.pan_offset)
            screen_x2, screen_y2 = self._get_screen_coordinates(self.selected_points[1][0], self.selected_points[1][1],
                                                           self.zoom_level, self.pan_offset)
            cv2.line(self.image_copy, (screen_x1, screen_y1), (screen_x2, screen_y2), (0, 255, 255), 2)

        instructions = [
            "Left Click: Select point",
            "Right Drag: Pan",
            "Ctrl+Wheel: Zoom",
            "U: Undo last point",
            "C: Calculate transform",
            "S: Save",
            "ESC: Exit"
        ]
        y_pos = 20
        for inst in instructions:
            cv2.putText(self.image_copy, inst, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 0), 1, cv2.LINE_AA)
            y_pos += 15

        cv2.putText(self.image_copy, f"Zoom: {self.zoom_level:.2f}x", (10, canvas_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Select Points", self.image_copy)

    def _mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False
            self.pan_start = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning and self.pan_start is not None:
                dx = x - self.pan_start[0]
                dy = y - self.pan_start[1]
                self.pan_offset[0] += dx
                self.pan_offset[1] += dy
                self.pan_start = (x, y)
                self._update_display()
            elif len(self.selected_points) == 1:
                self._update_display()
        elif event == cv2.EVENT_LBUTTONDOWN and len(self.selected_points) < 2:
            img_x, img_y = self._get_display_coordinates(x, y, self.zoom_level, self.pan_offset)
            h, w = self.original_image.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.selected_points.append((img_x, img_y))
                print(f"Selected point {len(self.selected_points)}: ({img_x}, {img_y})")
                self._update_display()
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.zoom_level
            if flags > 0:
                self.zoom_level *= 1.1
            else:
                self.zoom_level /= 1.1
            self.zoom_level = max(0.1, min(self.zoom_level, 10.0))

            zoom_ratio = self.zoom_level / old_zoom
            self.pan_offset[0] = x - (x - self.pan_offset[0]) * zoom_ratio
            self.pan_offset[1] = y - (y - self.pan_offset[1]) * zoom_ratio
            self._update_display()

    def _update_transformed_display(self):
        if self.transformed_image is None:
            return

        h, w = self.transformed_image.shape[:2]
        new_w = int(w * self.transformed_zoom)
        new_h = int(h * self.transformed_zoom)
        zoomed = cv2.resize(self.transformed_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas_h, canvas_w = 800, 800
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        x_start = int(max(0, -self.transformed_pan[0]))
        y_start = int(max(0, -self.transformed_pan[1]))
        x_end = int(min(new_w, canvas_w - self.transformed_pan[0]))
        y_end = int(min(new_h, canvas_h - self.transformed_pan[1]))

        paste_x = int(max(0, self.transformed_pan[0]))
        paste_y = int(max(0, self.transformed_pan[1]))

        if x_end > x_start and y_end > y_start:
            paste_w = int(x_end - x_start)
            paste_h = int(y_end - y_start)
            canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = zoomed[y_start:y_end, x_start:x_end]

        display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        instructions = ["Right Drag: Pan", "Ctrl+Wheel: Zoom"]
        y_pos = 20
        for inst in instructions:
            cv2.putText(display, inst, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 0), 1, cv2.LINE_AA)
            y_pos += 15

        cv2.putText(display, f"Zoom: {self.transformed_zoom:.2f}x", (10, canvas_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Transformed Image", display)

    def _transformed_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.transformed_panning = True
            self.transformed_pan_start = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.transformed_panning = False
            self.transformed_pan_start = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.transformed_panning and self.transformed_pan_start is not None:
                dx = x - self.transformed_pan_start[0]
                dy = y - self.transformed_pan_start[1]
                self.transformed_pan[0] += dx
                self.transformed_pan[1] += dy
                self.transformed_pan_start = (x, y)
                self._update_transformed_display()
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_CTRLKEY):
            old_zoom = self.transformed_zoom
            if flags > 0:
                self.transformed_zoom *= 1.1
            else:
                self.transformed_zoom /= 1.1
            self.transformed_zoom = max(0.1, min(self.transformed_zoom, 10.0))

            zoom_ratio = self.transformed_zoom / old_zoom
            self.transformed_pan[0] = x - (x - self.transformed_pan[0]) * zoom_ratio
            self.transformed_pan[1] = y - (y - self.transformed_pan[1]) * zoom_ratio
            self._update_transformed_display()

    def _calculate_transform(self):
        if len(self.selected_points) != 2:
            return

        point1, point2 = self.selected_points
        start_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

        h, w = self.original_image.shape[:2]
        image_center = (w / 2, h / 2)

        distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        print(f"Distance between points: {distance:.2f} pixels")

        new_x_vector = np.array(point1) - np.array(start_point)
        new_x_unit = new_x_vector / np.linalg.norm(new_x_vector)
        angle_radians = np.arctan2(new_x_unit[1], new_x_unit[0]) + np.pi / 2
        displacement = np.array(image_center) - np.array(start_point)

        self.rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), np.rad2deg(angle_radians), 1.0)
        self.rotation_matrix[0, 2] += displacement[0]
        self.rotation_matrix[1, 2] += displacement[1]

        self.transformed_image = cv2.warpAffine(self.original_image, self.rotation_matrix, (w, h),
                                               flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=self.fill_value)

        print(f"Transformation calculated successfully")
        print(f"Rotation angle: {np.rad2deg(angle_radians):.2f} degrees")

    def run(self):
        """Run the interactive transformation tool"""
        self.original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if self.original_image is None:
            raise FileNotFoundError(f"Image not found at: {self.image_path}")

        h, w = self.original_image.shape[:2]
        print(f"Original resolution: {w} x {h}")

        cv2.namedWindow("Select Points", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Points", 800, 800)
        cv2.setMouseCallback("Select Points", self._mouse_callback)
        self._update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("Exiting...")
                break

            elif key == ord('u') or key == ord('U'):
                if len(self.selected_points) > 0:
                    removed = self.selected_points.pop()
                    point_num = "1st" if len(self.selected_points) == 0 else "2nd"
                    print(f"Undone {point_num} point: {removed}")
                    self.transform_calculated = False
                    self._update_display()

            elif key == ord('c') or key == ord('C'):
                if len(self.selected_points) == 2 and not self.transform_calculated:
                    self._calculate_transform()
                    self.transform_calculated = True

                    self.transformed_zoom = 1.0
                    self.transformed_pan = [0, 0]

                    cv2.namedWindow("Transformed Image", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Transformed Image", 800, 800)
                    cv2.setMouseCallback("Transformed Image", self._transformed_mouse_callback)
                    self._update_transformed_display()
                elif len(self.selected_points) < 2:
                    print("Please select 2 points first")

            elif key == ord('s') or key == ord('S'):
                if self.transform_calculated:
                    os.makedirs(self.save_dir, exist_ok=True)

                    # Extract filename from path
                    basename = os.path.splitext(os.path.basename(self.image_path))[0]

                    transformed_image_path = os.path.join(self.save_dir, f"{basename}_transformed.jpg")
                    rotation_matrix_path = os.path.join(self.save_dir, f"{basename}_rotation_matrix.npy")

                    th, tw = self.transformed_image.shape[:2]
                    if (th, tw) != (h, w):
                        print(f"WARNING: Resolution changed from {w}x{h} to {tw}x{th}")
                    else:
                        print(f"Resolution verified: {w} x {h}")

                    cv2.imwrite(transformed_image_path, self.transformed_image)
                    np.save(rotation_matrix_path, self.rotation_matrix)

                    print(f"Transformed image saved as '{transformed_image_path}'")
                    print(f"Rotation matrix saved as '{rotation_matrix_path}'")
                else:
                    print("Please calculate transformation first (press 'C')")

        cv2.destroyAllWindows()
