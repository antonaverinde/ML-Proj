"""
Test script for cylindrical image transformation.

Tests both forward (unwrap/undistort) and inverse (wrap/distort) transformations.
Can work with synthetic gradient patterns or real images.

Usage:
    python test_cylindrical_transform.py
"""

import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend - commented out to show plots
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from PIL import Image
import os


def load_image(image_path):
    """
    Load an image from file and convert to grayscale.

    Args:
        image_path: Path to image file

    Returns:
        img: Grayscale image as numpy array (H x W), normalized to [0, 1]
    """
    img = Image.open(image_path)

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    img_array = np.array(img, dtype=np.float32) / 255.0

    print(f"Loaded image from: {image_path}")
    print(f"  Size: {img_array.shape[0]} x {img_array.shape[1]}")
    print(f"  Range: [{img_array.min():.3f}, {img_array.max():.3f}]")

    return img_array




def planar_to_cylindrical(x, y, f, cx, cy, R):
    """
    Convert planar image coordinates (x, y) to cylindrical coordinates (θ, h).

    Args:
        x: Image x-coordinate (pixels)
        y: Image y-coordinate (pixels)
        f: Focal length (pixels)
        cx: Principal point x-coordinate (pixels)
        cy: Principal point y-coordinate (pixels)
        R: Cylinder radius

    Returns:
        theta, h: Cylindrical coordinates (angle, height)
    """
    theta = np.arctan2(x - cx, f)
    h = (y - cy) * R / (f * np.cos(theta))
    return theta, h


def cylindrical_to_planar(theta, h, f, cx, cy, R):
    """
    Convert cylindrical coordinates (θ, h) to planar image coordinates (x, y).

    Args:
        theta: Angle coordinate (radians)
        h: Height coordinate
        f: Focal length (pixels)
        cx: Principal point x-coordinate (pixels)
        cy: Principal point y-coordinate (pixels)
        R: Cylinder radius

    Returns:
        x, y: Planar image coordinates
    """
    x = cx + f * np.tan(theta)
    y = cy + h * f * np.cos(theta) / R
    return x, y


def unwarp_cylindrical(img, f, R, cx=None, cy=None, return_mapping=False):
    """
    Unwarp a cylindrical image to planar coordinates (INVERSE TRANSFORM).

    Converts a distorted cylindrical image back to planar representation.
    Use this to straighten/flatten images of cylindrical surfaces.

    Args:
        img: Input distorted image (H x W)
        f: Focal length (pixels)
        R: Cylinder radius
        cx: Principal point x (default: image center)
        cy: Principal point y (default: image center)
        return_mapping: If True, returns (unwarped_img, x_src, y_src)

    Returns:
        unwarped_img: Unwarped planar image
        OR (if return_mapping=True):
        unwarped_img, x_src, y_src: Image and coordinate mapping arrays
    """
    height, width = img.shape

    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2

    # Create output coordinate grids
    y_out, x_out = np.mgrid[0:height, 0:width]

    # Convert output coordinates to cylindrical using inverse transform
    theta = -np.arctan2(x_out - cx, f)  # Negated angle for inverse
    h = (y_out - cy) / np.sqrt((x_out - cx)**2 + f**2)  # New height formula

    # Determine angular and height ranges
    theta_min, theta_max = theta.min(), theta.max()
    h_min, h_max = h.min(), h.max()

    # Map cylindrical coordinates to normalized [0, width-1] and [0, height-1]
    # Reverse x mapping to fix horizontal mirroring
    x_src = (theta_max - theta) / (theta_max - theta_min) * (width - 1)
    y_src = (h - h_min) / (h_max - h_min) * (height - 1)

    # Sample from source image using bilinear interpolation
    coords = np.array([y_src, x_src])
    unwarped_img = map_coordinates(img, coords, order=1, mode='constant', cval=0)

    if return_mapping:
        return unwarped_img, x_src, y_src
    else:
        return unwarped_img


def warp_cylindrical(img, f, R, cx=None, cy=None, return_mapping=False):
    """
    Warp a planar image to cylindrical coordinates (FORWARD TRANSFORM).

    Applies cylindrical distortion to simulate photographing a planar pattern
    on a cylinder.

    Args:
        img: Input planar image (H x W)
        f: Focal length (pixels)
        R: Cylinder radius
        cx: Principal point x (default: image center)
        cy: Principal point y (default: image center)
        return_mapping: If True, returns (warped_img, x_src, y_src)

    Returns:
        warped_img: Cylindrically distorted image
        OR (if return_mapping=True):
        warped_img, x_src, y_src: Image and coordinate mapping arrays
    """
    height, width = img.shape

    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2

    # Create output coordinate grids
    y_out, x_out = np.mgrid[0:height, 0:width]

    # For each output pixel, find the corresponding cylindrical coordinates
    theta, h = planar_to_cylindrical(x_out, y_out, f, cx, cy, R)

    # Map to source image coordinates
    # Assume cylindrical space spans the full input image
    theta_min, theta_max = theta.min(), theta.max()
    h_min, h_max = h.min(), h.max()

    # Inverse mapping: where in the source image should we sample from?
    x_src = (theta - theta_min) / (theta_max - theta_min) * (width - 1)
    y_src = (h - h_min) / (h_max - h_min) * (height - 1)

    # Sample from source image
    coords = np.array([y_src, x_src])
    warped_img = map_coordinates(img, coords, order=1, mode='constant', cval=0)

    if return_mapping:
        return warped_img, x_src, y_src
    else:
        return warped_img


def test_transform(image_source='gradient_vertical', f=1000, R=100,
                   transform_type='both', save_output=True, save_transform_matrix=True,
                   output_dir=None):
    """
    Test cylindrical transformation with various options.

    Args:
        image_source: 'gradient_vertical', 'gradient_horizontal', or path to image file
        f: Focal length (pixels)
        R: Cylinder radius
        transform_type: 'forward' (warp), 'inverse' (unwarp), or 'both'
        save_output: Whether to save transformed images
        save_transform_matrix: Whether to save transformation coordinate mapping
        output_dir: Directory to save transformed images and matrices
    """
    # Set default output directory if not provided
    if output_dir is None:
        try:
            from config import BINARY_MASKS_PATH
            output_dir = str(BINARY_MASKS_PATH / "Barrel_Images2_croped_Transformed_V2")
        except ImportError:
            # Fallback if config not available
            output_dir = "Barrel_Images2_croped_Transformed_V2"

    print("=" * 70)
    print(f"Cylindrical Transformation Test")
    print("=" * 70)
    print(f"Image source: {image_source}")
    print(f"Focal length (f): {f} pixels")
    print(f"Cylinder radius (R): {R} units")
    print(f"Transform type: {transform_type}")


    # Assume it's a file path
    print(f"\nLoading image from file...")
    img = load_image(image_source)
    input_filename = os.path.basename(image_source)
    source_name = os.path.splitext(input_filename)[0]

    # Create output directory if it doesn't exist and we're saving
    if save_output:
        os.makedirs(output_dir, exist_ok=True)

    # Apply transformations
    results = {'original': img}

    if transform_type in ['forward', 'both']:
        print("\nApplying FORWARD transform (warp to cylindrical)...")
        img_warped, x_map, y_map = warp_cylindrical(img, f, R, return_mapping=True)
        results['warped'] = img_warped
        print(f"  Warped range: [{img_warped.min():.3f}, {img_warped.max():.3f}]")

        # Save warped image
        if save_output:
            output_path = os.path.join(output_dir, input_filename)
            img_to_save = (img_warped * 255).astype(np.uint8)
            Image.fromarray(img_to_save, mode='L').save(output_path)
            print(f"  Saved warped image to: {output_path}")

        # Save transformation parameters
        if save_transform_matrix:
            base_name = os.path.splitext(input_filename)[0]
            params_path = os.path.join(output_dir, f"{base_name}_forward_params.npz")
            np.savez(params_path, f=f, R=R, transform_type='forward')
            print(f"  Saved forward transformation parameters to: {params_path}")

    if transform_type in ['inverse', 'both']:
        print("\nApplying INVERSE transform (unwarp from cylindrical)...")
        img_unwarped, x_map, y_map = unwarp_cylindrical(img, f, R, return_mapping=True)
        results['unwarped'] = img_unwarped
        print(f"  Unwarped range: [{img_unwarped.min():.3f}, {img_unwarped.max():.3f}]")

        # Save unwarped image
        if save_output:
            output_path = os.path.join(output_dir, input_filename)
            img_to_save = (img_unwarped * 255).astype(np.uint8)
            Image.fromarray(img_to_save, mode='L').save(output_path)
            print(f"  Saved unwarped image to: {output_path}")

        # Save transformation parameters
        if save_transform_matrix:
            base_name = os.path.splitext(input_filename)[0]
            params_path = os.path.join(output_dir, f"{base_name}_inverse_params.npz")
            np.savez(params_path, f=f, R=R, transform_type='inverse')
            print(f"  Saved inverse transformation parameters to: {params_path}")

    # Visualization
    num_plots = len(results)
    fig, axes = plt.subplots(1, num_plots, figsize=(8*num_plots, 12))

    if num_plots == 1:
        axes = [axes]

    titles = {
        'original': 'Original Image',
        'warped': f'Forward: Warped\n(f={f}, R={R})',
        'unwarped': f'Inverse: Unwarped\n(f={f}, R={R})'
    }

    for ax, (key, img_data) in zip(axes, results.items()):
        im = ax.imshow(img_data, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title(titles[key], fontsize=12)
        #ax.set_xlabel('Width (pixels)')
        #ax.set_ylabel('Height (pixels)')
        
        #plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return results




