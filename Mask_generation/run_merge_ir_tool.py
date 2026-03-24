"""
Script to run the interactive IR image merge tool across multiple locations.

This script:
1. Loads visible image (S{i}_Merged.jpg)
2. Iterates through IR measurement locations
3. For each location:
   - Presents first power/side mode for manual alignment
   - User selects correspondence points and previews merge
   - Upon save, automatically applies transformation to other power/side modes
   - Only proceeds to next location after successful save

Folder Structure:
-----------------
IR data folders follow pattern: {prefix}_{location}
Example: tarisir_zika_50hz_veryclose_5sec_2kw_left_pos3_bottom_left

Where:
- {prefix} contains power and side information
- {location} is the measurement location (e.g., bottom_left, top_right)
- Each location typically has 3 power/side mode variations

Usage:
------
1. Set SAMPLE_INDEX to desired sample number
2. Verify IMAGE1_BASE_PATH points to merged visible images
3. Verify IR_BASE_PATH points to IR measurement data
4. Run the script
5. For each location:
   - Select correspondence points on both images
   - Preview the merge
   - Press 'S' to save and proceed to next location
   - Press ESC to skip this location
"""

from ImageMergeToolIr import ImageMergeToolIr
from ImageMergeToolIrMulti import ImageMergeToolIrMulti
from config import BINARY_MASKS_PATH, IR_DATA_PATH
import os
import glob
import cv2


# ========== Configuration ==========
# Sample index
SAMPLE_INDEX = 5

# Path to the visible merged image
IMAGE1_BASE_PATH = str(BINARY_MASKS_PATH / "Barrel_Images2_croped_Merged_V2")
# Alternative path (commented): Barrel_Images2_Merged_Cylinder
# IMAGE1_BASE_PATH = str(BINARY_MASKS_PATH / "Barrel_Images2_Merged_Cylinder")

# Base path to IR measurement data
IR_BASE_PATH = str(IR_DATA_PATH)

# Flip visible image horizontally?
FLIP_IMAGE1 = False

# Rotate visible reference image by 180 degrees?
ROTATE_REFERENCE = True  # Set to True to rotate reference image 180°

# NPZ filename patterns
NPZ_FILENAME_280 = "PPT_a=0_width=280.npz"
NPZ_FILENAME_110 = "PPT_a=0_width=110.npz"

# Start from arbitrary location (1-9, or None to start from beginning)
# Example: START_FROM_LOCATION = 3 to start from the 3rd location
START_FROM_LOCATION = None

# MULTI-IMAGE MODE: Set to True to use scrollable image lists with Phase+Amp data
# When enabled, loads Amp[0,1] + Phase[1-5] from 280.npz and Phase[1-2] from 110.npz
# Total: 9 IR images
USE_MULTI_IMAGE_MODE = True

# Multi-image mode settings (only used when USE_MULTI_IMAGE_MODE = True)
USE_COLORMAP = True  # Apply viridis colormap to IR images
COLORMAP = cv2.COLORMAP_VIRIDIS  # Colormap to use
HISTOGRAM_THRESHOLD_LOW = 0.25   # Factor for left threshold (0.0-1.0)
HISTOGRAM_THRESHOLD_HIGH = 0.30  # Factor for right threshold (0.0-1.0)
HISTOGRAM_BINS = 50              # Number of histogram bins

# Homography computation method
# 0: All points method (standard, least squares)
# cv2.RANSAC: RANSAC-based robust method
# cv2.LMEDS: Least-Median robust method
# cv2.RHO: PROSAC-based robust method
HOMOGRAPHY_METHOD = cv2.LMEDS  # Use 0 for standard method (all points, no outlier rejection)

# RESEARCH MODE: Set to True to compare different transformation methods
# When enabled, runs transformation research instead of normal merge
RESEARCH_MODE = False
# ===================================


def find_location_groups(measure_dir):
    """
    Find all IR data folders and group them by location.

    Parameters:
    -----------
    measure_dir : str
        Path to measure_S{i} directory

    Returns:
    --------
    dict : Dictionary mapping location names to lists of folder paths
           Example: {'bottom_left': [folder1, folder2, folder3], ...}
    """
    postprocessed_dir = os.path.join(measure_dir, "postprocessed_data")

    if not os.path.exists(postprocessed_dir):
        raise FileNotFoundError(f"Postprocessed data directory not found: {postprocessed_dir}")

    # Find all subdirectories
    all_folders = [f for f in glob.glob(os.path.join(postprocessed_dir, "*"))
                   if os.path.isdir(f)]

    if not all_folders:
        raise ValueError(f"No folders found in: {postprocessed_dir}")

    # Group by location (last two parts: e.g., "bottom_left", "top_right")
    location_groups = {}
    for folder in all_folders:
        folder_name = os.path.basename(folder)

        # Extract location (last two parts after splitting by '_')
        parts = folder_name.rsplit('_', 2)
        if len(parts) < 3:
            print(f"Warning: Skipping folder with unexpected name format: {folder_name}")
            continue

        location = f"{parts[1]}_{parts[2]}"

        if location not in location_groups:
            location_groups[location] = []

        location_groups[location].append(folder)

    # Sort folders within each location for consistency
    for location in location_groups:
        location_groups[location].sort()

    return location_groups


def process_location(image1_path, location_folders, location_name, research_mode=False):
    """
    Process a single location by merging visible image with IR data.

    Parameters:
    -----------
    image1_path : str
        Path to visible merged image
    location_folders : list
        List of folder paths for this location
    location_name : str
        Name of the location (e.g., 'bottom_left')
    research_mode : bool
        If True, run transformation comparison instead of normal merge

    Returns:
    --------
    bool : True if successfully saved, False if cancelled
    """
    print("\n" + "="*70)
    print(f"PROCESSING LOCATION: {location_name}")
    print("="*70)
    print(f"Found {len(location_folders)} power/side mode variations:")
    for i, folder in enumerate(location_folders, 1):
        print(f"  {i}. {os.path.basename(folder)}")
    print("="*70)

    # Use 4kw_both folder for manual alignment if available, otherwise first folder
    primary_folder = None
    for folder in location_folders:
        folder_name = os.path.basename(folder)
        if '4kw_both' in folder_name.lower():
            primary_folder = folder
            break

    if primary_folder is None:
        print("Warning: No 4kw_both folder found, using first folder")
        primary_folder = location_folders[0]

    npz_path_280 = os.path.join(primary_folder, NPZ_FILENAME_280)
    npz_path_110 = os.path.join(primary_folder, NPZ_FILENAME_110)

    print(f"\nUsing primary folder for manual alignment:")
    print(f"  {os.path.basename(primary_folder)}")

    if research_mode:
        # Run transformation comparison research
        from TransformationResearch import research_mode as run_research
        print("\n*** RESEARCH MODE ENABLED ***")
        print("Comparing different transformation methods...")

        results = run_research(
            image1_path=image1_path,
            npz_path=npz_path_280,
            flip_image1=FLIP_IMAGE1
        )

        # Research mode doesn't save results, just displays comparison
        return results is not None
    elif USE_MULTI_IMAGE_MODE:
        # Multi-image mode with scrollable Phase images
        print("\n*** MULTI-IMAGE MODE ENABLED ***")
        print("Using scrollable image lists with Phase data")
        print(f"Colormap: {'Viridis' if USE_COLORMAP else 'Grayscale'}")
        print(f"Histogram thresholds: LOW={HISTOGRAM_THRESHOLD_LOW}, HIGH={HISTOGRAM_THRESHOLD_HIGH}")

        if not os.path.exists(npz_path_280):
            print(f"Error: NPZ file not found: {npz_path_280}")
            return False

        # Check if 110 file exists (optional)
        if not os.path.exists(npz_path_110):
            print(f"Warning: NPZ file not found: {npz_path_110} (will only load from 280.npz)")

        print(f"\nTransformation will be auto-applied to {len(location_folders)-1} other variations")

        # Create and run multi-image merge tool
        tool = ImageMergeToolIrMulti(
            image1_path=image1_path,
            npz_path_280=npz_path_280,
            npz_path_110=npz_path_110,
            flip_image1=FLIP_IMAGE1,
            rotate_reference=ROTATE_REFERENCE,
            use_colormap=USE_COLORMAP,
            histogram_threshold_low=HISTOGRAM_THRESHOLD_LOW,
            histogram_threshold_high=HISTOGRAM_THRESHOLD_HIGH,
            histogram_bins=HISTOGRAM_BINS,
            homography_method=HOMOGRAPHY_METHOD
        )

        success = tool.run()

        return success
    else:
        # Normal merge mode (original single-image version)
        if not os.path.exists(npz_path_280):
            print(f"Error: NPZ file not found: {npz_path_280}")
            return False

        print(f"\nTransformation will be auto-applied to {len(location_folders)-1} other variations")

        # Create and run merge tool
        tool = ImageMergeToolIr(
            image1_path=image1_path,
            npz_path=npz_path_280,
            flip_image1=FLIP_IMAGE1
        )

        success = tool.run()

        return success


def main():
    """Main execution function"""
    print("="*70)
    if RESEARCH_MODE:
        print("IR IMAGE MERGE TOOL - RESEARCH MODE")
        print("="*70)
        print("*** TRANSFORMATION COMPARISON MODE ***")
        print("\nThis mode will:")
        print("  - Let you select correspondence points")
        print("  - Compare 6 different transformation methods")
        print("  - Show visual comparison with IR reference")
        print("  - Not save any results (research only)")
    elif USE_MULTI_IMAGE_MODE:
        print("IR IMAGE MERGE TOOL - MULTI-IMAGE MODE")
        print("="*70)
        print("*** SCROLLABLE IMAGE LISTS ***")
        print("\nThis mode features:")
        print("  - Scrollable IR images (N/P keys, 1-9 for direct jump)")
        print("  - Amp[0,1] + Phase[1-5] from PPT_a=0_width=280.npz")
        print("  - Phase[1-2] from PPT_a=0_width=110.npz")
        print("  - Total: 9 IR images")
        print(f"  - Viridis colormap: {'Enabled' if USE_COLORMAP else 'Disabled'}")
        print(f"  - Histogram thresholding: LOW={HISTOGRAM_THRESHOLD_LOW}, HIGH={HISTOGRAM_THRESHOLD_HIGH}")
        print("  - Correspondence points persist across all images")
        print("  - Visible reference window in step 1")
        print(f"  - Rotate reference: {'Enabled' if ROTATE_REFERENCE else 'Disabled'}")
    else:
        print("IR IMAGE MERGE TOOL - BATCH PROCESSOR")
    print("="*70)
    print(f"Sample Index: {SAMPLE_INDEX}")
    print(f"Research Mode: {RESEARCH_MODE}")
    print(f"Multi-Image Mode: {USE_MULTI_IMAGE_MODE}")
    print("="*70)

    # Construct paths
    image1_filename = f"S{SAMPLE_INDEX}_Merged.jpg"
    image1_path = os.path.join(IMAGE1_BASE_PATH, image1_filename)

    measure_dir = os.path.join(IR_BASE_PATH, f"measure_S{SAMPLE_INDEX}")

    # Verify paths exist
    if not os.path.exists(image1_path):
        print(f"Error: Visible image not found: {image1_path}")
        return

    if not os.path.exists(measure_dir):
        print(f"Error: Measurement directory not found: {measure_dir}")
        return

    print(f"\nVisible Image: {image1_path}")
    print(f"IR Data Base:  {measure_dir}")

    # Find location groups
    print("\nScanning for IR measurement locations...")
    try:
        location_groups = find_location_groups(measure_dir)
    except Exception as e:
        print(f"Error finding location groups: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nFound {len(location_groups)} unique locations:")
    for location, folders in sorted(location_groups.items()):
        print(f"  - {location}: {len(folders)} power/side variations")

    # Process each location sequentially
    total_locations = len(location_groups)
    processed_count = 0
    skipped_count = 0
    last_processed_location = None
    interrupted = False

    # Determine starting index
    start_idx = START_FROM_LOCATION if START_FROM_LOCATION is not None else 1
    if start_idx > total_locations:
        print(f"\nError: START_FROM_LOCATION ({start_idx}) exceeds total locations ({total_locations})")
        return
    if start_idx > 1:
        print(f"\n*** STARTING FROM LOCATION {start_idx} (skipping first {start_idx-1} locations) ***\n")

    sorted_locations = sorted(location_groups.items())

    for idx, (location_name, location_folders) in enumerate(sorted_locations, 1):
        # Skip locations before start_idx
        if idx < start_idx:
            continue

        print(f"\n{'='*70}")
        print(f"LOCATION {idx}/{total_locations}: {location_name}")
        print(f"{'='*70}")

        success = process_location(image1_path, location_folders, location_name, research_mode=RESEARCH_MODE)

        last_processed_location = (idx, location_name)

        if success:
            processed_count += 1
            print(f"\n✓ Location '{location_name}' completed successfully!")
        else:
            skipped_count += 1
            print(f"\n✗ Location '{location_name}' was skipped or cancelled")

            # Check if user pressed ESC (tool returned False)
            # If so, stop immediately without prompting
            print("\n" + "="*70)
            print("PROCESSING INTERRUPTED BY USER (ESC)")
            print("="*70)
            print(f"\nLast location processed: #{idx} - {location_name}")
            if idx < total_locations:
                print(f"\nTo continue from where you left off:")
                print(f"  Set START_FROM_LOCATION = {idx + 1}")
                print(f"  Next location: #{idx + 1} - {sorted_locations[idx][0] if idx < len(sorted_locations) else 'N/A'}")
            interrupted = True
            break

        # Prompt to continue or stop
        if idx < total_locations:
            print(f"\nProgress: {idx}/{total_locations} locations reviewed")
            print(f"Processed: {processed_count} | Skipped: {skipped_count}")
            print("\nPress Enter to continue to next location, Ctrl+C to stop, or ESC to exit...")
            try:
                user_input = input()
                # Check if ESC was entered (though input() doesn't capture raw ESC)
                if user_input.lower() == 'esc' or user_input.lower() == 'q':
                    print("\n\nBatch processing stopped by user (ESC)")
                    interrupted = True
                    break
            except KeyboardInterrupt:
                print("\n\nBatch processing interrupted by user (Ctrl+C)")
                interrupted = True
                break

    # Final summary
    print("\n" + "="*70)
    if interrupted:
        print("BATCH PROCESSING INTERRUPTED")
    else:
        print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total locations: {total_locations}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped/Cancelled: {skipped_count}")

    if last_processed_location is not None:
        idx, name = last_processed_location
        print(f"\nLast processed location: {idx}. {name}")

        if interrupted and idx < total_locations:
            next_idx = idx + 1
            next_name = sorted_locations[next_idx - 1][0] if next_idx <= total_locations else "N/A"
            print(f"Next location to process: {next_idx}. {next_name}")
            print(f"\nTo continue from location {next_idx}, set:")
            print(f"    START_FROM_LOCATION = {next_idx}")

    print("="*70)


if __name__ == "__main__":
    main()
