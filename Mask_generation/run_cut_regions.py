"""
Script to run the interactive region crop tool across multiple IR measurement locations.

This script:
1. Iterates through IR measurement locations for a given sample
2. For each location:
   - Displays merged image
   - User selects rectangular crop region
   - Crops all npy/npz/jpg files in all power/side mode folders
   - Saves cropped data to 'cutted_V1' subfolder
3. Only proceeds to next location after successful save

Folder Structure:
-----------------
IR data folders follow pattern: {prefix}_{location}
Example: tarisir_zika_50hz_veryclose_5sec_2kw_left_pos3_bottom_left

Where:
- {prefix} contains power and side information
- {location} is the measurement location (e.g., bottom_left, top_right)
- Each location typically has 3 power/side mode variations

Output Structure:
-----------------
For each location folder, creates:
  location_folder/
    └── cutted_V1/
        ├── file1.npy (cropped)
        ├── file2.npz (cropped)
        └── merged_image.jpg (cropped)

Usage:
------
1. Set SAMPLE_INDEX to desired sample number
2. Verify IR_BASE_PATH points to IR measurement data
3. Run the script
4. For each location:
   - Click and drag to select crop region
   - Press 'S' to save and proceed to next location
   - Press 'R' to reset selection
   - Press ESC to skip this location
"""

from RunCutTool import RegionCutTool
from config import IR_DATA_PATH
import os
import glob


# ========== Configuration ==========
# Sample index
SAMPLE_INDEX = 6

# Base path to IR measurement data
IR_BASE_PATH = str(IR_DATA_PATH)

# NPZ filename pattern
NPZ_FILENAME = "PPT_a=0_width=280.npz"

# Start from arbitrary location (1-9, or None to start from beginning)
# Example: START_FROM_LOCATION = 3 to start from the 3rd location
START_FROM_LOCATION = None

# Output subfolder name (will be created in each location folder)
OUTPUT_SUBFOLDER = "cutted_V1"
SUBPATH="Masks_V3"
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
        mask_dir = os.path.join(folder, SUBPATH)

        # 🔹 skip or warn if Masks_V3 does not exist
        if not os.path.isdir(mask_dir):
            print(f"[WARNING] Missing {SUBPATH}: {mask_dir}")
            continue
        if location not in location_groups:
            location_groups[location] = []
        location_groups.setdefault(location, []).append(mask_dir)
        #location_groups[location].append(folder)

    # Sort folders within each location for consistency
    for location in location_groups:
        location_groups[location].sort()
   
    for location, folders in location_groups.items():
        print(f"Location: {location}")
        for f in folders:
            print(f"  - {f}")
    return location_groups


def process_location(location_folders, location_name):
    """
    Process a single location by cropping region from all files.

    Parameters:
    -----------
    location_folders : list
        List of folder paths for this location
    location_name : str
        Name of the location (e.g., 'bottom_left')

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

    # Create and run crop tool
    tool = RegionCutTool(
        location_folders=location_folders,
        location_name=location_name,
        npz_filename=NPZ_FILENAME,SUBPATH=SUBPATH
    )

    success = tool.run()

    return success


def main():
    """Main execution function"""
    print("="*70)
    print("IR REGION CROP TOOL - BATCH PROCESSOR")
    print("="*70)
    print(f"Sample Index: {SAMPLE_INDEX}")
    print(f"NPZ Filename: {NPZ_FILENAME}")
    print(f"Output subfolder: {OUTPUT_SUBFOLDER}")
    print("="*70)

    # Construct measurement directory path
    measure_dir = os.path.join(IR_BASE_PATH, f"measure_S{SAMPLE_INDEX}")

    # Verify path exists
    if not os.path.exists(measure_dir):
        print(f"Error: Measurement directory not found: {measure_dir}")
        return

    print(f"\nIR Data Base: {measure_dir}")

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

        success = process_location(location_folders, location_name)

        last_processed_location = (idx, location_name)

        if success:
            processed_count += 1
            print(f"\n✓ Location '{location_name}' completed successfully!")
        else:
            skipped_count += 1
            print(f"\n✗ Location '{location_name}' was skipped or cancelled")

            # User pressed ESC - stop processing
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
            print("\nPress Enter to continue to next location, or Ctrl+C to stop...")
            try:
                input()
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
# if __name__ == "__main__":
#     measure_dir = r"//Gfs01/g71/Thermo_Daten-MX2/2025/2025-11-04-Av-ZIKA-Mirko-Taris-Hologen-2kw-measurements/Taris/Experiment_1/measure_S2"

#     folders = find_location_groups(measure_dir)
   
