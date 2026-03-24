"""
Batch runner for ImprovedMaskTool

This script:
1. Iterates through samples
2. For each sample, finds all measurement locations
3. For each location (containing 3 regime/power folders):
   - Applies homography transformation to masks
   - Saves transformed masks (MaskV2.npy, MaskV2_2sDiff.npy)
   - Saves visualization (mask_comparison.png)

Usage:
------
1. Set SAMPLE_INDICES to list of samples to process
2. Run the script
3. Transformed masks will be saved in each location folder
"""

from ImprovedMaskTool import ImprovedMaskTool
from config import BINARY_MASKS_PATH, IR_DATA_PATH
import os
import glob


# ========== Configuration ==========
# Sample indices to process
SAMPLE_INDICES = [1, 2, 3]  # Adjust as needed

# Base path to mask directory
MASK_BASE_PATH = str(BINARY_MASKS_PATH / "Barrel_Images2_croped_Merged_V2_Mask")

# Base path to IR measurement data
IR_BASE_PATH = str(IR_DATA_PATH)

# Start from specific sample (or None to start from beginning)
START_FROM_SAMPLE = None

# Skip already processed locations (where MaskV2.npy already exists)
SKIP_EXISTING = True
# ===================================


def find_all_location_folders(measure_dir):
    """
    Find all IR data folders (all regime/power variations).

    Parameters:
    -----------
    measure_dir : str
        Path to measure_S{i} directory

    Returns:
    --------
    list : List of all folder paths
    """
    postprocessed_dir = os.path.join(measure_dir, "postprocessed_data")

    if not os.path.exists(postprocessed_dir):
        raise FileNotFoundError(f"Postprocessed data directory not found: {postprocessed_dir}")

    # Find all subdirectories
    all_folders = [f for f in glob.glob(os.path.join(postprocessed_dir, "*"))
                   if os.path.isdir(f)]

    if not all_folders:
        raise ValueError(f"No folders found in: {postprocessed_dir}")

    # Sort for consistency
    all_folders.sort()

    return all_folders


def process_sample(sample_number):
    """
    Process all locations for a given sample.

    Parameters:
    -----------
    sample_number : int
        Sample number to process

    Returns:
    --------
    tuple : (success_count, skipped_count, failed_count)
    """
    print("\n" + "="*70)
    print(f"PROCESSING SAMPLE {sample_number}")
    print("="*70)

    measure_dir = os.path.join(IR_BASE_PATH, f"measure_S{sample_number}")

    # Verify measurement directory exists
    if not os.path.exists(measure_dir):
        print(f"Error: Measurement directory not found: {measure_dir}")
        return (0, 0, 1)

    # Find all location folders
    print("\nScanning for location folders...")
    try:
        location_folders = find_all_location_folders(measure_dir)
    except Exception as e:
        print(f"Error finding location folders: {e}")
        import traceback
        traceback.print_exc()
        return (0, 0, 1)

    print(f"Found {len(location_folders)} location folders")

    # Process each location
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, location_folder in enumerate(location_folders, 1):
        folder_name = os.path.basename(location_folder)
        print(f"\n[{idx}/{len(location_folders)}] {folder_name}")

        # Check if already processed
        output_mask_path = os.path.join(location_folder, "MaskV2.npy")
        if SKIP_EXISTING and os.path.exists(output_mask_path):
            print(f"  ⊘ Skipping (already processed)")
            skipped_count += 1
            continue

        # Process this location
        tool = ImprovedMaskTool(sample_number, location_folder, MASK_BASE_PATH)
        success = tool.process()

        if success:
            success_count += 1
        else:
            failed_count += 1

    # Summary for this sample
    print("\n" + "-"*70)
    print(f"Sample {sample_number} Summary:")
    print(f"  Total locations: {len(location_folders)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped (existing): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print("-"*70)

    return (success_count, skipped_count, failed_count)


def main():
    """Main execution function"""
    print("="*70)
    print("IMPROVED MASK BATCH PROCESSOR")
    print("="*70)
    print(f"Samples to process: {SAMPLE_INDICES}")
    print(f"Skip existing: {SKIP_EXISTING}")
    print("="*70)

    # Verify mask base path exists
    if not os.path.exists(MASK_BASE_PATH):
        print(f"Error: Mask base path not found: {MASK_BASE_PATH}")
        return

    # Determine starting index
    start_idx = 0
    if START_FROM_SAMPLE is not None:
        try:
            start_idx = SAMPLE_INDICES.index(START_FROM_SAMPLE)
            print(f"\n*** STARTING FROM SAMPLE {START_FROM_SAMPLE} ***\n")
        except ValueError:
            print(f"Warning: START_FROM_SAMPLE {START_FROM_SAMPLE} not in SAMPLE_INDICES")
            return

    # Process each sample
    total_success = 0
    total_skipped = 0
    total_failed = 0

    for sample_idx in SAMPLE_INDICES[start_idx:]:
        try:
            success, skipped, failed = process_sample(sample_idx)
            total_success += success
            total_skipped += skipped
            total_failed += failed

        except KeyboardInterrupt:
            print("\n\nBatch processing interrupted by user (Ctrl+C)")
            break

        except Exception as e:
            print(f"\nUnexpected error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1

    # Final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total samples processed: {len(SAMPLE_INDICES[start_idx:])}")
    print(f"Total locations successfully processed: {total_success}")
    print(f"Total locations skipped (existing): {total_skipped}")
    print(f"Total locations failed: {total_failed}")
    print("="*70)


if __name__ == "__main__":
    main()
