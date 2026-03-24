# Mask Generation Tools

A comprehensive image processing pipeline for thermal imaging mask generation, including cylindrical image flattening, image alignment, and mask transformation tools.

## Overview

This toolkit processes thermographic images from laser pulse thermography measurements, performing:
- Cylindrical image correction and flattening
- Interactive image rotation and alignment
- Visible and IR image merging
- Mask generation and transformation
- Region cropping and cleanup

## Setup

### Prerequisites

- Python 3.7+
- Access to your local data directories (visible images and IR measurement data)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Mask_generation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure paths:**
   ```bash
   cp .env.example .env
   ```

4. **Edit `.env` with your local paths:**

   Open `.env` in a text editor and replace the placeholder paths:

   ```env
   # Base directory for binary masks processing
   BINARY_MASKS_BASE=/path/to/your/BinaryMasks

   # Base directory for IR measurement data
   IR_DATA_BASE=/path/to/your/IR/data/Experiment_1

   # Optional: Parent project directory
   # PROJECT_BASE=/path/to/your/project
   ```

   **Example (Windows):**
   ```env
   BINARY_MASKS_BASE=C:\Users\YourName\project\BinaryMasks
   IR_DATA_BASE=D:\Data\IR_Measurements\Experiment_1
   ```

   **Example (Linux):**
   ```env
   BINARY_MASKS_BASE=/home/yourname/project/BinaryMasks
   IR_DATA_BASE=/mnt/data/IR_Measurements/Experiment_1
   ```

## Processing Pipeline

See [`Execution_order.md`](Execution_order.md) for the complete step-by-step workflow.

**Quick Overview:**
1. Image rotation and axis alignment (`run_rotation_tool.py`)
2. Cylindrical flattening (`run_flatten_V2.py`)
3. Visible image merging (`run_merge_tool.py`)
4. IR and visible image alignment (`run_merge_ir_tool.py`)
5. Mask transformation (`run_improved_masks_load.py`)
6. Region cropping (`run_cut_regions.py`)

## Key Tools

### Core Processing Classes

- **`CylinderFlattener_V2.py`** - Cylindrical image transformation (flatten/unwrap)
- **`ImageRotationTool.py`** - Interactive image rotation and displacement
- **`ImageMergeToolIrMulti.py`** - IR and visible image merging with scrollable multi-channel support
- **`ImprovedMaskTool.py`** - Apply homography transformations to masks

### Executable Scripts

All `run_*.py` scripts are configured via their internal configuration sections. Simply edit the parameters at the top of each file before running.

**Examples:**

```bash
# Rotate and align images
python run_rotation_tool.py

# Flatten cylindrical images
python run_flatten_V2.py

# Merge IR and visible images
python run_merge_ir_tool.py
```

## Interactive Tools

Many tools provide interactive OpenCV windows for point selection:

**Controls:**
- **Left Click:** Select correspondence points
- **Right Drag:** Pan the image
- **Ctrl + Mouse Wheel:** Zoom in/out
- **U:** Undo last point
- **N/P:** Navigate through image lists (multi-image mode)
- **Q:** Finish selection
- **S:** Save and proceed
- **ESC:** Cancel

## Data Organization

### Expected Directory Structure

```
BinaryMasks/                          # BINARY_MASKS_BASE
├── Barrel_Images2_Masked/            # Masked visible images
├── Barrel_Images2_croped_NewAxes/    # Rotated images
├── Barrel_Images2_croped_Transformed_V2/  # Flattened images
├── Barrel_Images2_croped_Merged_V2/  # Merged visible images
└── Barrel_Images2_croped_Merged_V2_Mask/  # Generated masks

IR_Data/                              # IR_DATA_BASE
└── measure_S{i}/                     # Sample measurements
    └── postprocessed_data/
        └── {prefix}_{location}/      # Power/position variants
            ├── PPT_a=0_width=280.npz
            ├── PPT_a=0_width=110.npz
            └── Masks_V3/             # Generated masks
```

### Output Files

Each processing step creates outputs in designated subdirectories:
- Transformed images: `.jpg` files
- Transformation matrices: `.npy` files
- Masks: `.npy` files (binary or grayscale)
- Visualizations: `.png` files

## Configuration

All path configuration is managed via environment variables in `.env`:

```python
from config import BINARY_MASKS_PATH, IR_DATA_PATH

# Use paths in your code
image_path = BINARY_MASKS_PATH / "subfolder" / "image.jpg"
```

**Available paths:**
- `BINARY_MASKS_PATH` - Base directory for mask processing
- `IR_DATA_PATH` - Base directory for IR measurements
- `PROJECT_PATH` - (Optional) Parent project directory

## Troubleshooting

### Configuration Errors

If you see:
```
Configuration Error: Environment variable 'BINARY_MASKS_BASE' is not set.
```

**Solution:**
1. Ensure `.env` file exists in the BinaryMasks directory
2. Check that paths are set correctly in `.env`
3. Verify no typos in variable names

### Path Issues

- **Windows:** Use either forward slashes (`/`) or double backslashes (`\\`) in `.env`
  ```env
  BINARY_MASKS_BASE=C:/Users/YourName/project/BinaryMasks
  # OR
  BINARY_MASKS_BASE=C:\\Users\\YourName\\project\\BinaryMasks
  ```

- **Network paths:** Use forward slashes for consistency
  ```env
  IR_DATA_BASE=//server/share/path/to/data
  ```

### Import Errors

If tools fail to import `config`:
- Ensure you're running scripts from the BinaryMasks directory
- Check that `config.py` and `.env` are in the same directory

## Contributing

When contributing, ensure:
1. Never commit `.env` file (it's gitignored)
2. Update `.env.example` if adding new environment variables
3. Use `config.py` for all path references
4. Test with fresh `.env` setup

## File Reference

**Configuration:**
- `config.py` - Path loading and validation
- `.env` - Your local paths (gitignored)
- `.env.example` - Template for new users
- `requirements.txt` - Python dependencies

**Processing Tools:**
- `CylinderFlattener_V2.py` - Cylindrical transformations
- `ImageRotationTool.py` - Image rotation
- `ImageMergeTool.py` - Visible image merging
- `ImageMergeToolIrMulti.py` - IR/visible merging (multi-channel)
- `ImprovedMaskTool.py` - Mask transformations

**Executable Scripts:**
- `run_rotation_tool.py`
- `run_flatten_V2.py`
- `run_merge_tool.py`
- `run_merge_ir_tool.py`
- `run_improved_masks_load.py`
- `run_cut_regions.py`

**Documentation:**
- `README.md` - This file
- `Execution_order.md` - Processing workflow

## License

[Add your license information here]

## Contact

[Add your contact information here]
