## Execution Order for Mask Generation Pipeline

1. **run_rotation_tool.py** - Change axes of image so its centers align with barrel cut centers (needed to make next transformation more convenient).

2. **run_flatten_V2.py** - Used to flatten cylindrical images.

3. **run_merge_tool.py** - Used to merge flattened visible images.

4. **plot_image_mask.py** - It hasn't been saved but can be easily recreated. For each overlayed VIS image masks a bit corrupted, we reconstruct them with scikit-image closing, erosion, etc. Save folder was Barrel_Images2_croped_Merged_V2_Mask.

5. **run_merge_ir_tool.py** - Merge IR and visible images.

6. **run_improved_masks_load.py** - Original masks were a bit corrupted, we recreated them using plot_image_mask.py. In this file we cut masks into corresponding patches (patch_size=thermogram size) and load them into data storage folders.

7. **run_cut_regions.py** - Here we cut aligned patches (from run_merge_ir_tool.py) for each location. The point is that during alignment some edges could not be aligned perfectly, and there is usually not much heating there - so we remove them. We apply these cuts to masks, postprocessed data, etc. and save them in a new folder cutted_V1 inside each sample/location folder.

## Next Steps

After completing the above steps, proceed to the Preliminary_check folder (set in your .env as PROJECT_BASE) for:
- **organize_ml_data.py** - Rewrite all data in a convenient way together in separate folder
- **rename_with_number.py** - Change name format

**Note:** Ensure PROJECT_BASE is set in your .env file to the parent project directory.