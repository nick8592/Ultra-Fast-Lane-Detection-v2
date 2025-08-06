# Modified demo_custom.py Usage Guide

The `demo_custom.py` has been enhanced to support both the original dataset mode and a new custom folder mode for processing images directly from a folder without requiring `.txt` split files.

## New Features

### Custom Folder Mode
Process images directly from any folder containing image files:

```bash
python demo_custom.py configs/tusimple_res18.py --folder ./my_images/
```

### Original Dataset Mode
If no `--folder` argument is provided, it works exactly like the original script with dataset splits:

```bash
python demo_custom.py configs/tusimple_res18.py
```

## Usage Examples

### Basic folder processing:
```bash
python demo_custom.py configs/tusimple_res18.py --folder ./road_images/
```

### Custom output filename and FPS:
```bash
python demo_custom.py configs/culane_res18.py \
    --folder ./test_images/ \
    --output my_detection_results.mp4 \
    --fps 25
```

### Process specific image types only:
```bash
python demo_custom.py configs/tusimple_res18.py \
    --folder ./mixed_folder/ \
    --extensions jpg png \
    --max-images 50
```

### Process with custom model checkpoint:
```bash
python demo_custom.py configs/tusimple_res18.py \
    --folder ./test_images/ \
    --model ./checkpoints/custom_model.pth \
    --output custom_results.mp4
```

### Process with TuSimple model:
```bash
python demo_custom.py configs/tusimple_res18.py \
    --folder ./highway_images/ \
    --output highway_lanes.mp4 \
    --fps 30
```

### Process with CULane model:
```bash
python demo_custom.py configs/culane_res34.py \
    --folder ./urban_images/ \
    --output urban_lanes.mp4
```

### Mix and match models and configs:
```bash
python demo_custom.py configs/culane_res18.py \
    --folder ./test_images/ \
    --model ./checkpoints/tusimple_res34.pth \
    --output mixed_model_test.mp4
```

## Command Line Arguments

### Required:
- `config`: Path to model configuration file (e.g., `configs/tusimple_res18.py`)

### Optional:
- `--folder`: Input folder containing images (enables custom folder mode)
- `--model`: Path to model checkpoint (overrides config test_model)
- `--output`: Output video file path (default: `{folder_name}_lane.mp4`)
- `--fps`: Output video FPS (default: 30.0)
- `--extensions`: Image file extensions to process (default: jpg jpeg png bmp tiff tif)
- `--max-images`: Maximum number of images to process

## Supported Image Formats
- JPG/JPEG
- PNG
- BMP
- TIFF/TIF
- Case-insensitive (both uppercase and lowercase extensions)

## Output
- Creates an MP4 video file with detected lanes (in folder mode) or AVI files (in dataset mode)
- Uses green dots (circles) to mark lane points
- Maintains original image dimensions based on dataset type:
  - TuSimple: 1280x720
  - CULane: 1640x590

## Key Differences from Original

1. **Flexible Input**: Can process any folder of images, not just predefined dataset splits
2. **Custom Model Loading**: Can override config model with `--model` argument
3. **Recursive Search**: Finds images in subfolders automatically
4. **Error Handling**: Skips corrupted images and continues processing
5. **Progress Tracking**: Shows processing progress with tqdm
6. **Automatic Resizing**: Handles different input image sizes
7. **Backward Compatible**: Original dataset mode still works when `--folder` is not specified
8. **Model Flexibility**: Mix and match any model checkpoint with any config file

## Technical Details

- Uses the same preprocessing pipeline as the original (resize + crop from bottom)
- Maintains identical coordinate prediction logic
- Automatically detects and handles CUDA availability
- Uses MJPG codec for output video compression
- Processes images in sorted order for consistent video output
- Supports model checkpoint override for testing different model-config combinations
- Provides clear error messages for missing files or folders
