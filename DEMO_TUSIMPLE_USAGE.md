# TuSimple Dataset Demo Usage Guide

The `demo_tusimple.py` script is specifically designed to process the TuSimple dataset structure with nested date and sequence folders.

## TuSimple Dataset Structure

The script expects the following directory structure:
```
dataset_root/
├── 0530/
│   ├── 1492626047222176976_0/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ... (20 images total)
│   ├── 1492626126171818168_0/
│   │   ├── image1.jpg
│   │   └── ... (20 images)
│   └── ...
├── 0531/
│   ├── sequence_folder_1/
│   └── sequence_folder_2/
└── ...
```

## Usage Examples

### Basic usage - Process all sequences:
```bash
python demo_tusimple.py configs/tusimple_res18.py \
    --dataset-root /path/to/tusimple/dataset
```

### Process with custom model:
```bash
python demo_tusimple.py configs/tusimple_res18.py \
    --dataset-root /path/to/tusimple/dataset \
    --model ./checkpoints/custom_tusimple_model.pth
```

### Verify image order and show frame numbers:
```bash
python demo_tusimple.py configs/tusimple_res18.py \
    --dataset-root /path/to/tusimple/dataset \
    --verify-order \
    --show-frame-numbers \
    --max-sequences 2
```

### Create individual videos for each sequence:
```bash
python demo_tusimple.py configs/tusimple_res18.py \
    --dataset-root /path/to/tusimple/dataset \
    --output-dir ./tusimple_results \
    --fps 15 \
    --show-info
```

### Create single combined video:
```bash
python demo_tusimple.py configs/tusimple_res18.py \
    --dataset-root /path/to/tusimple/dataset \
    --single-video \
    --fps 10 \
    --show-info
```

### Process limited sequences with dots-only visualization:
```bash
python demo_tusimple.py configs/tusimple_res18.py \
    --dataset-root /path/to/tusimple/dataset \
    --max-sequences 5 \
    --dots-only \
    --output-dir ./test_results
```

## Command Line Arguments

### Required:
- `config`: Path to model configuration file (should be TuSimple config like `configs/tusimple_res18.py`)
- `--dataset-root`: Root path to TuSimple dataset containing date folders

### Optional:
- `--model`: Path to model checkpoint (overrides config test_model)
- `--output-dir`: Output directory for videos (default: `./tusimple_output`)
- `--fps`: Output video FPS (default: 10.0 for comfortable sequence viewing)
- `--dots-only`: Draw dots only without connecting lines
- `--max-sequences`: Maximum number of sequences to process (useful for testing)
- `--single-video`: Create one combined video instead of individual sequence videos
- `--show-info`: Show sequence and frame information on video frames
- `--show-frame-numbers`: Show frame numbers and image filenames on video frames
- `--verify-order`: Print detailed image order for verification (useful for debugging)

## Output

### Individual Videos Mode (Default)
- Creates one MP4 video file per sequence: `{date}_{sequence}.mp4`
- Each video shows the 20 images from that sequence with lane detection
- Example: `0530_1492626047222176976_0.mp4`

### Single Video Mode
- Creates one combined video: `tusimple_all_sequences.mp4`
- Contains all sequences concatenated together
- Useful for reviewing entire dataset processing

### Processing Summary
- Creates `processing_summary.json` with details about processed sequences
- Includes sequence counts, paths, and processing statistics

## Features

1. **Automatic Structure Detection**: Recursively finds all date/sequence folders
2. **Sequence Processing**: Maintains temporal order of images within sequences
3. **Flexible Output**: Choose between individual or combined videos
4. **Progress Tracking**: Shows real-time processing progress with tqdm
5. **Error Handling**: Skips corrupted images and continues processing
6. **Information Overlay**: Optional sequence and frame information on videos
7. **Natural Sorting**: Ensures correct chronological order of images and sequences
8. **Order Verification**: Option to verify and display image processing order
9. **Frame Numbering**: Optional frame numbers and filenames on video for debugging

## Technical Details

- **Image Dimensions**: Fixed to 1280x720 (TuSimple standard)
- **Video Codec**: MP4V for broad compatibility
- **Preprocessing**: Same pipeline as original demo (resize + crop from bottom)
- **Coordinate System**: Uses TuSimple row/column anchor system
- **Memory Efficient**: Processes sequences one at a time
- **CUDA Accelerated**: Uses GPU for model inference

## Example Output Structure

```
tusimple_output/
├── 0530_1492626047222176976_0.mp4
├── 0530_1492626126171818168_0.mp4
├── 0530_1492626127172745520_0.mp4
├── ...
└── processing_summary.json
```

Or with `--single-video`:
```
tusimple_output/
├── tusimple_all_sequences.mp4
└── processing_summary.json
```

## Performance Notes

- Processing speed depends on GPU and number of images
- Typical processing: 20-50 FPS on modern GPUs
- Each sequence (20 images) takes approximately 1-2 seconds to process
- Output video file sizes: ~1-5MB per sequence depending on FPS and content
