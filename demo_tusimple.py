#!/usr/bin/env python3
"""
Ultra-Fast Lane Detection v2 - TuSimple Dataset Demo
Process TuSimple dataset structure with nested folders containing image sequences
"""

import torch
import os
import cv2
from utils.dist_utils import dist_print
import torch
from utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
import glob
import argparse
from pathlib import Path
import json
import re

def natural_sort_key(text):
    """Generate a key for natural sorting that handles numbers correctly"""
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_tusimple_sequences(dataset_root):
    """Get all TuSimple sequence folders"""
    sequence_folders = []
    
    # Check if dataset_root is directly a date folder (contains sequence folders)
    # or a parent folder containing date folders
    root_contents = sorted(os.listdir(dataset_root), key=natural_sort_key)
    
    # Check if any item in root is a directory with numeric sequence-like names
    has_sequence_folders = False
    for item in root_contents:
        item_path = os.path.join(dataset_root, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) > 10:  # TuSimple sequences are long numbers
            has_sequence_folders = True
            break
    
    if has_sequence_folders:
        # dataset_root is directly a date folder, process its sequence folders
        date_name = os.path.basename(dataset_root)
        print(f"Processing date folder directly: {date_name}")
        
        for seq_folder in root_contents:
            seq_path = os.path.join(dataset_root, seq_folder)
            if os.path.isdir(seq_path):
                # Check if folder contains images
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(glob.glob(os.path.join(seq_path, ext)))
                    image_files.extend(glob.glob(os.path.join(seq_path, ext.upper())))
                
                if image_files:
                    # Sort images naturally to ensure correct order
                    sorted_images = sorted(image_files, key=lambda x: natural_sort_key(os.path.basename(x)))
                    
                    sequence_folders.append({
                        'path': seq_path,
                        'date': date_name,
                        'sequence': seq_folder,
                        'images': sorted_images,
                        'count': len(sorted_images)
                    })
    else:
        # dataset_root contains date folders, process each date folder
        print("Processing parent folder containing date folders")
        
        for date_folder in root_contents:
            date_path = os.path.join(dataset_root, date_folder)
            if os.path.isdir(date_path):
                # Look for sequence folders within date folder - sort naturally
                seq_folders = sorted(os.listdir(date_path), key=natural_sort_key)
                
                for seq_folder in seq_folders:
                    seq_path = os.path.join(date_path, seq_folder)
                    if os.path.isdir(seq_path):
                        # Check if folder contains images
                        image_files = []
                        for ext in ['*.jpg', '*.jpeg', '*.png']:
                            image_files.extend(glob.glob(os.path.join(seq_path, ext)))
                            image_files.extend(glob.glob(os.path.join(seq_path, ext.upper())))
                        
                        if image_files:
                            # Sort images naturally to ensure correct order
                            sorted_images = sorted(image_files, key=lambda x: natural_sort_key(os.path.basename(x)))
                            
                            sequence_folders.append({
                                'path': seq_path,
                                'date': date_folder,
                                'sequence': seq_folder,
                                'images': sorted_images,
                                'count': len(sorted_images)
                            })
    
    return sequence_folders

def preprocess_image(image_path, img_transforms, crop_size):
    """Preprocess a single image for model input"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    from PIL import Image
    pil_image = Image.fromarray(image_rgb)
    # Apply transforms (resize + normalize)
    processed = img_transforms(pil_image)
    
    # Crop from bottom like LaneTestDataset does
    processed = processed[:, -crop_size:, :]
    
    # Add batch dimension
    return processed.unsqueeze(0), image

def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590):
    """Convert model predictions to lane coordinates"""
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width), 
                                                    min(num_grid_row-1, max_indices_row[0, k, i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width), 
                                                    min(num_grid_col-1, max_indices_col[0, k, i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords

def draw_lanes(image, coords, dots_only=False, line_thickness=3, point_radius=4):
    """Draw detected lanes on the image"""
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for lane_idx, lane in enumerate(coords):
        color = colors[lane_idx % len(colors)]
        
        # Draw points
        for coord in lane:
            cv2.circle(image, coord, point_radius, color, -1)
        
        # Draw lines connecting points (unless dots_only is True)
        if not dots_only and len(lane) > 1:
            import numpy as np
            points = np.array(lane, dtype=np.int32)
            cv2.polylines(image, [points], False, color, line_thickness)
    
    return image

def parse_args():
    parser = argparse.ArgumentParser(description='Ultra-Fast Lane Detection v2 - TuSimple Dataset Demo')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--dataset-root', type=str, required=True, help='Root path to TuSimple data (can be a single date folder like /path/to/0601 or parent folder containing date folders)')
    parser.add_argument('--model', type=str, help='Path to model checkpoint (overrides config test_model)')
    parser.add_argument('--output-dir', type=str, default='./tusimple_output', help='Output directory for videos (default: ./tusimple_output)')
    parser.add_argument('--fps', type=float, default=30.0, help='Output video FPS (default: 30.0 for sequence viewing)')
    parser.add_argument('--dots-only', action='store_true', help='Draw dots only (no connecting lines)')
    parser.add_argument('--max-sequences', type=int, help='Maximum number of sequences to process')
    parser.add_argument('--single-video', action='store_true', help='Create single video for all sequences instead of individual videos')
    parser.add_argument('--show-info', action='store_true', help='Show sequence info on video frames')
    parser.add_argument('--show-frame-numbers', action='store_true', help='Show frame numbers on video frames')
    parser.add_argument('--verify-order', action='store_true', help='Print detailed image order for verification')
    return parser.parse_args()

def main():
    custom_args = parse_args()
    
    # Setup sys.argv for merge_config
    import sys
    sys.argv = ['demo_tusimple.py', custom_args.config]
    if custom_args.model:
        sys.argv.extend(['--test_model', custom_args.model])
    
    args, cfg = merge_config()
    cfg.batch_size = 1
    print('Setting batch_size to 1 for demo generation')

    dist_print('Start TuSimple dataset processing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    # TuSimple specific settings
    img_w, img_h = 1280, 720
    print(f"Using TuSimple image dimensions: {img_w}x{img_h}")

    # Load model
    net = get_model(cfg)
    
    # Load model weights
    model_path = custom_args.model if custom_args.model else cfg.test_model
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from: {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
        net.load_state_dict(compatible_state_dict, strict=False)
    else:
        print("Warning: No model weights loaded! Using random weights.")
        if model_path:
            print(f"Model file not found: {model_path}")
    
    net.eval()
    net = net.cuda()

    # Setup image transforms
    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Get all TuSimple sequences
    print(f"Scanning TuSimple dataset in: {custom_args.dataset_root}")
    sequences = get_tusimple_sequences(custom_args.dataset_root)
    
    if not sequences:
        print(f"Error: No image sequences found in '{custom_args.dataset_root}'")
        print("Expected structure:")
        print("  Option 1: dataset_root/sequence_folder/images (single date folder)")
        print("  Option 2: dataset_root/date_folder/sequence_folder/images (parent folder)")
        return
    
    print(f"Found {len(sequences)} sequences:")
    for seq in sequences[:10]:  # Show first 10
        print(f"  {seq['date']}/{seq['sequence']}: {seq['count']} images")
        # Show first few image names to verify order
        if seq['count'] > 0:
            sample_images = [os.path.basename(img) for img in seq['images'][:3]]
            if seq['count'] > 3:
                sample_images.append('...')
                sample_images.extend([os.path.basename(img) for img in seq['images'][-2:]])
            print(f"    Order: {' -> '.join(sample_images)}")
    if len(sequences) > 10:
        print(f"  ... and {len(sequences) - 10} more sequences")
    
    # Limit sequences if specified
    if custom_args.max_sequences:
        sequences = sequences[:custom_args.max_sequences]
        print(f"Processing first {len(sequences)} sequences")
    
    # Verify image order if requested
    if custom_args.verify_order:
        print("\nDetailed image order verification:")
        for seq in sequences[:3]:  # Show first 3 sequences in detail
            print(f"\n{seq['date']}/{seq['sequence']} ({seq['count']} images):")
            for i, img_path in enumerate(seq['images']):
                print(f"  {i+1:2d}: {os.path.basename(img_path)}")
            if len(sequences) > 3:
                print("  ... (use --verify-order to see all)")
                break
    
    # Create output directory
    output_dir = Path(custom_args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup video writer(s)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if custom_args.single_video:
        # Single video for all sequences
        output_path = output_dir / "tusimple_all_sequences.mp4"
        vout = cv2.VideoWriter(str(output_path), fourcc, custom_args.fps, (img_w, img_h))
        print(f"Creating single video: {output_path}")
    
    # Process sequences
    total_images = sum(seq['count'] for seq in sequences)
    processed_images = 0
    
    print(f"Processing {total_images} images from {len(sequences)} sequences...")
    
    # Create summary info
    summary = {
        'total_sequences': len(sequences),
        'total_images': total_images,
        'sequences': []
    }
    
    with tqdm.tqdm(total=total_images, desc="Processing images") as pbar:
        for seq_idx, sequence in enumerate(sequences):
            seq_info = {
                'date': sequence['date'],
                'sequence': sequence['sequence'],
                'image_count': sequence['count'],
                'path': sequence['path']
            }
            
            if not custom_args.single_video:
                # Individual video for each sequence
                output_path = output_dir / f"{sequence['date']}_{sequence['sequence']}.mp4"
                vout = cv2.VideoWriter(str(output_path), fourcc, custom_args.fps, (img_w, img_h))
            
            for img_idx, image_path in enumerate(sequence['images']):
                try:
                    # Preprocess image
                    processed_img, original_img = preprocess_image(image_path, img_transforms, cfg.train_height)
                    processed_img = processed_img.cuda()
                    
                    # Inference
                    with torch.no_grad():
                        pred = net(processed_img)
                    
                    # Get coordinates
                    coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, 
                                       original_image_width=img_w, original_image_height=img_h)
                    
                    # Draw lanes
                    vis = original_img.copy()
                    vis = draw_lanes(vis, coords, custom_args.dots_only)
                    
                    # Add sequence info if requested
                    if custom_args.show_info:
                        info_text = f"{sequence['date']}/{sequence['sequence']} - Frame {img_idx+1}/{sequence['count']}"
                        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    
                    # Add frame numbers if requested
                    if custom_args.show_frame_numbers:
                        frame_text = f"Frame: {img_idx+1}/{sequence['count']}"
                        image_name = os.path.basename(image_path)
                        name_text = f"Image: {image_name}"
                        
                        # Frame number (top right)
                        cv2.putText(vis, frame_text, (vis.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(vis, frame_text, (vis.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        
                        # Image name (bottom)
                        cv2.putText(vis, name_text, (10, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(vis, name_text, (10, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Resize if needed
                    if vis.shape[1] != img_w or vis.shape[0] != img_h:
                        vis = cv2.resize(vis, (img_w, img_h))
                    
                    vout.write(vis)
                    processed_images += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError processing {image_path}: {e}")
                    pbar.update(1)
                    continue
            
            if not custom_args.single_video:
                vout.release()
                print(f"Sequence video saved: {output_path}")
            
            summary['sequences'].append(seq_info)
    
    if custom_args.single_video:
        vout.release()
        print(f"Combined video saved: {output_path}")
    
    # Save processing summary
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {processed_images}/{total_images} images")
    print(f"From {len(sequences)} sequences")
    print(f"Output directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
