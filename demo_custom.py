import torch, os, cv2
from utils.dist_utils import dist_print
import torch, os
from utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
import glob
import argparse
from pathlib import Path

def get_image_files(folder_path, extensions=None):
    """Get sorted list of image files from folder"""
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
        # Also check uppercase extensions
        pattern = os.path.join(folder_path, '**', ext.upper())
        image_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    return image_files

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

def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords
def parse_args():
    parser = argparse.ArgumentParser(description='Ultra-Fast Lane Detection v2 - Custom Demo')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--folder', type=str, help='Input folder containing images (if not provided, uses original dataset mode)')
    parser.add_argument('--model', type=str, help='Path to model checkpoint (overrides config test_model)')
    parser.add_argument('--output', type=str, help='Output video file path (default: folder_name_demo.avi)')
    parser.add_argument('--fps', type=float, default=30.0, help='Output video FPS (default: 30.0)')
    parser.add_argument('--extensions', nargs='+', default=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'], 
                       help='Image file extensions to process (default: jpg jpeg png bmp tiff tif)')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process')
    return parser.parse_args()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # Parse arguments
    custom_args = parse_args()
    
    # Setup sys.argv for merge_config
    import sys
    sys.argv = ['demo_custom.py', custom_args.config]
    if custom_args.model:
        sys.argv.extend(['--test_model', custom_args.model])
    
    args, cfg = merge_config()
    cfg.batch_size = 1
    print('setting batch_size to 1 for demo generation')

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        img_w, img_h = 1280, 720
    else:
        raise NotImplementedError

    net = get_model(cfg)

    # Load model weights - use custom model if provided, otherwise use config
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

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Check if folder mode is requested
    if custom_args.folder:
        # Custom folder mode
        print(f"Processing images from folder: {custom_args.folder}")
        
        if not os.path.exists(custom_args.folder):
            print(f"Error: Folder '{custom_args.folder}' not found!")
            exit(1)
        
        # Get image files
        extensions_with_wildcards = [f'*.{ext}' for ext in custom_args.extensions]
        image_files = get_image_files(custom_args.folder, extensions_with_wildcards)
        
        if not image_files:
            print(f"Error: No image files found in '{custom_args.folder}' with extensions: {custom_args.extensions}")
            exit(1)
        
        print(f"Found {len(image_files)} images")
        
        # Limit number of images if specified
        if custom_args.max_images:
            image_files = image_files[:custom_args.max_images]
            print(f"Processing first {len(image_files)} images")
        
        # Setup output video
        if custom_args.output:
            output_path = custom_args.output
        else:
            folder_name = Path(custom_args.folder).name
            output_path = f"{folder_name}_lane.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vout = cv2.VideoWriter(output_path, fourcc, custom_args.fps, (img_w, img_h))
        
        print(f"Output video: {output_path}")
        print("Processing images...")
        
        for i, image_path in enumerate(tqdm.tqdm(image_files)):
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
                
                # Draw lanes (dots only like original)
                vis = original_img.copy()
                for lane in coords:
                    for coord in lane:
                        cv2.circle(vis, coord, 5, (0, 255, 0), -1)
                
                # Resize if needed to match output dimensions
                if vis.shape[1] != img_w or vis.shape[0] != img_h:
                    vis = cv2.resize(vis, (img_w, img_h))
                
                vout.write(vis)
                
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                continue
        
        vout.release()
        print(f"Video saved to: {output_path}")
        
    else:
        # Original dataset mode
        if cfg.dataset == 'CULane':
            splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
            datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
        elif cfg.dataset == 'Tusimple':
            splits = ['test.txt']
            datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
        else:
            raise NotImplementedError
            
        for split, dataset in zip(splits, datasets):
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            print(split[:-3]+'avi')
            vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
            for i, data in enumerate(tqdm.tqdm(loader)):
                imgs, names = data
                imgs = imgs.cuda()
                with torch.no_grad():
                    pred = net(imgs)

                vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
                coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h)
                for lane in coords:
                    for coord in lane:
                        cv2.circle(vis,coord,5,(0,255,0),-1)
                vout.write(vis)
            
            vout.release()
