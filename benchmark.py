#!/usr/bin/env python3
"""
Ultra-Fast Lane Detection v2 - Model Benchmarking Tool
Calculates FLOPs, parameters, and inference speed (ms/FPS)
"""

import torch
import time
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Ultra-Fast Lane Detection v2 model')
    parser.add_argument('--config', help='path to config file')
    parser.add_argument('--model', type=str, help='path to model checkpoint', default=None)
    parser.add_argument('--warmup', type=int, default=10, help='number of warmup runs')
    parser.add_argument('--runs', type=int, default=100, help='number of test runs')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for testing')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
    parser.add_argument('--force-input-shape', type=str, help='Force specific input shape as HxW, e.g., 320x800')
    return parser.parse_args()

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_flops(model, input_shape):
    """Calculate FLOPs using different methods"""
    results = {}
    
    # Method 1: Try thop
    try:
        from thop import profile, clever_format
        dummy_input = torch.randn(input_shape)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        results['thop_flops'] = flops
        results['thop_params'] = params
        flops_str, params_str = clever_format([flops, params], "%.3f")
        results['thop_flops_str'] = flops_str
        results['thop_params_str'] = params_str
    except ImportError:
        print("thop not available. Install with: pip install thop")
        results['thop_flops'] = None
        
    # Method 2: Try fvcore
    try:
        from fvcore.nn import FlopCountMode, flop_count
        dummy_input = torch.randn(input_shape)
        flop_dict, _ = flop_count(model, (dummy_input,), supported_ops=None)
        total_flops = sum(flop_dict.values())
        results['fvcore_flops'] = total_flops
        results['fvcore_flops_str'] = f"{total_flops/1e9:.3f}G"
    except ImportError:
        print("fvcore not available. Install with: pip install fvcore")
        results['fvcore_flops'] = None
        
    return results

def benchmark_speed(model, input_tensor, warmup_runs=10, test_runs=100, device='cuda'):
    """Benchmark inference speed"""
    model.eval()
    
    print(f"Warming up with {warmup_runs} runs...")
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print(f"Benchmarking with {test_runs} runs...")
    # Actual timing
    times = []
    with torch.no_grad():
        for i in range(test_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model(input_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{test_runs} runs")
    
    times = np.array(times)
    results = {
        'times_ms': times,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'fps_mean': 1000.0 / np.mean(times),
        'fps_max': 1000.0 / np.min(times)
    }
    
    return results

def main():
    args = parse_args()
    
    # Import after parsing args to handle missing dependencies gracefully
    from utils.common import merge_config, get_model
    
    # Load config
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import sys
    sys.argv = ['benchmark.py', args.config]
    if args.model:
        sys.argv.extend(['--test_model', args.model])
    
    _, cfg = merge_config()
    
    # Set device
    if args.cpu or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'
    
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = get_model(cfg)
    
    # Load weights if specified
    if args.model or cfg.test_model:
        model_path = args.model or cfg.test_model
        print(f"Loading weights from: {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
        model.load_state_dict(compatible_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # Determine input size
    if args.force_input_shape:
        # Parse forced input shape
        try:
            h, w = args.force_input_shape.split('x')
            input_height = int(h)
            input_width = int(w)
            print(f"Forcing input shape to: {input_height}x{input_width}")
        except:
            print(f"Invalid format for --force-input-shape: {args.force_input_shape}")
            print("Expected format: HxW (e.g., 320x800)")
            return
    elif hasattr(cfg, 'train_height') and hasattr(cfg, 'train_width'):
        # Use the original training dimensions, not the cropped ones
        input_height = cfg.train_height
        input_width = cfg.train_width
    else:
        # Default values
        input_height = 320
        input_width = 800
    
    input_shape = (args.batch_size, 3, input_height, input_width)
    print(f"Input shape: {input_shape}")
    
    # Check if forced input shape matches model expectations
    if args.force_input_shape and hasattr(cfg, 'train_height') and hasattr(cfg, 'train_width'):
        expected_height = cfg.train_height
        expected_width = cfg.train_width
        if input_height != expected_height or input_width != expected_width:
            print(f"WARNING: Forced input shape ({input_height}x{input_width}) differs from model's expected shape ({expected_height}x{expected_width})")
            print("This may cause errors or incorrect results!")
            
            # Calculate expected vs actual input_dim
            expected_dim = expected_height // 32 * expected_width // 32 * 8
            actual_dim = input_height // 32 * input_width // 32 * 8
            print(f"Expected input_dim: {expected_dim}, Actual: {actual_dim}")
            
            if expected_dim != actual_dim:
                print("ERROR: input_dim mismatch detected. Model will likely fail!")
                print("Consider using the model's native input shape or retraining with the desired shape.")
                return
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    # Calculate FLOPs
    print("\nCalculating FLOPs...")
    flop_results = calculate_flops(model.cpu(), input_shape)
    model = model.to(device)  # Move back to device
    
    # Benchmark speed
    print(f"\nBenchmarking inference speed on {device}...")
    input_tensor = torch.randn(input_shape).to(device)
    speed_results = benchmark_speed(model, input_tensor, args.warmup, args.runs, device)
    
    # Print results
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"Model: {cfg.dataset} with backbone {cfg.backbone}")
    print(f"Input shape: {input_shape}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    
    print(f"\nModel Complexity:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    if flop_results.get('thop_flops'):
        print(f"  FLOPs (thop): {flop_results['thop_flops_str']}")
    if flop_results.get('fvcore_flops'):
        print(f"  FLOPs (fvcore): {flop_results['fvcore_flops_str']}")
    
    print(f"\nInference Speed ({args.runs} runs):")
    print(f"  Mean: {speed_results['mean_ms']:.2f} ± {speed_results['std_ms']:.2f} ms")
    print(f"  Median: {speed_results['median_ms']:.2f} ms")
    print(f"  Min: {speed_results['min_ms']:.2f} ms")
    print(f"  Max: {speed_results['max_ms']:.2f} ms")
    print(f"  Mean FPS: {speed_results['fps_mean']:.2f}")
    print(f"  Max FPS: {speed_results['fps_max']:.2f}")
    
    if device == 'cuda':
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Save results
    output_file = f"benchmark_{cfg.dataset}_{cfg.backbone}_{device}.txt"
    with open(output_file, 'w') as f:
        f.write("ULTRA-FAST LANE DETECTION V2 - BENCHMARK RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Model: {cfg.dataset} with backbone {cfg.backbone}\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Test runs: {args.runs}\n")
        f.write(f"Warmup runs: {args.warmup}\n")
        f.write(f"\nModel Complexity:\n")
        f.write(f"  Total parameters: {total_params:,}\n")
        f.write(f"  Trainable parameters: {trainable_params:,}\n")
        
        if flop_results.get('thop_flops'):
            f.write(f"  FLOPs (thop): {flop_results['thop_flops_str']}\n")
        if flop_results.get('fvcore_flops'):
            f.write(f"  FLOPs (fvcore): {flop_results['fvcore_flops_str']}\n")
        
        f.write(f"\nInference Speed ({args.runs} runs):\n")
        f.write(f"  Mean: {speed_results['mean_ms']:.2f} ± {speed_results['std_ms']:.2f} ms\n")
        f.write(f"  Median: {speed_results['median_ms']:.2f} ms\n")
        f.write(f"  Min: {speed_results['min_ms']:.2f} ms\n")
        f.write(f"  Max: {speed_results['max_ms']:.2f} ms\n")
        f.write(f"  Mean FPS: {speed_results['fps_mean']:.2f}\n")
        f.write(f"  Max FPS: {speed_results['fps_max']:.2f}\n")
        
        if device == 'cuda':
            f.write(f"\nGPU Memory:\n")
            f.write(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB\n")
            f.write(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB\n")
    
    print(f"\nResults saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
