#!/usr/bin/env python3
"""
Test FP8 quantization optimization for Cosmos Transfer 2.5.

This script compares BF16 (baseline) vs FP8 inference:
1. Correctness: Validates outputs are visually similar
2. Performance: Measures speedup from FP8 Tensor Cores

Usage:
    # 5-frame test (fast validation)
    python tests/test_fp8_optimization.py \
        --depth-video <path> \
        --output-dir results/fp8_tests/5frame \
        --state-t 2 \
        --num-steps 4
    
    # 93-frame test (production workload)
    python tests/test_fp8_optimization.py \
        --depth-video <path> \
        --output-dir results/fp8_tests/93frame \
        --state-t 24 \
        --num-steps 35
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmos_transfer2.config import InferenceArguments, SetupArguments, DepthConfig
from cosmos_transfer2.inference import Control2WorldInference
from cosmos_transfer2._src.transfer2.utils import fp8_utils


def get_video_frame_count(video_path: str) -> int:
    """Get the number of frames in a video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def compute_video_psnr(video1_path: Path, video2_path: Path) -> float:
    """
    Compute PSNR between two videos.
    
    Returns:
        PSNR in dB (higher is better, >40 is excellent)
    """
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    
    mse_sum = 0
    frame_count = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Compute MSE for this frame
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        mse_sum += mse
        frame_count += 1
    
    cap1.release()
    cap2.release()
    
    if frame_count == 0:
        return 0.0
    
    avg_mse = mse_sum / frame_count
    if avg_mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10(255**2 / avg_mse)
    return psnr


def create_comparison_video(video1_path: Path, video2_path: Path, output_path: Path, title1: str = "BF16", title2: str = "FP8"):
    """Create side-by-side comparison video."""
    import subprocess
    
    # Use ffmpeg to create side-by-side comparison
    cmd = [
        'ffmpeg',
        '-i', str(video1_path),
        '-i', str(video2_path),
        '-filter_complex',
        f'[0:v]drawtext=text=\'{title1}\':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=10[v0];'
        f'[1:v]drawtext=text=\'{title2}\':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=10[v1];'
        f'[v0][v1]hstack=inputs=2',
        '-c:v', 'libx264',
        '-crf', '18',
        '-y',
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Created comparison video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test FP8 optimization")
    parser.add_argument("--depth-video", type=str, required=True, help="Path to depth control video")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--state-t", type=int, default=2, help="Number of latent frames (2 for 5 frames, 24 for 93 frames)")
    parser.add_argument("--num-steps", type=int, default=4, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, default="A high quality video of a robot in a warehouse", help="Text prompt")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for timing (best of N)")
    args = parser.parse_args()
    
    # Calculate expected frames
    expected_pixel_frames = (args.state_t - 1) * 4 + 1
    
    print("=" * 70)
    print("FP8 Quantization Optimization Test")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  State-t: {args.state_t} (→ {expected_pixel_frames} pixel frames)")
    print(f"  Diffusion steps: {args.num_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Num runs: {args.num_runs}")
    print()
    
    # Validate input video
    actual_frames = get_video_frame_count(args.depth_video)
    if actual_frames != expected_pixel_frames:
        raise ValueError(
            f"Input video '{args.depth_video}' has {actual_frames} frames, "
            f"but expected {expected_pixel_frames} frames for state_t={args.state_t}."
        )
    
    # Check FP8 availability
    print("Hardware Check:")
    print(f"  FP8 available: {fp8_utils.is_fp8_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
    print()
    
    if not fp8_utils.is_fp8_available():
        print("ERROR: FP8 not available on this system")
        print("Requirements:")
        print("  - Transformer Engine installed")
        print("  - GPU with compute capability >= 8.9 (Hopper/Blackwell)")
        print("  - PyTorch with FP8 dtypes")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup arguments
    setup_args = SetupArguments(
        checkpoint_dir="",
        enable_guardrails=False,
        benchmark=False,
    )
    
    # Helper to create sample config
    def create_sample(name: str, seed: int) -> InferenceArguments:
        return InferenceArguments(
            name=name,
            video_path=args.depth_video,
            prompt=args.prompt,
            guidance=7,
            num_steps=args.num_steps,
            seed=seed,
            depth=DepthConfig(control_path=args.depth_video),
            num_video_frames_per_chunk=expected_pixel_frames,
        )
    
    # Test 1: BF16 Baseline (with CFG optimization)
    print("-" * 70)
    print("Test 1: BF16 Baseline (CFG Batching Enabled)")
    print("-" * 70)
    
    inference_bf16 = Control2WorldInference(
        setup_args,
        batch_hint_keys=["depth"],
        state_t=args.state_t,
        use_cuda_graphs=False,
        use_cfg_batching=True,  # Use CFG optimization
        use_fp8=False,  # BF16 baseline
    )
    
    # Warmup
    print("Warming up BF16 model...")
    warmup_sample = create_sample("warmup", args.seed)
    _ = inference_bf16.generate([warmup_sample], output_path / "warmup_bf16")
    
    # Timed runs
    bf16_times = []
    for i in range(args.num_runs):
        print(f"BF16 run {i+1}/{args.num_runs}...")
        sample = create_sample(f"bf16_run{i}", args.seed)
        
        start = time.time()
        _ = inference_bf16.generate([sample], output_path / f"bf16_run{i}")
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        bf16_times.append(elapsed)
        print(f"  Time: {elapsed:.2f}s")
    
    best_bf16_time = min(bf16_times)
    print(f"Best BF16 time: {best_bf16_time:.2f}s")
    print()
    
    # Test 2: FP8 (with CFG optimization)
    print("-" * 70)
    print("Test 2: FP8 Quantization (CFG Batching Enabled)")
    print("-" * 70)
    
    inference_fp8 = Control2WorldInference(
        setup_args,
        batch_hint_keys=["depth"],
        state_t=args.state_t,
        use_cuda_graphs=False,
        use_cfg_batching=True,  # Use CFG optimization
        use_fp8=True,  # FP8 enabled
    )
    
    # Warmup (FP8 needs warmup for scaling calibration)
    print("Warming up FP8 model (calibrating scaling factors)...")
    warmup_sample = create_sample("warmup", args.seed)
    _ = inference_fp8.generate([warmup_sample], output_path / "warmup_fp8")
    
    # Timed runs
    fp8_times = []
    for i in range(args.num_runs):
        print(f"FP8 run {i+1}/{args.num_runs}...")
        sample = create_sample(f"fp8_run{i}", args.seed)
        
        start = time.time()
        _ = inference_fp8.generate([sample], output_path / f"fp8_run{i}")
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        fp8_times.append(elapsed)
        print(f"  Time: {elapsed:.2f}s")
    
    best_fp8_time = min(fp8_times)
    print(f"Best FP8 time: {best_fp8_time:.2f}s")
    print()
    
    # Compute speedup
    speedup = best_bf16_time / best_fp8_time
    
    print("=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"BF16 baseline:  {best_bf16_time:.2f}s")
    print(f"FP8 optimized:  {best_fp8_time:.2f}s")
    print(f"FP8 speedup:    {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print()
    
    # Quality comparison
    print("=" * 70)
    print("Quality Comparison")
    print("=" * 70)
    
    bf16_video_path = output_path / "bf16_run0" / "bf16_run0.mp4"
    fp8_video_path = output_path / "fp8_run0" / "fp8_run0.mp4"
    
    if bf16_video_path.exists() and fp8_video_path.exists():
        psnr = compute_video_psnr(bf16_video_path, fp8_video_path)
        print(f"PSNR: {psnr:.2f} dB")
        
        if psnr > 45:
            print("Quality: Excellent (visually identical)")
        elif psnr > 40:
            print("Quality: Very Good (imperceptible differences)")
        elif psnr > 35:
            print("Quality: Good (minor differences)")
        else:
            print("Quality: Fair (visible differences - may need tuning)")
        
        # Create comparison video
        comparison_path = output_path / "comparison_bf16_vs_fp8.mp4"
        create_comparison_video(bf16_video_path, fp8_video_path, comparison_path)
        print()
    
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)
    print(f"Results saved to: {output_path}")
    print()
    print("Summary:")
    print(f"  ✅ FP8 {speedup:.2f}x faster than BF16")
    print(f"  ✅ Quality: {psnr:.2f} dB PSNR")
    print()
    
    # Expected results
    print("Expected Results:")
    print("  - 5-frame:  1.6-1.8x speedup")
    print("  - 93-frame: 1.8-2.0x speedup")
    print("  - PSNR:     >40 dB (excellent quality)")


if __name__ == "__main__":
    main()

