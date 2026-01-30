#!/usr/bin/env python3
"""
Test batch inference with real model and data.

This test loads the actual Cosmos model and runs batch inference
on real video inputs.

Usage:
    cd /home/horde/cosmos/cosmos-transfer2.5
    source .venv/bin/activate
    python tests/test_batch_inference_real.py --input-dir /path/to/captures

Or with Isaac Lab captures:
    python tests/test_batch_inference_real.py \
        --video1 /path/to/capture1/depth_video.mp4 \
        --video2 /path/to/capture2/depth_video.mp4 \
        --output-dir ./batch_test_output
"""

import argparse
import sys
import time
from pathlib import Path

import torch


def test_batch_inference_real(
    video1_path: str,
    video2_path: str,
    output_dir: str,
    prompt: str = "A robotic arm manipulating objects on a table in an industrial setting.",
    num_steps: int = 4,
    guidance: int = 7,
):
    """Run batch inference with 2 real videos."""
    
    print("=" * 60)
    print("Cosmos Batch Inference - Real Model Test")
    print("=" * 60)
    
    # Check inputs exist
    video1 = Path(video1_path)
    video2 = Path(video2_path)
    
    if not video1.exists():
        print(f"Error: Video 1 not found: {video1}")
        return False
    if not video2.exists():
        print(f"Error: Video 2 not found: {video2}")
        return False
    
    print(f"Video 1: {video1}")
    print(f"Video 2: {video2}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {num_steps}, Guidance: {guidance}")
    print()
    
    # Import cosmos modules
    print("Loading Cosmos modules...")
    from cosmos_transfer2.config import InferenceArguments, SetupArguments, ModelKey, MODEL_CHECKPOINTS
    from cosmos_transfer2.inference import Control2WorldInference
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create setup args - model="depth" since we're using depth videos
    setup_args = SetupArguments(
        model="depth",  # Required: use depth model since inputs are depth videos
        output_dir=output_path,
        disable_guardrails=True,  # Skip for testing
    )
    
    # Create sample configs - depth control is implicit via model="depth"
    sample1 = InferenceArguments(
        name="batch_sample_1",
        video_path=video1,
        prompt=prompt,
        guidance=guidance,
        num_steps=num_steps,
        seed=42,
    )
    
    sample2 = InferenceArguments(
        name="batch_sample_2", 
        video_path=video2,
        prompt=prompt,
        guidance=guidance,
        num_steps=num_steps,
        seed=142,  # Different seed
    )
    
    # Initialize inference
    print("\nInitializing model (this may take a while)...")
    # For depth model, the hint key is just "depth"
    batch_hint_keys = ["depth"]
    
    start_init = time.time()
    inference = Control2WorldInference(setup_args, batch_hint_keys=batch_hint_keys)
    init_time = time.time() - start_init
    print(f"Model loaded in {init_time:.1f}s")
    
    # Run batch inference
    print("\n" + "=" * 60)
    print("Running BATCH inference (2 videos in parallel)...")
    print("=" * 60)
    
    torch.cuda.reset_peak_memory_stats()
    start_batch = time.time()
    
    batch_outputs = inference.generate_batch(
        samples=[sample1, sample2],
        output_dir=output_path,
        batch_size=2,
    )
    
    batch_time = time.time() - start_batch
    batch_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\nBatch inference complete!")
    print(f"  Time: {batch_time:.1f}s for 2 videos")
    print(f"  Per-video: {batch_time/2:.1f}s")
    print(f"  Peak GPU memory: {batch_memory:.1f} GB")
    print(f"  Outputs: {batch_outputs}")
    
    # Compare with sequential (optional)
    print("\n" + "=" * 60)
    print("Running SEQUENTIAL inference (for comparison)...")
    print("=" * 60)
    
    torch.cuda.reset_peak_memory_stats()
    start_seq = time.time()
    
    seq_outputs = inference.generate(
        samples=[sample1, sample2],
        output_dir=output_path / "sequential",
    )
    
    seq_time = time.time() - start_seq
    seq_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\nSequential inference complete!")
    print(f"  Time: {seq_time:.1f}s for 2 videos")
    print(f"  Per-video: {seq_time/2:.1f}s")
    print(f"  Peak GPU memory: {seq_memory:.1f} GB")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Batch mode:      {batch_time:.1f}s ({batch_time/2:.1f}s per video)")
    print(f"Sequential mode: {seq_time:.1f}s ({seq_time/2:.1f}s per video)")
    print(f"Speedup:         {seq_time/batch_time:.2f}x")
    print(f"Memory (batch):  {batch_memory:.1f} GB")
    print(f"Memory (seq):    {seq_memory:.1f} GB")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test batch inference with real data")
    parser.add_argument("--video1", required=True, help="Path to first video")
    parser.add_argument("--video2", required=True, help="Path to second video")
    parser.add_argument("--output-dir", default="./batch_test_output", help="Output directory")
    parser.add_argument("--prompt", default="A robotic arm manipulating objects on a table.", help="Prompt")
    parser.add_argument("--num-steps", type=int, default=4, help="Diffusion steps")
    parser.add_argument("--guidance", type=int, default=7, help="Guidance scale")
    
    args = parser.parse_args()
    
    success = test_batch_inference_real(
        video1_path=args.video1,
        video2_path=args.video2,
        output_dir=args.output_dir,
        prompt=args.prompt,
        num_steps=args.num_steps,
        guidance=args.guidance,
    )
    
    sys.exit(0 if success else 1)

