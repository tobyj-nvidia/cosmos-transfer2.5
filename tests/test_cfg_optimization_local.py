#!/usr/bin/env python3
"""
Test CFG optimization locally with real model and verify correctness + performance.

This test:
1. Loads the actual Cosmos Transfer2 model  
2. Runs inference with BOTH sequential CFG (original) and batched CFG (optimized)
3. Verifies outputs are identical (using same seed) for correctness
4. Measures and reports performance improvement

Test Strategy:
- Run 0 for both sequential and batched: Uses the SAME seed (--seed arg) for pixel-perfect comparison
- Runs 1+ for both: Use different seeds to measure timing robustness across noise samples

Usage:
    cd /home/tobyj/code/notes/daily-notes/src/reference/dex/experiments/octi/cosmos-transfer2.5
    source .venv/bin/activate
    
    # Test 1: With 5-frame input (state_t=2, fast):
    python tests/test_cfg_optimization_local.py \
        --depth-video /home/tobyj/code/isaaclab-cosmos-experiments/results/test_inputs_5frame/depth_5frame_000.mp4 \
        --output-dir ./cfg_opt_test_5frame \
        --state-t 2 \
        --seed 42 \
        --num-steps 4
    
    # Test 2: With 93-frame input (state_t=24, default Cosmos Transfer size):
    python tests/test_cfg_optimization_local.py \
        --depth-video /home/tobyj/code/isaaclab-cosmos-experiments/results/batch_test_fix_verification/batch_sample_1_control_depth.mp4 \
        --output-dir ./cfg_opt_test_93frame \
        --state-t 24 \
        --seed 42 \
        --num-steps 35

Note: --num-steps controls diffusion sampling steps (4 is fast, 35 is high quality)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch


def get_video_frame_count(video_path: str) -> int:
    """Get the total number of frames in a video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Test CFG optimization with real model")
    parser.add_argument("--depth-video", type=str, required=True,
                        help="Path to depth video input")
    parser.add_argument("--output-dir", type=str, default="./cfg_opt_test",
                        help="Output directory for results")
    parser.add_argument("--prompt", type=str, 
                        default="A high quality video of a robot in a warehouse",
                        help="Text prompt")
    parser.add_argument("--num-steps", type=int, default=4,
                        help="Number of diffusion sampling steps (4=fast, 35=high quality)")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of runs for timing (best of N)")
    parser.add_argument("--state-t", type=int, default=2,
                        help="Number of latent temporal frames (2→5 frames, 24→93 frames)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--profile", action="store_true",
                        help="Enable NVTX markers for profiling (reduces num_runs to 1)")
    
    args = parser.parse_args()
    
    # For profiling, only do 1 run of each to keep the profile clean
    if args.profile:
        args.num_runs = 1
    
    print("="*70)
    print("CFG Optimization Test with Real Model")
    print("="*70)
    print(f"Depth video: {args.depth_video}")
    print(f"Output dir: {args.output_dir}")
    print(f"Prompt: {args.prompt}")
    print(f"Diffusion steps: {args.num_steps} (configurable via --num-steps)")
    print(f"State_t: {args.state_t} → {(args.state_t - 1) * 4 + 1} pixel frames expected")
    print(f"Seed: {args.seed}")
    print()
    
    # Validate input video has correct number of frames
    print("Validating input video...")
    expected_pixel_frames = (args.state_t - 1) * 4 + 1
    actual_frames = get_video_frame_count(args.depth_video)
    print(f"  Input video has {actual_frames} frames")
    print(f"  Expected {expected_pixel_frames} frames for state_t={args.state_t}")
    
    if actual_frames != expected_pixel_frames:
        print()
        print(f"ERROR: Frame count mismatch!")
        print(f"  Input video: {actual_frames} frames")
        print(f"  Expected:    {expected_pixel_frames} frames")
        print()
        print("Suggestions:")
        if actual_frames == 93:
            print(f"  - For 93-frame videos, use --state-t 24 (default Cosmos Transfer size)")
        elif actual_frames == 5:
            print(f"  - For 5-frame videos, use --state-t 2")
        else:
            # Calculate correct state_t
            correct_state_t = (actual_frames - 1) // 4 + 1
            print(f"  - For {actual_frames}-frame videos, try --state-t {correct_state_t}")
        print(f"  - Or create a {expected_pixel_frames}-frame video using scripts/extract_rolling_frames.py")
        sys.exit(1)
    print("  ✓ Frame count matches expected")
    print()
    
    # Import cosmos modules
    print("Loading Cosmos modules...")
    from cosmos_transfer2.config import InferenceArguments, SetupArguments, DepthConfig
    from cosmos_transfer2.inference import Control2WorldInference
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create setup args
    setup_args = SetupArguments(
        model="depth",
        output_dir=output_path,
        disable_guardrails=True,
    )
    
    # Expected pixel frames: (state_t - 1) * temporal_compression_factor + 1
    # For Cosmos Transfer 2.5, temporal_compression_factor = 4
    # Example: state_t=2 → (2-1)*4+1 = 5 frames
    expected_pixel_frames = (args.state_t - 1) * 4 + 1
    
    # Helper to create sample config with specific seed
    def create_sample(name: str, seed: int) -> InferenceArguments:
        return InferenceArguments(
            name=name,
            video_path=args.depth_video,
            prompt=args.prompt,
            num_video_frames_per_chunk=expected_pixel_frames,
            num_steps=args.num_steps,
            seed=seed,
            depth=DepthConfig(control_path=args.depth_video),
        )
    
    print(f"\nInitializing model (state_t={args.state_t})...")
    start_init = time.time()
    inference = Control2WorldInference(
        setup_args,
        batch_hint_keys=["depth"],
        state_t=args.state_t,
        use_cuda_graphs=False,  # Disable for fair comparison
        use_cfg_batching=False,  # Will be toggled between tests
    )
    init_time = time.time() - start_init
    print(f"Model initialized in {init_time:.2f}s")
    print()
    
    # Warmup run
    if args.profile:
        torch.cuda.nvtx.range_push("WARMUP")
    
    print("Running warmup...")
    warmup_sample = create_sample("warmup", seed=999)
    _ = inference.generate(
        samples=[warmup_sample],
        output_dir=output_path / "warmup"
    )
    torch.cuda.synchronize()
    print("Warmup complete")
    print()
    
    if args.profile:
        torch.cuda.nvtx.range_pop()
    
    # =================================================================
    # Monkey-patch to capture diffusion-only timing
    # =================================================================
    from cosmos_transfer2._src.predict2.models.text2world_model_rectified_flow import Text2WorldModelRectifiedFlow
    original_generate = Text2WorldModelRectifiedFlow.generate_samples_from_batch
    
    diffusion_times = {"sequential": [], "batched": []}
    current_mode = [None]  # Use list to allow modification in nested function
    
    def timed_generate(self, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_generate(self, *args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if current_mode[0]:
            diffusion_times[current_mode[0]].append(elapsed)
        return result
    
    Text2WorldModelRectifiedFlow.generate_samples_from_batch = timed_generate
    
    # =================================================================
    # Test 1: Original sequential CFG (use_cfg_batching=False)
    # =================================================================
    if args.profile:
        torch.cuda.nvtx.range_push("SEQUENTIAL_CFG_TEST")
    
    print("="*70)
    print("Test 1: Original Sequential CFG")
    print("="*70)
    
    # Disable CFG batching optimization
    inference.use_cfg_batching = False
    inference.inference_pipeline.use_cfg_batching = False
    current_mode[0] = "sequential"
    
    sequential_times = []
    for run in range(args.num_runs):
        # Use deterministic seed for run 0 (for correctness comparison)
        # Use different seeds for subsequent runs (for timing robustness)
        run_seed = args.seed if run == 0 else args.seed + run
        sample_seq = create_sample(f"sequential_run{run}", seed=run_seed)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output_seq = inference.generate(
            samples=[sample_seq],
            output_dir=output_path / f"sequential_run{run}",
        )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        sequential_times.append(elapsed)
        diffusion_time = diffusion_times["sequential"][-1]
        print(f"  Run {run+1}/{args.num_runs} (seed={run_seed}): {elapsed:.3f}s total, {diffusion_time:.3f}s diffusion")
    
    best_sequential = min(sequential_times)
    avg_sequential = sum(sequential_times) / len(sequential_times)
    print(f"\nSequential CFG:")
    print(f"  Best: {best_sequential:.3f}s")
    print(f"  Average: {avg_sequential:.3f}s")
    print()
    
    if args.profile:
        torch.cuda.nvtx.range_pop()
    
    # =================================================================
    # Test 2: Optimized batched CFG (use_cfg_batching=True)
    # =================================================================
    if args.profile:
        torch.cuda.nvtx.range_push("BATCHED_CFG_TEST")
    
    print("="*70)
    print("Test 2: Optimized Batched CFG")
    print("="*70)
    
    # Enable CFG batching optimization
    inference.use_cfg_batching = True
    inference.inference_pipeline.use_cfg_batching = True
    current_mode[0] = "batched"
    
    batched_times = []
    for run in range(args.num_runs):
        # Use SAME seed as sequential for run 0 (for correctness comparison)
        # Use different seeds for subsequent runs (for timing robustness)
        run_seed = args.seed if run == 0 else args.seed + run
        sample_batch = create_sample(f"batched_run{run}", seed=run_seed)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output_batch = inference.generate(
            samples=[sample_batch],
            output_dir=output_path / f"batched_run{run}",
        )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        batched_times.append(elapsed)
        diffusion_time = diffusion_times["batched"][-1]
        print(f"  Run {run+1}/{args.num_runs} (seed={run_seed}): {elapsed:.3f}s total, {diffusion_time:.3f}s diffusion")
    
    best_batched = min(batched_times)
    avg_batched = sum(batched_times) / len(batched_times)
    print(f"\nBatched CFG:")
    print(f"  Best: {best_batched:.3f}s")
    print(f"  Average: {avg_batched:.3f}s")
    print()
    
    if args.profile:
        torch.cuda.nvtx.range_pop()
    
    # =================================================================
    # Compare Results
    # =================================================================
    print("="*70)
    print("Performance Comparison")
    print("="*70)
    
    # Total end-to-end timing
    speedup_best = (best_sequential / best_batched - 1) * 100
    speedup_avg = (avg_sequential / avg_batched - 1) * 100
    
    print("End-to-End (total):")
    print(f"  Sequential (original):  {best_sequential:.3f}s (best), {avg_sequential:.3f}s (avg)")
    print(f"  Batched (optimized):    {best_batched:.3f}s (best), {avg_batched:.3f}s (avg)")
    print(f"  Speedup:                {speedup_best:+.1f}% (best), {speedup_avg:+.1f}% (avg)")
    print()
    
    # Diffusion-only timing (what we're optimizing)
    best_seq_diffusion = min(diffusion_times["sequential"])
    avg_seq_diffusion = sum(diffusion_times["sequential"]) / len(diffusion_times["sequential"])
    best_batch_diffusion = min(diffusion_times["batched"])
    avg_batch_diffusion = sum(diffusion_times["batched"]) / len(diffusion_times["batched"])
    
    speedup_diffusion_best = (best_seq_diffusion / best_batch_diffusion - 1) * 100
    speedup_diffusion_avg = (avg_seq_diffusion / avg_batch_diffusion - 1) * 100
    
    print("Diffusion Only (optimized part):")
    print(f"  Sequential (original):  {best_seq_diffusion:.3f}s (best), {avg_seq_diffusion:.3f}s (avg)")
    print(f"  Batched (optimized):    {best_batch_diffusion:.3f}s (best), {avg_batch_diffusion:.3f}s (avg)")
    print(f"  Speedup:                {speedup_diffusion_best:+.1f}% (best), {speedup_diffusion_avg:+.1f}% (avg)")
    print()
    
    if speedup_diffusion_best > 0:
        print(f"✓ CFG optimization successful: {speedup_diffusion_best:.1f}% faster diffusion!")
    else:
        print(f"⚠ No speedup observed in diffusion (unexpected)")
    print()
    
    # =================================================================
    # Verify Correctness (compare video outputs)
    # =================================================================
    print("="*70)
    print("Correctness Verification")
    print("="*70)
    print(f"Comparing run 0 outputs (both used seed={args.seed})")
    print()
    
    # Load the output videos and compare
    import cv2
    import numpy as np
    
    seq_video_path = output_path / "sequential_run0" / "sequential_run0.mp4"
    batch_video_path = output_path / "batched_run0" / "batched_run0.mp4"
    
    if seq_video_path.exists() and batch_video_path.exists():
        cap_seq = cv2.VideoCapture(str(seq_video_path))
        cap_batch = cv2.VideoCapture(str(batch_video_path))
        
        frame_diffs = []
        frame_idx = 0
        
        while True:
            ret_seq, frame_seq = cap_seq.read()
            ret_batch, frame_batch = cap_batch.read()
            
            if not (ret_seq and ret_batch):
                break
            
            # Compute per-pixel difference
            diff = np.abs(frame_seq.astype(float) - frame_batch.astype(float))
            max_diff = diff.max()
            mean_diff = diff.mean()
            frame_diffs.append((max_diff, mean_diff))
            frame_idx += 1
        
        cap_seq.release()
        cap_batch.release()
        
        if frame_diffs:
            max_diffs = [d[0] for d in frame_diffs]
            mean_diffs = [d[1] for d in frame_diffs]
            
            print(f"Compared {len(frame_diffs)} frames")
            print(f"Max pixel difference:  {max(max_diffs):.2f} (across all frames)")
            print(f"Mean pixel difference: {np.mean(mean_diffs):.2f}")
            print()
            
            # Tolerance check (allow for small numerical differences)
            if max(max_diffs) < 10.0:  # Within 10/255 for 8-bit video
                print("✓ Outputs are nearly identical (within tolerance)")
            else:
                print(f"⚠ Outputs differ more than expected (max diff: {max(max_diffs):.2f})")
            
            # Create side-by-side comparison video
            print()
            print("Creating visual comparison video...")
            comparison_path = output_path / "comparison_sequential_vs_batched.mp4"
            
            cap_seq = cv2.VideoCapture(str(seq_video_path))
            cap_batch = cv2.VideoCapture(str(batch_video_path))
            
            # Get video properties
            fps = cap_seq.get(cv2.CAP_PROP_FPS)
            width = int(cap_seq.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_seq.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer for side-by-side comparison
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(comparison_path), fourcc, fps, (width * 2, height))
            
            while True:
                ret_seq, frame_seq = cap_seq.read()
                ret_batch, frame_batch = cap_batch.read()
                
                if not (ret_seq and ret_batch):
                    break
                
                # Add labels
                cv2.putText(frame_seq, "Sequential", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_batch, "Batched CFG", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Concatenate horizontally
                combined = np.hstack([frame_seq, frame_batch])
                out.write(combined)
            
            cap_seq.release()
            cap_batch.release()
            out.release()
            
            print(f"✓ Comparison video saved to: {comparison_path}")
        else:
            print("⚠ Could not compare frames")
    else:
        print(f"⚠ Output videos not found for comparison")
        print(f"  Expected: {seq_video_path}")
        print(f"  Expected: {batch_video_path}")
    print()
    
    print("="*70)
    print("Test Complete")
    print("="*70)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

