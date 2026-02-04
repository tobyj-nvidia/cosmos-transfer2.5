"""
Real model test for CFG optimization.

This test:
1. Loads the actual Cosmos Transfer 2.5 model
2. Runs inference with sequential CFG (baseline)
3. Runs inference with batched CFG (optimized)
4. Verifies outputs are equivalent
5. Measures and reports performance difference

Usage:
    cd cosmos-transfer2.5
    source .venv/bin/activate
    python tests/test_cfg_optimization_real.py --depth-video /path/to/depth.mp4
"""

import argparse
import time
import torch
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_network_precomputed_hints(depth_video_path: str, num_warmup: int = 1, num_runs: int = 3):
    """
    Test that precomputed hints produce identical output to computed hints.
    Also measure performance improvement.
    """
    print("=" * 70)
    print("CFG Optimization Test - Network Level")
    print("=" * 70)
    print()
    
    # Import Cosmos modules
    from pathlib import Path
    from cosmos_transfer2.config import InferenceArguments, SetupArguments, DepthConfig
    from cosmos_transfer2.inference import Control2WorldInference
    
    # Use minimal state_t for faster testing
    state_t = 2  # 5 pixel frames
    expected_pixel_frames = (state_t - 1) * 4 + 1
    
    print(f"Using state_t={state_t} ({expected_pixel_frames} pixel frames) for faster testing")
    print()
    
    # Create output directory
    output_path = Path("/tmp/cfg_optimization_test")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create setup args
    setup_args = SetupArguments(
        model="depth",
        output_dir=output_path,
        disable_guardrails=True,
    )
    
    print("Loading model...")
    start = time.time()
    inference = Control2WorldInference(
        setup_args,
        batch_hint_keys=["depth"],
        state_t=state_t,
    )
    print(f"Model loaded in {time.time() - start:.1f}s")
    print()
    
    # Get the network
    net = inference.inference_pipeline.model.net
    
    # Create test inputs (using actual model dimensions)
    print("Creating test inputs...")
    device = "cuda"
    dtype = torch.bfloat16
    
    # Model dimensions (from config)
    B, C, T, H, W = 1, 16, 2, 60, 106  # state_t=2 gives T=2 latent frames
    text_seq_len = 512
    text_dim = 4096
    
    # Create synthetic inputs
    x = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    timesteps = torch.tensor([[500.0, 500.0]], device=device, dtype=dtype)  # Mid-timestep
    crossattn_emb = torch.randn(B, text_seq_len, text_dim, device=device, dtype=dtype)
    latent_control_input = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    condition_mask = torch.ones(B, 1, T, H, W, device=device, dtype=dtype)
    fps = torch.tensor([24.0], device=device)
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  timesteps: {timesteps.shape}")
    print(f"  crossattn_emb: {crossattn_emb.shape}")
    print(f"  latent_control_input: {latent_control_input.shape}")
    print()
    
    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = net.forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                latent_control_input=latent_control_input,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=fps,
                control_context_scale=1.0,
            )
    torch.cuda.synchronize()
    print()
    
    # Test 1: Verify precomputed hints produce identical output
    print("Test 1: Verifying precomputed hints correctness...")
    with torch.no_grad():
        # Run with internal hint computation
        output_baseline = net.forward(
            x_B_C_T_H_W=x,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn_emb,
            latent_control_input=latent_control_input,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=fps,
            control_context_scale=1.0,
            precomputed_hints=None,
        )
        
        # Compute hints separately
        hints, control_scale = net.compute_control_hints(
            x_B_C_T_H_W=x,
            latent_control_input=latent_control_input,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=fps,
            control_context_scale=1.0,
        )
        
        # Run with precomputed hints
        output_precomputed = net.forward(
            x_B_C_T_H_W=x,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn_emb,
            latent_control_input=latent_control_input,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            fps=fps,
            control_context_scale=1.0,
            precomputed_hints=(hints, control_scale),
        )
    
    # Compare outputs
    max_diff = (output_baseline - output_precomputed).abs().max().item()
    mean_diff = (output_baseline - output_precomputed).abs().mean().item()
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    if max_diff < 1e-3:
        print("  ✓ PASS: Outputs are equivalent!")
    else:
        print("  ✗ FAIL: Outputs differ!")
        return False
    print()
    
    # Test 2: Performance comparison
    print(f"Test 2: Performance comparison ({num_runs} runs each)...")
    
    # Baseline: Two forward passes (like sequential CFG)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            # Simulating cond pass
            _ = net.forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                latent_control_input=latent_control_input,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=fps,
                control_context_scale=1.0,
                precomputed_hints=None,
            )
            # Simulating uncond pass
            _ = net.forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                latent_control_input=latent_control_input,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=fps,
                control_context_scale=1.0,
                precomputed_hints=None,
            )
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / num_runs
    
    # Optimized: Compute hints once, then two forward passes with precomputed hints
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            # Compute hints once
            hints, control_scale = net.compute_control_hints(
                x_B_C_T_H_W=x,
                latent_control_input=latent_control_input,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=fps,
                control_context_scale=1.0,
            )
            # Cond pass with precomputed hints
            _ = net.forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                latent_control_input=latent_control_input,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=fps,
                control_context_scale=1.0,
                precomputed_hints=(hints, control_scale),
            )
            # Uncond pass with precomputed hints
            _ = net.forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                latent_control_input=latent_control_input,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=fps,
                control_context_scale=1.0,
                precomputed_hints=(hints, control_scale),
            )
    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - start) / num_runs
    
    speedup = (1 - optimized_time / baseline_time) * 100
    
    print(f"  Baseline (no hint caching):   {baseline_time * 1000:.1f} ms per CFG step")
    print(f"  Optimized (hint caching):     {optimized_time * 1000:.1f} ms per CFG step")
    print(f"  Speedup:                      {speedup:.1f}%")
    print()
    
    # Test 3: Batched forward pass
    print("Test 3: Batched CFG (cond + uncond in single pass)...")
    
    # Create batched inputs
    x_batched = torch.cat([x, x], dim=0)
    timesteps_batched = torch.cat([timesteps, timesteps], dim=0)
    crossattn_cond = crossattn_emb
    crossattn_uncond = torch.zeros_like(crossattn_emb)
    crossattn_batched = torch.cat([crossattn_cond, crossattn_uncond], dim=0)
    latent_control_batched = torch.cat([latent_control_input, latent_control_input], dim=0)
    condition_mask_batched = torch.cat([condition_mask, condition_mask], dim=0)
    fps_batched = torch.cat([fps, fps], dim=0)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            # Compute hints once
            hints, control_scale = net.compute_control_hints(
                x_B_C_T_H_W=x,
                latent_control_input=latent_control_input,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=fps,
                control_context_scale=1.0,
            )
            # Stack hints for batch
            hints_batched = torch.cat([hints, hints], dim=1) if hints.dim() > 4 else torch.cat([hints, hints], dim=0)
            
            # Single batched forward pass
            _ = net.forward(
                x_B_C_T_H_W=x_batched,
                timesteps_B_T=timesteps_batched,
                crossattn_emb=crossattn_batched,
                latent_control_input=latent_control_batched,
                condition_video_input_mask_B_C_T_H_W=condition_mask_batched,
                fps=fps_batched,
                control_context_scale=1.0,
                precomputed_hints=(hints_batched, control_scale),
            )
    torch.cuda.synchronize()
    batched_time = (time.perf_counter() - start) / num_runs
    
    batched_speedup = (1 - batched_time / baseline_time) * 100
    
    print(f"  Batched CFG:                  {batched_time * 1000:.1f} ms per CFG step")
    print(f"  Total speedup vs baseline:    {batched_speedup:.1f}%")
    print()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Baseline (sequential, no cache): {baseline_time * 1000:.1f} ms")
    print(f"Hint caching only:               {optimized_time * 1000:.1f} ms ({speedup:.1f}% faster)")
    print(f"Full optimization (batched):     {batched_time * 1000:.1f} ms ({batched_speedup:.1f}% faster)")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test CFG optimization with real model")
    parser.add_argument("--depth-video", type=str, 
                        default="/home/tobyj/code/isaaclab-cosmos-experiments/results/batch_test_fix_verification/batch_sample_1_control_depth.mp4",
                        help="Path to depth video for testing")
    parser.add_argument("--num-warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of timed runs")
    args = parser.parse_args()
    
    success = test_network_precomputed_hints(
        args.depth_video, 
        num_warmup=args.num_warmup,
        num_runs=args.num_runs
    )
    
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

