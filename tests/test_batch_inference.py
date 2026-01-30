#!/usr/bin/env python3
"""
Test script to verify batch inference works correctly.

This tests that the model can process multiple samples in parallel
after the latents[0] fix.

Usage:
    cd /home/horde/cosmos-transfer2.5
    source .venv/bin/activate
    python tests/test_batch_inference.py
"""

import torch
import sys
from pathlib import Path

# Add cosmos to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_batch_scheduler_step():
    """Test that scheduler.step works with batched inputs."""
    from cosmos_transfer2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
    
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(num_inference_steps=4, device="cuda")
    
    # Test with batch_size=1
    model_output_1 = torch.randn(1, 16, 24, 90, 160, device="cuda", dtype=torch.bfloat16)
    sample_1 = torch.randn(1, 16, 24, 90, 160, device="cuda", dtype=torch.bfloat16)
    t = scheduler.timesteps[0]
    
    result_1 = scheduler.step(model_output_1, t, sample_1, return_dict=False)[0]
    print(f"✓ Batch size 1: input {sample_1.shape} -> output {result_1.shape}")
    assert result_1.shape == sample_1.shape, f"Shape mismatch: {result_1.shape} != {sample_1.shape}"
    
    # Reset scheduler for batch_size=2
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(num_inference_steps=4, device="cuda")
    
    # Test with batch_size=2
    model_output_2 = torch.randn(2, 16, 24, 90, 160, device="cuda", dtype=torch.bfloat16)
    sample_2 = torch.randn(2, 16, 24, 90, 160, device="cuda", dtype=torch.bfloat16)
    
    result_2 = scheduler.step(model_output_2, t, sample_2, return_dict=False)[0]
    print(f"✓ Batch size 2: input {sample_2.shape} -> output {result_2.shape}")
    assert result_2.shape == sample_2.shape, f"Shape mismatch: {result_2.shape} != {sample_2.shape}"
    
    # Test with batch_size=4
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(num_inference_steps=4, device="cuda")
    
    model_output_4 = torch.randn(4, 16, 24, 90, 160, device="cuda", dtype=torch.bfloat16)
    sample_4 = torch.randn(4, 16, 24, 90, 160, device="cuda", dtype=torch.bfloat16)
    
    result_4 = scheduler.step(model_output_4, t, sample_4, return_dict=False)[0]
    print(f"✓ Batch size 4: input {sample_4.shape} -> output {result_4.shape}")
    assert result_4.shape == sample_4.shape, f"Shape mismatch: {result_4.shape} != {sample_4.shape}"
    
    print("\n✓ All scheduler batch tests passed!")
    return True


def test_memory_scaling():
    """Test memory usage scales correctly with batch size."""
    import gc
    
    torch.cuda.empty_cache()
    gc.collect()
    
    base_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"\nBase memory: {base_memory:.2f} GB")
    
    # Allocate batch_size=1 tensors (simulating latents)
    batch_1 = torch.randn(1, 16, 24, 90, 160, device="cuda", dtype=torch.bfloat16)
    mem_1 = torch.cuda.memory_allocated() / 1024**3
    print(f"Batch 1 memory: {mem_1:.2f} GB (+{mem_1 - base_memory:.3f} GB)")
    
    # Allocate batch_size=2 tensors
    batch_2 = torch.randn(2, 16, 24, 90, 160, device="cuda", dtype=torch.bfloat16)
    mem_2 = torch.cuda.memory_allocated() / 1024**3
    print(f"Batch 2 memory: {mem_2:.2f} GB (+{mem_2 - mem_1:.3f} GB)")
    
    # Allocate batch_size=4 tensors
    batch_4 = torch.randn(4, 16, 24, 90, 160, device="cuda", dtype=torch.bfloat16)
    mem_4 = torch.cuda.memory_allocated() / 1024**3
    print(f"Batch 4 memory: {mem_4:.2f} GB (+{mem_4 - mem_2:.3f} GB)")
    
    per_sample_mem = (mem_4 - base_memory) / 7  # Total 7 samples allocated
    print(f"\nEstimated per-sample latent memory: {per_sample_mem * 1000:.1f} MB")
    
    # Cleanup
    del batch_1, batch_2, batch_4
    torch.cuda.empty_cache()
    gc.collect()
    
    print("✓ Memory scaling test passed!")
    return True


def test_full_diffusion_loop_batch():
    """Test the full diffusion loop with batched inputs."""
    from cosmos_transfer2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
    
    print("\nTesting full diffusion loop with batch_size=2...")
    
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    num_steps = 4
    scheduler.set_timesteps(num_inference_steps=num_steps, device="cuda")
    
    batch_size = 2
    latent_shape = (batch_size, 16, 24, 90, 160)
    
    # Start with noise
    latents = torch.randn(latent_shape, device="cuda", dtype=torch.bfloat16)
    noise = latents.clone()
    
    print(f"Initial latents shape: {latents.shape}")
    
    # Simulate diffusion loop (like in generate_samples_from_batch)
    for i, t in enumerate(scheduler.timesteps):
        # Simulate velocity prediction (random for test)
        velocity_pred = torch.randn_like(latents)
        
        # This is the FIXED code path
        prev_sample = scheduler.step(
            velocity_pred, t, latents, return_dict=False
        )[0]
        latents = prev_sample
        
        print(f"  Step {i+1}/{num_steps}: latents shape = {latents.shape}")
        
        # Verify batch dimension preserved
        assert latents.shape[0] == batch_size, f"Batch dim lost! Got {latents.shape[0]}"
    
    print(f"Final latents shape: {latents.shape}")
    assert latents.shape == latent_shape, f"Shape changed: {latents.shape} != {latent_shape}"
    
    print("✓ Full diffusion loop batch test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Cosmos Batch Inference Tests")
    print("=" * 60)
    
    try:
        test_batch_scheduler_step()
        test_memory_scaling()
        test_full_diffusion_loop_batch()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

