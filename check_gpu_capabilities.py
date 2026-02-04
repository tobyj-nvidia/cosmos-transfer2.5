#!/usr/bin/env python3
"""Check GPU capabilities for optimization opportunities."""

import torch

print("=" * 70)
print("GPU Hardware Capabilities")
print("=" * 70)

# Basic GPU info
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
compute_cap = torch.cuda.get_device_capability(0)
print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

# Check for FP8 support (PyTorch 2.1+)
has_fp8 = hasattr(torch, 'float8_e4m3fn') and hasattr(torch, 'float8_e5m2')
print(f"PyTorch FP8 dtypes available: {has_fp8}")

if compute_cap[0] >= 9:  # Blackwell (9.0) and later
    print("✅ Hardware supports FP8 Tensor Cores (5th gen)")
elif compute_cap[0] >= 8 and compute_cap[1] >= 9:  # Hopper (8.9)
    print("✅ Hardware supports FP8 Tensor Cores (4th gen)")
else:
    print("❌ Hardware does not support FP8 Tensor Cores")

print("\n" + "=" * 70)
print("Available dtypes in PyTorch")
print("=" * 70)
print(f"torch.float32: ✅")
print(f"torch.float16: ✅")
print(f"torch.bfloat16: ✅")
if has_fp8:
    print(f"torch.float8_e4m3fn: ✅")
    print(f"torch.float8_e5m2: ✅")

print("\n" + "=" * 70)
print("Current Cosmos Model dtype")
print("=" * 70)

# Try to check what Cosmos is using
try:
    from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import ControlLVGDiT
    print("Checking Cosmos code for dtype usage...")
    
    # Check the source for dtype hints
    import inspect
    source = inspect.getsource(ControlLVGDiT.__init__)
    if 'bfloat16' in source:
        print("Found: Model uses bfloat16 ✅")
    elif 'float16' in source:
        print("Found: Model uses float16")
    elif 'float32' in source:
        print("Found: Model uses float32")
    else:
        print("Could not determine dtype from source")
        
except Exception as e:
    print(f"Could not inspect Cosmos code: {e}")

print("\n" + "=" * 70)
print("Optimization Recommendations")
print("=" * 70)

if compute_cap[0] >= 9:
    print("🚀 MAJOR OPPORTUNITY: Your Blackwell GPU has FP8 Tensor Cores!")
    if has_fp8:
        print("   → PyTorch 2.1+ with FP8 support detected")
        print("   → Quantizing to FP8 could give ~2x speedup")
        print("   → Tools to try: transformer-engine, quanto, torchao")
    else:
        print("   → Upgrade PyTorch to 2.1+ for FP8 support")
        print("   → Or use NVIDIA Transformer Engine")

print("\n💡 Flash Attention 2.7.3 is already optimal for current dtype")
print("   → Flash Attention 3 not widely released yet (as of early 2025)")

if compute_cap[0] >= 9:
    print("\n🔧 Next steps:")
    print("   1. Try FP8 quantization (biggest potential gain)")
    print("   2. Export to TensorRT for graph-level optimization")
    print("   3. Experiment with fewer diffusion steps (20-25)")

