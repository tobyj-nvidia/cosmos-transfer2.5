#!/usr/bin/env python3
"""
Batch inference test with granular NVTX v3 profiling markers.

This wraps test_batch_inference_real.py with v3 NVTX patches applied
for detailed visibility into WHERE time is spent:
- VELOCITY_FN_SETUP: Everything before diffusion (conditioning, VAE encode, control)
- GET_DATA_AND_CONDITION: Control input processing
- DATA_PREP_*: Input normalization/augmentation
- STEP_N: Individual diffusion steps
- DENOISE_COND/UNCOND: CFG passes
- VAE_DECODE: Final decoding
"""
import sys
import os

# Add the repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply NVTX patches BEFORE importing cosmos modules
# Use v3 for granular profiling
from scripts.nvtx_profiling_v3 import apply_all_patches
apply_all_patches()

# Now run the actual test
from tests.test_batch_inference_real import test_batch_inference_real
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test batch inference with granular NVTX v3 profiling")
    parser.add_argument("--video1", required=True, help="Path to first video")
    parser.add_argument("--video2", required=True, help="Path to second video")
    parser.add_argument("--output-dir", default="./batch_test_output", help="Output directory")
    parser.add_argument("--prompt", default="A robotic arm manipulating objects on a table.", help="Prompt")
    parser.add_argument("--num-steps", type=int, default=4, help="Diffusion steps")
    parser.add_argument("--guidance", type=int, default=7, help="Guidance scale")
    parser.add_argument("--state-t", type=int, default=24, 
                        help="Latent temporal frames (2=5 pixel, 4=13 pixel, 7=25 pixel, 24=93 pixel)")
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="Enable CUDA Graphs to reduce kernel launch overhead")
    
    args = parser.parse_args()
    
    success = test_batch_inference_real(
        video1_path=args.video1,
        video2_path=args.video2,
        output_dir=args.output_dir,
        prompt=args.prompt,
        num_steps=args.num_steps,
        guidance=args.guidance,
        state_t=args.state_t,
        use_cuda_graphs=args.cuda_graphs,
    )
    
    sys.exit(0 if success else 1)

