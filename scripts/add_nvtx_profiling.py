#!/usr/bin/env python3
"""
NVTX profiling wrapper for Cosmos Transfer2 batch inference.

This script patches key functions to add NVTX range markers for better
visibility in Nsight Systems profiles.

Usage:
    python add_nvtx_profiling.py --config <config.yaml> --output_dir <output>
"""

import torch
import functools

# Try to import nvtx - if not available, create no-op decorator
try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    class nvtx:
        @staticmethod
        def annotate(message="", color=None):
            def decorator(func):
                return func
            return decorator
        
        @staticmethod
        def push_range(message="", color=None):
            pass
        
        @staticmethod
        def pop_range():
            pass


def patch_with_nvtx(cls, method_name, nvtx_name, color="blue"):
    """Patch a class method to add NVTX range annotation."""
    original_method = getattr(cls, method_name)
    
    @functools.wraps(original_method)
    def wrapped(self, *args, **kwargs):
        nvtx.push_range(nvtx_name, color=color)
        try:
            result = original_method(self, *args, **kwargs)
        finally:
            nvtx.pop_range()
        return result
    
    setattr(cls, method_name, wrapped)
    print(f"Patched {cls.__name__}.{method_name} with NVTX marker: {nvtx_name}")


def patch_velocity_fn_with_nvtx(original_get_velocity_fn):
    """Wrap velocity_fn creation to add NVTX markers for condition/uncondition."""
    
    @functools.wraps(original_get_velocity_fn)
    def patched_get_velocity_fn(self, data_batch, guidance=1.5, is_negative_prompt=False):
        # Get the original velocity_fn
        original_velocity_fn = original_get_velocity_fn(self, data_batch, guidance, is_negative_prompt)
        
        # Track diffusion step counter
        step_counter = [0]
        
        @functools.wraps(original_velocity_fn)
        def wrapped_velocity_fn(noise, noise_x, timestep):
            step_counter[0] += 1
            step_name = f"Diffusion Step {step_counter[0]}"
            
            nvtx.push_range(step_name, color="green")
            try:
                # The original velocity_fn does:
                # cond_v = self.denoise(condition)
                # uncond_v = self.denoise(uncondition)
                # We can't easily separate these without modifying the original code
                result = original_velocity_fn(noise, noise_x, timestep)
            finally:
                nvtx.pop_range()
            return result
        
        return wrapped_velocity_fn
    
    return patched_get_velocity_fn


def apply_nvtx_patches():
    """Apply NVTX patches to Cosmos Transfer2 inference code."""
    
    if not HAS_NVTX:
        print("WARNING: nvtx not available. Install with: pip install nvtx")
        print("Profiling markers will be no-ops.")
    
    # Patch the inference pipeline
    try:
        from cosmos_transfer2._src.transfer2.inference.inference_pipeline import Img2WorldInferencePipeline
        patch_with_nvtx(Img2WorldInferencePipeline, "generate_img2world", "generate_img2world (sequential)", color="blue")
        patch_with_nvtx(Img2WorldInferencePipeline, "generate_img2world_batch", "generate_img2world_batch", color="purple")
    except ImportError as e:
        print(f"Could not patch inference pipeline: {e}")
    
    # Patch the model's generate_samples_from_batch
    try:
        from cosmos_transfer2._src.predict2.models.text2world_model_rectified_flow import Text2WorldModelRectifiedFlow
        patch_with_nvtx(Text2WorldModelRectifiedFlow, "generate_samples_from_batch", "generate_samples_from_batch", color="yellow")
    except ImportError as e:
        print(f"Could not patch Text2WorldModelRectifiedFlow: {e}")
    
    # Patch the control model's velocity function
    try:
        from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import ControlVideo2WorldModelRectifiedFlow
        original_method = ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch
        ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch = patch_velocity_fn_with_nvtx(original_method)
        print("Patched ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch with diffusion step markers")
    except ImportError as e:
        print(f"Could not patch velocity function: {e}")
    
    # Patch denoise to see condition vs uncondition
    try:
        from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import ControlVideo2WorldModelRectifiedFlow
        
        original_denoise = ControlVideo2WorldModelRectifiedFlow.denoise
        denoise_counter = [0]
        
        @functools.wraps(original_denoise)
        def patched_denoise(self, noise, xt, timesteps, condition):
            denoise_counter[0] += 1
            # Alternate between condition and uncondition
            phase = "Conditioned" if denoise_counter[0] % 2 == 1 else "Unconditioned"
            nvtx.push_range(f"Denoise ({phase})", color="orange" if phase == "Conditioned" else "red")
            try:
                result = original_denoise(self, noise, xt, timesteps, condition)
            finally:
                nvtx.pop_range()
            return result
        
        ControlVideo2WorldModelRectifiedFlow.denoise = patched_denoise
        print("Patched denoise with condition/uncondition markers")
    except Exception as e:
        print(f"Could not patch denoise: {e}")
    
    # Patch VAE encode/decode
    try:
        from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import CausalVAETokenizer
        patch_with_nvtx(CausalVAETokenizer, "encode", "VAE Encode", color="cyan")
        patch_with_nvtx(CausalVAETokenizer, "decode", "VAE Decode", color="magenta")
    except ImportError as e:
        print(f"Could not patch VAE tokenizer: {e}")
    
    print("\nNVTX patches applied. Run with nsys profile to see markers.")


if __name__ == "__main__":
    print("NVTX Profiling Patches for Cosmos Transfer2")
    print("=" * 50)
    apply_nvtx_patches()
    
    print("\nTo use these patches, import this module before running inference:")
    print("  import add_nvtx_profiling")
    print("  add_nvtx_profiling.apply_nvtx_patches()")
    print("\nThen run with nsys:")
    print("  nsys profile --trace=cuda,nvtx python your_inference_script.py")

