#!/usr/bin/env python3
"""
NVTX profiling for CFG optimization test.

This is adapted from nvtx_profiling_v3.py but SKIPS the top-level
Control2WorldInference.generate wrapper to allow the CFG test's own
markers (SEQUENTIAL_CFG_TEST, BATCHED_CFG_TEST) to show through.

Adds all the granular markers:
- DIFFUSION_SAMPLING(steps=N)
- STEP_1, STEP_2, ... (per diffusion step)
- VAE_ENCODE, VAE_DECODE
- DATA_PREP_NORMALIZE, DATA_PREP_AUGMENT
- GET_DATA_AND_CONDITION
- VELOCITY_FN_SETUP
- DiT_FORWARD
- And more
"""

import functools
import torch

# Use torch.cuda.nvtx for Nsight Systems compatibility
HAS_NVTX = hasattr(torch.cuda, 'nvtx')
if HAS_NVTX:
    print("Using torch.cuda.nvtx for profiling markers")
else:
    print("WARNING: torch.cuda.nvtx not available")


class NVTXContext:
    """Context manager for NVTX ranges using torch.cuda.nvtx."""
    def __init__(self, message, color=None):
        self.message = message
        
    def __enter__(self):
        if HAS_NVTX:
            torch.cuda.nvtx.range_push(self.message)
        return self
    
    def __exit__(self, *args):
        if HAS_NVTX:
            torch.cuda.nvtx.range_pop()


# Global state for tracking
_state = {
    'diffusion_step': 0,
    'denoise_call': 0,
}


def reset_state():
    """Reset tracking state."""
    _state['diffusion_step'] = 0
    _state['denoise_call'] = 0


def apply_all_patches():
    """Apply all NVTX patches for comprehensive profiling (CFG-compatible version)."""
    
    print("=" * 70)
    print("Applying CFG-compatible NVTX profiling patches...")
    print("=" * 70)
    
    # NOTE: We do NOT patch Control2WorldInference.generate at the top level
    # because the CFG test adds its own markers (SEQUENTIAL_CFG_TEST, BATCHED_CFG_TEST)
    
    # 1. Patch generate_samples_from_batch with granular markers
    try:
        print("  Patching Text2WorldModelRectifiedFlow...", flush=True)
        from cosmos_transfer2._src.predict2.models.text2world_model_rectified_flow import Text2WorldModelRectifiedFlow
        
        # Patch _normalize_video_databatch_inplace
        if hasattr(Text2WorldModelRectifiedFlow, '_normalize_video_databatch_inplace'):
            original_normalize = Text2WorldModelRectifiedFlow._normalize_video_databatch_inplace
            @functools.wraps(original_normalize)
            def patched_normalize(self, *args, **kwargs):
                with NVTXContext("DATA_PREP_NORMALIZE"):
                    return original_normalize(self, *args, **kwargs)
            Text2WorldModelRectifiedFlow._normalize_video_databatch_inplace = patched_normalize
            print("    ✓ Patched _normalize_video_databatch_inplace")
        
        # Patch _augment_image_dim_inplace
        if hasattr(Text2WorldModelRectifiedFlow, '_augment_image_dim_inplace'):
            original_augment = Text2WorldModelRectifiedFlow._augment_image_dim_inplace
            @functools.wraps(original_augment)
            def patched_augment(self, *args, **kwargs):
                with NVTXContext("DATA_PREP_AUGMENT"):
                    return original_augment(self, *args, **kwargs)
            Text2WorldModelRectifiedFlow._augment_image_dim_inplace = patched_augment
            print("    ✓ Patched _augment_image_dim_inplace")
        
        # Patch the main diffusion loop
        original_gen_samples = Text2WorldModelRectifiedFlow.generate_samples_from_batch
        @functools.wraps(original_gen_samples)
        def patched_gen_samples(self, data_batch, num_steps=35, **kwargs):
            reset_state()  # Reset step counter for each inference
            with NVTXContext(f"DIFFUSION_SAMPLING(steps={num_steps})"):
                return original_gen_samples(self, data_batch, num_steps=num_steps, **kwargs)
        Text2WorldModelRectifiedFlow.generate_samples_from_batch = patched_gen_samples
        print("    ✓ Patched generate_samples_from_batch")
        
    except Exception as e:
        print(f"    ✗ Could not patch Text2WorldModelRectifiedFlow: {type(e).__name__}: {e}")
    
    # 2. Patch ControlVideo2WorldModelRectifiedFlow methods
    try:
        print("  Patching ControlVideo2WorldModelRectifiedFlow...", flush=True)
        from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import ControlVideo2WorldModelRectifiedFlow
        
        # Patch get_velocity_fn_from_batch (where conditioning setup happens)
        original_get_vel = ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch
        @functools.wraps(original_get_vel)
        def patched_get_vel(self, data_batch, guidance=1.5, is_negative_prompt=False, **kwargs):
            with NVTXContext("VELOCITY_FN_SETUP"):
                original_velocity_fn = original_get_vel(self, data_batch, guidance, is_negative_prompt, **kwargs)
            
            @functools.wraps(original_velocity_fn)
            def wrapped_velocity_fn(noise, noise_x, timestep):
                _state['diffusion_step'] += 1
                step = _state['diffusion_step']
                with NVTXContext(f"STEP_{step}"):
                    return original_velocity_fn(noise, noise_x, timestep)
            
            return wrapped_velocity_fn
        
        ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch = patched_get_vel
        print("    ✓ Patched get_velocity_fn_from_batch with per-step markers")
        
        # Patch get_data_and_condition (control input encoding)
        if hasattr(ControlVideo2WorldModelRectifiedFlow, 'get_data_and_condition'):
            original_get_data = ControlVideo2WorldModelRectifiedFlow.get_data_and_condition
            @functools.wraps(original_get_data)
            def patched_get_data(self, data_batch):
                with NVTXContext("GET_DATA_AND_CONDITION"):
                    return original_get_data(self, data_batch)
            ControlVideo2WorldModelRectifiedFlow.get_data_and_condition = patched_get_data
            print("    ✓ Patched get_data_and_condition")
        
        # Patch denoise with condition/uncondition tracking
        if hasattr(ControlVideo2WorldModelRectifiedFlow, 'denoise'):
            original_denoise = ControlVideo2WorldModelRectifiedFlow.denoise
            @functools.wraps(original_denoise)
            def patched_denoise(self, noise, xt, timesteps, condition):
                _state['denoise_call'] += 1
                is_cond = (_state['denoise_call'] % 2) == 1
                phase = "COND" if is_cond else "UNCOND"
                
                with NVTXContext(f"DENOISE_{phase}"):
                    return original_denoise(self, noise, xt, timesteps, condition)
            
            ControlVideo2WorldModelRectifiedFlow.denoise = patched_denoise
            print("    ✓ Patched denoise with COND/UNCOND tracking")
        
    except Exception as e:
        print(f"    ✗ Could not patch ControlVideo2WorldModelRectifiedFlow: {type(e).__name__}: {e}")
    
    # 3. Patch VAE tokenizer encode/decode
    try:
        print("  Patching VAE tokenizer...", flush=True)
        from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        
        original_encode = Wan2pt1VAEInterface.encode
        @functools.wraps(original_encode)
        def patched_encode(self, *args, **kwargs):
            with NVTXContext("VAE_ENCODE"):
                return original_encode(self, *args, **kwargs)
        Wan2pt1VAEInterface.encode = patched_encode
        print("    ✓ Patched Wan2pt1VAEInterface.encode")
        
        original_decode = Wan2pt1VAEInterface.decode
        @functools.wraps(original_decode)
        def patched_decode(self, *args, **kwargs):
            with NVTXContext("VAE_DECODE"):
                return original_decode(self, *args, **kwargs)
        Wan2pt1VAEInterface.decode = patched_decode
        print("    ✓ Patched Wan2pt1VAEInterface.decode")
        
    except Exception as e:
        print(f"    ✗ Could not patch VAE tokenizer: {type(e).__name__}: {e}")
    
    # 4. Patch the DiT network forward
    try:
        print("  Patching DiT network...", flush=True)
        from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import MinimalV4LVGControlVaceDiT
        
        original_net_forward = MinimalV4LVGControlVaceDiT.forward
        @functools.wraps(original_net_forward)
        def patched_net_forward(self, *args, **kwargs):
            with NVTXContext("DiT_FORWARD"):
                return original_net_forward(self, *args, **kwargs)
        MinimalV4LVGControlVaceDiT.forward = patched_net_forward
        print("    ✓ Patched DiT forward")
        
    except Exception as e:
        print(f"    ✗ Could not patch DiT network: {type(e).__name__}: {e}")
    
    # 5. Patch model.decode (at the model level)
    try:
        print("  Patching model encode/decode...", flush=True)
        from cosmos_transfer2._src.predict2.models.text2world_model_rectified_flow import Text2WorldModelRectifiedFlow
        
        if hasattr(Text2WorldModelRectifiedFlow, 'decode'):
            original_model_decode = Text2WorldModelRectifiedFlow.decode
            @functools.wraps(original_model_decode)
            def patched_model_decode(self, *args, **kwargs):
                with NVTXContext("MODEL_DECODE"):
                    return original_model_decode(self, *args, **kwargs)
            Text2WorldModelRectifiedFlow.decode = patched_model_decode
            print("    ✓ Patched Text2WorldModelRectifiedFlow.decode")
        
        if hasattr(Text2WorldModelRectifiedFlow, 'encode'):
            original_model_encode = Text2WorldModelRectifiedFlow.encode
            @functools.wraps(original_model_encode)
            def patched_model_encode(self, *args, **kwargs):
                with NVTXContext("MODEL_ENCODE"):
                    return original_model_encode(self, *args, **kwargs)
            Text2WorldModelRectifiedFlow.encode = patched_model_encode
            print("    ✓ Patched Text2WorldModelRectifiedFlow.encode")
            
    except Exception as e:
        print(f"    ✗ Could not patch model encode/decode: {type(e).__name__}: {e}")
    
    # 6. Patch pipeline methods (generate_img2world, etc) - these are from inference_pipeline.py
    try:
        print("  Patching ControlVideo2WorldInference pipeline...", flush=True)
        from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
        
        if hasattr(ControlVideo2WorldInference, 'generate_img2world'):
            original_gen_img2world = ControlVideo2WorldInference.generate_img2world
            @functools.wraps(original_gen_img2world)
            def patched_gen_img2world(self, *args, **kwargs):
                with NVTXContext("GENERATE_IMG2WORLD"):
                    return original_gen_img2world(self, *args, **kwargs)
            ControlVideo2WorldInference.generate_img2world = patched_gen_img2world
            print("    ✓ Patched generate_img2world")
        
        if hasattr(ControlVideo2WorldInference, 'generate_img2world_batch'):
            original_gen_batch = ControlVideo2WorldInference.generate_img2world_batch
            @functools.wraps(original_gen_batch)
            def patched_gen_batch(self, *args, **kwargs):
                with NVTXContext("GENERATE_IMG2WORLD_BATCH"):
                    return original_gen_batch(self, *args, **kwargs)
            ControlVideo2WorldInference.generate_img2world_batch = patched_gen_batch
            print("    ✓ Patched generate_img2world_batch")
        
        if hasattr(ControlVideo2WorldInference, '_get_data_batch_input'):
            original_get_data_batch = ControlVideo2WorldInference._get_data_batch_input
            @functools.wraps(original_get_data_batch)
            def patched_get_data_batch(self, *args, **kwargs):
                with NVTXContext("GET_DATA_BATCH_INPUT"):
                    return original_get_data_batch(self, *args, **kwargs)
            ControlVideo2WorldInference._get_data_batch_input = patched_get_data_batch
            print("    ✓ Patched _get_data_batch_input")
        
    except Exception as e:
        print(f"    ✗ Could not patch pipeline: {type(e).__name__}: {e}")
    
    # 7. Patch Control2WorldInference._generate_batch and _generate_sample
    try:
        print("  Patching Control2WorldInference batch/sample generation...", flush=True)
        from cosmos_transfer2.inference import Control2WorldInference
        
        if hasattr(Control2WorldInference, '_generate_batch'):
            original_gen_batch = Control2WorldInference._generate_batch
            @functools.wraps(original_gen_batch)
            def patched_gen_batch(self, *args, **kwargs):
                with NVTXContext("_GENERATE_BATCH"):
                    return original_gen_batch(self, *args, **kwargs)
            Control2WorldInference._generate_batch = patched_gen_batch
            print("    ✓ Patched _generate_batch")
        
        if hasattr(Control2WorldInference, '_generate_sample'):
            original_gen_sample = Control2WorldInference._generate_sample
            @functools.wraps(original_gen_sample)
            def patched_gen_sample(self, *args, **kwargs):
                with NVTXContext("_GENERATE_SAMPLE"):
                    return original_gen_sample(self, *args, **kwargs)
            Control2WorldInference._generate_sample = patched_gen_sample
            print("    ✓ Patched _generate_sample")
        
    except Exception as e:
        print(f"    ✗ Could not patch Control2WorldInference: {type(e).__name__}: {e}")
    
    print()
    print("=" * 70)
    print("NVTX profiling patches applied successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    apply_all_patches()
    print("\nNVTX patches applied. Run with nsys profile to see markers.")

