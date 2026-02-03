#!/usr/bin/env python3
"""
Enhanced NVTX profiling v3 for Cosmos Transfer2 inference.

Adds granular high-level markers to identify WHERE time is spent:
- DATA_PREP: Normalizing and augmenting input data
- CONDITIONING: Text/prompt embedding computation
- VAE_ENCODE_INPUT: Encoding main input video to latent space
- VAE_ENCODE_CONTROL: Encoding control inputs (depth, etc) to latent space
- CONTROL_SETUP: Setting up control conditions
- VELOCITY_FN_SETUP: Creating the velocity function (includes conditioning)
- DIFFUSION_STEP_N: Individual diffusion steps
- DENOISE_COND/UNCOND: Classifier-free guidance passes
- VAE_DECODE: Decoding latents to video

Usage:
    import nvtx_profiling_v3
    nvtx_profiling_v3.apply_all_patches()
"""

import functools
import torch

# Use torch.cuda.nvtx for Nsight Systems compatibility
# This is more reliable than the standalone nvtx package
HAS_NVTX = hasattr(torch.cuda, 'nvtx')
if HAS_NVTX:
    print("Using torch.cuda.nvtx for profiling markers")
else:
    print("WARNING: torch.cuda.nvtx not available")


# Color scheme for easy identification in Nsight
COLORS = {
    'batch_inference': 0x00FF00,      # Green
    'sequential_inference': 0xFF0000,  # Red
    'warmup': 0x808080,               # Gray
    'data_prep': 0xFFD700,            # Gold
    'conditioning': 0x9400D3,          # Purple
    'vae_encode_input': 0x00FFFF,     # Cyan
    'vae_encode_control': 0x00CED1,   # Dark Cyan
    'control_setup': 0xFF8C00,        # Dark Orange
    'velocity_fn_setup': 0xBA55D3,    # Medium Orchid
    'diffusion_step': 0xFFFF00,       # Yellow
    'denoise_cond': 0xFFA500,         # Orange
    'denoise_uncond': 0xFF00FF,       # Magenta
    'vae_decode': 0xFF69B4,           # Pink
    'dit_forward': 0xADD8E6,          # Light Blue
}


class NVTXContext:
    """Context manager for NVTX ranges using torch.cuda.nvtx."""
    def __init__(self, message, color=None):
        self.message = message
        # Note: torch.cuda.nvtx doesn't support colors, only messages
        
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
    'mode': 'unknown',
}


def reset_state():
    """Reset tracking state."""
    _state['diffusion_step'] = 0
    _state['denoise_call'] = 0


def apply_all_patches():
    """Apply all NVTX patches for comprehensive profiling."""
    import sys
    
    print("Applying NVTX profiling patches v3 (granular)...", flush=True)
    
    # 1. Patch the high-level pipeline entry points (Control2WorldInference)
    try:
        print("  Attempting to import Control2WorldInference...", flush=True)
        from cosmos_transfer2.inference import Control2WorldInference
        print("  Import successful, patching...", flush=True)
        
        original_generate = Control2WorldInference.generate
        @functools.wraps(original_generate)
        def patched_generate(self, *args, **kwargs):
            _state['mode'] = 'sequential'
            reset_state()
            with NVTXContext("=== SEQUENTIAL INFERENCE ===", COLORS['sequential_inference']):
                return original_generate(self, *args, **kwargs)
        Control2WorldInference.generate = patched_generate
        print("  ✓ Patched Control2WorldInference.generate", flush=True)
        
        if hasattr(Control2WorldInference, 'generate_batch'):
            original_gen_batch = Control2WorldInference.generate_batch
            @functools.wraps(original_gen_batch)
            def patched_gen_batch(self, *args, **kwargs):
                _state['mode'] = 'batch'
                reset_state()
                with NVTXContext("=== BATCH INFERENCE ===", COLORS['batch_inference']):
                    return original_gen_batch(self, *args, **kwargs)
            Control2WorldInference.generate_batch = patched_gen_batch
            print("  ✓ Patched Control2WorldInference.generate_batch", flush=True)
        else:
            print("  ✗ Control2WorldInference has no generate_batch method", flush=True)
            
    except Exception as e:
        print(f"  ✗ Could not patch Control2WorldInference: {type(e).__name__}: {e}", flush=True)
    
    # 2. Patch generate_samples_from_batch with granular markers
    try:
        from cosmos_transfer2._src.predict2.models.text2world_model_rectified_flow import Text2WorldModelRectifiedFlow
        
        # Patch _normalize_video_databatch_inplace
        if hasattr(Text2WorldModelRectifiedFlow, '_normalize_video_databatch_inplace'):
            original_normalize = Text2WorldModelRectifiedFlow._normalize_video_databatch_inplace
            @functools.wraps(original_normalize)
            def patched_normalize(self, *args, **kwargs):
                with NVTXContext("DATA_PREP_NORMALIZE", COLORS['data_prep']):
                    return original_normalize(self, *args, **kwargs)
            Text2WorldModelRectifiedFlow._normalize_video_databatch_inplace = patched_normalize
            print("  ✓ Patched _normalize_video_databatch_inplace")
        
        # Patch _augment_image_dim_inplace
        if hasattr(Text2WorldModelRectifiedFlow, '_augment_image_dim_inplace'):
            original_augment = Text2WorldModelRectifiedFlow._augment_image_dim_inplace
            @functools.wraps(original_augment)
            def patched_augment(self, *args, **kwargs):
                with NVTXContext("DATA_PREP_AUGMENT", COLORS['data_prep']):
                    return original_augment(self, *args, **kwargs)
            Text2WorldModelRectifiedFlow._augment_image_dim_inplace = patched_augment
            print("  ✓ Patched _augment_image_dim_inplace")
        
        # Patch the main diffusion loop
        original_gen_samples = Text2WorldModelRectifiedFlow.generate_samples_from_batch
        @functools.wraps(original_gen_samples)
        def patched_gen_samples(self, data_batch, num_steps=35, **kwargs):
            with NVTXContext(f"DIFFUSION_SAMPLING(steps={num_steps})", COLORS['diffusion_step']):
                return original_gen_samples(self, data_batch, num_steps=num_steps, **kwargs)
        Text2WorldModelRectifiedFlow.generate_samples_from_batch = patched_gen_samples
        print("  ✓ Patched generate_samples_from_batch")
        
    except ImportError as e:
        print(f"  ✗ Could not patch Text2WorldModelRectifiedFlow: {e}")
    
    # 3. Patch ControlVideo2WorldModelRectifiedFlow methods
    try:
        from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import ControlVideo2WorldModelRectifiedFlow
        
        # Patch get_velocity_fn_from_batch (where conditioning setup happens)
        original_get_vel = ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch
        @functools.wraps(original_get_vel)
        def patched_get_vel(self, data_batch, guidance=1.5, is_negative_prompt=False):
            with NVTXContext("VELOCITY_FN_SETUP", COLORS['velocity_fn_setup']):
                original_velocity_fn = original_get_vel(self, data_batch, guidance, is_negative_prompt)
            
            @functools.wraps(original_velocity_fn)
            def wrapped_velocity_fn(noise, noise_x, timestep):
                _state['diffusion_step'] += 1
                step = _state['diffusion_step']
                with NVTXContext(f"STEP_{step}", COLORS['diffusion_step']):
                    return original_velocity_fn(noise, noise_x, timestep)
            
            return wrapped_velocity_fn
        
        ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch = patched_get_vel
        print("  ✓ Patched get_velocity_fn_from_batch with VELOCITY_FN_SETUP marker")
        
        # Patch get_data_and_condition (control input encoding)
        original_get_data = ControlVideo2WorldModelRectifiedFlow.get_data_and_condition
        @functools.wraps(original_get_data)
        def patched_get_data(self, data_batch):
            with NVTXContext("GET_DATA_AND_CONDITION", COLORS['vae_encode_control']):
                return original_get_data(self, data_batch)
        ControlVideo2WorldModelRectifiedFlow.get_data_and_condition = patched_get_data
        print("  ✓ Patched get_data_and_condition")
        
        # Patch denoise with condition/uncondition tracking
        original_denoise = ControlVideo2WorldModelRectifiedFlow.denoise
        @functools.wraps(original_denoise)
        def patched_denoise(self, noise, xt, timesteps, condition):
            _state['denoise_call'] += 1
            is_cond = (_state['denoise_call'] % 2) == 1
            phase = "COND" if is_cond else "UNCOND"
            color = COLORS['denoise_cond'] if is_cond else COLORS['denoise_uncond']
            
            with NVTXContext(f"DENOISE_{phase}", color):
                return original_denoise(self, noise, xt, timesteps, condition)
        
        ControlVideo2WorldModelRectifiedFlow.denoise = patched_denoise
        print("  ✓ Patched denoise with COND/UNCOND tracking")
        
    except ImportError as e:
        print(f"  ✗ Could not patch ControlVideo2WorldModelRectifiedFlow: {e}")
    
    # 4. Patch conditioner for conditioning setup
    try:
        from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import ConditionerConfig
        
        # Try to patch get_condition_with_negative_prompt
        try:
            from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import get_condition_with_negative_prompt
            import cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner as cond_module
            
            original_get_cond = get_condition_with_negative_prompt
            @functools.wraps(original_get_cond)
            def patched_get_cond(*args, **kwargs):
                with NVTXContext("GET_CONDITIONING", COLORS['conditioning']):
                    return original_get_cond(*args, **kwargs)
            cond_module.get_condition_with_negative_prompt = patched_get_cond
            print("  ✓ Patched get_condition_with_negative_prompt")
        except (ImportError, AttributeError):
            pass
        
    except ImportError as e:
        print(f"  ✗ Could not patch conditioner: {e}")
    
    # 5. Patch VAE tokenizer encode/decode
    try:
        from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        
        original_encode = Wan2pt1VAEInterface.encode
        @functools.wraps(original_encode)
        def patched_encode(self, *args, **kwargs):
            with NVTXContext("VAE_ENCODE", COLORS['vae_encode_input']):
                return original_encode(self, *args, **kwargs)
        Wan2pt1VAEInterface.encode = patched_encode
        print("  ✓ Patched Wan2pt1VAEInterface.encode")
        
        original_decode = Wan2pt1VAEInterface.decode
        @functools.wraps(original_decode)
        def patched_decode(self, *args, **kwargs):
            with NVTXContext("VAE_DECODE", COLORS['vae_decode']):
                return original_decode(self, *args, **kwargs)
        Wan2pt1VAEInterface.decode = patched_decode
        print("  ✓ Patched Wan2pt1VAEInterface.decode")
        
    except ImportError as e:
        print(f"  ✗ Could not patch VAE tokenizer: {e}")
    
    # 6. Patch the DiT network forward
    try:
        from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import ControlLVGDiT
        
        original_net_forward = ControlLVGDiT.forward
        @functools.wraps(original_net_forward)
        def patched_net_forward(self, *args, **kwargs):
            with NVTXContext("DiT_FORWARD", COLORS['dit_forward']):
                return original_net_forward(self, *args, **kwargs)
        ControlLVGDiT.forward = patched_net_forward
        print("  ✓ Patched ControlLVGDiT.forward")
        
    except ImportError as e:
        print(f"  ✗ Could not patch DiT network: {e}")
    
    # 7. Patch model.decode (at the model level) - use Text2WorldModelRectifiedFlow
    try:
        # Text2WorldModelRectifiedFlow has the actual encode/decode methods
        if hasattr(Text2WorldModelRectifiedFlow, 'decode'):
            original_model_decode = Text2WorldModelRectifiedFlow.decode
            @functools.wraps(original_model_decode)
            def patched_model_decode(self, *args, **kwargs):
                with NVTXContext("MODEL_DECODE", COLORS['vae_decode']):
                    return original_model_decode(self, *args, **kwargs)
            Text2WorldModelRectifiedFlow.decode = patched_model_decode
            print("  ✓ Patched Text2WorldModelRectifiedFlow.decode")
            
    except Exception as e:
        print(f"  ✗ Could not patch model decode: {e}")
    
    # 8. Patch model.encode (for input video encoding)
    try:
        if hasattr(Text2WorldModelRectifiedFlow, 'encode'):
            original_model_encode = Text2WorldModelRectifiedFlow.encode
            @functools.wraps(original_model_encode)
            def patched_model_encode(self, *args, **kwargs):
                with NVTXContext("MODEL_ENCODE", COLORS['vae_encode_input']):
                    return original_model_encode(self, *args, **kwargs)
            Text2WorldModelRectifiedFlow.encode = patched_model_encode
            print("  ✓ Patched Text2WorldModelRectifiedFlow.encode")
            
    except Exception as e:
        print(f"  ✗ Could not patch model encode: {e}")
    
    # 9. Patch the inference pipeline methods
    try:
        from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
        
        # Patch generate_img2world for overall tracking
        if hasattr(ControlVideo2WorldInference, 'generate_img2world'):
            original_gen_img2world = ControlVideo2WorldInference.generate_img2world
            @functools.wraps(original_gen_img2world)
            def patched_gen_img2world(self, *args, **kwargs):
                with NVTXContext("GENERATE_IMG2WORLD", COLORS['sequential_inference']):
                    return original_gen_img2world(self, *args, **kwargs)
            ControlVideo2WorldInference.generate_img2world = patched_gen_img2world
            print("  ✓ Patched ControlVideo2WorldInference.generate_img2world")
        
        # Patch generate_img2world_batch for batch inference
        if hasattr(ControlVideo2WorldInference, 'generate_img2world_batch'):
            original_gen_img2world_batch = ControlVideo2WorldInference.generate_img2world_batch
            @functools.wraps(original_gen_img2world_batch)
            def patched_gen_img2world_batch(self, *args, **kwargs):
                with NVTXContext("GENERATE_IMG2WORLD_BATCH", COLORS['batch_inference']):
                    return original_gen_img2world_batch(self, *args, **kwargs)
            ControlVideo2WorldInference.generate_img2world_batch = patched_gen_img2world_batch
            print("  ✓ Patched ControlVideo2WorldInference.generate_img2world_batch")
        
        # Patch _get_data_batch_input for data preparation
        if hasattr(ControlVideo2WorldInference, '_get_data_batch_input'):
            original_get_batch = ControlVideo2WorldInference._get_data_batch_input
            @functools.wraps(original_get_batch)
            def patched_get_batch(self, *args, **kwargs):
                with NVTXContext("GET_DATA_BATCH_INPUT", COLORS['data_prep']):
                    return original_get_batch(self, *args, **kwargs)
            ControlVideo2WorldInference._get_data_batch_input = patched_get_batch
            print("  ✓ Patched ControlVideo2WorldInference._get_data_batch_input")
            
    except Exception as e:
        print(f"  ✗ Could not patch ControlVideo2WorldInference: {e}")
    
    # 9b. Patch Control2WorldInference._generate_batch (high-level batch handler)
    try:
        from cosmos_transfer2.inference import Control2WorldInference
        
        if hasattr(Control2WorldInference, '_generate_batch'):
            original_gen_batch_internal = Control2WorldInference._generate_batch
            @functools.wraps(original_gen_batch_internal)
            def patched_gen_batch_internal(self, batch_samples, output_dir, batch_offset=0):
                with NVTXContext("_GENERATE_BATCH", COLORS['batch_inference']):
                    return original_gen_batch_internal(self, batch_samples, output_dir, batch_offset)
            Control2WorldInference._generate_batch = patched_gen_batch_internal
            print("  ✓ Patched Control2WorldInference._generate_batch")
            
    except Exception as e:
        print(f"  ✗ Could not patch _generate_batch: {e}")
    
    # 10. Patch text encoder
    try:
        from cosmos_transfer2._src.transfer2.inference.utils import get_t5_from_prompt
        import cosmos_transfer2._src.transfer2.inference.utils as utils_module
        
        original_get_t5 = get_t5_from_prompt
        @functools.wraps(original_get_t5)
        def patched_get_t5(*args, **kwargs):
            with NVTXContext("TEXT_EMBEDDING_T5", COLORS['conditioning']):
                return original_get_t5(*args, **kwargs)
        utils_module.get_t5_from_prompt = patched_get_t5
        print("  ✓ Patched get_t5_from_prompt")
        
    except Exception as e:
        print(f"  ✗ Could not patch text encoder: {e}")
    
    # 11. Patch video reading
    try:
        from cosmos_transfer2._src.transfer2.inference.utils import read_and_process_video
        import cosmos_transfer2._src.transfer2.inference.utils as utils_module
        
        original_read_video = read_and_process_video
        @functools.wraps(original_read_video)
        def patched_read_video(*args, **kwargs):
            with NVTXContext("READ_INPUT_VIDEO", COLORS['data_prep']):
                return original_read_video(*args, **kwargs)
        utils_module.read_and_process_video = patched_read_video
        print("  ✓ Patched read_and_process_video")
        
    except Exception as e:
        print(f"  ✗ Could not patch video reading: {e}")
    
    # 12. Patch control input reading
    try:
        from cosmos_transfer2._src.transfer2.inference.utils import read_and_process_control_input
        import cosmos_transfer2._src.transfer2.inference.utils as utils_module
        
        original_read_control = read_and_process_control_input
        @functools.wraps(original_read_control)
        def patched_read_control(*args, **kwargs):
            with NVTXContext("READ_CONTROL_INPUT", COLORS['control_setup']):
                return original_read_control(*args, **kwargs)
        utils_module.read_and_process_control_input = patched_read_control
        print("  ✓ Patched read_and_process_control_input")
        
    except Exception as e:
        print(f"  ✗ Could not patch control input reading: {e}")
    
    # 13. Patch torch.stack and torch.cat for batching visibility (adds noise but shows data prep)
    # Skip this - too noisy
    
    # 14. Patch img_or_video saving for output visibility  
    try:
        from cosmos_transfer2._src.predict2.utils.io import save_img_or_video
        import cosmos_transfer2._src.predict2.utils.io as io_module
        
        original_save = save_img_or_video
        @functools.wraps(original_save)
        def patched_save(*args, **kwargs):
            with NVTXContext("SAVE_OUTPUT", 0x808080):  # Gray
                return original_save(*args, **kwargs)
        io_module.save_img_or_video = patched_save
        print("  ✓ Patched save_img_or_video")
        
    except Exception as e:
        print(f"  ✗ Could not patch save_img_or_video: {e}")
    
    print("\nNVTX profiling v3 patches applied!")
    print("\nExpected markers in profile:")
    print("  GREEN:       === BATCH INFERENCE === (Control2WorldInference.generate_batch)")
    print("  GREEN:       GENERATE_IMG2WORLD_BATCH, _GENERATE_BATCH")
    print("  RED:         === SEQUENTIAL INFERENCE === (Control2WorldInference.generate)")
    print("  RED:         GENERATE_IMG2WORLD")
    print("  GOLD:        DATA_PREP_NORMALIZE, DATA_PREP_AUGMENT, READ_INPUT_VIDEO")
    print("  PURPLE:      GET_CONDITIONING, TEXT_EMBEDDING_T5")
    print("  MED ORCHID:  VELOCITY_FN_SETUP (includes VAE encode + condition setup)")
    print("  DARK CYAN:   GET_DATA_AND_CONDITION (control input processing)")
    print("  ORANGE:      READ_CONTROL_INPUT")
    print("  CYAN:        VAE_ENCODE, MODEL_ENCODE")
    print("  YELLOW:      DIFFUSION_SAMPLING, STEP_N")
    print("  ORANGE:      DENOISE_COND")
    print("  MAGENTA:     DENOISE_UNCOND")
    print("  PINK:        VAE_DECODE, MODEL_DECODE")
    print("  LIGHT BLUE:  DiT_FORWARD")
    print("  GRAY:        SAVE_OUTPUT")


if __name__ == "__main__":
    apply_all_patches()

