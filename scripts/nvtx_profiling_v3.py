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

try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    print("WARNING: nvtx not available. Install with: pip install nvtx")
    
    class _FakeNvtx:
        @staticmethod
        def annotate(message="", color=None):
            def decorator(func):
                return func
            return decorator
        
        @staticmethod
        def start_range(message="", color=None):
            return None
        
        @staticmethod
        def end_range(range_id):
            pass
    
    nvtx = _FakeNvtx()


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
    """Context manager for NVTX ranges using nvtx.annotate()."""
    def __init__(self, message, color=None):
        self.message = message
        self.color = color
        self._ctx = None
        
    def __enter__(self):
        if HAS_NVTX:
            # Use nvtx.annotate() which is the recommended context manager API
            self._ctx = nvtx.annotate(message=self.message, color=self.color)
            self._ctx.__enter__()
        return self
    
    def __exit__(self, *args):
        if HAS_NVTX and self._ctx is not None:
            self._ctx.__exit__(*args)


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
        from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import CausalVAETokenizer
        
        original_encode = CausalVAETokenizer.encode
        @functools.wraps(original_encode)
        def patched_encode(self, *args, **kwargs):
            with NVTXContext("VAE_ENCODE", COLORS['vae_encode_input']):
                return original_encode(self, *args, **kwargs)
        CausalVAETokenizer.encode = patched_encode
        print("  ✓ Patched CausalVAETokenizer.encode")
        
        original_decode = CausalVAETokenizer.decode
        @functools.wraps(original_decode)
        def patched_decode(self, *args, **kwargs):
            with NVTXContext("VAE_DECODE", COLORS['vae_decode']):
                return original_decode(self, *args, **kwargs)
        CausalVAETokenizer.decode = patched_decode
        print("  ✓ Patched CausalVAETokenizer.decode")
        
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
    
    # 7. Patch model.decode (at the model level)
    try:
        from cosmos_transfer2._src.predict2.models.text2world_model import Text2WorldModel
        
        if hasattr(Text2WorldModel, 'decode'):
            original_model_decode = Text2WorldModel.decode
            @functools.wraps(original_model_decode)
            def patched_model_decode(self, *args, **kwargs):
                with NVTXContext("MODEL_DECODE", COLORS['vae_decode']):
                    return original_model_decode(self, *args, **kwargs)
            Text2WorldModel.decode = patched_model_decode
            print("  ✓ Patched Text2WorldModel.decode")
            
    except ImportError as e:
        print(f"  ✗ Could not patch model decode: {e}")
    
    # 8. Patch model.encode (for input video encoding)
    try:
        from cosmos_transfer2._src.predict2.models.text2world_model import Text2WorldModel
        
        if hasattr(Text2WorldModel, 'encode'):
            original_model_encode = Text2WorldModel.encode
            @functools.wraps(original_model_encode)
            def patched_model_encode(self, *args, **kwargs):
                with NVTXContext("MODEL_ENCODE", COLORS['vae_encode_input']):
                    return original_model_encode(self, *args, **kwargs)
            Text2WorldModel.encode = patched_model_encode
            print("  ✓ Patched Text2WorldModel.encode")
            
    except ImportError as e:
        print(f"  ✗ Could not patch model encode: {e}")
    
    print("\nNVTX profiling v3 patches applied!")
    print("\nExpected markers in profile:")
    print("  GREEN:       === BATCH INFERENCE === (Control2WorldInference.generate_batch)")
    print("  RED:         === SEQUENTIAL INFERENCE === (Control2WorldInference.generate)")
    print("  GOLD:        DATA_PREP_NORMALIZE, DATA_PREP_AUGMENT")
    print("  PURPLE:      GET_CONDITIONING")
    print("  MED ORCHID:  VELOCITY_FN_SETUP (includes VAE encode + condition setup)")
    print("  DARK CYAN:   GET_DATA_AND_CONDITION (control input processing)")
    print("  CYAN:        VAE_ENCODE, MODEL_ENCODE")
    print("  YELLOW:      DIFFUSION_SAMPLING, STEP_N")
    print("  ORANGE:      DENOISE_COND")
    print("  MAGENTA:     DENOISE_UNCOND")
    print("  PINK:        VAE_DECODE, MODEL_DECODE")
    print("  LIGHT BLUE:  DiT_FORWARD")


if __name__ == "__main__":
    apply_all_patches()

