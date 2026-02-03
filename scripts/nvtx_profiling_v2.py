#!/usr/bin/env python3
"""
Enhanced NVTX profiling for Cosmos Transfer2 inference.

Adds clear high-level markers for:
- Overall test phases (BATCH_INFERENCE, SEQUENTIAL_INFERENCE)
- Per-sample processing
- Diffusion steps with condition/uncondition
- VAE encode/decode
- Data loading

Usage:
    import nvtx_profiling_v2
    nvtx_profiling_v2.apply_all_patches()
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


# Color scheme for easy identification
COLORS = {
    'batch_inference': 0x00FF00,    # Green - batch mode
    'sequential_inference': 0xFF0000,  # Red - sequential mode
    'sample': 0x0000FF,             # Blue - individual sample
    'diffusion_step': 0xFFFF00,     # Yellow - diffusion steps
    'denoise_cond': 0xFFA500,       # Orange - conditioned
    'denoise_uncond': 0xFF00FF,     # Magenta - unconditioned
    'vae_encode': 0x00FFFF,         # Cyan - VAE encode
    'vae_decode': 0xFF69B4,         # Pink - VAE decode
    'data_load': 0x808080,          # Gray - data loading
    'text_embed': 0x9400D3,         # Purple - text embedding
}


class NVTXContext:
    """Context manager for NVTX ranges."""
    def __init__(self, message, color=None):
        self.message = message
        self.color = color
        self._range_id = None
        
    def __enter__(self):
        if HAS_NVTX:
            # nvtx package uses start_range/end_range, not push_range/pop_range
            self._range_id = nvtx.start_range(message=self.message, color=self.color)
        return self
    
    def __exit__(self, *args):
        if HAS_NVTX and self._range_id is not None:
            nvtx.end_range(self._range_id)


def nvtx_range(message, color=None):
    """Decorator to wrap a function in an NVTX range."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with NVTXContext(message, color):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global state for tracking
_state = {
    'diffusion_step': 0,
    'denoise_call': 0,
    'sample_index': 0,
    'mode': 'unknown',
}


def reset_state():
    """Reset tracking state."""
    _state['diffusion_step'] = 0
    _state['denoise_call'] = 0
    _state['sample_index'] = 0


def apply_all_patches():
    """Apply all NVTX patches for comprehensive profiling."""
    
    print("Applying NVTX profiling patches v2...")
    
    # 1. Patch the main inference entry points
    try:
        from cosmos_transfer2.inference import CosmosTransfer2Pipeline
        
        original_generate = CosmosTransfer2Pipeline.generate
        @functools.wraps(original_generate)
        def patched_generate(self, *args, **kwargs):
            _state['mode'] = 'sequential'
            reset_state()
            with NVTXContext("=== SEQUENTIAL_INFERENCE ===", COLORS['sequential_inference']):
                return original_generate(self, *args, **kwargs)
        CosmosTransfer2Pipeline.generate = patched_generate
        print("  ✓ Patched CosmosTransfer2Pipeline.generate")
        
        # Patch _generate_sample for per-sample tracking
        if hasattr(CosmosTransfer2Pipeline, '_generate_sample'):
            original_gen_sample = CosmosTransfer2Pipeline._generate_sample
            @functools.wraps(original_gen_sample)
            def patched_gen_sample(self, *args, **kwargs):
                _state['sample_index'] += 1
                reset_state()
                with NVTXContext(f"SAMPLE_{_state['sample_index']}", COLORS['sample']):
                    return original_gen_sample(self, *args, **kwargs)
            CosmosTransfer2Pipeline._generate_sample = patched_gen_sample
            print("  ✓ Patched CosmosTransfer2Pipeline._generate_sample")
        
        # Patch _generate_batch
        if hasattr(CosmosTransfer2Pipeline, '_generate_batch'):
            original_gen_batch = CosmosTransfer2Pipeline._generate_batch
            @functools.wraps(original_gen_batch)
            def patched_gen_batch(self, *args, **kwargs):
                _state['mode'] = 'batch'
                reset_state()
                with NVTXContext("=== BATCH_INFERENCE ===", COLORS['batch_inference']):
                    return original_gen_batch(self, *args, **kwargs)
            CosmosTransfer2Pipeline._generate_batch = patched_gen_batch
            print("  ✓ Patched CosmosTransfer2Pipeline._generate_batch")
            
    except ImportError as e:
        print(f"  ✗ Could not patch CosmosTransfer2Pipeline: {e}")
    
    # 2. Patch the inference pipeline methods
    try:
        from cosmos_transfer2._src.transfer2.inference.inference_pipeline import Img2WorldInferencePipeline
        
        # Patch generate_img2world (sequential)
        original_gen = Img2WorldInferencePipeline.generate_img2world
        @functools.wraps(original_gen)
        def patched_gen(self, *args, **kwargs):
            _state['sample_index'] += 1
            sample_name = kwargs.get('sample_name', f'sample_{_state["sample_index"]}')
            with NVTXContext(f"generate_img2world({sample_name})", COLORS['sequential_inference']):
                return original_gen(self, *args, **kwargs)
        Img2WorldInferencePipeline.generate_img2world = patched_gen
        print("  ✓ Patched Img2WorldInferencePipeline.generate_img2world")
        
        # Patch generate_img2world_batch
        if hasattr(Img2WorldInferencePipeline, 'generate_img2world_batch'):
            original_batch = Img2WorldInferencePipeline.generate_img2world_batch
            @functools.wraps(original_batch)
            def patched_batch(self, *args, **kwargs):
                batch_size = kwargs.get('batch_size', len(kwargs.get('prompts', [])))
                with NVTXContext(f"generate_img2world_batch(n={batch_size})", COLORS['batch_inference']):
                    return original_batch(self, *args, **kwargs)
            Img2WorldInferencePipeline.generate_img2world_batch = patched_batch
            print("  ✓ Patched Img2WorldInferencePipeline.generate_img2world_batch")
            
    except ImportError as e:
        print(f"  ✗ Could not patch Img2WorldInferencePipeline: {e}")
    
    # 3. Patch generate_samples_from_batch (diffusion loop)
    try:
        from cosmos_transfer2._src.predict2.models.text2world_model_rectified_flow import Text2WorldModelRectifiedFlow
        
        original_gen_samples = Text2WorldModelRectifiedFlow.generate_samples_from_batch
        @functools.wraps(original_gen_samples)
        def patched_gen_samples(self, data_batch, num_steps=35, **kwargs):
            with NVTXContext(f"DIFFUSION_LOOP(steps={num_steps})", COLORS['diffusion_step']):
                return original_gen_samples(self, data_batch, num_steps=num_steps, **kwargs)
        Text2WorldModelRectifiedFlow.generate_samples_from_batch = patched_gen_samples
        print("  ✓ Patched Text2WorldModelRectifiedFlow.generate_samples_from_batch")
        
    except ImportError as e:
        print(f"  ✗ Could not patch generate_samples_from_batch: {e}")
    
    # 4. Patch velocity_fn with step tracking
    try:
        from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import ControlVideo2WorldModelRectifiedFlow
        
        original_get_vel = ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch
        @functools.wraps(original_get_vel)
        def patched_get_vel(self, data_batch, guidance=1.5, is_negative_prompt=False):
            original_velocity_fn = original_get_vel(self, data_batch, guidance, is_negative_prompt)
            
            @functools.wraps(original_velocity_fn)
            def wrapped_velocity_fn(noise, noise_x, timestep):
                _state['diffusion_step'] += 1
                step = _state['diffusion_step']
                with NVTXContext(f"STEP_{step}", COLORS['diffusion_step']):
                    return original_velocity_fn(noise, noise_x, timestep)
            
            return wrapped_velocity_fn
        
        ControlVideo2WorldModelRectifiedFlow.get_velocity_fn_from_batch = patched_get_vel
        print("  ✓ Patched velocity_fn with step tracking")
        
        # 5. Patch denoise with condition/uncondition tracking
        original_denoise = ControlVideo2WorldModelRectifiedFlow.denoise
        @functools.wraps(original_denoise)
        def patched_denoise(self, noise, xt, timesteps, condition):
            _state['denoise_call'] += 1
            # Odd calls are conditioned, even are unconditioned (due to CFG)
            is_cond = (_state['denoise_call'] % 2) == 1
            phase = "COND" if is_cond else "UNCOND"
            color = COLORS['denoise_cond'] if is_cond else COLORS['denoise_uncond']
            
            with NVTXContext(f"DENOISE_{phase}", color):
                return original_denoise(self, noise, xt, timesteps, condition)
        
        ControlVideo2WorldModelRectifiedFlow.denoise = patched_denoise
        print("  ✓ Patched denoise with COND/UNCOND tracking")
        
    except ImportError as e:
        print(f"  ✗ Could not patch velocity/denoise: {e}")
    
    # 6. Patch VAE encode/decode
    try:
        from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import CausalVAETokenizer
        
        original_encode = CausalVAETokenizer.encode
        @functools.wraps(original_encode)
        def patched_encode(self, *args, **kwargs):
            with NVTXContext("VAE_ENCODE", COLORS['vae_encode']):
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
    
    # 7. Patch video loading
    try:
        from cosmos_transfer2._src.transfer2.inference.utils import read_video_or_image_into_frames_BCTHW
        import cosmos_transfer2._src.transfer2.inference.utils as utils_module
        
        original_read = read_video_or_image_into_frames_BCTHW
        @functools.wraps(original_read)
        def patched_read(*args, **kwargs):
            with NVTXContext("LOAD_VIDEO", COLORS['data_load']):
                return original_read(*args, **kwargs)
        utils_module.read_video_or_image_into_frames_BCTHW = patched_read
        print("  ✓ Patched read_video_or_image_into_frames_BCTHW")
        
    except ImportError as e:
        print(f"  ✗ Could not patch video loading: {e}")
    
    # 8. Patch text embedding
    try:
        from cosmos_transfer2._src.predict2.conditioner import GeneralConditioner
        
        if hasattr(GeneralConditioner, 'forward'):
            original_forward = GeneralConditioner.forward
            @functools.wraps(original_forward)
            def patched_forward(self, *args, **kwargs):
                with NVTXContext("TEXT_CONDITIONING", COLORS['text_embed']):
                    return original_forward(self, *args, **kwargs)
            GeneralConditioner.forward = patched_forward
            print("  ✓ Patched GeneralConditioner.forward")
            
    except ImportError as e:
        print(f"  ✗ Could not patch conditioner: {e}")
    
    # 9. Patch the DiT network forward (where attention happens)
    try:
        from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import ControlLVGDiT
        
        original_net_forward = ControlLVGDiT.forward
        @functools.wraps(original_net_forward)
        def patched_net_forward(self, *args, **kwargs):
            with NVTXContext("DiT_FORWARD", 0xADD8E6):  # Light blue
                return original_net_forward(self, *args, **kwargs)
        ControlLVGDiT.forward = patched_net_forward
        print("  ✓ Patched ControlLVGDiT.forward (DiT network)")
        
    except ImportError as e:
        print(f"  ✗ Could not patch DiT network: {e}")
    
    # 10. Patch attention if accessible
    try:
        from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import Attention
        
        original_attn = Attention.forward
        @functools.wraps(original_attn)
        def patched_attn(self, *args, **kwargs):
            with NVTXContext("ATTENTION", 0x32CD32):  # Lime green
                return original_attn(self, *args, **kwargs)
        Attention.forward = patched_attn
        print("  ✓ Patched Attention.forward")
        
    except ImportError as e:
        print(f"  ✗ Could not patch Attention: {e}")
    
    print("\nNVTX profiling v2 patches applied!")
    print("\nExpected markers in profile:")
    print("  GREEN:   === BATCH_INFERENCE ===")
    print("  RED:     === SEQUENTIAL_INFERENCE ===")
    print("  BLUE:    SAMPLE_N")
    print("  YELLOW:  STEP_N, DIFFUSION_LOOP")
    print("  ORANGE:  DENOISE_COND")
    print("  MAGENTA: DENOISE_UNCOND")
    print("  CYAN:    VAE_ENCODE")
    print("  PINK:    VAE_DECODE")
    print("  GRAY:    LOAD_VIDEO")
    print("  PURPLE:  TEXT_CONDITIONING")


if __name__ == "__main__":
    apply_all_patches()

