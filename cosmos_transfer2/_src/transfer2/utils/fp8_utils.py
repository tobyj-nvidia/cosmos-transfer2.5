"""
FP8 utilities for Cosmos Transfer 2.5 inference.

Enables FP8 precision for Transformer Engine layers on Blackwell GPUs.
"""

import torch
from contextlib import contextmanager
from typing import Optional

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
except ImportError:
    HAS_TE = False


def create_fp8_recipe(
    margin: int = 0,
    interval: int = 1,
    fp8_format: str = "HYBRID",
    amax_history_len: int = 1024,
    amax_compute_algo: str = "max",
) -> Optional:
    """
    Create FP8 recipe for inference.
    
    Args:
        margin: Safety margin for scaling (0 for inference)
        interval: How often to update scaling factors (1 for inference)
        fp8_format: "HYBRID" uses E4M3 for forward, E5M2 for backward
        amax_history_len: History length for max absolute value tracking
        amax_compute_algo: Algorithm for computing amax ("max" or "most_recent")
    
    Returns:
        DelayedScaling recipe if Transformer Engine available, else None
    """
    if not HAS_TE:
        return None
    
    return DelayedScaling(
        margin=margin,
        interval=interval,
        fp8_format=getattr(Format, fp8_format),
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
    )


@contextmanager
def fp8_autocast(
    enabled: bool = True,
    fp8_recipe: Optional = None,
    fp8_group: Optional = None,
):
    """
    Context manager for FP8 inference.
    
    Usage:
        with fp8_autocast(enabled=True):
            output = model(input)
    
    Args:
        enabled: Whether to use FP8 (falls back to BF16 if False)
        fp8_recipe: FP8 recipe (created automatically if None)
        fp8_group: Process group for distributed inference (None for single GPU)
    
    Yields:
        Context with FP8 enabled
    """
    if not enabled or not HAS_TE:
        # No-op context if FP8 disabled or TE not available
        yield
        return
    
    # Create default recipe if not provided
    if fp8_recipe is None:
        fp8_recipe = create_fp8_recipe()
    
    # Use Transformer Engine's FP8 autocast
    with te.fp8_autocast(
        enabled=True,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
    ):
        yield


def is_fp8_available() -> bool:
    """
    Check if FP8 is available on current GPU.
    
    Returns:
        True if:
        - Transformer Engine is installed
        - GPU has compute capability >= 8.9 (Hopper/Blackwell)
        - PyTorch has FP8 dtypes
    """
    if not HAS_TE:
        return False
    
    if not torch.cuda.is_available():
        return False
    
    # Check compute capability (Hopper 8.9+, Blackwell 9.0+, or newer 12.0+)
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        return False
    if major == 8 and minor < 9:
        return False
    
    # Check PyTorch FP8 support
    if not (hasattr(torch, 'float8_e4m3fn') and hasattr(torch, 'float8_e5m2')):
        return False
    
    return True


def get_fp8_stats(model: torch.nn.Module) -> dict:
    """
    Get FP8 statistics from a model using Transformer Engine layers.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with FP8 statistics (scaling factors, amax values, etc.)
    """
    stats = {
        'num_fp8_modules': 0,
        'modules': [],
    }
    
    if not HAS_TE:
        return stats
    
    for name, module in model.named_modules():
        if hasattr(module, 'fp8_meta'):
            stats['num_fp8_modules'] += 1
            module_stats = {
                'name': name,
                'type': type(module).__name__,
            }
            
            # Get FP8 metadata if available
            if hasattr(module.fp8_meta, 'amax_history'):
                module_stats['amax_history_len'] = len(module.fp8_meta['amax_history'])
            
            if hasattr(module.fp8_meta, 'scale'):
                module_stats['scale'] = module.fp8_meta['scale'].item()
            
            if hasattr(module.fp8_meta, 'scale_inv'):
                module_stats['scale_inv'] = module.fp8_meta['scale_inv'].item()
            
            stats['modules'].append(module_stats)
    
    return stats


def warmup_fp8(model: torch.nn.Module, dummy_input: torch.Tensor, num_steps: int = 10):
    """
    Warmup FP8 scaling factors with dummy forward passes.
    
    FP8 requires a few forward passes to calibrate scaling factors before
    producing accurate results.
    
    Args:
        model: PyTorch model
        dummy_input: Dummy input tensor for warmup
        num_steps: Number of warmup steps (default: 10)
    """
    if not HAS_TE:
        return
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            _ = model(dummy_input)


# Monkey-patch to enable FP8 in Cosmos attention
def enable_fp8_in_attention():
    """
    Enable FP8 support in Cosmos attention layers.
    
    This patches the attention backend selection to enable FP8 when available.
    Call this once at startup before model loading.
    """
    try:
        from cosmos_transfer2._src.imaginaire.attention.cudnn import cudnn_forward
        
        # Store original function
        original_get_cudnn_dtype_map = cudnn_forward._get_cudnn_dtype_map
        
        def patched_get_cudnn_dtype_map(arch_tag):
            """Modified version that enables FP8 for Blackwell."""
            result = original_get_cudnn_dtype_map(arch_tag)
            
            # Enable FP8 for Blackwell (12.0) and Hopper (8.9+)
            if arch_tag >= 89 or arch_tag == 120:  # Hopper 8.9+ or Blackwell 12.0
                import cudnn
                result.update({
                    torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
                    torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
                })
            
            return result
        
        # Apply patch
        cudnn_forward._get_cudnn_dtype_map = patched_get_cudnn_dtype_map
        print("✅ FP8 support enabled in Cosmos attention")
        
    except Exception as e:
        print(f"⚠️  Could not enable FP8 in attention: {e}")


if __name__ == "__main__":
    # Test script
    print("FP8 Availability Check")
    print("=" * 50)
    print(f"Transformer Engine available: {HAS_TE}")
    print(f"FP8 available: {is_fp8_available()}")
    
    if is_fp8_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
        
        # Create test recipe
        recipe = create_fp8_recipe()
        print(f"FP8 Recipe created: {recipe is not None}")
        
        # Test context manager
        with fp8_autocast(enabled=True):
            print("✅ FP8 autocast context working")

