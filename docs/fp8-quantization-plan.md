# FP8 Quantization with NVIDIA Transformer Engine

## Goal
Accelerate Cosmos Transfer 2.5 inference by leveraging FP8 Tensor Cores on Blackwell GB202 GPU.

## Hardware Capabilities
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Compute Capability**: 12.0
- **Tensor Cores**: 5th generation with native FP8 support
- **Memory**: 96GB GDDR7

## Current State
- **Model dtype**: bfloat16
- **Flash Attention**: 2.7.3 (optimal for bf16)
- **CFG Optimization**: 49.2% speedup already achieved

## Expected Gains from FP8
- **Throughput**: ~2x faster matrix operations on Tensor Cores
- **Memory**: 50% reduction (8-bit vs 16-bit)
- **Quality**: <1% degradation with proper calibration
- **Combined with CFG**: ~3x total speedup over baseline

## NVIDIA Transformer Engine Approach

### Why Transformer Engine?
1. **Drop-in replacement** for attention layers
2. **Automatic mixed precision** (FP8/BF16 switching)
3. **Production-ready** (used in NVIDIA inference servers)
4. **Optimized kernels** for Blackwell architecture
5. **Handles calibration** automatically during warmup

### Architecture Integration Points

The Cosmos DiT uses these key components (all in bfloat16):
1. **Self-Attention blocks**: `Attention` class in `minimal_v4_dit.py`
2. **Cross-Attention blocks**: Same `Attention` class
3. **MLP layers**: `FeedForward` class
4. **Control branch**: Similar attention in `minimal_v4_lvg_dit_control_vace.py`

### Transformer Engine Replacements

```python
# Current: PyTorch native attention (bfloat16)
class Attention(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

# Replace with: Transformer Engine (FP8/BF16 mixed)
import transformer_engine.pytorch as te

class Attention(te.TransformerLayer):
    # Automatic FP8/BF16 precision management
    # FP8 for matrix multiplies, BF16 for accumulation
```

## Implementation Plan

### Phase 1: Setup and Baseline (Today)
1. ✅ Create `feature/fp8-quantization` branch
2. ✅ Detect GPU capabilities
3. Install Transformer Engine
4. Create baseline benchmarks (bfloat16 with CFG optimization)

### Phase 2: Minimal Integration (Day 1-2)
1. Replace self-attention layers with `te.DotProductAttention`
2. Replace MLPs with `te.LayerNormMLP`
3. Add FP8 recipe configuration
4. Test with 5-frame input (fast validation)

### Phase 3: Validation (Day 2-3)
1. Compare output quality (pixel-by-pixel vs baseline)
2. Measure PSNR/SSIM metrics
3. Visual inspection of generated videos
4. Verify no quality degradation

### Phase 4: Performance Testing (Day 3-4)
1. Benchmark 5-frame inference (with CFG + FP8)
2. Benchmark 93-frame inference (production workload)
3. Profile with Nsight to verify FP8 kernel usage
4. Measure memory usage reduction

### Phase 5: Documentation (Day 4-5)
1. Document speedup results
2. Compare CFG-only vs CFG+FP8
3. Provide recommendations for deployment
4. Update README with FP8 instructions

## Installation

```bash
# On horde machine
cd /home/horde/cosmos/cosmos-transfer2.5
source .venv/bin/activate

# Install Transformer Engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Or use pre-built wheel if available
pip install transformer-engine[pytorch]
```

## Key Configuration Parameters

```python
# FP8 Recipe for inference
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_recipe = DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=Format.HYBRID,  # E4M3 for fwd, E5M2 for bwd (inference uses fwd only)
    amax_history_len=1024,
    amax_compute_algo="max",
)
```

## Testing Strategy

### Correctness Test
```python
# Run same input through both models
output_bf16 = model_bf16(input)  # Current baseline
output_fp8 = model_fp8(input)    # FP8 version

# Compare
diff = torch.abs(output_bf16 - output_fp8).max()
psnr = compute_psnr(output_bf16, output_fp8)

assert diff < 0.01  # Should be very close
assert psnr > 40    # High quality threshold
```

### Performance Test
```python
# Warmup for calibration (required for FP8)
for _ in range(10):
    _ = model_fp8(warmup_input)

# Timed inference
with torch.cuda.nvtx.range("FP8_INFERENCE"):
    start = time.time()
    output = model_fp8(input)
    torch.cuda.synchronize()
    fp8_time = time.time() - start

speedup = bf16_time / fp8_time
print(f"FP8 speedup: {speedup:.2f}x")
```

## Expected Results

### Conservative Estimates
- **5-frame test**: 1.6-1.8x faster (smaller batches, overhead matters more)
- **93-frame test**: 1.8-2.0x faster (larger batches, better Tensor Core utilization)

### Combined with CFG (49.2% speedup)
- **5-frame total**: ~2.4x faster than original
- **93-frame total**: ~3.0x faster than original

### Memory Savings
- **Current**: ~60GB VRAM for 93 frames
- **With FP8**: ~35-40GB VRAM (50% weight reduction, activations still bf16)

## Risks and Mitigation

### Risk 1: Quality Degradation
- **Mitigation**: Extensive visual comparison, PSNR/SSIM metrics
- **Fallback**: Keep bf16 as default, FP8 as optional flag

### Risk 2: Compatibility Issues
- **Mitigation**: Start with minimal changes, test incrementally
- **Fallback**: Use quantization-aware training if post-training quantization fails

### Risk 3: Installation Problems
- **Mitigation**: Use pre-built wheels, Docker container
- **Fallback**: Try torchao or quanto if TE doesn't install cleanly

## Success Criteria

1. ✅ **Installation**: Transformer Engine installs without errors
2. ✅ **Integration**: Model runs with FP8 layers
3. ✅ **Quality**: Output visually identical to bf16 (PSNR > 40dB)
4. ✅ **Performance**: >1.5x speedup on 93-frame test
5. ✅ **Memory**: <50GB VRAM for 93-frame inference

## References

- [Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [FP8 Training Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [Blackwell FP8 Specs](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

