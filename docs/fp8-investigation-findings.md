# FP8 Investigation Findings

## Current Status: **FP8 Not Yet Working**

### Test Results (5-frame, 4 diffusion steps)
- **BF16 baseline**: 3.80s
- **FP8 "optimized"**: 3.97s (**4% slower!**)
- **PSNR**: Infinite (byte-for-byte identical)
- **Conclusion**: FP8 is not actually being used

### Evidence FP8 Isn't Active
1. ❌ No speedup (actually slower due to context overhead)
2. ❌ Outputs are identical (not expected with quantization)
3. ❌ No "FP8 inference enabled" log messages
4. ❌ No FP8 recipe/scaling logs from Transformer Engine

## Root Cause Analysis

### What We Implemented
```python
# In inference_pipeline.py
with fp8_utils.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
    sample = self.model.generate_samples_from_batch(...)
```

### Why It's Not Working

**Hypothesis 1: Runtime vs Init-time Configuration**
- Transformer Engine's `DotProductAttention` may need FP8 configured **at initialization**
- The `fp8_autocast` context might not affect pre-initialized layers
- Current code: FP8 context at inference time
- May need: FP8 enabled when model is loaded

**Hypothesis 2: Missing FP8 Module Wrapping**
- Transformer Engine may require explicit module wrapping
- Need to wrap attention layers with `te.fp8_autocast` at module level
- Or use `te.Linear` with FP8 enabled instead of `nn.Linear`

**Hypothesis 3: FP8 Recipe Not Propagating**
- The FP8 recipe might not be reaching the attention layers
- May need to set FP8 globally before model initialization
- Or pass FP8 config through model's `exp_override_opts`

## Hardware Confirmation
✅ **FP8 is available:**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Compute Capability: 12.0 (Blackwell)
- Transformer Engine: 2.2+cu128
- PyTorch FP8 dtypes: Available

## Next Steps to Fix

### Option 1: Enable FP8 at Model Load (Most Likely Fix)
```python
# In inference.py, before model loading:
if self.use_fp8:
    import transformer_engine.pytorch as te
    # Set FP8 globally before model init
    te.fp8.FP8GlobalStateManager.set_fp8_enabled(True)
    # Or add to exp_override_opts
    exp_override_opts.append("model.config.use_fp8=True")
```

### Option 2: Wrap Attention Modules with FP8
```python
# After model loading, wrap attention modules:
for name, module in model.named_modules():
    if isinstance(module, DotProductAttention):
        # Wrap with FP8-enabled version
        module.fp8_enabled = True
```

### Option 3: Use Transformer Engine's fp8_model_init
```python
# Before model loading:
with te.fp8_model_init(enabled=True):
    model = load_model(...)
```

### Option 4: Check Cosmos's FP8 Integration
- Cosmos imports Transformer Engine already
- Check if there's an existing FP8 config parameter
- Look for `use_fp8` or `fp8_enabled` in model configs
- May already have infrastructure we're not using

## Investigation Steps

### Step 1: Add Debug Logging ✅
**Status**: Added debug prints to `fp8_utils.py`

**Next**: Re-run test to see if context is entered

### Step 2: Check TE Documentation
- Review Transformer Engine's attention API
- Find correct way to enable FP8 for pre-initialized layers
- Check if `fp8_autocast` works with `DotProductAttention`

### Step 3: Search Cosmos Codebase
```bash
# Search for existing FP8 infrastructure
grep -r "fp8.*enable\|use_fp8\|fp8.*config" cosmos_transfer2/_src/
```

### Step 4: Test with Explicit FP8 Enable
- Try setting `te.fp8.FP8GlobalStateManager` before model load
- Or modify attention init to accept FP8 parameter

### Step 5: Minimal Reproduction
Create minimal test:
```python
import transformer_engine.pytorch as te

# Create attention layer
attn = te.attention.DotProductAttention(...)

# Try FP8 context
with te.fp8_autocast(enabled=True):
    output = attn(q, k, v)
    
# Check if FP8 was used (should see scaling factors)
```

## Expected Results (Once Fixed)
- **5-frame**: 1.6-1.8x speedup
- **93-frame**: 1.8-2.0x speedup
- **PSNR**: 40-50 dB (slight differences due to quantization)
- **Memory**: ~50% reduction for model weights

## Alternate Approaches (If TE fp8_autocast Doesn't Work)

### A. TensorRT Export
- Export model to TensorRT
- Let TensorRT handle FP8 optimization
- Pros: Production-ready, automatic optimization
- Cons: One-time export cost, less flexible

### B. Manual FP8 Quantization (torchao/quanto)
- Use `torchao` or `quanto` for explicit quantization
- More control over which layers to quantize
- Pros: Clear what's quantized, easier debugging
- Cons: More manual work

### C. Check for Existing Cosmos FP8 Support
- Cosmos may already have FP8 infrastructure
- Look in model configs for FP8 flags
- Check experiment configs for FP8 options

## References
- [Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [TE FP8 API Docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)
- [Blackwell FP8 Specs](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

## Current Branch Status
- ✅ FP8 utilities implemented
- ✅ FP8 context added to inference pipeline
- ✅ Test script created
- ❌ FP8 not actually activating (needs investigation)
- ⏸️  Paused for debugging

