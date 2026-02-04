#!/usr/bin/bash
# Test FP8 quantization with 5-frame input (fast validation)
#
# This script tests FP8 vs BF16 inference:
# - Validates FP8 is working on Blackwell GPU
# - Measures speedup from FP8 Tensor Cores
# - Compares output quality (PSNR)
#
# Usage:
#   cd /home/horde/cosmos/cosmos-transfer2.5
#   bash scripts/test_fp8_5frame.sh
#
# Prerequisites:
#   - Blackwell GB202 GPU with FP8 support
#   - Transformer Engine 2.2+
#   - Input depth video (5 frames)
#   - ~10-15 minutes for full test

set -e

# Configuration
COSMOS_DIR="${COSMOS_DIR:-/home/horde/cosmos/cosmos-transfer2.5}"
DEPTH_VIDEO="${DEPTH_VIDEO:-/home/horde/cosmos/cosmos-transfer2.5/results/short_temporal_tests/5frame_output/batch_sample_1_control_depth.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/fp8_tests/5frame}"
STATE_T="${STATE_T:-2}" # 2 for 5 frames
NUM_STEPS="${NUM_STEPS:-4}" # 4 for fast
SEED="${SEED:-42}"
PROMPT="${PROMPT:-A high quality video of a robot in a warehouse}"
NUM_RUNS="${NUM_RUNS:-3}" # Best of 3

echo "========================================================================"
echo "FP8 Quantization Test: 5-Frame Input"
echo "========================================================================"
echo "Configuration:"
echo "  Depth video: $DEPTH_VIDEO"
echo "  State-t: $STATE_T ($(($(($STATE_T - 1)) * 4 + 1)) pixel frames)"
echo "  Num steps: $NUM_STEPS"
echo "  Num runs: $NUM_RUNS"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check prerequisites
if [ ! -f "$DEPTH_VIDEO" ]; then
    echo "ERROR: Depth video not found: $DEPTH_VIDEO"
    exit 1
fi

# Setup environment
cd "$COSMOS_DIR"
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running FP8 test (BF16 vs FP8 comparison)..."
echo ""

python tests/test_fp8_optimization.py \
    --depth-video "$DEPTH_VIDEO" \
    --output-dir "$OUTPUT_DIR" \
    --state-t $STATE_T \
    --num-steps $NUM_STEPS \
    --seed $SEED \
    --num-runs $NUM_RUNS \
    --prompt "$PROMPT"

echo ""
echo "========================================================================"
echo "Test Complete"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - $OUTPUT_DIR/bf16_run0/bf16_run0.mp4 (BF16 baseline)"
echo "  - $OUTPUT_DIR/fp8_run0/fp8_run0.mp4 (FP8 optimized)"
echo "  - $OUTPUT_DIR/comparison_bf16_vs_fp8.mp4 (side-by-side)"
echo ""
echo "Expected Results:"
echo "  - FP8 Speedup: 1.6-1.8x faster"
echo "  - Quality: >40 dB PSNR (excellent)"
echo "  - Combined with CFG: ~2.4-2.7x total speedup vs original"
echo ""

