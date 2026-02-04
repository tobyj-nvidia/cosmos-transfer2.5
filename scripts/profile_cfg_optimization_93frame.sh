#!/bin/bash
#
# Profile CFG optimization with 93-frame (default) input and 35 diffusion steps
#

set -e

# Configuration for 93-frame test
DEPTH_VIDEO="/home/horde/isaaclab-cosmos-experiments/results/batch_test_20260130_180547/captures/video_1/depth_video.mp4"
OUTPUT_DIR="/home/horde/cosmos/cosmos-transfer2.5/results/cfg_opt_profiles_93frame"
STATE_T=24  # (24-1)*4+1 = 93 pixel frames
SEED=42
NUM_STEPS=35  # Default high-quality
PROMPT="A robotic arm manipulating objects on a table in a warehouse"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Nsight Systems Profiling: CFG Optimization (93-frame)"
echo "========================================================================"
echo "Configuration:"
echo "  Depth video: $DEPTH_VIDEO"
echo "  State-t: $STATE_T (93 pixel frames)"
echo "  Num steps: $NUM_STEPS (high quality)"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Capturing both sequential and batched CFG in one profile..."
echo ""

nsys profile \
    --output="$OUTPUT_DIR/cfg_optimization_93frame.nsys-rep" \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --stats=true \
    /home/horde/cosmos/cosmos-transfer2.5/.venv/bin/python \
    tests/test_cfg_optimization_nvtx_v3.py \
    --depth-video "$DEPTH_VIDEO" \
    --output-dir "$OUTPUT_DIR/comparison_run" \
    --state-t $STATE_T \
    --seed $SEED \
    --num-steps $NUM_STEPS \
    --prompt "$PROMPT" \
    --profile

echo ""
echo "========================================================================"
echo "Profiling Complete"
echo "========================================================================"
echo "Profile saved to: $OUTPUT_DIR/cfg_optimization_93frame.nsys-rep"
echo ""
echo "The profile contains:"
echo "  - WARMUP (separate phase)"
echo "  - SEQUENTIAL_CFG_TEST (original implementation, 93 frames, 35 steps)"
echo "  - BATCHED_CFG_TEST (optimized implementation, 93 frames, 35 steps)"
echo ""
echo "To analyze:"
echo "  1. Download profile:"
echo "     scp horde:$OUTPUT_DIR/cfg_optimization_93frame.nsys-rep /home/tobyj/Downloads/"
echo ""
echo "  2. Open in Nsight Systems UI"
echo ""
echo "  3. Compare side-by-side with 5-frame results to see if speedup improves"
echo ""

