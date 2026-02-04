#!/bin/bash
#
# Profile CFG optimization (sequential vs batched) with Nsight Systems
#
# This script captures both sequential and batched CFG in a single profile
# with clear NVTX markers to distinguish them in the timeline.
#

set -e

# Configuration
DEPTH_VIDEO="/home/horde/cosmos/cosmos-transfer2.5/results/short_temporal_tests/5frame_output/batch_sample_1_control_depth.mp4"
OUTPUT_DIR="/home/horde/cosmos/cosmos-transfer2.5/results/cfg_opt_profiles"
STATE_T=2
SEED=42
NUM_STEPS=4
PROMPT="A high quality video"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Nsight Systems Profiling: CFG Optimization Comparison"
echo "========================================================================"
echo "Configuration:"
echo "  Depth video: $DEPTH_VIDEO"
echo "  State-t: $STATE_T (5 pixel frames)"
echo "  Num steps: $NUM_STEPS"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Capturing both sequential and batched CFG in one profile..."
echo ""

nsys profile \
    --output="$OUTPUT_DIR/cfg_optimization_comparison.nsys-rep" \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --stats=true \
    /home/horde/cosmos/cosmos-transfer2.5/.venv/bin/python \
    tests/test_cfg_optimization_local.py \
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
echo "Profile saved to: $OUTPUT_DIR/cfg_optimization_comparison.nsys-rep"
echo ""
echo "The profile contains:"
echo "  - WARMUP (separate phase)"
echo "  - SEQUENTIAL_CFG_TEST (original implementation)"
echo "  - BATCHED_CFG_TEST (optimized implementation)"
echo ""
echo "To analyze:"
echo "  1. Download profile:"
echo "     scp horde:$OUTPUT_DIR/cfg_optimization_comparison.nsys-rep /home/tobyj/Downloads/"
echo ""
echo "  2. Open in Nsight Systems UI"
echo ""
echo "  3. Compare side-by-side:"
echo "     - Zoom to SEQUENTIAL_CFG_TEST vs BATCHED_CFG_TEST ranges"
echo "     - Compare DiT forward pass time"
echo "     - Compare control branch computation"
echo "     - Compare number of kernel launches"
echo "     - Compare GPU utilization"
echo ""

