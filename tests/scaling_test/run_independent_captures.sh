#!/bin/bash
# Capture 93 independent Isaac Lab simulations for temporally independent frames
#
# Goal: Create a 93-frame video where each frame shows 8 environments at step 10
#       of INDEPENDENT simulations (no temporal correlation between frames)
#
# Usage:
#   ./run_independent_captures.sh [--no-albedo]
#
# Output structure:
#   tests/temporal-stability/
#   ├── run_000/                 # First capture run
#   │   ├── rgb_frames/
#   │   │   └── frame_00009.png  # 10th frame (8 envs tiled)
#   │   ├── depth_frames/
#   │   ├── seg_frames/
#   │   └── albedo_frames/
#   ├── run_001/
#   │   └── ...
#   └── run_092/
#
# After this, run: python assemble_independent_video.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$HOME/isaaclab-warp"
CAPTURE_SCRIPT="$ISAACLAB_DIR/scripts/capture_depth_video.py"
OUTPUT_BASE="$SCRIPT_DIR/../temporal-stability"

# Parse arguments
NO_ALBEDO=""
for arg in "$@"; do
    if [ "$arg" == "--no-albedo" ]; then
        NO_ALBEDO="--no-albedo"
    fi
done

# Configuration
NUM_RUNS=93        # 93 frames for Cosmos
NUM_ENVS=8         # 8 environments per frame
GRID_COLS=4        # 4x2 grid
CAPTURE_FRAMES=10  # Capture 10 frames, use the 10th (index 9)

echo "=============================================="
echo "Independent Capture: $NUM_RUNS runs × $NUM_ENVS envs"
echo "=============================================="
echo ""
echo "Each run captures $CAPTURE_FRAMES frames of $NUM_ENVS environments."
echo "We'll use frame 10 (index 9) from each run."
echo "Total independent states: $((NUM_RUNS * NUM_ENVS)) = $(($NUM_RUNS * $NUM_ENVS))"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Setup Isaac Lab environment
cd "$ISAACLAB_DIR"

# Deactivate any current environment
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
fi

# Initialize and activate conda
source ~/miniconda/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate env_isaaclab

echo "Starting captures..."
echo ""

START_TIME=$(date +%s)

for run_idx in $(seq 0 $((NUM_RUNS - 1))); do
    run_num=$(printf "%03d" $run_idx)
    run_dir="$OUTPUT_BASE/run_$run_num"
    
    echo "[$run_num/$((NUM_RUNS-1))] Capturing run_$run_num..."
    
    # Build capture command
    cmd="python $CAPTURE_SCRIPT"
    cmd="$cmd --num-envs $NUM_ENVS"
    cmd="$cmd --num-frames $CAPTURE_FRAMES"
    cmd="$cmd --depth-max 10"
    cmd="$cmd --tiled --grid-cols $GRID_COLS"
    cmd="$cmd --output-dir $run_dir"
    
    if [ -n "$NO_ALBEDO" ]; then
        cmd="$cmd --no-albedo"
    fi
    
    # Run capture (suppress most output)
    eval "$cmd" > /dev/null 2>&1
    
    # Progress update every 10 runs
    if [ $((run_idx % 10)) -eq 9 ]; then
        elapsed=$(($(date +%s) - START_TIME))
        remaining_runs=$((NUM_RUNS - run_idx - 1))
        avg_time=$((elapsed / (run_idx + 1)))
        eta=$((remaining_runs * avg_time))
        echo "  Progress: $((run_idx + 1))/$NUM_RUNS runs complete. ETA: ${eta}s"
    fi
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "All captures complete!"
echo "=============================================="
echo ""
echo "Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s)"
echo "Output: $OUTPUT_BASE"
echo ""
echo "Next step: Assemble the final video"
echo "  python assemble_independent_video.py"
echo ""

