#!/bin/bash
# Capture Isaac Lab videos for all tile sizes
# Usage: ./run_all_captures.sh [tile_count] [--no-albedo]
#   If tile_count is provided, only capture that size
#   Otherwise, capture all sizes
#   Use --no-albedo if Newton albedo branch is not installed

set -e

# Parse arguments
NO_ALBEDO=""
TILE_COUNT=""
for arg in "$@"; do
    if [ "$arg" == "--no-albedo" ]; then
        NO_ALBEDO="--no-albedo"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        TILE_COUNT="$arg"
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$HOME/isaaclab-warp"
CAPTURE_SCRIPT="$ISAACLAB_DIR/scripts/capture_depth_video.py"

# Tile configurations: num_envs grid_cols
declare -A TILE_CONFIGS=(
    ["001"]="1 1"
    ["002"]="2 2"
    ["004"]="4 2"
    ["008"]="8 4"
    ["016"]="16 4"
    ["032"]="32 8"
    ["064"]="64 8"
    ["128"]="128 16"
    ["256"]="256 16"
    ["512"]="512 32"
)

capture_tile() {
    local tile_num=$1
    local num_envs=$2
    local grid_cols=$3
    
    local tile_dir="$SCRIPT_DIR/tile_$tile_num"
    local capture_dir="$tile_dir/capture"
    
    echo "=============================================="
    echo "Capturing tile_$tile_num: $num_envs envs, $grid_cols cols"
    echo "=============================================="
    
    # Create directories
    mkdir -p "$capture_dir"
    mkdir -p "$tile_dir/specs"
    mkdir -p "$tile_dir/output"
    
    # Build capture command
    local cmd="python $CAPTURE_SCRIPT"
    cmd="$cmd --num-envs $num_envs"
    cmd="$cmd --num-frames 93"
    cmd="$cmd --depth-max 10"
    cmd="$cmd --no-reset"
    cmd="$cmd --output-dir $capture_dir"
    
    # Add tiled options for multi-env
    if [ "$num_envs" -gt 1 ]; then
        cmd="$cmd --tiled --grid-cols $grid_cols"
    fi
    
    # Add no-albedo flag if requested
    if [ -n "$NO_ALBEDO" ]; then
        cmd="$cmd --no-albedo"
    fi
    
    echo "Running: $cmd"
    echo ""
    
    # Deactivate any current environment and activate Isaac Lab
    cd "$ISAACLAB_DIR"
    
    # Deactivate cosmos venv if active
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
    fi
    
    # Initialize and activate conda
    source ~/miniconda/etc/profile.d/conda.sh
    conda deactivate 2>/dev/null || true
    conda activate env_isaaclab
    
    eval "$cmd"
    
    echo ""
    echo "Captured: $capture_dir"
    echo "  - rgb_video.mp4"
    echo "  - depth_video.mp4"
    echo "  - seg_video.mp4"
    if [ -z "$NO_ALBEDO" ]; then
        echo "  - albedo_video.mp4"
    fi
    echo ""
}

# Check if specific tile count requested
if [ -n "$TILE_COUNT" ]; then
    tile_num=$(printf "%03d" $TILE_COUNT)
    if [ -z "${TILE_CONFIGS[$tile_num]}" ]; then
        echo "Error: Invalid tile count '$TILE_COUNT'. Valid options: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512"
        exit 1
    fi
    read num_envs grid_cols <<< "${TILE_CONFIGS[$tile_num]}"
    capture_tile "$tile_num" "$num_envs" "$grid_cols"
else
    # Capture all tile sizes
    echo "Capturing all tile sizes..."
    echo ""
    
    for tile_num in 001 002 004 008 016 032 064 128 256 512; do
        read num_envs grid_cols <<< "${TILE_CONFIGS[$tile_num]}"
        capture_tile "$tile_num" "$num_envs" "$grid_cols"
    done
    
    echo "=============================================="
    echo "All captures complete!"
    echo "=============================================="
fi

