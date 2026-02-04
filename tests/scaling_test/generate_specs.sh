#!/bin/bash
# Generate Cosmos spec files for all tile sizes
# Run this after capturing Isaac Lab videos

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source shared configuration
source "$SCRIPT_DIR/config.sh"

# Grid descriptions for prompts (with zero-padded keys for this script)
declare -A GRID_DESC_PADDED=(
    ["001"]="${GRID_DESC[1]}"
    ["002"]="${GRID_DESC[2]}"
    ["004"]="${GRID_DESC[4]}"
    ["008"]="${GRID_DESC[8]}"
    ["016"]="${GRID_DESC[16]}"
    ["032"]="${GRID_DESC[32]}"
    ["064"]="${GRID_DESC[64]}"
    ["128"]="${GRID_DESC[128]}"
    ["256"]="${GRID_DESC[256]}"
    ["512"]="${GRID_DESC[512]}"
)

generate_spec() {
    local tile_num=$1
    local tile_dir="$SCRIPT_DIR/tile_$tile_num"
    local specs_dir="$tile_dir/specs"
    
    # Build prompt from shared config
    local prompt="$BASE_PROMPT"
    if [ -n "${GRID_DESC_PADDED[$tile_num]}" ]; then
        prompt="$prompt, ${GRID_DESC_PADDED[$tile_num]}"
    fi
    
    echo "Generating specs for tile_$tile_num..."
    
    mkdir -p "$specs_dir"
    
    # Depth-only spec
    cat > "$specs_dir/depth_only.json" << EOF
{
  "name": "depth_$tile_num",
  "prompt": "$prompt",
  "video_path": "../capture/depth_video.mp4",
  "guidance": 7.0,
  "num_steps": 35,
  "image_context_path": "../../kuka-reference-image.png",
  "depth": {
    "control_path": "../capture/depth_video.mp4",
    "control_weight": 1.0
  }
}
EOF
    
    # Depth + Seg spec
    cat > "$specs_dir/depth_seg.json" << EOF
{
  "name": "depth_seg_$tile_num",
  "prompt": "$prompt",
  "video_path": "../capture/depth_video.mp4",
  "guidance": 7.0,
  "num_steps": 35,
  "image_context_path": "../../kuka-reference-image.png",
  "depth": {
    "control_path": "../capture/depth_video.mp4",
    "control_weight": 1.0
  },
  "seg": {
    "control_path": "../capture/seg_video.mp4",
    "control_weight": 1.0
  }
}
EOF
    
    echo "  Created: $specs_dir/depth_only.json"
    echo "  Created: $specs_dir/depth_seg.json"
}

echo "Generating Cosmos spec files..."
echo ""

for tile_num in 001 002 004 008 016 032 064 128 256 512; do
    generate_spec "$tile_num"
done

echo ""
echo "All spec files generated!"
echo ""
echo "To run Cosmos inference:"
echo "  cd ~/cosmos/cosmos-transfer2.5"
echo "  source .venv/bin/activate"
echo "  export LD_LIBRARY_PATH=\$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:\$LD_LIBRARY_PATH"
echo ""
echo "  # Depth-only"
echo "  python examples/inference.py tests/scaling_test/tile_008/specs/depth_only.json --output_dir tests/scaling_test/tile_008/output"
echo ""
echo "  # Depth + Seg"
echo "  python examples/inference.py tests/scaling_test/tile_008/specs/depth_seg.json --output_dir tests/scaling_test/tile_008/output"

