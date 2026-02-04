#!/bin/bash
# Run Cosmos inference for all tile sizes
# Usage: ./run_all_cosmos.sh [tile_count] [mode]
#   tile_count: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 (default: all)
#   mode: depth, seg, both (default: both)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSMOS_DIR="$HOME/cosmos/cosmos-transfer2.5"
INFERENCE_SCRIPT="$COSMOS_DIR/examples/inference.py"

# Setup Cosmos environment
setup_cosmos() {
    echo "Setting up Cosmos environment..."
    cd "$COSMOS_DIR"
    source .venv/bin/activate
    export LD_LIBRARY_PATH="$COSMOS_DIR/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
    echo "Environment ready."
    echo ""
}

run_inference() {
    local tile_num=$1
    local mode=$2  # "depth" or "seg"
    
    local tile_dir="$SCRIPT_DIR/tile_$tile_num"
    local output_dir="$tile_dir/output"
    
    if [ "$mode" == "depth" ]; then
        local spec_file="$tile_dir/specs/depth_only.json"
        local output_name="depth_$tile_num"
    else
        local spec_file="$tile_dir/specs/depth_seg.json"
        local output_name="depth_seg_$tile_num"
    fi
    
    # Check if spec file exists
    if [ ! -f "$spec_file" ]; then
        echo "Warning: Spec file not found: $spec_file"
        echo "Run ./generate_specs.sh first"
        return 1
    fi
    
    # Check if capture exists
    if [ ! -f "$tile_dir/capture/depth_video.mp4" ]; then
        echo "Warning: Capture not found for tile_$tile_num"
        echo "Run ./run_all_captures.sh $((10#$tile_num)) first"
        return 1
    fi
    
    echo "=============================================="
    echo "Running: tile_$tile_num ($mode)"
    echo "  Spec: $spec_file"
    echo "  Output: $output_dir"
    echo "=============================================="
    
    mkdir -p "$output_dir"
    
    # Run inference
    # Format: inference.py -i SPEC_FILE --output-dir DIR control:TYPE
    # Note: args must come BEFORE the subcommand
    python "$INFERENCE_SCRIPT" \
        -i "$spec_file" \
        --output-dir "$output_dir" \
        control:depth
    
    # Rename output to match convention
    local cosmos_output="$output_dir/${output_name}.mp4"
    if [ -f "$cosmos_output" ]; then
        echo "Output saved: $cosmos_output"
    else
        # Find the generated file and rename it
        local generated=$(find "$output_dir" -name "*.mp4" -newer "$spec_file" | head -1)
        if [ -n "$generated" ] && [ "$generated" != "$cosmos_output" ]; then
            mv "$generated" "$cosmos_output"
            echo "Output renamed to: $cosmos_output"
        fi
    fi
    
    echo ""
}

# Parse arguments
TILE_COUNT="${1:-all}"
MODE="${2:-both}"

# Setup environment first
setup_cosmos

if [ "$TILE_COUNT" == "all" ]; then
    TILES="001 002 004 008 016 032 064 128 256 512"
else
    TILES=$(printf "%03d" $TILE_COUNT)
fi

for tile_num in $TILES; do
    if [ "$MODE" == "both" ] || [ "$MODE" == "depth" ]; then
        run_inference "$tile_num" "depth"
    fi
    
    if [ "$MODE" == "both" ] || [ "$MODE" == "seg" ]; then
        run_inference "$tile_num" "seg"
    fi
done

echo "=============================================="
echo "All Cosmos inferences complete!"
echo "=============================================="
echo ""
echo "Results in:"
for tile_num in $TILES; do
    echo "  tile_$tile_num/output/"
done

