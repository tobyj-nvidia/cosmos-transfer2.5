#!/bin/bash
# Generate overlay videos for all Cosmos outputs
# Usage: ./run_all_overlays.sh [tile_count] [alpha]
#   tile_count: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, or "all" (default: all)
#   alpha: 0.0-1.0 (default: 0.6)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_SCRIPT="$SCRIPT_DIR/overlay_depth.py"

# Parse arguments
TILE_COUNT="${1:-all}"
ALPHA="${2:-0.6}"

generate_overlay() {
    local tile_num=$1
    local mode=$2  # "depth" or "depth_seg"
    
    local tile_dir="$SCRIPT_DIR/tile_$tile_num"
    local capture_dir="$tile_dir/capture"
    local output_dir="$tile_dir/output"
    
    # Determine input/output files based on mode
    if [ "$mode" == "depth" ]; then
        local cosmos_output="$output_dir/depth_$tile_num.mp4"
        local overlay_input="$capture_dir/depth_video.mp4"
        local overlay_output="$output_dir/depth_${tile_num}_overlay.mp4"
    else
        local cosmos_output="$output_dir/depth_seg_$tile_num.mp4"
        local overlay_input="$capture_dir/depth_video.mp4"
        local overlay_output="$output_dir/depth_seg_${tile_num}_overlay.mp4"
    fi
    
    # Check if Cosmos output exists
    if [ ! -f "$cosmos_output" ]; then
        echo "  Skipping: $cosmos_output (not found)"
        return 0
    fi
    
    echo "  Generating overlay: $(basename $overlay_output)"
    
    python "$OVERLAY_SCRIPT" "$cosmos_output" "$overlay_input" \
        --alpha "$ALPHA" \
        --output "$overlay_output"
}

generate_seg_overlay() {
    local tile_num=$1
    
    local tile_dir="$SCRIPT_DIR/tile_$tile_num"
    local capture_dir="$tile_dir/capture"
    local output_dir="$tile_dir/output"
    
    local cosmos_output="$output_dir/depth_seg_$tile_num.mp4"
    local seg_input="$capture_dir/seg_video.mp4"
    local overlay_output="$output_dir/depth_seg_${tile_num}_seg_overlay.mp4"
    
    # Check if Cosmos output and seg video exist
    if [ ! -f "$cosmos_output" ]; then
        return 0
    fi
    if [ ! -f "$seg_input" ]; then
        return 0
    fi
    
    echo "  Generating seg overlay: $(basename $overlay_output)"
    
    python "$OVERLAY_SCRIPT" "$cosmos_output" "$seg_input" \
        --alpha "$ALPHA" \
        --output "$overlay_output"
}

# Check overlay script exists
if [ ! -f "$OVERLAY_SCRIPT" ]; then
    echo "Error: overlay_depth.py not found at $OVERLAY_SCRIPT"
    echo "Copy it with: cp ../overlay_depth.py ."
    exit 1
fi

echo "=============================================="
echo "Generating Overlay Videos (alpha=$ALPHA)"
echo "=============================================="
echo ""

if [ "$TILE_COUNT" == "all" ]; then
    TILES="001 002 004 008 016 032 064 128 256 512"
else
    TILES=$(printf "%03d" $TILE_COUNT)
fi

for tile_num in $TILES; do
    tile_dir="$SCRIPT_DIR/tile_$tile_num"
    
    if [ ! -d "$tile_dir/output" ]; then
        echo "Skipping tile_$tile_num (no output directory)"
        continue
    fi
    
    echo "Processing tile_$tile_num..."
    
    # Generate depth overlay for depth-only output
    generate_overlay "$tile_num" "depth"
    
    # Generate depth overlay for depth+seg output  
    generate_overlay "$tile_num" "depth_seg"
    
    # Generate seg overlay for depth+seg output
    generate_seg_overlay "$tile_num"
    
    echo ""
done

echo "=============================================="
echo "All overlays complete!"
echo "=============================================="
echo ""
echo "Overlay files:"
for tile_num in $TILES; do
    output_dir="$SCRIPT_DIR/tile_$tile_num/output"
    if [ -d "$output_dir" ]; then
        for f in "$output_dir"/*_overlay.mp4; do
            [ -f "$f" ] && echo "  $f"
        done
    fi
done

