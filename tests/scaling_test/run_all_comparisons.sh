#!/bin/bash
# Generate comparison videos (2x3 tiled grid) for all Cosmos outputs
# Usage: ./run_all_comparisons.sh [tile_count] [mode]
#   tile_count: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, or "all" (default: all)
#   mode: depth, seg, albedo, all (default: all)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARISON_SCRIPT="$SCRIPT_DIR/create_comparison.py"

# Parse arguments
TILE_COUNT="${1:-all}"
MODE="${2:-all}"

# Check comparison script exists
if [ ! -f "$COMPARISON_SCRIPT" ]; then
    echo "Error: create_comparison.py not found at $COMPARISON_SCRIPT"
    exit 1
fi

echo "=============================================="
echo "Generating Comparison Videos (mode=$MODE)"
echo "=============================================="
echo ""
echo "Layout: 2x3 grid"
echo "  ┌──────────────┬──────────────┬──────────────┐"
echo "  │ Input: Depth │ Input: Seg   │ Input: Albedo│"
echo "  ├──────────────┼──────────────┼──────────────┤"
echo "  │ Cosmos Out   │ + Overlay    │   (empty)    │"
echo "  └──────────────┴──────────────┴──────────────┘"
echo ""
echo "Unused inputs show black with '(Not Used)' label"
echo ""

if [ "$TILE_COUNT" == "all" ]; then
    TILES="001 002 004 008 016 032 064 128 256 512"
else
    TILES=$(printf "%03d" $TILE_COUNT)
fi

SUCCESS=0
TOTAL=0

for tile_num in $TILES; do
    tile_dir="$SCRIPT_DIR/tile_$tile_num"
    
    if [ ! -d "$tile_dir" ]; then
        echo "Skipping tile_$tile_num (directory not found)"
        continue
    fi
    
    echo "Processing tile_$tile_num..."
    
    # Determine modes to run
    if [ "$MODE" == "all" ]; then
        MODES="depth seg"
        # Add albedo mode if albedo video exists
        if [ -f "$tile_dir/capture/albedo_video.mp4" ]; then
            MODES="$MODES albedo"
        fi
    else
        MODES="$MODE"
    fi
    
    for m in $MODES; do
        TOTAL=$((TOTAL + 1))
        
        if python "$COMPARISON_SCRIPT" "tile_$tile_num" "$m" 2>/dev/null; then
            SUCCESS=$((SUCCESS + 1))
        fi
    done
    
    echo ""
done

echo "=============================================="
echo "Complete: $SUCCESS/$TOTAL comparison videos created"
echo "=============================================="

if [ $SUCCESS -gt 0 ]; then
    echo ""
    echo "Output files:"
    for tile_num in $TILES; do
        output_dir="$SCRIPT_DIR/tile_$tile_num/output"
        if [ -d "$output_dir" ]; then
            for f in "$output_dir"/comparison_*.mp4; do
                [ -f "$f" ] && echo "  $f"
            done
        fi
    done
fi
