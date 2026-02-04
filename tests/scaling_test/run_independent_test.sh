#!/bin/bash
# Full workflow for independent simulation test
#
# Creates a 93-frame video where each frame shows 8 robots at step 10
# of INDEPENDENT simulations, then runs Cosmos Transfer on it.
#
# Usage:
#   ./run_independent_test.sh [--skip-capture] [--skip-cosmos] [--no-albedo]
#
# Steps:
#   1. Capture: 93 runs × 8 envs × 10 frames each
#   2. Assemble: Extract frame 10 from each run → 93-frame video
#   3. Cosmos: Run depth and depth+seg transfer
#   4. Post-process: Overlays and comparisons
#
# Output:
#   tests/temporal-stability/
#   ├── run_*/                   # 93 capture directories
#   ├── final/                   # Assembled videos
#   │   ├── rgb_video.mp4
#   │   ├── depth_video.mp4
#   │   ├── seg_video.mp4
#   │   └── albedo_video.mp4
#   ├── specs/                   # Cosmos spec files
#   │   ├── depth_only.json
#   │   └── depth_seg.json
#   └── output/                  # Cosmos outputs + comparisons
#       ├── depth_independent.mp4
#       ├── depth_seg_independent.mp4
#       └── comparison_*.mp4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$SCRIPT_DIR/../temporal-stability"
COSMOS_DIR="$HOME/cosmos/cosmos-transfer2.5"

# Source shared configuration (for prompts, settings)
source "$SCRIPT_DIR/config.sh"

# Parse arguments
SKIP_CAPTURE=false
SKIP_COSMOS=false
NO_ALBEDO=""

for arg in "$@"; do
    case "$arg" in
        --skip-capture)
            SKIP_CAPTURE=true
            ;;
        --skip-cosmos)
            SKIP_COSMOS=true
            ;;
        --no-albedo)
            NO_ALBEDO="--no-albedo"
            ;;
    esac
done

echo "=============================================="
echo "Independent Simulation Test"
echo "=============================================="
echo ""
echo "Goal: 93-frame video where each frame shows 8 robots"
echo "      at step 10 of INDEPENDENT simulations"
echo ""
echo "Options:"
[ "$SKIP_CAPTURE" = true ] && echo "  - Skipping capture (using existing)"
[ "$SKIP_COSMOS" = true ] && echo "  - Skipping Cosmos inference"
[ -n "$NO_ALBEDO" ] && echo "  - No albedo capture"
echo ""

#-------------------------------------------
# Step 1: Capture
#-------------------------------------------
if [ "$SKIP_CAPTURE" = false ]; then
    echo "=============================================="
    echo "Step 1: Capturing 93 independent runs..."
    echo "=============================================="
    echo ""
    
    "$SCRIPT_DIR/run_independent_captures.sh" $NO_ALBEDO
    
    echo ""
fi

#-------------------------------------------
# Step 2: Assemble videos
#-------------------------------------------
echo "=============================================="
echo "Step 2: Assembling final videos..."
echo "=============================================="
echo ""

python "$SCRIPT_DIR/assemble_independent_video.py" \
    --input-dir "$TEST_DIR" \
    --output-dir "$TEST_DIR/final"

echo ""

#-------------------------------------------
# Step 3: Generate Cosmos specs
#-------------------------------------------
echo "=============================================="
echo "Step 3: Generating Cosmos spec files..."
echo "=============================================="
echo ""

mkdir -p "$TEST_DIR/specs"
mkdir -p "$TEST_DIR/output"

# Get prompt from shared config (8 envs = uses GRID_DESC[8])
PROMPT=$(get_prompt 8)

# Depth-only spec
cat > "$TEST_DIR/specs/depth_only.json" << EOF
{
  "name": "depth_independent",
  "prompt": "$PROMPT",
  "video_path": "../final/depth_video.mp4",
  "guidance": 7.0,
  "num_steps": 35,
  "image_context_path": "../../kuka-reference-image.png",
  "depth": {
    "control_path": "../final/depth_video.mp4",
    "control_weight": 1.0
  }
}
EOF
echo "  Created: $TEST_DIR/specs/depth_only.json"

# Depth + Seg spec
cat > "$TEST_DIR/specs/depth_seg.json" << EOF
{
  "name": "depth_seg_independent",
  "prompt": "$PROMPT",
  "video_path": "../final/depth_video.mp4",
  "guidance": 7.0,
  "num_steps": 35,
  "image_context_path": "../../kuka-reference-image.png",
  "depth": {
    "control_path": "../final/depth_video.mp4",
    "control_weight": 1.0
  },
  "seg": {
    "control_path": "../final/seg_video.mp4",
    "control_weight": 1.0
  }
}
EOF
echo "  Created: $TEST_DIR/specs/depth_seg.json"

echo ""

#-------------------------------------------
# Step 4: Run Cosmos inference
#-------------------------------------------
if [ "$SKIP_COSMOS" = false ]; then
    echo "=============================================="
    echo "Step 4: Running Cosmos inference..."
    echo "=============================================="
    echo ""
    
    # Setup Cosmos environment
    cd "$COSMOS_DIR"
    source .venv/bin/activate
    export LD_LIBRARY_PATH="$COSMOS_DIR/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
    
    echo "Running depth-only inference..."
    python examples/inference.py \
        -i "$TEST_DIR/specs/depth_only.json" \
        --output-dir "$TEST_DIR/output" \
        control:depth
    
    echo ""
    echo "Running depth+seg inference..."
    python examples/inference.py \
        -i "$TEST_DIR/specs/depth_seg.json" \
        --output-dir "$TEST_DIR/output" \
        control:depth
    
    echo ""
fi

#-------------------------------------------
# Step 5: Generate overlays
#-------------------------------------------
echo "=============================================="
echo "Step 5: Generating overlays..."
echo "=============================================="
echo ""

# Check if overlay script exists
OVERLAY_SCRIPT="$SCRIPT_DIR/overlay_depth.py"
if [ -f "$OVERLAY_SCRIPT" ]; then
    # Depth-only overlay
    if [ -f "$TEST_DIR/output/depth_independent.mp4" ]; then
        python "$OVERLAY_SCRIPT" \
            "$TEST_DIR/output/depth_independent.mp4" \
            "$TEST_DIR/final/depth_video.mp4" \
            --alpha 0.6 \
            --output "$TEST_DIR/output/depth_independent_overlay.mp4"
        echo "  Created: depth_independent_overlay.mp4"
    fi
    
    # Depth+seg overlay
    if [ -f "$TEST_DIR/output/depth_seg_independent.mp4" ]; then
        python "$OVERLAY_SCRIPT" \
            "$TEST_DIR/output/depth_seg_independent.mp4" \
            "$TEST_DIR/final/depth_video.mp4" \
            --alpha 0.6 \
            --output "$TEST_DIR/output/depth_seg_independent_overlay.mp4"
        echo "  Created: depth_seg_independent_overlay.mp4"
    fi
else
    echo "  Skipping overlays (overlay_depth.py not found)"
fi

echo ""

#-------------------------------------------
# Step 6: Generate comparison grids
#-------------------------------------------
echo "=============================================="
echo "Step 6: Generating comparison videos..."
echo "=============================================="
echo ""

# We need to create comparison videos manually since this isn't in tile_* structure
# Use ffmpeg directly

FINAL_DIR="$TEST_DIR/final"
OUTPUT_DIR="$TEST_DIR/output"

# Check if we have the needed files
if [ -f "$OUTPUT_DIR/depth_independent.mp4" ] && [ -f "$OUTPUT_DIR/depth_independent_overlay.mp4" ]; then
    echo "Creating depth-only comparison..."
    
    ffmpeg -y \
        -i "$FINAL_DIR/depth_video.mp4" \
        -i "$OUTPUT_DIR/depth_independent.mp4" \
        -i "$OUTPUT_DIR/depth_independent_overlay.mp4" \
        -filter_complex \
        "[0:v]scale=426:352,drawtext=text='Input\\: Depth':fontsize=18:fontcolor=yellow:x=10:y=10:box=1:boxcolor=black@0.7[depth];
         color=black:s=426x352:d=1,loop=-1:size=93,drawtext=text='(Seg Not Used)':fontsize=16:fontcolor=gray:x=(w-text_w)/2:y=(h-text_h)/2[seg];
         color=black:s=426x352:d=1,loop=-1:size=93,drawtext=text='(Albedo Not Used)':fontsize=16:fontcolor=gray:x=(w-text_w)/2:y=(h-text_h)/2[albedo];
         [1:v]scale=426:352,drawtext=text='Cosmos Output':fontsize=18:fontcolor=cyan:x=10:y=10:box=1:boxcolor=black@0.7[cosmos];
         [2:v]scale=426:352,drawtext=text='Output + Depth':fontsize=18:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[overlay];
         color=black:s=426x352:d=1,loop=-1:size=93[empty];
         [depth][seg][albedo]hstack=inputs=3[top];
         [cosmos][overlay][empty]hstack=inputs=3[bottom];
         [top][bottom]vstack[out]" \
        -map "[out]" \
        -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -shortest \
        "$OUTPUT_DIR/comparison_depth_independent.mp4" 2>/dev/null
    
    echo "  Created: comparison_depth_independent.mp4"
fi

if [ -f "$OUTPUT_DIR/depth_seg_independent.mp4" ] && [ -f "$OUTPUT_DIR/depth_seg_independent_overlay.mp4" ]; then
    echo "Creating depth+seg comparison..."
    
    ffmpeg -y \
        -i "$FINAL_DIR/depth_video.mp4" \
        -i "$FINAL_DIR/seg_video.mp4" \
        -i "$OUTPUT_DIR/depth_seg_independent.mp4" \
        -i "$OUTPUT_DIR/depth_seg_independent_overlay.mp4" \
        -filter_complex \
        "[0:v]scale=426:352,drawtext=text='Input\\: Depth':fontsize=18:fontcolor=yellow:x=10:y=10:box=1:boxcolor=black@0.7[depth];
         [1:v]scale=426:352,drawtext=text='Input\\: Segmentation':fontsize=18:fontcolor=yellow:x=10:y=10:box=1:boxcolor=black@0.7[seg];
         color=black:s=426x352:d=1,loop=-1:size=93,drawtext=text='(Albedo Not Used)':fontsize=16:fontcolor=gray:x=(w-text_w)/2:y=(h-text_h)/2[albedo];
         [2:v]scale=426:352,drawtext=text='Cosmos Output':fontsize=18:fontcolor=cyan:x=10:y=10:box=1:boxcolor=black@0.7[cosmos];
         [3:v]scale=426:352,drawtext=text='Output + Depth':fontsize=18:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[overlay];
         color=black:s=426x352:d=1,loop=-1:size=93[empty];
         [depth][seg][albedo]hstack=inputs=3[top];
         [cosmos][overlay][empty]hstack=inputs=3[bottom];
         [top][bottom]vstack[out]" \
        -map "[out]" \
        -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -shortest \
        "$OUTPUT_DIR/comparison_depth_seg_independent.mp4" 2>/dev/null
    
    echo "  Created: comparison_depth_seg_independent.mp4"
fi

echo ""

#-------------------------------------------
# Done
#-------------------------------------------
echo "=============================================="
echo "Independent Test Complete!"
echo "=============================================="
echo ""
echo "Output directory: $TEST_DIR"
echo ""
echo "Videos:"
ls -la "$TEST_DIR/final/"*.mp4 2>/dev/null || echo "  (no final videos)"
echo ""
echo "Cosmos outputs:"
ls -la "$TEST_DIR/output/"*.mp4 2>/dev/null || echo "  (no cosmos outputs)"
echo ""

