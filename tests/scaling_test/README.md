# Cosmos Transfer 2.5 Scaling Tests

Systematic testing of Isaac Lab + Cosmos with varying environment counts.

## Folder Structure

```
scaling_test/
├── README.md                    # This file
├── run_all_captures.sh          # Capture all Isaac Lab videos
├── generate_specs.sh            # Generate Cosmos spec files
├── run_all_cosmos.sh            # Run all Cosmos inferences
├── run_all_overlays.sh          # Generate depth overlay videos
├── run_all_comparisons.sh       # Generate 2x3 comparison grids
├── create_comparison.py         # Comparison video generator
├── overlay_depth.py             # Overlay tool (copy from parent)
├── kuka-reference-image.png     # Style reference (copy from tests/)
│
├── tile_001/                    # 1 environment (1280x704)
│   ├── capture/                 # Isaac Lab outputs
│   │   ├── rgb_video.mp4
│   │   ├── depth_video.mp4
│   │   ├── seg_video.mp4
│   │   └── albedo_video.mp4     # (if Newton albedo branch)
│   ├── specs/                   # Cosmos spec files
│   │   ├── depth_only.json
│   │   └── depth_seg.json
│   └── output/                  # Cosmos outputs
│       ├── depth_001.mp4             # Depth-only generation
│       ├── depth_001_overlay.mp4     # With depth overlay (60%)
│       ├── depth_seg_001.mp4         # Depth+Seg generation
│       ├── depth_seg_001_overlay.mp4 # With depth overlay
│       ├── comparison_depth_001.mp4      # 2x3 grid (depth only)
│       ├── comparison_depth_seg_001.mp4  # 2x3 grid (depth+seg)
│       └── comparison_all_001.mp4        # 2x3 grid (all inputs)
│
├── tile_002/                    # 2 environments (640x704 per tile)
├── tile_004/                    # 4 environments
├── tile_008/                    # 8 environments
├── tile_016/                    # 16 environments
├── tile_032/                    # 32 environments
├── tile_064/                    # 64 environments
├── tile_128/                    # 128 environments
├── tile_256/                    # 256 environments
└── tile_512/                    # 512 environments
```

## Comparison Video Layout

The final output is a 2x3 grid showing inputs and outputs:

```
┌──────────────┬──────────────┬──────────────┐
│ Input: Depth │ Input: Seg   │ Input: Albedo│
├──────────────┼──────────────┼──────────────┤
│ Cosmos Out   │ + Overlay    │   (empty)    │
└──────────────┴──────────────┴──────────────┘
```

Unused input tiles show black with "(Not Used)" label.

## Quick Start

### 1. Setup (on Horde)

```bash
cd ~/cosmos/cosmos-transfer2.5/tests/scaling_test

# Copy reference image and overlay script
cp ../kuka-reference-image.png .
cp ../overlay_depth.py .

# Make scripts executable
chmod +x *.sh
```

### 2. Capture Isaac Lab Videos

```bash
# Capture all tile sizes with depth, seg, and albedo (~15 min)
./run_all_captures.sh

# Capture without albedo (if Newton albedo branch not installed)
./run_all_captures.sh --no-albedo

# Or capture single tile size
./run_all_captures.sh 8
./run_all_captures.sh 8 --no-albedo
```

### 3. Generate Cosmos Spec Files

```bash
./generate_specs.sh
```

### 4. Run Cosmos Inference

```bash
# Setup Cosmos environment
cd ~/cosmos/cosmos-transfer2.5
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# Run all inferences
cd tests/scaling_test
./run_all_cosmos.sh

# Or run specific tile/mode
./run_all_cosmos.sh 8 depth     # Depth-only
./run_all_cosmos.sh 8 seg       # Depth+Seg
./run_all_cosmos.sh all both    # All tiles, both modes
```

### 5. Generate Overlay Videos

```bash
# Generate all overlays (60% opacity default)
./run_all_overlays.sh

# Or specific tile with custom alpha
./run_all_overlays.sh 8 0.5
```

### 6. Generate Comparison Grids

```bash
# Generate all 2x3 comparison videos
./run_all_comparisons.sh

# Or specific modes
./run_all_comparisons.sh all depth   # Depth-only comparisons
./run_all_comparisons.sh all seg     # Depth+Seg comparisons
./run_all_comparisons.sh all albedo  # All inputs (with albedo display)
./run_all_comparisons.sh 8 seg       # Single tile, depth+seg
```

## Complete Workflow

Run everything in sequence:

```bash
# 1. Isaac Lab captures
./run_all_captures.sh

# 2. Generate spec files
./generate_specs.sh

# 3. Cosmos inference (requires Cosmos environment)
cd ~/cosmos/cosmos-transfer2.5
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
cd tests/scaling_test
./run_all_cosmos.sh

# 4. Post-processing
./run_all_overlays.sh
./run_all_comparisons.sh

# Final outputs in tile_*/output/comparison_*.mp4
```

## Spec File Format

### Depth-Only (`depth_only.json`)
```json
{
  "name": "depth_008",
  "prompt": "Photorealistic Kuka robot arms...",
  "video_path": "../capture/depth_video.mp4",
  "guidance": 7.0,
  "num_steps": 35,
  "image_context_path": "../../kuka-reference-image.png",
  "depth": {
    "control_path": "../capture/depth_video.mp4",
    "control_weight": 1.0
  }
}
```

### Depth + Segmentation (`depth_seg.json`)
```json
{
  "name": "depth_seg_008",
  "prompt": "Photorealistic Kuka robot arms...",
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
```

## Output Naming

| Mode | Cosmos Output | Overlay | Comparison |
|------|---------------|---------|------------|
| Depth only | `depth_001.mp4` | `depth_001_overlay.mp4` | `comparison_depth_001.mp4` |
| Depth + Seg | `depth_seg_001.mp4` | `depth_seg_001_overlay.mp4` | `comparison_depth_seg_001.mp4` |
| All inputs | (uses depth+seg) | (uses depth+seg) | `comparison_all_001.mp4` |

## Notes

- **Captures**: Use `--no-reset` to prevent simulation resets during video
- **Depth**: Uses `--depth-max 10` for proper gradient distribution
- **Segmentation**: Uses Cosmos-compatible colors (PREDEFINED_COLORS_SEGMENTATION)
- **Albedo**: For reference display only (Cosmos doesn't use it as control)
- **Video length**: 93 frames at 30fps (native Cosmos chunk size)
- **Resolution**: Always 1280x704 total (720p), tiles are subdivided

## Tile Sizes

| Envs | Grid | Per-Tile Size |
|------|------|---------------|
| 1 | 1×1 | 1280×704 |
| 2 | 2×1 | 640×704 |
| 4 | 2×2 | 640×352 |
| 8 | 4×2 | 320×352 |
| 16 | 4×4 | 320×176 |
| 32 | 8×4 | 160×176 |
| 64 | 8×8 | 160×88 |
| 128 | 16×8 | 80×88 |
| 256 | 16×16 | 80×44 |
| 512 | 32×16 | 40×44 |
