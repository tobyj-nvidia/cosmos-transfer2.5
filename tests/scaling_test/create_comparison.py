#!/usr/bin/env python3
"""
Create comparison video with 2x3 grid layout.

Layout (always 2 rows x 3 columns):
    ┌──────────────┬──────────────┬──────────────┐
    │ Input: Depth │ Input: Seg   │ Input: Albedo│
    ├──────────────┼──────────────┼──────────────┤
    │ Cosmos Out   │ + Overlay    │   (empty)    │
    └──────────────┴──────────────┴──────────────┘

Unused input tiles show black with "(Not Used)" label.

Modes:
    depth       - Only depth was used as control
    seg         - Depth + Segmentation were used
    albedo      - Depth + Seg + Albedo display (Albedo for reference only)

Usage:
    python create_comparison.py tile_001 depth
    python create_comparison.py tile_001 seg
    python create_comparison.py tile_001 albedo
    python create_comparison.py --all
"""

__version__ = "0.6.1"

import argparse
import subprocess
import sys
from pathlib import Path


def create_comparison(tile_dir: Path, mode: str, output_name: str = None):
    """Create a 2x3 comparison video for a tile.
    
    Args:
        tile_dir: Path to tile directory (e.g., tile_001/)
        mode: "depth" for depth-only, "seg" for depth+seg, "albedo" for depth+seg+albedo
        output_name: Custom output filename (optional)
    """
    capture_dir = tile_dir / "capture"
    output_dir = tile_dir / "output"
    
    # Input files
    depth_video = capture_dir / "depth_video.mp4"
    seg_video = capture_dir / "seg_video.mp4"
    albedo_video = capture_dir / "albedo_video.mp4"
    
    tile_num = tile_dir.name.replace("tile_", "")
    
    # Determine which inputs were used and output files
    if mode == "depth":
        cosmos_video = output_dir / f"depth_{tile_num}.mp4"
        overlay_video = output_dir / f"depth_{tile_num}_overlay.mp4"
        default_output = output_dir / f"comparison_depth_{tile_num}.mp4"
        use_seg = False
        use_albedo = False
    elif mode == "albedo":
        cosmos_video = output_dir / f"depth_seg_{tile_num}.mp4"
        overlay_video = output_dir / f"depth_seg_{tile_num}_overlay.mp4"
        default_output = output_dir / f"comparison_all_{tile_num}.mp4"
        use_seg = True
        use_albedo = True
    else:  # mode == "seg"
        cosmos_video = output_dir / f"depth_seg_{tile_num}.mp4"
        overlay_video = output_dir / f"depth_seg_{tile_num}_overlay.mp4"
        default_output = output_dir / f"comparison_depth_seg_{tile_num}.mp4"
        use_seg = True
        use_albedo = False
    
    output_file = Path(output_name) if output_name else default_output
    
    # Check required inputs exist
    required_files = [
        (depth_video, "depth"),
        (cosmos_video, "cosmos"),
        (overlay_video, "overlay")
    ]
    if use_seg:
        required_files.append((seg_video, "seg"))
    if use_albedo:
        required_files.append((albedo_video, "albedo"))
    
    missing = []
    for f, name in required_files:
        if not f.exists():
            missing.append(f"{name}: {f}")
    
    if missing:
        print(f"  Missing files for {tile_dir.name} ({mode}):")
        for m in missing:
            print(f"    - {m}")
        return False
    
    print(f"  Creating: {output_file.name}")
    
    # Tile dimensions: 1280/3 ≈ 426, 704/2 = 352
    tile_w = 426
    tile_h = 352
    
    # Build ffmpeg command for 2x3 grid
    # Input order: 0=depth, 1=cosmos, 2=overlay
    # Then optionally: 3=seg, 4=albedo
    
    inputs = [
        "-i", str(depth_video),
        "-i", str(cosmos_video),
        "-i", str(overlay_video),
    ]
    
    if use_seg:
        inputs.extend(["-i", str(seg_video)])
    if use_albedo:
        inputs.extend(["-i", str(albedo_video)])
    
    # Build filter complex - NO TEXT LABELS (drawtext is extremely slow on this system)
    filter_parts = []
    
    # Top row: Depth | Seg | Albedo
    filter_parts.append(f"[0:v]scale={tile_w}:{tile_h}[depth]")
    
    # v1 = seg or black placeholder
    if use_seg:
        seg_input = "3" if not use_albedo else "3"
        filter_parts.append(f"[{seg_input}:v]scale={tile_w}:{tile_h}[seg]")
    else:
        filter_parts.append(f"color=black:s={tile_w}x{tile_h}:d=4:r=30,trim=end_frame=93[seg]")
    
    # v2 = albedo or black placeholder
    if use_albedo:
        albedo_input = "4"
        filter_parts.append(f"[{albedo_input}:v]scale={tile_w}:{tile_h}[albedo]")
    else:
        filter_parts.append(f"color=black:s={tile_w}x{tile_h}:d=4:r=30,trim=end_frame=93[albedo]")
    
    # Bottom row: Cosmos | Overlay | Empty
    filter_parts.append(f"[1:v]scale={tile_w}:{tile_h}[cosmos]")
    filter_parts.append(f"[2:v]scale={tile_w}:{tile_h}[overlay]")
    filter_parts.append(f"color=black:s={tile_w}x{tile_h}:d=4:r=30,trim=end_frame=93[empty]")
    
    # Stack into 2x3 grid
    filter_parts.append("[depth][seg][albedo]hstack=inputs=3[top]")
    filter_parts.append("[cosmos][overlay][empty]hstack=inputs=3[bottom]")
    filter_parts.append("[top][bottom]vstack[out]")
    
    filter_complex = ";".join(filter_parts)
    
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-stats",  # Show encoding progress
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", "18",  # High quality (near-lossless)
        "-preset", "medium",  # Balanced speed/compression
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(output_file)
    ]
    
    try:
        # Show progress (don't capture stderr so we see ffmpeg stats)
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=False)
        print(f"    ✓ Saved: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Error: ffmpeg failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create 2x3 comparison videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("tile", nargs="?", type=str,
                        help="Tile directory name (e.g., tile_001) or 'all'")
    parser.add_argument("mode", nargs="?", type=str, choices=["depth", "seg", "albedo", "all"],
                        default="seg",
                        help="Mode: depth, seg, albedo, or all (default: seg)")
    parser.add_argument("--all", action="store_true",
                        help="Process all tile directories")
    parser.add_argument("--output", "-o", type=str,
                        help="Custom output filename")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    print(f"Comparison Video Generator v{__version__}")
    print("=" * 50)
    print("Layout: 2x3 grid")
    print("  Row 1: Depth | Segmentation | Albedo")
    print("  Row 2: Cosmos | Overlay | (empty)")
    print("=" * 50)
    
    # Determine which tiles to process
    if args.all or args.tile == "all":
        tile_dirs = sorted(script_dir.glob("tile_*"))
        if not tile_dirs:
            print("No tile directories found!")
            sys.exit(1)
    elif args.tile:
        tile_path = script_dir / args.tile
        if not tile_path.exists():
            # Try adding tile_ prefix
            tile_path = script_dir / f"tile_{args.tile.zfill(3)}"
        if not tile_path.exists():
            print(f"Error: Tile directory not found: {args.tile}")
            sys.exit(1)
        tile_dirs = [tile_path]
    else:
        parser.print_help()
        sys.exit(1)
    
    # Process each tile
    success_count = 0
    total_count = 0
    
    for tile_dir in tile_dirs:
        print(f"\nProcessing {tile_dir.name}...")
        
        # Determine which modes to run
        if args.mode == "all":
            modes = ["depth", "seg"]
            # Only add albedo mode if albedo video exists
            if (tile_dir / "capture" / "albedo_video.mp4").exists():
                modes.append("albedo")
        else:
            modes = [args.mode]
        
        for mode in modes:
            total_count += 1
            if create_comparison(tile_dir, mode, args.output):
                success_count += 1
    
    print()
    print("=" * 50)
    print(f"Complete: {success_count}/{total_count} comparisons created")
    
    if success_count > 0:
        print("\nOutput files:")
        for tile_dir in tile_dirs:
            for f in sorted((tile_dir / "output").glob("comparison_*.mp4")):
                print(f"  {f}")


if __name__ == "__main__":
    main()
