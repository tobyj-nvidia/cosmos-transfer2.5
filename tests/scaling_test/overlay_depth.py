#!/usr/bin/env python3
"""
Overlay depth video on top of a generated RGB video with alpha blending.
Uses ffmpeg only - no OpenCV required.

Usage:
    python overlay_depth.py <rgb_video> <depth_video> [--alpha 0.6] [--output overlay.mp4]

Examples:
    # Basic overlay with 60% depth opacity
    python overlay_depth.py cosmos_output.mp4 depth_video.mp4

    # Custom alpha (30% depth visibility)
    python overlay_depth.py cosmos_output.mp4 depth_video.mp4 --alpha 0.3

    # Custom output path
    python overlay_depth.py cosmos_output.mp4 depth_video.mp4 --output my_overlay.mp4
"""

__version__ = "0.2.0"

import argparse
import subprocess
import sys
from pathlib import Path


def create_overlay(rgb_video: Path, overlay_video: Path, output_path: Path, alpha: float = 0.6):
    """Create an overlay video using ffmpeg.
    
    Args:
        rgb_video: Path to the base RGB video (from Cosmos)
        overlay_video: Path to depth or segmentation video to overlay
        output_path: Output video path
        alpha: Overlay opacity (0.0-1.0, default: 0.6)
    """
    # ffmpeg blend filter: blend=all_expr='A*(1-{alpha})+B*{alpha}'
    # Or use the overlay filter with colorchannelmixer for opacity
    
    # Using blend filter for pixel-level blending
    # Format: [0:v][1:v]blend=all_expr='A*(1-alpha)+B*alpha'
    
    blend_expr = f"A*{1-alpha}+B*{alpha}"
    
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(rgb_video),
        "-i", str(overlay_video),
        "-filter_complex",
        f"[0:v][1:v]blend=all_expr='{blend_expr}'[out]",
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr[:500] if e.stderr else 'unknown error'}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Overlay depth/segmentation video on generated RGB video (ffmpeg-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("rgb_video", type=str,
                        help="Path to generated RGB video (from Cosmos)")
    parser.add_argument("overlay_video", type=str,
                        help="Path to depth or segmentation video to overlay")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Overlay opacity (0.0-1.0, default: 0.6)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output video path (default: auto-generated)")
    
    args = parser.parse_args()
    
    rgb_path = Path(args.rgb_video).resolve()
    overlay_path = Path(args.overlay_video).resolve()
    
    if not rgb_path.exists():
        print(f"Error: RGB video not found: {rgb_path}")
        sys.exit(1)
    if not overlay_path.exists():
        print(f"Error: Overlay video not found: {overlay_path}")
        sys.exit(1)
    
    # Auto-generate output name if not provided
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = rgb_path.parent / f"{rgb_path.stem}_overlay.mp4"
    
    print(f"Overlay Tool v{__version__} (ffmpeg-only)")
    print("=" * 50)
    print(f"RGB Video:     {rgb_path.name}")
    print(f"Overlay Video: {overlay_path.name}")
    print(f"Alpha:         {args.alpha}")
    print(f"Output:        {output_path.name}")
    print()
    
    print("Creating overlay...")
    if create_overlay(rgb_path, overlay_path, output_path, args.alpha):
        print(f"Done! Saved to: {output_path}")
    else:
        print("Failed to create overlay")
        sys.exit(1)


if __name__ == "__main__":
    main()

