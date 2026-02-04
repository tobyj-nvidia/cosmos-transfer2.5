#!/usr/bin/env python3
"""
Assemble independent simulation frames into a single video.

Takes frame 10 (index 9) from each of 93 independent capture runs and
combines them into a 93-frame video for Cosmos Transfer.

Usage:
    python assemble_independent_video.py [--frame-idx 9] [--output-dir OUTPUT]

Input structure (from run_independent_captures.sh):
    tests/temporal-stability/
    ├── run_000/
    │   ├── rgb_frames/frame_00009.png
    │   ├── depth_frames/frame_00009.png
    │   ├── seg_frames/frame_00009.png
    │   └── albedo_frames/frame_00009.png (optional)
    ├── run_001/
    └── ...

Output:
    tests/temporal-stability/final/
    ├── rgb_video.mp4
    ├── depth_video.mp4
    ├── seg_video.mp4
    └── albedo_video.mp4 (if available)
"""

__version__ = "1.0.0"

import argparse
import subprocess
import sys
from pathlib import Path


def assemble_video(input_dir: Path, output_dir: Path, frame_idx: int = 9, fps: int = 30):
    """Assemble videos from independent capture runs.
    
    Args:
        input_dir: Directory containing run_000/, run_001/, etc.
        output_dir: Directory for output videos
        frame_idx: Which frame to extract from each run (default: 9 = 10th frame)
        fps: Output video frame rate
    """
    # Find all run directories
    run_dirs = sorted(input_dir.glob("run_*"))
    if not run_dirs:
        print(f"Error: No run_* directories found in {input_dir}")
        return False
    
    num_runs = len(run_dirs)
    print(f"Found {num_runs} capture runs")
    print(f"Extracting frame index {frame_idx} (frame {frame_idx + 1}) from each run")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Frame types to process
    frame_types = ["rgb", "depth", "seg", "albedo"]
    
    for frame_type in frame_types:
        frame_subdir = f"{frame_type}_frames"
        frame_filename = f"frame_{frame_idx:05d}.png"
        
        # Check if this frame type exists in first run
        first_frame = run_dirs[0] / frame_subdir / frame_filename
        if not first_frame.exists():
            if frame_type == "albedo":
                print(f"  Skipping {frame_type} (not captured)")
            else:
                print(f"  Warning: {frame_type} frames not found in {run_dirs[0]}")
            continue
        
        print(f"Assembling {frame_type} video...")
        
        # Create temporary directory for renamed frames
        temp_dir = output_dir / f"temp_{frame_type}"
        temp_dir.mkdir(exist_ok=True)
        
        # Copy/link frames with sequential numbering
        missing_count = 0
        for i, run_dir in enumerate(run_dirs):
            src_frame = run_dir / frame_subdir / frame_filename
            dst_frame = temp_dir / f"frame_{i:05d}.png"
            
            if src_frame.exists():
                # Create symlink for efficiency
                if dst_frame.exists():
                    dst_frame.unlink()
                dst_frame.symlink_to(src_frame.resolve())
            else:
                missing_count += 1
                print(f"    Warning: Missing {src_frame}")
        
        if missing_count > 0:
            print(f"    {missing_count} missing frames")
        
        # Assemble video with ffmpeg
        output_video = output_dir / f"{frame_type}_video.mp4"
        
        # Use lossless encoding for depth/seg to preserve exact values
        if frame_type in ["depth", "seg"]:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(temp_dir / "frame_%05d.png"),
                "-c:v", "libx264",
                "-crf", "0",  # Lossless
                "-preset", "fast",
                "-pix_fmt", "yuv444p",  # No chroma subsampling
                str(output_video)
            ]
        else:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(temp_dir / "frame_%05d.png"),
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                str(output_video)
            ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            print(f"  Created: {output_video}")
        except subprocess.CalledProcessError as e:
            print(f"  Error creating {frame_type} video: {e.stderr[:200]}")
        
        # Clean up temp directory
        for f in temp_dir.glob("*.png"):
            f.unlink()
        temp_dir.rmdir()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Assemble independent capture frames into videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input-dir", "-i", type=str, default="../temporal-stability",
                        help="Input directory with run_* subdirectories (default: ../temporal-stability)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for videos (default: INPUT_DIR/final)")
    parser.add_argument("--frame-idx", "-f", type=int, default=9,
                        help="Frame index to extract from each run (default: 9 = 10th frame)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video frame rate (default: 30)")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Run './run_independent_captures.sh' first to generate captures.")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "final"
    
    print(f"Independent Video Assembler v{__version__}")
    print("=" * 50)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Frame:  {args.frame_idx} (0-indexed)")
    print(f"FPS:    {args.fps}")
    print("=" * 50)
    print()
    
    success = assemble_video(input_dir, output_dir, args.frame_idx, args.fps)
    
    if success:
        print()
        print("=" * 50)
        print("Assembly complete!")
        print("=" * 50)
        print()
        print("Output videos:")
        for f in sorted(output_dir.glob("*_video.mp4")):
            print(f"  {f}")
        print()
        print("These videos have 93 frames, each showing 8 environments")
        print("at step 10 of independent simulations.")
        print()
        print("Next: Run Cosmos Transfer on these videos")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

