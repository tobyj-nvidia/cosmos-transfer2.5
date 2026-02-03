#!/usr/bin/env python3
"""
Extract rolling N-frame windows from a video.

Creates a set of videos, each containing N consecutive frames with a stride of 1.
For example, with N=5 and a 93-frame video:
  - window_000.mp4: frames [0, 1, 2, 3, 4]
  - window_001.mp4: frames [1, 2, 3, 4, 5]
  - window_002.mp4: frames [2, 3, 4, 5, 6]
  - ...
  - window_088.mp4: frames [88, 89, 90, 91, 92]

This is useful for testing short temporal windows with Cosmos Transfer.

Usage:
    python scripts/extract_rolling_frames.py input.mp4 output_dir/ --num-frames 5
    
    # Or with custom stride:
    python scripts/extract_rolling_frames.py input.mp4 output_dir/ --num-frames 5 --stride 2
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_video_frame_count(video_path: str) -> int:
    """Get the total number of frames in a video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def get_video_fps(video_path: str) -> float:
    """Get the frame rate of a video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    fps_str = result.stdout.strip()
    # Parse fraction like "30/1" or "30000/1001"
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str)


def extract_frame_window(
    input_path: str,
    output_path: str,
    start_frame: int,
    num_frames: int,
    fps: float,
) -> bool:
    """
    Extract a window of frames from a video.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        start_frame: Starting frame index (0-based)
        num_frames: Number of frames to extract
        fps: Frame rate for output video
        
    Returns:
        True if successful
    """
    # Use select filter to get exact frames
    # eq(n,start) selects frame at index start
    # between(n,start,end) selects frames from start to end inclusive
    end_frame = start_frame + num_frames - 1
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", input_path,
        "-vf", f"select='between(n,{start_frame},{end_frame})',setpts=N/FRAME_RATE/TB",
        "-fps_mode", "vfr",
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", "18",
        "-an",  # No audio
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract rolling N-frame windows from a video"
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output_dir", help="Output directory for frame windows")
    parser.add_argument(
        "--num-frames", "-n", type=int, default=5,
        help="Number of frames per window (default: 5)"
    )
    parser.add_argument(
        "--stride", "-s", type=int, default=1,
        help="Stride between windows (default: 1)"
    )
    parser.add_argument(
        "--max-windows", "-m", type=int, default=None,
        help="Maximum number of windows to extract (default: all)"
    )
    parser.add_argument(
        "--prefix", "-p", default="window",
        help="Output filename prefix (default: 'window')"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"Error: Input video not found: {input_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    print(f"Analyzing video: {input_path}")
    total_frames = get_video_frame_count(str(input_path))
    fps = get_video_fps(str(input_path))
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Window size: {args.num_frames} frames")
    print(f"  Stride: {args.stride}")
    
    # Calculate number of windows
    num_windows = (total_frames - args.num_frames) // args.stride + 1
    if args.max_windows:
        num_windows = min(num_windows, args.max_windows)
    
    print(f"  Windows to extract: {num_windows}")
    print()
    
    # Extract windows
    for i in range(num_windows):
        start_frame = i * args.stride
        output_path = output_dir / f"{args.prefix}_{i:03d}.mp4"
        
        print(f"Extracting window {i+1}/{num_windows}: frames [{start_frame}..{start_frame + args.num_frames - 1}] -> {output_path.name}")
        
        success = extract_frame_window(
            str(input_path),
            str(output_path),
            start_frame,
            args.num_frames,
            fps,
        )
        
        if not success:
            print(f"  ERROR: Failed to extract window {i}")
            sys.exit(1)
    
    print()
    print(f"Done! Extracted {num_windows} windows to {output_dir}/")
    print(f"Each window contains {args.num_frames} frames at {fps:.2f} fps")
    
    # Print summary of files
    print()
    print("Files created:")
    for f in sorted(output_dir.glob(f"{args.prefix}_*.mp4"))[:5]:
        print(f"  {f.name}")
    if num_windows > 5:
        print(f"  ... ({num_windows - 5} more)")


if __name__ == "__main__":
    main()

