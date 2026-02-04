#!/usr/bin/env python3
"""
Overlay depth video on top of a generated RGB video with alpha blending.

Useful for comparing how well Cosmos followed the depth control input.

Usage:
    python overlay_depth.py <rgb_video> <depth_video> [--alpha 0.3] [--output overlay.mp4]

Examples:
    # Basic overlay with 30% depth opacity
    python overlay_depth.py cosmos_output.mp4 depth_video.mp4

    # Custom alpha (50% depth visibility)
    python overlay_depth.py cosmos_output.mp4 depth_video.mp4 --alpha 0.5

    # Side-by-side comparison instead of overlay
    python overlay_depth.py cosmos_output.mp4 depth_video.mp4 --mode side-by-side

    # Overlay segmentation instead
    python overlay_depth.py cosmos_output.mp4 seg_video.mp4 --alpha 0.4 --colorize
"""

__version__ = "0.1.0"

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_video_frames(video_path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    """Load all frames from a video file.
    
    Returns:
        frames: List of RGB numpy arrays
        fps: Frames per second
        size: (width, height)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames, fps, (width, height)


def colorize_depth(depth_frame: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """Apply a colormap to a grayscale depth frame.
    
    Args:
        depth_frame: RGB frame where R=G=B (grayscale depth)
        colormap: OpenCV colormap name ('turbo', 'jet', 'viridis', 'magma')
    
    Returns:
        Colorized RGB frame
    """
    # Convert to grayscale if needed
    if depth_frame.ndim == 3:
        gray = depth_frame[:, :, 0]  # R channel (they're all the same)
    else:
        gray = depth_frame
    
    # Apply colormap
    colormap_cv = {
        'turbo': cv2.COLORMAP_TURBO,
        'jet': cv2.COLORMAP_JET,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'magma': cv2.COLORMAP_MAGMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'plasma': cv2.COLORMAP_PLASMA,
    }.get(colormap, cv2.COLORMAP_TURBO)
    
    colored = cv2.applyColorMap(gray, colormap_cv)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def blend_frames(rgb_frame: np.ndarray, overlay_frame: np.ndarray, alpha: float) -> np.ndarray:
    """Blend two frames with alpha.
    
    Args:
        rgb_frame: Base RGB frame
        overlay_frame: Overlay RGB frame  
        alpha: Overlay opacity (0.0 = invisible, 1.0 = fully opaque)
    
    Returns:
        Blended RGB frame
    """
    # Resize overlay to match rgb if needed
    if overlay_frame.shape[:2] != rgb_frame.shape[:2]:
        overlay_frame = cv2.resize(overlay_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))
    
    # Alpha blend: result = rgb * (1 - alpha) + overlay * alpha
    blended = (rgb_frame.astype(np.float32) * (1 - alpha) + 
               overlay_frame.astype(np.float32) * alpha)
    return blended.clip(0, 255).astype(np.uint8)


def side_by_side(rgb_frame: np.ndarray, overlay_frame: np.ndarray, gap: int = 4) -> np.ndarray:
    """Create side-by-side comparison.
    
    Args:
        rgb_frame: Left frame (generated RGB)
        overlay_frame: Right frame (depth/seg)
        gap: Pixel gap between frames
    
    Returns:
        Combined frame with both side by side
    """
    # Resize overlay to match rgb height
    if overlay_frame.shape[0] != rgb_frame.shape[0]:
        scale = rgb_frame.shape[0] / overlay_frame.shape[0]
        new_width = int(overlay_frame.shape[1] * scale)
        overlay_frame = cv2.resize(overlay_frame, (new_width, rgb_frame.shape[0]))
    
    # Create gap
    gap_strip = np.zeros((rgb_frame.shape[0], gap, 3), dtype=np.uint8)
    
    # Concatenate
    combined = np.concatenate([rgb_frame, gap_strip, overlay_frame], axis=1)
    return combined


def save_video(frames: list[np.ndarray], output_path: Path, fps: float):
    """Save frames as MP4 video using ffmpeg for quality.
    
    Args:
        frames: List of RGB numpy arrays
        output_path: Output video path
        fps: Frames per second
    """
    if not frames:
        print("No frames to save!")
        return
    
    height, width = frames[0].shape[:2]
    
    # Save frames to temp directory, then use ffmpeg
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        print(f"    Saving {len(frames)} frames...")
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame, mode='RGB')
            img.save(tmpdir / f"frame_{i:05d}.png")
        
        print(f"    Encoding video...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-crf", "18",  # Good quality
            "-pix_fmt", "yuv420p",  # Wide compatibility
            "-preset", "medium",
            str(output_path)
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"    Saved: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"    ffmpeg error: {e.stderr.decode()}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Overlay depth/segmentation video on generated RGB video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("rgb_video", type=str,
                        help="Path to generated RGB video (from Cosmos)")
    parser.add_argument("overlay_video", type=str,
                        help="Path to depth or segmentation video to overlay")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Overlay opacity (0.0-1.0, default: 0.6)")
    parser.add_argument("--mode", type=str, default="overlay",
                        choices=["overlay", "side-by-side", "both"],
                        help="Comparison mode (default: overlay)")
    parser.add_argument("--colorize", action="store_true",
                        help="Apply colormap to grayscale depth (turbo by default)")
    parser.add_argument("--colormap", type=str, default="turbo",
                        choices=["turbo", "jet", "viridis", "magma", "inferno", "plasma"],
                        help="Colormap for depth (default: turbo)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output video path (default: auto-generated)")
    
    args = parser.parse_args()
    
    print(f"Depth/Seg Overlay Tool v{__version__}")
    print("=" * 50)
    
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
        suffix = f"_{args.mode.replace('-', '')}_a{int(args.alpha*100)}"
        output_path = rgb_path.parent / f"{rgb_path.stem}{suffix}.mp4"
    
    print(f"RGB Video:     {rgb_path}")
    print(f"Overlay Video: {overlay_path}")
    print(f"Mode:          {args.mode}")
    print(f"Alpha:         {args.alpha}")
    print(f"Colorize:      {args.colorize}")
    print(f"Output:        {output_path}")
    print()
    
    # Load videos
    print("[1/3] Loading videos...")
    rgb_frames, rgb_fps, rgb_size = load_video_frames(rgb_path)
    overlay_frames, overlay_fps, overlay_size = load_video_frames(overlay_path)
    
    print(f"    RGB: {len(rgb_frames)} frames, {rgb_size[0]}x{rgb_size[1]}, {rgb_fps:.1f} fps")
    print(f"    Overlay: {len(overlay_frames)} frames, {overlay_size[0]}x{overlay_size[1]}, {overlay_fps:.1f} fps")
    
    # Match frame counts (use minimum)
    num_frames = min(len(rgb_frames), len(overlay_frames))
    if len(rgb_frames) != len(overlay_frames):
        print(f"    Warning: Frame count mismatch, using first {num_frames} frames")
    
    rgb_frames = rgb_frames[:num_frames]
    overlay_frames = overlay_frames[:num_frames]
    
    # Process frames
    print(f"\n[2/3] Processing {num_frames} frames...")
    
    output_frames = []
    for i, (rgb_frame, overlay_frame) in enumerate(zip(rgb_frames, overlay_frames)):
        # Optionally colorize depth
        if args.colorize:
            overlay_frame = colorize_depth(overlay_frame, args.colormap)
        
        if args.mode == "overlay":
            result = blend_frames(rgb_frame, overlay_frame, args.alpha)
        elif args.mode == "side-by-side":
            result = side_by_side(rgb_frame, overlay_frame)
        elif args.mode == "both":
            # Create 2x2 grid: [rgb, overlay] / [blended, side-by-side]
            blended = blend_frames(rgb_frame, overlay_frame, args.alpha)
            sbs = side_by_side(rgb_frame, overlay_frame, gap=2)
            
            # Scale to fit
            target_width = max(rgb_frame.shape[1] * 2, sbs.shape[1])
            
            # Top row: rgb and overlay side by side
            top_row = side_by_side(rgb_frame, overlay_frame, gap=2)
            # Bottom: blended (scaled to match width)
            blended_scaled = cv2.resize(blended, (target_width, blended.shape[0]))
            
            # Stack vertically
            gap_strip = np.zeros((4, target_width, 3), dtype=np.uint8)
            result = np.concatenate([
                cv2.resize(top_row, (target_width, top_row.shape[0])),
                gap_strip,
                blended_scaled
            ], axis=0)
        
        output_frames.append(result)
        
        if (i + 1) % 20 == 0 or i == num_frames - 1:
            print(f"    Processed {i + 1}/{num_frames} frames")
    
    # Save output
    print(f"\n[3/3] Saving video...")
    save_video(output_frames, output_path, rgb_fps)
    
    print()
    print("=" * 50)
    print(f"Done! Output saved to: {output_path}")


if __name__ == "__main__":
    main()

