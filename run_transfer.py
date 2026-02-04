#!/usr/bin/env python3
"""
Cosmos Transfer 2.5 - Simple inference wrapper with timing and GPU monitoring.

Usage:
    # Single control (depth)
    python run_transfer.py /path/to/depth.mp4 "A robot arm" --control depth --use-input-as-control
    
    # Multi-control (depth + segmentation)
    python run_transfer.py /path/to/depth.mp4 "A robot arm" --control depth --use-input-as-control \
        --seg-video /path/to/seg.mp4
    
    # With style reference image
    python run_transfer.py /path/to/depth.mp4 "A robot arm" --control depth --use-input-as-control \
        --seg-video /path/to/seg.mp4 --style-image /path/to/reference.png
"""

__version__ = "0.5.1"

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


class GPUMonitor:
    """Background GPU monitoring using nvidia-smi."""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
    
    def _sample_gpu(self) -> dict | None:
        """Get current GPU stats from nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 5:
                    return {
                        "timestamp": time.perf_counter(),
                        "memory_used_mb": float(parts[0]),
                        "memory_total_mb": float(parts[1]),
                        "gpu_utilization_pct": float(parts[2]),
                        "power_draw_w": float(parts[3]),
                        "temperature_c": float(parts[4]),
                    }
        except Exception:
            pass
        return None
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            sample = self._sample_gpu()
            if sample:
                self.samples.append(sample)
            self._stop_event.wait(self.interval)
    
    def start(self):
        """Start monitoring."""
        self._stop_event.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> dict:
        """Stop monitoring and return statistics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        
        if not self.samples:
            return {}
        
        # Compute statistics
        memory_used = [s["memory_used_mb"] for s in self.samples]
        gpu_util = [s["gpu_utilization_pct"] for s in self.samples]
        power = [s["power_draw_w"] for s in self.samples]
        temp = [s["temperature_c"] for s in self.samples]
        
        return {
            "num_samples": len(self.samples),
            "sample_interval_sec": self.interval,
            "memory_mb": {
                "min": min(memory_used),
                "max": max(memory_used),
                "avg": sum(memory_used) / len(memory_used),
                "total": self.samples[0]["memory_total_mb"],
            },
            "gpu_utilization_pct": {
                "min": min(gpu_util),
                "max": max(gpu_util),
                "avg": sum(gpu_util) / len(gpu_util),
            },
            "power_watts": {
                "min": min(power),
                "max": max(power),
                "avg": sum(power) / len(power),
            },
            "temperature_c": {
                "min": min(temp),
                "max": max(temp),
                "avg": sum(temp) / len(temp),
            },
        }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.2f}s"


def format_memory(mb: float) -> str:
    """Format memory in GB."""
    return f"{mb / 1024:.1f} GB"


def main():
    parser = argparse.ArgumentParser(description="Run Cosmos Transfer 2.5 inference")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("prompt", type=str, help="Text prompt describing desired output")
    parser.add_argument("--control", type=str, default="depth", 
                        choices=["depth", "edge", "seg", "vis"],
                        help="Control modality (default: depth)")
    parser.add_argument("--control-weight", type=float, default=1.0,
                        help="Control weight (default: 1.0)")
    parser.add_argument("--guidance", type=float, default=3.0,
                        help="Guidance scale (default: 3.0)")
    parser.add_argument("--control-video", type=str, default=None,
                        help="Path to control video (optional, auto-generated if not provided)")
    parser.add_argument("--use-input-as-control", action="store_true",
                        help="Use the input video directly as the control video (skip extraction)")
    parser.add_argument("--seg-video", type=str, default=None,
                        help="Path to segmentation video for multi-control (depth+seg)")
    parser.add_argument("--seg-weight", type=float, default=1.0,
                        help="Segmentation control weight for multi-control (default: 1.0)")
    parser.add_argument("--style-image", type=str, default=None,
                        help="Path to RGB image to use as visual style reference")
    parser.add_argument("--num-steps", type=int, default=35,
                        help="Number of diffusion sampling steps (default: 35, lower=faster)")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5,
                        help="GPU sampling interval in seconds (default: 0.5)")
    args = parser.parse_args()

    # Resolve paths
    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # Create output directory next to input video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = video_path.parent / f"cosmos_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create spec file
    spec = {
        "name": video_path.stem,
        "prompt": args.prompt,
        "video_path": str(video_path),
        "guidance": args.guidance,
        "num_steps": args.num_steps,
        args.control: {
            "control_weight": args.control_weight
        }
    }
    
    # Add control video path if provided
    if args.use_input_as_control:
        # Use the input video directly as the control (skip extraction)
        spec[args.control]["control_path"] = str(video_path)
    elif args.control_video:
        control_path = Path(args.control_video).resolve()
        if not control_path.exists():
            print(f"Error: Control video not found: {control_path}")
            sys.exit(1)
        spec[args.control]["control_path"] = str(control_path)

    # Add segmentation video for multi-control (depth + seg)
    if args.seg_video:
        seg_path = Path(args.seg_video).resolve()
        if not seg_path.exists():
            print(f"Error: Segmentation video not found: {seg_path}")
            sys.exit(1)
        spec["seg"] = {
            "control_path": str(seg_path),
            "control_weight": args.seg_weight
        }

    # Add style reference image if provided
    if args.style_image:
        style_path = Path(args.style_image).resolve()
        if not style_path.exists():
            print(f"Error: Style image not found: {style_path}")
            sys.exit(1)
        spec["image_context_path"] = str(style_path)

    spec_path = output_dir / "inference_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    # Print configuration
    print("=" * 60)
    print(f"Cosmos Transfer 2.5 Inference (run_transfer.py v{__version__})")
    print("=" * 60)
    print(f"Input video:    {video_path}")
    print(f"Prompt:         {args.prompt}")
    print(f"Control:        {args.control} (weight={args.control_weight})")
    if args.use_input_as_control:
        print(f"Control video:  (using input video directly)")
    elif args.control_video:
        print(f"Control video:  {args.control_video}")
    else:
        print(f"Control video:  (will auto-extract from input)")
    if args.seg_video:
        print(f"Seg video:      {args.seg_video} (weight={args.seg_weight})")
        print(f"Mode:           MULTI-CONTROL (depth + seg)")
    if args.style_image:
        print(f"Style image:    {args.style_image}")
    print(f"Guidance:       {args.guidance}")
    print(f"Num steps:      {args.num_steps}")
    print(f"Output dir:     {output_dir}")
    print("=" * 60)

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(interval=args.gpu_sample_interval)

    # Timing dictionary
    timings = {
        "script_version": __version__,
        "start_time": datetime.now().isoformat(),
        "input_video": str(video_path),
        "prompt": args.prompt,
        "control": args.control,
        "control_weight": args.control_weight,
        "seg_video": args.seg_video,
        "seg_weight": args.seg_weight if args.seg_video else None,
        "multi_control": args.seg_video is not None,
        "guidance": args.guidance,
        "num_steps": args.num_steps,
        "use_input_as_control": args.use_input_as_control,
        "style_image": args.style_image,
    }

    # Start GPU monitoring
    gpu_monitor.start()

    # Import and run inference
    total_start = time.perf_counter()

    print("\n[1/3] Loading model...")
    load_start = time.perf_counter()
    
    from cosmos_oss.init import cleanup_environment, init_environment
    init_environment()
    
    from cosmos_transfer2.config import InferenceArguments, SetupArguments
    from cosmos_transfer2.inference import Control2WorldInference

    # Model name: always use the primary control type
    # Multi-control is handled via batch_hint_keys from the spec file
    setup = SetupArguments(output_dir=str(output_dir), model=args.control)
    inference_samples, batch_hint_keys = InferenceArguments.from_files([spec_path])
    inference = Control2WorldInference(setup, batch_hint_keys=batch_hint_keys)
    
    load_time = time.perf_counter() - load_start
    timings["model_load_time"] = load_time
    print(f"    Model loaded in {format_duration(load_time)}")

    print("\n[2/3] Generating video...")
    gen_start = time.perf_counter()
    
    inference.generate(inference_samples, output_dir=output_dir)
    
    gen_time = time.perf_counter() - gen_start
    timings["generation_time"] = gen_time
    print(f"    Generation completed in {format_duration(gen_time)}")

    print("\n[3/3] Cleaning up...")
    cleanup_start = time.perf_counter()
    cleanup_environment()
    cleanup_time = time.perf_counter() - cleanup_start
    timings["cleanup_time"] = cleanup_time

    total_time = time.perf_counter() - total_start
    timings["total_time"] = total_time
    timings["end_time"] = datetime.now().isoformat()

    # Stop GPU monitoring and get stats
    gpu_stats = gpu_monitor.stop()
    timings["gpu_stats"] = gpu_stats

    # Find output video
    output_videos = list(output_dir.glob("*.mp4"))
    if output_videos:
        timings["output_video"] = str(output_videos[0])

    # Save timing report
    timing_path = output_dir / "timing_report.json"
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"Model Load:     {format_duration(load_time):>12}")
    print(f"Generation:     {format_duration(gen_time):>12}")
    print(f"Cleanup:        {format_duration(cleanup_time):>12}")
    print("-" * 60)
    print(f"TOTAL:          {format_duration(total_time):>12}")
    print("=" * 60)
    
    # Print GPU stats
    if gpu_stats:
        print("\nGPU STATISTICS")
        print("=" * 60)
        mem = gpu_stats["memory_mb"]
        print(f"Memory Used:    {format_memory(mem['avg']):>8} avg, {format_memory(mem['max']):>8} max / {format_memory(mem['total'])} total")
        util = gpu_stats["gpu_utilization_pct"]
        print(f"GPU Util:       {util['avg']:>7.1f}% avg, {util['max']:>7.1f}% max")
        pwr = gpu_stats["power_watts"]
        print(f"Power Draw:     {pwr['avg']:>7.1f}W avg, {pwr['max']:>7.1f}W max")
        temp = gpu_stats["temperature_c"]
        print(f"Temperature:    {temp['avg']:>7.1f}°C avg, {temp['max']:>7.1f}°C max")
        print(f"Samples:        {gpu_stats['num_samples']} @ {gpu_stats['sample_interval_sec']}s interval")
        print("=" * 60)
    
    if output_videos:
        print(f"\nOutput video: {output_videos[0]}")
    print(f"Timing report: {timing_path}")


if __name__ == "__main__":
    main()
