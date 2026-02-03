#!/usr/bin/env python3
"""
Test batch inference with real model and data.

This test loads the actual Cosmos model and runs batch inference
on real video inputs, with GPU monitoring to measure utilization.

Usage:
    cd /home/horde/cosmos/cosmos-transfer2.5
    source .venv/bin/activate
    export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
    python tests/test_batch_inference_real.py \
        --video1 /path/to/capture1/depth_video.mp4 \
        --video2 /path/to/capture2/depth_video.mp4 \
        --output-dir ./batch_test_output
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Try nvtx package first, fall back to torch.cuda.nvtx
try:
    import nvtx as _nvtx
    # Use nvtx.annotate() context manager API
    class nvtx:
        """Wrapper for nvtx package using annotate() API."""
        _ctx_stack = []
        
        @staticmethod
        def push_range(message="", color=None):
            ctx = _nvtx.annotate(message=message, color=color)
            ctx.__enter__()
            nvtx._ctx_stack.append(ctx)
        
        @staticmethod
        def pop_range():
            if nvtx._ctx_stack:
                ctx = nvtx._ctx_stack.pop()
                ctx.__exit__(None, None, None)
    print("Using nvtx package for profiling markers")
except ImportError:
    # Fall back to torch.cuda.nvtx
    class nvtx:
        """Wrapper for torch.cuda.nvtx."""
        @staticmethod
        def push_range(message="", color=None):
            torch.cuda.nvtx.range_push(message)
        
        @staticmethod
        def pop_range():
            torch.cuda.nvtx.range_pop()
    print("Using torch.cuda.nvtx for profiling markers")


# ============================================================================
# GPU Monitor (embedded to avoid cross-repo imports)
# ============================================================================

@dataclass
class GPUSample:
    """Single GPU measurement sample."""
    timestamp: float
    memory_used_mb: float
    memory_total_mb: float
    utilization_percent: float
    temperature_c: float | None = None


@dataclass
class GPUMetrics:
    """Aggregated GPU metrics from monitoring session."""
    samples: list[GPUSample] = field(default_factory=list)

    @property
    def peak_memory_mb(self) -> float:
        if not self.samples:
            return 0.0
        return max(s.memory_used_mb for s in self.samples)

    @property
    def avg_memory_mb(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.memory_used_mb for s in self.samples) / len(self.samples)

    @property
    def avg_utilization(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.utilization_percent for s in self.samples) / len(self.samples)

    @property
    def peak_utilization(self) -> float:
        if not self.samples:
            return 0.0
        return max(s.utilization_percent for s in self.samples)

    @property
    def min_utilization(self) -> float:
        if not self.samples:
            return 0.0
        return min(s.utilization_percent for s in self.samples)

    @property
    def duration_seconds(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": {
                "peak_mb": round(self.peak_memory_mb, 1),
                "avg_mb": round(self.avg_memory_mb, 1),
            },
            "utilization": {
                "peak_percent": round(self.peak_utilization, 1),
                "avg_percent": round(self.avg_utilization, 1),
                "min_percent": round(self.min_utilization, 1),
            },
            "duration_seconds": round(self.duration_seconds, 2),
            "num_samples": len(self.samples),
        }

    def get_samples_list(self) -> list[dict[str, Any]]:
        if not self.samples:
            return []
        start_time = self.samples[0].timestamp
        return [
            {
                "time_offset_s": round(s.timestamp - start_time, 3),
                "memory_mb": round(s.memory_used_mb, 1),
                "utilization_percent": round(s.utilization_percent, 1),
            }
            for s in self.samples
        ]


class GPUMonitor:
    """Monitor GPU metrics in a background thread."""

    def __init__(self, interval: float = 0.5, device_id: int = 0):
        self.interval = interval
        self.device_id = device_id
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._metrics = GPUMetrics()

    def start(self) -> None:
        self._stop_event.clear()
        self._metrics = GPUMetrics()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUMetrics:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._metrics

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_gpu()
            if sample:
                self._metrics.samples.append(sample)
            self._stop_event.wait(timeout=self.interval)

    def _sample_gpu(self) -> GPUSample | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode != 0:
                return None
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                return GPUSample(
                    timestamp=time.time(),
                    memory_used_mb=float(parts[0]),
                    memory_total_mb=float(parts[1]),
                    utilization_percent=float(parts[2]),
                    temperature_c=float(parts[3]) if len(parts) > 3 else None,
                )
        except Exception:
            pass
        return None


# ============================================================================
# Test Functions
# ============================================================================

def test_batch_inference_real(
    video1_path: str,
    video2_path: str,
    output_dir: str,
    prompt: str = "A robotic arm manipulating objects on a table in an industrial setting.",
    num_steps: int = 4,
    guidance: int = 7,
    state_t: int = 24,
    use_cuda_graphs: bool = False,
):
    """
    Run batch inference with 2 real videos and GPU monitoring.
    
    Args:
        video1_path: Path to first depth control video
        video2_path: Path to second depth control video
        output_dir: Output directory for results
        prompt: Text prompt for generation
        num_steps: Number of diffusion steps
        guidance: Guidance scale
        state_t: Latent temporal frames. Controls video length:
                 - state_t=2 → 5 pixel frames
                 - state_t=4 → 13 pixel frames
                 - state_t=7 → 25 pixel frames
                 - state_t=24 → 93 pixel frames (default)
    """
    # Calculate expected pixel frames from state_t
    expected_pixel_frames = (state_t - 1) * 4 + 1
    
    print("=" * 60)
    print("Cosmos Batch Inference - Real Model Test (with GPU Monitoring)")
    print("=" * 60)
    
    # Check inputs exist
    video1 = Path(video1_path)
    video2 = Path(video2_path)
    
    if not video1.exists():
        print(f"Error: Video 1 not found: {video1}")
        return False
    if not video2.exists():
        print(f"Error: Video 2 not found: {video2}")
        return False
    
    print(f"Video 1: {video1}")
    print(f"Video 2: {video2}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {num_steps}, Guidance: {guidance}")
    print(f"State_t: {state_t} (expecting {expected_pixel_frames} pixel frames)")
    print()
    
    # Import cosmos modules
    print("Loading Cosmos modules...")
    from cosmos_transfer2.config import InferenceArguments, SetupArguments, DepthConfig
    from cosmos_transfer2.inference import Control2WorldInference
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create setup args
    setup_args = SetupArguments(
        model="depth",
        output_dir=output_path,
        disable_guardrails=True,
    )
    
    # Create sample configs
    sample1 = InferenceArguments(
        name="batch_sample_1",
        video_path=video1,
        prompt=prompt,
        guidance=guidance,
        num_steps=num_steps,
        seed=42,
        depth=DepthConfig(control_path=video1),
        num_video_frames_per_chunk=expected_pixel_frames,
    )
    
    sample2 = InferenceArguments(
        name="batch_sample_2", 
        video_path=video2,
        prompt=prompt,
        guidance=guidance,
        num_steps=num_steps,
        seed=142,
        depth=DepthConfig(control_path=video2),
        num_video_frames_per_chunk=expected_pixel_frames,
    )
    
    # Initialize inference
    print("\nInitializing model (this may take a while)...")
    batch_hint_keys = ["depth"]
    
    start_init = time.time()
    inference = Control2WorldInference(
        setup_args, 
        batch_hint_keys=batch_hint_keys,
        state_t=state_t,
        use_cuda_graphs=use_cuda_graphs,
    )
    init_time = time.time() - start_init
    cuda_graphs_str = " with CUDA Graphs" if use_cuda_graphs else ""
    print(f"Model loaded in {init_time:.1f}s (state_t={state_t}){cuda_graphs_str}")
    
    # ========== WARMUP RUN ==========
    # Run a single sample first to warm up JIT compilation, CUDA kernels, etc.
    # This ensures neither batch nor sequential is penalized by warmup overhead.
    print("\n" + "=" * 60)
    print("Running WARMUP inference (1 sample to warm JIT/kernels)...")
    print("=" * 60)
    
    warmup_sample = InferenceArguments(
        name="warmup",
        video_path=video1,
        prompt=prompt,
        guidance=guidance,
        num_steps=num_steps,
        seed=999,
        depth=DepthConfig(control_path=video1),
        num_video_frames_per_chunk=expected_pixel_frames,
    )
    
    nvtx.push_range("=== WARMUP (exclude from analysis) ===", color=0x808080)  # Gray
    warmup_start = time.time()
    warmup_output = inference.generate(
        samples=[warmup_sample],
        output_dir=output_path / "warmup",
    )
    warmup_time = time.time() - warmup_start
    nvtx.pop_range()
    print(f"  Warmup complete in {warmup_time:.1f}s (excluded from comparisons)")
    
    # Clear CUDA cache after warmup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # ========== BATCH INFERENCE ==========
    print("\n" + "=" * 60)
    print("Running BATCH inference (2 videos in parallel)...")
    print("=" * 60)
    
    torch.cuda.reset_peak_memory_stats()
    gpu_monitor_batch = GPUMonitor(interval=0.5)
    gpu_monitor_batch.start()
    
    nvtx.push_range("=== BATCH INFERENCE ===", color=0x00FF00)  # Green
    start_batch = time.time()
    batch_outputs = inference.generate_batch(
        samples=[sample1, sample2],
        output_dir=output_path,
        batch_size=2,
    )
    batch_time = time.time() - start_batch
    nvtx.pop_range()
    
    batch_gpu_metrics = gpu_monitor_batch.stop()
    batch_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\nBatch inference complete!")
    print(f"  Time: {batch_time:.1f}s for 2 videos")
    print(f"  Per-video: {batch_time/2:.1f}s")
    print(f"  Peak GPU memory (torch): {batch_memory:.1f} GB")
    print(f"  GPU Utilization: avg={batch_gpu_metrics.avg_utilization:.1f}%, "
          f"peak={batch_gpu_metrics.peak_utilization:.1f}%, "
          f"min={batch_gpu_metrics.min_utilization:.1f}%")
    print(f"  Samples collected: {len(batch_gpu_metrics.samples)}")
    print(f"  Outputs: {batch_outputs}")
    
    # ========== SEQUENTIAL INFERENCE ==========
    print("\n" + "=" * 60)
    print("Running SEQUENTIAL inference (for comparison)...")
    print("=" * 60)
    
    torch.cuda.reset_peak_memory_stats()
    gpu_monitor_seq = GPUMonitor(interval=0.5)
    gpu_monitor_seq.start()
    
    nvtx.push_range("=== SEQUENTIAL INFERENCE ===", color=0xFF0000)  # Red
    start_seq = time.time()
    seq_outputs = inference.generate(
        samples=[sample1, sample2],
        output_dir=output_path / "sequential",
    )
    seq_time = time.time() - start_seq
    nvtx.pop_range()
    
    seq_gpu_metrics = gpu_monitor_seq.stop()
    seq_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\nSequential inference complete!")
    print(f"  Time: {seq_time:.1f}s for 2 videos")
    print(f"  Per-video: {seq_time/2:.1f}s")
    print(f"  Peak GPU memory (torch): {seq_memory:.1f} GB")
    print(f"  GPU Utilization: avg={seq_gpu_metrics.avg_utilization:.1f}%, "
          f"peak={seq_gpu_metrics.peak_utilization:.1f}%, "
          f"min={seq_gpu_metrics.min_utilization:.1f}%")
    print(f"  Samples collected: {len(seq_gpu_metrics.samples)}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("SUMMARY (after warmup)")
    print("=" * 60)
    print(f"Warmup time (excluded): {warmup_time:.1f}s")
    print()
    print(f"{'Metric':<25} {'Batch':>15} {'Sequential':>15}")
    print("-" * 55)
    print(f"{'Total time (s)':<25} {batch_time:>15.1f} {seq_time:>15.1f}")
    print(f"{'Per-video time (s)':<25} {batch_time/2:>15.1f} {seq_time/2:>15.1f}")
    print(f"{'Peak memory (GB)':<25} {batch_memory:>15.1f} {seq_memory:>15.1f}")
    print(f"{'Avg GPU util (%)':<25} {batch_gpu_metrics.avg_utilization:>15.1f} {seq_gpu_metrics.avg_utilization:>15.1f}")
    print(f"{'Peak GPU util (%)':<25} {batch_gpu_metrics.peak_utilization:>15.1f} {seq_gpu_metrics.peak_utilization:>15.1f}")
    print(f"{'Min GPU util (%)':<25} {batch_gpu_metrics.min_utilization:>15.1f} {seq_gpu_metrics.min_utilization:>15.1f}")
    print("-" * 55)
    print(f"{'Speedup':<25} {seq_time/batch_time:>15.2f}x")
    
    # Save detailed metrics to JSON
    metrics_file = output_path / "gpu_metrics.json"
    metrics_data = {
        "warmup_seconds": round(warmup_time, 2),
        "batch": {
            "time_seconds": round(batch_time, 2),
            "memory_gb": round(batch_memory, 2),
            "gpu_metrics": batch_gpu_metrics.to_dict(),
            "gpu_samples": batch_gpu_metrics.get_samples_list(),
        },
        "sequential": {
            "time_seconds": round(seq_time, 2),
            "memory_gb": round(seq_memory, 2),
            "gpu_metrics": seq_gpu_metrics.to_dict(),
            "gpu_samples": seq_gpu_metrics.get_samples_list(),
        },
        "speedup": round(seq_time / batch_time, 3),
        "config": {
            "num_steps": num_steps,
            "guidance": guidance,
            "prompt": prompt,
            "state_t": state_t,
            "expected_pixel_frames": expected_pixel_frames,
        }
    }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nDetailed metrics saved to: {metrics_file}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test batch inference with real data")
    parser.add_argument("--video1", required=True, help="Path to first video")
    parser.add_argument("--video2", required=True, help="Path to second video")
    parser.add_argument("--output-dir", default="./batch_test_output", help="Output directory")
    parser.add_argument("--prompt", default="A robotic arm manipulating objects on a table.", help="Prompt")
    parser.add_argument("--num-steps", type=int, default=4, help="Diffusion steps")
    parser.add_argument("--guidance", type=int, default=7, help="Guidance scale")
    parser.add_argument("--state-t", type=int, default=24, 
                        help="Latent temporal frames (2=5 pixel, 4=13 pixel, 7=25 pixel, 24=93 pixel)")
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="Enable CUDA Graphs to reduce kernel launch overhead")
    
    args = parser.parse_args()
    
    success = test_batch_inference_real(
        video1_path=args.video1,
        video2_path=args.video2,
        output_dir=args.output_dir,
        prompt=args.prompt,
        num_steps=args.num_steps,
        guidance=args.guidance,
        state_t=args.state_t,
        use_cuda_graphs=args.cuda_graphs,
    )
    
    sys.exit(0 if success else 1)
