#!/bin/bash
# Profile Cosmos batch inference with Nsight Systems
#
# This script captures detailed GPU profiles for both batch and sequential
# inference to identify bottlenecks and confirm compute-bound hypothesis.
#
# Usage:
#   cd /home/horde/cosmos/cosmos-transfer2.5
#   ./tests/profile_batch_inference.sh
#
# Prerequisites:
#   - nsys installed (nsight-systems package)
#   - Video inputs from previous batch test
#   - ~10 minutes per profile (batch + sequential)

set -e

# Configuration
COSMOS_DIR="${COSMOS_DIR:-/home/horde/cosmos/cosmos-transfer2.5}"
VIDEO1="${VIDEO1:-/home/horde/isaaclab-cosmos-experiments/results/batch_test_20260130_180547/captures/video_1/depth_video.mp4}"
VIDEO2="${VIDEO2:-/home/horde/isaaclab-cosmos-experiments/results/batch_test_20260130_180547/captures/video_2/depth_video.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-./nsight_profiles_$(date +%Y%m%d_%H%M%S)}"
NUM_STEPS="${NUM_STEPS:-4}"

echo "============================================================"
echo "Cosmos Batch Inference - Nsight Profiling"
echo "============================================================"
echo "COSMOS_DIR: $COSMOS_DIR"
echo "VIDEO1: $VIDEO1"
echo "VIDEO2: $VIDEO2"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NUM_STEPS: $NUM_STEPS"
echo ""

# Check prerequisites
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys not found. Install with:"
    echo "  wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_1/nsight-systems-2026.1.1_2026.1.1.204-1_amd64.deb"
    echo "  sudo dpkg -i nsight-systems-*.deb"
    exit 1
fi

if [ ! -f "$VIDEO1" ]; then
    echo "ERROR: Video 1 not found: $VIDEO1"
    exit 1
fi

if [ ! -f "$VIDEO2" ]; then
    echo "ERROR: Video 2 not found: $VIDEO2"
    exit 1
fi

# Setup environment
cd "$COSMOS_DIR"
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# Enable perf events (may require sudo)
echo "Enabling performance counters (may need sudo password)..."
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid' 2>/dev/null || echo "Warning: Could not set perf_event_paranoid"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "Profile 1: BATCH Inference (2 videos in parallel)"
echo "============================================================"

nsys profile \
    --output="$OUTPUT_DIR/cosmos_batch_inference" \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --gpu-metrics-device all \
    --gpu-metrics-frequency 10000 \
    --stats=true \
    --force-overwrite=true \
    python tests/test_batch_inference_real.py \
        --video1 "$VIDEO1" \
        --video2 "$VIDEO2" \
        --output-dir "$OUTPUT_DIR/batch_output" \
        --num-steps "$NUM_STEPS"

echo ""
echo "============================================================"
echo "Profile 2: SEQUENTIAL Inference (baseline)"
echo "============================================================"

# Create a simpler sequential-only test script
cat > "$OUTPUT_DIR/test_sequential_only.py" << 'PYEOF'
#!/usr/bin/env python3
"""Sequential-only inference for profiling comparison."""
import sys
import time
from pathlib import Path

import torch

def main():
    video1 = sys.argv[1]
    video2 = sys.argv[2]
    output_dir = sys.argv[3]
    num_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    
    print(f"Loading Cosmos modules...")
    from cosmos_transfer2.config import InferenceArguments, SetupArguments, DepthConfig
    from cosmos_transfer2.inference import Control2WorldInference
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    setup_args = SetupArguments(
        model="depth",
        output_dir=output_path,
        disable_guardrails=True,
    )
    
    sample1 = InferenceArguments(
        name="seq_sample_1",
        video_path=Path(video1),
        prompt="A robotic arm manipulating objects on a table.",
        guidance=7,
        num_steps=num_steps,
        seed=42,
        depth=DepthConfig(control_path=Path(video1)),
    )
    
    sample2 = InferenceArguments(
        name="seq_sample_2",
        video_path=Path(video2),
        prompt="A robotic arm manipulating objects on a table.",
        guidance=7,
        num_steps=num_steps,
        seed=142,
        depth=DepthConfig(control_path=Path(video2)),
    )
    
    print("Initializing model...")
    inference = Control2WorldInference(setup_args, batch_hint_keys=["depth"])
    
    print("Running SEQUENTIAL inference...")
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    
    # Process one at a time (standard sequential)
    inference.generate(samples=[sample1, sample2], output_dir=output_path)
    
    elapsed = time.time() - start
    memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Sequential complete: {elapsed:.1f}s, {memory:.1f} GB peak")

if __name__ == "__main__":
    main()
PYEOF

nsys profile \
    --output="$OUTPUT_DIR/cosmos_sequential_inference" \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --gpu-metrics-device all \
    --gpu-metrics-frequency 10000 \
    --stats=true \
    --force-overwrite=true \
    python "$OUTPUT_DIR/test_sequential_only.py" \
        "$VIDEO1" \
        "$VIDEO2" \
        "$OUTPUT_DIR/sequential_output" \
        "$NUM_STEPS"

echo ""
echo "============================================================"
echo "Profiling Complete!"
echo "============================================================"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.nsys-rep 2>/dev/null || echo "  (no .nsys-rep files found)"
echo ""
echo "To copy to local machine:"
echo "  scp horde:$COSMOS_DIR/$OUTPUT_DIR/*.nsys-rep ./"
echo ""
echo "To view profiles:"
echo "  nsys-ui cosmos_batch_inference.nsys-rep"
echo "  nsys-ui cosmos_sequential_inference.nsys-rep"
echo ""
echo "Key things to look for:"
echo "  1. GPU kernel execution time vs idle time"
echo "  2. Memory transfer patterns"
echo "  3. DiT forward pass timing"
echo "  4. Whether batch kernels are larger or just 2x as many"

