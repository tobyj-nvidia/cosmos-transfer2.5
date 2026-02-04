#!/usr/bin/env python3
"""
Run CFG optimization test with detailed NVTX profiling markers.

This script applies the v3 NVTX profiling patches to add detailed markers
throughout the Cosmos codebase, then runs the CFG optimization test.

Usage:
    python tests/test_cfg_optimization_nvtx_v3.py \
        --depth-video <path> \
        --output-dir <path> \
        --state-t 2 \
        --seed 42 \
        --num-steps 4 \
        --profile
"""

import sys
from pathlib import Path

# Add scripts directory to path to import nvtx_profiling_v3
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

print("=" * 70)
print("Applying NVTX v3 profiling patches...")
print("=" * 70)

# Import and apply all NVTX patches
from nvtx_profiling_v3 import apply_all_patches
apply_all_patches()

print("\n" + "=" * 70)
print("NVTX patches applied successfully")
print("=" * 70)
print()

# Now import and run the actual test
from test_cfg_optimization_local import main

if __name__ == "__main__":
    main()

