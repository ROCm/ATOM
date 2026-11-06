#!/usr/bin/env python3
"""Prebuild the V9 HIP kernels into the kernel cache (otherwise built on first use)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsv41_v9 import kernels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter", type=Path, help="AITER checkout (default: imported)")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--cache-dir", type=Path, help="default: $DSV41_V9_KERNEL_DIR")
    args = parser.parse_args()
    for name in kernels.SOURCES:
        print(kernels.library_path(name, args.aiter, args.arch, args.cache_dir))
