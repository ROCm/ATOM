#!/usr/bin/env python3
"""Backward-compatible wrapper — use merge_dsv4_bf16_gemm_gfx942.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_dsv4_bf16_gemm_gfx942 import main

if __name__ == "__main__":
    raise SystemExit(main())
