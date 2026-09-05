"""
TurboQuant — 3-bit KV Cache Compression for AMD ROCm

Implements asymmetric KV cache quantization:
- Keys: 2-bit Lloyd-Max + 1-bit QJL correction = 3 bits/coordinate
- Values: 3-bit Lloyd-Max = 3 bits/coordinate
- ~5× compression vs FP16, cosine similarity > 0.98

Based on: https://github.com/DevTechJr/turboquant_cutile
Ported from NVIDIA Blackwell/cuTile to AMD MI355X via PyTorch + Baybridge.
"""

from .engine import TurboQuantEngine
from .codebook import LloydMaxCodebook, solve_lloyd_max

__all__ = ["TurboQuantEngine", "LloydMaxCodebook", "solve_lloyd_max"]
