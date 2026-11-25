# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import torch

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_cuda
except (ImportError, OSError):
    _hadamard_cuda = None

from atomrl.lumen.ops.dsv4.rope import (
    wrapped_precompute_freqs_cis,  # noqa: F401 — public compatibility export
)


def _hadamard_torch(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Walsh-Hadamard on the last dim (power-of-2). Fallback when CUDA ext ABI mismatches."""
    n = x.size(-1)
    if n == 1:
        return x * scale
    if n & (n - 1):
        raise ValueError(f"hadamard size must be a power of 2, got {n}")
    out = x
    h = 1
    while h < n:
        out = out.reshape(*out.shape[:-1], n // (2 * h), h, 2)
        a = out[..., 0]
        b = out[..., 1]
        out = torch.stack((a + b, a - b), dim=-1).reshape(*x.shape)
        h *= 2
    return out * scale


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform used to redistribute activation energy before QAT."""
    assert x.dtype == torch.bfloat16
    scale = x.size(-1) ** -0.5
    if _hadamard_cuda is not None:
        try:
            return _hadamard_cuda(x, scale=scale)
        except (RuntimeError, OSError):
            pass
    return _hadamard_torch(x, scale)
