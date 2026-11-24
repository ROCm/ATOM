# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""HIP C++ fused FP8 cast + transpose + amax kernel.

Loads the optional prebuilt ``atomrl.lumen.csrc._fused_quant_transpose``
extension and exposes the same API as
:func:`~atomrl.lumen.ops.quantize.cast_transpose.cast_transpose_amax_fp8`.
Build it by installing ATOM with ``LUMEN_BUILD_EXTENSIONS=1`` on a configured
ROCm host. When it is absent, callers retain the Python/Triton fallback.

Enable via ``LUMEN_FUSED_QUANT_TRANSPOSE_CPP=1``.
"""

from __future__ import annotations

import functools
import logging

import torch

from atomrl.lumen.ops.quantize.quant_amax_fused import _get_amax_scratch

logger = logging.getLogger(__name__)

_module = None


def _load_module():
    """Load the pre-built HIP extension from the csrc package."""
    global _module
    if _module is not None:
        return _module

    import importlib

    _module = importlib.import_module("atomrl.lumen.csrc._fused_quant_transpose")
    return _module


@functools.lru_cache(maxsize=1)
def _probe_hip_cast_transpose() -> bool:
    """Return True if the HIP fused quant+transpose kernel can be compiled."""
    try:
        _load_module()
        return True
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning("HIP cast+transpose kernel unavailable: %s", e)
        return False


def cast_transpose_amax_fp8_hip(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    clamp_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused FP8 cast + transpose + amax via HIP C++ kernel.

    API-compatible with
    :func:`~atomrl.lumen.ops.quantize.cast_transpose.cast_transpose_amax_fp8`.

    The HIP kernel reads BF16/FP16 input (M, N) once and produces:
      - out_row (M, N) in FP8 row-major
      - out_col (N, M) in FP8 (transposed)
      - amax    (1,)   float32 max(abs(input))
    """
    mod = _load_module()

    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert x.is_cuda, "cast_transpose_amax_fp8_hip requires a CUDA tensor"

    M, N = x.shape
    out_row = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    out_col = torch.empty((N, M), dtype=fp8_dtype, device=x.device)
    amax_out = _get_amax_scratch(x.device)
    amax_out.zero_()

    if scale.dtype == torch.float32 and scale.ndim == 1 and scale.is_contiguous():
        scale_1 = scale
    else:
        scale_1 = scale.float().reshape(-1).contiguous()
    if scale_1.device != x.device:
        scale_1 = scale_1.to(device=x.device)

    x_2d = x.contiguous()
    mod.static_quant_transpose_amax(out_row, out_col, amax_out, x_2d, scale_1)

    return out_row, out_col, amax_out.clone()
