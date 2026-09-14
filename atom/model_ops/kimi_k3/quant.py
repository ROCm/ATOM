# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone activation quant for Kimi-K3, fused with a strided read.

The one kernel here exists because a column slice of a fused GEMM output has
to be made row-contiguous before it can feed the next GEMM, and under a
per-token quant scheme it then has to be quantized -- two full [T, D] HBM
round-trips for what is one load and one store of work.
``strided_per_token_quant`` does the gather and the quant together: it reads
at the source's row stride and writes the contiguous quantized result, so the
intermediate bf16 copy never exists.

Why not an aiter quant? Both of aiter's per-token entry points were checked:

* ``get_hip_quant(QuantType.per_Token)`` asserts a contiguous input
  (``AITER_CHECK(input.is_contiguous())``, csrc/kernels/quant_kernels.cu), and
  the kernel derives its row offset as ``blockIdx.x * cols`` with no stride
  argument at all. Passing a slice does not raise -- it ``abort()``s the
  process. This is still the fallback below, paid for with a ``.contiguous()``.
* ``aiter.ops.triton.quant.dynamic_per_token_quant_fp8_i8`` *does* take
  ``x_in.stride(0)``, but its kernel reuses that one offset for the store as
  well as the load (``offs = pid * x_in_stride_r + ...``, used by both
  ``tl.load`` and ``tl.store``). With the contiguous ``[T, D]`` output the
  consuming GEMM needs, a strided input therefore writes out of bounds -- 1792
  stray elements at T=17/D=128/stride=1024, and a hard IMA at T=4096. It also
  divides by a zero scale on an all-zero row (NaN) where the HIP path does not.

So the ``strided_`` prefix is load-bearing, and the kernel below is bit-exact
against ``get_hip_quant(QuantType.per_Token)`` (verified over T in 1..4096 and
non-power-of-two D) so that the fused and fallback paths stay interchangeable.
"""

from __future__ import annotations

import torch
from aiter import QuantType, get_hip_quant

from atom.utils.decorators import mark_trace

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _strided_per_token_quant_kernel(
        x_ptr,
        y_ptr,
        s_ptr,
        D,
        fp8_max,
        inv_fp8_max,
        stride_xm,
        stride_ym,
        BLOCK: tl.constexpr,  # next_pow2(D) -- one tile spans the whole row
    ):
        # One program per token. The row is loaded once, reduced to its amax,
        # and stored quantized -- so the strided source read costs no more than
        # the contiguous copy it replaces, and the quant rides along free.
        #
        # BLOCK spans all of D deliberately: a row split across tiles would need
        # either a second pass (amax is not known until every tile is seen) or a
        # cross-tile atomic, and both cost more than the register pressure of a
        # single wide tile at the D this is used for (head_dim).
        tok = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        # stride_xm is the SOURCE row stride: for a column slice of a fused GEMM
        # output that is the fused N, not D. The feature stride is 1 (a column
        # slice preserves it), so no explicit column stride is needed.
        x = tl.load(x_ptr + tok * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x))
        # amax * (1/fp8_max), NOT amax / fp8_max. Triton's fdiv on ROCm returns a
        # result 1 ulp off IEEE here, which shifts ~0.15% of elements to an
        # adjacent FP8 code versus aiter's per-token quant; the multiply is exact
        # and reproduces aiter bit for bit (which matters: the two are used
        # interchangeably for the same activation, including as each other's
        # fallback). fp8_max is a power of two, so 1/fp8_max is exact too.
        scale = amax * inv_fp8_max
        # An all-zero row has amax 0; 1/0 would poison the row with NaN, so the
        # reciprocal is forced to 0 there (the stored row is 0 either way, and
        # the scale is stored as 0 to match aiter's per-token quant).
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        q = tl.minimum(tl.maximum(x * inv, -fp8_max), fp8_max)
        tl.store(
            y_ptr + tok * stride_ym + cols, q.to(y_ptr.dtype.element_ty), mask=mask
        )
        tl.store(s_ptr + tok, scale)


@mark_trace
def strided_per_token_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token quantize a possibly-strided ``[T, D]`` activation.

    Fuses the row gather into the quant: ``x`` may be a column slice of a wider
    tensor (feature stride 1, row stride > D), which the kernel reads at its
    own stride. The result is a fresh contiguous ``[T, D]`` tensor, so the
    ``.contiguous()`` such a slice would otherwise need before its consuming
    GEMM is subsumed rather than merely reordered.

    Returns ``(quantized [T, D], scale [T, 1] float32)``, the layout a
    per-token a8w8 GEMM takes as ``x_scale=``. Bit-exact against
    ``get_hip_quant(QuantType.per_Token)`` -- which is the fallback when triton
    is unavailable -- including the zero scale on an all-zero row.
    """
    assert x.ndim == 2, f"expected [T, D], got {tuple(x.shape)}"
    t, d = x.shape
    if not _HAS_TRITON or t == 0:
        # aiter's kernel needs a contiguous input, so this fallback pays the
        # copy the fused path exists to avoid.
        return get_hip_quant(QuantType.per_Token)(
            x.contiguous(), quant_dtype=quant_dtype
        )
    assert x.stride(1) == 1, (
        "strided_per_token_quant reads rows at x.stride(0) with a unit feature stride; "
        f"got strides {tuple(x.stride())}"
    )
    fp8_max = float(torch.finfo(quant_dtype).max)
    out = torch.empty((t, d), dtype=quant_dtype, device=x.device)
    scale = torch.empty((t, 1), dtype=torch.float32, device=x.device)
    _strided_per_token_quant_kernel[(t,)](
        x,
        out,
        scale,
        d,
        fp8_max,
        1.0 / fp8_max,
        x.stride(0),
        out.stride(0),
        BLOCK=triton.next_power_of_2(d),
    )
    return out, scale
