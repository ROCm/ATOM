# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 Gluon KDA o_norm (decode): rmsnorm_gated fused with o_proj's E8M0 quant.

Byte-identical to ``rmsnorm_gated`` (the one-row Triton kernel, num_warps=4)
followed by ``per_1x128_e8m0_quant``. A KDA head is 128 wide, one quant group, so
a row is one (token, head). The Triton kernel spreads a row over four waves and
reduces across them through LDS; with the quant's amax as a second cross-wave
reduction that cost more than the standalone quant it replaced (T >= 1024). Here
one wave owns a row:

* The [4, 32] tile puts row element ``32 * k + lane`` in lane ``lane``, register
  ``k``: register k is what wave k held in the Triton kernel, so the variance sum
  takes that kernel's order (the lane butterfly, then the four wave partials as
  ``(p0 + p2) + (p1 + p3)``).
* The Triton kernel rounds ``(x * rstd * w) * sigmoid(gate)`` to BF16 once
  (``v_fma_mixlo_bf16``, a +0 addend); here the product is rounded to FP32 first.
  The two differ only where the FP32 product sits on a BF16 tie, which is then
  stepped one FP32 ulp toward the exact product (``fma`` gives the error).
* Programs are remapped so each XCD takes a contiguous run of rows: the
  column-major scale of neighbouring tokens shares lines, and written from every
  XCD those lines cost more than the whole norm at T >= 1024.
"""

from __future__ import annotations

import functools

import torch
from aiter import dtypes
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language import PropagateNan

_PROPAGATE_NAN = gl.constexpr(PropagateNan.ALL)  # keep NaN through the FP8 clamp


@gluon.jit
def _rmsnorm_gated_e8m0_kernel(
    x_ptr,
    w_ptr,
    g_ptr,
    y_ptr,
    s_ptr,
    H,
    eps,
    stride_xm,
    stride_ym,
    stride_g_outer,
    stride_g_head,
    stride_s_outer,
    stride_s_head,
    HEADS: gl.constexpr,
    FP8_MAX: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    row = gl.program_id(0)
    if NUM_XCDS > 1:
        G = gl.num_programs(0)
        per = (G + NUM_XCDS - 1) // NUM_XCDS
        tall = G % NUM_XCDS
        tall += (tall == 0).to(gl.int32) * NUM_XCDS
        xcd = row % NUM_XCDS
        row = xcd * per + row // NUM_XCDS - (xcd - tall) * (xcd > tall).to(gl.int32)
    k = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    lane = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    cols = gl.expand_dims(k, 1) * 32 + gl.expand_dims(lane, 0)
    outer = (row // HEADS).to(gl.int64)
    head = (row % HEADS).to(gl.int64)
    x = gl.load(x_ptr + row.to(gl.int64) * stride_xm + cols).to(gl.float32)

    part = gl.sum(x * x, axis=1)
    p0 = gl.sum(gl.where(k == 0, part, 0.0), axis=0)
    p1 = gl.sum(gl.where(k == 1, part, 0.0), axis=0)
    p2 = gl.sum(gl.where(k == 2, part, 0.0), axis=0)
    p3 = gl.sum(gl.where(k == 3, part, 0.0), axis=0)
    rstd = 1.0 / gl.sqrt(((p0 + p2) + (p1 + p3)) / H + eps)
    w = gl.load(w_ptr + cols).to(gl.float32)
    gate = gl.load(g_ptr + outer * stride_g_outer + head * stride_g_head + cols)
    a = x * rstd * w
    sg = 1 / (1 + gl.exp(-gate.to(gl.float32)))
    p = a * sg + 0.0  # as the +0 addend: a -0 product rounds to +0
    err = gl.fma(a, sg, -p)
    bits = p.to(gl.uint32, bitcast=True)
    tie = ((bits & 0xFFFF) == 0x8000) & (err != 0.0)
    bits = gl.where(tie, gl.where((err > 0.0) == (p > 0.0), bits + 1, bits - 1), bits)
    y = bits.to(gl.float32, bitcast=True).to(x_ptr.dtype.element_ty).to(gl.float32)

    # triton_e8m0_quant.fp8_e8m0_group_scale (Gluon cannot call it)
    amax = gl.max(gl.max(gl.abs(y), axis=1), axis=0)
    amax = gl.minimum(gl.maximum(amax, 1e-10), 448.0 * 2.0**119)
    bits = (amax * (1.0 / FP8_MAX)).to(gl.uint32, bitcast=True)
    code = (bits >> 23) & 0xFF
    code += ((code < 0xFF) & ((bits & 0x7FFFFF) != 0)).to(gl.uint32)
    y = y * ((254 - code) << 23).to(gl.float32, bitcast=True)
    y = gl.maximum(y, -FP8_MAX, propagate_nan=_PROPAGATE_NAN)
    y = gl.minimum(y, FP8_MAX, propagate_nan=_PROPAGATE_NAN)
    gl.store(s_ptr + outer * stride_s_outer + head * stride_s_head, code.to(gl.uint8))
    gl.store(y_ptr + row.to(gl.int64) * stride_ym + cols, y.to(y_ptr.dtype.element_ty))


@functools.cache
def _num_xcds() -> int:
    from aiter.ops.triton.utils.device_info import get_num_xcds

    return get_num_xcds()


def rmsnorm_gated_e8m0(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    quant_dtype: torch.dtype,
    col_major: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(fp8 [t, heads * 128], E8M0 scale [t, heads])`` of x [t, heads, 128] and a
    gate [t, heads, 128] that may be strided; ``col_major`` stores the scale
    transposed, as aiter's preshuffled GEMM reads it."""
    t, heads, h = x.shape
    m = t * heads
    x2 = x.reshape(m, h).contiguous()
    y = torch.empty((m, h), dtype=quant_dtype, device=x.device)
    scale = torch.empty(
        (heads, t) if col_major else (t, heads), dtype=torch.uint8, device=x.device
    )
    if m:
        _rmsnorm_gated_e8m0_kernel[(m,)](
            x2,
            weight,
            gate,
            y,
            scale,
            h,
            float(eps),
            x2.stride(0),
            y.stride(0),
            gate.stride(0),
            gate.stride(1),
            1 if col_major else heads,
            t if col_major else 1,
            HEADS=heads,
            FP8_MAX=float(torch.finfo(quant_dtype).max),
            NUM_XCDS=_num_xcds(),
            num_warps=1,
        )
    return y.view(t, heads * h), scale.view(t, heads).view(dtypes.fp8_e8m0)
