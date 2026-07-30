# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused activation and gated RMSNorm operations for Kimi-K3."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _situ_and_mul_kernel(
        x_ptr,
        y_ptr,
        M,
        D,
        stride_xm,
        stride_ym,
        beta,
        inv_beta,
        linear_beta,
        inv_linear_beta,
        HAS_LINEAR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        g = tl.load(x_ptr + row * stride_xm + col, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * stride_xm + D + col, mask=mask, other=0.0).to(
            tl.float32
        )
        # SiTUv2 gate: beta * tanh(gate/beta) * sigmoid(gate); tanh via sigmoid
        # identity (tanh(z) = 2*sigmoid(2z) - 1) for portability across triton.
        out = beta * (2.0 * tl.sigmoid(2.0 * g * inv_beta) - 1.0) * tl.sigmoid(g)
        if HAS_LINEAR:
            u = linear_beta * (2.0 * tl.sigmoid(2.0 * u * inv_linear_beta) - 1.0)
        y = out * u
        tl.store(y_ptr + row * stride_ym + col, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_gated_kernel(
        x_ptr,
        w_ptr,
        g_ptr,
        y_ptr,
        H,
        eps,
        stride_xm,
        stride_ym,
        stride_g_outer,
        stride_g_head,
        HEADS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < H
        # x / y are row-contiguous [M, H]; gate may be strided. Its logical row
        # `row` decomposes into (outer, head) so the token-boundary jump
        # (stride_g_outer) and per-head step (stride_g_head) are read directly,
        # avoiding a contiguous copy of the strided gate slice.
        g_off = (row // HEADS) * stride_g_outer + (row % HEADS) * stride_g_head + cols
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / H
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd * w) * tl.sigmoid(gate)
        tl.store(
            y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask
        )


def situ_and_mul(
    x: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    """SiTUv2 gated activation over the last dim (x[..., :D] gate, x[..., D:] up)."""
    *lead, two_d = x.shape
    assert two_d % 2 == 0
    d = two_d // 2
    x2 = x.reshape(-1, two_d)
    m = x2.shape[0]
    y = torch.empty((m, d), dtype=x.dtype, device=x.device)
    if not _HAS_TRITON or m == 0:
        return _situ_and_mul_torch(x, beta, linear_beta)
    BLOCK = 1024
    grid = (m, triton.cdiv(d, BLOCK))
    has_linear = linear_beta is not None
    _situ_and_mul_kernel[grid](
        x2,
        y,
        m,
        d,
        x2.stride(0),
        y.stride(0),
        float(beta),
        1.0 / float(beta),
        float(linear_beta) if has_linear else 0.0,
        (1.0 / float(linear_beta)) if has_linear else 0.0,
        HAS_LINEAR=has_linear,
        BLOCK=BLOCK,
    )
    return y.reshape(*lead, d)


def rmsnorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """rmsnorm(x) over last dim * weight * sigmoid(gate), written into ``out``.

    ``gate`` may be strided (e.g. a column slice of a fused GEMM output): the
    kernel reads it via (outer, head) strides so no contiguous copy is needed.
    ``x`` is normed row-wise and is made contiguous (cheap; the caller's ``out``
    already is). Supports a 2D ``[M, H]`` or 3D ``[outer, heads, H]`` gate.

    ``out`` is a contiguous buffer holding ``x``'s element count that the kernel
    writes into directly, and which is returned. There is deliberately no
    allocating form: the only caller is the KDA mixer, which hands over the
    cudagraph-owned buffer its splitting op was given, so the result lands at
    its final address instead of being allocated here and copied. ``out`` may
    alias ``x`` -- one program per row, whole row loaded before it is stored.
    """
    h = x.shape[-1]
    x2 = x.reshape(-1, h)
    m = x2.shape[0]
    # Must end up a view of `out`, never a copy, or the kernel writes where the
    # caller never looks. A contiguous `out` reshapes to [m, h] for free.
    assert out.is_contiguous() and out.numel() == x2.numel(), (
        "rmsnorm_gated(out=...) needs a contiguous buffer holding x's "
        f"element count; got {tuple(out.shape)} "
        f"contiguous={out.is_contiguous()} for x {tuple(x.shape)}"
    )
    if not _HAS_TRITON or m == 0 or h > 8192:
        # The torch reference is a plain expression, so it allocates; landing it
        # in `out` needs this copy. Unreachable for KDA (triton present, h=128,
        # m>0), and copies nothing on the m==0 batch that can reach it.
        return out.view_as(x).copy_(_rmsnorm_gated_torch(x, weight, gate, eps))
    if gate.ndim == 3:
        heads = gate.shape[1]
        stride_g_outer, stride_g_head = gate.stride(0), gate.stride(1)
    else:
        # 2D: one logical head per row; the head term drops out (row % 1 == 0).
        heads = 1
        stride_g_outer, stride_g_head = gate.stride(0), 0
    x2 = x2.contiguous()
    y = out.view(m, h)
    BLOCK = triton.next_power_of_2(h)
    _rmsnorm_gated_kernel[(m,)](
        x2,
        weight,
        gate,
        y,
        h,
        float(eps),
        x2.stride(0),
        y.stride(0),
        stride_g_outer,
        stride_g_head,
        HEADS=heads,
        BLOCK=BLOCK,
    )
    return out


# --------------------------------------------------------------------------- #
# torch references (also the fallback when triton is unavailable)
# --------------------------------------------------------------------------- #
def _situ_and_mul_torch(
    x: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate_f = gate.float()
    up_f = up.float()
    out = beta * torch.tanh(gate_f / beta) * torch.sigmoid(gate_f)
    if linear_beta is not None:
        up_f = linear_beta * torch.tanh(up_f / linear_beta)
    return (out * up_f).to(x.dtype)


def _rmsnorm_gated_torch(
    x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float
) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    xn = x_f * torch.rsqrt(var + eps)
    return (xn.to(dtype) * weight.to(dtype)) * torch.sigmoid(gate)
