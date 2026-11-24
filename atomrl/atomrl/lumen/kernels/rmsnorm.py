# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen-owned RMSNorm Triton kernels for short rows.

Contains only kernels that AITER does not provide. AITER has a narrow-N forward
(``_rmsnorm_kernel_large_m_small_n``) but no matching backward, so its backward
walks one row per program. At the per-head QK-norm shape used by Qwen3-style
models -- ``N = head_dim = 128`` over hundreds of thousands of rows -- a row is
128 elements, which leaves most of the wavefront idle and measures ~4x slower
than the forward despite moving only 3x the bytes.

Both kernels here tile ``BLOCK_M`` rows at a time so a program always has a full
tile of work, and the backward keeps its per-row reduction inside the tile.

    y      = x * rsigma * w,   rsigma = 1 / sqrt(mean(x^2) + eps)
    g      = dy * w
    dx     = rsigma * (g - x * rsigma^2 * sum_n(g * x) / N)
    dw     = sum_m (dy * x * rsigma)

``dw`` reduces across every row, so each program accumulates a private partial
and a second pass folds the partials together.
"""

import torch
import triton
import triton.language as tl

__all__ = [
    "NARROW_RMSNORM_MAX_N",
    "narrow_rmsnorm_backward",
    "narrow_rmsnorm_forward",
]

# Above this the row itself fills a wavefront and AITER's own kernels are at
# least as fast, so there is nothing for a row-tiling kernel to win.
NARROW_RMSNORM_MAX_N = 512

# A 16384-element tile keeps the wavefront full for any N in range; measured
# best on gfx950 across M from 1.3e5 to 5.2e5.
_TILE_ELEMS = 16384

# Programs per SM for the backward's persistent loop. Two gave the best balance
# between occupancy and the size of the dw partial buffer.
_BWD_WAVES = 2


@triton.jit
def _narrow_rmsnorm_fwd_kernel(
    X,
    Y,
    W,
    RSIGMA,
    M,
    N,
    eps,
    stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    m_off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    mask_m = m_off < M
    mask_n = n_off < N
    mask = mask_m[:, None] & mask_n[None, :]
    offs = m_off[:, None] * stride_m + n_off[None, :]

    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + n_off, mask=mask_n, other=0.0).to(tl.float32)

    rsigma = tl.math.rsqrt(tl.sum(x * x, axis=1) / N + eps)
    y = x * rsigma[:, None] * w[None, :]

    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(RSIGMA + m_off, rsigma, mask=mask_m)


@triton.jit
def _narrow_rmsnorm_bwd_kernel(
    DY,
    X,
    W,
    RSIGMA,
    DX,
    DW_PARTIAL,
    M,
    N,
    stride_m,
    NUM_PRGMS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_off = tl.arange(0, BLOCK_N)
    mask_n = n_off < N
    w = tl.load(W + n_off, mask=mask_n, other=0.0).to(tl.float32)
    dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for blk in range(pid, tl.cdiv(M, BLOCK_M), NUM_PRGMS):
        m_off = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = m_off < M
        mask = mask_m[:, None] & mask_n[None, :]
        offs = m_off[:, None] * stride_m + n_off[None, :]

        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        rsigma = tl.load(RSIGMA + m_off, mask=mask_m, other=0.0)

        g = dy * w[None, :]
        dot = tl.sum(g * x, axis=1)
        dx = rsigma[:, None] * (g - x * (rsigma * rsigma * dot / N)[:, None])

        tl.store(DX + offs, dx.to(DX.dtype.element_ty), mask=mask)
        dw_acc += tl.sum(dy * x * rsigma[:, None], axis=0)

    tl.store(DW_PARTIAL + pid * N + n_off, dw_acc, mask=mask_n)


@triton.jit
def _narrow_rmsnorm_dw_reduce_kernel(
    DW_PARTIAL,
    DW,
    ROWS,
    N,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_off = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_off < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for r0 in range(0, ROWS, BLOCK_R):
        r_off = r0 + tl.arange(0, BLOCK_R)
        mask = (r_off < ROWS)[:, None] & mask_n[None, :]
        acc += tl.sum(
            tl.load(
                DW_PARTIAL + r_off[:, None] * N + n_off[None, :], mask=mask, other=0.0
            ),
            axis=0,
        )

    tl.store(DW + n_off, acc.to(DW.dtype.element_ty), mask=mask_n)


def _tile_shape(n):
    block_n = triton.next_power_of_2(n)
    return max(8, min(128, _TILE_ELEMS // block_n)), block_n


def narrow_rmsnorm_forward(x, weight, eps):
    """RMSNorm forward over a contiguous ``(M, N)`` input with small ``N``.

    Returns ``(y, rsigma)``; ``rsigma`` is kept in fp32 for the backward.
    """
    m, n = x.shape
    block_m, block_n = _tile_shape(n)
    y = torch.empty_like(x)
    rsigma = torch.empty(m, dtype=torch.float32, device=x.device)

    _narrow_rmsnorm_fwd_kernel[(triton.cdiv(m, block_m),)](
        x,
        y,
        weight,
        rsigma,
        m,
        n,
        eps,
        x.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )
    return y, rsigma


def narrow_rmsnorm_backward(dy, x, weight, rsigma, num_sms):
    """Gradients of :func:`narrow_rmsnorm_forward`. Returns ``(dx, dw)``."""
    m, n = x.shape
    block_m, block_n = _tile_shape(n)
    num_prgms = min(triton.cdiv(m, block_m), num_sms * _BWD_WAVES)

    dx = torch.empty_like(x)
    dw_partial = torch.empty((num_prgms, n), dtype=torch.float32, device=x.device)

    _narrow_rmsnorm_bwd_kernel[(num_prgms,)](
        dy,
        x,
        weight,
        rsigma,
        dx,
        dw_partial,
        m,
        n,
        x.stride(0),
        num_prgms,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )

    dw = torch.empty_like(weight)
    _narrow_rmsnorm_dw_reduce_kernel[(triton.cdiv(n, 64),)](
        dw_partial,
        dw,
        num_prgms,
        n,
        BLOCK_R=64,
        BLOCK_N=64,
        num_warps=4,
    )
    return dx, dw
