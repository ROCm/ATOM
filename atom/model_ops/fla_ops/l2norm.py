# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import torch

import triton
import triton.language as tl

BT_LIST = [8, 16, 32, 64, 128]

USE_DEFAULT_FLA_NORM = int(os.getenv("USE_DEFAULT_FLA_NORM", "0"))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["D"],
)
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["NB"])
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    xs = tl.load(X + (rindex + N * row_idx), xmask).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)


@triton.jit
def l2norm_fwd_qk_kernel(
    X_Q,
    X_K,
    Y_Q,
    Y_K,
    eps,
    M,
    N: tl.constexpr,
    MBLOCK: tl.constexpr,
):
    """Fused l2-norm over two same-shape inputs (q, k). Each program
    normalises MBLOCK rows of EITHER q (when program_id(1) == 0) or k
    (when program_id(1) == 1). The grid is therefore (cdiv(M, MBLOCK), 2)
    — twice as many blocks per launch as a single-input l2norm, which
    helps fill the GPU when M*N is small. Numerically identical to two
    back-to-back l2norm_fwd_kernel2 calls (same reduction order per row).
    """
    xoffset = tl.program_id(0) * MBLOCK
    is_k = tl.program_id(1)
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    # Branch the base pointers via a Python-level if on the program_id
    # value. Triton compiles both branches into the same kernel; the
    # runtime takes only one branch per program.
    if is_k == 0:
        xs = tl.load(X_Q + (rindex + N * row_idx), xmask).to(tl.float32)
    else:
        xs = tl.load(X_K + (rindex + N * row_idx), xmask).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    if is_k == 0:
        tl.store(Y_Q + (rindex + N * row_idx), xs * rsqrt, xmask)
    else:
        tl.store(Y_K + (rindex + N * row_idx), xs * rsqrt, xmask)


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if not USE_DEFAULT_FLA_NORM:
        MBLOCK = 32
        # M, N = x.shape
        l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](
            x,
            y,
            eps,
            T,
            D,
            MBLOCK,
        )
    else:
        if D <= 512:
            NB = triton.cdiv(T, 2048)

            def grid(meta):
                return (triton.cdiv(T, meta["BT"]),)

            l2norm_fwd_kernel[grid](
                x,
                y,
                eps,
                NB=NB,
                T=T,
                D=D,
                BD=BD,
            )
        else:
            l2norm_fwd_kernel1[(T,)](
                x,
                y,
                eps=eps,
                D=D,
                BD=BD,
            )

    return y.view(x_shape_og)


def l2norm_fwd_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused l2-norm over q and k in a single kernel launch.

    Equivalent to `l2norm_fwd(q), l2norm_fwd(k)` but launches one kernel
    with twice the grid instead of two back-to-back kernels — saves a
    launch round-trip and doubles SM occupancy on the launch, which
    matters when `T*H` is small enough that a single l2norm doesn't fill
    the device.

    Requires `q` and `k` to have identical shape AND identical strides
    (so the kernel can address both with the same `(N * row + col)`
    arithmetic). The GDN forward call site upstream of this satisfies
    that — `causal_conv1d_fn` allocates `q` and `k` as fresh contiguous
    `[cu_seqlen, k_dim]` tensors, then both are `.view(1, T, H, K)`'d
    identically.

    Numerics: bit-exact w.r.t. two `l2norm_fwd(...)` calls when both
    take the `USE_DEFAULT_FLA_NORM=0` path (the default). Same per-row
    reduction order, same fp32 accumulation, same rsqrt.
    """
    assert q.shape == k.shape, f"l2norm_fwd_qk: q.shape {q.shape} != k.shape {k.shape}"
    assert q.dtype == k.dtype, f"l2norm_fwd_qk: q.dtype {q.dtype} != k.dtype {k.dtype}"
    assert (
        q.stride() == k.stride()
    ), f"l2norm_fwd_qk: q.stride {q.stride()} != k.stride {k.stride()}"

    x_shape_og = q.shape
    q_flat = q.view(-1, q.shape[-1])
    k_flat = k.view(-1, k.shape[-1])

    out_dtype = output_dtype if output_dtype is not None else q.dtype
    y_q = torch.empty_like(q_flat, dtype=out_dtype)
    y_k = torch.empty_like(k_flat, dtype=out_dtype)
    assert y_q.stride(-1) == 1 and y_k.stride(-1) == 1

    T, D = q_flat.shape
    # Feature-dim safety bound matches the existing l2norm_fwd helper.
    MAX_FUSED_SIZE = 65536 // q.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("l2norm_fwd_qk: feature dim >= 64KB is unsupported.")

    MBLOCK = 32
    grid = (triton.cdiv(T, MBLOCK), 2)
    l2norm_fwd_qk_kernel[grid](
        q_flat,
        k_flat,
        y_q,
        y_k,
        eps,
        T,
        D,
        MBLOCK,
    )

    return y_q.view(x_shape_og), y_k.view(x_shape_og)
