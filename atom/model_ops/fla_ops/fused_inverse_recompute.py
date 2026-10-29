# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""(I + A)^-1 of each 64-token chunk and the WY vectors it gives, in one kernel.

Per chunk and value head, from the strictly lower triangular A of
`fused_cumsum_kkt`:

    Ai = (I + A)^-1,   w = Ai (beta exp(g_cumsum) k),   u = Ai (beta v)

It replaces `solve_tril_16x16_kernel` -> `fused_merge_recompute`: the diagonal
16 x 16 inverses no longer go through HBM. Ai is built in 16 x 16 blocks: each
diagonal block as the product (I + N)(I + N^2)(I + N^4)(I + N^8) of its
nilpotent N = -A_ii, the off-diagonal blocks by block forward substitution, all
on fp32 WMMA. The w / u products fold the per-token scale into Ai's columns,
w = (Ai diag(beta exp(g))) k, so k and v enter bf16 WMMA exactly; the scaled Ai
goes in as a bf16 high and low part (two WMMAs, ~16 bits of mantissa), which
keeps w and u at the bf16 rounding of the fp32 result. The products run over
NC column slices of k / v.

One warp a program: the 16 x 16 block algebra would repeat in every warp of a
wider program. The launch configuration is fixed (gfx1250 sweep, see
`fused_inverse_recompute`): no autotuning.
"""

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices


@triton.jit
def _inv16(a, eye):
    """(I + a)^-1 for a strictly lower triangular 16 x 16 fp32 tile."""
    n1 = -a
    n2 = tl.dot(n1, n1, input_precision="ieee")
    n4 = tl.dot(n2, n2, input_precision="ieee")
    n8 = tl.dot(n4, n4, input_precision="ieee")
    p = eye + n1 + n2 + tl.dot(n1, n2, input_precision="ieee")
    p = p + tl.dot(p, n4, input_precision="ieee")
    return p + tl.dot(p, n8, input_precision="ieee")


@triton.jit
def _tile(ptr, row0, col0, stride, T_seq):
    """[16, 16] fp32 tile at (row0, col0); rows past the sequence read 0."""
    r = row0 + tl.arange(0, 16)
    c = col0 + tl.arange(0, 16)
    return tl.load(
        ptr + r[:, None] * stride + c[None, :], mask=(r < T_seq)[:, None], other=0.0
    )


@triton.jit
def _vec(ptr, row0, n):
    r = row0 + tl.arange(0, 16)
    return tl.load(ptr + r, mask=r < n, other=0.0)


@triton.jit
def _split(x, SPLIT: tl.constexpr):
    """x as a bf16 high part and (with SPLIT) the bf16 rounding of what it leaves."""
    hi = x.to(tl.bfloat16)
    if SPLIT:
        return hi, (x - hi.to(tl.float32)).to(tl.bfloat16)
    return hi, hi


@triton.jit
def _mac(acc, hi, lo, rhs, SPLIT: tl.constexpr):
    acc = tl.dot(hi, rhs, acc)
    if SPLIT:
        acc = tl.dot(lo, rhs, acc)
    return acc


@triton.jit
def _wy(
    src,
    s_src,
    dst,
    s_dst,
    row0,
    T_seq,
    cols,
    x11h,
    x11l,
    x21h,
    x21l,
    x22h,
    x22l,
    x31h,
    x31l,
    x32h,
    x32l,
    x33h,
    x33l,
    x41h,
    x41l,
    x42h,
    x42l,
    x43h,
    x43l,
    x44h,
    x44l,
    SPLIT: tl.constexpr,
    N: tl.constexpr,
):
    """dst rows = (X src) over one column slice, X the scaled Ai in 16-row blocks."""
    r = row0 + tl.arange(0, 16)
    p = src + r[:, None] * s_src + cols[None, :]
    s1 = tl.load(p, mask=(r < T_seq)[:, None], other=0.0)
    s2 = tl.load(p + 16 * s_src, mask=(r + 16 < T_seq)[:, None], other=0.0)
    s3 = tl.load(p + 32 * s_src, mask=(r + 32 < T_seq)[:, None], other=0.0)
    s4 = tl.load(p + 48 * s_src, mask=(r + 48 < T_seq)[:, None], other=0.0)
    q = dst + r[:, None] * s_dst + cols[None, :]
    acc = _mac(tl.zeros([16, N], dtype=tl.float32), x11h, x11l, s1, SPLIT)
    tl.store(q, acc.to(dst.dtype.element_ty), mask=(r < T_seq)[:, None])
    acc = _mac(tl.zeros([16, N], dtype=tl.float32), x21h, x21l, s1, SPLIT)
    acc = _mac(acc, x22h, x22l, s2, SPLIT)
    tl.store(
        q + 16 * s_dst, acc.to(dst.dtype.element_ty), mask=(r + 16 < T_seq)[:, None]
    )
    acc = _mac(tl.zeros([16, N], dtype=tl.float32), x31h, x31l, s1, SPLIT)
    acc = _mac(acc, x32h, x32l, s2, SPLIT)
    acc = _mac(acc, x33h, x33l, s3, SPLIT)
    tl.store(
        q + 32 * s_dst, acc.to(dst.dtype.element_ty), mask=(r + 32 < T_seq)[:, None]
    )
    acc = _mac(tl.zeros([16, N], dtype=tl.float32), x41h, x41l, s1, SPLIT)
    acc = _mac(acc, x42h, x42l, s2, SPLIT)
    acc = _mac(acc, x43h, x43l, s3, SPLIT)
    acc = _mac(acc, x44h, x44l, s4, SPLIT)
    tl.store(
        q + 48 * s_dst, acc.to(dst.dtype.element_ty), mask=(r + 48 < T_seq)[:, None]
    )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NT": lambda args: args["nt_ptr"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _fused_inverse_recompute_kernel(
    k_ptr,
    v_ptr,
    A_ptr,
    beta_hm_ptr,
    scale_hm_ptr,
    w_ptr,
    u_ptr,
    cu_seqlens,
    chunk_indices,
    nt_ptr,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    NC: tl.constexpr,
    SPLIT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_NT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN and HAS_NT:
        # a bucket-shaped grid: programs past the step's chunk count have no work
        if i_t >= tl.load(nt_ptr):
            return
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        T_seq = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
    else:
        bos = i_b * T
        T_seq = T
    t0 = i_t * BT

    o = tl.arange(0, 16)
    eye = (o[:, None] == o[None, :]).to(tl.float32)
    a_base = A_ptr + (bos * H + i_h) * BT
    sa = H * BT
    ai11 = _inv16(_tile(a_base, t0, 0, sa, T_seq), eye)
    ai22 = _inv16(_tile(a_base, t0 + 16, 16, sa, T_seq), eye)
    ai33 = _inv16(_tile(a_base, t0 + 32, 32, sa, T_seq), eye)
    ai44 = _inv16(_tile(a_base, t0 + 48, 48, sa, T_seq), eye)
    a21 = _tile(a_base, t0 + 16, 0, sa, T_seq)
    a32 = _tile(a_base, t0 + 32, 16, sa, T_seq)
    a43 = _tile(a_base, t0 + 48, 32, sa, T_seq)
    ai21 = -tl.dot(
        tl.dot(ai22, a21, input_precision="ieee"), ai11, input_precision="ieee"
    )
    ai32 = -tl.dot(
        tl.dot(ai33, a32, input_precision="ieee"), ai22, input_precision="ieee"
    )
    ai43 = -tl.dot(
        tl.dot(ai44, a43, input_precision="ieee"), ai33, input_precision="ieee"
    )
    a31 = _tile(a_base, t0 + 32, 0, sa, T_seq)
    ai31 = -tl.dot(
        ai33,
        tl.dot(a31, ai11, input_precision="ieee")
        + tl.dot(a32, ai21, input_precision="ieee"),
        input_precision="ieee",
    )
    a42 = _tile(a_base, t0 + 48, 16, sa, T_seq)
    ai42 = -tl.dot(
        ai44,
        tl.dot(a42, ai22, input_precision="ieee")
        + tl.dot(a43, ai32, input_precision="ieee"),
        input_precision="ieee",
    )
    a41 = _tile(a_base, t0 + 48, 0, sa, T_seq)
    ai41 = -tl.dot(
        ai44,
        tl.dot(a41, ai11, input_precision="ieee")
        + tl.dot(a42, ai21, input_precision="ieee")
        + tl.dot(a43, ai31, input_precision="ieee"),
        input_precision="ieee",
    )

    # the column scales, one contiguous run of the head-major vectors each
    hm = (i_b * H + i_h) * T + (bos - i_b * T) + t0
    n = T_seq - t0
    s1 = _vec(scale_hm_ptr + hm, 0, n)
    s2 = _vec(scale_hm_ptr + hm, 16, n)
    s3 = _vec(scale_hm_ptr + hm, 32, n)
    s4 = _vec(scale_hm_ptr + hm, 48, n)
    b1 = _vec(beta_hm_ptr + hm, 0, n)
    b2 = _vec(beta_hm_ptr + hm, 16, n)
    b3 = _vec(beta_hm_ptr + hm, 32, n)
    b4 = _vec(beta_hm_ptr + hm, 48, n)

    # w = (Ai diag(beta exp(g))) k
    x11h, x11l = _split(ai11 * s1[None, :], SPLIT)
    x21h, x21l = _split(ai21 * s1[None, :], SPLIT)
    x22h, x22l = _split(ai22 * s2[None, :], SPLIT)
    x31h, x31l = _split(ai31 * s1[None, :], SPLIT)
    x32h, x32l = _split(ai32 * s2[None, :], SPLIT)
    x33h, x33l = _split(ai33 * s3[None, :], SPLIT)
    x41h, x41l = _split(ai41 * s1[None, :], SPLIT)
    x42h, x42l = _split(ai42 * s2[None, :], SPLIT)
    x43h, x43l = _split(ai43 * s3[None, :], SPLIT)
    x44h, x44l = _split(ai44 * s4[None, :], SPLIT)
    k_base = k_ptr + (bos * Hg + i_h // (H // Hg)) * K
    w_base = w_ptr + (bos * H + i_h) * K
    for c in tl.static_range(NC):
        _wy(
            k_base,
            Hg * K,
            w_base,
            H * K,
            t0,
            T_seq,
            c * (K // NC) + tl.arange(0, K // NC),
            x11h,
            x11l,
            x21h,
            x21l,
            x22h,
            x22l,
            x31h,
            x31l,
            x32h,
            x32l,
            x33h,
            x33l,
            x41h,
            x41l,
            x42h,
            x42l,
            x43h,
            x43l,
            x44h,
            x44l,
            SPLIT,
            K // NC,
        )
    # u = (Ai diag(beta)) v
    x11h, x11l = _split(ai11 * b1[None, :], SPLIT)
    x21h, x21l = _split(ai21 * b1[None, :], SPLIT)
    x22h, x22l = _split(ai22 * b2[None, :], SPLIT)
    x31h, x31l = _split(ai31 * b1[None, :], SPLIT)
    x32h, x32l = _split(ai32 * b2[None, :], SPLIT)
    x33h, x33l = _split(ai33 * b3[None, :], SPLIT)
    x41h, x41l = _split(ai41 * b1[None, :], SPLIT)
    x42h, x42l = _split(ai42 * b2[None, :], SPLIT)
    x43h, x43l = _split(ai43 * b3[None, :], SPLIT)
    x44h, x44l = _split(ai44 * b4[None, :], SPLIT)
    v_base = v_ptr + (bos * H + i_h) * V
    u_base = u_ptr + (bos * H + i_h) * V
    for c in tl.static_range(NC):
        _wy(
            v_base,
            H * V,
            u_base,
            H * V,
            t0,
            T_seq,
            c * (V // NC) + tl.arange(0, V // NC),
            x11h,
            x11l,
            x21h,
            x21l,
            x22h,
            x22l,
            x31h,
            x31l,
            x32h,
            x32l,
            x33h,
            x33l,
            x41h,
            x41l,
            x42h,
            x42l,
            x43h,
            x43l,
            x44h,
            x44l,
            SPLIT,
            V // NC,
        )


def fused_inverse_recompute(
    k: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    beta_hm: torch.Tensor,
    scale_hm: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
    plan=None,
):
    """w and u of every 64-token chunk from its A. ``plan``: see `fused_cumsum_kkt`.

    Args:
        k: [B, T, Hg, K] bf16
        v: [B, T, H, V] bf16
        A: [B, T, H, chunk_size] fp32, strictly lower triangular
        beta_hm, scale_hm: [B, H, T] fp32, beta and beta * exp(g_cumsum)

    Returns:
        w: [B, T, H, K], u: [B, T, H, V] (k's dtype)
    """
    assert chunk_size == 64
    B, T, H, _ = A.shape
    Hg, K = k.shape[2], k.shape[3]
    V = v.shape[3]
    nt_ptr = None
    if plan is not None:
        chunk_indices, NT, nt_ptr = plan.chunk_indices_64, plan.nt64_max, plan.nt64
    elif cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, chunk_size)
    w = torch.empty(B, T, H, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    # gfx1250 sweep (T = 16k, 30 sequences, 32 heads; cold caches): one warp with a
    # 2-wave register budget and 4 column slices, 118 us; 2 / 4 / 8 warps 150-650 us
    _fused_inverse_recompute_kernel[(NT, B * H)](
        k,
        v,
        A,
        beta_hm,
        scale_hm,
        w,
        u,
        cu_seqlens,
        chunk_indices,
        nt_ptr,
        T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=chunk_size,
        NC=4,
        SPLIT=True,
        num_warps=1,
        num_stages=1,
        waves_per_eu=2,
    )
    return w, u
