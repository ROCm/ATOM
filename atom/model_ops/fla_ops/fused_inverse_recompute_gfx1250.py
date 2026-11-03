# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`fused_inverse_recompute` as a Gluon kernel for gfx1250.

Per 64-token chunk and value head, from the strictly lower triangular A:

    Ai = (I + A)^-1,   w = Ai diag(beta exp(g_cumsum)) k,   u = Ai diag(beta) v

Ai is built in 16 x 16 blocks with fp32 WMMA (16x16x4) as in the Triton kernel:
each diagonal block as (I + N)(I + N^2)(I + N^4)(I + N^8), N = -A_ii, the
off-diagonal blocks by block forward substitution. The A tiles load straight
into the WMMA layout. The scaled Ai goes into the bf16 products as a high and a
low bf16 part. Two 16 x 16 blocks of a row of X = Ai diag(s) make one
[16, 32] A operand of a 16x16x32 WMMA, against two 16-row blocks of k or v.
The tensor data mover brings the chunk's k and v rows into LDS while the
inverse runs; rows past the sequence read 0.
"""

import functools

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .index import prepare_chunk_indices


@gluon.jit
def _mm(x, y, acc, OA: gl.constexpr, OB: gl.constexpr):
    """acc + x @ y for fp32 16 x 16 tiles."""
    return gl.amd.gfx1250.wmma(gl.convert_layout(x, OA), gl.convert_layout(y, OB), acc)


@gluon.jit
def _inv16(a, eye, z, OA: gl.constexpr, OB: gl.constexpr):
    """(I + a)^-1 for a strictly lower triangular 16 x 16 fp32 tile."""
    n1 = -a
    n2 = _mm(n1, n1, z, OA, OB)
    n4 = _mm(n2, n2, z, OA, OB)
    n8 = _mm(n4, n4, z, OA, OB)
    p = eye + n1 + n2 + _mm(n1, n2, z, OA, OB)
    p = p + _mm(p, n4, z, OA, OB)
    return p + _mm(p, n8, z, OA, OB)


@gluon.jit
def _a_tile(a_base, row0, col0, n, stride, L: gl.constexpr):
    """[16, 16] fp32 tile of A at (row0, col0); rows at or past n read 0 (buffer
    loads: a masked lane reads out of range, with no branch)."""
    r = row0 + gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    c = col0 + gl.arange(0, 16, layout=gl.SliceLayout(0, L))
    return gl.amd.gfx1250.buffer_load(
        a_base,
        r[:, None] * stride + c[None, :],
        mask=(r < n)[:, None] & (c >= 0)[None, :],
        other=0.0,
    )


@gluon.jit
def _pair(x, y, SPLIT: gl.constexpr, OA16: gl.constexpr):
    """[x | y] as bf16 A operands [16, 32]: the high part and the bf16 rounding
    of what it leaves."""
    xy = gl.reshape(gl.permute(gl.join(x, y), [0, 2, 1]), [16, 32])
    hi = xy.to(gl.bfloat16)
    lo = (xy - hi.to(gl.float32)).to(gl.bfloat16)
    if not SPLIT:
        lo = hi
    return gl.convert_layout(hi, OA16), gl.convert_layout(lo, OA16)


@gluon.jit
def _wy(
    src_smem,
    dst,
    s_dst,
    n,
    p11h,
    p11l,
    p21h,
    p21l,
    p31h,
    p31l,
    p33h,
    p33l,
    p41h,
    p41l,
    p43h,
    p43l,
    N: gl.constexpr,
    NC: gl.constexpr,
    L16: gl.constexpr,
    OB16: gl.constexpr,
):
    """dst rows = X src over the chunk, NC column slices of N; src [64, NC * N]
    in LDS; X given as its row pairs (row i: [X_i1 | X_i2] and [X_i3 | X_i4],
    high and low parts)."""
    rows = gl.arange(0, 16, layout=gl.SliceLayout(1, L16))
    z = gl.zeros([16, N], gl.float32, layout=L16)
    for cs in gl.static_range(NC):
        cols = cs * N + gl.arange(0, N, layout=gl.SliceLayout(0, L16))
        b12 = src_smem.slice(0, 32, dim=0).slice(cs * N, N, dim=1).load(OB16)
        b34 = src_smem.slice(32, 32, dim=0).slice(cs * N, N, dim=1).load(OB16)
        o = dst + rows[:, None] * s_dst + cols[None, :]
        acc = gl.amd.gfx1250.wmma(p11l, b12, gl.amd.gfx1250.wmma(p11h, b12, z))
        gl.store(o, acc.to(dst.dtype.element_ty), mask=(rows < n)[:, None])
        acc = gl.amd.gfx1250.wmma(p21l, b12, gl.amd.gfx1250.wmma(p21h, b12, z))
        gl.store(
            o + 16 * s_dst, acc.to(dst.dtype.element_ty), mask=(rows + 16 < n)[:, None]
        )
        acc = gl.amd.gfx1250.wmma(p31l, b12, gl.amd.gfx1250.wmma(p31h, b12, z))
        acc = gl.amd.gfx1250.wmma(p33l, b34, gl.amd.gfx1250.wmma(p33h, b34, acc))
        gl.store(
            o + 32 * s_dst, acc.to(dst.dtype.element_ty), mask=(rows + 32 < n)[:, None]
        )
        acc = gl.amd.gfx1250.wmma(p41l, b12, gl.amd.gfx1250.wmma(p41h, b12, z))
        acc = gl.amd.gfx1250.wmma(p43l, b34, gl.amd.gfx1250.wmma(p43h, b34, acc))
        gl.store(
            o + 48 * s_dst, acc.to(dst.dtype.element_ty), mask=(rows + 48 < n)[:, None]
        )


@gluon.jit
def _fused_inverse_recompute_gfx1250_kernel(
    chunk_indices,
    cu_seqlens,
    nt_ptr,
    A_ptr,
    scale_hm_ptr,
    beta_hm_ptr,
    k_ptr,
    v_ptr,
    w_ptr,
    u_ptr,
    T,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    NK: gl.constexpr,
    NV: gl.constexpr,
    SPLIT: gl.constexpr,
    HAS_NT: gl.constexpr,
    L: gl.constexpr,
    OA: gl.constexpr,
    OB: gl.constexpr,
    L16: gl.constexpr,
    OA16: gl.constexpr,
    OB16: gl.constexpr,
    SMEM_K: gl.constexpr,
    SMEM_V: gl.constexpr,
):
    pid_t = gl.program_id(0)
    i_h = gl.program_id(1)
    if HAS_NT:  # noqa: SIM102 (a constexpr, then a device load)
        # a bucket-shaped grid: programs past the step's chunk count have no work
        if pid_t >= gl.load(nt_ptr):
            return
    i_n = gl.load(chunk_indices + pid_t * 2).to(gl.int32)
    t0 = gl.load(chunk_indices + pid_t * 2 + 1).to(gl.int32) * 64
    bos = gl.load(cu_seqlens + i_n).to(gl.int32)
    n = gl.load(cu_seqlens + i_n + 1).to(gl.int32) - bos - t0
    row0 = (bos + t0).to(gl.int64)

    # the chunk's k and v rows into LDS while the inverse runs
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr + (row0 * Hg + i_h // (H // Hg)) * K,
        shape=(n, K),
        strides=(Hg * K, 1),
        block_shape=(64, K),
        layout=SMEM_K,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr + (row0 * H + i_h) * V,
        shape=(n, V),
        strides=(H * V, 1),
        block_shape=(64, V),
        layout=SMEM_V,
    )
    k_smem = gl.allocate_shared_memory(k_ptr.type.element_ty, [64, K], SMEM_K)
    v_smem = gl.allocate_shared_memory(v_ptr.type.element_ty, [64, V], SMEM_V)
    gl.amd.gfx1250.tdm.async_load(k_desc, [0, 0], k_smem)
    gl.amd.gfx1250.tdm.async_load(v_desc, [0, 0], v_smem)

    # A in 16 x 16 tiles, straight into the WMMA layout
    a_base = A_ptr + (row0 * H + i_h) * 64
    a11 = _a_tile(a_base, 0, 0, n, H * 64, L)
    a21 = _a_tile(a_base, 16, 0, n, H * 64, L)
    a22 = _a_tile(a_base, 16, 16, n, H * 64, L)
    a31 = _a_tile(a_base, 32, 0, n, H * 64, L)
    a32 = _a_tile(a_base, 32, 16, n, H * 64, L)
    a33 = _a_tile(a_base, 32, 32, n, H * 64, L)
    a41 = _a_tile(a_base, 48, 0, n, H * 64, L)
    a42 = _a_tile(a_base, 48, 16, n, H * 64, L)
    a43 = _a_tile(a_base, 48, 32, n, H * 64, L)
    a44 = _a_tile(a_base, 48, 48, n, H * 64, L)
    # the column scales of each block, head-major runs
    cols = gl.arange(0, 16, layout=gl.SliceLayout(0, L))
    hm = i_h.to(gl.int64) * T + row0 + cols
    s1 = gl.load(scale_hm_ptr + hm, mask=cols < n, other=0.0)
    s2 = gl.load(scale_hm_ptr + hm + 16, mask=cols + 16 < n, other=0.0)
    s3 = gl.load(scale_hm_ptr + hm + 32, mask=cols + 32 < n, other=0.0)
    s4 = gl.load(scale_hm_ptr + hm + 48, mask=cols + 48 < n, other=0.0)
    b1 = gl.load(beta_hm_ptr + hm, mask=cols < n, other=0.0)
    b2 = gl.load(beta_hm_ptr + hm + 16, mask=cols + 16 < n, other=0.0)
    b3 = gl.load(beta_hm_ptr + hm + 32, mask=cols + 32 < n, other=0.0)
    b4 = gl.load(beta_hm_ptr + hm + 48, mask=cols + 48 < n, other=0.0)

    # Ai = (I + A)^-1 in 16 x 16 blocks
    rows = gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    eye = (rows[:, None] == cols[None, :]).to(gl.float32)
    z = gl.zeros([16, 16], gl.float32, layout=L)
    ai11 = _inv16(a11, eye, z, OA, OB)
    ai22 = _inv16(a22, eye, z, OA, OB)
    ai33 = _inv16(a33, eye, z, OA, OB)
    ai44 = _inv16(a44, eye, z, OA, OB)
    ai21 = -_mm(_mm(ai22, a21, z, OA, OB), ai11, z, OA, OB)
    ai32 = -_mm(_mm(ai33, a32, z, OA, OB), ai22, z, OA, OB)
    ai43 = -_mm(_mm(ai44, a43, z, OA, OB), ai33, z, OA, OB)
    ai31 = -_mm(ai33, _mm(a31, ai11, z, OA, OB) + _mm(a32, ai21, z, OA, OB), z, OA, OB)
    ai42 = -_mm(ai44, _mm(a42, ai22, z, OA, OB) + _mm(a43, ai32, z, OA, OB), z, OA, OB)
    ai41 = -_mm(
        ai44,
        _mm(a41, ai11, z, OA, OB)
        + _mm(a42, ai21, z, OA, OB)
        + _mm(a43, ai31, z, OA, OB),
        z,
        OA,
        OB,
    )

    # w = (Ai diag(beta exp(g))) k
    p11h, p11l = _pair(ai11 * s1[None, :], z, SPLIT, OA16)
    p21h, p21l = _pair(ai21 * s1[None, :], ai22 * s2[None, :], SPLIT, OA16)
    p31h, p31l = _pair(ai31 * s1[None, :], ai32 * s2[None, :], SPLIT, OA16)
    p33h, p33l = _pair(ai33 * s3[None, :], z, SPLIT, OA16)
    p41h, p41l = _pair(ai41 * s1[None, :], ai42 * s2[None, :], SPLIT, OA16)
    p43h, p43l = _pair(ai43 * s3[None, :], ai44 * s4[None, :], SPLIT, OA16)
    gl.amd.gfx1250.tdm.async_wait(1)
    gl.barrier()
    _wy(
        k_smem,
        w_ptr + (row0 * H + i_h) * K,
        H * K,
        n,
        p11h,
        p11l,
        p21h,
        p21l,
        p31h,
        p31l,
        p33h,
        p33l,
        p41h,
        p41l,
        p43h,
        p43l,
        K // NK,
        NK,
        L16,
        OB16,
    )
    # u = (Ai diag(beta)) v
    p11h, p11l = _pair(ai11 * b1[None, :], z, SPLIT, OA16)
    p21h, p21l = _pair(ai21 * b1[None, :], ai22 * b2[None, :], SPLIT, OA16)
    p31h, p31l = _pair(ai31 * b1[None, :], ai32 * b2[None, :], SPLIT, OA16)
    p33h, p33l = _pair(ai33 * b3[None, :], z, SPLIT, OA16)
    p41h, p41l = _pair(ai41 * b1[None, :], ai42 * b2[None, :], SPLIT, OA16)
    p43h, p43l = _pair(ai43 * b3[None, :], ai44 * b4[None, :], SPLIT, OA16)
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    _wy(
        v_smem,
        u_ptr + (row0 * H + i_h) * V,
        H * V,
        n,
        p11h,
        p11l,
        p21h,
        p21l,
        p31h,
        p31l,
        p33h,
        p33l,
        p41h,
        p41l,
        p43h,
        p43l,
        V // NV,
        NV,
        L16,
        OB16,
    )


@functools.cache
def _layouts(K, V):
    L = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[], instr_shape=[16, 16, 4]
    )
    L16 = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[], instr_shape=[16, 16, 32]
    )
    return (
        L,
        gl.DotOperandLayout(operand_index=0, parent=L, k_width=2),
        gl.DotOperandLayout(operand_index=1, parent=L, k_width=2),
        L16,
        gl.DotOperandLayout(operand_index=0, parent=L16, k_width=8),
        gl.DotOperandLayout(operand_index=1, parent=L16, k_width=8),
        gl.PaddedSharedLayout.with_identity_for([[K, 16]], [64, K], [1, 0]),
        gl.PaddedSharedLayout.with_identity_for([[V, 16]], [64, V], [1, 0]),
    )


def fused_inverse_recompute_gfx1250(
    k: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    beta_hm: torch.Tensor,
    scale_hm: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan,
):
    """w and u of every 64-token chunk from its A, as `fused_inverse_recompute`
    (varlen, B = 1).

    Args:
        k: [1, T, Hg, K] bf16, v: [1, T, H, V] bf16
        A: [1, T, H, 64] fp32, strictly lower triangular
        beta_hm, scale_hm: [1, H, T] fp32, beta and beta * exp(g_cumsum)
        cu_seqlens: [N + 1]; plan: see `fused_cumsum_kkt`

    Returns:
        w: [1, T, H, K], u: [1, T, H, V] (k's dtype)
    """
    _, T, H, _ = A.shape
    Hg, K, V = k.shape[2], k.shape[3], v.shape[3]
    assert k.is_contiguous() and v.is_contiguous() and A.is_contiguous()
    nt_ptr = None
    if plan is not None:
        chunk_indices, NT, nt_ptr = plan.chunk_indices_64, plan.nt64_max, plan.nt64
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
        NT = len(chunk_indices)
    w = torch.empty(1, T, H, K, device=k.device, dtype=k.dtype)
    u = torch.empty(1, T, H, V, device=k.device, dtype=k.dtype)
    L, OA, OB, L16, OA16, OB16, SMEM_K, SMEM_V = _layouts(K, V)
    _fused_inverse_recompute_gfx1250_kernel[(NT, H)](
        chunk_indices,
        cu_seqlens,
        nt_ptr,
        A,
        scale_hm,
        beta_hm,
        k,
        v,
        w,
        u,
        T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        NK=2,
        NV=2,
        SPLIT=True,
        HAS_NT=nt_ptr is not None,
        L=L,
        OA=OA,
        OB=OB,
        L16=L16,
        OA16=OA16,
        OB16=OB16,
        SMEM_K=SMEM_K,
        SMEM_V=SMEM_V,
        num_warps=1,
    )
    return w, u
