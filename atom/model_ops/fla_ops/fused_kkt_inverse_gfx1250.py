# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gating, KKT and inverse/WY without a global A intermediate."""

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
def _add(x, y):
    return x + y


@gluon.jit
def _quarters(x):
    x = gl.permute(gl.reshape(x, [4, 16]), [1, 0])
    x = gl.reshape(x, [16, 2, 2])
    x = gl.convert_layout(
        x, gl.BlockedLayout([1, 2, 2], [32, 1, 1], [1, 1, 1], [0, 1, 2])
    )
    even, odd = gl.split(x)
    x0, x2 = gl.split(even)
    x1, x3 = gl.split(odd)
    return x0, x1, x2, x3


@gluon.jit
def _kkt(
    kr,
    kc,
    gr,
    gc,
    beta,
    row0: gl.constexpr,
    col0: gl.constexpr,
    L: gl.constexpr,
    L16: gl.constexpr,
):
    kk = gl.amd.gfx1250.wmma(kr, kc, gl.zeros([16, 16], gl.float32, L16))
    kk = gl.convert_layout(kk, L)
    rows = row0 + gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    cols = col0 + gl.arange(0, 16, layout=gl.SliceLayout(0, L))
    gr = gl.convert_layout(gr, gl.SliceLayout(1, L))
    gc = gl.convert_layout(gc, gl.SliceLayout(0, L))
    beta = gl.convert_layout(beta, gl.SliceLayout(1, L))
    lower = rows[:, None] > cols[None, :]
    return gl.where(
        lower,
        kk * gl.exp(gl.where(lower, gr[:, None] - gc[None, :], 0.0)) * beta[:, None],
        0.0,
    )


@gluon.jit
def _pair(x, y, OA16: gl.constexpr):
    """Split [x | y] into bf16 high and residual [16, 32] WMMA operands."""
    xy = gl.reshape(gl.permute(gl.join(x, y), [0, 2, 1]), [16, 32])
    hi = xy.to(gl.bfloat16)
    lo = (xy - hi.to(gl.float32)).to(gl.bfloat16)
    return gl.convert_layout(hi, OA16), gl.convert_layout(lo, OA16)


@gluon.jit
def _wy(
    src_smem,
    dst,
    stride_dst_m,
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
    SMEM: gl.constexpr,
):
    """Multiply split-bf16 inverse row pairs by the LDS tile and store dst."""
    z = gl.zeros([16, N], gl.float32, layout=L16)
    for cs in gl.static_range(NC):
        b12 = src_smem.slice(0, 32, dim=0).slice(cs * N, N, dim=1).load(OB16)
        b34 = src_smem.slice(32, 32, dim=0).slice(cs * N, N, dim=1).load(OB16)
        acc = gl.amd.gfx1250.wmma(p11l, b12, gl.amd.gfx1250.wmma(p11h, b12, z))
        src_smem.slice(0, 16, dim=0).slice(cs * N, N, dim=1).store(
            acc.to(src_smem.dtype)
        )
        acc = gl.amd.gfx1250.wmma(p21l, b12, gl.amd.gfx1250.wmma(p21h, b12, z))
        src_smem.slice(16, 16, dim=0).slice(cs * N, N, dim=1).store(
            acc.to(src_smem.dtype)
        )
        acc = gl.amd.gfx1250.wmma(p31l, b12, gl.amd.gfx1250.wmma(p31h, b12, z))
        acc = gl.amd.gfx1250.wmma(p33l, b34, gl.amd.gfx1250.wmma(p33h, b34, acc))
        src_smem.slice(32, 16, dim=0).slice(cs * N, N, dim=1).store(
            acc.to(src_smem.dtype)
        )
        acc = gl.amd.gfx1250.wmma(p41l, b12, gl.amd.gfx1250.wmma(p41h, b12, z))
        acc = gl.amd.gfx1250.wmma(p43l, b34, gl.amd.gfx1250.wmma(p43h, b34, acc))
        src_smem.slice(48, 16, dim=0).slice(cs * N, N, dim=1).store(
            acc.to(src_smem.dtype)
        )
    dst_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=dst,
        shape=(n, NC * N),
        strides=(stride_dst_m, 1),
        block_shape=(64, NC * N),
        layout=SMEM,
    )
    gl.barrier()
    gl.amd.gfx1250.tdm.async_store(dst_desc, [0, 0], src_smem)


@gluon.jit
def _fused_kkt_inverse_gfx1250_kernel(
    chunk_indices,
    cu_seqlens,
    nt_ptr,
    a_ptr,
    b_ptr,
    A_log,
    dt_bias,
    g_cumsum_ptr,
    stride_ab,
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
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_V: gl.constexpr,
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
    pid_m = gl.program_id(0)
    pid_h = gl.program_id(1)
    if HAS_NT:  # noqa: SIM102 (a constexpr, then a device load)
        # Padded programs must return before reading chunk metadata.
        if pid_m >= gl.load(nt_ptr):
            return
    i_n = gl.load(chunk_indices + pid_m * 2).to(gl.int32)
    t0 = gl.load(chunk_indices + pid_m * 2 + 1).to(gl.int32) * BLOCK_M
    bos = gl.load(cu_seqlens + i_n).to(gl.int32)
    M = gl.load(cu_seqlens + i_n + 1).to(gl.int32) - bos - t0
    start_m = (bos + t0).to(gl.int64)

    # Load K first; the same LDS tile holds V after W is stored.
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr + (start_m * Hg + pid_h // (H // Hg)) * K,
        shape=(M, K),
        strides=(Hg * K, 1),
        block_shape=(BLOCK_M, K),
        layout=SMEM_K,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr + (start_m * H + pid_h) * V,
        shape=(M, V),
        strides=(H * V, 1),
        block_shape=(BLOCK_M, V),
        layout=SMEM_V,
    )
    k_smem = gl.allocate_shared_memory(k_ptr.type.element_ty, [BLOCK_M, K], SMEM_K)
    gl.amd.gfx1250.tdm.async_load(k_desc, [0, 0], k_smem)

    # The gates stay in fp32, with the original bf16 rounding of beta.
    LG: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offs_m = gl.arange(0, BLOCK_M, layout=LG)
    mask_m = offs_m < M
    off_ab = (start_m + offs_m) * stride_ab + pid_h
    x = gl.load(a_ptr + off_ab, mask=mask_m, other=0.0).to(gl.float32)
    x = x + gl.load(dt_bias + pid_h).to(gl.float32)
    softplus = gl.where(x <= 20.0, gl.log(1 + gl.exp(x)), x)
    gate = -gl.exp(gl.load(A_log + pid_h).to(gl.float32)) * softplus
    gc = gl.associative_scan(gl.where(mask_m, gate, 0.0), 0, _add)
    bv = gl.load(b_ptr + off_ab, mask=mask_m, other=0.0).to(gl.float32)
    beta = gl.where(
        mask_m, (1.0 / (1.0 + gl.exp(-bv))).to(gl.bfloat16).to(gl.float32), 0.0
    )
    scale = beta * gl.exp(gc)
    SG: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    gs = gl.allocate_shared_memory(gl.float32, [BLOCK_M, 1], SG)
    bs = gl.allocate_shared_memory(gl.float32, [1, BLOCK_M], SG)
    ss = gl.allocate_shared_memory(gl.float32, [1, BLOCK_M], SG)
    gs.store(
        gl.convert_layout(
            gc, gl.SliceLayout(1, gl.BlockedLayout([1, 1], [32, 1], [1, 1], [0, 1]))
        )[:, None]
    )
    bs.store(
        gl.convert_layout(
            beta, gl.SliceLayout(0, gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0]))
        )[None, :]
    )
    ss.store(
        gl.convert_layout(
            scale, gl.SliceLayout(0, gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0]))
        )[None, :]
    )
    gd = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=g_cumsum_ptr + start_m * H + pid_h,
        shape=(M, 1),
        strides=(H, 1),
        block_shape=(BLOCK_M, 1),
        layout=SG,
    )
    bd = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=beta_hm_ptr + pid_h.to(gl.int64) * T + start_m,
        shape=(1, M),
        strides=(T, 1),
        block_shape=(1, BLOCK_M),
        layout=SG,
    )
    sd = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=scale_hm_ptr + pid_h.to(gl.int64) * T + start_m,
        shape=(1, M),
        strides=(T, 1),
        block_shape=(1, BLOCK_M),
        layout=SG,
    )
    gl.barrier()
    g1, g2, g3, g4 = _quarters(gc)
    b1, b2, b3, b4 = _quarters(beta)
    s1, s2, s3, s4 = _quarters(scale)

    # The K tile supplies KKT and the later WY product.
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    k1 = k_smem.slice(0, 16, dim=0).load(OA16)
    k2 = k_smem.slice(16, 16, dim=0).load(OA16)
    k3 = k_smem.slice(32, 16, dim=0).load(OA16)
    k4 = k_smem.slice(48, 16, dim=0).load(OA16)
    kt1 = gl.convert_layout(gl.permute(k1, [1, 0]), OB16)
    kt2 = gl.convert_layout(gl.permute(k2, [1, 0]), OB16)
    kt3 = gl.convert_layout(gl.permute(k3, [1, 0]), OB16)
    kt4 = gl.convert_layout(gl.permute(k4, [1, 0]), OB16)
    gl.amd.gfx1250.tdm.async_store(gd, [0, 0], gs)
    gl.amd.gfx1250.tdm.async_store(bd, [0, 0], bs)
    gl.amd.gfx1250.tdm.async_store(sd, [0, 0], ss)
    a11 = _kkt(k1, kt1, g1, g1, b1, 0, 0, L, L16)
    a21 = _kkt(k2, kt1, g2, g1, b2, 16, 0, L, L16)
    a22 = _kkt(k2, kt2, g2, g2, b2, 16, 16, L, L16)
    a31 = _kkt(k3, kt1, g3, g1, b3, 32, 0, L, L16)
    a32 = _kkt(k3, kt2, g3, g2, b3, 32, 16, L, L16)
    a33 = _kkt(k3, kt3, g3, g3, b3, 32, 32, L, L16)
    a41 = _kkt(k4, kt1, g4, g1, b4, 48, 0, L, L16)
    a42 = _kkt(k4, kt2, g4, g2, b4, 48, 16, L, L16)
    a43 = _kkt(k4, kt3, g4, g3, b4, 48, 32, L, L16)
    a44 = _kkt(k4, kt4, g4, g4, b4, 48, 48, L, L16)
    offs_n = gl.arange(0, 16, layout=gl.SliceLayout(0, L))
    b1 = gl.convert_layout(b1, gl.SliceLayout(0, L))
    b2 = gl.convert_layout(b2, gl.SliceLayout(0, L))
    b3 = gl.convert_layout(b3, gl.SliceLayout(0, L))
    b4 = gl.convert_layout(b4, gl.SliceLayout(0, L))
    s1 = gl.convert_layout(s1, gl.SliceLayout(0, L))
    s2 = gl.convert_layout(s2, gl.SliceLayout(0, L))
    s3 = gl.convert_layout(s3, gl.SliceLayout(0, L))
    s4 = gl.convert_layout(s4, gl.SliceLayout(0, L))

    # Ai = (I + A)^-1 in 16 x 16 blocks
    offs_m = gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    eye = (offs_m[:, None] == offs_n[None, :]).to(gl.float32)
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
    p11h, p11l = _pair(ai11 * s1[None, :], z, OA16)
    p21h, p21l = _pair(ai21 * s1[None, :], ai22 * s2[None, :], OA16)
    p31h, p31l = _pair(ai31 * s1[None, :], ai32 * s2[None, :], OA16)
    p33h, p33l = _pair(ai33 * s3[None, :], z, OA16)
    p41h, p41l = _pair(ai41 * s1[None, :], ai42 * s2[None, :], OA16)
    p43h, p43l = _pair(ai43 * s3[None, :], ai44 * s4[None, :], OA16)
    gl.amd.gfx1250.tdm.async_wait(0)
    gs._keep_alive()
    bs._keep_alive()
    ss._keep_alive()
    gl.barrier()
    _wy(
        k_smem,
        w_ptr + (start_m * H + pid_h) * K,
        H * K,
        M,
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
        BLOCK_K,
        K // BLOCK_K,
        L16,
        OB16,
        SMEM_K,
    )
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.amd.gfx1250.tdm.async_load(v_desc, [0, 0], k_smem)
    # u = (Ai diag(beta)) v
    p11h, p11l = _pair(ai11 * b1[None, :], z, OA16)
    p21h, p21l = _pair(ai21 * b1[None, :], ai22 * b2[None, :], OA16)
    p31h, p31l = _pair(ai31 * b1[None, :], ai32 * b2[None, :], OA16)
    p33h, p33l = _pair(ai33 * b3[None, :], z, OA16)
    p41h, p41l = _pair(ai41 * b1[None, :], ai42 * b2[None, :], OA16)
    p43h, p43l = _pair(ai43 * b3[None, :], ai44 * b4[None, :], OA16)
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    _wy(
        k_smem,
        u_ptr + (start_m * H + pid_h) * V,
        H * V,
        M,
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
        BLOCK_V,
        V // BLOCK_V,
        L16,
        OB16,
        SMEM_V,
    )
    gl.amd.gfx1250.tdm.async_wait(0)


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


def fused_kkt_inverse_gfx1250(k, v, cu_seqlens, plan, gating):
    """Compute cumulative gates and WY factors for packed bf16 inputs.

    k/v are contiguous [1, T, 16/32, 128]; cu_seqlens marks sequence bounds.
    plan is the padded chunk plan or None; gating is (a, b, A_log, dt_bias),
    with bf16 a/b sharing a row stride and contiguous head columns.
    Return fp32 g [1, T, 32], bf16 w/u [1, T, 32, 128], and fp32
    beta/scale [1, 32, T]. Requires gfx1250 and 64-token chunks.
    """
    a, b, A_log, dt_bias = gating
    _, T, Hg, K = k.shape
    H, V = v.shape[2], v.shape[3]
    assert k.is_contiguous() and v.is_contiguous()
    assert k.shape[0] == 1 and a.stride(1) == 1 and b.stride(1) == 1
    assert a.stride(0) == b.stride(0)
    nt_ptr = None
    if plan is not None:
        chunk_indices, NT, nt_ptr = plan.chunk_indices_64, plan.nt64_max, plan.nt64
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
        NT = len(chunk_indices)
    g_cumsum = torch.empty(1, T, H, device=k.device, dtype=torch.float32)
    beta_hm = torch.empty(1, H, T, device=k.device, dtype=torch.float32)
    scale_hm = torch.empty(1, H, T, device=k.device, dtype=torch.float32)
    w = torch.empty(1, T, H, K, device=k.device, dtype=k.dtype)
    u = torch.empty(1, T, H, V, device=k.device, dtype=k.dtype)
    L, OA, OB, L16, OA16, OB16, SMEM_K, SMEM_V = _layouts(K, V)
    _fused_kkt_inverse_gfx1250_kernel[(NT, H)](
        chunk_indices,
        cu_seqlens,
        nt_ptr,
        a,
        b,
        A_log,
        dt_bias,
        g_cumsum,
        a.stride(0),
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
        BLOCK_M=64,
        BLOCK_K=16,
        BLOCK_V=16,
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
    return g_cumsum, w, u, beta_hm, scale_hm
