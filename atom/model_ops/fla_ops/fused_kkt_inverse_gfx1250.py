# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gated delta rule prefill on gfx1250: per 64-token chunk and value head, the gates, A = beta k k^T exp(g_i - g_j),
(I + A)^-1 and the WY factors w, u in one kernel (A stays on chip)."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .index import prepare_chunk_indices


@gluon.jit
def _add(x, y):
    return x + y


@gluon.jit
def _kk(kr, kc, L16: gl.constexpr):
    return gl.amd.gfx1250.wmma(kr, kc, gl.zeros([16, 16], gl.float32, L16))


@gluon.jit
def _gate(kk, gr, gc, beta, row0: gl.constexpr, col0: gl.constexpr, L: gl.constexpr):
    """beta_i exp(g_i - g_j) kk below the diagonal."""
    rows = row0 + gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    cols = col0 + gl.arange(0, 16, layout=gl.SliceLayout(0, L))
    lower = rows[:, None] > cols[None, :]
    return gl.where(
        lower,
        kk * gl.exp(gl.where(lower, gr[:, None] - gc[None, :], 0.0)) * beta[:, None],
        0.0,
    )


@gluon.jit
def _gate_fast(
    kk,
    sr,
    ec,
    row0: gl.constexpr,
    col0: gl.constexpr,
    DIAG: gl.constexpr,
    L: gl.constexpr,
):
    """beta_i exp(g_i - g_j) kk as scale_i exp(-g_j) kk."""
    a = kk * (sr[:, None] * ec[None, :])
    if DIAG:
        rows = row0 + gl.arange(0, 16, layout=gl.SliceLayout(1, L))
        cols = col0 + gl.arange(0, 16, layout=gl.SliceLayout(0, L))
        a = gl.where(rows[:, None] > cols[None, :], a, 0.0)
    return a


@gluon.jit
def _wm(a0, a1, b0t, b1t, acc, OA16: gl.constexpr, OB16: gl.constexpr):
    """acc + [a0 | a1] @ [b0; b1] for bf16 16 x 16 tiles, the B blocks given transposed."""
    a = gl.reshape(gl.permute(gl.join(a0, a1), [0, 2, 1]), [16, 32])
    b = gl.reshape(gl.permute(gl.join(b0t, b1t), [2, 1, 0]), [32, 16])
    return gl.amd.gfx1250.wmma(
        gl.convert_layout(a, OA16), gl.convert_layout(b, OB16), acc
    )


@gluon.jit
def _split(x):
    """x as bf16 high and residual parts."""
    h = x.to(gl.bfloat16)
    return h, (x - h.to(gl.float32)).to(gl.bfloat16)


@gluon.jit
def _tr(x, buf, L16: gl.constexpr):
    """x transposed through its own LDS tile."""
    buf.store(x)
    return buf.permute([1, 0]).load(L16)


@gluon.jit
def _bf(x):
    return x.to(gl.bfloat16)


@gluon.jit
def _f(x):
    return x.to(gl.float32)


@gluon.jit
def _dbl(x, xt, m, half, z, zb, OA16: gl.constexpr, OB16: gl.constexpr):
    """X_2S = X_S + X_S C X_S and its transpose, C = m between the S-row halves of each 2S block (half = 0/1)."""
    # yt = (C X)^T
    yt = _bf(_wm(xt, zb, m * half, zb, z, OA16, OB16))
    xn = _wm(x, zb, yt, zb, _f(x), OA16, OB16)
    xtn = _wm(yt, zb, x, zb, _f(xt), OA16, OB16)
    return _bf(xn), _bf(xtn)


@gluon.jit
def _diag(m, mt, eye, z, zb, h1, h2, h4, h8, OA16: gl.constexpr, OB16: gl.constexpr):
    """(I - m)^-1 of a strictly lower m and its transpose, by block doubling from 2 x 2 blocks."""
    x = _bf(eye + _f(m * h1))
    xt = _bf(eye + _f(mt * h1))
    x, xt = _dbl(x, xt, m, h2, z, zb, OA16, OB16)
    x, xt = _dbl(x, xt, m, h4, z, zb, OA16, OB16)
    return _dbl(x, xt, m, h8, z, zb, OA16, OB16)


@gluon.jit
def _half(rows, cols, S: gl.constexpr):
    """1 between the S-row halves of each 2S block, else 0, as bf16."""
    h = (rows[:, None] // (2 * S) == cols[None, :] // (2 * S)) & (
        rows[:, None] // S != cols[None, :] // S
    )
    return h.to(gl.bfloat16)


@gluon.jit
def _pair(x, y, OA16: gl.constexpr):
    """Split [x | y] into bf16 high and residual [16, 32] WMMA operands."""
    xy = gl.reshape(gl.permute(gl.join(x, y), [0, 2, 1]), [16, 32])
    # Move fp32 values into the operand layout first, so bf16 packing needs no 16-bit shuffles.
    xy = gl.convert_layout(xy, OA16)
    hi = xy.to(gl.bfloat16)
    lo = (xy - hi.to(gl.float32)).to(gl.bfloat16)
    return hi, lo


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
    decay_ptr,
    T,
    NT,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_V: gl.constexpr,
    HAS_NT: gl.constexpr,
    STORE_DECAY: gl.constexpr,
):
    L16: gl.constexpr = _layout("L16", K, V)
    OA16: gl.constexpr = _layout("OA16", K, V)
    OB16: gl.constexpr = _layout("OB16", K, V)
    SMEM_K: gl.constexpr = _layout("SMEM_K", K, V)
    SMEM_V: gl.constexpr = _layout("SMEM_V", K, V)
    pid_m = gl.program_id(0)
    pid_h = gl.program_id(1)
    # Padded programs must return before reading chunk metadata.
    if HAS_NT:
        if pid_m >= gl.load(nt_ptr):
            return
    elif pid_m >= NT:
        return
    seq = gl.load(chunk_indices + pid_m * 2).to(gl.int32)
    t0 = gl.load(chunk_indices + pid_m * 2 + 1).to(gl.int32) * BLOCK_M
    bos = gl.load(cu_seqlens + seq).to(gl.int32)
    M = gl.load(cu_seqlens + seq + 1).to(gl.int32) - bos - t0
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

    # The gates in fp32; beta is rounded to bf16 as in the unfused KKT kernel.
    LG: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    LV: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    offs_m = gl.arange(0, BLOCK_M, layout=LG)
    mask_m = offs_m < M
    # Rows past the chunk end read the last row; the masks below drop them.
    off_ab = (start_m + gl.minimum(offs_m, M - 1)) * stride_ab + pid_h
    xa = gl.load(a_ptr + off_ab)
    bv = gl.load(b_ptr + off_ab).to(gl.float32)
    x = xa.to(gl.float32) + gl.load(dt_bias + pid_h).to(gl.float32)
    softplus = gl.where(x <= 20.0, gl.log(1 + gl.exp(x)), x)
    gate = -gl.exp(gl.load(A_log + pid_h).to(gl.float32)) * softplus
    gc = gl.associative_scan(gl.where(mask_m, gate, 0.0), 0, _add)
    beta = gl.where(
        mask_m, (1.0 / (1.0 + gl.exp(-bv))).to(gl.bfloat16).to(gl.float32), 0.0
    )
    scale = beta * gl.exp(gc)
    # Chunks that decay by less than e^60 factor exp(g_i - g_j) into exp(g_i) exp(-g_j).
    fast = gl.min(gc, 0) >= -60.0
    SG: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    gs = gl.allocate_shared_memory(gl.float32, [BLOCK_M, 1], SG)
    bs = gl.allocate_shared_memory(gl.float32, [1, BLOCK_M], SG)
    ss = gl.allocate_shared_memory(gl.float32, [1, BLOCK_M], SG)
    es = gl.allocate_shared_memory(gl.float32, [1, BLOCK_M], SG)
    es.store(gl.convert_layout(gl.exp(-gc), gl.SliceLayout(0, LV))[None, :])
    gs.store(
        gl.convert_layout(
            gc, gl.SliceLayout(1, gl.BlockedLayout([1, 1], [32, 1], [1, 1], [0, 1]))
        )[:, None]
    )
    bs.store(gl.convert_layout(beta, gl.SliceLayout(0, LV))[None, :])
    ss.store(gl.convert_layout(scale, gl.SliceLayout(0, LV))[None, :])
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
    if STORE_DECAY:
        # The state walk's factors: exp(g_last - g_i) of the chunk's rows (0 past its end), then exp(g_last).
        g_last = gl.sum(gl.where(offs_m == gl.minimum(M, BLOCK_M) - 1, gc, 0.0), 0)
        dsm = gl.allocate_shared_memory(gl.float32, [2, BLOCK_M], SG)
        dsm.slice(0, 1, dim=0).store(
            gl.convert_layout(
                gl.where(mask_m, gl.exp(g_last - gc), 0.0), gl.SliceLayout(0, LV)
            )[None, :]
        )
        dsm.slice(1, 1, dim=0).store(
            gl.zeros([1, BLOCK_M], gl.float32, LV) + gl.exp(g_last)
        )
        dd = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=decay_ptr + (pid_m * H + pid_h).to(gl.int64) * 2 * BLOCK_M,
            shape=(2, BLOCK_M),
            strides=(BLOCK_M, 1),
            block_shape=(2, BLOCK_M),
            layout=SG,
        )
    gl.barrier()
    # Gate quarters straight from the gate tiles, in the layouts KKT and WY use.
    LR: gl.constexpr = gl.SliceLayout(1, L16)
    LC: gl.constexpr = gl.SliceLayout(0, L16)
    S1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    g_lds = gs.reinterpret(gl.float32, [BLOCK_M], S1)
    b_lds = bs.reinterpret(gl.float32, [BLOCK_M], S1)
    s_lds = ss.reinterpret(gl.float32, [BLOCK_M], S1)
    e_lds = es.reinterpret(gl.float32, [BLOCK_M], S1)
    g1 = g_lds.slice(0, 16).load(LR)
    g2 = g_lds.slice(16, 16).load(LR)
    g3 = g_lds.slice(32, 16).load(LR)
    g4 = g_lds.slice(48, 16).load(LR)
    r1 = b_lds.slice(0, 16).load(LR)
    r2 = b_lds.slice(16, 16).load(LR)
    r3 = b_lds.slice(32, 16).load(LR)
    r4 = b_lds.slice(48, 16).load(LR)

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
    if STORE_DECAY:
        gl.amd.gfx1250.tdm.async_store(dd, [0, 0], dsm)
    kk11 = _kk(k1, kt1, L16)
    kk21 = _kk(k2, kt1, L16)
    kk31 = _kk(k3, kt1, L16)
    kk41 = _kk(k4, kt1, L16)
    kk22 = _kk(k2, kt2, L16)
    kk32 = _kk(k3, kt2, L16)
    kk42 = _kk(k4, kt2, L16)
    kk33 = _kk(k3, kt3, L16)
    kk43 = _kk(k4, kt3, L16)
    kk44 = _kk(k4, kt4, L16)
    if fast:
        q1 = s_lds.slice(0, 16).load(LR)
        q2 = s_lds.slice(16, 16).load(LR)
        q3 = s_lds.slice(32, 16).load(LR)
        q4 = s_lds.slice(48, 16).load(LR)
        c = e_lds.slice(0, 16).load(LC)
        a11 = _gate_fast(kk11, q1, c, 0, 0, True, L16)
        a21 = _gate_fast(kk21, q2, c, 16, 0, False, L16)
        a31 = _gate_fast(kk31, q3, c, 32, 0, False, L16)
        a41 = _gate_fast(kk41, q4, c, 48, 0, False, L16)
        c = e_lds.slice(16, 16).load(LC)
        a22 = _gate_fast(kk22, q2, c, 16, 16, True, L16)
        a32 = _gate_fast(kk32, q3, c, 32, 16, False, L16)
        a42 = _gate_fast(kk42, q4, c, 48, 16, False, L16)
        c = e_lds.slice(32, 16).load(LC)
        a33 = _gate_fast(kk33, q3, c, 32, 32, True, L16)
        a43 = _gate_fast(kk43, q4, c, 48, 32, False, L16)
        c = e_lds.slice(48, 16).load(LC)
        a44 = _gate_fast(kk44, q4, c, 48, 48, True, L16)
    else:
        c = g_lds.slice(0, 16).load(LC)
        a11 = _gate(kk11, g1, c, r1, 0, 0, L16)
        a21 = _gate(kk21, g2, c, r2, 16, 0, L16)
        a31 = _gate(kk31, g3, c, r3, 32, 0, L16)
        a41 = _gate(kk41, g4, c, r4, 48, 0, L16)
        c = g_lds.slice(16, 16).load(LC)
        a22 = _gate(kk22, g2, c, r2, 16, 16, L16)
        a32 = _gate(kk32, g3, c, r3, 32, 16, L16)
        a42 = _gate(kk42, g4, c, r4, 48, 16, L16)
        c = g_lds.slice(32, 16).load(LC)
        a33 = _gate(kk33, g3, c, r3, 32, 32, L16)
        a43 = _gate(kk43, g4, c, r4, 48, 32, L16)
        c = g_lds.slice(48, 16).load(LC)
        a44 = _gate(kk44, g4, c, r4, 48, 48, L16)

    # V is read after W is stored; warm the L2 with it now. The prefetch clamps out-of-bounds rows to
    # base + shape[0] * stride[0] - 1, so its descriptor starts at head 0 of the chunk's first row.
    vp_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr + start_m * H * V,
        shape=(M, H * V),
        strides=(H * V, 1),
        block_shape=(BLOCK_M, V),
        layout=SMEM_V,
    )
    gl.amd.gfx1250.tdm.prefetch(vp_desc, [0, pid_h * V])
    offs_i = gl.arange(0, 16, layout=LR)
    offs_j = gl.arange(0, 16, layout=LC)
    eye = (offs_i[:, None] == offs_j[None, :]).to(gl.float32)
    z = gl.zeros([16, 16], gl.float32, layout=L16)
    zb = gl.zeros([16, 16], gl.bfloat16, layout=L16)
    # -A in bf16 high and residual parts: X0 uses the high parts, the Newton step both.
    h11, l11 = _split(-a11)
    h21, l21 = _split(-a21)
    h31, l31 = _split(-a31)
    h41, l41 = _split(-a41)
    h22, l22 = _split(-a22)
    h32, l32 = _split(-a32)
    h42, l42 = _split(-a42)
    h33, l33 = _split(-a33)
    h43, l43 = _split(-a43)
    h44, l44 = _split(-a44)
    gl.amd.hint.sched_barrier()
    # X0 ~ (I + A)^-1 from bf16 products; each block also transposed, the B operands of later products.
    TB: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[16, 8]], [16, 16], [1, 0]
    )
    tr0 = gl.allocate_shared_memory(gl.bfloat16, [16, 16], TB)
    tr1 = gl.allocate_shared_memory(gl.bfloat16, [16, 16], TB)
    tr2 = gl.allocate_shared_memory(gl.bfloat16, [16, 16], TB)
    tr3 = gl.allocate_shared_memory(gl.bfloat16, [16, 16], TB)
    e1 = _half(offs_i, offs_j, 1)
    e2 = _half(offs_i, offs_j, 2)
    e4 = _half(offs_i, offs_j, 4)
    e8 = _half(offs_i, offs_j, 8)
    m1t = _tr(h11, tr0, L16)
    m2t = _tr(h22, tr1, L16)
    m3t = _tr(h33, tr2, L16)
    m4t = _tr(h44, tr3, L16)
    gl.amd.hint.sched_barrier()
    x11, x11t = _diag(h11, m1t, eye, z, zb, e1, e2, e4, e8, OA16, OB16)
    x22, x22t = _diag(h22, m2t, eye, z, zb, e1, e2, e4, e8, OA16, OB16)
    gl.amd.hint.sched_barrier()
    x33, x33t = _diag(h33, m3t, eye, z, zb, e1, e2, e4, e8, OA16, OB16)
    x44, x44t = _diag(h44, m4t, eye, z, zb, e1, e2, e4, e8, OA16, OB16)
    gl.amd.hint.sched_barrier()
    # X_i,i-1 = X_ii Y with Y = (-A_i,i-1) X_i-1,i-1; yt = Y^T
    yt = _bf(_wm(x11t, zb, h21, zb, z, OA16, OB16))
    x21 = _bf(_wm(x22, zb, yt, zb, z, OA16, OB16))
    x21t = _bf(_wm(yt, zb, x22, zb, z, OA16, OB16))
    yt = _bf(_wm(x22t, zb, h32, zb, z, OA16, OB16))
    x32 = _bf(_wm(x33, zb, yt, zb, z, OA16, OB16))
    x32t = _bf(_wm(yt, zb, x33, zb, z, OA16, OB16))
    yt = _bf(_wm(x33t, zb, h43, zb, z, OA16, OB16))
    x43 = _bf(_wm(x44, zb, yt, zb, z, OA16, OB16))
    x43t = _bf(_wm(yt, zb, x44, zb, z, OA16, OB16))
    # X31 = X33 ((-A31) X11 + (-A32) X21), X42 likewise
    yt = _bf(_wm(x11t, x21t, h31, h32, z, OA16, OB16))
    x31 = _bf(_wm(x33, zb, yt, zb, z, OA16, OB16))
    x31t = _bf(_wm(yt, zb, x33, zb, z, OA16, OB16))
    yt = _bf(_wm(x22t, x32t, h42, h43, z, OA16, OB16))
    x42 = _bf(_wm(x44, zb, yt, zb, z, OA16, OB16))
    x42t = _bf(_wm(yt, zb, x44, zb, z, OA16, OB16))
    # X41 = X44 ((-A41) X11 + (-A42) X21 + (-A43) X31)
    yt = _bf(
        _wm(x11t, x21t, h41, h42, _wm(x31t, zb, h43, zb, z, OA16, OB16), OA16, OB16)
    )
    x41 = _bf(_wm(x44, zb, yt, zb, z, OA16, OB16))
    x41t = _bf(_wm(yt, zb, x44, zb, z, OA16, OB16))
    gl.amd.hint.sched_barrier()
    # Newton step: R = I - (I + A) X0, as R^T = I - X0^T + X0^T (-A)^T with both parts of -A.
    r11 = _bf(_wm(x11t, x11t, h11, l11, eye - _f(x11t), OA16, OB16))
    r22 = _bf(_wm(x22t, x22t, h22, l22, eye - _f(x22t), OA16, OB16))
    r33 = _bf(_wm(x33t, x33t, h33, l33, eye - _f(x33t), OA16, OB16))
    r44 = _bf(_wm(x44t, x44t, h44, l44, eye - _f(x44t), OA16, OB16))
    r21 = _wm(x11t, x21t, h21, h22, -_f(x21t), OA16, OB16)
    r21 = _bf(_wm(x11t, x21t, l21, l22, r21, OA16, OB16))
    r32 = _wm(x22t, x32t, h32, h33, -_f(x32t), OA16, OB16)
    r32 = _bf(_wm(x22t, x32t, l32, l33, r32, OA16, OB16))
    r43 = _wm(x33t, x43t, h43, h44, -_f(x43t), OA16, OB16)
    r43 = _bf(_wm(x33t, x43t, l43, l44, r43, OA16, OB16))
    r31 = _wm(x31t, x31t, h33, l33, -_f(x31t), OA16, OB16)
    r31 = _wm(x11t, x21t, h31, h32, r31, OA16, OB16)
    r31 = _bf(_wm(x11t, x21t, l31, l32, r31, OA16, OB16))
    r42 = _wm(x42t, x42t, h44, l44, -_f(x42t), OA16, OB16)
    r42 = _wm(x22t, x32t, h42, h43, r42, OA16, OB16)
    r42 = _bf(_wm(x22t, x32t, l42, l43, r42, OA16, OB16))
    r41 = _wm(x11t, x21t, h41, h42, -_f(x41t), OA16, OB16)
    r41 = _wm(x11t, x21t, l41, l42, r41, OA16, OB16)
    r41 = _wm(x31t, x41t, h43, h44, r41, OA16, OB16)
    r41 = _bf(_wm(x31t, x41t, l43, l44, r41, OA16, OB16))
    gl.amd.hint.sched_barrier()
    # Ai = X0 + X0 R (r_ij holds R_ij transposed)
    ai11 = _wm(x11, zb, r11, zb, _f(x11), OA16, OB16)
    ai21 = _wm(x21, x22, r11, r21, _f(x21), OA16, OB16)
    ai31 = _wm(
        x31, x32, r11, r21, _wm(x33, zb, r31, zb, _f(x31), OA16, OB16), OA16, OB16
    )
    ai41 = _wm(
        x41, x42, r11, r21, _wm(x43, x44, r31, r41, _f(x41), OA16, OB16), OA16, OB16
    )
    ai22 = _wm(x22, zb, r22, zb, _f(x22), OA16, OB16)
    ai32 = _wm(x32, x33, r22, r32, _f(x32), OA16, OB16)
    ai42 = _wm(
        x42, x43, r22, r32, _wm(x44, zb, r42, zb, _f(x42), OA16, OB16), OA16, OB16
    )
    ai33 = _wm(x33, zb, r33, zb, _f(x33), OA16, OB16)
    ai43 = _wm(x43, x44, r33, r43, _f(x43), OA16, OB16)
    ai44 = _wm(x44, zb, r44, zb, _f(x44), OA16, OB16)

    gl.amd.hint.sched_barrier()
    # w = (Ai diag(beta exp(g))) k
    s1 = s_lds.slice(0, 16).load(LC)
    s2 = s_lds.slice(16, 16).load(LC)
    s3 = s_lds.slice(32, 16).load(LC)
    s4 = s_lds.slice(48, 16).load(LC)
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
    es._keep_alive()
    if STORE_DECAY:
        dsm._keep_alive()
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
    b1 = b_lds.slice(0, 16).load(LC)
    b2 = b_lds.slice(16, 16).load(LC)
    b3 = b_lds.slice(32, 16).load(LC)
    b4 = b_lds.slice(48, 16).load(LC)
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


@gluon.constexpr_function
def _layout(name, K, V):
    """Layout `name` of the kernel: the bf16 WMMA layout, its operands, the K and V LDS tiles."""
    L16 = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[], instr_shape=[16, 16, 32]
    )
    return {
        "L16": L16,
        "OA16": gl.DotOperandLayout(operand_index=0, parent=L16, k_width=8),
        "OB16": gl.DotOperandLayout(operand_index=1, parent=L16, k_width=8),
        "SMEM_K": gl.PaddedSharedLayout.with_identity_for([[K, 16]], [64, K], [1, 0]),
        "SMEM_V": gl.PaddedSharedLayout.with_identity_for([[V, 16]], [64, V], [1, 0]),
    }[name]


def fused_kkt_inverse_gfx1250(k, v, cu_seqlens, plan, gating, with_decay):
    """Compute cumulative gates and WY factors for packed bf16 inputs.

    k/v are contiguous [1, T, 16/32, 128]; cu_seqlens marks sequence bounds.
    plan is the padded chunk plan or None; gating is (a, b, A_log, dt_bias),
    with bf16 a/b sharing a row stride and contiguous head columns.
    Return fp32 g [1, T, 32], bf16 w/u [1, T, 32, 128], and fp32
    beta/scale [1, 32, T]. with_decay: also return fp32 decay [NT, 32, 2, 64] per (chunk, head):
    row 0 exp(g_last - g_i) (0 past the sequence end), row 1 exp(g_last); else None.
    Requires gfx1250 and 64-token chunks.
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
    decay = (
        torch.empty(NT, H, 2, 64, device=k.device, dtype=torch.float32)
        if with_decay
        else None
    )
    # A chunk-grid multiple of 8 puts both heads of a K head on one XCD (round-robin over 8).
    _fused_kkt_inverse_gfx1250_kernel[(-(-NT // 8) * 8, H)](
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
        decay if with_decay else w,  # unused pointer without decay
        T,
        NT,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BLOCK_M=64,
        BLOCK_K=32,
        BLOCK_V=32,
        HAS_NT=nt_ptr is not None,
        STORE_DECAY=with_decay,
        num_warps=1,
    )
    return g_cumsum, decay, w, u, beta_hm, scale_hm
