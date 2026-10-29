# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The chunked Gated DeltaNet output (`chunk_fwd_o`) as a Gluon kernel for gfx1250,
optionally with the layer's gated RMSNorm in its epilogue.

One program computes one chunk of one value head, all V columns:

    o = scale * (exp(g) q @ h  +  (exp(g_i - g_j) q k^T, causal) @ v_new)

and, with a gate, ``norm(o) * weight * silu(z)`` per row (the RMSNorm group is
the head), so the un-normalized output never goes to memory. The tensor data
mover brings every operand tile into LDS at once -- the chunk is bounded at its
sequence's end, so a partial last chunk reads zeros and stores only its rows --
and stores the result from LDS. The launch configuration is fixed.
"""

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_TDM_MAX_PAD_INTERVAL_DWORDS = 256


def _pad_interval(extent, elem_bits=16):
    return min(extent, _TDM_MAX_PAD_INTERVAL_DWORDS * 32 // elem_bits)


@gluon.jit
def _chunk_o_gfx1250_kernel(
    q,
    k,
    v,
    h,
    g,
    o,
    z,
    norm_weight,
    cu_seqlens,
    chunk_indices,
    nt_ptr,
    scale,
    eps,
    T,
    stride_z_t,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    USE_G: gl.constexpr,
    GATED_NORM: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    HAS_NT: gl.constexpr,
    WMMA: gl.constexpr,
    OP_A: gl.constexpr,
    OP_B: gl.constexpr,
    SMEM_ROW_K: gl.constexpr,
    SMEM_ROW_V: gl.constexpr,
    SMEM_H: gl.constexpr,
    SMEM_A: gl.constexpr,
    SMEM_G: gl.constexpr,
    NORM_LAYOUT: gl.constexpr,
):
    # heads vary fastest: the value heads of a key head read the same q and k
    # tiles back to back, from L2
    i_bh = gl.program_id(0)
    i_t = gl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    if IS_VARLEN and HAS_NT:  # noqa: SIM102 (a constexpr, then a device load)
        if i_t >= gl.load(nt_ptr):
            return
    if IS_VARLEN:
        i_tg = i_t
        i_n = gl.load(chunk_indices + i_t * 2).to(gl.int32)
        i_t = gl.load(chunk_indices + i_t * 2 + 1).to(gl.int32)
        bos = gl.load(cu_seqlens + i_n).to(gl.int32)
        eos = gl.load(cu_seqlens + i_n + 1).to(gl.int32)
        T = eos - bos
    else:
        i_tg = i_b * gl.cdiv(T, BT) + i_t
        bos = i_b * T
    t0 = i_t * BT
    bos64 = bos.to(gl.int64)

    q_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=q + (bos64 * Hg + i_h // (H // Hg)) * K,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BT, K),
        layout=SMEM_ROW_K,
    )
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k + (bos64 * Hg + i_h // (H // Hg)) * K,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BT, K),
        layout=SMEM_ROW_K,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v + (bos64 * H + i_h) * V,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BT, V),
        layout=SMEM_ROW_V,
    )
    h_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=h + (i_tg.to(gl.int64) * H + i_h) * K * V,
        shape=(K, V),
        strides=(V, 1),
        block_shape=(K, V),
        layout=SMEM_H,
    )
    o_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=o + (bos64 * H + i_h) * V,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BT, V),
        layout=SMEM_ROW_K,
    )
    q_smem = gl.allocate_shared_memory(q.type.element_ty, [BT, K], SMEM_ROW_K)
    k_smem = gl.allocate_shared_memory(k.type.element_ty, [BT, K], SMEM_ROW_K)
    v_smem = gl.allocate_shared_memory(v.type.element_ty, [BT, V], SMEM_ROW_V)
    h_smem = gl.allocate_shared_memory(h.type.element_ty, [K, V], SMEM_H)
    a_smem = gl.allocate_shared_memory(v.type.element_ty, [BT, BT], SMEM_A)
    gl.amd.gfx1250.tdm.async_load(q_desc, [t0, 0], q_smem)
    gl.amd.gfx1250.tdm.async_load(h_desc, [0, 0], h_smem)
    gl.amd.gfx1250.tdm.async_load(k_desc, [t0, 0], k_smem)
    gl.amd.gfx1250.tdm.async_load(v_desc, [t0, 0], v_smem)
    if USE_G:
        g_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=g + bos64 * H + i_h,
            shape=(T, 1),
            strides=(H, 1),
            block_shape=(BT, 1),
            layout=SMEM_G,
        )
        g_smem = gl.allocate_shared_memory(gl.float32, [BT, 1], SMEM_G)
        gl.amd.gfx1250.tdm.async_load(g_desc, [t0, 0], g_smem)
    if GATED_NORM:
        z_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=z + bos64 * stride_z_t + i_h * V,
            shape=(T, V),
            strides=(stride_z_t, 1),
            block_shape=(BT, V),
            layout=SMEM_ROW_V,
        )
        z_smem = gl.allocate_shared_memory(z.type.element_ty, [BT, V], SMEM_ROW_V)
        gl.amd.gfx1250.tdm.async_load(z_desc, [t0, 0], z_smem)

    rows = gl.arange(0, BT, layout=gl.SliceLayout(1, WMMA))
    cols_t = gl.arange(0, BT, layout=gl.SliceLayout(0, WMMA))
    cols_v = gl.arange(0, V, layout=gl.SliceLayout(0, WMMA))
    if GATED_NORM:
        weight = gl.load(norm_weight + cols_v).to(gl.float32)

    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    b_q = q_smem.load(OP_A)
    b_o = gl.amd.gfx1250.wmma(
        b_q, h_smem.load(OP_B), gl.zeros([BT, V], gl.float32, layout=WMMA)
    )
    b_A = gl.amd.gfx1250.wmma(
        b_q,
        k_smem.permute([1, 0]).load(OP_B),
        gl.zeros([BT, BT], gl.float32, layout=WMMA),
    )
    if USE_G:
        b_g = g_smem.reshape([BT]).load(gl.SliceLayout(1, WMMA))
        b_gc = g_smem.reshape([BT]).load(gl.SliceLayout(0, WMMA))
        b_o = b_o * gl.exp(b_g)[:, None]
        b_A = b_A * gl.exp(b_g[:, None] - b_gc[None, :])
    live = T - t0
    m_A = (
        (rows[:, None] >= cols_t[None, :])
        & (rows[:, None] < live)
        & (cols_t[None, :] < live)
    )
    b_A = gl.where(m_A, b_A, 0.0)
    a_smem.store(b_A.to(a_smem.dtype))
    gl.barrier()
    b_pv = gl.amd.gfx1250.wmma(
        a_smem.load(OP_A), v_smem.load(OP_B), gl.zeros([BT, V], gl.float32, layout=WMMA)
    )
    b_o = b_o * scale + b_pv * scale
    if GATED_NORM:
        # the unfused chain stores o in bf16 and normalizes what it reads back
        x = b_o.to(o.dtype.element_ty).to(gl.float32)
        # the sum of squares in layer_norm_fwd's register layout, so its order of
        # additions -- and the result, bit for bit -- is the unfused kernel's
        x_rows = gl.convert_layout(x, NORM_LAYOUT)
        rstd = gl.rsqrt(gl.sum(x_rows * x_rows, 1) / V + eps)
        rstd = gl.convert_layout(rstd, gl.SliceLayout(1, WMMA))
        b_z = z_smem.load(WMMA).to(gl.float32)
        # silu(z) = z * sigmoid(z), sigmoid written as tl.sigmoid computes it
        b_o = x * rstd[:, None] * weight[None, :] * (b_z * (1.0 / (1.0 + gl.exp(-b_z))))
    # the output tile reuses q's buffer: every wave read q before the barrier
    # that published the A tile
    o_smem = q_smem
    o_smem.store(b_o.to(o_smem.dtype))
    gl.barrier()
    gl.amd.gfx1250.tdm.async_store(o_desc, [t0, 0], o_smem)
    gl.amd.gfx1250.tdm.async_wait(0)


@functools.cache
def _layouts(K, V, BT, num_warps):
    warp_bases = [[1, 0], [2, 0], [4, 0]][: num_warps.bit_length() - 1]
    wmma = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]
    )
    op_a = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    op_b = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    smem_row_k = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(K), 8]], [BT, K], [1, 0]
    )
    smem_row_v = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(V), 16]], [BT, V], [1, 0]
    )
    smem_h = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(V), 16]], [K, V], [1, 0]
    )
    smem_a = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(BT), 8]], [BT, BT], [1, 0]
    )
    smem_g = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    # layer_norm_fwd's layout for a [rows, 128] bf16 tile (one warp, 8-wide loads)
    norm_layout = gl.BlockedLayout([1, 8], [2, 16], [num_warps, 1], [1, 0])
    return wmma, op_a, op_b, smem_row_k, smem_row_v, smem_h, smem_a, smem_g, norm_layout


NUM_WARPS = 4


def chunk_o_gfx1250(
    q,
    k,
    v,
    h,
    g,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    nt_ptr,
    NT,
    T,
    B,
    H,
    Hg,
    K,
    V,
    BT,
    gate=None,
):
    """``gate`` = (z [.., T, H, V] with unit V stride and H stride V, norm weight
    [V], eps): write ``norm(o) * weight * silu(z)`` instead of ``o``."""
    layouts = _layouts(K, V, BT, NUM_WARPS)
    if gate is not None:
        z, weight, eps = gate
        z2 = z.reshape(-1, *z.shape[-2:]) if z.dim() > 3 else z
        assert (
            z2.stride(-1) == 1 and z2.stride(-2) == V
        ), "z rows must be [H, V] contiguous"
        stride_z_t = z2.stride(0)
    else:
        z = weight = None
        eps = 0.0
        stride_z_t = 0
    _chunk_o_gfx1250_kernel[(B * H, NT)](
        q,
        k,
        v,
        h,
        g,
        o,
        z,
        weight,
        cu_seqlens,
        chunk_indices,
        nt_ptr,
        scale,
        eps,
        T,
        stride_z_t,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        GATED_NORM=gate is not None,
        IS_VARLEN=cu_seqlens is not None,
        HAS_NT=nt_ptr is not None,
        WMMA=layouts[0],
        OP_A=layouts[1],
        OP_B=layouts[2],
        SMEM_ROW_K=layouts[3],
        SMEM_ROW_V=layouts[4],
        SMEM_H=layouts[5],
        SMEM_A=layouts[6],
        SMEM_G=layouts[7],
        NORM_LAYOUT=layouts[8],
        num_warps=NUM_WARPS,
    )
    return o
