# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The chunked Gated DeltaNet state recurrence (`chunk_delta_h`) as a Gluon kernel
for gfx1250.

One program walks one (sequence, value head, V block) over its chunks; the
recurrence is serial in the chunks, so the program is a chain of two dependent
GEMMs a chunk:

    v_new = u - w @ h                    [BT, BV]  (stored, before the gate)
    h     = exp(g_last) h + k^T @ (exp(g_last - g) v_new)

The state stays fp32 in WMMA accumulator registers for the whole walk. The
chunk's operands do not depend on the state, so the tensor data mover loads the
w, k and u tiles of the next chunks into LDS while the current chunk computes,
and stores the per-chunk state and v_new tiles from LDS without the waves
waiting on them. Out-of-sequence rows of a partial last chunk read as zero (the
descriptors are bounded at the sequence end) and are not stored.

The launch configuration is a function of the shapes alone; nothing is tuned at
run time.
"""

import functools

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_BT = 64
_TDM_MAX_PAD_INTERVAL_DWORDS = 256


def _pad_interval(extent, elem_bits=16):
    return min(extent, _TDM_MAX_PAD_INTERVAL_DWORDS * 32 // elem_bits)


@gluon.jit
def _chunk_h_gfx1250_kernel(
    k,
    u,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    n_ptr,
    h0_idx,
    ht_idx,
    h0_flag,
    T,
    NT_ALL,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    USE_G: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    STORE_FINAL_STATE: gl.constexpr,
    SAVE_NEW_VALUE: gl.constexpr,
    STORE_H: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    HAS_N: gl.constexpr,
    USE_STATE_IDX: gl.constexpr,
    WMMA: gl.constexpr,
    OP_A: gl.constexpr,
    OP_B: gl.constexpr,
    SMEM_A_ROW: gl.constexpr,
    SMEM_A_COL: gl.constexpr,
    SMEM_TILE: gl.constexpr,
    SMEM_STATE: gl.constexpr,
    SMEM_G: gl.constexpr,
):
    i_v = gl.program_id(0)
    i_nh = gl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H
    if IS_VARLEN and HAS_N:
        # a bucket-shaped grid: programs past the step's sequence count have no work
        if i_n >= gl.load(n_ptr):
            return
    if IS_VARLEN:
        bos = gl.load(cu_seqlens + i_n).to(gl.int32)
        eos = gl.load(cu_seqlens + i_n + 1).to(gl.int32)
        T = eos - bos
        boh = gl.load(chunk_offsets + i_n).to(gl.int32)
    else:
        bos = i_n * T
        boh = i_n * gl.cdiv(T, BT)
    NT = gl.cdiv(T, BT)
    v0 = i_v * BV

    # ---- descriptors, bounded at the sequence end ----
    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=w + (bos.to(gl.int64) * H + i_h) * K,
        shape=(T, K),
        strides=(H * K, 1),
        block_shape=(BT, K),
        layout=SMEM_A_ROW,
    )
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k + (bos.to(gl.int64) * Hg + i_h // (H // Hg)) * K,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BT, K),
        layout=SMEM_A_COL,
    )
    u_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=u + (bos.to(gl.int64) * H + i_h) * V,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BT, BV),
        layout=SMEM_TILE,
    )
    if USE_G:
        # the gates of the chunk ride with its operands: one fp32 column of [T, H]
        g_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=g + bos.to(gl.int64) * H + i_h,
            shape=(T, 1),
            strides=(H, 1),
            block_shape=(BT, 1),
            layout=SMEM_G,
        )
    NB: gl.constexpr = NUM_BUFFERS
    w_smem = gl.allocate_shared_memory(w.type.element_ty, [NB, BT, K], SMEM_A_ROW)
    k_smem = gl.allocate_shared_memory(k.type.element_ty, [NB, BT, K], SMEM_A_COL)
    u_smem = gl.allocate_shared_memory(u.type.element_ty, [NB, BT, BV], SMEM_TILE)
    if USE_G:
        g_smem = gl.allocate_shared_memory(gl.float32, [NB, BT, 1], SMEM_G)
    # the state entering a chunk, bf16: the B operand of w @ h and the h[i_t]
    # store; v_new, bf16, for its store. NB of each, so the store of a chunk can
    # drain while the next chunks write the others: a buffer is written again only
    # after the wait that covers its store.
    h_smem = gl.allocate_shared_memory(k.type.element_ty, [NB, K, BV], SMEM_STATE)
    if SAVE_NEW_VALUE:
        vn_smem = gl.allocate_shared_memory(k.type.element_ty, [NB, BT, BV], SMEM_TILE)
    # the gated v_new, bf16: the B operand of k^T @ v
    vg_smem = gl.allocate_shared_memory(k.type.element_ty, [BT, BV], SMEM_TILE)
    if STORE_H:
        h_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=h,
            shape=(NT_ALL * H * K, V),
            strides=(V, 1),
            block_shape=(K, BV),
            layout=SMEM_STATE,
        )
    if SAVE_NEW_VALUE:
        vn_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=v_new + (bos.to(gl.int64) * H + i_h) * V,
            shape=(T, V),
            strides=(H * V, 1),
            block_shape=(BT, BV),
            layout=SMEM_TILE,
        )
    # Tensor-data-mover operations of a chunk, in issue order: the h store, the
    # loads of a later chunk's operands, the v_new store. Every chunk issues the
    # same count (loads past the last chunk re-read the last chunk), so each wait
    # below is a fixed count of the operations issued after the chunk's loads.
    LOADS: gl.constexpr = 3 + USE_G
    PER_CHUNK: gl.constexpr = STORE_H + LOADS + SAVE_NEW_VALUE

    # ---- the initial state ----
    rows_k = gl.arange(0, K, layout=gl.SliceLayout(1, WMMA))
    cols_v = gl.arange(0, BV, layout=gl.SliceLayout(0, WMMA))
    state_off = rows_k[:, None] * V + (v0 + cols_v)[None, :]
    if USE_STATE_IDX:
        if USE_INITIAL_STATE:
            h0 = h0 + (gl.load(h0_idx + i_n).to(gl.int64) * H + i_h) * K * V
        if STORE_FINAL_STATE:
            ht = ht + (gl.load(ht_idx + i_n).to(gl.int64) * H + i_h) * K * V
        use_h0 = gl.load(h0_flag + i_n) != 0
    else:
        if USE_INITIAL_STATE:
            h0 = h0 + i_nh.to(gl.int64) * K * V
        if STORE_FINAL_STATE:
            ht = ht + i_nh.to(gl.int64) * K * V
        use_h0 = True
    # the first chunks' operands
    for i in gl.static_range(NB - 1):
        c = gl.minimum(i, NT - 1) * BT
        gl.amd.gfx1250.tdm.async_load(w_desc, [c, 0], w_smem.index(i))
        gl.amd.gfx1250.tdm.async_load(k_desc, [c, 0], k_smem.index(i))
        gl.amd.gfx1250.tdm.async_load(u_desc, [c, v0], u_smem.index(i))
        if USE_G:
            gl.amd.gfx1250.tdm.async_load(g_desc, [c, 0], g_smem.index(i))
    if USE_INITIAL_STATE:
        state = gl.load(h0 + state_off, mask=(state_off >= 0) & use_h0, other=0.0).to(gl.float32)
    else:
        state = gl.zeros([K, BV], gl.float32, layout=WMMA)
    # the state entering chunk 0, bf16; each chunk writes the next one's
    h_smem.index(0).store(state.to(h_smem.dtype))

    rows_t = gl.arange(0, BT, layout=gl.SliceLayout(1, WMMA))
    zero_v = gl.zeros([BT, BV], gl.float32, layout=WMMA)
    zero_h = gl.zeros([K, BV], gl.float32, layout=WMMA)

    for i_t in range(NT):
        buf = i_t % NB
        if i_t == 0:
            gl.amd.gfx1250.tdm.async_wait(LOADS * (NB - 2))
        elif NB == 3 and i_t == 1:
            gl.amd.gfx1250.tdm.async_wait(PER_CHUNK)
        else:
            gl.amd.gfx1250.tdm.async_wait(SAVE_NEW_VALUE + PER_CHUNK * (NB - 2))
        # every wave's share of the copies has landed and its state rows are in
        # h_smem, and every wave is done with the buffers the next copies overwrite
        gl.barrier()
        h_cur = h_smem.index(buf)
        if STORE_H:
            gl.amd.gfx1250.tdm.async_store(h_desc, [((boh + i_t) * H + i_h) * K, v0], h_cur)
        nxt = gl.minimum(i_t + NB - 1, NT - 1) * BT
        nbuf = (i_t + NB - 1) % NB
        gl.amd.gfx1250.tdm.async_load(w_desc, [nxt, 0], w_smem.index(nbuf))
        gl.amd.gfx1250.tdm.async_load(k_desc, [nxt, 0], k_smem.index(nbuf))
        gl.amd.gfx1250.tdm.async_load(u_desc, [nxt, v0], u_smem.index(nbuf))
        if USE_G:
            gl.amd.gfx1250.tdm.async_load(g_desc, [nxt, 0], g_smem.index(nbuf))
        t0 = i_t * BT
        m_t = (t0 + rows_t) < T

        b_h = h_cur.load(OP_B)
        b_w = w_smem.index(buf).load(OP_A)
        b_v = gl.amd.gfx1250.wmma(b_w, b_h, zero_v)
        b_u = u_smem.index(buf).load(WMMA).to(gl.float32)
        b_v = b_u - b_v
        if USE_G:
            b_g = g_smem.index(buf).reshape([BT]).load(gl.SliceLayout(1, WMMA))
            b_g_last = gl.sum(gl.where(rows_t == gl.minimum(BT, T - t0) - 1, b_g, 0.0), 0)
            b_vg = b_v * gl.where(m_t, gl.exp(b_g_last - b_g), 0.0)[:, None]
            state = state * gl.exp(b_g_last)
        else:
            b_vg = b_v
        if SAVE_NEW_VALUE:
            vn_cur = vn_smem.index(buf)
            vn_cur.store(b_v.to(vn_cur.dtype))
        vg_smem.store(b_vg.to(vg_smem.dtype))
        gl.barrier()
        if SAVE_NEW_VALUE:
            gl.amd.gfx1250.tdm.async_store(vn_desc, [t0, v0], vn_cur)
        b_k = k_smem.index(buf).permute([1, 0]).load(OP_A)
        b_vgb = vg_smem.load(OP_B)
        state = state + gl.amd.gfx1250.wmma(b_k, b_vgb, zero_h)
        # the state entering the next chunk
        h_smem.index((i_t + 1) % NB).store(state.to(h_smem.dtype))

    gl.amd.gfx1250.tdm.async_wait(0)
    if STORE_FINAL_STATE:
        gl.store(ht + state_off, state.to(ht.dtype.element_ty))


@functools.lru_cache(maxsize=None)
def _num_cus(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=None)
def _layouts(K, BT, BV, num_warps, split="M"):
    if split == "M":
        warp_bases = [[1, 0], [2, 0], [4, 0]][: num_warps.bit_length() - 1]
    else:
        # each wave owns BV / num_warps columns of the state and of v_new, so both
        # GEMMs of a chunk are wave-local; every wave reads all of w and k
        warp_bases = [[0, 1], [0, 2], [0, 4]][: num_warps.bit_length() - 1]
    wmma = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]
    )
    op_a = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    op_b = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    # [BT, K] row-major tiles: w is read as the A operand ([M=BT, K]); k is read
    # transposed as the A operand of k^T @ v ([M=K, BT]).
    smem_a_row = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(K), 8]], [BT, K], [1, 0]
    )
    smem_a_col = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(K), 8]], [BT, K], [1, 0]
    )
    smem_tile = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(BV), 16]], [BT, BV], [1, 0]
    )
    smem_state = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(BV), 16]], [K, BV], [1, 0]
    )
    smem_g = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    return wmma, op_a, op_b, smem_a_row, smem_a_col, smem_tile, smem_state, smem_g


def launch_config(N, H, num_cus):
    """(BV, operand buffers, warps) from host-known shapes alone.

    A (sequence, head) is one serial walk over its chunks; its V blocks are
    independent programs that re-read the same w and k tiles. With enough walks
    to give every CU two programs, the widest block wins (no re-reads, the GPU
    is throughput-bound). With fewer, the step is as long as its longest walk:
    narrower blocks shorten the per-chunk chain, and a third operand buffer keeps
    the copies further ahead of it. Measured on gfx1250 (256 CUs) at T = 16384,
    K = V = 128, H = 32 / 8, 1 to 52 sequences.
    """
    walks = N * H
    if walks >= 2 * num_cus:
        return 128, 2, 4
    if walks >= num_cus // 2:
        return 64, 3, 4
    return 32, 3, 4


def chunk_h_gfx1250(
    k, w, u, g, h, v_new, h0, ht, cu_seqlens, chunk_offsets, n_ptr, h0_idx, ht_idx,
    h0_flag, T, N, NT_ALL, H, Hg, K, V, BT, store_h=True, BV=None, num_warps=None,
    num_buffers=None, split="M",
):
    bv, nb, nw = launch_config(N, H, _num_cus(k.device.index))
    BV = BV or bv
    num_buffers = num_buffers or nb
    num_warps = num_warps or nw
    layouts = _layouts(K, BT, BV, num_warps, split)
    _chunk_h_gfx1250_kernel[(triton.cdiv(V, BV), N * H)](
        k, u, w, v_new, g, h, h0, ht, cu_seqlens, chunk_offsets, n_ptr, h0_idx,
        ht_idx, h0_flag, T, NT_ALL,
        H=H, Hg=Hg, K=K, V=V, BT=BT, BV=BV, NUM_BUFFERS=num_buffers,
        USE_G=g is not None,
        USE_INITIAL_STATE=h0 is not None,
        STORE_FINAL_STATE=ht is not None,
        SAVE_NEW_VALUE=v_new is not None,
        STORE_H=store_h,
        IS_VARLEN=cu_seqlens is not None,
        HAS_N=n_ptr is not None,
        USE_STATE_IDX=h0_idx is not None,
        WMMA=layouts[0], OP_A=layouts[1], OP_B=layouts[2], SMEM_A_ROW=layouts[3],
        SMEM_A_COL=layouts[4], SMEM_TILE=layouts[5], SMEM_STATE=layouts[6],
        SMEM_G=layouts[7],
        num_warps=num_warps,
    )
