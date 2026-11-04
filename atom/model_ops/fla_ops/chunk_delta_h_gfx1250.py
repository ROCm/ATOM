# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Serial chunked gated delta-rule state update for gfx1250.

Each chunk writes its incoming bf16 state and v_new = u - w @ bf16(state),
then updates the fp32 state with exp(g_last) * state +
k.T @ bf16(exp(g_last - g) * v_new). TDM buffers overlap operand loads and stores.
"""

import functools

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from atom.utils import envs

_NUM_WARPS = 4
_CONFIGS = {
    128: {
        "buffers": 2,
        "warp_bases": [[0, 1], [1, 0]],
        "k_padding": 8,
        "latency": 1.13,
        "cost": 2.1,
    },
    64: {
        "buffers": 3,
        "warp_bases": [[1, 0], [2, 0]],
        "k_padding": 8,
        "latency": 0.83,
        "cost": 1.82,
    },
    32: {
        "buffers": 4,
        "warp_bases": [[1, 0], [2, 0]],
        "k_padding": 16,
        "latency": 0.71,
        "cost": 1.39,
    },
}


@gluon.jit
def _tile_descriptor(desc, row, col):
    return gl.amd.gfx1250.tdm.update_tensor_descriptor(
        desc, add_offsets=[row, col], clamp_bounds=True
    )


@gluon.jit
def _load_pairs(
    w_desc, k_desc, u_desc, g_desc, w_dst, k_dst, u_dst, g_dst, start_m, start_n
):
    # Waves 0-1 load w/u; waves 2-3 load k/g.
    gl.amd.gfx1250.tdm.async_load_fused(
        [
            (_tile_descriptor(w_desc, start_m, 0), w_dst, 3),
            (_tile_descriptor(k_desc, start_m, 0), k_dst, 12),
        ]
    )
    gl.amd.gfx1250.tdm.async_load_fused(
        [
            (_tile_descriptor(u_desc, start_m, start_n), u_dst, 3),
            (_tile_descriptor(g_desc, start_m, 0), g_dst, 12),
        ]
    )


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
    ckpt_pool,
    ckpt_slot,
    T,
    NT_ALL,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    USE_G: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    STORE_FINAL_STATE: gl.constexpr,
    SAVE_NEW_VALUE: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    HAS_N: gl.constexpr,
    USE_STATE_IDX: gl.constexpr,
    STORE_CKPT: gl.constexpr,
    PAIRED: gl.constexpr,
    WMMA: gl.constexpr,
    OP_A: gl.constexpr,
    OP_B: gl.constexpr,
    SMEM_A_ROW: gl.constexpr,
    SMEM_A_COL: gl.constexpr,
    SMEM_TILE: gl.constexpr,
    SMEM_STATE: gl.constexpr,
    SMEM_G: gl.constexpr,
):
    pid_n = gl.program_id(0)
    pid_m = gl.program_id(1)
    sequence = pid_m // H
    head = pid_m % H
    if IS_VARLEN and HAS_N:  # noqa: SIM102 (a constexpr, then a device load)
        if sequence >= gl.load(n_ptr):
            return
    if IS_VARLEN:
        start_token = gl.load(cu_seqlens + sequence).to(gl.int32)
        end_token = gl.load(cu_seqlens + sequence + 1).to(gl.int32)
        T = end_token - start_token
        first_chunk = gl.load(chunk_offsets + sequence).to(gl.int32)
    else:
        start_token = sequence * T
        first_chunk = sequence * gl.cdiv(T, BLOCK_M)
    num_chunks = gl.cdiv(T, BLOCK_M)
    start_n = pid_n * BLOCK_N

    # Bound operand descriptors at the sequence end.
    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=w + (start_token.to(gl.int64) * H + head) * K,
        shape=(T, K),
        strides=(H * K, 1),
        block_shape=(BLOCK_M, K),
        layout=SMEM_A_ROW,
    )
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k + (start_token.to(gl.int64) * Hg + head // (H // Hg)) * K,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BLOCK_M, K),
        layout=SMEM_A_COL,
    )
    u_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=u + (start_token.to(gl.int64) * H + head) * V,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=SMEM_TILE,
    )
    if USE_G:
        g_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=g + start_token.to(gl.int64) * H + head,
            shape=(T, 1),
            strides=(H, 1),
            block_shape=(BLOCK_M, 1),
            layout=SMEM_G,
        )
    NB: gl.constexpr = NUM_BUFFERS
    w_smem = gl.allocate_shared_memory(w.type.element_ty, [NB, BLOCK_M, K], SMEM_A_ROW)
    k_smem = gl.allocate_shared_memory(k.type.element_ty, [NB, BLOCK_M, K], SMEM_A_COL)
    u_smem = gl.allocate_shared_memory(
        u.type.element_ty, [NB, BLOCK_M, BLOCK_N], SMEM_TILE
    )
    if USE_G:
        g_smem = gl.allocate_shared_memory(gl.float32, [NB, BLOCK_M, 1], SMEM_G)
    # Ring buffers keep bf16 states and residuals live until their stores drain.
    h_smem = gl.allocate_shared_memory(k.type.element_ty, [NB, K, BLOCK_N], SMEM_STATE)
    if SAVE_NEW_VALUE:
        vn_smem = gl.allocate_shared_memory(
            k.type.element_ty, [NB, BLOCK_M, BLOCK_N], SMEM_TILE
        )
    vg_smem = gl.allocate_shared_memory(
        k.type.element_ty, [BLOCK_M, BLOCK_N], SMEM_TILE
    )
    h_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=h,
        shape=(NT_ALL * H * K, V),
        strides=(V, 1),
        block_shape=(K, BLOCK_N),
        layout=SMEM_STATE,
    )
    if SAVE_NEW_VALUE:
        vn_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=v_new + (start_token.to(gl.int64) * H + head) * V,
            shape=(T, V),
            strides=(H * V, 1),
            block_shape=(BLOCK_M, BLOCK_N),
            layout=SMEM_TILE,
        )
    # Every chunk issues the same TDM count, including prefetches past the end.
    LOADS: gl.constexpr = 2 if PAIRED else 3 + USE_G
    PER_CHUNK: gl.constexpr = 1 + LOADS + SAVE_NEW_VALUE

    offs_k = gl.arange(0, K, layout=gl.SliceLayout(1, WMMA))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA))
    # The nonnegative-offset masks preserve pointer-width inference.
    offs_state = offs_k[:, None] * V + (start_n + offs_n)[None, :]
    if USE_STATE_IDX:
        if USE_INITIAL_STATE:
            h0 = h0 + (gl.load(h0_idx + sequence).to(gl.int64) * H + head) * K * V
        if STORE_FINAL_STATE:
            ht = ht + (gl.load(ht_idx + sequence).to(gl.int64) * H + head) * K * V
        use_h0 = gl.load(h0_flag + sequence) != 0
    else:
        if USE_INITIAL_STATE:
            h0 = h0 + pid_m.to(gl.int64) * K * V
        if STORE_FINAL_STATE:
            ht = ht + pid_m.to(gl.int64) * K * V
        use_h0 = True
    # Prefetch operands before loading the initial state.
    for i in gl.static_range(NB - 1):
        start_m = gl.minimum(i, num_chunks - 1) * BLOCK_M
        if PAIRED:
            _load_pairs(
                w_desc,
                k_desc,
                u_desc,
                g_desc,
                w_smem.index(i),
                k_smem.index(i),
                u_smem.index(i),
                g_smem.index(i),
                start_m,
                start_n,
            )
        else:
            gl.amd.gfx1250.tdm.async_load(w_desc, [start_m, 0], w_smem.index(i))
            gl.amd.gfx1250.tdm.async_load(k_desc, [start_m, 0], k_smem.index(i))
            gl.amd.gfx1250.tdm.async_load(u_desc, [start_m, start_n], u_smem.index(i))
            if USE_G:
                gl.amd.gfx1250.tdm.async_load(g_desc, [start_m, 0], g_smem.index(i))
    if USE_INITIAL_STATE:
        state = gl.load(h0 + offs_state, mask=(offs_state >= 0) & use_h0, other=0.0).to(
            gl.float32
        )
    else:
        state = gl.zeros([K, BLOCK_N], gl.float32, layout=WMMA)
    h_smem.index(0).store(state.to(h_smem.dtype))

    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA))
    zero_v = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=WMMA)

    for chunk in range(num_chunks):
        buf = chunk % NB
        if chunk == 0:
            gl.amd.gfx1250.tdm.async_wait(LOADS * (NB - 2))
        elif NB >= 3 and chunk == 1:
            gl.amd.gfx1250.tdm.async_wait(PER_CHUNK + LOADS * (NB - 3))
        elif NB == 4 and chunk == 2:
            gl.amd.gfx1250.tdm.async_wait(2 * PER_CHUNK)
        else:
            gl.amd.gfx1250.tdm.async_wait(SAVE_NEW_VALUE + PER_CHUNK * (NB - 2))
        # Wait for all waves before reading or reusing shared buffers.
        gl.barrier()
        if STORE_CKPT:
            # Checkpoints preserve the incoming fp32 state before bf16 rounding.
            cslot = gl.load(ckpt_slot + first_chunk + chunk)
            gl.store(
                ckpt_pool
                + (gl.maximum(cslot, 0).to(gl.int64) * H + head) * K * V
                + offs_state,
                state.to(ckpt_pool.dtype.element_ty),
                mask=(offs_state >= 0) & (cslot >= 0),
            )
        h_cur = h_smem.index(buf)
        gl.amd.gfx1250.tdm.async_store(
            h_desc, [((first_chunk + chunk) * H + head) * K, start_n], h_cur
        )
        next_m = gl.minimum(chunk + NB - 1, num_chunks - 1) * BLOCK_M
        next_buf = (chunk + NB - 1) % NB
        if PAIRED:
            _load_pairs(
                w_desc,
                k_desc,
                u_desc,
                g_desc,
                w_smem.index(next_buf),
                k_smem.index(next_buf),
                u_smem.index(next_buf),
                g_smem.index(next_buf),
                next_m,
                start_n,
            )
        else:
            gl.amd.gfx1250.tdm.async_load(w_desc, [next_m, 0], w_smem.index(next_buf))
            gl.amd.gfx1250.tdm.async_load(k_desc, [next_m, 0], k_smem.index(next_buf))
            gl.amd.gfx1250.tdm.async_load(
                u_desc, [next_m, start_n], u_smem.index(next_buf)
            )
            if USE_G:
                gl.amd.gfx1250.tdm.async_load(
                    g_desc, [next_m, 0], g_smem.index(next_buf)
                )
        start_m = chunk * BLOCK_M
        mask_m = (start_m + offs_m) < T

        b_h = h_cur.load(OP_B)
        b_w = w_smem.index(buf).load(OP_A)
        if PAIRED:
            # Read k while the first dependent product runs.
            b_k = k_smem.index(buf).permute([1, 0]).load(OP_A)
        b_v = gl.amd.gfx1250.wmma(b_w, b_h, zero_v)
        b_u = u_smem.index(buf).load(WMMA).to(gl.float32)
        b_v = b_u - b_v
        if USE_G:
            b_g = g_smem.index(buf).reshape([BLOCK_M]).load(gl.SliceLayout(1, WMMA))
            last = gl.minimum(BLOCK_M, T - start_m) - 1
            last_index = gl.full(
                [1, 1],
                last,
                gl.int32,
                gl.BlockedLayout([1, 1], [1, 32], [gl.num_warps(), 1], [1, 0]),
            )
            # Preserve the reduction's positive zero for a last gate of -0.0.
            b_g_last = (
                gl.sum(gl.sum(g_smem.index(buf).gather(last_index, 0), 0), 0) + 0.0
            )
            b_vg = b_v * gl.where(mask_m, gl.exp(b_g_last - b_g), 0.0)[:, None]
            state = state * gl.exp(b_g_last)
        else:
            b_vg = b_v
        if SAVE_NEW_VALUE:
            vn_cur = vn_smem.index(buf)
            vn_cur.store(b_v.to(vn_cur.dtype))
        vg_smem.store(b_vg.to(vg_smem.dtype))
        gl.barrier()
        if SAVE_NEW_VALUE:
            gl.amd.gfx1250.tdm.async_store(vn_desc, [start_m, start_n], vn_cur)
        if not PAIRED:
            b_k = k_smem.index(buf).permute([1, 0]).load(OP_A)
        b_vgb = vg_smem.load(OP_B)
        # Accumulate directly into the decayed state to preserve rounding.
        state = gl.amd.gfx1250.wmma(b_k, b_vgb, state)
        h_smem.index((chunk + 1) % NB).store(state.to(h_smem.dtype))

    gl.amd.gfx1250.tdm.async_wait(0)
    if STORE_FINAL_STATE:
        gl.store(ht + offs_state, state.to(ht.dtype.element_ty))


@functools.cache
def _num_cus(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _layouts(K, BLOCK_M, BLOCK_N):
    config = _CONFIGS[BLOCK_N]
    wmma = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=config["warp_bases"],
        instr_shape=[16, 16, 32],
    )
    op_a = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    op_b = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    smem_a_row = gl.PaddedSharedLayout.with_identity_for([[K, 8]], [BLOCK_M, K], [1, 0])
    smem_a_col = gl.PaddedSharedLayout.with_identity_for(
        [[K, config["k_padding"]]], [BLOCK_M, K], [1, 0]
    )
    smem_tile = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_M, BLOCK_N], [1, 0]
    )
    smem_state = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [K, BLOCK_N], [1, 0]
    )
    smem_g = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    return wmma, op_a, op_b, smem_a_row, smem_a_col, smem_tile, smem_state, smem_g


def launch_config(H, V, total_chunks, longest_chunks, num_cus):
    """Return (value tile width, buffers, warps) from chunk counts and CU count."""

    def step_estimate(block_n):
        walk = longest_chunks * _CONFIGS[block_n]["latency"]
        work = (
            total_chunks
            * H
            * triton.cdiv(V, block_n)
            * _CONFIGS[block_n]["cost"]
            / num_cus
        )
        return max(walk, work)

    block_n = min(_CONFIGS, key=step_estimate)
    return block_n, _CONFIGS[block_n]["buffers"], _NUM_WARPS


def chunk_h_gfx1250(
    k,
    w,
    u,
    g,
    h,
    v_new,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    n_ptr,
    h0_idx,
    ht_idx,
    h0_flag,
    T,
    N,
    NT_ALL,
    longest_chunks,
    H,
    Hg,
    K,
    V,
    BLOCK_M,
    ckpt_pool,
    ckpt_slot,
):
    """Write chunk states, residual values, final states and requested checkpoints.

    k/w/u are contiguous bf16 token-major tensors; g holds fp32 log gates.
    Sequence and state indices select pool slots; ckpt_slot=-1 skips a checkpoint.
    None disables the gate, initial/final state, residual output or checkpoints.
    Writes supplied buffers and returns None. Requires K=128, BLOCK_M=64,
    V divisible by the selected value tile width, and nonempty sequences.
    """
    BLOCK_N, num_buffers, num_warps = launch_config(
        H, V, NT_ALL, longest_chunks, _num_cus(k.device.index)
    )
    layouts = _layouts(K, BLOCK_M, BLOCK_N)
    _chunk_h_gfx1250_kernel[(triton.cdiv(V, BLOCK_N), N * H)](
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
        ckpt_pool if ckpt_slot is not None else h,
        ckpt_slot if ckpt_slot is not None else h,
        T,
        NT_ALL,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_BUFFERS=num_buffers,
        USE_G=g is not None,
        USE_INITIAL_STATE=h0 is not None,
        STORE_FINAL_STATE=ht is not None,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        HAS_N=n_ptr is not None,
        USE_STATE_IDX=h0_idx is not None,
        STORE_CKPT=ckpt_slot is not None,
        PAIRED=envs.ATOM_GDN_CHUNK_H_PAIRED_LOADS and g is not None,
        WMMA=layouts[0],
        OP_A=layouts[1],
        OP_B=layouts[2],
        SMEM_A_ROW=layouts[3],
        SMEM_A_COL=layouts[4],
        SMEM_TILE=layouts[5],
        SMEM_STATE=layouts[6],
        SMEM_G=layouts[7],
        num_warps=num_warps,
    )
