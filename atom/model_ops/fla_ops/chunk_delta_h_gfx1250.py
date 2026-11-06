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
# Per value tile width: ring depth (of long and of short walks), wave layouts of the
# two products, k padding, early operand reads, and the step latency / work cost
# that launch_config weighs.
_CONFIGS = {
    128: {
        "buffers": 2,
        "short_buffers": 2,
        "warp_bases": [[0, 1], [1, 0]],
        "value_warp_bases": [[0, 1], [1, 0]],
        "k_padding": 8,
        "latency": 1.13,
        "cost": 2.1,
        "early_reads": False,
    },
    64: {
        "buffers": 4,
        "short_buffers": 3,
        "warp_bases": [[1, 0], [2, 0]],
        "value_warp_bases": [[0, 1], [1, 0]],
        "k_padding": 8,
        "latency": 0.83,
        "cost": 1.15,
        "early_reads": True,
    },
    32: {
        "buffers": 4,
        "short_buffers": 3,
        "warp_bases": [[1, 0], [2, 0]],
        "value_warp_bases": [[1, 0], [2, 0]],
        "k_padding": 16,
        "latency": 0.71,
        "cost": 0.88,
        "early_reads": False,
    },
}
# Walks of up to this many chunks use the short-walk ring depth (at most one
# buffer more than the walk's chunks).
_SHORT_WALK = 16
# Step-latency units per extra wave of workgroups beyond one per CU.
_WAVE_COST = 2.0


@gluon.jit
def _tile_descriptor(desc, row, col):
    return gl.amd.gfx1250.tdm.update_tensor_descriptor(
        desc, add_offsets=[row, col], clamp_bounds=True
    )


@gluon.jit
def _load_chunk(
    w_desc,
    k_desc,
    u_desc,
    g_desc,
    w_smem,
    k_smem,
    u_smem,
    g_smem,
    buf,
    start_m,
    start_n,
    g_row,
    PAIRED: gl.constexpr,
):
    # TDM loads of one chunk's w, k, u tiles and gate row into ring slot buf.
    if PAIRED:
        w_dst = w_smem.index(buf)
        k_dst = k_smem.index(buf)
        u_dst = u_smem.index(buf)
        g_dst = g_smem.index(buf)
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
                (_tile_descriptor(g_desc, g_row, 0), g_dst, 12),
            ]
        )
    else:
        gl.amd.gfx1250.tdm.async_load(w_desc, [start_m, 0], w_smem.index(buf))
        gl.amd.gfx1250.tdm.async_load(k_desc, [start_m, 0], k_smem.index(buf))
        gl.amd.gfx1250.tdm.async_load(u_desc, [start_m, start_n], u_smem.index(buf))
        gl.amd.gfx1250.tdm.async_load(g_desc, [g_row, 0], g_smem.index(buf))


@gluon.jit
def _chunk_h_gfx1250_kernel(
    k,
    u,
    w,
    v_new,
    g,
    decay,
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
    ckpt_pool,
    ckpt_slot,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    EARLY_READS: gl.constexpr,
    USE_DECAY: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    STORE_FINAL_STATE: gl.constexpr,
    SAVE_NEW_VALUE: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    HAS_N: gl.constexpr,
    USE_STATE_IDX: gl.constexpr,
    STORE_CKPT: gl.constexpr,
    PAIRED: gl.constexpr,
):
    WMMA: gl.constexpr = _layout("WMMA", K, BLOCK_M, BLOCK_N)
    OP_A: gl.constexpr = _layout("OP_A", K, BLOCK_M, BLOCK_N)
    OP_B: gl.constexpr = _layout("OP_B", K, BLOCK_M, BLOCK_N)
    WMMA_V: gl.constexpr = _layout("WMMA_V", K, BLOCK_M, BLOCK_N)
    OP_A_V: gl.constexpr = _layout("OP_A_V", K, BLOCK_M, BLOCK_N)
    OP_B_V: gl.constexpr = _layout("OP_B_V", K, BLOCK_M, BLOCK_N)
    SMEM_A_ROW: gl.constexpr = _layout("SMEM_A_ROW", K, BLOCK_M, BLOCK_N)
    SMEM_A_COL: gl.constexpr = _layout("SMEM_A_COL", K, BLOCK_M, BLOCK_N)
    SMEM_TILE: gl.constexpr = _layout("SMEM_TILE", K, BLOCK_M, BLOCK_N)
    SMEM_STATE: gl.constexpr = _layout("SMEM_STATE", K, BLOCK_M, BLOCK_N)
    SMEM_G: gl.constexpr = _layout("SMEM_G", K, BLOCK_M, BLOCK_N)
    if Hg % 8 == 0:
        # Workgroups go round-robin over 8 XCDs: keep the value tiles of a head
        # and the heads of a k head on one XCD.
        TILES: gl.constexpr = V // BLOCK_N
        GROUP: gl.constexpr = H // Hg
        wg = gl.program_id(1) * TILES + gl.program_id(0)
        pid_n = (wg // 8) % TILES
        pid_g = (wg // (8 * TILES * GROUP)) * 8 + wg % 8  # sequence * Hg + k head
        pid_m = pid_g * GROUP + (wg // (8 * TILES)) % GROUP
    else:
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
    if USE_DECAY:
        # A row per (chunk, head): exp(g_last - g_i) of its rows, then exp(g_last) repeated.
        g_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=decay,
            shape=(NT_ALL * H, 2 * BLOCK_M),
            strides=(2 * BLOCK_M, 1),
            block_shape=(1, 2 * BLOCK_M),
            layout=SMEM_G,
        )
        g_base = first_chunk * H + head
        G_STEP: gl.constexpr = H
    else:
        g_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=g + start_token.to(gl.int64) * H + head,
            shape=(T, 1),
            strides=(H, 1),
            block_shape=(BLOCK_M, 1),
            layout=SMEM_G,
        )
        g_base = 0
        G_STEP: gl.constexpr = BLOCK_M
    NB: gl.constexpr = NUM_BUFFERS
    w_smem = gl.allocate_shared_memory(w.type.element_ty, [NB, BLOCK_M, K], SMEM_A_ROW)
    k_smem = gl.allocate_shared_memory(k.type.element_ty, [NB, BLOCK_M, K], SMEM_A_COL)
    u_smem = gl.allocate_shared_memory(
        u.type.element_ty, [NB, BLOCK_M, BLOCK_N], SMEM_TILE
    )
    if USE_DECAY:
        g_smem = gl.allocate_shared_memory(gl.float32, [NB, 1, 2 * BLOCK_M], SMEM_G)
    else:
        g_smem = gl.allocate_shared_memory(gl.float32, [NB, BLOCK_M, 1], SMEM_G)
    # Ring buffers keep bf16 states and residuals live until their stores drain.
    # K-major [BLOCK_N, K] state tiles: b128 stores of the state's WMMA rows and
    # untransposed B operand loads.
    h_smem = gl.allocate_shared_memory(k.type.element_ty, [NB, BLOCK_N, K], SMEM_STATE)
    if SAVE_NEW_VALUE:
        vn_smem = gl.allocate_shared_memory(
            k.type.element_ty, [NB, BLOCK_M, BLOCK_N], SMEM_TILE
        )
    vg_smem = gl.allocate_shared_memory(
        k.type.element_ty, [BLOCK_M, BLOCK_N], SMEM_TILE
    )
    h_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=h,
        shape=(NT_ALL * H * V, K),
        strides=(K, 1),
        block_shape=(BLOCK_N, K),
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
    LOADS: gl.constexpr = 2 if PAIRED else 4
    PER_CHUNK: gl.constexpr = 1 + LOADS + SAVE_NEW_VALUE

    offs_k = gl.arange(0, K, layout=gl.SliceLayout(1, WMMA))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA))
    # The nonnegative-offset masks preserve pointer-width inference.
    offs_state = offs_k[:, None] * V + (start_n + offs_n)[None, :]
    # Initial and final states move in rows of V, converted to and from the WMMA layout.
    IO_LANES_N: gl.constexpr = 4 if BLOCK_N == 32 else 8
    STATE_IO: gl.constexpr = gl.BlockedLayout(
        [1, 8], [32 // IO_LANES_N, IO_LANES_N], [gl.num_warps(), 1], [1, 0]
    )
    offs_k_io = gl.arange(0, K, layout=gl.SliceLayout(1, STATE_IO))
    offs_n_io = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, STATE_IO))
    offs_io = offs_k_io[:, None] * V + (start_n + offs_n_io)[None, :]
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
    # Prefetch operands before loading the initial state. A repeated u load stands in
    # for each store of a loop chunk, so every chunk waits for the same TDM count.
    for i in gl.static_range(NB - 1):
        start_m = gl.minimum(i, num_chunks - 1) * BLOCK_M
        g_row = g_base + gl.minimum(i, num_chunks - 1) * G_STEP
        gl.amd.gfx1250.tdm.async_load(u_desc, [start_m, start_n], u_smem.index(i))
        _load_chunk(
            w_desc,
            k_desc,
            u_desc,
            g_desc,
            w_smem,
            k_smem,
            u_smem,
            g_smem,
            i,
            start_m,
            start_n,
            g_row,
            PAIRED,
        )
        if SAVE_NEW_VALUE:
            gl.amd.gfx1250.tdm.async_load(u_desc, [start_m, start_n], u_smem.index(i))
    if USE_INITIAL_STATE and h0.dtype.element_ty == gl.bfloat16:
        # A bf16 state converts in the shared memory left beside the ring buffers.
        b_h0 = gl.load(h0 + offs_io, mask=(offs_io >= 0) & use_h0, other=0.0)
        state = gl.convert_layout(b_h0, WMMA).to(gl.float32)
    elif USE_INITIAL_STATE:
        state = gl.load(h0 + offs_state, mask=(offs_state >= 0) & use_h0, other=0.0).to(
            gl.float32
        )
    else:
        state = gl.zeros([K, BLOCK_N], gl.float32, layout=WMMA)
    h_smem.index(0).permute([1, 0]).store(state.to(h_smem.dtype))

    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_V))
    zero_v = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=WMMA_V)

    for chunk in range(num_chunks):
        buf = chunk % NB
        gl.amd.gfx1250.tdm.async_wait(SAVE_NEW_VALUE + PER_CHUNK * (NB - 2))
        # Wait for all waves before reading or reusing shared buffers.
        gl.barrier()
        if STORE_CKPT:
            # Checkpoints preserve the incoming fp32 state before bf16 rounding.
            cslot = gl.load(ckpt_slot + first_chunk + chunk)
            if cslot >= 0:
                gl.store(
                    ckpt_pool + (cslot.to(gl.int64) * H + head) * K * V + offs_state,
                    state.to(ckpt_pool.dtype.element_ty),
                    mask=offs_state >= 0,
                )
        h_cur = h_smem.index(buf)
        if EARLY_READS:
            # Read the operands of the first product before the TDM issue.
            b_h = h_cur.permute([1, 0]).load(OP_B_V)
            b_w = w_smem.index(buf).load(OP_A_V)
            if PAIRED:
                b_k = k_smem.index(buf).permute([1, 0]).load(OP_A)
        gl.amd.gfx1250.tdm.async_store(
            h_desc,
            [((first_chunk + chunk) * H + head) * V + start_n, 0],
            h_cur,
        )
        next_c = gl.minimum(chunk + NB - 1, num_chunks - 1)
        next_m = next_c * BLOCK_M
        next_g = g_base + next_c * G_STEP
        next_buf = (chunk + NB - 1) % NB
        _load_chunk(
            w_desc,
            k_desc,
            u_desc,
            g_desc,
            w_smem,
            k_smem,
            u_smem,
            g_smem,
            next_buf,
            next_m,
            start_n,
            next_g,
            PAIRED,
        )
        start_m = chunk * BLOCK_M
        if not EARLY_READS:
            b_h = h_cur.permute([1, 0]).load(OP_B_V)
            b_w = w_smem.index(buf).load(OP_A_V)
            if PAIRED:
                # Read k while the first dependent product runs.
                b_k = k_smem.index(buf).permute([1, 0]).load(OP_A)
        b_v = gl.amd.gfx1250.wmma(b_w, b_h, zero_v)
        b_u = u_smem.index(buf).load(WMMA_V).to(gl.float32)
        b_v = b_u - b_v
        if USE_DECAY:
            g_cur = g_smem.index(buf).reshape([2 * BLOCK_M])
            b_e = g_cur.slice(0, BLOCK_M).load(gl.SliceLayout(1, WMMA_V))
            ONE: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
            b_decay = gl.sum(g_cur.slice(BLOCK_M, 1).load(ONE), 0)
            b_vg = b_v * b_e[:, None]
            state = state * b_decay
        else:
            b_g = g_smem.index(buf).reshape([BLOCK_M]).load(gl.SliceLayout(1, WMMA_V))
            mask_m = (start_m + offs_m) < T
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
        h_smem.index((chunk + 1) % NB).permute([1, 0]).store(state.to(h_smem.dtype))

    gl.amd.gfx1250.tdm.async_wait(0)
    if STORE_FINAL_STATE:
        gl.store(ht + offs_io, gl.convert_layout(state.to(ht.dtype.element_ty), STATE_IO))


@functools.cache
def _num_cus(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@gluon.constexpr_function
def _layout(name, K, BLOCK_M, BLOCK_N):
    """Layout `name` of the kernel (WMMA, its operands, LDS tiles) for a value tile width."""
    config = _CONFIGS[BLOCK_N]
    # The state's D layout holds runs of K: b128 stores into the K-major state tile.
    wmma = gl.amd.AMDWMMALayout(
        version=3,
        transposed=False,
        warp_bases=config["warp_bases"],
        instr_shape=[16, 16, 32],
    )
    op_a = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    op_b = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    # Layout of the first product and of its value tiles (u, v_new).
    wmma_v = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=config["value_warp_bases"],
        instr_shape=[16, 16, 32],
    )
    op_a_v = gl.DotOperandLayout(operand_index=0, parent=wmma_v, k_width=8)
    op_b_v = gl.DotOperandLayout(operand_index=1, parent=wmma_v, k_width=8)
    smem_a_row = gl.PaddedSharedLayout.with_identity_for([[K, 8]], [BLOCK_M, K], [1, 0])
    smem_a_col = gl.PaddedSharedLayout.with_identity_for(
        [[K, config["k_padding"]]], [BLOCK_M, K], [1, 0]
    )
    smem_tile = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_M, BLOCK_N], [1, 0]
    )
    smem_state = gl.PaddedSharedLayout.with_identity_for([[K, 8]], [BLOCK_N, K], [1, 0])
    smem_g = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    return {
        "WMMA": wmma,
        "OP_A": op_a,
        "OP_B": op_b,
        "WMMA_V": wmma_v,
        "OP_A_V": op_a_v,
        "OP_B_V": op_b_v,
        "SMEM_A_ROW": smem_a_row,
        "SMEM_A_COL": smem_a_col,
        "SMEM_TILE": smem_tile,
        "SMEM_STATE": smem_state,
        "SMEM_G": smem_g,
    }[name]


@functools.lru_cache(maxsize=64)
def launch_config(N, H, V, total_chunks, longest_chunks, num_cus):
    """Return (value tile width, buffers, warps) from the grid, chunk counts and CU count."""

    def step_estimate(block_n):
        workgroups = N * H * triton.cdiv(V, block_n)
        walk = longest_chunks * _CONFIGS[block_n]["latency"]
        if workgroups <= num_cus:
            return walk
        # More workgroups than CUs: the work spread over the CUs, plus each extra wave.
        work = (
            total_chunks
            * H
            * triton.cdiv(V, block_n)
            * _CONFIGS[block_n]["cost"]
            / num_cus
        )
        return max(walk, work) + _WAVE_COST * (triton.cdiv(workgroups, num_cus) - 1)

    block_n = min(_CONFIGS, key=step_estimate)
    if longest_chunks <= _SHORT_WALK:
        buffers = min(_CONFIGS[block_n]["short_buffers"], longest_chunks + 1)
    else:
        buffers = _CONFIGS[block_n]["buffers"]
    return block_n, buffers, _NUM_WARPS


def chunk_h_gfx1250(
    k,
    w,
    u,
    g,
    decay,
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
    decay (fp32 [NT_ALL, H, 2, 64] or None) replaces g: per (chunk, head) the rows
    exp(g_last - g_i) (0 past the sequence end) and exp(g_last).
    Sequence and state indices select pool slots; ckpt_slot=-1 skips a checkpoint.
    None disables the initial/final state, residual output or checkpoints.
    Writes supplied buffers and returns None. Requires K=128, BLOCK_M=64,
    V divisible by the selected value tile width, and nonempty sequences.
    """
    BLOCK_N, num_buffers, num_warps = launch_config(
        N, H, V, NT_ALL, longest_chunks, _num_cus(k.device.index)
    )
    _chunk_h_gfx1250_kernel[(triton.cdiv(V, BLOCK_N), N * H)](
        k,
        u,
        w,
        v_new,
        g,
        decay if decay is not None else h,
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
        ckpt_pool if ckpt_slot is not None else h,
        ckpt_slot if ckpt_slot is not None else h,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_BUFFERS=num_buffers,
        EARLY_READS=_CONFIGS[BLOCK_N]["early_reads"],
        USE_DECAY=decay is not None,
        USE_INITIAL_STATE=h0 is not None,
        STORE_FINAL_STATE=ht is not None,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        HAS_N=n_ptr is not None,
        USE_STATE_IDX=h0_idx is not None,
        STORE_CKPT=ckpt_slot is not None,
        PAIRED=envs.ATOM_GDN_CHUNK_H_PAIRED_LOADS,
        num_warps=num_warps,
    )
