# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language import core


@core.extern
def _tanh(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(core.dtype("fp32"),): ("llvm.amdgcn.tanh.f32", core.dtype("fp32"))},
        is_pure=True,
        _semantic=_semantic,
    )


@gluon.jit
def _causal_conv1d_persistent_kernel(
    x_ptr,
    w_ptr,
    conv_states_ptr,
    cache_indices_ptr,
    cache_indices_in_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    eps,
    dim: gl.constexpr,
    k_dim_size: gl.constexpr,
    num_cache_lines: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_istate_seq: gl.constexpr,
    stride_istate_dim: gl.constexpr,
    stride_istate_token: gl.constexpr,
    stride_cache_indices: gl.constexpr,
    query_token_stride: gl.constexpr,
    key_token_stride: gl.constexpr,
    value_token_stride: gl.constexpr,
    pad_slot_id: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_T: gl.constexpr,
):
    """y[m, n] = silu(sum_k w[n, k] * x[m - 3 + k, n]) for one block of tokens.

    x rows are [q | k | v] channels; a program does BLOCK_N channels of one of them.
    q and k get an L2 norm over each head of 128 channels. Window positions before the
    sequence start read the initial state (or zero). The program of the first token
    block writes the last three tokens of the sequence to the output state slot.
    """
    gl.static_assert(BLOCK_M % BLOCK_T == 0)
    gl.static_assert(BLOCK_N <= k_dim_size)
    HEAD_DIM: gl.constexpr = 128
    LAYOUT_N: gl.constexpr = gl.BlockedLayout([8], [32], [2], [0])
    LAYOUT_MN: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 2], [1, 0])
    SHARED: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_T, BLOCK_N], [1, 0]
    )
    # Consecutive programs take the column blocks of one token block.
    pid = gl.program_id(0) + gl.num_programs(0) * gl.program_id(1)
    pid_m = pid // (dim // BLOCK_N)
    pid_n = pid % (dim // BLOCK_N)
    seq = gl.load(batch_ptr + pid_m).to(gl.int64)
    chunk = gl.load(token_chunk_offset_ptr + pid_m)
    if seq == pad_slot_id:
        return
    seq_start = gl.load(query_start_loc_ptr + seq).to(gl.int64)
    seq_end = gl.load(query_start_loc_ptr + seq + 1).to(gl.int64)
    seqlen = (seq_end - seq_start).to(gl.int32)
    slot_in = gl.load(cache_indices_in_ptr + seq * stride_cache_indices).to(gl.int64)
    if USE_PAD_SLOT:  # noqa: SIM102
        if slot_in == pad_slot_id:
            return
    load_init = (chunk == 0) & gl.load(has_initial_states_ptr + seq)
    start_m = chunk * BLOCK_M
    start_n = pid_n * BLOCK_N
    offs_n = start_n + gl.arange(0, BLOCK_N, layout=LAYOUT_N)
    mask_n = offs_n < dim
    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        x_ptr + seq_start * stride_x_token,
        [seqlen, dim],
        [stride_x_token, 1],
        [BLOCK_T, BLOCK_N],
        SHARED,
    )
    # One input tile: the next tile loads into it when the current one is in registers.
    x_smem = gl.allocate_shared_memory(
        x_ptr.dtype.element_ty, [BLOCK_T, BLOCK_N], SHARED
    )
    gl.amd.gfx1250.tdm.async_load(x_desc, [start_m, start_n], x_smem)

    # Half weights: the taps give h = x / 2, and silu(x) = h + h * tanh(h).
    w_cols = w_ptr + offs_n * stride_w_dim
    w0 = gl.load(w_cols, mask_n, 0).to(gl.float32) * 0.5
    w1 = gl.load(w_cols + stride_w_width, mask_n, 0).to(gl.float32) * 0.5
    w2 = gl.load(w_cols + 2 * stride_w_width, mask_n, 0).to(gl.float32) * 0.5
    w3 = gl.load(w_cols + 3 * stride_w_width, mask_n, 0).to(gl.float32) * 0.5

    # Window x[start_m - 3 : start_m]: earlier tokens, or the initial state (block 0).
    x_cols = x_ptr + seq_start * stride_x_token + offs_n
    mask_prev = mask_n & (chunk > 0)
    x0 = gl.load(x_cols + (start_m - 3) * stride_x_token, mask_prev, 0)
    x1 = gl.load(x_cols + (start_m - 2) * stride_x_token, mask_prev, 0)
    x2 = gl.load(x_cols + (start_m - 1) * stride_x_token, mask_prev, 0)
    state_cols = (
        conv_states_ptr + slot_in * stride_istate_seq + offs_n * stride_istate_dim
    )
    s0 = gl.load(state_cols, mask_n & load_init, 0)
    s1 = gl.load(state_cols + stride_istate_token, mask_n & load_init, 0)
    s2 = gl.load(state_cols + 2 * stride_istate_token, mask_n & load_init, 0)
    x0 = gl.where(chunk == 0, s0, x0)
    x1 = gl.where(chunk == 0, s1, x1)
    x2 = gl.where(chunk == 0, s2, x2)

    # The program's columns are in one of q, k, v.
    is_q = start_n < k_dim_size
    is_qk = start_n < 2 * k_dim_size
    y_ptr = gl.where(is_q, query_ptr, gl.where(is_qk, key_ptr, value_ptr))
    y_width = gl.where(is_qk, k_dim_size, dim - 2 * k_dim_size)
    y_col = start_n - gl.where(is_q, 0, gl.where(is_qk, k_dim_size, 2 * k_dim_size))
    stride_ym = gl.where(
        is_q, query_token_stride, gl.where(is_qk, key_token_stride, value_token_stride)
    )
    y_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        y_ptr + seq_start * stride_ym,
        [seqlen, y_width],
        [stride_ym, 1],
        [BLOCK_T, BLOCK_N],
        SHARED,
    )
    # Two output tiles: the store of the previous tile stays in flight.
    y_smem = gl.allocate_shared_memory(
        query_ptr.dtype.element_ty, [2, BLOCK_T, BLOCK_N], SHARED
    )
    offs_t = gl.arange(0, BLOCK_T, layout=gl.SliceLayout(1, LAYOUT_MN))
    num_tiles = gl.cdiv(gl.minimum(BLOCK_M, seqlen - start_m), BLOCK_T)
    gl.amd.gfx1250.tdm.async_wait(0)
    for tile in range(num_tiles):
        gl.amd.gfx1250.tdm.async_wait(1)
        x = x_smem.load(LAYOUT_MN)
        gl.barrier()
        y_tile = y_smem.index(tile % 2)
        tile_m = start_m + tile * BLOCK_T
        if tile + 1 < num_tiles:
            gl.amd.gfx1250.tdm.async_load(x_desc, [tile_m + BLOCK_T, start_n], x_smem)
        # Slide the window over the tile's rows.
        rows = ()
        for t in gl.static_range(BLOCK_T):
            x3 = gl.convert_layout(
                gl.reshape(
                    gl.gather(x, gl.full((1, BLOCK_N), t, gl.int32, LAYOUT_MN), 0),
                    (BLOCK_N,),
                ),
                LAYOUT_N,
            )
            h = x0.to(gl.float32) * w0
            h = gl.fma(x1.to(gl.float32), w1, h)
            h = gl.fma(x2.to(gl.float32), w2, h)
            h = gl.fma(x3.to(gl.float32), w3, h)
            x0 = x1
            x1 = x2
            x2 = x3
            rows += (h,)
        acc = gl.full((BLOCK_T, BLOCK_N), 0, gl.float32, LAYOUT_MN)
        for t in gl.static_range(BLOCK_T):
            h_row = gl.convert_layout(rows[t], gl.SliceLayout(0, LAYOUT_MN))[None, :]
            acc = gl.where(offs_t[:, None] == t, h_row, acc)
        acc = gl.fma(acc, _tanh(acc), acc)
        if is_qk:
            y = gl.reshape(acc, (BLOCK_T, BLOCK_N // HEAD_DIM, HEAD_DIM))
            rstd = gl.rsqrt(gl.sum(y * y, axis=2) + eps)
            acc = gl.reshape(y * rstd[:, :, None], (BLOCK_T, BLOCK_N))
        y_tile.store(acc.to(query_ptr.dtype.element_ty))
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(y_desc, [tile_m, y_col], y_tile)
    gl.amd.gfx1250.tdm.async_wait(0)

    # Block 0 writes the final state: the last three tokens of x or the initial state.
    if chunk == 0:
        offs_s = gl.arange(0, 4, layout=gl.SliceLayout(1, LAYOUT_MN))
        offs_n_r = gl.convert_layout(offs_n, gl.SliceLayout(0, LAYOUT_MN))
        src = seqlen - 3 + offs_s
        mask_s = (offs_s[:, None] < 3) & (offs_n_r[None, :] < dim)
        x_last = gl.load(
            x_ptr + (seq_start + src[:, None]) * stride_x_token + offs_n_r[None, :],
            mask_s & (src[:, None] >= 0),
            0,
        )
        state_last = gl.load(
            conv_states_ptr
            + slot_in * stride_istate_seq
            + offs_n_r[None, :] * stride_istate_dim
            + (src[:, None] + 3) * stride_istate_token,
            mask_s & (src[:, None] < 0) & load_init & (slot_in < num_cache_lines),
            0,
        )
        new_state = gl.where(src[:, None] >= 0, x_last, state_last)
        slot_out = gl.load(cache_indices_ptr + seq * stride_cache_indices).to(gl.int64)
        gl.barrier()
        gl.store(
            conv_states_ptr
            + slot_out * stride_istate_seq
            + offs_n_r[None, :] * stride_istate_dim
            + offs_s[:, None] * stride_istate_token,
            new_state,
            mask_s,
        )


PERSISTENT_CONFIG = {
    "BLOCK_M": 32,
    "BLOCK_N": 512,
    "BLOCK_T": 8,
    "num_warps": 2,
    "waves_per_eu": 6,
}
PERSISTENT_BLOCK_M = PERSISTENT_CONFIG["BLOCK_M"]
