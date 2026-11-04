# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice


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
    """Compute width-four SiLU convolution, normalized Q/K, V, and final states.

    Inputs are bf16 token rows, per-channel weights, and indexed sequence states.
    Q/K heads have 128 channels; each program stays within one Q, K, or V slice.
    """
    gl.static_assert(BLOCK_M % BLOCK_T == 0)
    gl.static_assert(BLOCK_N <= k_dim_size)
    HEAD_DIM: gl.constexpr = 128
    LAYOUT_N: gl.constexpr = gl.BlockedLayout([8], [32], [2], [0])
    LAYOUT_MN: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, 2], [1, 0])
    SHARED: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_T, BLOCK_N], [1, 0]
    )
    pid = gl.program_id(0) + gl.num_programs(0) * gl.program_id(1)
    pid_m = pid // (dim // BLOCK_N)
    pid_n = pid % (dim // BLOCK_N)
    idx_seq = gl.load(batch_ptr + pid_m).to(gl.int64)
    chunk_offset = gl.load(token_chunk_offset_ptr + pid_m)
    if idx_seq == pad_slot_id:
        return
    seq_start = gl.load(query_start_loc_ptr + idx_seq).to(gl.int64)
    seq_end = gl.load(query_start_loc_ptr + idx_seq + 1).to(gl.int64)
    seqlen = (seq_end - seq_start).to(gl.int32)
    slot_in = gl.load(cache_indices_in_ptr + idx_seq * stride_cache_indices).to(
        gl.int64
    )
    if USE_PAD_SLOT:  # noqa: SIM102
        if slot_in == pad_slot_id:
            return
    load_init = (chunk_offset == 0) & gl.load(has_initial_states_ptr + idx_seq)
    start_m = chunk_offset * BLOCK_M
    start_n = pid_n * BLOCK_N
    offs_n = start_n + gl.arange(0, BLOCK_N, layout=LAYOUT_N)
    mask_n = offs_n < dim
    w0 = gl.load(w_ptr + offs_n * stride_w_dim, mask_n, 0).to(gl.float32)
    w1 = gl.load(w_ptr + offs_n * stride_w_dim + stride_w_width, mask_n, 0).to(
        gl.float32
    )
    w2 = gl.load(w_ptr + offs_n * stride_w_dim + 2 * stride_w_width, mask_n, 0).to(
        gl.float32
    )
    w3 = gl.load(w_ptr + offs_n * stride_w_dim + 3 * stride_w_width, mask_n, 0).to(
        gl.float32
    )
    bias = gl.full((BLOCK_N,), 0, gl.float32, LAYOUT_N)
    state_base = (
        conv_states_ptr + slot_in * stride_istate_seq + offs_n * stride_istate_dim
    )
    input_base = x_ptr + seq_start * stride_x_token + offs_n
    col0 = gl.load(
        input_base + (start_m - 3) * stride_x_token, mask_n & (chunk_offset > 0), 0
    )
    col1 = gl.load(
        input_base + (start_m - 2) * stride_x_token, mask_n & (chunk_offset > 0), 0
    )
    col2 = gl.load(
        input_base + (start_m - 1) * stride_x_token, mask_n & (chunk_offset > 0), 0
    )
    old0 = gl.load(state_base, mask_n & load_init, 0)
    old1 = gl.load(state_base + stride_istate_token, mask_n & load_init, 0)
    old2 = gl.load(state_base + 2 * stride_istate_token, mask_n & load_init, 0)
    col0 = gl.where(chunk_offset == 0, old0, col0)
    col1 = gl.where(chunk_offset == 0, old1, col1)
    col2 = gl.where(chunk_offset == 0, old2, col2)
    output = gl.where(
        start_n < k_dim_size,
        query_ptr,
        gl.where(start_n < 2 * k_dim_size, key_ptr, value_ptr),
    )
    width = gl.where(start_n < 2 * k_dim_size, k_dim_size, dim - 2 * k_dim_size)
    output_col = start_n - gl.where(
        start_n < k_dim_size,
        0,
        gl.where(start_n < 2 * k_dim_size, k_dim_size, 2 * k_dim_size),
    )
    stride_ym = gl.where(
        start_n < k_dim_size,
        query_token_stride,
        gl.where(start_n < 2 * k_dim_size, key_token_stride, value_token_stride),
    )
    desc_x = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        x_ptr + seq_start * stride_x_token,
        [seqlen, dim],
        [stride_x_token, 1],
        [BLOCK_T, BLOCK_N],
        SHARED,
    )
    desc_y = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        output + seq_start * stride_ym,
        [seqlen, width],
        [stride_ym, 1],
        [BLOCK_T, BLOCK_N],
        SHARED,
    )
    # Reuse the input tile after loading registers; alternate pending output tiles.
    input_tile = gl.allocate_shared_memory(
        x_ptr.dtype.element_ty, [BLOCK_T, BLOCK_N], SHARED
    )
    output_buffers = gl.allocate_shared_memory(
        query_ptr.dtype.element_ty, [2, BLOCK_T, BLOCK_N], SHARED
    )
    tiles = gl.cdiv(gl.minimum(BLOCK_M, seqlen - start_m), BLOCK_T)
    gl.amd.gfx1250.tdm.async_load(desc_x, [start_m, start_n], input_tile)
    gl.amd.gfx1250.tdm.async_wait(0)
    for tile in range(tiles):
        if tile > 0:
            gl.amd.gfx1250.tdm.async_wait(1)
        current_matrix = input_tile.load(LAYOUT_MN)
        gl.barrier()
        output_tile = output_buffers.index(tile % 2)
        tile_row = start_m + tile * BLOCK_T
        if tile + 1 < tiles:
            gl.amd.gfx1250.tdm.async_load(
                desc_x, [tile_row + BLOCK_T, start_n], input_tile
            )
        accumulators = ()
        for index in gl.static_range(BLOCK_T):
            row_index = gl.full((1, BLOCK_N), index, gl.int32, LAYOUT_MN)
            current = gl.convert_layout(
                gl.reshape(gl.gather(current_matrix, row_index, 0), (BLOCK_N,)),
                LAYOUT_N,
            )
            acc_row = gl.fma(col0.to(gl.float32), w0, bias)
            acc_row = gl.fma(col1.to(gl.float32), w1, acc_row)
            acc_row = gl.fma(col2.to(gl.float32), w2, acc_row)
            acc_row = gl.fma(current.to(gl.float32), w3, acc_row)
            col0 = col1
            col1 = col2
            col2 = current
            accumulators += (acc_row,)
        offs_m = gl.arange(0, BLOCK_T, layout=gl.SliceLayout(1, LAYOUT_MN))
        acc = gl.full((BLOCK_T, BLOCK_N), 0, gl.float32, LAYOUT_MN)
        for index in gl.static_range(BLOCK_T):
            expanded_acc = gl.convert_layout(
                accumulators[index], gl.SliceLayout(0, LAYOUT_MN)
            )[None, :]
            acc = gl.where(offs_m[:, None] == index, expanded_acc, acc)
        acc = libdevice.fast_dividef(acc, 1.0 + libdevice.fast_expf(-acc))
        if start_n < 2 * k_dim_size:
            # Normalize the rounded convolution, matching a separate Q/K epilogue.
            rounded = acc.to(query_ptr.dtype.element_ty).to(gl.float32)
            y = gl.reshape(rounded, (BLOCK_T, BLOCK_N // HEAD_DIM, HEAD_DIM))
            rstd = gl.rsqrt(gl.sum(y * y, axis=2) + eps)
            acc = gl.reshape(y * rstd[:, :, None], (BLOCK_T, BLOCK_N))
        row_indices = offs_m[:, None] + gl.full(
            (BLOCK_T, BLOCK_N), 0, gl.int32, LAYOUT_MN
        )
        output_tile.scatter(acc.to(query_ptr.dtype.element_ty), row_indices, 0)
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(desc_y, [tile_row, output_col], output_tile)
    gl.amd.gfx1250.tdm.async_wait(0)
    # Only the first program for a sequence publishes its final state.
    if chunk_offset == 0:
        offs_s = gl.arange(0, 4, layout=gl.SliceLayout(1, LAYOUT_MN))
        offs_n_r = gl.convert_layout(offs_n, gl.SliceLayout(0, LAYOUT_MN))
        src = seqlen - 3 + offs_s
        new_x = gl.load(
            x_ptr + (seq_start + src[:, None]) * stride_x_token + offs_n_r[None, :],
            (offs_s[:, None] < 3) & (src[:, None] >= 0) & (offs_n_r[None, :] < dim),
            0,
        )
        state_old = gl.load(
            conv_states_ptr
            + slot_in * stride_istate_seq
            + offs_n_r[None, :] * stride_istate_dim
            + (src[:, None] + 3) * stride_istate_token,
            (offs_s[:, None] < 3)
            & (src[:, None] < 0)
            & load_init
            & (slot_in < num_cache_lines)
            & (offs_n_r[None, :] < dim),
            0,
        )
        new_state = gl.where(src[:, None] >= 0, new_x, state_old)
        slot_out = gl.load(cache_indices_ptr + idx_seq * stride_cache_indices).to(
            gl.int64
        )
        gl.barrier()
        gl.store(
            conv_states_ptr
            + slot_out * stride_istate_seq
            + offs_n_r[None, :] * stride_istate_dim
            + offs_s[:, None] * stride_istate_token,
            new_state,
            (offs_s[:, None] < 3) & (offs_n_r[None, :] < dim),
        )


PERSISTENT_CONFIG = {"BLOCK_M": 32, "BLOCK_N": 512, "BLOCK_T": 8, "num_warps": 2}
PERSISTENT_BLOCK_M = PERSISTENT_CONFIG["BLOCK_M"]
