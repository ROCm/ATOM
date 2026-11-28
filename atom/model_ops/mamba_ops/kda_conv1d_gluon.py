# SPDX-License-Identifier: Apache-2.0
"""Opt-in gfx1250 Gluon/TDM path for KDA W4 five-token verification.

The host dispatch restricts this experimental kernel to batch 128, 4608 BF16
channels, FP32 contiguous four-tap weights, and SiLU without bias or APC.
A request-bounded TDM transfer fills an eight-row LDS tile; unused rows are
zero-filled without fetching the next request. History and vectorized weights
use explicit CDNA5 buffer accesses while that transfer is outstanding. Input
and history registers serve both convolution and state update.

One workgroup owns a request/channel tile. The source places the first history
partial sum before the TDM wait; generated code overlaps history/weight products
with the transfer, while additions may follow the wait. Five convolution steps
are unrolled. This overlap is not a steady multistage token pipeline.
Read/write state slots may differ, and writes preserve the unused state tail.

Metadata is read as one uniform scalar-memory group before pad/empty filtering.
The host must provide contiguous int32 query bounds with at least batch + 1
entries and accepted-token counts with at least batch entries. Both int32 slot
index arrays must cover batch entries with the same positive stride. Metadata
storage must also cover padded and empty requests; their values never drive
input, state, or output accesses after the inactive-request check.
"""

import triton.experimental.gluon.language as gl
from triton.experimental import gluon


@gluon.jit
def _load_metadata_smem_group(in_slot_ptr, bounds_ptr, accepted_ptr, out_slot_ptr):
    # All pointers are uniform, valid global addresses. Complete every scalar
    # read inside this block so LLVM cannot reuse a source/output register while
    # its memory operation is pending. Early-clobber outputs stay disjoint from
    # all input pointer pairs. The memory clobber describes hidden memory reads.
    return gl.inline_asm_elementwise(
        asm="""
        s_load_b32 $0, $4, 0x0
        s_load_b64 $1, $5, 0x0
        s_load_b32 $2, $6, 0x0
        s_load_b32 $3, $7, 0x0
        s_wait_kmcnt 0x0
        """,
        constraints="=&s,=&s,=&s,=&s,s,s,s,s,~{memory}",
        args=(
            in_slot_ptr.to(gl.uint64),
            bounds_ptr.to(gl.uint64),
            accepted_ptr.to(gl.uint64),
            out_slot_ptr.to(gl.uint64),
        ),
        dtype=(gl.int32, gl.int64, gl.int32, gl.int32),
        is_pure=False,
        pack=1,
    )


@gluon.jit
def kda_conv1d_gluon_tdm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    conv_state_indices_in_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,
    block_idx_last_scheduled_token,
    initial_state_idx,
    query_ptr,
    key_ptr,
    value_ptr,
    batch,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    k_dim_size: gl.constexpr,
    v_dim_size: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_q_seq: gl.constexpr,
    stride_q_token: gl.constexpr,
    stride_q_dim: gl.constexpr,
    stride_k_seq: gl.constexpr,
    stride_k_token: gl.constexpr,
    stride_k_dim: gl.constexpr,
    stride_v_seq: gl.constexpr,
    stride_v_token: gl.constexpr,
    stride_v_dim: gl.constexpr,
    pad_slot_id: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    IS_APC_ENABLED: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    gl.static_assert(not IS_APC_ENABLED and (not HAS_BIAS))
    gl.static_assert(IS_VARLEN and IS_SPEC_DECODING and SILU_ACTIVATION)
    gl.static_assert(BLOCK_N == 256 and gl.num_warps() == 4)
    gl.static_assert(KERNEL_WIDTH == 4)
    gl.static_assert(seqlen == 5 and state_len == 7)
    gl.static_assert(IS_SPEC_DECODING)
    gl.static_assert(stride_x_dim == 1)
    gl.static_assert(x_ptr.dtype.element_ty == gl.bfloat16)
    gl.static_assert(conv_state_ptr.dtype.element_ty == gl.bfloat16)
    gl.static_assert(w_ptr.dtype.element_ty == gl.float32)
    idx_seq = gl.program_id(0)
    if idx_seq >= batch:
        return
    state_init = 0
    state_last = 0
    raw_in_slot, raw_bounds, early_accepted_tokens, early_out_slot = (
        _load_metadata_smem_group(
            conv_state_indices_in_ptr + idx_seq * stride_state_indices + state_init,
            query_start_loc_ptr + idx_seq,
            num_accepted_tokens_ptr + idx_seq,
            conv_state_indices_ptr + idx_seq * stride_state_indices + state_last,
        )
    )
    in_slot = raw_in_slot.to(gl.int64)
    # Each half is a signed int32 before extension, matching the original
    # independent loads even when a bound has its high bit set.
    start = raw_bounds.to(gl.int32).to(gl.int64)
    end = (raw_bounds >> 32).to(gl.int32).to(gl.int64)
    total_tokens = gl.load(query_start_loc_ptr + batch).to(gl.int64)
    actual_t = end - start
    q_base = start * stride_q_seq
    k_base = start * stride_k_seq
    v_base = start * stride_v_seq
    q_t_stride = stride_q_seq
    k_t_stride = stride_k_seq
    v_t_stride = stride_v_seq
    inactive = actual_t == 0
    if USE_PAD_SLOT:
        # Bitwise OR evaluates both predicates; neither load is made
        # control-dependent on the other predicate at the source level.
        inactive = inactive | (in_slot == pad_slot_id)
    if inactive:
        return
    accepted_offset = early_accepted_tokens.to(gl.int64) - 1
    out_slot = early_out_slot.to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [gl.num_warps()], [0])
    row_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    channel_base = gl.program_id(1) * BLOCK_N
    channel = channel_base + gl.arange(0, BLOCK_N, layout=layout)
    valid_channel = channel < dim
    FIRST_ROWS: gl.constexpr = 8
    desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=x_ptr,
        shape=(total_tokens.to(gl.int32), dim),
        strides=(stride_x_token, 1),
        block_shape=(FIRST_ROWS, BLOCK_N),
        layout=shared,
    )
    x_buf = gl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    # Bounds are relative to the updated origin and stop at this request.
    desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        desc,
        add_offsets=[start.to(gl.int32), channel_base],
        set_bounds=[actual_t.to(gl.int32), dim - channel_base],
    )
    gl.amd.cdna5.tdm.async_load(desc, dest=x_buf, warp_used_hint=1)
    history_base = (
        conv_state_ptr
        + in_slot * stride_conv_state_seq
        + accepted_offset * stride_conv_state_tok
    )
    history_desc = gl.amd.cdna5.buffer.make_buffer_descriptor(
        history_base, (2 * stride_conv_state_tok + dim * stride_conv_state_dim) * 2
    )
    h_off = (channel * stride_conv_state_dim).to(gl.int32)
    col0 = gl.amd.cdna5.buffer.load(history_desc, h_off, valid_channel, 0)
    col1 = gl.amd.cdna5.buffer.load(
        history_desc, h_off + stride_conv_state_tok, valid_channel, 0
    )
    col2 = gl.amd.cdna5.buffer.load(
        history_desc, h_off + 2 * stride_conv_state_tok, valid_channel, 0
    )
    weight_desc = gl.amd.cdna5.buffer.make_buffer_descriptor(
        w_ptr, dim * stride_w_dim * 4
    )
    weight_layout: gl.constexpr = gl.BlockedLayout(
        [2, 4], [32, 1], [gl.num_warps(), 1], [1, 0]
    )
    wc = channel_base + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, weight_layout))
    wt = gl.arange(0, 4, layout=gl.SliceLayout(0, weight_layout))
    weights = gl.amd.cdna5.buffer.load(
        weight_desc,
        (wc[:, None] * stride_w_dim + wt[None, :] * stride_w_width).to(gl.int32),
        wc[:, None] < dim,
        0,
    )
    w02, w13 = gl.split(gl.reshape(weights, (BLOCK_N, 2, 2)))
    w0, w2 = gl.split(w02)
    w1, w3 = gl.split(w13)
    w0 = gl.convert_layout(w0, layout)
    w1 = gl.convert_layout(w1, layout)
    w2 = gl.convert_layout(w2, layout)
    w3 = gl.convert_layout(w3, layout)
    bias = gl.full((BLOCK_N,), 0, gl.float32, layout)
    state_desc = gl.amd.cdna5.buffer.make_buffer_descriptor(
        conv_state_ptr + out_slot * stride_conv_state_seq,
        ((seqlen + 1) * stride_conv_state_tok + dim * stride_conv_state_dim) * 2,
    )
    query_desc = gl.amd.cdna5.buffer.make_buffer_descriptor(
        query_ptr + q_base,
        ((actual_t - 1) * q_t_stride + (k_dim_size - 1) * stride_q_dim + 1).to(gl.int32)
        * 2,
    )
    key_desc = gl.amd.cdna5.buffer.make_buffer_descriptor(
        key_ptr + k_base,
        ((actual_t - 1) * k_t_stride + (k_dim_size - 1) * stride_k_dim + 1).to(gl.int32)
        * 2,
    )
    value_desc = gl.amd.cdna5.buffer.make_buffer_descriptor(
        value_ptr + v_base,
        ((actual_t - 1) * v_t_stride + (v_dim_size - 1) * stride_v_dim + 1).to(gl.int32)
        * 2,
    )
    # Preserve the baseline FP32 accumulation order; launch without FMA fusion.
    partial0 = col0.to(gl.float32) * w0 + bias
    partial0 = partial0 + col1.to(gl.float32) * w1
    partial0 = partial0 + col2.to(gl.float32) * w2
    gl.amd.cdna5.buffer.store(col1, state_desc, h_off, valid_channel)
    gl.amd.cdna5.buffer.store(
        col2, state_desc, h_off + stride_conv_state_tok, valid_channel
    )
    # Gluon also supplies local synchronization for the LDS consumer warps.
    gl.amd.cdna5.tdm.async_wait(0)
    for t in gl.static_range(5):
        x = gl.reshape(x_buf.slice(t, 1).load(row_layout), (BLOCK_N,))
        x = gl.convert_layout(x, layout)
        active = (t < actual_t) & valid_channel
        if t == 0:
            acc = partial0
        else:
            acc = col0.to(gl.float32) * w0 + bias
            acc = acc + col1.to(gl.float32) * w1
            acc = acc + col2.to(gl.float32) * w2
        acc = acc + x.to(gl.float32) * w3
        col0 = col1
        col1 = col2
        col2 = x
        acc = acc / (1 + gl.exp(-acc))
        gl.amd.cdna5.buffer.store(
            x, state_desc, h_off + (t + 2) * stride_conv_state_tok, active
        )
        gl.amd.cdna5.buffer.store(
            acc.to(query_ptr.dtype.element_ty),
            query_desc,
            (t * q_t_stride + channel * stride_q_dim).to(gl.int32),
            active & (channel < k_dim_size),
        )
        gl.amd.cdna5.buffer.store(
            acc.to(key_ptr.dtype.element_ty),
            key_desc,
            (t * k_t_stride + (channel - k_dim_size) * stride_k_dim).to(gl.int32),
            active & (channel >= k_dim_size) & (channel < 2 * k_dim_size),
        )
        gl.amd.cdna5.buffer.store(
            acc.to(value_ptr.dtype.element_ty),
            value_desc,
            (t * v_t_stride + (channel - 2 * k_dim_size) * stride_v_dim).to(gl.int32),
            active & (channel >= 2 * k_dim_size),
        )
