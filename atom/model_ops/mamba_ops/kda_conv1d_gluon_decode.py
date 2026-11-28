# SPDX-License-Identifier: Apache-2.0
"""GFX1250 KDA W4 single-token decode using explicit Gluon buffer accesses.

Channels are assigned in adjacent pairs per lane. Four contiguous FP32 weight
taps are loaded together, and input/history registers serve both convolution
and state updates. Input is issued before history and weights to overlap latency.

One descriptor per tensor covers this request's rows; request and state-slot
bases remain 64-bit while bounded within-request offsets are int32. Read and
write state slots may differ, and state storage beyond the three updated rows
is preserved. The dispatcher restricts this path to validated KDA shapes.
"""

import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.amd.cdna5 import buffer


@gluon.jit
def _conv4(h0, h1, h2, x, w0, w1, w2, w3):
    acc = h0.to(gl.float32) * w0 + 0.0
    acc = acc + h1.to(gl.float32) * w1
    acc = acc + h2.to(gl.float32) * w2
    acc = acc + x.to(gl.float32) * w3
    return acc / (1.0 + gl.exp(-acc))


@gluon.jit
def _write_qkv_buffer(
    acc,
    q,
    k,
    v,
    row,
    c,
    dim: gl.constexpr,
    split: gl.constexpr,
    qs: gl.constexpr,
    ks: gl.constexpr,
    vs: gl.constexpr,
    qd: gl.constexpr,
    kd: gl.constexpr,
    vd: gl.constexpr,
):
    # q/k/v are descriptors based at this request's first output row.
    # Cast is safe: row is a token index in [0, seqlen), never a global row.
    buffer.store(acc, q, (row * qs + c * qd).to(gl.int32), c < split)
    buffer.store(
        acc,
        k,
        (row * ks + (c - split) * kd).to(gl.int32),
        (c >= split) & (c < 2 * split),
    )
    buffer.store(
        acc,
        v,
        (row * vs + (c - 2 * split) * vd).to(gl.int32),
        (c >= 2 * split) & (c < dim),
    )


@gluon.jit
def kda_conv1d_gluon_decode_kernel(
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
    batch: int,
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
    gl.static_assert(KERNEL_WIDTH == 4 and not HAS_BIAS and not IS_APC_ENABLED)
    gl.static_assert(SILU_ACTIVATION and seqlen == 1 and state_len == 3)
    gl.static_assert(not IS_VARLEN and not IS_SPEC_DECODING)
    gl.static_assert(dim == 4608 and k_dim_size == 1536 and v_dim_size == 1536)
    gl.static_assert(BLOCK_N == 512)
    gl.static_assert(x_ptr.dtype.element_ty == gl.bfloat16)
    gl.static_assert(conv_state_ptr.dtype.element_ty == gl.bfloat16)
    gl.static_assert(w_ptr.dtype.element_ty == gl.float32)
    gl.static_assert(query_ptr.dtype.element_ty == gl.bfloat16)
    gl.static_assert(key_ptr.dtype.element_ty == gl.bfloat16)
    gl.static_assert(value_ptr.dtype.element_ty == gl.bfloat16)
    i = gl.program_id(0)
    if i >= batch:
        return
    read_slot = gl.load(conv_state_indices_in_ptr + i * stride_state_indices).to(
        gl.int64
    )
    if USE_PAD_SLOT:  # noqa: SIM102 - specialize before the runtime slot check
        if read_slot == pad_slot_id:
            return
    row = i.to(gl.int64)
    nt = 1
    xbase = x_ptr + i.to(gl.int64) * stride_x_seq
    off = 0
    write_slot = gl.load(conv_state_indices_ptr + i * stride_state_indices).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [gl.num_warps()], [0])
    c = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
    valid = c < dim
    gl.static_assert(stride_x_dim == 1 and stride_conv_state_dim == 1)
    gl.static_assert(stride_w_dim == 4 and stride_w_width == 1)
    gl.static_assert(stride_q_dim == 1 and stride_k_dim == 1 and stride_v_dim == 1)
    x_offset = c * stride_x_dim
    state_offset = c * stride_conv_state_dim
    X_BOUND: gl.constexpr = ((seqlen - 1) * stride_x_token + dim) * 2
    HISTORY_BOUND: gl.constexpr = (2 * stride_conv_state_tok + dim) * 2
    STATE_BOUND: gl.constexpr = ((seqlen + 1) * stride_conv_state_tok + dim) * 2
    Q_BOUND: gl.constexpr = ((seqlen - 1) * stride_q_seq + k_dim_size) * 2
    K_BOUND: gl.constexpr = ((seqlen - 1) * stride_k_seq + k_dim_size) * 2
    V_BOUND: gl.constexpr = ((seqlen - 1) * stride_v_seq + v_dim_size) * 2
    gl.static_assert(0 < X_BOUND < 2**31 and 0 < HISTORY_BOUND < 2**31)
    gl.static_assert(0 < STATE_BOUND < 2**31 and 0 < dim * stride_w_dim * 4 < 2**31)
    gl.static_assert(
        0 < Q_BOUND < 2**31 and 0 < K_BOUND < 2**31 and 0 < V_BOUND < 2**31
    )
    x_desc = buffer.make_buffer_descriptor(xbase, X_BOUND)
    # Issue x before the independent history/weight loads.
    x_decode = buffer.load(x_desc, x_offset, valid, 0)
    src = (
        conv_state_ptr + read_slot * stride_conv_state_seq + off * stride_conv_state_tok
    )
    history_desc = buffer.make_buffer_descriptor(src, HISTORY_BOUND)
    h0 = buffer.load(history_desc, state_offset, valid, 0)
    h1 = buffer.load(history_desc, state_offset + stride_conv_state_tok, valid, 0)
    h2 = buffer.load(history_desc, state_offset + 2 * stride_conv_state_tok, valid, 0)
    weight_desc = buffer.make_buffer_descriptor(w_ptr, dim * stride_w_dim * 4)
    # Each lane owns four contiguous FP32 taps. The tap axis is register-only.
    # Splitting it changes register indexing, without cross-lane shuffles.
    weight_layout: gl.constexpr = gl.BlockedLayout(
        [2, 4], [32, 1], [gl.num_warps(), 1], [1, 0]
    )
    wc = gl.program_id(1) * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, weight_layout)
    )
    wt = gl.arange(0, 4, layout=gl.SliceLayout(0, weight_layout))
    weights = buffer.load(
        weight_desc,
        wc[:, None] * stride_w_dim + wt[None, :] * stride_w_width,
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

    dst = conv_state_ptr + write_slot * stride_conv_state_seq
    state_desc = buffer.make_buffer_descriptor(dst, STATE_BOUND)
    q_desc = buffer.make_buffer_descriptor(query_ptr + row * stride_q_seq, Q_BOUND)
    k_desc = buffer.make_buffer_descriptor(key_ptr + row * stride_k_seq, K_BOUND)
    v_desc = buffer.make_buffer_descriptor(value_ptr + row * stride_v_seq, V_BOUND)
    buffer.store(h1, state_desc, state_offset, valid)
    buffer.store(h2, state_desc, state_offset + stride_conv_state_tok, valid)
    for t in range(nt):
        xt = x_decode
        buffer.store(
            xt,
            state_desc,
            (state_offset + (t + 2) * stride_conv_state_tok).to(gl.int32),
            valid,
        )
        acc = _conv4(h0, h1, h2, xt, w0, w1, w2, w3)
        _write_qkv_buffer(
            acc,
            q_desc,
            k_desc,
            v_desc,
            t,
            c,
            dim,
            k_dim_size,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_q_dim,
            stride_k_dim,
            stride_v_dim,
        )
        h0, h1, h2 = h1, h2, xt


# Launch with grid=(batch, 9), BLOCK_N=512, num_warps=4, enable_fp_fusion=False.
