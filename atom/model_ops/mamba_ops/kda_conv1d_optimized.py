# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 W4 Conv1D: reuse history/input registers for convolution and state.

Supported dispatch: bf16 inputs/state/output, fp32 weights, no bias/APC;
single-token decode or speculative varlen with maximum length five.
Read and write state indices may differ. Writes beyond the effective state
prefix are preserved. One thread owns each channel throughout the operation.
"""

import triton
import triton.language as tl


@triton.jit
def _conv4(h0, h1, h2, x, w0, w1, w2, w3):
    acc = h0.to(tl.float32) * w0 + 0.0
    acc = acc + h1.to(tl.float32) * w1
    acc = acc + h2.to(tl.float32) * w2
    acc = acc + x.to(tl.float32) * w3
    return acc / (1.0 + tl.exp(-acc))


@triton.jit
def _write_qkv(
    acc,
    q,
    k,
    v,
    row,
    c,
    dim: tl.constexpr,
    split: tl.constexpr,
    qs: tl.constexpr,
    ks: tl.constexpr,
    vs: tl.constexpr,
    qd: tl.constexpr,
    kd: tl.constexpr,
    vd: tl.constexpr,
):
    # Separate stores avoid AMD pointer-canonicalization issues in the baseline.
    tl.store(q + row * qs + c * qd, acc, c < split)
    tl.store(k + row * ks + (c - split) * kd, acc, (c >= split) & (c < 2 * split))
    tl.store(v + row * vs + (c - 2 * split) * vd, acc, (c >= 2 * split) & (c < dim))


@triton.jit
def kda_conv1d_reuse_kernel(
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
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    k_dim_size: tl.constexpr,
    v_dim_size: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_q_seq: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_seq: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_seq: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_dim: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    T_MODE: tl.constexpr = 0,
):
    tl.static_assert(KERNEL_WIDTH == 4 and not HAS_BIAS and not IS_APC_ENABLED)
    tl.static_assert(SILU_ACTIVATION and (seqlen == 1 or seqlen == 5))
    tl.static_assert(
        (not IS_VARLEN and not IS_SPEC_DECODING and seqlen == 1)
        or (IS_VARLEN and IS_SPEC_DECODING and seqlen == 5)
    )
    tl.static_assert(state_len == seqlen + 2)
    tl.static_assert(x_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(conv_state_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(w_ptr.dtype.element_ty == tl.float32)
    i = tl.program_id(0)
    if i >= batch:
        return
    read_slot = tl.load(conv_state_indices_in_ptr + i * stride_state_indices).to(
        tl.int64
    )
    if USE_PAD_SLOT:  # noqa: SIM102 - specialize before the runtime slot check
        if read_slot == pad_slot_id:
            return
    if IS_VARLEN:
        row = tl.load(query_start_loc_ptr + i).to(tl.int64)
        nt = tl.load(query_start_loc_ptr + i + 1).to(tl.int64) - row
        xbase = x_ptr + row * stride_x_token
    else:
        row = i.to(tl.int64)
        nt = 1
        xbase = x_ptr + i.to(tl.int64) * stride_x_seq
    if nt == 0:
        return
    if IS_SPEC_DECODING:
        off = tl.load(num_accepted_tokens_ptr + i).to(tl.int64) - 1
    else:
        off = 0
    write_slot = tl.load(conv_state_indices_ptr + i * stride_state_indices).to(tl.int64)
    c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = c < dim
    xb = xbase + c * stride_x_dim
    # Issue decode x before the history/weight loads. Reusing a late x load
    # saves instructions but can expose its miss latency with rotating buffers.
    if seqlen == 1:
        x_decode = tl.load(xb, valid, 0)
    src = conv_state_ptr + read_slot * stride_conv_state_seq + c * stride_conv_state_dim
    dst = (
        conv_state_ptr + write_slot * stride_conv_state_seq + c * stride_conv_state_dim
    )
    # Identical 1-D ownership for every history load and state store: no
    # cross-thread state movement and no block-wide synchronization required.
    h0 = tl.load(src + off * stride_conv_state_tok, valid, 0)
    h1 = tl.load(src + (off + 1) * stride_conv_state_tok, valid, 0)
    h2 = tl.load(src + (off + 2) * stride_conv_state_tok, valid, 0)
    wb = w_ptr + c * stride_w_dim
    w0 = tl.load(wb, valid, 0)
    w1 = tl.load(wb + stride_w_width, valid, 0)
    w2 = tl.load(wb + 2 * stride_w_width, valid, 0)
    w3 = tl.load(wb + 3 * stride_w_width, valid, 0)

    # MODE 2 eagerly loads the whole bounded verification window. Every input
    # vector is consumed by both state movement and convolution exactly once.
    if T_MODE == 2 and seqlen == 5:
        x0 = tl.load(xb, valid & (nt > 0), 0)
        x1 = tl.load(xb + stride_x_token, valid & (nt > 1), 0)
        x2 = tl.load(xb + 2 * stride_x_token, valid & (nt > 2), 0)
        x3 = tl.load(xb + 3 * stride_x_token, valid & (nt > 3), 0)
        x4 = tl.load(xb + 4 * stride_x_token, valid & (nt > 4), 0)

    tl.store(dst, h1, valid)
    tl.store(dst + stride_conv_state_tok, h2, valid)
    if T_MODE == 0:
        for t in tl.range(nt):
            if seqlen == 1:
                xt = x_decode
            else:
                xt = tl.load(xb + t * stride_x_token, valid, 0)
            tl.store(dst + (t + 2) * stride_conv_state_tok, xt, valid)
            acc = _conv4(h0, h1, h2, xt, w0, w1, w2, w3)
            _write_qkv(
                acc,
                query_ptr,
                key_ptr,
                value_ptr,
                row + t,
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
    else:
        for t in tl.static_range(seqlen):
            if t < nt:
                if seqlen == 1:
                    xt = x_decode
                elif T_MODE == 2 and seqlen == 5:
                    if t == 0:
                        xt = x0
                    elif t == 1:
                        xt = x1
                    elif t == 2:
                        xt = x2
                    elif t == 3:
                        xt = x3
                    else:
                        xt = x4
                else:
                    xt = tl.load(xb + t * stride_x_token, valid, 0)
                tl.store(dst + (t + 2) * stride_conv_state_tok, xt, valid)
                acc = _conv4(h0, h1, h2, xt, w0, w1, w2, w3)
                _write_qkv(
                    acc,
                    query_ptr,
                    key_ptr,
                    value_ptr,
                    row + t,
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
