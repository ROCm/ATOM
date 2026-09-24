# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""PLE tensor operators: n-gram state/hash, FP8 lookup, gate and convolution."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _advance_ngram_state_kernel(
    Ids,
    Starts,
    State,
    In,
    Out,
    Has,
    Context,
    Accepted,
    stride_ids,
    stride_starts,
    stride_state_slot,
    stride_state_col,
    stride_in,
    stride_out,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    SPEC: tl.constexpr,
    EOS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    start = tl.load(Starts + req * stride_starts)
    end = tl.load(Starts + (req + 1) * stride_starts)
    src = tl.load(In + req * stride_in)
    dst = tl.load(Out + req * stride_out)
    valid = (dst >= 0) & (end > start)
    has = tl.load(Has + req) & (src >= 0)
    offset = tl.load(Accepted + req) - 1 if SPEC else 0
    past = tl.load(
        State + src * stride_state_slot + (offset + j) * stride_state_col,
        valid & has & (j < WIDTH),
        EOS,
    )
    tl.store(Context + req * WIDTH + j, past, j < WIDTH)
    # After verification, retain the history after the anchor plus every
    # candidate. The next call selects the accepted prefix's window in-place.
    position = (start + 1 if SPEC else end) - WIDTH + j
    token = tl.load(
        Ids + position * stride_ids,
        valid & (j < CAPACITY) & (position >= start) & (position < end),
        EOS,
    )
    # Snapshot the whole old window before updating an in-place slot. A short
    # chunk carries its oldest surviving tokens; a long one replaces it all.
    carried = tl.gather(past, tl.minimum(WIDTH + position - start, WIDTH - 1), axis=0)
    tl.store(
        State + dst * stride_state_slot + j * stride_state_col,
        tl.where(position >= start, token, carried),
        valid & (j < CAPACITY),
    )


def advance_ngram_state(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    state: torch.Tensor,
    state_indices_in: torch.Tensor,
    state_indices_out: torch.Tensor,
    has_initial_state: torch.Tensor,
    eos_token_id: int,
    history_width: int | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read each request's context and commit its actual input tokens on GPU.

    The n-gram window shares PLE/GDN's state slots and fork contract: live
    destinations are unique, and a shared fork source is read-only. Cold or
    padded requests read EOS; empty/padded requests never write state. Even a
    one-token chunk writes a complete destination window.
    """
    requests = state_indices_out.numel()
    width = state.shape[1] if history_width is None else history_width
    capacity = state.shape[1] if num_accepted_tokens is not None else width
    context = state.new_empty((requests, width))
    if requests and width:
        _advance_ngram_state_kernel[(requests,)](
            input_ids,
            query_start_loc,
            state,
            state_indices_in,
            state_indices_out,
            has_initial_state,
            context,
            num_accepted_tokens,
            input_ids.stride(0),
            query_start_loc.stride(0),
            state.stride(0),
            state.stride(1),
            state_indices_in.stride(0),
            state_indices_out.stride(0),
            width,
            capacity,
            num_accepted_tokens is not None,
            eos_token_id,
            triton.next_power_of_2(capacity),
            num_warps=1,
        )
    return context


@triton.jit
def _ngram_ids_kernel(
    ids_ptr,
    starts_ptr,
    context_ptr,
    multipliers_ptr,
    sizes_ptr,
    offsets_ptr,
    out_ptr,
    num_tokens,
    num_reqs,
    stride_ids,
    stride_starts,
    stride_context_req,
    stride_context_col,
    NGRAM: tl.constexpr,
    HEADS: tl.constexpr,
    EOS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    positions = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    valid = positions < num_tokens
    # upper_bound(starts[1:], position), including zero-length requests and
    # graph padding. Derive positions here instead of casting/indexing several
    # token-sized ATen intermediates on every model step.
    lo = tl.full((BLOCK_TOKENS,), 0, tl.int32)
    hi = tl.full((BLOCK_TOKENS,), num_reqs, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        boundary = tl.load(starts_ptr + (mid + 1) * stride_starts, mid < num_reqs, 0)
        right = positions >= boundary
        active = lo < hi
        hi = tl.where(active & ~right, mid, hi)
        lo = tl.where(active & right, mid + 1, lo)
    requests = tl.minimum(lo, num_reqs - 1)
    local = positions - tl.load(starts_ptr + requests * stride_starts)
    mixed = tl.load(ids_ptr + positions * stride_ids, valid, 0).to(tl.int64)
    mixed *= tl.load(multipliers_ptr)
    live = tl.full((BLOCK_TOKENS,), True, tl.int1)
    heads = tl.arange(0, BLOCK_HEADS)
    for shift in tl.static_range(1, NGRAM):
        from_input = local >= shift
        context_col = tl.minimum(tl.maximum(NGRAM - 1 + local - shift, 0), NGRAM - 2)
        history = tl.load(
            ids_ptr + (positions - shift) * stride_ids,
            valid & from_input,
            0,
        ).to(tl.int64)
        context = tl.load(
            context_ptr
            + requests * stride_context_req
            + context_col * stride_context_col,
            valid & ~from_input,
            0,
        ).to(tl.int64)
        previous = tl.where(from_input, history, context)
        live &= previous != EOS
        previous = tl.where(live, previous, EOS)
        # Keep signed int64 wraparound/XOR and PyTorch's nonnegative remainder.
        # Triton's signed % truncates toward zero, unlike torch.remainder.
        mixed ^= previous * tl.load(multipliers_ptr + shift)
        head_index = (shift - 1) * HEADS + heads
        size = tl.load(sizes_ptr + head_index, heads < HEADS, 1)
        offset = tl.load(offsets_ptr + head_index, heads < HEADS, 0)
        remainder = mixed[:, None] % size[None, :]
        remainder = tl.where(remainder < 0, remainder + size[None, :], remainder)
        tl.store(
            out_ptr + positions[:, None] * ((NGRAM - 1) * HEADS) + head_index[None, :],
            remainder + offset[None, :],
            valid[:, None] & (heads[None, :] < HEADS),
        )


def compute_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Hash flat tokens with per-request history and EOS boundaries."""
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    input_ids = input_ids.reshape(-1)
    num_tokens = input_ids.numel()
    num_reqs = query_start_loc.numel() - 1
    if query_start_loc.ndim != 1 or num_reqs < 0:
        raise ValueError(
            "PLE query_start_loc must include at least the initial boundary"
        )
    if ngram_context.shape != (num_reqs, ngram_size - 1):
        raise ValueError("PLE ngram_context must be [requests, ngram_size - 1]")
    out = torch.empty(
        (num_tokens, ngram_heads), dtype=torch.int64, device=input_ids.device
    )
    if not num_tokens:
        return out
    if num_reqs == 0:
        raise ValueError("PLE nonempty tokens require at least one request")
    _ngram_ids_kernel[(triton.cdiv(num_tokens, 32),)](
        input_ids,
        query_start_loc,
        ngram_context,
        layer_multipliers,
        ngram_heads_vocab_sizes,
        ngram_heads_offsets,
        out,
        num_tokens,
        num_reqs,
        input_ids.stride(0),
        query_start_loc.stride(0),
        ngram_context.stride(0),
        ngram_context.stride(1),
        NGRAM=ngram_size,
        HEADS=heads_per_ngram,
        EOS=eos_token_id,
        BLOCK_TOKENS=32,
        BLOCK_HEADS=triton.next_power_of_2(heads_per_ngram),
        num_warps=4,
    )
    return out


@triton.jit
def _fp8_embedding_kernel(
    Ids,
    Weight,
    Scale,
    Output,
    START: tl.constexpr,
    END: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.load(Ids + row)
    cols = tl.arange(0, BLOCK)
    valid = (token >= START) & (token < END) & (cols < DIM)
    weight = tl.load(Weight + (token.to(tl.int64) - START) * DIM + cols, valid, 0.0)
    dtype = Output.dtype.element_ty
    scale = tl.load(Scale).to(dtype).to(tl.float32)
    values = weight.to(dtype).to(tl.float32) * scale
    tl.store(Output + row * DIM + cols, values, cols < DIM)


def fp8_embedding_lookup(
    ids: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize selected rows of a local FP8 shard; other shards yield zero."""
    dim = weight.shape[1]
    output = torch.empty((ids.numel(), dim), dtype=output_dtype, device=ids.device)
    if ids.numel() == 0:
        return output
    _fp8_embedding_kernel[(ids.numel(),)](
        ids,
        weight,
        scale,
        output,
        vocab_start_idx,
        vocab_end_idx,
        dim,
        triton.next_power_of_2(dim),
    )
    return output


@triton.jit
def _ple_gate_kernel(
    key_ptr,
    query_ptr,
    value_ptr,
    out_ptr,
    stride_key_token,
    stride_key_head,
    stride_query_token,
    stride_query_head,
    stride_value_token,
    HC: tl.constexpr,
    HIDDEN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    dtype = key_ptr.dtype.element_ty
    key = tl.load(
        key_ptr + token * stride_key_token + head * stride_key_head + cols, mask, 0
    ).to(tl.float32)
    query = tl.load(
        query_ptr + token * stride_query_token + head * stride_query_head + cols,
        mask,
        0,
    ).to(tl.float32)
    # Every eager BF16 result is a rounding boundary, including the product
    # before the reduction. Keeping only the final cast changes PLE's gate.
    product = (key * query).to(dtype).to(tl.float32)
    score = tl.sum(product, 0).to(dtype).to(tl.float32)
    score = (score / SCALE).to(dtype).to(tl.float32)
    magnitude = (
        tl.maximum(tl.abs(score), 1e-6, propagate_nan=tl.PropagateNan.ALL)
        .to(dtype)
        .to(tl.float32)
    )
    root = tl.sqrt(magnitude).to(dtype).to(tl.float32)
    sign = tl.where(score > 0, 1.0, tl.where(score < 0, -1.0, 0.0))
    gate = tl.sigmoid(sign * root).to(dtype).to(tl.float32)
    value = tl.load(value_ptr + token * stride_value_token + cols, mask, 0).to(
        tl.float32
    )
    tl.store(out_ptr + (token * HC + head) * HIDDEN + cols, gate * value, mask)


def ple_gate(
    key: torch.Tensor, query: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """sigmoid(signed_sqrt(sum(key * query) / sqrt(H))) * value, per HC stream.

    Key and query are already Gemma-normalized. Reuse that shared norm, but
    fuse the PLE-specific reduction and gate without token-sized temporaries.
    """
    if key.ndim != 3 or query.shape != key.shape:
        raise ValueError("PLE gate expects matching [tokens, hc, hidden] key/query")
    tokens, hc, hidden = key.shape
    if value.shape != (tokens, hidden):
        raise ValueError("PLE gate value must be [tokens, hidden]")
    if any(x.stride(-1) != 1 for x in (key, query, value)):
        raise ValueError("PLE gate requires contiguous hidden dimensions")
    if key.dtype != query.dtype or key.dtype != value.dtype:
        raise ValueError("PLE gate inputs must have the same dtype")
    out = key.new_empty(key.shape)
    if tokens:
        _ple_gate_kernel[(tokens, hc)](
            key,
            query,
            value,
            out,
            key.stride(0),
            key.stride(1),
            query.stride(0),
            query.stride(1),
            value.stride(0),
            HC=hc,
            HIDDEN=hidden,
            SCALE=math.sqrt(hidden),
            BLOCK=triton.next_power_of_2(hidden),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _conv(
    X,
    W,
    S,
    Starts,
    In,
    Out,
    Has,
    Y,
    Accepted,
    C: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    SPEC: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    # upper_bound on cu_seqlens: zero-length/padded requests are skipped.
    lo, hi = 0, R
    while lo < hi:
        mid = (lo + hi) // 2
        end = tl.load(Starts + mid + 1)
        before_end = token < end
        hi = tl.where(before_end, mid, hi)
        lo = tl.where(before_end, lo, mid + 1)
    req = lo
    valid_req = req < R
    start = tl.load(Starts + req, valid_req, 0)
    src = tl.load(In + req * IS, valid_req, -1)
    dst = tl.load(Out + req * OS, valid_req, -1)
    offset = tl.load(Accepted + req, valid_req, 1) - 1 if SPEC else 0
    has = tl.load(Has + req, valid_req, False) & (src >= 0)
    valid = valid_req & (dst >= 0) & (c < C)
    value = tl.full((B,), 0, tl.float32)
    for tap in tl.static_range(K):
        pos = token - (K - 1 - tap) * D
        x = tl.load(X + pos * C + c, valid & (pos >= start), 0).to(tl.float32)
        s_pos = (K - 1) * D + pos - start
        s = tl.load(
            S + src * SS0 + c * SS1 + offset + s_pos,
            valid & has & (pos < start) & (s_pos >= 0),
            0,
        ).to(tl.float32)
        w = tl.load(W + c * K + tap, c < C, 0).to(tl.float32)
        value += tl.where(pos >= start, x, s) * w
    # torch conv rounds to input dtype before the SiLU activation.
    value = value.to(Y.dtype.element_ty).to(tl.float32)
    value = value * tl.sigmoid(value)
    tl.store(Y + token * C + c, tl.where(valid, value, 0), c < C)


@triton.jit
def _update(
    X,
    S,
    Starts,
    In,
    Out,
    Has,
    Accepted,
    C: tl.constexpr,
    L: tl.constexpr,
    CAPACITY: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    SPEC: tl.constexpr,
    B: tl.constexpr,
    BL: tl.constexpr,
):
    req = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    j = tl.arange(0, BL)
    start, end = tl.load(Starts + req), tl.load(Starts + req + 1)
    src, dst = tl.load(In + req * IS), tl.load(Out + req * OS)
    offset = tl.load(Accepted + req) - 1 if SPEC else 0
    has = tl.load(Has + req) & (src >= 0)
    valid = (dst >= 0) & (end > start) & (c[:, None] < C) & (j[None, :] < CAPACITY)
    pos = (start + 1 if SPEC else end) - L + j[None, :]
    x = tl.load(X + pos * C + c[:, None], valid & (pos >= start) & (pos < end), 0)
    old_pos = L + pos - start
    old = tl.load(
        S + src * SS0 + c[:, None] * SS1 + offset + old_pos,
        valid & has & (pos < start) & (old_pos >= 0),
        0,
    )
    # Each program owns whole histories for its channels: all old entries
    # are loaded before stores, including when src == dst.
    value = tl.where(pos >= start, x, old)
    tl.store(S + dst * SS0 + c[:, None] * SS1 + j[None, :], value, valid)


def dilated_causal_conv1d(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices_in: torch.Tensor,
    state_indices_out: torch.Tensor,
    has_initial_state: torch.Tensor,
    dilation: int,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """SiLU(conv(inputs)), updating only valid, nonempty destination slots.

    inputs: [tokens, channels]; weight: [channels, kernel]; state:
    [slots, channels, history]. Destination slots must be unique among real
    requests; a source shared by forks must not also be a destination.
    Negative slots are graph padding. Workspace is O(tokens * channels), not
    O(requests * longest_request * channels). Separate output and state kernels
    ensure every token reads the old history before any in-place update.
    """
    channels, kernel = weight.shape
    history = (kernel - 1) * dilation
    capacity = state.shape[2] if num_accepted_tokens is not None else history
    requests = state_indices_out.numel()
    if inputs.ndim != 2 or inputs.shape[1] != channels or dilation < 1:
        raise ValueError("invalid dilated convolution input geometry")
    if state.shape[1] != channels or state.shape[2] < history or state.stride(2) != 1:
        raise ValueError("state must contain a contiguous history per channel")
    if query_start_loc.numel() != requests + 1:
        raise ValueError("query_start_loc must have requests + 1 entries")
    inputs, weight = inputs.contiguous(), weight.contiguous()
    output = torch.empty_like(inputs)
    if inputs.shape[0] == 0:
        return output
    _conv[(inputs.shape[0], triton.cdiv(channels, 128))](
        inputs,
        weight,
        state,
        query_start_loc,
        state_indices_in,
        state_indices_out,
        has_initial_state,
        output,
        num_accepted_tokens,
        channels,
        kernel,
        dilation,
        state.stride(0),
        state.stride(1),
        state_indices_in.stride(0),
        state_indices_out.stride(0),
        num_accepted_tokens is not None,
        requests,
        128,
        enable_fp_fusion=False,
    )
    if history and requests:
        _update[(requests, triton.cdiv(channels, 32))](
            inputs,
            state,
            query_start_loc,
            state_indices_in,
            state_indices_out,
            has_initial_state,
            num_accepted_tokens,
            channels,
            history,
            capacity,
            state.stride(0),
            state.stride(1),
            state_indices_in.stride(0),
            state_indices_out.stride(0),
            num_accepted_tokens is not None,
            32,
            triton.next_power_of_2(capacity),
        )
    return output
