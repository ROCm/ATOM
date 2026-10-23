# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A step's inputs from its tokens in one launch: ``h0`` and ``emb``.

ATOM's ops take six: the token embeddings, their copy into the four HC
streams, ``advance_ngram_state``, ``compute_ngram_ids``, the n-gram table's
lookup and its copy into ``emb``. A few rows make each a few microseconds of
launch for almost no work. :class:`StepPrologue` runs them as one kernel of
two kinds of programs:

- one per request: its n-gram window read and rewritten as
  ``advance_ngram_state`` does, its tokens hashed as ``compute_ngram_ids``
  does from the window it read, and their table rows copied into ``emb``
  (``TB`` tokens' rows, every head's, in one load);
- one per ``BLOCK_H`` columns of a token's embedding, copied into its streams.

``compute_ngram_ids`` gives a token the request whose span holds it and the
last request every token past the spans; the request programs split the tokens
the same way, the spans being ascending (``query_start_loc``). A request's
program reads its window before it rewrites it, as ``advance_ngram_state``'s
does, so the requests' slots keep that op's contract: live destinations are
unique, a shared source is read only.

Bit-identical to the ops: every value is a copy or the same integer hash.
"""

import torch
import triton
import triton.language as tl

BF16 = torch.bfloat16


@triton.jit
def _request_program(
    req,
    ids_ptr,
    starts_ptr,
    state_ptr,
    in_ptr,
    out_ptr,
    has_ptr,
    accepted_ptr,
    mult_ptr,
    sizes_ptr,
    offsets_ptr,
    table_ptr,
    emb_ptr,
    n_tok,
    n_req,
    stride_ids,
    stride_starts,
    stride_state_slot,
    stride_state_col,
    stride_in,
    stride_out,
    stride_table,
    stride_emb,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    SPEC: tl.constexpr,
    EOS: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    TB: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # advance_ngram_state's program: the window it reads (``past``), then its rewrite
    j = tl.arange(0, BLOCK)
    start = tl.load(starts_ptr + req * stride_starts)
    end = tl.load(starts_ptr + (req + 1) * stride_starts)
    src = tl.load(in_ptr + req * stride_in)
    dst = tl.load(out_ptr + req * stride_out)
    valid = (dst >= 0) & (end > start)
    has = tl.load(has_ptr + req) & (src >= 0)
    offset = tl.load(accepted_ptr + req) - 1 if SPEC else 0
    past = tl.load(
        state_ptr + src * stride_state_slot + (offset + j) * stride_state_col,
        valid & has & (j < WIDTH),
        EOS,
    )

    # compute_ngram_ids on the request's tokens, its context the window read;
    # TB tokens at a time, all their heads' table rows in one load: a row's
    # load waits on nothing but its hash
    last = tl.where(req == n_req - 1, n_tok, end)
    KP: tl.constexpr = WIDTH * HEADS
    k = tl.arange(0, BLOCK_K)
    km = k < KP
    k_shift = k // HEADS + 1
    size = tl.load(sizes_ptr + k, km, 1)[None, :]
    base = tl.load(offsets_ptr + k, km, 0)[None, :]
    d = tl.arange(0, BLOCK_D)
    for t0 in range(start, last, TB):
        t = t0 + tl.arange(0, TB)
        live_t = (t < last) & (t < n_tok)
        local = t - start
        mixed = tl.load(ids_ptr + t * stride_ids, live_t, 0).to(tl.int64)
        mixed *= tl.load(mult_ptr)
        live = tl.full((TB,), True, tl.int1)
        mixed_k = tl.zeros((TB, BLOCK_K), tl.int64)
        for shift in tl.static_range(1, WIDTH + 1):
            from_input = local >= shift
            col = tl.minimum(tl.maximum(WIDTH + local - shift, 0), WIDTH - 1)
            history = tl.load(
                ids_ptr + (t - shift) * stride_ids, live_t & from_input, 0
            ).to(tl.int64)
            # past[col]: the one column that matches
            context = tl.sum(tl.where(j[None, :] == col[:, None], past[None, :], 0), 1)
            previous = tl.where(from_input, history, context.to(tl.int64))
            live &= previous != EOS
            previous = tl.where(live, previous, EOS)
            # signed int64 wraparound / XOR and torch's nonnegative remainder
            mixed ^= previous * tl.load(mult_ptr + shift)
            mixed_k = tl.where((k_shift == shift)[None, :], mixed[:, None], mixed_k)
        r = mixed_k % size
        row = tl.where(r < 0, r + size, r) + base
        m = live_t[:, None, None] & km[None, :, None] & (d < HEAD_DIM)[None, None, :]
        v = tl.load(table_ptr + row[:, :, None] * stride_table + d[None, None, :], m)
        tl.store(
            emb_ptr
            + t[:, None, None] * stride_emb
            + (k * HEAD_DIM)[None, :, None]
            + d[None, None, :],
            v,
            m,
        )

    position = (start + 1 if SPEC else end) - WIDTH + j
    token = tl.load(
        ids_ptr + position * stride_ids,
        valid & (j < CAPACITY) & (position >= start) & (position < end),
        EOS,
    )
    carried = tl.gather(past, tl.minimum(WIDTH + position - start, WIDTH - 1), axis=0)
    tl.store(
        state_ptr + dst * stride_state_slot + j * stride_state_col,
        tl.where(position >= start, token, carried),
        valid & (j < CAPACITY),
    )


@triton.jit
def _step_prologue_kernel(
    ids_ptr,
    starts_ptr,
    state_ptr,
    in_ptr,
    out_ptr,
    has_ptr,
    accepted_ptr,
    mult_ptr,
    sizes_ptr,
    offsets_ptr,
    table_ptr,
    emb_ptr,
    embed_ptr,
    h0_ptr,
    n_tok,
    n_req,
    stride_ids,
    stride_starts,
    stride_state_slot,
    stride_state_col,
    stride_in,
    stride_out,
    stride_table,
    stride_emb,
    stride_embed,
    stride_h0,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    SPEC: tl.constexpr,
    EOS: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
    TB: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < n_req:
        _request_program(
            pid,
            ids_ptr,
            starts_ptr,
            state_ptr,
            in_ptr,
            out_ptr,
            has_ptr,
            accepted_ptr,
            mult_ptr,
            sizes_ptr,
            offsets_ptr,
            table_ptr,
            emb_ptr,
            n_tok,
            n_req,
            stride_ids,
            stride_starts,
            stride_state_slot,
            stride_state_col,
            stride_in,
            stride_out,
            stride_table,
            stride_emb,
            WIDTH,
            CAPACITY,
            SPEC,
            EOS,
            HEADS,
            HEAD_DIM,
            BLOCK,
            TB,
            BLOCK_K,
            BLOCK_D,
        )
    else:
        # a token's BLOCK_H columns of its embedding into its HC streams
        blocks: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
        t = (pid - n_req) // blocks
        c = ((pid - n_req) % blocks) * BLOCK_H + tl.arange(0, BLOCK_H)
        m = c < H
        tok = tl.load(ids_ptr + t * stride_ids).to(tl.int64)
        e = tl.load(embed_ptr + tok * stride_embed + c, m)
        for s in tl.static_range(HC):
            tl.store(h0_ptr + t * stride_h0 + s * H + c, e, m)


def refusal(embed_tokens, ple_embedding, hc_width) -> str | None:
    """Why :class:`StepPrologue` cannot take a step's tokens to its ``h0`` and
    ``emb``; None if it can."""
    table = ple_embedding.ngram_embedding
    for name, lookup in (("embed_tokens", embed_tokens), ("the n-gram table", table)):
        w = getattr(lookup, "weight", None)
        if getattr(lookup, "tp_size", 1) != 1:
            return f"{name} is sharded"
        if w is None or w.dtype != BF16 or w.dim() != 2 or w.stride(1) != 1:
            return f"{name} is not a bf16 table"
    if table.weight.shape[1] != ple_embedding.head_dim:
        return "the n-gram table's rows are not one head wide"
    if hc_width % embed_tokens.weight.shape[1]:
        return "the HC streams are not whole embeddings"
    return None


class StepPrologue:
    """``(ids [rows], the step's PLE metadata)`` into ``h0 [rows, hc_width]``
    and ``emb [rows, ple_dim]``, as ``Qwen4MonoDecodeRunner.embed``'s ops."""

    BLOCK_H = 512
    # tokens a request's program hashes and looks up at a time: a verify span
    TB = 4

    def __init__(self, embed_tokens, ple_embedding):
        e = ple_embedding
        self.embed = embed_tokens.weight
        self.table = e.ngram_embedding.weight
        self.mult = e.layer_multipliers
        self.sizes = e.ngram_heads_vocab_sizes
        self.offsets = e.ngram_heads_offsets
        self.width = e.ngram_size - 1
        self.heads = e.heads_per_ngram
        self.head_dim = e.head_dim
        self.eos = e.eos_token_id

    def __call__(self, ids, md, h0, emb):
        n_tok, hidden = ids.numel(), self.embed.shape[1]
        starts, state = md.query_start_loc, md.ngram_state
        n_req = md.state_indices_out.numel()
        spec = md.num_accepted_tokens is not None
        assert n_req and starts.numel() == n_req + 1
        assert h0.shape[0] == emb.shape[0] == n_tok and h0.stride(1) == 1
        assert emb.shape[1] == self.width * self.heads * self.head_dim
        assert emb.stride(1) == 1
        capacity = state.shape[1] if spec else self.width
        blocks = triton.cdiv(hidden, self.BLOCK_H)
        _step_prologue_kernel[(n_req + n_tok * blocks,)](
            ids,
            starts,
            state,
            md.state_indices_in,
            md.state_indices_out,
            md.has_initial_state,
            md.num_accepted_tokens if spec else starts,
            self.mult,
            self.sizes,
            self.offsets,
            self.table,
            emb,
            self.embed,
            h0,
            n_tok,
            n_req,
            ids.stride(0),
            starts.stride(0),
            state.stride(0),
            state.stride(1),
            md.state_indices_in.stride(0),
            md.state_indices_out.stride(0),
            self.table.stride(0),
            emb.stride(0),
            self.embed.stride(0),
            h0.stride(0),
            WIDTH=self.width,
            CAPACITY=capacity,
            SPEC=spec,
            EOS=self.eos,
            HEADS=self.heads,
            HEAD_DIM=self.head_dim,
            H=hidden,
            HC=h0.shape[1] // hidden,
            BLOCK=triton.next_power_of_2(capacity),
            TB=self.TB,
            BLOCK_K=triton.next_power_of_2(self.width * self.heads),
            BLOCK_D=triton.next_power_of_2(self.head_dim),
            BLOCK_H=self.BLOCK_H,
            num_warps=4,
        )
