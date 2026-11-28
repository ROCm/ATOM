# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The kernels of `presence_penalty.py`.

`penalize_rows` subtracts each row's penalty, in place, from the logits of the
tokens its request has generated: the bits of its table row, and on a
verification row the drafts before it (`input_ids[first + 1 .. row]`). Only
those logits are read and written. `mark_emitted` sets the bits of the tokens a
step emitted, -1 entries (rejected drafts) excluded. Before it penalizes,
`penalize_rows` clears the rows the step claimed and marks a resumed request's
earlier output.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.presence_penalty import PresenceRows

_BLOCK = 4096


@triton.jit
def _penalize_kernel(
    logits_ptr,
    logits_stride,
    vocab,
    seen_ptr,
    seen_stride,
    slots_ptr,
    penalties_ptr,
    drafts_ptr,
    input_ids_ptr,
    MAX_DRAFTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row)
    if slot >= 0:
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inb = cols < vocab
        words = tl.load(
            seen_ptr + slot.to(tl.int64) * seen_stride + cols // 32, mask=inb, other=0
        )
        hit = ((words >> (cols % 32)) & 1) != 0
        if MAX_DRAFTS > 0:
            drafts = tl.load(drafts_ptr + row)
            for k in tl.static_range(1, MAX_DRAFTS + 1):
                tok = tl.load(
                    input_ids_ptr + row - drafts + k, mask=k <= drafts, other=-1
                )
                hit = hit | (cols == tok)
        hit = hit & inb
        ptrs = logits_ptr + row.to(tl.int64) * logits_stride + cols
        x = tl.load(ptrs, mask=hit, other=0.0).to(tl.float32)
        penalty = tl.load(penalties_ptr + row)
        tl.store(ptrs, (x - penalty).to(logits_ptr.dtype.element_ty), mask=hit)


@triton.jit
def _mark_kernel(
    tokens_ptr,
    tokens_stride,
    num_cols,
    emit_ptr,
    seen_ptr,
    seen_stride,
    vocab,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    slot = tl.load(emit_ptr + r)
    if slot >= 0:
        offs = tl.arange(0, BLOCK)
        tok = tl.load(
            tokens_ptr + r * tokens_stride + offs, mask=offs < num_cols, other=-1
        ).to(tl.int32)
        ok = (tok >= 0) & (tok < vocab)
        tok = tl.where(ok, tok, 0)
        tl.atomic_or(
            seen_ptr + slot.to(tl.int64) * seen_stride + tok // 32,
            1 << (tok % 32),
            mask=ok,
        )


def penalize_rows(
    seen: torch.Tensor,
    rows: PresenceRows,
    logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    max_drafts: int,
) -> None:
    """Prepare the rows the step claimed, then penalize `logits` in place. Row
    `t` of `logits` is the step's row `t` (the source logits, before any
    gather); `input_ids` are the step's input ids on a verification step."""
    if rows.fresh.numel():
        seen.index_fill_(0, rows.fresh.long(), 0)
    vocab = min(logits.shape[-1], seen.shape[1] * 32)
    if rows.resumed_ids.numel():
        _mark(seen, rows.resumed_slots, rows.resumed_ids.view(-1, 1), vocab)
    n = rows.slots.numel()
    if n == 0:
        return
    assert logits.stride(-1) == 1
    _penalize_kernel[(n, triton.cdiv(vocab, _BLOCK))](
        logits,
        logits.stride(0),
        vocab,
        seen,
        seen.stride(0),
        rows.slots,
        rows.penalties,
        rows.drafts_before,
        rows.slots if input_ids is None else input_ids,
        MAX_DRAFTS=0 if input_ids is None else max_drafts,
        BLOCK=_BLOCK,
    )


def mark_emitted(
    seen: torch.Tensor, rows: PresenceRows, tokens: torch.Tensor, vocab: int
) -> None:
    """Set the bits of the tokens each request emitted: `tokens` holds one row
    per request (a verify's accepted drafts and bonus, -1 past them)."""
    bs = rows.emit_slots.numel()
    if bs:
        _mark(seen, rows.emit_slots, tokens.reshape(tokens.shape[0], -1)[:bs], vocab)


def _mark(seen, slots, tokens, vocab) -> None:
    """Set the bits of `tokens` row r in table row `slots[r]` (-1: none)."""
    assert tokens.stride(-1) == 1
    _mark_kernel[(slots.numel(),)](
        tokens,
        tokens.stride(0),
        tokens.shape[1],
        slots,
        seen,
        seen.stride(0),
        vocab,
        BLOCK=triton.next_power_of_2(max(tokens.shape[1], 2)),
    )
