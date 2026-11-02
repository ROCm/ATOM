# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hold a step's target logits to each request's `RequiredChannel`.

A request with a required channel (Kimi-K3's tools channel under
`tool_choice="required"`) may not end before the channel opens, and once its
output ends with `after` the next tokens are `opening`. Both are decided per
logits row, from the state the output has reached by that row:

* a decode row's state is its sequence's state after the anchor;
* a verification row's state is that, advanced over the drafts before it --
  the row checks the draft that follows those, so it must be constrained as
  if they had been accepted;
* the bonus row comes after every draft.

Masking the target rows is enough for every sampling path. A sampled target
row that disagrees with its draft rejects it, greedy verification compares
against the argmax of the masked row, and the bonus is sampled from its masked
row like a plain decode.

Where the anchor's state comes from:

* the host, when it has every token up to the anchor (`batch` state, staged
  with the other sampling parameters);
* the previous forward's output, on the GPU, when deferred output means the
  host has not seen that forward's tokens yet (`carried_from` names the row).
  `advance_states` computes that output after each forward from the tokens
  actually emitted, so the next one reads it in place, as it reads the anchor
  itself.

The host cannot patch a token after the fact: the next forward has already
taken the sampled ids as its input, and its KV and recurrent state with them.

`constrain_logits` runs `VOCAB_PROGRAMS` programs per request: a forced row
spans K3's whole 163,840-id vocabulary, and one program per request wrote it
serially at ~58 us a row (~400 us for a DSpark7 step's seven). Each program
writes every `VOCAB_PROGRAMS`-th block of the row, -inf everywhere but the
forced id, without reading it. Few enough programs per request that a step
with no forced row costs what one program per request did. `advance_states`
runs one program per request. The pure-torch twins follow
`RequiredChannel.step` one token at a time and are what the CPU tests check.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from atom.sampling_params import UNCONSTRAINED, RequiredChannel

BLOCK_V = 1024
"""Vocabulary columns a `constrain_logits` program writes at a time."""

VOCAB_PROGRAMS = 16
"""`constrain_logits` programs per request, splitting a forced row's blocks.

Measured on MI355X over K3's vocabulary: a DSpark7 step with all seven draft
rows forced takes 14 / 20 / 52 us at bs 1 / 16 / 64, and one with none forced
5 / 7 / 7 us, as with one program per request. A program per block (160)
halves the first at bs 1 but quadruples the second at bs 64, and that is the
step a request takes every time until its channel opens."""

# The host's own sentinel, so both sides keep one definition of it.
_UNCONSTRAINED = tl.constexpr(UNCONSTRAINED)


class DeviceChannel(NamedTuple):
    """A `RequiredChannel` as the kernels read it."""

    table: torch.Tensor  # int32 `RequiredChannel.table`
    after_len: int
    open_len: int
    opens_at: int
    end_len: int

    @classmethod
    def of(cls, channel: RequiredChannel, device) -> "DeviceChannel":
        return cls(
            torch.tensor(channel.table, dtype=torch.int32, device=device),
            len(channel.after),
            len(channel.opening),
            channel.opens_at,
            len(channel.end_ids),
        )


@triton.jit
def _step(
    s,
    tok,
    table_ptr,
    AFTER_LEN: tl.constexpr,
    OPEN_LEN: tl.constexpr,
    OPENS_AT: tl.constexpr,
):
    """`RequiredChannel.step`, move for move."""
    forced_end = AFTER_LEN + OPEN_LEN
    in_after = s < AFTER_LEN
    forcing = (s >= AFTER_LEN) & (s < forced_end)
    own = s - forced_end + 1  # past the forced states: it ends with opening[:own]
    want = tl.load(
        table_ptr
        + tl.minimum(
            tl.maximum(tl.where(in_after, s, AFTER_LEN + own), 0), forced_end - 1
        )
    )
    own_next = tl.where(own + 1 >= OPENS_AT, _UNCONSTRAINED, s + 1)
    matched = tl.where(in_after, s + 1, own_next)
    if OPENS_AT == 1:
        own_first = _UNCONSTRAINED
    else:
        own_first = forced_end
    missed = tl.where(
        tok == tl.load(table_ptr),
        1,
        tl.where(tok == tl.load(table_ptr + AFTER_LEN), own_first, 0),
    )
    forced = tl.where(s + 1 < forced_end, s + 1, _UNCONSTRAINED)
    return tl.where(
        s < 0,
        s,
        tl.where(forcing, forced, tl.where(tok == want, matched, missed)),
    )


@triton.jit
def _constrain_row_block(
    row_ptr,
    s,
    vb,
    nprog,
    vocab_size,
    table_ptr,
    AFTER_LEN: tl.constexpr,
    OPEN_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Mask program `vb`'s share of one logits row for state `s`.

    A forced row is the whole vocabulary, so each of the request's `nprog`
    programs writes every `nprog`-th block of it; a row that only bans the end
    ids is program 0's alone. An `UNCONSTRAINED` row is left alone.
    """
    neg_inf = float("-inf")
    forced_end = AFTER_LEN + OPEN_LEN
    if (s >= AFTER_LEN) & (s < forced_end):
        forced = tl.load(table_ptr + s)
        # Written, not read: only the forced column keeps a value.
        for first in range(vb * BLOCK_V, vocab_size, nprog * BLOCK_V):
            col = first + tl.arange(0, BLOCK_V)
            tl.store(row_ptr + col, neg_inf, mask=(col < vocab_size) & (col != forced))
        if (forced // BLOCK_V) % nprog == vb:
            # Its own value, unless the model ruled it out too: then 0, so the
            # row keeps a finite logit and its softmax is not NaN.
            x = tl.load(row_ptr + forced)
            tl.store(row_ptr + forced, tl.where(x == neg_inf, 0.0, x))
    elif (s >= 0) & (vb == 0):
        # No all -inf guard here, unlike the forced branch: this writes only
        # the few end ids and leaves every other column as the LM head wrote
        # it, finite, so the row keeps all of them. A guard would have to read
        # the whole row to know, on the step a request takes every time until
        # its channel opens.
        for e in tl.static_range(END_LEN):
            tl.store(row_ptr + tl.load(table_ptr + forced_end + e), neg_inf)


@triton.jit(do_not_specialize=["vocab_size"])
def _constrain_logits_kernel(
    last_ptr,  # [>=bs, V] -- each request's plain decode or bonus row
    last_stride,
    target_ptr,  # [num_drafts, V] -- verification rows, when HAS_DRAFTS
    target_stride,
    vocab_size,
    states_ptr,  # [bs] int32 -- the host's state after each anchor
    carried_from_ptr,  # [bs] int32 -- previous forward's row, or < 0
    carried_ptr,  # [num_prev] int32 -- previous forward's states
    base_ptr,  # [bs] int32 out -- the state each request starts this step in
    cu_num_draft_tokens_ptr,  # [bs] int32 -- inclusive prefix sum of drafts
    input_ids_ptr,  # the step's input ids
    target_logits_indices_ptr,  # [num_drafts]
    table_ptr,
    HAS_CARRY: tl.constexpr,
    HAS_DRAFTS: tl.constexpr,
    AFTER_LEN: tl.constexpr,
    OPEN_LEN: tl.constexpr,
    OPENS_AT: tl.constexpr,
    END_LEN: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Every one of the request's programs walks its drafts for the states;
    # each writes only its own share of the rows.
    i = tl.program_id(0)
    vb = tl.program_id(1)
    nprog = tl.num_programs(1)
    s = tl.load(states_ptr + i)
    if HAS_CARRY:
        src = tl.load(carried_from_ptr + i)
        carried = tl.load(carried_ptr + tl.maximum(src, 0))
        s = tl.where(src >= 0, carried, s)
    if vb == 0:
        tl.store(base_ptr + i, s)
    if s < 0:  # UNCONSTRAINED
        return

    if HAS_DRAFTS:
        start = tl.load(cu_num_draft_tokens_ptr + i - 1, mask=i > 0, other=0)
        end = tl.load(cu_num_draft_tokens_ptr + i)
        for r in range(start, end):
            _constrain_row_block(
                target_ptr + r.to(tl.int64) * target_stride,
                s,
                vb,
                nprog,
                vocab_size,
                table_ptr,
                AFTER_LEN,
                OPEN_LEN,
                END_LEN,
                BLOCK_V,
            )
            draft = tl.load(input_ids_ptr + tl.load(target_logits_indices_ptr + r) + 1)
            s = _step(s, draft, table_ptr, AFTER_LEN, OPEN_LEN, OPENS_AT)
    _constrain_row_block(
        last_ptr + i.to(tl.int64) * last_stride,
        s,
        vb,
        nprog,
        vocab_size,
        table_ptr,
        AFTER_LEN,
        OPEN_LEN,
        END_LEN,
        BLOCK_V,
    )


@triton.jit(do_not_specialize=["width"])
def _advance_states_kernel(
    base_ptr,  # [bs] int32
    sampled_ptr,  # [bs, width] int32 -- emitted ids, -1 after the last
    sampled_stride,
    width,
    out_ptr,  # [bs] int32
    table_ptr,
    AFTER_LEN: tl.constexpr,
    OPEN_LEN: tl.constexpr,
    OPENS_AT: tl.constexpr,
):
    i = tl.program_id(0)
    s = tl.load(base_ptr + i)
    if s >= 0:
        live = s >= 0
        for j in range(width):
            tok = tl.load(sampled_ptr + i.to(tl.int64) * sampled_stride + j)
            live = live & (tok >= 0)
            s = tl.where(
                live, _step(s, tok, table_ptr, AFTER_LEN, OPEN_LEN, OPENS_AT), s
            )
    tl.store(out_ptr + i, s)


def constrain_logits(
    channel: DeviceChannel,
    states: torch.Tensor,
    last_logits: torch.Tensor,
    *,
    carried_from: torch.Tensor | None = None,
    carried_states: torch.Tensor | None = None,
    target_logits: torch.Tensor | None = None,
    cu_num_draft_tokens: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    target_logits_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask the step's target rows in place; return each request's base state.

    Args:
      channel:      the batch's channel, on the device.
      states:       `[bs]` int32 -- the host's state after each anchor, or
                    `UNCONSTRAINED`.
      last_logits:  `[>= bs, V]` -- row `i` is request `i`'s decode row, or
                    its bonus row on a verification step.
      carried_from: `[bs]` int32 -- the row each request held in the previous
                    forward, negative for one the host has caught up with;
                    None when no row carries over.
      carried_states: `[num_prev]` int32 -- `advance_states` of the previous
                    forward, which `carried_from` indexes.
      target_logits, cu_num_draft_tokens, input_ids, target_logits_indices:
                    a verification step's rows, laid out as the rejection
                    sampler reads them; draft `r` is
                    `input_ids[target_logits_indices[r] + 1]`.

    The base state is what `advance_states` folds this step's output into.
    """
    bs = int(states.shape[0])
    base = torch.empty_like(states)
    if bs == 0:
        return base
    has_carry = carried_from is not None
    has_drafts = target_logits is not None
    assert last_logits.stride(1) == 1
    assert not has_drafts or target_logits.stride(1) == 1
    vocab_size = last_logits.shape[1]
    nprog = min(VOCAB_PROGRAMS, triton.cdiv(vocab_size, BLOCK_V))
    _constrain_logits_kernel[(bs, nprog)](
        last_logits,
        last_logits.stride(0),
        target_logits if has_drafts else last_logits,
        target_logits.stride(0) if has_drafts else 0,
        vocab_size,
        states,
        carried_from if has_carry else states,
        carried_states if has_carry else states,
        base,
        cu_num_draft_tokens if has_drafts else states,
        input_ids if has_drafts else states,
        target_logits_indices if has_drafts else states,
        channel.table,
        HAS_CARRY=has_carry,
        HAS_DRAFTS=has_drafts,
        AFTER_LEN=channel.after_len,
        OPEN_LEN=channel.open_len,
        OPENS_AT=channel.opens_at,
        END_LEN=channel.end_len,
        BLOCK_V=BLOCK_V,
    )
    return base


def advance_states(
    channel: DeviceChannel, base: torch.Tensor, sampled_tokens: torch.Tensor
) -> torch.Tensor:
    """Each request's state after the tokens this step emitted.

    `sampled_tokens` is `[bs]` or `[bs, width]`, with -1 after each row's last
    emitted id -- the rejection sampler's layout."""
    bs = int(base.shape[0])
    out = torch.empty_like(base)
    if bs == 0:
        return out
    sampled = sampled_tokens.view(bs, -1)
    _advance_states_kernel[(bs,)](
        base,
        sampled,
        sampled.stride(0),
        sampled.shape[1],
        out,
        channel.table,
        AFTER_LEN=channel.after_len,
        OPEN_LEN=channel.open_len,
        OPENS_AT=channel.opens_at,
    )
    return out


def _constrain_row_reference(row: torch.Tensor, state: int, channel: RequiredChannel):
    if state < 0:
        return
    if len(channel.after) <= state < len(channel.after) + len(channel.opening):
        forced = channel.table[state]
        kept = row[forced].clone()
        if kept == float("-inf"):
            kept.zero_()
        row.fill_(float("-inf"))
        row[forced] = kept
    else:
        row[list(channel.end_ids)] = float("-inf")


def constrain_logits_reference(
    channel: RequiredChannel,
    states: torch.Tensor,
    last_logits: torch.Tensor,
    *,
    carried_from: torch.Tensor | None = None,
    carried_states: torch.Tensor | None = None,
    target_logits: torch.Tensor | None = None,
    cu_num_draft_tokens: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    target_logits_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-torch twin of `constrain_logits`, runnable on CPU.

    A loop over requests and rows through `RequiredChannel.step`, so it states
    which row gets which state rather than re-deriving the kernel's moves.
    """
    base = states.clone()
    if carried_from is not None:
        for i, src in enumerate(carried_from.tolist()):
            if src >= 0:
                base[i] = carried_states[src]
    for i, state in enumerate(base.tolist()):
        if state < 0:
            continue
        if target_logits is not None:
            start = 0 if i == 0 else int(cu_num_draft_tokens[i - 1])
            for r in range(start, int(cu_num_draft_tokens[i])):
                _constrain_row_reference(target_logits[r], state, channel)
                draft = int(input_ids[int(target_logits_indices[r]) + 1])
                state = channel.step(state, draft)
        _constrain_row_reference(last_logits[i], state, channel)
    return base


def advance_states_reference(
    channel: RequiredChannel, base: torch.Tensor, sampled_tokens: torch.Tensor
) -> torch.Tensor:
    """Pure-torch twin of `advance_states`, runnable on CPU."""
    rows = sampled_tokens.view(base.shape[0], -1).tolist()
    out = []
    for state, row in zip(base.tolist(), rows):
        emitted = row[: row.index(-1)] if -1 in row else row
        out.append(channel.advance(state, emitted))
    return torch.tensor(out, dtype=base.dtype)
