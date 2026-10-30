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

Both kernels run one program per request. Their pure-torch twins follow
`RequiredChannel.step` one token at a time and are what the CPU tests check.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from atom.sampling_params import RequiredChannel

BLOCK_V = 4096
"""Vocabulary columns one loop iteration of a forced row rewrites."""


class DeviceChannel(NamedTuple):
    """A `RequiredChannel` as the kernels read it."""

    table: torch.Tensor  # int32 `RequiredChannel.table`
    after_len: int
    open_len: int
    end_len: int

    @classmethod
    def of(cls, channel: RequiredChannel, device) -> "DeviceChannel":
        return cls(
            torch.tensor(channel.table, dtype=torch.int32, device=device),
            len(channel.after),
            len(channel.opening),
            len(channel.end_ids),
        )


@triton.jit
def _step(s, tok, table_ptr, AFTER_LEN: tl.constexpr, OPEN_STATE: tl.constexpr):
    """`RequiredChannel.step`: an opened channel is `UNCONSTRAINED` (-1), and
    stays so."""
    want = tl.load(table_ptr + tl.maximum(tl.minimum(s, AFTER_LEN - 1), 0))
    first = tl.load(table_ptr)
    matched = tl.where(tok == want, s + 1, tl.where(tok == first, 1, 0))
    forced = tl.where(s + 1 < OPEN_STATE, s + 1, -1)
    return tl.where(s < 0, s, tl.where(s < AFTER_LEN, matched, forced))


@triton.jit
def _constrain_row(
    row_ptr,
    s,
    vocab_size,
    table_ptr,
    AFTER_LEN: tl.constexpr,
    OPEN_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Mask one logits row for state `s`; an `UNCONSTRAINED` row is left alone."""
    neg_inf = float("-inf")
    if s >= 0:
        if s < AFTER_LEN:
            for e in tl.static_range(END_LEN):
                end_id = tl.load(table_ptr + AFTER_LEN + OPEN_LEN + e)
                tl.store(row_ptr + end_id, neg_inf)
        elif s < AFTER_LEN + OPEN_LEN:
            # Every column but the forced one, in one pass. The forced logit
            # keeps its own value unless that is -inf too, which would leave
            # no finite logit in the row and a NaN softmax: it is then 0.
            forced = tl.load(table_ptr + s)
            for start in range(0, vocab_size, BLOCK_V):
                col = start + tl.arange(0, BLOCK_V)
                live = col < vocab_size
                x = tl.load(row_ptr + col, mask=live)
                kept = tl.where(x == neg_inf, 0.0, x)
                tl.store(
                    row_ptr + col, tl.where(col == forced, kept, neg_inf), mask=live
                )


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
    END_LEN: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    i = tl.program_id(0)
    s = tl.load(states_ptr + i)
    if HAS_CARRY:
        src = tl.load(carried_from_ptr + i)
        carried = tl.load(carried_ptr + tl.maximum(src, 0))
        s = tl.where(src >= 0, carried, s)
    tl.store(base_ptr + i, s)
    if s < 0:  # UNCONSTRAINED
        return

    if HAS_DRAFTS:
        start = tl.load(cu_num_draft_tokens_ptr + i - 1, mask=i > 0, other=0)
        end = tl.load(cu_num_draft_tokens_ptr + i)
        for r in range(start, end):
            _constrain_row(
                target_ptr + r.to(tl.int64) * target_stride,
                s,
                vocab_size,
                table_ptr,
                AFTER_LEN,
                OPEN_LEN,
                END_LEN,
                BLOCK_V,
            )
            draft = tl.load(input_ids_ptr + tl.load(target_logits_indices_ptr + r) + 1)
            s = _step(s, draft, table_ptr, AFTER_LEN, AFTER_LEN + OPEN_LEN)
    _constrain_row(
        last_ptr + i.to(tl.int64) * last_stride,
        s,
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
    OPEN_STATE: tl.constexpr,
):
    i = tl.program_id(0)
    s = tl.load(base_ptr + i)
    if s >= 0:
        live = s >= 0
        for j in range(width):
            tok = tl.load(sampled_ptr + i.to(tl.int64) * sampled_stride + j)
            live = live & (tok >= 0)
            s = tl.where(live, _step(s, tok, table_ptr, AFTER_LEN, OPEN_STATE), s)
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
    _constrain_logits_kernel[(bs,)](
        last_logits,
        last_logits.stride(0),
        target_logits if has_drafts else last_logits,
        target_logits.stride(0) if has_drafts else 0,
        last_logits.shape[1],
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
        OPEN_STATE=channel.after_len + channel.open_len,
    )
    return out


def _constrain_row_reference(row: torch.Tensor, state: int, channel: RequiredChannel):
    if state < 0:
        return
    if state < len(channel.after):
        row[list(channel.end_ids)] = float("-inf")
    elif state < channel.open_state:
        forced = channel.table[state]
        kept = row[forced].clone()
        if kept == float("-inf"):
            kept.zero_()
        row.fill_(float("-inf"))
        row[forced] = kept


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
