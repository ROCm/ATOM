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


def constrain_logits(*args, **kwargs):
    from atom.model_ops.required_channel_gpu import constrain_logits as execute
    return execute(*args, **kwargs)


def advance_states(*args, **kwargs):
    from atom.model_ops.required_channel_gpu import advance_states as execute
    return execute(*args, **kwargs)


def _constrain_row_reference(
    row: torch.Tensor, state: int, channel: RequiredChannel, preserve_masks=False
):
    if state < 0:
        return
    if len(channel.after) <= state < len(channel.after) + len(channel.opening):
        forced = channel.table[state]
        kept = row[forced].clone()
        if not preserve_masks and kept == float("-inf"):
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
    preserve_masks: bool | torch.Tensor = False,
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
        preserve = bool(preserve_masks[i]) if isinstance(preserve_masks, torch.Tensor) else preserve_masks
        if state < 0:
            continue
        if target_logits is not None:
            start = 0 if i == 0 else int(cu_num_draft_tokens[i - 1])
            for r in range(start, int(cu_num_draft_tokens[i])):
                _constrain_row_reference(
                    target_logits[r], state, channel, preserve
                )
                draft = int(input_ids[int(target_logits_indices[r]) + 1])
                state = channel.step(state, draft)
        _constrain_row_reference(last_logits[i], state, channel, preserve)
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
