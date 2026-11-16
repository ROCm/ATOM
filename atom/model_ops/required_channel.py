# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A `RequiredChannel` as the kernels read it, and its pure-torch twins.

The kernels are in `triton_required_channel.py`. This module holds what they
share with the CPU tests: `DeviceChannel`, the launch constants, and the twins
that follow `RequiredChannel.step` one token at a time.
"""

from typing import NamedTuple

import torch

from atom.sampling_params import RequiredChannel

BLOCK_V = 1024
"""Vocabulary columns a `constrain_logits` program writes at a time."""

VOCAB_PROGRAMS = 16
"""`constrain_logits` programs per request, splitting a forced row's blocks.

Measured on MI355X over K3's vocabulary: a DSpark7 step with all seven draft
rows forced takes 14 / 20 / 52 us at bs 1 / 16 / 64, and one with none forced
5 / 7 / 7 us, as with one program per request. A program per block (160)
halves the first at bs 1 but quadruples the second at bs 64, and that is the
step a request takes every time until its channel opens."""


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
