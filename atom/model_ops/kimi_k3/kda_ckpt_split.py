# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's prefill checkpoints, written by cutting the recurrence at them.

FlashKDA keeps only final states, so the head path re-runs each interior
checkpoint's head, ``[cu[row], cu[row] + off)`` from the row's initial state,
as one more varlen sequence through the gathered state: a 1025-token prefill
whose prompt-end anchor sits at 1024 runs 2049 tokens of recurrence.

Here the step's tokens are cut into blocks at the interior checkpoints and
each block runs FlashKDA on the paged state, so a block starts from the state
the block before it left in the rows' runtime slots. Right after the block
that ends at a cut, the checkpoints there are copied out of those slots by the
shared writer's end-target path, conv window and all. The recurrence runs each
token once, and nothing is gathered or scattered.

Every block is a varlen call. FlashKDA reads a call's sequence bounds and chunk
indices back from ``cu_seqlens`` behind one-entry caches, so a step's blocks
would sync the host on every call; the builder knows both, attaches them to
each block's ``cu_seqlens``, and `install_host_bounds` makes FlashKDA take them.

The blocks give the head path's bits when no call is segmented, every cut is on
FlashKDA's chunk grid and the pass-C route is the same for both; elsewhere the
segments a cut moves change the rounding (same order of error against fp32).

A block of at most `TAIL_RECURRENT_MAX` tokens per row whose rows all resume a
state -- typically the tail after a prompt-end anchor -- runs on the fp32
token recurrence KDA decode uses instead of FlashKDA.
"""

from __future__ import annotations

import functools
from itertools import pairwise
from typing import NamedTuple

import torch

#: FlashKDA's chunk; `CheckpointSplit.runs` turns a plan down if it is not.
CHUNK = 32
#: Longest block (per row) that runs on the token recurrence. Measured on
#: MI355X (Kimi-K3.1, TP8): a FlashKDA call costs 22-24 us whatever the block
#: length, the token recurrence 10.5 us at 1 token, 21.1 at 8, 33.2 at 16; so
#: it only wins up to 6-7 tokens. Its fp32 state is also closer to an fp32
#: reference (tail-block error 4.40e-3 -> 3.36e-3 at 8193 tokens).
TAIL_RECURRENT_MAX = 6
_BOUNDS = "_atom_seq_bounds"
_CHUNKS = "_atom_chunk_indices"
#: Keys of the int32 lists `plan_checkpoint_split` adds to the step's transfer.
_LISTS = (
    "split_slots",
    "split_init",
    "split_cu",
    "split_chunks",
    "split_ck_rows",
    "split_ck_slots",
    "split_ck_offs",
    "split_ck_ends",
    "split_ck_runtime",
)


def _flash_kda():
    from aiter.ops.triton._triton_kernels.kimi_delta_attn import flash_kda

    return flash_kda


def _chunk_fwd():
    from aiter.ops.triton._triton_kernels.kimi_delta_attn import chunk_fwd

    return chunk_fwd


def _host_chunk_indices(chunk_indices):
    def host_chunk_indices(cu_seqlens, chunk_size):
        chunks = getattr(cu_seqlens, _CHUNKS, None)
        if chunks is None or chunk_size != CHUNK:
            return chunk_indices(cu_seqlens, chunk_size)
        return chunks

    return host_chunk_indices


@functools.cache
def install_host_bounds() -> None:
    """Make FlashKDA read a block's sequence bounds and chunk indices off the
    attributes the builder attached to its ``cu_seqlens``; any other tensor
    takes FlashKDA's own (cached, host-syncing) path. ``chunk_delta_attn_fwd``
    builds the chunk indices before it calls FlashKDA, under its own name for
    the function. Once per process."""
    fk = _flash_kda()
    seq_bounds = fk._seq_bounds

    def host_seq_bounds(cu_seqlens):
        bounds = getattr(cu_seqlens, _BOUNDS, None)
        return seq_bounds(cu_seqlens) if bounds is None else bounds

    fk._seq_bounds = host_seq_bounds
    for module in (fk, _chunk_fwd()):
        module.prepare_chunk_indices = _host_chunk_indices(module.prepare_chunk_indices)


class SplitBlock:
    """One FlashKDA call: tokens ``[start, end)`` of the step."""

    def __init__(
        self,
        start,
        end,
        cu_seqlens,
        state_indices,
        has_initial_state,
        checkpoints,
        tail=None,
    ):
        self.start, self.end = start, end
        #: ``[2, rows, longest]`` read/write state slots when the block runs on
        #: the token recurrence (`TAIL_RECURRENT_MAX`), else None.
        self.tail = tail
        self.cu_seqlens = cu_seqlens
        self.state_indices = state_indices
        self.has_initial_state = has_initial_state
        #: `write_state_checkpoints`' rows, slots, offs, is_end, runtime_slots
        #: for the targets this block ends on, or None.
        self.checkpoints = checkpoints


class CheckpointSplit:
    def __init__(self, blocks):
        self.blocks = tuple(blocks)

    def runs(self, num_heads: int, head_dim: int) -> bool:
        """Whether the forward takes the blocks rather than the heads: the
        builder's cuts assume FlashKDA's chunk."""
        return _flash_kda().FLASH_KDA_CHUNK == CHUNK


class _HostBlock(NamedTuple):
    start: int
    end: int
    #: Each row's ``(start, end)`` within the block, as FlashKDA's
    #: ``_seq_bounds`` gives them.
    bounds: tuple[tuple[int, int], ...]
    targets: int
    #: The longest row when the block runs on the token recurrence, else 0.
    tail: int = 0

    @property
    def chunks(self) -> int:
        return sum(-(-(e - s) // CHUNK) for s, e in self.bounds)


class CheckpointSplitPlan:
    """The host half: `values` go into the builder's one transfer, and `bind`
    turns their device slices into a `CheckpointSplit`."""

    def __init__(self, blocks, values):
        self.blocks, self.values = blocks, values

    def bind(self, tensors: dict[str, torch.Tensor]) -> CheckpointSplit:
        init = tensors["split_init"] != 0
        chunk_indices = tensors["split_chunks"].view(-1, 2)
        ck = [tensors[name] for name in _LISTS[4:]]
        blocks, row, chunk, target, tail_at = [], 0, 0, 0, 0
        for b in self.blocks:
            tail = None
            if b.tail:
                n = 2 * len(b.bounds) * b.tail
                tail = tensors["split_tail"][tail_at : tail_at + n].view(2, -1, b.tail)
                tail_at += n
            rows = slice(row, row + len(b.bounds))
            targets = slice(target, target + b.targets)
            cu_seqlens = tensors["split_cu"][row + len(blocks) :][: len(b.bounds) + 1]
            setattr(cu_seqlens, _BOUNDS, b.bounds)
            setattr(cu_seqlens, _CHUNKS, chunk_indices[chunk : chunk + b.chunks])
            blocks.append(
                SplitBlock(
                    start=b.start,
                    end=b.end,
                    cu_seqlens=cu_seqlens,
                    state_indices=tensors["split_slots"][rows],
                    has_initial_state=init[rows],
                    checkpoints=(tuple(t[targets] for t in ck) if b.targets else None),
                    tail=tail,
                )
            )
            row += len(b.bounds)
            chunk += b.chunks
            target += b.targets
        return CheckpointSplit(blocks)


def plan_checkpoint_split(targets, batch) -> CheckpointSplitPlan | None:
    """Blocks for this step's `midstep_checkpoint_targets`, or None where the
    head re-run stays.

    Interior targets cut the step; end targets are written after the last
    block, when every row's slot holds its final state. None when there is no
    interior target, or when a row has no state slot of its own, forks one (its
    recurrence reads one slot and writes another) or has no tokens (FlashKDA
    numbers only the sequences that have chunks).
    """
    scheduled = [int(n) for n in batch.num_scheduled_tokens]
    forks = getattr(batch, "state_fork_srcs", None) or ()
    if (
        list(batch.state_rows) != list(range(len(scheduled)))
        or any(int(src) >= 0 for src in forks)
        or 0 in scheduled
    ):
        return None
    starts = [0]
    for n in scheduled:
        starts.append(starts[-1] + n)
    interior = [t for t in targets if not t[3]]
    cuts = sorted({starts[row] + off for row, _, off, _, _ in interior})
    if not cuts:
        return None
    slots = [int(s) for s in batch.state_slots_committed]
    cached = [int(n) for n in batch.num_cached_tokens]
    values = {name: [] for name in _LISTS}
    blocks = []
    for lo, hi in pairwise([0, *cuts, starts[-1]]):
        rows = [
            r for r in range(len(scheduled)) if starts[r] < hi and lo < starts[r + 1]
        ]
        bounds = tuple(
            (max(lo, starts[r]) - lo, min(hi, starts[r + 1]) - lo) for r in rows
        )
        values["split_cu"] += [0, *(e for _, e in bounds)]
        for i, (s, e) in enumerate(bounds):
            for c in range(-(-(e - s) // CHUNK)):
                values["split_chunks"] += [i, c]
        resumes = [int(cached[r] > 0 or lo > starts[r]) for r in rows]
        for r, resume in zip(rows, resumes):
            values["split_slots"].append(slots[r])
            values["split_init"].append(resume)
        tail = max(e - s for s, e in bounds)
        tail = tail if all(resumes) and tail <= TAIL_RECURRENT_MAX else 0
        if tail:
            # The recurrence reads column 0's slot and writes column t's after
            # token t; -1 skips the write.
            idx = values.setdefault("split_tail", [])
            for r in rows:
                idx += [slots[r]] * tail
            for r, (s, e) in zip(rows, bounds):
                idx += [-1] * (e - s - 1) + [slots[r]] + [-1] * (tail - (e - s))
        if hi == starts[-1]:
            here = [t for t in targets if t[3]]
        else:
            here = [t for t in interior if starts[t[0]] + t[2] == hi]
        for row, dst, off, _, runtime in here:
            values["split_ck_rows"].append(row)
            values["split_ck_slots"].append(dst)
            values["split_ck_offs"].append(off)
            values["split_ck_ends"].append(1)
            values["split_ck_runtime"].append(runtime)
        blocks.append(_HostBlock(lo, hi, bounds, len(here), tail))
    return CheckpointSplitPlan(blocks, values)


def run_checkpoint_split(
    layer,
    q,
    k,
    v,
    gate,
    beta,
    kda_metadata,
    mixed_qkv,
    conv_state,
    ssm_state,
    split: CheckpointSplit,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """A split step's recurrence, final states and checkpoints, in place of
    the head re-run; returns ``[1, T, heads, head_dim]``, ``out``'s first
    rows when it has room (`_kda_recurrence`'s buffer).

    A block's checkpoints are written before the next block runs: that block
    moves the same runtime slots past them.
    """
    from atom.model_ops.fla_ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints

    install_host_bounds()
    if out is not None and out.shape[0] >= v.shape[1] and out.is_contiguous():
        out = out[: v.shape[1]].unsqueeze(0)
    else:
        out = torch.empty_like(v)
    beta = beta.float()
    query_start_loc = kda_metadata.non_spec_query_start_loc
    for block in split.blocks:
        rows = slice(block.start, block.end)
        if block.tail is not None:
            fused_sigmoid_gating_delta_rule_update(
                A_log=layer.A_log,
                a=gate[:, rows],
                b=beta[:, rows],
                dt_bias=layer.dt_bias,
                q=q[:, rows],
                k=k[:, rows],
                v=v[:, rows],
                o=out[0, rows],
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=block.cu_seqlens,
                ssm_state_indices=block.tail[1],
                ssm_state_indices_in=block.tail[0],
                use_qk_l2norm_in_kernel=True,
                is_kda=True,
                lower_bound=layer._kda_gate_lower_bound,
            )
        else:
            layer._chunk_kda(
                q=q[:, rows],
                k=k[:, rows],
                v=v[:, rows],
                g=gate[:, rows],
                beta=beta[:, rows],
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=layer._kda_gate_lower_bound is not None,
                lower_bound=layer._kda_gate_lower_bound,
                cu_seqlens=block.cu_seqlens,
                state_v_first=True,
                out=out[:, rows],
                state_cache=ssm_state,
                state_indices=block.state_indices,
                has_initial_state=block.has_initial_state,
            )
        if block.checkpoints is not None:
            write_state_checkpoints(
                ssm_state,
                ssm_state,
                mixed_qkv,
                conv_state,
                *block.checkpoints,
                query_start_loc,
                query_start_loc,
                chunk_size=1,
                conv_state_len=kda_metadata.ssm_conv_state_len,
            )
    return out
