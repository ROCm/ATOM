# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Vocab-parallel sampling for a speculative verify under TP.

Under TP the LM head leaves each rank a ``[rows, V / tp]`` slice of the logits. Instead of all-gathering
``[rows, V]`` and sampling every row over the whole vocabulary on every rank, each rank picks a winner per row over
its own slice with the Gumbel-max trick, the ranks all-gather those winners (``[rows, 2]``: score, global token id)
and every rank takes the same global winner.

The noise of (row, token) is Philox on the step's seed at ``row * V + global id``, so the pick is the one a single GPU
would make over the whole vocabulary with that noise, at any TP size, and the ranks agree without a broadcast.
Gumbel-max samples softmax(logits / T) exactly; a row at or below ``eps`` takes the argmax, lowest id on a tie, as the
sampler's ``tie="low"`` does.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_BLOCK = 2048


@dataclass(frozen=True)
class LocalLogits:
    """A step's logits as the LM head leaves them on this rank: every row over the vocab slice
    ``[vocab_start, vocab_start + local.shape[1])`` of ``vocab_size``. Not a tuple: ``ModelRunner.postprocess`` reads
    a tuple as a head's (ids, verdict)."""

    local: torch.Tensor
    vocab_start: int
    vocab_size: int


@triton.jit(do_not_specialize=["seed"])
def _vp_pick_kernel(
    logits_ptr,
    rows_ptr,
    temps_ptr,
    out_ptr,
    stride_row,
    n_local,
    vocab_start,
    vocab_size,
    seed,
    eps,
    BLOCK: tl.constexpr,
):
    """Program r: row ``rows[r]`` of the slice at temperature ``temps[r]``; stores (best score, its global id) as
    fp32 at ``out[r]``. The noise counter is keyed on r and the global id, never on the rank.
    """
    r = tl.program_id(0)
    src = tl.load(rows_ptr + r).to(tl.int64)
    t = tl.load(temps_ptr + r).to(tl.float32)
    greedy = t <= eps
    inv_t = 1.0 / tl.maximum(t, eps)
    offs = tl.arange(0, BLOCK)
    best = tl.full((), -float("inf"), dtype=tl.float32)
    no_id = tl.full((), 2147483647, dtype=tl.int32)
    best_id = no_id
    base = r * vocab_size + vocab_start
    for start in tl.range(0, n_local, BLOCK):
        cols = start + offs
        mask = cols < n_local
        x = tl.load(
            logits_ptr + src * stride_row + cols, mask=mask, other=-float("inf")
        ).to(tl.float32)
        u = tl.rand(seed, (base + cols).to(tl.int32))
        u = tl.minimum(tl.maximum(u, 1e-12), 1.0 - 1e-7)
        s = tl.where(greedy, x, x * inv_t - tl.log(-tl.log(u)))
        s = tl.where(mask, s, -float("inf"))
        block_best = tl.max(s, axis=0)
        gid = (vocab_start + cols).to(tl.int32)
        block_id = tl.min(tl.where((s == block_best) & mask, gid, no_id), axis=0)
        take = (block_best > best) | ((block_best == best) & (block_id < best_id))
        best = tl.where(take, block_best, best)
        best_id = tl.where(take, block_id, best_id)
    tl.store(out_ptr + 2 * r, best)
    tl.store(out_ptr + 2 * r + 1, best_id.to(tl.float32))


def vp_pick(
    logits: LocalLogits, rows: torch.Tensor, temps: torch.Tensor, seed: int, eps: float
) -> torch.Tensor:
    """This rank's winner of every row ``rows[i]`` at temperature ``temps[i]``: ``[len(rows), 2]`` fp32 (score, global
    id; ids below 2^24 are exact). Row i's noise is keyed on i, so a row picked twice in one call draws twice.
    """
    local = logits.local
    n = rows.shape[0]
    out = torch.empty((n, 2), dtype=torch.float32, device=local.device)
    if n:
        assert local.stride(1) == 1 and rows.is_contiguous() and temps.is_contiguous()
        assert n * logits.vocab_size < 2**31, "the noise counter is int32"
        assert logits.vocab_size < 2**24, "ids travel as fp32"
        _vp_pick_kernel[(n,)](
            local,
            rows,
            temps,
            out,
            local.stride(0),
            local.shape[1],
            logits.vocab_start,
            logits.vocab_size,
            seed,
            eps,
            BLOCK=_BLOCK,
            num_warps=8,
        )
    return out


def vp_reduce(gathered: torch.Tensor, n: int) -> torch.Tensor:
    """The global winner of each of ``n`` rows from every rank's ``vp_pick`` stacked in rank order (``[ranks * n, 2]``):
    the highest score, the lowest id on a tie (``argmax`` takes the first, the lowest rank's slice holds the lowest
    ids). int32 ids, the same on every rank that reduces the same input."""
    g = gathered.view(-1, n, 2)
    winner = g[:, :, 0].argmax(dim=0)
    return g[:, :, 1].gather(0, winner.unsqueeze(0)).squeeze(0).to(torch.int32)


def vp_sample(
    logits: LocalLogits,
    rows: torch.Tensor,
    temps: torch.Tensor,
    seed: int,
    eps: float,
    group,
    use_custom: bool,
) -> torch.Tensor:
    """``rows`` of the step's logits sampled over the whole vocabulary from this rank's slice: pick, all-gather the
    winners over the TP ``group``, reduce. Every rank of the group gets the same int32 ids.
    """
    picked = vp_pick(logits, rows, temps, seed, eps)
    gathered = group.all_gather(picked, dim=0, use_custom=use_custom)
    return vp_reduce(gathered, rows.shape[0])
