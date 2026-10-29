# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Eager oracles for the lossless speculative-sampling kernels, for tests.

Kept out of `atom/model_ops/dspark_markov_sample.py` so the shipped module
holds only what serving runs. The level grid (bins, levels per nat, the
19-nat floor) is imported from the kernel module rather than restated, so a
change there reaches the oracle too.

Not a test file (no ``test_`` prefix): it is imported, not collected. It
imports the kernel module, so import it only below a test file's AITER / GPU
guards.
"""

import torch

from atom.model_ops.dspark_markov_sample import (
    _Q_BINS,
    _Q_LEVELS_PER_NAT,
    _Q_MAX_GAP,
)


def sparse_q_reference(
    s: torch.Tensor,
    store_width: int,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
) -> torch.Tensor:
    """The q the sampled op stores for scores ``s`` [B, V] fp32, densified:
    ``[B, V]``, 0 off the stored set. For tests.

    Selects by (level, id) with a stable sort rather than through per-tile
    quotas, so it checks the kernels' bookkeeping rather than repeating it.
    The normalization sums in another order, so probabilities agree with the
    kernels' to rounding, the set exactly (the top-p cut sums level masses in
    another order too, so at a top-p it may differ by a level on a near tie).
    A row with nothing to store gets the delta on its lowest-id max.
    """
    rows, vocab = s.shape
    row_max = s.max(dim=-1, keepdim=True).values
    gap = row_max - s
    level = (1 + torch.floor(gap * _Q_LEVELS_PER_NAT)).to(torch.int64)
    level = torch.where(gap == 0, 0, level)
    sink = _Q_BINS - 1
    level = torch.where((gap >= 0) & (gap <= _Q_MAX_GAP), level, sink)
    counts = torch.zeros(rows, _Q_BINS, dtype=torch.int64, device=s.device)
    counts.scatter_add_(1, level, torch.ones_like(level))
    counts[:, sink] = 0
    limit = torch.full((rows,), store_width, dtype=torch.int64, device=s.device)
    if top_k is not None:
        limit = torch.where(top_k > 0, torch.minimum(top_k.long(), limit), limit)
    n = torch.minimum(limit, counts.sum(-1))
    if top_p is not None:
        bins = torch.arange(_Q_BINS, device=s.device, dtype=torch.float32)
        mid_gap = torch.where(bins == 0, 0.0, (bins - 0.5) / _Q_LEVELS_PER_NAT)
        mass = counts.float() * torch.exp(-mid_gap)
        before = mass.cumsum(-1) - mass
        inside = before < top_p[:, None].float() * mass.sum(-1, keepdim=True)
        nucleus = torch.where(inside, counts, 0).sum(-1)
        n = torch.where(top_p < 1.0, torch.minimum(n, nucleus), n)
    order = torch.sort(level, dim=-1, stable=True).indices
    chosen = torch.arange(vocab, device=s.device)[None, :] < n[:, None]
    weights = torch.zeros_like(s)
    weights.scatter_(1, order, torch.where(chosen, 1.0, 0.0))
    weights = torch.where(weights > 0, torch.exp(s - row_max), 0.0)
    q = weights / weights.sum(-1, keepdim=True)
    delta = torch.zeros_like(s).scatter_(1, s.argmax(-1, keepdim=True), 1.0)
    return torch.where((n > 0)[:, None], q, delta)


def dspark_markov_sample_reference(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
    store_width: int,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The eager spelling of the op, for tests: ``(ids [B], q [B, V])`` with q
    densified (`sparse_q_reference`), a greedy row's the delta on its argmax.

    Takes its ``[B, V]`` Exp(1) noise as an argument rather than drawing it,
    so a test can feed it the same draws the kernel would make, or none.
    """
    markov_embed = markov_w1[prev_ids]
    bias = torch.matmul(markov_embed.float(), markov_w2.float().t())
    logits = base_logits.float() + bias
    q = sparse_q_reference(
        logits / temperatures[:, None].float(), store_width, top_k, top_p
    )
    greedy = greedy_mask.bool()
    argmax = logits.argmax(dim=-1, keepdim=True)
    q = torch.where(greedy[:, None], torch.zeros_like(q).scatter_(1, argmax, 1.0), q)
    noise = torch.where(greedy[:, None], 1.0, exp_noise)
    return (q / noise).argmax(dim=-1), q
