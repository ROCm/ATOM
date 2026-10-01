# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Distributed top-k threshold selection for the DCP sparse-prefill indexer.

Each DCP rank scores only its own round-robin shard, so no rank can rank the
global candidates. What crosses the wire is therefore not the candidates but
the global K-th score: a token in the global top-K is in its own rank's local
top-K (fewer tokens outrank it locally than globally), so every rank already
holds every global winner it owns and only needs to know where to cut.

Two collectives agree on that cut:

  * an all-gather of four per-row scalars, reduced to a bracket [lo, hi] that
    provably contains the global K-th score;
  * an all-reduce of a per-row linear histogram over that bracket.

The cut admits the whole threshold bin, so the emitted set is a SUPERSET of the
exact global top-K -- nothing the single-rank path would select is lost.

The decode twin is ``dcp_ops.dcp_decode_candidate_exchange_fused``, which
exchanges the candidates themselves. That does not transfer: decode has ~10^2
rows, prefill has up to ``max_num_batched_tokens``.
"""

from __future__ import annotations

import torch

NEG_INF = float("-inf")
POS_INF = float("inf")

# Column layout of the exchanged per-row bracket statistics.
_MAX, _KTH, _MIN, _CNT = 0, 1, 2, 3


def row_bracket_stats(local_val: torch.Tensor) -> torch.Tensor:
    """Per-row bracket scalars for one rank's local top-k values.

    ``local_val`` is ``[rows, K]`` fp32 as ``top_k_per_row_prefill`` wrote it:
    the K winners ordered by COLUMN INDEX rather than by score, with a row
    holding fewer than K candidates tail-padded ``-inf``. "Valid" means finite.
    Every reduction below is order-independent, so nothing here depends on that
    ordering.

    Returns ``[rows, 4]`` fp32:
      0 ``local_max``        largest valid value, ``-inf`` if the row is empty
      1 ``local_kth``        the K-th largest IF the row returned K valid
                             entries, else ``-inf``
      2 ``local_min_valid``  smallest valid value, ``+inf`` if the row is empty
      3 ``local_valid_count`` number of valid entries (exact in fp32 below 2^24)

    ``local_kth`` falls out of ``local_min_valid``: when the kernel returns a
    full K entries those entries ARE the row's top-K, so their minimum is the
    K-th largest. A short row proves nothing about where the global K-th sits,
    which is what ``-inf`` encodes -- ``reduce_bracket``'s ``max`` then ignores
    that rank.
    """
    assert local_val.dtype == torch.float32, local_val.dtype
    assert local_val.dim() == 2, local_val.shape
    valid = torch.isfinite(local_val)
    rows, k = local_val.shape
    out = local_val.new_empty((rows, 4))
    neg_inf = local_val.new_full((), NEG_INF)
    out[:, _MAX] = torch.where(valid, local_val, neg_inf).amax(1)
    min_valid = torch.where(valid, local_val, local_val.new_full((), POS_INF)).amin(1)
    count = valid.sum(1, dtype=torch.float32)
    out[:, _MIN] = min_valid
    out[:, _CNT] = count
    out[:, _KTH] = torch.where(count == k, min_valid, neg_inf)
    return out


def reduce_bracket(gathered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce the all-gathered ``[W, rows, 4]`` statistics to ``(lo, hi)``.

    ``hi = max_r local_max`` is the global largest candidate.

    ``lo = max( max_r local_kth , min_r local_min_valid )``. The first term is
    the tightest valid lower bound: the rank achieving it holds K candidates at
    or above that value, so the global K-th is at least that, and nothing below
    it can enter the global top-K. The second term only takes over when NO rank
    has K local candidates -- every query token in the first K positions of a
    prefill -- where the correct answer is "select everything" and widening the
    bracket to the global minimum delivers exactly that.

    Deliberately a reduce over a gathered tensor rather than a reduction
    collective: every rank must land on bit-identical ``lo``/``hi`` or the
    per-rank owned sets stop being a partition of one global selection, and a
    fixed-order torch reduce over the same bytes guarantees that.
    """
    assert gathered.dim() == 3 and gathered.shape[-1] == 4, gathered.shape
    hi = gathered[:, :, _MAX].amax(0)
    lo = torch.maximum(gathered[:, :, _KTH].amax(0), gathered[:, :, _MIN].amin(0))
    return lo, hi
