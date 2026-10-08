# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DCP prefill indexer's LOCAL causal window, checked against the layout.

A query token's global window is [0, p]; locally it is [0, how many of those p+1
positions this rank owns]. Getting that count wrong by one silently scores the
wrong keys, so it is pinned here against `dcp_owner_rank`, the function that
defines ownership.
"""

import numpy as np
import pytest

from atom.distributed.dcp_layout import (
    dcp_local_prefix_count,
    dcp_owner_rank,
    dcp_prefill_local_window,
)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_prefix_count_matches_brute_force_ownership(world, interleave):
    n_max = 300
    pos = np.arange(n_max)
    for rank in range(world):
        owned = (dcp_owner_rank(pos, world, interleave) == rank).astype(np.int64)
        brute = np.concatenate([[0], np.cumsum(owned)])  # brute[n] = count in [0, n)
        got = dcp_local_prefix_count(np.arange(n_max + 1), rank, world, interleave)
        assert np.array_equal(got, brute), (world, interleave, rank)


@pytest.mark.parametrize("world", [2, 4])
def test_prefix_count_never_exceeds_the_padded_local_length(world):
    """The scorer must stay inside the region cp_gather_indexer_k_quant_cache read."""
    interleave = 1
    for g in range(1, 200):
        lpad = -(-g // (interleave * world)) * interleave
        for rank in range(world):
            assert dcp_local_prefix_count(g, rank, world, interleave) <= lpad


def _brute_force_window(g_lens, q_counts, rank, world, interleave):
    """Reference: per query token, (local region base, base + owned-in-window)."""
    cu_pad, acc = [], 0
    for g in g_lens:
        cu_pad.append(acc)
        acc += -(-int(g) // (interleave * world)) * interleave
    ks, ke = [], []
    for b, (g, q) in enumerate(zip(g_lens, q_counts)):
        cached = int(g) - int(q)
        for t in range(int(q)):
            p = cached + t  # this token's position within its sequence
            owned = sum(
                1 for j in range(p + 1) if dcp_owner_rank(j, world, interleave) == rank
            )
            ks.append(cu_pad[b])
            ke.append(cu_pad[b] + owned)
    return np.array(ks, dtype=np.int64), np.array(ke, dtype=np.int64)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 4])
def test_prefill_local_window_matches_brute_force(world, interleave):
    """Chunked prefill: cached prefix plus this chunk's new tokens, 2 requests."""
    g_lens = np.array([37, 64], dtype=np.int64)  # full context per request
    q_counts = np.array([5, 64], dtype=np.int64)  # req 0 is a continuation chunk
    cu_pad = np.zeros(len(g_lens) + 1, dtype=np.int64)
    np.cumsum(
        ((g_lens + interleave * world - 1) // (interleave * world)) * interleave,
        out=cu_pad[1:],
    )
    ks, ke = dcp_prefill_local_window(
        cu_pad, g_lens, q_counts, world - 1, world, interleave
    )
    want_ks, want_ke = _brute_force_window(
        g_lens, q_counts, world - 1, world, interleave
    )
    assert np.array_equal(ks, want_ks)
    assert np.array_equal(ke, want_ke)


def test_prefill_local_window_handles_an_empty_batch():
    cu_pad = np.zeros(1, dtype=np.int64)
    empty = np.zeros(0, dtype=np.int64)
    ks, ke = dcp_prefill_local_window(cu_pad, empty, empty, 0, 4, 1)
    assert ks.shape == (0,) and ke.shape == (0,)
