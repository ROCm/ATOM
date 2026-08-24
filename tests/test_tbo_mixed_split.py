# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Token-midpoint splitting of a mixed ``[prefill | decode]`` batch.

Two properties matter and neither is obvious from reading either function
alone:

1. `_precompute_mixed_token_split` and `split_mixed_token_midpoint` must agree
   on the ub0/ub1 token counts. The former's numbers are MAX-reduced across DP
   to size the per-ubatch buffers; if the slices the latter builds are a
   different size, the all_gather mismatches and the group hangs.

2. Each slice must carry its OWN prefill/decode boundary. The cut lands
   wherever the token midpoint falls, so the two ubatches get different mixes,
   and nothing downstream can rederive the boundary from the parent batch.

Everything here is pure arithmetic -- no GPU, no model.
"""

import numpy as np
import pytest

from atom.utils.tbo.ubatch_splitting import split_mixed_token_midpoint
from atom.utils.tbo.ubatching import _precompute_mixed_token_split


def _mixed(n_pref_seqs, pref_len, n_dec_seqs, dec_len=1):
    """Build (num_scheduled_tokens, num_reqs, n_pref_seqs, n_pref_tokens)."""
    toks = [pref_len] * n_pref_seqs + [dec_len] * n_dec_seqs
    return (
        np.asarray(toks, dtype=np.int64),
        n_pref_seqs + n_dec_seqs,
        n_pref_seqs,
        pref_len * n_pref_seqs,
    )


# (n_pref_seqs, pref_len, n_dec_seqs) covering: cut inside prefill, cut inside
# decode, cut exactly on the boundary, single prefill request, odd totals.
_SHAPES = [
    (4, 4096, 8),  # cut lands deep inside the prefill region
    (1, 8192, 400),  # one long prefill, many decode rows
    (2, 100, 200),  # decode dominates -> cut lands inside decode
    (1, 200, 200),  # cut lands exactly on the prefill/decode boundary
    (3, 777, 5),  # odd token counts
    (16, 870, 64),  # the gsm8k-like shape: many short prefills
]


@pytest.mark.parametrize("n_pref_seqs,pref_len,n_dec_seqs", _SHAPES)
def test_precompute_and_slices_agree(n_pref_seqs, pref_len, n_dec_seqs):
    """Property 1: the reported ub0/ub1 are the sizes actually produced."""
    toks, n_reqs, n_ps, n_pt = _mixed(n_pref_seqs, pref_len, n_dec_seqs)
    _, can_split, ub0, ub1 = _precompute_mixed_token_split(toks, n_reqs, n_pt, 0)
    assert can_split

    slices = split_mixed_token_midpoint(n_reqs, n_ps, n_pt, toks)
    assert slices is not None and len(slices) == 2
    got = [s.token_slice.stop - s.token_slice.start for s in slices]
    assert got == [ub0, ub1], f"precompute said {[ub0, ub1]}, slices are {got}"


@pytest.mark.parametrize("n_pref_seqs,pref_len,n_dec_seqs", _SHAPES)
def test_slices_tile_the_batch(n_pref_seqs, pref_len, n_dec_seqs):
    """No token may be dropped or forwarded twice."""
    toks, n_reqs, n_ps, n_pt = _mixed(n_pref_seqs, pref_len, n_dec_seqs)
    slices = split_mixed_token_midpoint(n_reqs, n_ps, n_pt, toks)
    total = int(toks.sum())

    assert slices[0].token_slice.start == 0
    assert slices[-1].token_slice.stop == total
    for a, b in zip(slices, slices[1:]):
        assert a.token_slice.stop == b.token_slice.start
    for s in slices:
        assert s.token_slice.stop > s.token_slice.start, "empty ubatch"


@pytest.mark.parametrize("n_pref_seqs,pref_len,n_dec_seqs", _SHAPES)
def test_per_ubatch_prefill_boundary(n_pref_seqs, pref_len, n_dec_seqs):
    """Property 2: each slice's own prefill token/seq counts are right.

    Checked against the definition rather than the implementation: the prefill
    part of a slice is its intersection with the batch's leading prefill range.
    """
    toks, n_reqs, n_ps, n_pt = _mixed(n_pref_seqs, pref_len, n_dec_seqs)
    slices = split_mixed_token_midpoint(n_reqs, n_ps, n_pt, toks)

    seen_pref_tokens = 0
    for s in slices:
        ts, rs = s.token_slice, s.request_slice
        expect_tok = max(0, min(ts.stop, n_pt) - ts.start)
        expect_seq = max(0, min(rs.stop, n_ps) - rs.start)
        assert s.num_prefill_tokens == expect_tok
        assert s.num_prefill_seqs == expect_seq
        # A ubatch's prefill rows are always its LEADING rows: it can never
        # hold decode tokens before prefill ones.
        assert s.num_prefill_tokens <= ts.stop - ts.start
        seen_pref_tokens += s.num_prefill_tokens
    # Every prefill token lands in exactly one ubatch.
    assert seen_pref_tokens == n_pt


def test_cut_inside_prefill_gives_pure_prefill_ubatch0():
    """Prefill-dominated batch: ubatch 0 is all prefill, ubatch 1 straddles."""
    toks, n_reqs, n_ps, n_pt = _mixed(4, 4096, 8)  # 16384 prefill, 8 decode
    ub0, ub1 = split_mixed_token_midpoint(n_reqs, n_ps, n_pt, toks)
    assert ub0.num_prefill_tokens == ub0.token_slice.stop - ub0.token_slice.start
    assert ub1.num_prefill_tokens < ub1.token_slice.stop - ub1.token_slice.start


def test_cut_inside_decode_gives_pure_decode_ubatch1():
    """Decode-dominated batch: ubatch 1 holds no prefill at all."""
    toks, n_reqs, n_ps, n_pt = _mixed(2, 100, 200)  # 200 prefill, 200 decode
    ub0, ub1 = split_mixed_token_midpoint(n_reqs, n_ps, n_pt, toks)
    assert ub1.num_prefill_tokens == 0
    assert ub1.num_prefill_seqs == 0
    assert ub0.num_prefill_tokens == n_pt


def test_too_small_to_split():
    toks = np.asarray([1], dtype=np.int64)
    assert split_mixed_token_midpoint(1, 0, 0, toks) is None
    _, can_split, _, _ = _precompute_mixed_token_split(toks, 1, 0, 0)
    assert can_split is False


def test_min_token_bar_keys_off_prefill_only():
    """A big decode tail must not talk a small prefill past the bar."""
    toks, n_reqs, _, n_pt = _mixed(1, 100, 10000)  # 100 prefill, 10000 decode
    meets_min, can_split, _, _ = _precompute_mixed_token_split(toks, n_reqs, n_pt, 8192)
    assert can_split is True  # structurally splittable
    assert meets_min is False  # but not worth it: only 100 prefill tokens
