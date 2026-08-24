# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Token-midpoint splitting of a mixed ``[prefill | decode]`` batch.

Three properties, none of them obvious from either function alone:

1. `_precompute_mixed_token_split` and `split_mixed_token_midpoint` must agree
   on the ub0/ub1 token counts. The former's numbers are MAX-reduced across DP
   to size the per-ubatch buffers; if the slices the latter builds are a
   different size, the all_gather mismatches and the group hangs.

2. A batch is only splittable when the cut lands strictly inside the prefill
   region. One contiguous cut of a `[prefill | decode]` layout can never leave
   both halves mixed -- whichever side the cut falls on, the other is pure --
   and a pure-DECODE ubatch under a parent context that says is_prefill=True
   is something TBO cannot express. So the cut is required to fall in the
   prefill region, making ubatch 0 pure prefill and ubatch 1 the only mixed
   one. Decode-heavy batches are refused outright.

3. The straddling ubatch must carry its own prefill boundary; nothing
   downstream can rederive it.

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


# Prefill-dominated shapes -- the cut lands inside the prefill region, so these
# split. This is what real mixed batches look like: measured 15831 prefill
# against 379 decode tokens.
_SPLITTABLE = [
    (4, 4096, 8),  # cut deep inside the prefill region
    (1, 8192, 400),  # one long prefill, many decode rows
    (3, 777, 5),  # odd token counts
    (16, 870, 64),  # the gsm8k-like shape: many short prefills
    (2, 100, 99),  # only just prefill-dominated (cut at 99 < 200)
]

# Decode at or over half the batch -- the cut would land on or past the
# boundary, leaving a pure-decode ubatch. Refused.
_REFUSED = [
    (2, 100, 200),  # 200 prefill, 200 decode: cut exactly on the boundary
    (1, 200, 200),  # same, single prefill request
    (1, 100, 10000),  # decode dwarfs prefill
]


@pytest.mark.parametrize("n_pref_seqs,pref_len,n_dec_seqs", _SPLITTABLE)
def test_precompute_and_slices_agree(n_pref_seqs, pref_len, n_dec_seqs):
    """Property 1: the reported ub0/ub1 are the sizes actually produced."""
    toks, n_reqs, n_ps, n_pt = _mixed(n_pref_seqs, pref_len, n_dec_seqs)
    _, can_split, ub0, ub1 = _precompute_mixed_token_split(toks, n_reqs, n_pt, 0)
    assert can_split

    slices = split_mixed_token_midpoint(n_reqs, n_ps, n_pt, toks)
    assert slices is not None and len(slices) == 2
    got = [s.token_slice.stop - s.token_slice.start for s in slices]
    assert got == [ub0, ub1], f"precompute said {[ub0, ub1]}, slices are {got}"


@pytest.mark.parametrize("n_pref_seqs,pref_len,n_dec_seqs", _REFUSED)
def test_decode_heavy_is_refused_by_both(n_pref_seqs, pref_len, n_dec_seqs):
    """Property 2, and both sides must refuse in step.

    If only the splitter refused, `can_split` would still be True, the cross-DP
    AND-reduce would keep TBO on, and this rank would run one ubatch against
    its peers' two.
    """
    toks, n_reqs, n_ps, n_pt = _mixed(n_pref_seqs, pref_len, n_dec_seqs)
    _, can_split, ub0, ub1 = _precompute_mixed_token_split(toks, n_reqs, n_pt, 0)
    assert can_split is False
    assert (ub0, ub1) == (0, 0)
    assert split_mixed_token_midpoint(n_reqs, n_ps, n_pt, toks) is None


@pytest.mark.parametrize("n_pref_seqs,pref_len,n_dec_seqs", _SPLITTABLE)
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


@pytest.mark.parametrize("n_pref_seqs,pref_len,n_dec_seqs", _SPLITTABLE)
def test_ubatch0_pure_prefill_ubatch1_mixed(n_pref_seqs, pref_len, n_dec_seqs):
    """Properties 2 and 3: the shape the rest of the design relies on.

    ubatch 0 holds nothing but prefill (so it takes the ordinary prefill path),
    ubatch 1 holds the prefill tail plus every decode row, and its recorded
    boundary matches the definition -- its intersection with the batch's
    leading prefill range.
    """
    toks, n_reqs, n_ps, n_pt = _mixed(n_pref_seqs, pref_len, n_dec_seqs)
    ub0, ub1 = split_mixed_token_midpoint(n_reqs, n_ps, n_pt, toks)

    ub0_len = ub0.token_slice.stop - ub0.token_slice.start
    ub1_len = ub1.token_slice.stop - ub1.token_slice.start

    # ubatch 0: pure prefill.
    assert ub0.num_prefill_tokens == ub0_len

    # ubatch 1: straddles, so strictly between empty and full.
    assert 0 < ub1.num_prefill_tokens < ub1_len

    # Both against the definition, not the implementation.
    for s in (ub0, ub1):
        ts, rs = s.token_slice, s.request_slice
        assert s.num_prefill_tokens == max(0, min(ts.stop, n_pt) - ts.start)
        assert s.num_prefill_seqs == max(0, min(rs.stop, n_ps) - rs.start)

    # Every prefill token lands in exactly one ubatch.
    assert ub0.num_prefill_tokens + ub1.num_prefill_tokens == n_pt


def test_too_small_to_split():
    toks = np.asarray([1], dtype=np.int64)
    assert split_mixed_token_midpoint(1, 0, 0, toks) is None
    _, can_split, _, _ = _precompute_mixed_token_split(toks, 1, 0, 0)
    assert can_split is False


def test_min_token_bar_keys_off_prefill_only():
    """A decode tail must not talk a small prefill past the min-token bar.

    Prefill-dominated (so the batch IS splittable) but far below the 8192 bar:
    can_split says yes, meets_min_tokens says not worth it.
    """
    toks, n_reqs, _, n_pt = _mixed(1, 100, 50)  # 100 prefill, 50 decode
    meets_min, can_split, _, _ = _precompute_mixed_token_split(toks, n_reqs, n_pt, 8192)
    assert can_split is True
    assert meets_min is False


def test_maybe_create_ubatch_slices_dispatches_to_mixed():
    """The factory must route a mixed batch to the mixed splitter.

    Regression: the mixed splitter existed and was unit-tested while nothing
    called it -- `maybe_create_ubatch_slices` still took the prefill path,
    which counts only `num_scheduled_tokens[:num_prefill_seqs]` and would have
    produced slices that do not cover the decode rows.
    """
    from atom.utils.tbo.ubatch_splitting import maybe_create_ubatch_slices

    # The shape the first end-to-end attempt actually produced.
    toks = np.asarray([605] * 27 + [1], dtype=np.int64)
    total = int(toks.sum())
    slices = maybe_create_ubatch_slices(
        num_reqs=28,
        num_tokens=total,
        is_prefill=True,
        num_scheduled_tokens=toks,
        force=True,
        num_prefill_seqs=27,
        num_prefill_tokens=27 * 605,
    )
    assert slices is not None and len(slices) == 2
    # Mixed splitter ran: only it fills these in.
    assert all(s.num_prefill_tokens is not None for s in slices)
    # And the slices span the whole batch, decode rows included.
    assert slices[0].token_slice.start == 0
    assert slices[-1].token_slice.stop == total


def test_maybe_create_ubatch_slices_leaves_prefill_path_alone():
    """Without the mixed args the factory must behave exactly as before."""
    from atom.utils.tbo.ubatch_splitting import maybe_create_ubatch_slices

    toks = np.asarray([4096] * 4, dtype=np.int64)
    slices = maybe_create_ubatch_slices(
        num_reqs=4,
        num_tokens=int(toks.sum()),
        is_prefill=True,
        num_scheduled_tokens=toks,
        force=True,
    )
    assert slices is not None
    assert all(s.num_prefill_tokens is None for s in slices)
