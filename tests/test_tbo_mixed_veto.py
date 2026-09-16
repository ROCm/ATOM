# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A mixed prefill+decode batch must veto TBO through `can_split`.

`can_split` is AND-reduced across DP ranks so that one rank's structural
inability to split turns TBO off for the whole group. A rank that instead
reported can_split=True and then quietly declined to build slices would run one
ubatch while its peers ran two, and the per-ubatch collectives would deadlock.

Mixed batches cannot be split today (the split maths only sees the prefill rows
and `split_attn_metadata` has no branch for the nested per-segment metadata), so
they must surface that as can_split=False here rather than bailing out later in
`_maybe_create_tbo_slices`.
"""

import numpy as np
import pytest

from atom.utils.tbo.ubatching import local_tbo_precompute


class _Config:
    def __init__(self, enable_tbo=True, enable_tbo_decode=True):
        self.enable_tbo = enable_tbo
        self.enable_tbo_decode = enable_tbo_decode


class _Batch:
    """Only the attributes `local_tbo_precompute` reads."""

    def __init__(
        self,
        *,
        is_mixed=False,
        total_seqs_num_prefill=0,
        total_seqs_num_decode=0,
        total_tokens_num=0,
        is_dummy_run=False,
    ):
        self.is_mixed = is_mixed
        self.total_seqs_num_prefill = total_seqs_num_prefill
        self.total_seqs_num_decode = total_seqs_num_decode
        self.total_tokens_num = total_tokens_num
        self.is_dummy_run = is_dummy_run


# Well above ATOM_TBO_PREFILL_MIN_TOKENS (8192), so the min-token bar is never
# what decides these cases -- only can_split is under test.
_PREFILL_TOKENS = [8192, 8192]


def test_pure_prefill_can_split():
    """Baseline: a plain prefill batch big enough to split says so."""
    meets_min, can_split, ub0, ub1 = local_tbo_precompute(
        _Config(),
        _Batch(total_seqs_num_prefill=2),
        True,
        np.asarray(_PREFILL_TOKENS, dtype=np.int64),
    )
    assert can_split is True
    assert meets_min is True
    assert ub0 > 0 and ub1 > 0


def test_mixed_batch_vetoes_split():
    """The regression: a mixed batch must report can_split=False.

    Same prefill rows as the baseline above, plus decode rows. Without the veto
    the split maths happily returns can_split=True off the prefill rows alone,
    the cross-DP AND-reduce keeps TBO on, and the rank then declines to build
    slices -- one ubatch here, two on every peer.
    """
    meets_min, can_split, ub0, ub1 = local_tbo_precompute(
        _Config(),
        _Batch(is_mixed=True, total_seqs_num_prefill=2, total_seqs_num_decode=4),
        True,  # a mixed batch carries prefill rows, so callers pass is_prefill=True
        np.asarray(_PREFILL_TOKENS + [1, 1, 1, 1], dtype=np.int64),
    )
    assert can_split is False
    assert meets_min is False
    assert (ub0, ub1) == (0, 0)


def test_mixed_batch_vetoes_split_on_decode_path():
    """The veto must not depend on which path the caller would have taken."""
    _, can_split, ub0, ub1 = local_tbo_precompute(
        _Config(),
        _Batch(is_mixed=True, total_seqs_num_decode=8, total_tokens_num=8),
        False,
        np.ones(8, dtype=np.int64),
    )
    assert can_split is False
    assert (ub0, ub1) == (0, 0)


@pytest.mark.parametrize("is_prefill", [True, False])
def test_tbo_off_reports_no_split(is_prefill):
    """Sanity: with TBO off nothing splits, mixed or not."""
    _, can_split, _, _ = local_tbo_precompute(
        _Config(enable_tbo=False),
        _Batch(total_seqs_num_prefill=2, total_seqs_num_decode=2, total_tokens_num=2),
        is_prefill,
        np.asarray(_PREFILL_TOKENS, dtype=np.int64),
    )
    assert can_split is False
