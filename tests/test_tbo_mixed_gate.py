# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""With TBO on, a mixed batch takes the mixed split -- or refuses through `can_split`.

`can_split` is AND-reduced across DP ranks so that one rank's inability to
split turns TBO off for the whole group. A rank that reported can_split=True
and then declined to build slices would run one ubatch while its peers ran two,
and the per-ubatch collectives would deadlock. So whatever the mixed split
cannot handle -- a cut that would leave a pure-decode ubatch -- must surface
here as can_split=False.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from atom.utils.tbo.ubatching import local_tbo_precompute


@pytest.fixture(autouse=True)
def _min_tokens(monkeypatch):
    """Pin the min-token bar at its default so `_PREFILL_TOKENS` clears it."""
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", "8192")


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
        total_tokens_num_prefill=0,
        is_dummy_run=False,
    ):
        self.is_mixed = is_mixed
        self.total_seqs_num_prefill = total_seqs_num_prefill
        self.total_seqs_num_decode = total_seqs_num_decode
        self.total_seqs_num = total_seqs_num_prefill + total_seqs_num_decode
        self.total_tokens_num = total_tokens_num
        self.total_tokens_num_prefill = total_tokens_num_prefill
        self.is_dummy_run = is_dummy_run


# Well above ATOM_TBO_PREFILL_MIN_TOKENS (8192), so the min-token bar is never
# what decides these cases.
_PREFILL_TOKENS = [8192, 8192]


def test_pure_prefill_can_split():
    meets_min, can_split, ub0, ub1 = local_tbo_precompute(
        _Config(),
        _Batch(total_seqs_num_prefill=2),
        True,
        np.asarray(_PREFILL_TOKENS, dtype=np.int64),
    )
    assert can_split is True and meets_min is True
    assert ub0 > 0 and ub1 > 0


def test_mixed_batch_takes_the_mixed_split():
    """The cut lands inside the prefill rows: two prefill-mode ubatches whose
    sizes cover the whole batch, decode rows included."""
    toks = _PREFILL_TOKENS + [1, 1, 1, 1]
    meets_min, can_split, ub0, ub1 = local_tbo_precompute(
        _Config(),
        _Batch(
            is_mixed=True,
            total_seqs_num_prefill=2,
            total_seqs_num_decode=4,
            total_tokens_num=sum(toks),
            total_tokens_num_prefill=sum(_PREFILL_TOKENS),
        ),
        True,
        np.asarray(toks, dtype=np.int64),
    )
    assert can_split is True and meets_min is True
    assert ub0 > 0 and ub1 > 0 and ub0 + ub1 == sum(toks)
    assert ub0 < sum(_PREFILL_TOKENS)


def test_decode_heavy_mixed_batch_refuses():
    """A midpoint at or past the prefill tokens would leave ubatch 1 pure
    decode under a prefill-mode parent: refused as can_split=False."""
    toks = [16] + [8] * 16  # 16 prefill tokens, 128 decode tokens
    _, can_split, ub0, ub1 = local_tbo_precompute(
        _Config(),
        _Batch(
            is_mixed=True,
            total_seqs_num_prefill=1,
            total_seqs_num_decode=16,
            total_tokens_num=sum(toks),
            total_tokens_num_prefill=16,
        ),
        True,
        np.asarray(toks, dtype=np.int64),
    )
    assert can_split is False
    assert (ub0, ub1) == (0, 0)


@pytest.mark.parametrize("is_prefill", [True, False])
def test_tbo_off_reports_no_split(is_prefill):
    """With TBO off nothing splits, mixed or not."""
    _, can_split, _, _ = local_tbo_precompute(
        _Config(enable_tbo=False),
        _Batch(
            is_mixed=True,
            total_seqs_num_prefill=2,
            total_seqs_num_decode=2,
            total_tokens_num=sum(_PREFILL_TOKENS) + 2,
            total_tokens_num_prefill=sum(_PREFILL_TOKENS),
        ),
        is_prefill,
        np.asarray(_PREFILL_TOKENS + [1, 1], dtype=np.int64),
    )
    assert can_split is False


@pytest.mark.parametrize("pcp_size,is_mixed", [(1, True), (2, False), (2, True)])
def test_runner_rejects_mixed_pcp_before_collective_sizing(pcp_size, is_mixed):
    """The runtime backstop also protects callers that bypass Config validation."""
    from atom.model_engine.model_runner import ModelRunner

    config = _Config()
    config.prefill_context_parallel_size = pcp_size
    runner = SimpleNamespace(config=config)
    tokens = _PREFILL_TOKENS + ([1, 1] if is_mixed else [])
    batch = _Batch(
        is_mixed=is_mixed,
        total_seqs_num_prefill=2,
        total_seqs_num_decode=2 if is_mixed else 0,
        total_tokens_num=sum(tokens),
        total_tokens_num_prefill=sum(_PREFILL_TOKENS),
    )
    batch.num_scheduled_tokens = np.asarray(tokens, dtype=np.int64)
    if pcp_size > 1 and is_mixed:
        with pytest.raises(NotImplementedError, match="mixed prefill.*PCP"):
            ModelRunner._local_tbo_eligibility(runner, batch)
    else:
        meets_min, can_split, ub0, ub1 = ModelRunner._local_tbo_eligibility(
            runner, batch
        )
        assert meets_min and can_split
        assert ub0 + ub1 == sum(tokens) // pcp_size
