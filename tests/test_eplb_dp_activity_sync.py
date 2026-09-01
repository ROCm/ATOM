# SPDX-License-Identifier: MIT

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    from atom.utils.tbo.ubatching import sync_dp_metadata
except ImportError as _e:
    skip_if_dependency_missing(_e, "requires full atom import env")


def _mock_all_gather(monkeypatch, rows):
    def _all_gather(outputs, local, group=None):
        _ = group
        assert local.numel() == len(rows[0])
        for output, row in zip(outputs, rows):
            output.copy_(torch.tensor(row, dtype=torch.int32))

    monkeypatch.setattr(torch.distributed, "all_gather", _all_gather)


def test_dp_sync_distinguishes_real_decode_from_dummy(monkeypatch):
    # [scheduled_tokens, scheduled_bs, is_prefill, is_dummy]
    _mock_all_gather(
        monkeypatch,
        [
            [10, 2, 1, 0],  # real prefill
            [8, 4, 0, 0],  # real decode
            [1, 1, 0, 1],  # synchronization-only dummy decode
        ],
    )

    result = sync_dp_metadata(
        dp_group=object(),
        dp_size=3,
        scheduled_tokens=10,
        scheduled_bs=2,
        is_prefill=True,
        tbo_on=False,
    )

    assert result.num_tokens_across_dp.tolist() == [10, 8, 1]
    assert result.max_bs_across_dp == 4
    assert result.any_rank_has_prefill
    assert result.any_rank_has_real_prefill
    assert result.any_rank_has_decode


def test_dp_sync_preserves_tbo_and_dspark_field_offsets(monkeypatch):
    # Four base fields, four TBO fields, then max_seqlen_q.
    _mock_all_gather(
        monkeypatch,
        [
            [10, 3, 1, 0, 1, 1, 4, 6, 2],
            [8, 5, 0, 1, 0, 0, 0, 0, 4],
        ],
    )

    result = sync_dp_metadata(
        dp_group=object(),
        dp_size=2,
        scheduled_tokens=10,
        scheduled_bs=3,
        is_prefill=True,
        tbo_on=True,
        local_meets_min_tokens=True,
        local_can_split=True,
        local_ub_tokens=(4, 6),
        max_seqlen_q=2,
    )

    assert not result.tbo_collective_active
    assert result.ub_max_tokens_across_dp is None
    assert result.max_bs_across_dp == 5
    assert result.max_seqlen_q_across_dp == 4
    assert result.any_rank_has_real_prefill
    assert not result.any_rank_has_decode
