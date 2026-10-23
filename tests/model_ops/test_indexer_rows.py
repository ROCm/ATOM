# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The shared indexer's step rows (`atom.model_ops.indexer.rows`): how a step's
query rows are cut into bands, grouped by request, and split across DCP ranks.
Host arithmetic only."""

import numpy as np
import pytest
import torch

rows_module = pytest.importorskip(
    "atom.model_ops.indexer.rows",
    reason="the indexer's rows import aiter at load",
    exc_type=ImportError,
)

from atom.distributed.pcp_utils import pcp_owned_rows_by_request
from atom.model_ops import sparse_indexer_chunk as chunk
from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata
from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace


def _covers(bands, first, end):
    """`bands` tile `[first, end)` in order, none empty."""
    starts = [b for b, _ in bands]
    stops = [e for _, e in bands]
    return (
        (not bands or (starts[0] == first and stops[-1] == end))
        and starts[1:] == stops[:-1]
        and all(b < e for b, e in bands)
    )


@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
def test_request_rows_band_in_whole_requests(monkeypatch, next_n):
    """A request's `next_n` rows read its keys together, so no band may cut
    one: the bands tile the rows in whole requests whatever the budget."""
    monkeypatch.setattr(chunk, "plane_rows", lambda width: 7)
    requests = 9
    rows = rows_module.fp8_rows(
        torch.zeros(requests, dtype=torch.int32),
        4096,
        None,
        torch.zeros(requests, 4, dtype=torch.int32),
        next_n=next_n,
    )
    assert _covers(rows.bands, 0, requests * next_n)
    assert all(b % next_n == 0 and e % next_n == 0 for b, e in rows.bands)
    assert len(rows.bands) > 1
    assert rows.out_rows == requests * next_n


def test_rows_band_within_the_plane_they_are_cut_to():
    """Bands follow the workspace's plane, and the rows carry it, so the
    scorer reads the same plane the bands were cut to."""
    plane = LogitsWorkspace(3 * 1024, "cpu")
    rows = rows_module.fp8_rows(
        torch.zeros(10, dtype=torch.int32),
        1024,
        plane,
        torch.zeros(10, 4, dtype=torch.int32),
    )
    assert _covers(rows.bands, 0, 10)
    assert max(e - b for b, e in rows.bands) == 3
    assert rows.workspace is plane


def test_recorded_fp4_bands_keep_their_launch_arguments_in_place():
    """A captured graph replays the addresses it recorded: a recorded decode's
    band starts are rewritten in their homes step after step, not allocated."""
    plane = LogitsWorkspace(3 * 1024, "cpu")
    storage = rows_module.Fp4LaunchStorage("cpu", 10, 10)
    sequences = Fp4MqaRaggedMetadata(
        torch.arange(11, dtype=torch.int32), 1, torch.zeros(10, 4, dtype=torch.int32)
    )

    def band_starts():
        rows = rows_module.fp4_rows(
            torch.zeros(10, dtype=torch.int32),
            1024,
            plane,
            sequences,
            heads=32,
            page_size=64,
            storage=storage,
        )
        return [launch["query_start_loc"] for launch in rows.fp4_bands]

    first, second = band_starts(), band_starts()
    assert len(first) > 1
    assert [s.data_ptr() for s in first] == [s.data_ptr() for s in second]
    assert second[1].tolist() == [0] * 4 + [1, 2, 3] + [3] * 4


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("count", [1, 7, 12, 4097])
def test_dcp_split_bands_only_this_rank_s_rows(world, count):
    """Under DCP a rank scores its stride of the rows alone, and its slot of
    the gathered selection is a whole stride whether or not it has rows."""
    keys = rows_module.ConcatKeys(None, None, 1 << 20, None)
    covered = []
    for rank in range(world):
        split = rows_module.DcpSplit.of(count, rank, world)
        rows = rows_module.concat_rows(
            torch.zeros(count, dtype=torch.int32), keys, split
        )
        assert _covers(rows.bands, split.first, split.end)
        assert rows.out_rows == split.stride * world >= count
        assert split.slot == rank * split.stride
        covered += [r for b, e in rows.bands for r in range(b, e)]
    assert covered == list(range(count))
    # Without DCP, one rank's split: every row, nothing to gather.
    assert rows_module.DcpSplit.of(count) == (0, count, 0, count, 1)


def test_pcp_owned_rows_group_by_request_and_pad():
    """A PCP rank owns rows `first, first + stride, ...`; grouped by request
    from the host's `cu_seqlens_q`, the padding rows past the last request one
    group more."""
    cu = np.array([0, 3, 3, 8], dtype=np.int32)
    # Rank 1 of 2 owns rows 1, 3, 5, 7 of 8, then one padding row.
    starts = pcp_owned_rows_by_request(cu, 1, 2, 5)
    assert starts.tolist() == [0, 1, 1, 4, 5]
    assert pcp_owned_rows_by_request(cu, 0, 2, 4).tolist() == [0, 2, 2, 4]
