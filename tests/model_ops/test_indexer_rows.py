# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The shared indexer's step rows (`atom.model_ops.indexer.rows`): how a step's
query rows are cut into bands, grouped by request, and split across ranks.
Host arithmetic only."""

import itertools

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
def test_row_split_bands_only_this_rank_s_rows(world, count):
    """Split across ranks, a rank scores its stride of the rows alone, and its
    slot of the gathered selection is a whole stride whether or not it has
    rows."""
    keys = rows_module.ConcatKeys(None, None, 1 << 20, None)
    covered = []
    for rank in range(world):
        split = rows_module.RowSplit.of(count, rank, world)
        rows = rows_module.concat_rows(
            torch.zeros(count, dtype=torch.int32), keys, split
        )
        assert _covers(rows.bands, split.first, split.end)
        assert rows.out_rows == split.stride * world >= count
        assert split.slot == rank * split.stride
        covered += [r for b, e in rows.bands for r in range(b, e)]
    assert covered == list(range(count))
    # Without DCP, one rank's split: every row, nothing to gather.
    assert rows_module.RowSplit.of(count) == (0, count, 0, count, 1, "dcp")


def test_pcp_owned_rows_group_by_request_and_pad():
    """A PCP rank owns rows `first, first + stride, ...`; grouped by request
    from the host's `cu_seqlens_q`, the padding rows past the last request in
    its group. One group per request: the FP4 scorer reads group b's keys off
    `block_tables[b]`, a table with a row per request."""
    cu = np.array([0, 3, 3, 7], dtype=np.int32)
    # 7 tokens pad to 8: rank 1 of 2 owns rows 1, 3, 5 and padding row 7.
    assert pcp_owned_rows_by_request(cu, 1, 2, 4).tolist() == [0, 1, 1, 4]
    assert pcp_owned_rows_by_request(cu, 0, 2, 4).tolist() == [0, 2, 2, 4]
    # 6 tokens pad to 8: rank 3 of 4 owns row 3 and padding row 7, so the last
    # request's group holds padding only.
    cu = np.array([0, 5, 6], dtype=np.int32)
    assert pcp_owned_rows_by_request(cu, 3, 4, 2).tolist() == [0, 1, 2]


def _owned_rows_reference(lens, world, rank):
    """Brute force: walk the rows `rank` owns, padding rows to the last request."""
    total = sum(lens)
    padded = -(-total // world) * world
    bounds = np.cumsum([0, *lens])
    counts = [0] * len(lens)
    for row in range(rank, padded, world):
        if row >= total:
            counts[-1] += 1
        else:
            counts[int(np.searchsorted(bounds, row, side="right")) - 1] += 1
    return np.concatenate([[0], np.cumsum(counts)]).tolist()


def test_pcp_owned_rows_match_brute_force():
    for world in (2, 3, 4, 8):
        for lens in itertools.product((1, 2, 3, 5), repeat=3):
            cu = np.cumsum([0, *lens]).astype(np.int32)
            owned = -(-sum(lens) // world)
            for rank in range(world):
                got = pcp_owned_rows_by_request(cu, rank, world, owned)
                assert got.tolist() == _owned_rows_reference(lens, world, rank), (
                    world,
                    lens,
                    rank,
                )


# ---- The split's gather: which group, which communicator, which slot ---------

score_module = pytest.importorskip(
    "atom.model_ops.indexer.score",
    reason="the indexer's scorer imports aiter at load",
    exc_type=ImportError,
)


class _Pynccl:
    def __init__(self, disabled=False):
        self.disabled = disabled
        self.calls = []

    def all_gather(self, output, input_):
        self.calls.append((output, input_))


class _Group:
    def __init__(self, name, pynccl):
        self.device_group = name
        self.device_communicator = type("Comm", (), {"pynccl_comm": pynccl})()


def _groups(monkeypatch, tp_pynccl, dcp_pynccl):
    """Fake TP and DCP groups for the scorer, and a record of the torch
    ProcessGroup gathers it falls back to."""
    tp, dcp = _Group("tp", tp_pynccl), _Group("dcp", dcp_pynccl)
    monkeypatch.setattr(score_module, "get_tp_group", lambda: tp)
    monkeypatch.setattr(score_module, "get_dcp_group", lambda: dcp)
    torch_calls = []
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        lambda out, inp, group: torch_calls.append((out, inp, group)),
    )
    return torch_calls


@pytest.mark.parametrize("group", ["dcp", "tp"])
@pytest.mark.parametrize("count, rank, world", [(12, 7, 8), (13, 2, 4), (1, 1, 2)])
def test_split_gathers_its_own_slot_on_its_group_s_pynccl(
    monkeypatch, group, count, rank, world
):
    """The gather runs on the split's group's pynccl communicator (built with
    the group, so counted in the memory profile; the torch ProcessGroup's is
    created at the first prefill), in place: the input is this rank's whole
    slot of the output, an empty rank's (12 rows over 8, rank 7) included."""
    tp, dcp = _Pynccl(), _Pynccl()
    torch_calls = _groups(monkeypatch, tp, dcp)
    split = rows_module.RowSplit.of(count, rank, world, group=group)
    out = torch.zeros(split.stride * world + 3, 4, dtype=torch.int32)
    score_module._gather_split(out, split)
    used, idle = (tp, dcp) if group == "tp" else (dcp, tp)
    assert torch_calls == [] and idle.calls == []
    [(region, mine)] = used.calls
    assert region.data_ptr() == out.data_ptr()
    assert region.shape[0] == split.stride * world
    assert mine.data_ptr() == out[split.slot].data_ptr()
    assert mine.shape[0] == split.stride


@pytest.mark.parametrize("pynccl", [None, _Pynccl(disabled=True)])
def test_split_without_pynccl_gathers_on_the_process_group(monkeypatch, pynccl):
    """No usable pynccl on the DCP group: the gather falls back to its torch
    ProcessGroup, as before. (The TP split is not turned on without one.)"""
    torch_calls = _groups(monkeypatch, _Pynccl(), pynccl)
    split = rows_module.RowSplit.of(9, 1, 4)
    out = torch.zeros(split.stride * 4, 4, dtype=torch.int32)
    score_module._gather_split(out, split)
    [(region, mine, group)] = torch_calls
    assert group == "dcp" and region.shape[0] == split.stride * 4
    assert mine.data_ptr() == out[split.slot].data_ptr()


def test_unsplit_rows_gather_nothing(monkeypatch):
    tp, dcp = _Pynccl(), _Pynccl()
    torch_calls = _groups(monkeypatch, tp, dcp)
    score_module._gather_split(torch.zeros(5, 4), rows_module.RowSplit.of(5))
    assert torch_calls == [] and tp.calls == [] and dcp.calls == []


def test_row_split_refuses_an_unknown_group():
    """Anything but "dcp" / "tp" would otherwise gather on the DCP group."""
    with pytest.raises(ValueError):
        rows_module.RowSplit.of(8, 0, 2, group="TP")


@pytest.mark.parametrize(
    "usable_on_ranks, expected",
    [((1, 1), True), ((0, 0), False), ((1, 0), RuntimeError), ((0, 1), RuntimeError)],
)
def test_ranks_must_agree_on_the_gather_transport(
    monkeypatch, usable_on_ranks, expected
):
    """pynccl is built per rank and only warns on failure: ranks that disagree
    would gather over different transports (or not split at all) and hang, so
    the startup check raises instead. The fake all_reduce folds in the other
    rank's vote the way MIN over the CPU group would."""
    mine, other = usable_on_ranks
    group = _Group("tp", _Pynccl() if mine else None)
    group.cpu_group = "cpu"

    def all_reduce(votes, op, group):
        assert group == "cpu"
        votes.copy_(torch.minimum(votes, torch.tensor([other, -other])))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    if expected is RuntimeError:
        with pytest.raises(RuntimeError):
            score_module.split_gather_comm_on_every_rank(group)
    else:
        assert score_module.split_gather_comm_on_every_rank(group) is expected
