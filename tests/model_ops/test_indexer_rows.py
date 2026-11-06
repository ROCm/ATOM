# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The shared indexer's step rows (`atom.model_ops.indexer.rows`): how a step's
query rows are cut into bands, grouped by request, and split across ranks.
Host arithmetic only."""

import itertools
import threading
from types import SimpleNamespace

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


def test_concat_rows_band_within_the_plane_at_its_row_pitch():
    """FP8 prefill rows cut to a workspace band at `concat_plane_width`, the
    pitch aiter pads a plane to, and carry the plane to the scorer."""
    keys = rows_module.ConcatKeys(None, None, 1000, None)
    pitch = chunk.concat_plane_width(1000)
    assert pitch == 1024
    plane = LogitsWorkspace(3 * pitch, "cpu")
    rows = rows_module.concat_rows(
        torch.zeros(10, dtype=torch.int32), keys, workspace=plane
    )
    assert _covers(rows.bands, 0, 10)
    assert max(e - b for b, e in rows.bands) == 3
    assert rows.workspace is plane
    # Without one, the logits budget alone bands them and nothing is carried.
    rows = rows_module.concat_rows(torch.zeros(10, dtype=torch.int32), keys)
    assert rows.bands == ((0, 10),) and rows.workspace is None


def test_concat_bands_score_into_the_plane(monkeypatch):
    """Every band's logits are a view of the workspace plane at the band's
    rows and the step's key count, rows `concat_plane_width` apart: nothing
    the size of the context is allocated per band."""
    total, count, pitch = 1000, 10, chunk.concat_plane_width(1000)
    plane = LogitsWorkspace(3 * pitch, "cpu")
    keys = rows_module.ConcatKeys(
        None, None, total, torch.zeros(count, dtype=torch.int32)
    )
    rows = rows_module.concat_rows(
        torch.full((count,), total, dtype=torch.int32), keys, workspace=plane
    )
    seen = []

    def fake_logits(**kwargs):
        logits = kwargs["logits"]
        seen.append(logits)
        return logits

    monkeypatch.setattr(score_module, "fp8_mqa_logits", fake_logits)
    monkeypatch.setattr(score_module, "top_k_per_row_prefill", lambda *a, **k: None)
    base = plane.logits(1, 1).data_ptr()
    score_module._score_concat(
        torch.zeros(count, 1, 1),
        torch.zeros(count, 1),
        None,
        rows,
        4,
        False,
        torch.zeros(count, 4, dtype=torch.int32),
        (torch.zeros(total, 1), torch.zeros(total, 1)),
        True,
    )
    assert [t.shape[0] for t in seen] == [e - b for b, e in rows.bands]
    assert all(t.shape[1] == total and t.stride() == (pitch, 1) for t in seen)
    assert all(t.data_ptr() == base for t in seen)


def _concat_groups_of(cu_keys, cu_rows, runs):
    return tuple(
        rows_module.ConcatGroup(
            first,
            end,
            int(cu_rows[first]),
            int(cu_rows[end]),
            int(cu_keys[first]),
            int(cu_keys[end] - cu_keys[first]),
        )
        for first, end in runs
    )


def test_pack_concat_groups_keeps_each_run_within_capacity():
    """Requests pack in order without exceeding the fixed capacity."""
    cu = np.array([0, 500, 700, 1200, 1250, 2000])
    runs = chunk.pack_concat_groups(cu, 1000)
    assert runs == [(0, 2), (2, 4), (4, 5)]
    assert all(cu[end] - cu[first] <= 1000 for first, end in runs)
    assert chunk.pack_concat_groups(np.array([0, 10]), 1000) == [(0, 1)]
    with pytest.raises(ValueError, match="request 0 has 2000 keys"):
        chunk.pack_concat_groups(np.array([0, 2000, 2001]), 1000)


@pytest.mark.parametrize("world", [1, 2, 8])
def test_concat_rows_band_each_group_against_its_own_keys(world):
    """A band never spans two groups, the bands cover this rank's rows in
    order, and each holds as many rows as the plane fits at its group's
    pitch."""
    cu_keys = np.array([0, 3000, 3100, 6100])
    cu_rows = np.array([0, 40, 50, 90])
    groups = _concat_groups_of(cu_keys, cu_rows, [(0, 1), (1, 3)])
    keys = rows_module.ConcatKeys(None, None, int(cu_keys[-1]), None, groups=groups)
    plane = LogitsWorkspace(16 * chunk.concat_plane_width(3100), "cpu")
    covered = []
    for rank in range(world):
        split = rows_module.RowSplit.of(90, rank, world)
        rows = rows_module.concat_rows(
            torch.zeros(90, dtype=torch.int32), keys, split, workspace=plane
        )
        assert _covers(rows.bands, split.first, split.end)
        for first, end in rows.bands:
            group = next(g for g in groups if g.row_first <= first < g.row_end)
            assert end <= group.row_end
            pitch = chunk.concat_plane_width(group.total)
            assert (end - first) * pitch <= plane.capacity
        covered += [r for b, e in rows.bands for r in range(b, e)]
    assert covered == list(range(90))


@pytest.mark.parametrize("request_local", [True, False])
@pytest.mark.parametrize("world", [1, 2])
def test_concat_groups_score_against_their_own_keys(monkeypatch, request_local, world):
    """Each group's keys are gathered once, on every rank whether or not it
    scores rows of the group (DCP's all-gather); its rows score against them
    with starts and ends rebased to the group, into the plane at the group's
    pitch, and the top-k's places come back in the step's run -- less the
    row's request start when request-local."""
    cu_keys = np.array([0, 300, 310, 610])
    cu_rows = np.array([0, 4, 5, 9])
    groups = _concat_groups_of(cu_keys, cu_rows, [(0, 1), (1, 3)])
    requests = np.repeat(np.arange(3), np.diff(cu_rows))
    row_starts = torch.tensor(cu_keys[requests], dtype=torch.int32)
    row_ends = torch.tensor([1, 2, 3, 4, 7, 1, 2, 3, 4], dtype=torch.int32)
    keys = rows_module.ConcatKeys(None, None, 610, row_starts, groups=groups)
    plane = LogitsWorkspace(8 * chunk.concat_plane_width(310), "cpu")

    def fake_logits(**kwargs):
        scored.append((kwargs["cu_starts"].tolist(), kwargs["cu_ends"].tolist()))
        logits = kwargs["logits"]
        assert logits.stride() == (chunk.concat_plane_width(logits.shape[1]), 1)
        return logits

    def fake_topk(scores, starts, ends, out, _, rows, *strides, k, stable):
        out[:, 0] = ends - 1  # each row's last visible key, in the group's run
        out[:, 1:] = -1

    monkeypatch.setattr(score_module, "fp8_mqa_logits", fake_logits)
    monkeypatch.setattr(score_module, "top_k_per_row_prefill", fake_topk)
    out = torch.zeros(9, 3, dtype=torch.int32)
    for rank in range(world):
        gathered, scored = [], []

        def gather(group, seen=gathered):
            seen.append(group.first)
            return torch.zeros(group.total, 1), torch.zeros(group.total, 1)

        split = rows_module.RowSplit.of(9, rank, world)
        rows = rows_module.concat_rows(row_ends, keys, split, workspace=plane)
        score_module._score_concat(
            torch.zeros(9, 1, 1),
            torch.zeros(9, 1),
            None,
            rows,
            3,
            False,
            out,
            gather,
            request_local,
        )
        assert gathered == [0, 1]
        for starts, stops in scored:
            assert all(0 <= a < b <= 310 for a, b in zip(starts, stops))
    expected = row_ends - 1 + (0 if request_local else row_starts)
    assert out[:, 0].tolist() == expected.tolist()
    assert (out[:, 1:] == -1).all()


def test_a_dcp_group_gathers_its_requests_off_the_step_tables(monkeypatch):
    """Under DCP a group's gather reads its requests' rows of the step's block
    tables (`ConcatKeys`), not the forward's metadata's, which a TBO ubatch
    cuts to its own requests."""
    from atom.model_ops.indexer import indexer_forward as forward

    seen = {}

    def fake_gather(kv_cache, k_fp8, k_scale, block_tables, cu_seqlens, preshuffle):
        seen["tables"] = block_tables.clone()

    monkeypatch.setattr(forward, "cp_gather_indexer_k_quant_cache", fake_gather)
    monkeypatch.setattr(
        forward,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0, device_group=None),
    )
    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", lambda out, inp, group: None
    )
    workspace = chunk.ConcatKeyWorkspace(64, 4, torch.uint8, "cpu", 64)
    step_tables = torch.arange(12).view(6, 2)
    keys = rows_module.ConcatKeys(step_tables, None, 30, None, workspace=workspace)
    plan = rows_module.DcpConcatGather(
        10, torch.tensor([0, 4, 10], dtype=torch.int32), torch.arange(10)
    )
    group = rows_module.ConcatGroup(2, 4, 0, 5, 12, 10, plan)
    ubatch = SimpleNamespace(block_tables=torch.full((3, 2), -1))
    values, scales = forward._dcp_gather_indexer_k_prefill(
        None, ubatch, 4, "cpu", keys, group
    )
    assert torch.equal(seen["tables"], step_tables[2:4])
    assert values.shape == (10, 4) and scales.shape == (10, 1)


def test_a_ubatch_keeps_the_groups_its_rows_fall_in():
    """A TBO ubatch reads its rows' starts and the groups its rows fall in,
    their rows rebased to the ubatch and their keys whole."""
    groups = _concat_groups_of(
        np.array([0, 300, 310, 610]), np.array([0, 4, 5, 9]), [(0, 1), (1, 3)]
    )
    keys = rows_module.ConcatKeys(None, None, 610, torch.arange(9), groups=groups)
    ubatch = rows_module.ubatch_concat_keys(keys, 2, 6)
    assert ubatch.row_starts.tolist() == [2, 3, 4, 5]
    assert [(g.first, g.row_first, g.row_end, g.total) for g in ubatch.groups] == [
        (0, 0, 2, 300),
        (1, 2, 4, 310),
    ]
    ubatch = rows_module.ubatch_concat_keys(keys, 5, 9)
    assert [(g.first, g.row_first, g.row_end) for g in ubatch.groups] == [(1, 0, 4)]
    assert (
        rows_module.ubatch_concat_keys(keys._replace(groups=None), 2, 6).groups is None
    )


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
    assert rows_module.RowSplit.of(count) == (0, count, 0, count, 1, "dcp", False)


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


# ---- The split's gather: which group, which slot, which order --------------

score_module = pytest.importorskip(
    "atom.model_ops.indexer.score",
    reason="the indexer's scorer imports aiter at load",
    exc_type=ImportError,
)


class _Group:
    def __init__(self, name):
        self.device_group = name


def _groups(monkeypatch):
    """Fake TP and DCP groups for the scorer, and a record of the gathers it
    runs on their ProcessGroups."""
    monkeypatch.setattr(score_module, "get_tp_group", lambda: _Group("tp"))
    monkeypatch.setattr(score_module, "get_dcp_group", lambda: _Group("dcp"))
    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        lambda out, inp, group: calls.append((out, inp, group)),
    )
    return calls


@pytest.mark.parametrize("group", ["dcp", "tp"])
@pytest.mark.parametrize("make", ["of", "striped"])
@pytest.mark.parametrize("count, rank, world", [(12, 7, 8), (13, 2, 4), (1, 1, 2)])
def test_split_gathers_its_own_slot_on_its_group(
    monkeypatch, group, make, count, rank, world
):
    """The gather runs on the split's group's ProcessGroup, in place: the
    input is this rank's whole slot of the output, an empty rank's (12 rows
    over 8, rank 7) included."""
    calls = _groups(monkeypatch)
    split = getattr(rows_module.RowSplit, make)(count, rank, world, group=group)
    out = torch.zeros(split.stride * world + 3, 4, dtype=torch.int32)
    score_module._gather_split(out, split)
    [(region, mine, used)] = calls
    assert used == group
    assert region.data_ptr() == out.data_ptr()
    assert region.shape[0] == split.stride * world
    assert mine.data_ptr() == out[split.slot].data_ptr()
    assert mine.shape[0] == split.stride


def test_unsplit_rows_gather_nothing(monkeypatch):
    calls = _groups(monkeypatch)
    score_module._gather_split(torch.zeros(5, 4), rows_module.RowSplit.of(5))
    assert calls == []


@pytest.mark.parametrize("make", ["of", "striped"])
def test_row_split_refuses_an_unknown_group(make):
    """Anything but "dcp" / "tp" would otherwise gather on the DCP group."""
    with pytest.raises(ValueError):
        getattr(rows_module.RowSplit, make)(8, 0, 2, group="TP")


@pytest.mark.parametrize("world", [2, 3, 4, 8])
@pytest.mark.parametrize("count", [1, 7, 12, 13, 4097])
def test_striped_split_deals_every_row_once(world, count):
    """Rank r scores rows r, r + world, ...: its count of them from the
    front of its rows, into a whole stride's slot at its rank's place."""
    dealt = []
    for rank in range(world):
        split = rows_module.RowSplit.striped(count, rank, world)
        assert (split.first, split.end) == (0, len(range(rank, count, world)))
        assert split.stride == -(-count // world) >= split.end
        assert split.slot == rank * split.stride and split.rank == rank
        assert (split.world, split.group, split.stripe) == (world, "tp", True)
        dealt += range(rank, count, world)
    assert sorted(dealt) == list(range(count))


@pytest.mark.parametrize(
    "fp4, concat",
    [(False, False), (False, True), (True, False)],
    ids=["fp8", "fp8_concat", "fp4"],
)
@pytest.mark.parametrize("count, world", [(13, 4), (16, 4), (3, 4), (4097, 8)])
def test_striped_rows_come_back_rank_major(monkeypatch, fp4, concat, count, world):
    """Each rank scores its rows of the step's query, every `world`-th from
    its rank, into its slot; the slots gather rank-major and every rank gets
    them as they are, row i at `(i % world) * stride + i // world`. A row scores
    as its own id here, so the selection says which row landed where; its
    weights, and FP4's scales, are the same rows. FP4's kernels address their
    rows densely, so it scores copies."""
    topk = 3
    ids = torch.arange(count, dtype=torch.float32)
    values = ids.view(count, 1, 1).expand(count, 1, 2).contiguous()
    query = (values, values + 0.5) if fp4 else values
    weights = ids.view(count, 1)
    cache = rows_module.IndexCache(torch.zeros(1), torch.zeros(1) if fp4 else None, 64)

    def score(query, weights, rows, out):
        values = query[0] if fp4 else query
        if fp4:
            # Dense: a single row's stride too, which `is_contiguous` ignores.
            assert all(
                t.is_contiguous() and t.stride(0) == t.shape[1:].numel()
                for t in (*query, weights)
            )
            assert torch.equal(query[1], values + 0.5)
        assert torch.equal(weights[:, 0], values[:, 0, 0])
        mine = values[: rows.split.end, 0, :1].to(torch.int32)
        out[: rows.split.end] = mine.expand(-1, topk)

    monkeypatch.setattr(
        score_module,
        "_score_paged",
        lambda query, weights, cache, rows, topk, scale, stable, on_band, out: (
            score(query, weights, rows, out)
        ),
    )
    monkeypatch.setattr(
        score_module,
        "_score_concat",
        lambda query, weights, cache, rows, topk, stable, out, keys, local: (
            score(query, weights, rows, out)
        ),
    )
    monkeypatch.setattr(score_module, "get_tp_group", lambda: _Group("tp"))
    # The ranks run as threads; the gather hands each every rank's slot.
    caller, slots = threading.local(), [None] * world
    barrier = threading.Barrier(world, timeout=60)

    def all_gather_into_tensor(out, inp, group):
        slots[caller.rank] = inp.clone()
        barrier.wait()
        out.copy_(torch.cat(slots))

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", all_gather_into_tensor
    )
    results, errors = {}, []

    def rank_step(rank):
        caller.rank = rank
        split = rows_module.RowSplit.striped(count, rank, world)
        rows = rows_module.IndexRows(
            torch.zeros(split.end, dtype=torch.int32),
            1,
            ((0, split.end),),
            split,
            concat=SimpleNamespace() if concat else None,
        )
        try:
            results[rank] = score_module.score_topk(
                query, weights, cache, rows, topk=topk
            )
        except Exception as error:  # noqa: BLE001 - surfaced to the test thread
            errors.append(error)
            barrier.abort()

    threads = [threading.Thread(target=rank_step, args=(r,)) for r in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    stride = -(-count // world)
    row = torch.arange(count)
    at = row % world * stride + row // world
    want = row.to(torch.int32).unsqueeze(1).expand(-1, topk)
    for rank in range(world):
        assert results[rank].shape == (stride * world, topk)
        assert torch.equal(results[rank][at], want), rank


@pytest.mark.parametrize(
    "kwargs",
    [
        {"out": torch.zeros(4, 3, dtype=torch.int32)},
        {"on_band": lambda scores, span: None},
        {"keys": ()},
        {"request_local": False},
    ],
    ids=["out", "on_band", "keys", "request_global"],
)
def test_striped_split_refuses_to_read_its_rows_in_place(kwargs):
    """A striped split hands its selection back rank-major: `out`,
    `on_band`, `keys` and request-global ids all assume rows in order."""
    split = rows_module.RowSplit.striped(4, 0, 2)
    rows = rows_module.IndexRows(
        torch.zeros(split.end, dtype=torch.int32), 1, ((0, split.end),), split
    )
    cache = rows_module.IndexCache(torch.zeros(1), None, 64)
    with pytest.raises(ValueError):
        score_module.score_topk(
            torch.zeros(4, 1, 2), torch.zeros(4, 1), cache, rows, topk=3, **kwargs
        )


@pytest.mark.parametrize(
    "count, world",
    [(13, 4), (16, 4), (3, 4), (1, 2), (4097, 8)],
    ids=["max_tail", "no_tail", "short_tail", "one_row_max_tail", "tp8_max_tail"],
)
def test_the_model_s_entry_keeps_every_slot_behind_the_step_s_rows(
    monkeypatch, count, world
):
    """A striped split's selection is `IndexRows.out_rows` rank-major rows,
    its slots' tails among them; the V4 indexer's op entry hands back a
    view of the step's row count, in the shape its fake promises a compiled
    graph, whose storage still holds every slot for `csa_translate_pack` to
    read through the split."""
    import atom.models.deepseek_v4 as v4
    from atom.model_ops.module_dispatch_ops import _indexer_score_topk_fake

    topk = 8
    split = rows_module.RowSplit.striped(count, world - 1, world)
    rows = rows_module.IndexRows(
        torch.zeros(split.end, dtype=torch.int32), 1, ((0, split.end),), split
    )

    scored = []

    def score_topk(query, weights, cache, rows, *, topk, weight_scale):
        # Each row's selection tagged with its row.
        tags = torch.arange(rows.out_rows, dtype=torch.int32)
        scored.append(tags[:, None].repeat(1, topk))
        return scored[-1]

    monkeypatch.setattr(v4, "score_topk", score_topk)
    monkeypatch.setattr(
        v4,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=SimpleNamespace(index_rows=rows)),
    )
    layer = SimpleNamespace(index_cache=None, _weights_scale=1.0)
    query, weights = torch.zeros(count, 2, 4), torch.zeros(count, 2)
    got = v4.Indexer.indexer_score_topk(layer, query, weights, None, topk)
    fake = _indexer_score_topk_fake(query, weights, None, "layer", topk)
    assert got.shape == fake.shape == (count, topk)
    # A view of the selection itself, every slot behind it.
    assert got.data_ptr() == scored[0].data_ptr() and got.is_contiguous()
    held = got.untyped_storage().nbytes() // got.element_size()
    assert held >= rows.out_rows * topk


@pytest.mark.parametrize(
    "state, split",
    [
        ("PREFILL_NATIVE", rows_module.RowSplit.striped(13, 1, 4)),
        ("PREFILL_NATIVE", rows_module.RowSplit.of(13)),
        ("DECODE", rows_module.RowSplit.of(13)),
    ],
    ids=["prefill_striped", "prefill_whole", "decode"],
)
def test_the_csa_pack_reads_the_selection_through_the_step_s_split(
    monkeypatch, state, split
):
    """The CSA pack is handed the split the indexer scored under -- the same
    `attn_metadata.index_rows` -- so a striped split's rank-major rows are
    read as rows, and rows in order as they are."""
    import atom.models.deepseek_v4 as v4

    seen = {}
    monkeypatch.setattr(
        v4, "csa_translate_pack", lambda *args, **kwargs: seen.update(kwargs)
    )
    rows = rows_module.IndexRows(
        torch.zeros(split.end, dtype=torch.int32), 1, ((0, split.end),), split
    )
    attn_md = SimpleNamespace(
        state=getattr(v4.AttnState, state),
        index_rows=rows,
        **dict.fromkeys(
            (
                "kv_indptr_csa",
                "kv_indices_csa",
                "kv_indptr_prefix_csa",
                "kv_indices_prefix_csa",
                "skip_prefix_len_csa",
                "block_tables",
                "batch_id_per_q_token",
            )
        ),
        envelope_rows=8,
    )
    layer = SimpleNamespace(window_size=128, layer_name="layer")
    v4.DeepseekV4Attention._fill_csa_paged_compress(
        layer, attn_md, torch.zeros(13, 4), torch.zeros(13), 13
    )
    assert seen["stripe"] is split
