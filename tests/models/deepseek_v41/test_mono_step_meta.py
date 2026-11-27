# SPDX-License-Identifier: MIT
"""GPU metadata and selective resets against the host walk-plan reference."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize(
    "rows,pattern",
    [(1, "single"), (8, "six"), (13, "six"), (40, "mixed"), (64, "single")],
)
def test_parallel_plan_and_reset_replay(rows, pattern):
    from atom.mono.dsa_indexer.paged import plan as ip
    from atom.mono.dsa_indexer.paged.metadata import write_step_meta

    rng = np.random.default_rng(4140 + rows)
    owners = np.arange(rows, dtype=np.int32)
    if pattern == "six":
        owners //= 6
    elif pattern == "mixed":
        owners = np.cumsum(rng.integers(0, 2, rows)).astype(np.int32)
    if rows > 1:
        owners[-1] = -1
    positions = rng.choice([0, 510, 511, 512, 8191, 163839], rows).astype(np.int64)
    positions[owners < 0] = 0
    slots = np.arange(rows, dtype=np.int32)[::-1].copy()
    step = SimpleNamespace(
        batch_ids=torch.tensor(owners, device="cuda"),
        slots=torch.tensor(slots, device="cuda"),
        positions=torch.tensor(positions, device="cuda"),
        block_tables=torch.empty(rows, 1282, device="cuda", dtype=torch.int32),
    )
    geometry = SimpleNamespace(window_size=512)
    window = SimpleNamespace(slot_rows=8192, ring_slots=4096)
    ring = torch.empty(rows, device="cuda", dtype=torch.int32)
    meta = torch.empty(rows, 4, device="cuda", dtype=torch.int32)
    plans = {r: torch.empty(256, 4, device="cuda", dtype=torch.int32) for r in (1, 2)}
    backing = torch.full((131,), 71, device="cuda", dtype=torch.int32)
    resets = (backing[3:74], backing[83:126])

    def launch(lo, hi):
        write_step_meta(
            step,
            geometry,
            window,
            ring,
            meta,
            plans,
            (lo, hi - lo),
            resets,
            parallel_plans=True,
        )

    def check(lo, hi):
        pos = step.positions.cpu().numpy()
        live = owners >= 0
        batch = np.maximum(owners, 0)
        ref_ring = np.where(live, slots[batch] * 8192 + pos % 4096, -1)
        first = np.maximum(pos - 511, 0)
        ref_meta = np.stack(
            (
                first,
                np.where(live, pos + 1 - first, 0),
                batch * 1282,
                slots[batch] * 8192,
            ),
            axis=1,
        )
        torch.testing.assert_close(
            ring.cpu(), torch.from_numpy(ref_ring).int(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            meta.cpu(), torch.from_numpy(ref_meta).int(), rtol=0, atol=0
        )
        for ratio, plan in plans.items():
            ref = np.empty((256, 4), dtype=np.int32)
            ip.fill_walk_plan(
                (pos[lo:hi] + 1) // ratio, ip.walk_pieces(owners[lo:hi]), ref
            )
            ref[ref[:, 1] > 0, 0] += lo
            torch.testing.assert_close(
                plan.cpu(), torch.from_numpy(ref), rtol=0, atol=0
            )
        assert all(torch.count_nonzero(t).item() == 0 for t in resets)
        untouched = torch.cat((backing[:3], backing[74:83], backing[126:]))
        assert torch.equal(untouched, torch.full_like(untouched, 71))

    for lo, hi in ((0, rows), (rows // 3, rows)):
        backing.fill_(71)
        launch(lo, hi)
        check(lo, hi)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(0, rows)
    step.positions.add_(17)
    backing.fill_(71)
    graph.replay()
    check(0, rows)


@pytest.mark.parametrize(
    "world,rank,parallel", [(2, 1, False), (4, 0, False), (4, 3, True), (8, 7, True)]
)
def test_tp_rank_work_and_short_to_long_graph_switch(world, rank, parallel):
    from atom.mono.dsa_indexer.paged import plan as ip
    from atom.mono.dsa_indexer.paged import tp_plan as tp
    from atom.mono.dsa_indexer.paged.metadata import write_step_meta

    # Three causal query groups, including two-request mixed short/long
    # steps: a short request stays owned even when there are fewer than TP.
    owners = np.array([0] * 6 + [1] * 2 + [-1, -1], dtype=np.int32)
    rows = len(owners)
    step = SimpleNamespace(
        batch_ids=torch.tensor(owners, device="cuda"),
        slots=torch.arange(2, device="cuda", dtype=torch.int32),
        positions=torch.zeros(rows, device="cuda", dtype=torch.int64),
        block_tables=torch.empty(2, 8194, device="cuda", dtype=torch.int32),
    )
    ring = torch.empty(rows, device="cuda", dtype=torch.int32)
    meta = torch.empty(rows, 4, device="cuda", dtype=torch.int32)
    ix_meta = torch.empty_like(meta)
    plans = {
        ratio: torch.empty(256, 4, device="cuda", dtype=torch.int32) for ratio in (1, 2)
    }
    backing = torch.empty(79, device="cuda", dtype=torch.int32)
    peer = backing[3:74]

    def call():
        write_step_meta(
            step,
            SimpleNamespace(window_size=512),
            SimpleNamespace(slot_rows=8192, ring_slots=4096),
            ring,
            meta,
            plans,
            (0, rows),
            (),
            parallel_plans=parallel,
            ix_meta=ix_meta,
            tp_world=world,
            tp_rank=rank,
            peer_reset=peer,
        )

    # Capture once; contexts and request IDs change without host branching.
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for first, second in (
        (8191, 8191),
        (163839, 163839),
        (8192, 2047),
        (2047, 163839),
        (524287, 16383),
        (8191, 8191),
    ):
        positions = np.array(
            [first + i for i in range(6)] + [second, second + 1, 0, 0], dtype=np.int64
        )
        step.positions.copy_(torch.from_numpy(positions).to("cuda"))
        backing.fill_(71)
        graph.replay()
        actual_meta = ix_meta.cpu().numpy()
        np.testing.assert_array_equal(actual_meta[:, 0], [0] * 6 + [1] * 2 + [-1, -1])
        np.testing.assert_array_equal(actual_meta[:8, 1], [0] * 6 + [6] * 2)
        np.testing.assert_array_equal(actual_meta[:8, 2], [5] * 6 + [7] * 2)
        expected_flags = 2 | (65536 if max(first, second) + 1 <= 8192 else 0)
        assert np.all(actual_meta[:, 3] == expected_flags)
        assert torch.equal(
            peer, torch.full_like(peer, 71 if expected_flags & 65536 else 0)
        )
        assert torch.equal(
            torch.cat((backing[:3], backing[74:])),
            torch.full((8,), 71, device="cuda", dtype=torch.int32),
        )
        for ratio, plan in plans.items():
            bounds = np.where(owners >= 0, (positions + 1) // ratio, 0)
            work = tp.reference_work(bounds.tolist(), owners.tolist(), world, rank)
            lead, spans = ip.walk_pieces(owners)
            expected = {}
            for row, span in zip(lead, spans):
                w = work[row + span - 1]
                if w.scored:
                    expected[(row, span)] = list(
                        range(w.first // ip.CHUNK, ip.cdiv(w.first + w.count, ip.CHUNK))
                    )
            actual = {}
            for row, span, first_chunk, end_chunk in plan.cpu().numpy():
                if span:
                    actual.setdefault((row, span), []).extend(
                        range(first_chunk, end_chunk)
                    )
            assert actual == expected


@pytest.mark.parametrize("world", [2, 4])
def test_split_histogram_reset_tracks_live_graph_request_groups(world):
    from atom.mono.dsa_indexer.paged.metadata import write_step_meta

    rows = 8
    step = SimpleNamespace(
        batch_ids=torch.arange(rows, device="cuda", dtype=torch.int32),
        slots=torch.arange(rows, device="cuda", dtype=torch.int32),
        positions=torch.zeros(rows, device="cuda", dtype=torch.int64),
        block_tables=torch.empty(rows, 4098, device="cuda", dtype=torch.int32),
    )
    ring = torch.empty(rows, device="cuda", dtype=torch.int32)
    meta = torch.empty(rows, 4, device="cuda", dtype=torch.int32)
    flags = torch.empty(64, device="cuda", dtype=torch.int32)
    hist = torch.empty(128, device="cuda", dtype=torch.int32)
    backing = torch.empty(264, device="cuda", dtype=torch.int32)
    peer = backing[4:260]
    prefix_bytes = 64

    def call():
        write_step_meta(
            step,
            SimpleNamespace(window_size=512),
            SimpleNamespace(slot_rows=8192, ring_slots=4096),
            ring,
            meta,
            {},
            (0, rows),
            (flags,),
            tp_world=world,
            peer_reset=peer,
            split_reset_buffers=(hist,),
            peer_nonsplit_bytes=prefix_bytes,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    # One graph visits local, request ownership, key splitting, a dominant
    # request, and a causal group crossing the dominant-request threshold.
    states = [
        ([8191] * rows, list(range(rows)), False, False),
        ([163839] * rows, list(range(rows)), True, False),
        ([163839] * rows, [0] * 6 + [1] * 2, True, world == 4),
        ([524287] + [163839] * 7, list(range(rows)), True, True),
        ([262143 + i for i in range(6)] + [163839] * 2, [0] * 6 + [1] * 2, True, True),
        ([8191] * rows, list(range(rows)), False, False),
    ]
    for positions, ids, long_step, split_step in states * 2:
        step.positions.copy_(torch.tensor(positions, device="cuda", dtype=torch.int64))
        step.batch_ids.copy_(torch.tensor(ids, device="cuda", dtype=torch.int32))
        flags.fill_(71)
        hist.fill_(71)
        backing.fill_(71)
        graph.replay()
        assert torch.equal(flags, torch.zeros_like(flags))
        assert torch.equal(hist, torch.full_like(hist, 0 if split_step else 71))
        expected = torch.full_like(peer, 71)
        if long_step:
            expected[: peer.numel() if split_step else prefix_bytes // 4] = 0
        assert torch.equal(peer, expected)
        assert torch.equal(
            torch.cat((backing[:4], backing[260:])),
            torch.full((8,), 71, device="cuda", dtype=torch.int32),
        )
