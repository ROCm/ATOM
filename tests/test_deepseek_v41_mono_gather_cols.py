# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""DeepSeek-V4.1 mono column gather (``kernels.col_gather``): a sharded step
leaves each rank only its own columns, and one launch pushes them to every
rank and lands every column in place. Here, without a GPU, the plan:
every (rank, plane, row, column) pushed once, every (row, plane, column)
landed once, each in its own CG pairs inside the peer buffer's CG region,
and the runner's choice of gathers."""

import sys
from types import SimpleNamespace

import pytest
from aiter_stub import _drop, atom_without_aiter

pytest.importorskip("flydsl")


@pytest.fixture(scope="module")
def M():
    before = set(sys.modules)
    with atom_without_aiter() as load:
        cg = load("atom.models.deepseek_v41.mono.kernels.col_gather")
        layout = load("atom.models.deepseek_v41.mono.kernels.peer_layout")
        runner = load("atom.models.deepseek_v41.mono.runner")
        yield SimpleNamespace(cg=cg, layout=layout, runner=runner)
    for name in set(sys.modules) - before:
        if name.startswith("atom.models.deepseek_v41.mono"):
            _drop(name)


def _settle_past_a_wave(cg, state, taps, tp) -> bool:
    """A settle whose (row, quad) pushes span more than one wave (refused)."""
    lanes = cg.THREADS // cg.WAVES
    return state == cg.State.SETTLE and lanes % ((cg.HC + taps) * tp) != 0


def builds(cg):
    for state in cg.State:
        for taps in range(cg.MAX_TAPS + 1):
            if state == cg.State.NONE and taps == 0:
                continue
            for tp in (2, 4, 8):
                if _settle_past_a_wave(cg, state, taps, tp):
                    continue
                for rows in (1, 6, 11, 48):
                    yield cg.ColGatherBuild(rows=rows, taps=taps, state=state, tp=tp)


def test_a_settle_past_one_wave_is_refused(M):
    """SETTLE lands into the residual its pushes read: only where a (row,
    quad)'s pushes are one wave's lanes, every one reads before a land."""
    cg = M.cg
    refused = [
        (taps, tp)
        for taps in range(cg.MAX_TAPS + 1)
        for tp in (2, 4, 8)
        if _settle_past_a_wave(cg, cg.State.SETTLE, taps, tp)
    ]
    assert refused and (0, 8) not in refused  # the runner's settle builds
    for taps, tp in refused:
        with pytest.raises(AssertionError, match="past one wave"):
            cg.ColGatherBuild(rows=6, taps=taps, state=cg.State.SETTLE, tp=tp)


def test_every_column_is_pushed_to_every_rank_once(M):
    cg = M.cg
    hidden = cg.HIDDEN
    for p in builds(cg):
        for rank in range(p.tp):
            seen = {cg.push_task(g, p, rank) for g in range(p.pushes)}
            assert len(seen) == p.pushes
            width = hidden // p.tp
            assert {col for *_, col in seen} == set(
                range(rank * width, (rank + 1) * width, 8)
            )
            assert {(d, pl, t) for d, pl, t, _ in seen} == {
                (d, pl, t)
                for d in range(p.tp)
                for pl in range(p.planes)
                for t in range(p.rows)
            }


def test_every_column_lands_once(M):
    cg = M.cg
    for p in builds(cg):
        seen = {cg.land_task(g, p) for g in range(p.lands)}
        assert seen == {
            (t, pl, col)
            for t in range(p.rows)
            for pl in range(p.planes)
            for col in range(0, cg.HIDDEN, 8)
        }


def test_cg_pairs_are_distinct_and_inside_the_region(M):
    cg, layout = M.cg, M.layout
    for p in builds(cg):
        # a land reads pairs at and at + 2, of 2 pairs each: 4 pairs a quad
        pairs = set()
        for t, pl, col in (cg.land_task(g, p) for g in range(p.lands)):
            at = cg.cg_pair(t, pl, col)
            quad = set(range(at, at + 4))
            assert not pairs & quad
            pairs |= quad
        _, nbytes = layout.peer_layout(p.rows, layout.Dims(p.tp))["cg"]
        assert max(pairs) < nbytes // 8


def test_enough_threads_for_one_round_of_each(M):
    cg = M.cg
    for p in builds(cg):
        assert p.ctas <= cg.BLOCKS
        if p.ctas < cg.BLOCKS:
            assert p.threads >= max(p.pushes, p.lands)


@pytest.mark.parametrize(
    ("shards", "engram", "expect"),
    [
        (False, True, set()),
        (True, False, {"COLLAPSE"}),
        (True, True, {"COLLAPSE", "SETTLE"}),
    ],
)
def test_runner_builds_the_gathers_a_sharded_step_reads(M, shards, engram, expect):
    R, cg = M.runner, M.cg
    runner = R.V41MonoDecodeRunner.__new__(R.V41MonoDecodeRunner)
    runner.tp = 4
    runner.aux = SimpleNamespace(buffers=(object(), object(), object()))
    runner.model = SimpleNamespace(
        layers=[SimpleNamespace(engram=object() if engram else None)]
    )
    runner._shard = lambda s: shards
    got = runner._gathers(6)
    assert {state.name for state in got} == expect
    if shards:
        assert got[cg.State.COLLAPSE].taps == 3
        if engram:
            assert got[cg.State.SETTLE].taps == 0
