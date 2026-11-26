# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A V4.1 candidate-source indexer layer takes its packed bands' top-k over
the blocks it has just picked (`paged_scoring._topk_within_picks`), and the
shared scorer runs its plain top-k only when that does not."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs


@pytest.fixture(scope="module")
def M():
    # AITER and Triton stand in (`tests/aiter_stub.py`), so this runs on the
    # GPU-less CI; nothing here launches a kernel.
    with atom_under_stubs() as load:
        yield SimpleNamespace(
            scoring=load("atom.model_ops.deepseek_v41.paged_scoring"),
            score=load("atom.model_ops.indexer.score"),
            workspace=load("atom.model_ops.deepseek_v41.score_workspace"),
            geometry=load("atom.model_ops.attentions.pool_layout.v41_pool_geometry"),
        )


def _workspace(M, rows, width):
    """`ScoreWorkspace.candidate_scratch` over a plane `rows` x `width`."""
    space = SimpleNamespace(
        candidate_rows=rows,
        _candidate_plane=torch.zeros(rows * width),
        _candidate_lengths=torch.zeros(rows, dtype=torch.int32),
    )
    space.candidate_scratch = lambda *a: M.workspace.ScoreWorkspace.candidate_scratch(
        space, *a
    )
    return space


def test_only_more_blocks_than_the_top_k_hold_it(M):
    chosen = torch.zeros(4, 512, dtype=torch.int32)
    space = _workspace(M, 4, 512 * 8)
    assert M.scoring._topk_within_picks(chosen, 8, 512, space) is None
    assert M.scoring._topk_within_picks(chosen, 8, 511, space) is not None


def test_no_reserved_plane_keeps_the_whole_row_top_k(M):
    chosen = torch.zeros(4, 16, dtype=torch.int32)
    assert M.scoring._topk_within_picks(chosen, 8, 4, _workspace(M, 0, 0)) is None


def test_a_band_takes_its_top_k_within_its_picks(M, monkeypatch):
    calls = []
    monkeypatch.setattr(
        M.scoring, "top_k_within_candidates", lambda *a: calls.append(a)
    )
    monkeypatch.setattr(M.scoring, "candidate_topk_launch", lambda count: ("L", count))
    chosen = torch.arange(6 * 16, dtype=torch.int32).view(6, 16)
    space = _workspace(M, 4, 16 * 8)
    band_topk = M.scoring._topk_within_picks(chosen, 8, 4, space)
    out = torch.empty(2, 4, dtype=torch.int32)
    assert band_topk("scores", slice(2, 4), out, "starts", "ends")
    (args,) = calls
    assert args[:3] == ("scores", "starts", "ends")
    assert torch.equal(args[3], chosen[2:4])
    assert args[4] == 8 and args[5] is out and args[6] == 4
    plane, lengths, launch = args[7:]
    # Views of the reserved plane, not fresh allocations.
    assert plane.shape == (2, 128) and lengths.shape == (2,)
    assert plane.data_ptr() == space._candidate_plane.data_ptr()
    assert lengths.data_ptr() == space._candidate_lengths.data_ptr()
    assert launch == ("L", 16)


def test_launch_constants_are_worked_out_once_per_run(M, monkeypatch):
    counts = []
    monkeypatch.setattr(M.scoring, "top_k_within_candidates", lambda *a: None)
    monkeypatch.setattr(M.scoring, "candidate_topk_launch", counts.append)
    chosen = torch.zeros(6, 16, dtype=torch.int32)
    band_topk = M.scoring._topk_within_picks(chosen, 8, 4, _workspace(M, 4, 128))
    out = torch.empty(2, 4, dtype=torch.int32)
    for span in (slice(0, 2), slice(2, 4), slice(4, 6)):
        assert band_topk("s", span, out, "st", "en")
    assert counts == [16]


def test_a_band_too_tall_keeps_the_whole_row_top_k(M, monkeypatch):
    def refuse(*a):
        pytest.fail("a tall band took the compacted top-k")

    monkeypatch.setattr(M.scoring, "top_k_within_candidates", refuse)
    chosen = torch.zeros(3, 16, dtype=torch.int32)
    band_topk = M.scoring._topk_within_picks(chosen, 8, 4, _workspace(M, 2, 128))
    assert not band_topk("s", slice(0, 3), torch.empty(3, 4), "st", "en")


@pytest.mark.parametrize("index_fp4", [True, False])
@pytest.mark.parametrize("tokens", [8192, 40])
def test_the_reserved_plane_is_sized_from_the_configuration(M, tokens, index_fp4):
    """`min(max_tokens, CANDIDATE_TOPK_MAX_ROWS)` rows of the candidate list's
    width, held by an FP4 (packed) workspace with a candidate list only."""
    W = M.workspace
    geometry = M.geometry.V41PoolGeometry(
        2, ((0, 1), (1, 2)), 256, 128, 512, 128,
        index_block_rows=8, index_fp4=index_fp4,
    )  # fmt: skip
    blocks, columns = 64, 1048576 // geometry.block_size
    plain = W.ScoreWorkspace(geometry, tokens, columns, "meta", pages=columns)
    assert plain.candidate_rows == 0
    assert plain.candidate_scratch(1, blocks * 8) is None
    space = W.ScoreWorkspace(
        geometry, tokens, columns, "meta", pages=columns, candidate_blocks=blocks
    )
    if not index_fp4:
        assert space.candidate_rows == 0
        return
    rows = min(tokens, W.CANDIDATE_TOPK_MAX_ROWS)
    assert space.candidate_rows == rows
    assert space._candidate_plane.numel() == rows * blocks * 8
    plane, lengths = space.candidate_scratch(rows, blocks * 8)
    assert plane.shape == (rows, blocks * 8) and lengths.shape == (rows,)
    assert space.candidate_scratch(rows + 1, blocks * 8) is None
    assert space.candidate_scratch(rows, (blocks + 1) * 8) is None


def _packed_rows(bands):
    return SimpleNamespace(
        workspace=SimpleNamespace(packed_logits=lambda: "packed"),
        bands=bands,
        next_n=1,
        packed_offsets=torch.arange(4, dtype=torch.int32),
        row_ends=torch.arange(10, 14, dtype=torch.int32),
        width=64,
    )


@pytest.mark.parametrize("taken", [True, False])
def test_the_scorer_runs_its_top_k_only_when_the_band_did_not(M, monkeypatch, taken):
    score_mod = M.score
    plain, seen = [], []
    monkeypatch.setattr(
        score_mod, "top_k_per_row_decode", lambda *a, **k: plain.append(a[3])
    )
    monkeypatch.setattr(score_mod, "_fp4_band_logits", lambda *a, **k: None)
    cache = SimpleNamespace(fp4=True)

    def band_topk(scores, span, out, starts, ends):
        seen.append((scores, span, starts.tolist(), ends.tolist()))
        return taken

    out = torch.empty(4, 2, dtype=torch.int32)
    query = (torch.zeros(4, 1), torch.zeros(4, 1))
    score_mod._score_paged(
        query,
        torch.zeros(4, 1),
        cache,
        _packed_rows([(0, 2), (2, 4)]),
        2,
        1.0,
        True,
        None,
        out,
        band_topk,
    )
    assert seen == [
        ("packed", slice(0, 2), [0, 1], [10, 11]),
        ("packed", slice(2, 4), [2, 3], [12, 13]),
    ]
    assert len(plain) == (0 if taken else 2)


def test_a_striped_split_refuses_a_band_top_k(M):
    score_mod = M.score
    rows = SimpleNamespace(split=SimpleNamespace(stripe=True))
    with pytest.raises(ValueError, match="rank-major"):
        score_mod.score_topk(
            None, torch.zeros(1, 1), None, rows, topk=1, band_topk=lambda *a: True
        )
