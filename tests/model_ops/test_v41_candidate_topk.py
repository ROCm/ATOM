# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A V4.1 candidate-source indexer layer takes its packed bands' top-k over
the blocks it has just picked (`paged_scoring._topk_within_picks`), and the
shared scorer runs its plain top-k only when that does not."""

from types import SimpleNamespace

import pytest
import torch

try:
    from atom.model_ops.deepseek_v41 import paged_scoring
    from atom.model_ops.indexer import score as score_mod
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "requires full atom import env")


def test_only_more_blocks_than_the_top_k_hold_it():
    chosen = torch.zeros(4, 512, dtype=torch.int32)
    assert paged_scoring._topk_within_picks(chosen, 8, 512) is None
    assert paged_scoring._topk_within_picks(chosen, 8, 511) is not None


def test_a_band_takes_its_top_k_within_its_picks(monkeypatch):
    calls = []
    monkeypatch.setattr(
        paged_scoring, "top_k_within_candidates", lambda *a: calls.append(a)
    )
    chosen = torch.arange(6 * 16, dtype=torch.int32).view(6, 16)
    band_topk = paged_scoring._topk_within_picks(chosen, 8, 4)
    out = torch.empty(2, 4, dtype=torch.int32)
    assert band_topk("scores", slice(2, 4), out, "starts", "ends")
    (args,) = calls
    assert args[:3] == ("scores", "starts", "ends")
    assert torch.equal(args[3], chosen[2:4])
    assert args[4] == 8 and args[5] is out and args[6] == 4


def test_a_band_too_tall_keeps_the_whole_row_top_k(monkeypatch):
    monkeypatch.setattr(paged_scoring, "CANDIDATE_TOPK_MAX_ROWS", 2)

    def refuse(*a):
        pytest.fail("a tall band took the compacted top-k")

    monkeypatch.setattr(paged_scoring, "top_k_within_candidates", refuse)
    chosen = torch.zeros(3, 16, dtype=torch.int32)
    band_topk = paged_scoring._topk_within_picks(chosen, 8, 4)
    assert not band_topk("s", slice(0, 3), torch.empty(3, 4), "st", "en")


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
def test_the_scorer_runs_its_top_k_only_when_the_band_did_not(monkeypatch, taken):
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


def test_a_striped_split_refuses_a_band_top_k():
    rows = SimpleNamespace(split=SimpleNamespace(stripe=True))
    with pytest.raises(ValueError, match="rank-major"):
        score_mod.score_topk(
            None, torch.zeros(1, 1), None, rows, topk=1, band_topk=lambda *a: True
        )
