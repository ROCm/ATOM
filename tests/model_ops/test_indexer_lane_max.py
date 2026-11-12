# SPDX-License-Identifier: MIT
"""Lane maxima: held beside the packed FP4 logits, written only by an FP4
band launch whose config writes them."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs


@pytest.fixture(scope="module")
def score():
    with atom_under_stubs() as load:
        yield load("atom.model_ops.indexer.score")


def _launch(writes, schedule=None):
    return {"schedule": schedule, "cfg": SimpleNamespace(writes_lane_max=writes)}


def test_only_a_schedule_free_opus_launch_that_writes_them_gets_lane_max(score):
    """aiter refuses a stray `lane_max` rather than ignore it, so the scorer
    gets it only for an OPUS, schedule-free launch whose config writes it."""
    for layout in (score.NATURAL, score.MFMA):
        cache = SimpleNamespace(layout=layout)
        assert score._writes_lane_max(cache, _launch(True))
        assert not score._writes_lane_max(cache, _launch(False))
        assert not score._writes_lane_max(cache, _launch(True, schedule=object()))
    other = SimpleNamespace(layout="preshuffle")
    assert not score._writes_lane_max(other, _launch(True))


def _geometry(index_fp4):
    return SimpleNamespace(
        owners=((0, 1), (1, 2), (2, 2)),
        rows_per_page=lambda ratio: 64 // ratio,
        index_block_rows=8,
        index_blocks_per_page=lambda ratio: 64 // ratio // 8,
        index_fp4=index_fp4,
        speculative_tokens=5,
    )


def test_the_packed_workspace_holds_lane_max_sized_by_aiter():
    """The FP4 workspace allocates the lane maxima with the rest, at the size
    aiter's sizing gives for its packed logits; without the sizing (a chip
    whose scorer writes none), or on the FP8 plane, it holds none."""
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace

    def sizing(elems, rows):
        return elems // 4 + rows

    fp4 = ScoreWorkspace(_geometry(True), 40, 9, "cpu", pages=10, lane_max_elems=sizing)
    lane_max = fp4.packed_lane_max()
    assert lane_max.dtype == torch.float32 and lane_max.dim() == 1
    assert lane_max.numel() == sizing(fp4.capacity, 40)
    assert (
        ScoreWorkspace(_geometry(True), 40, 9, "cpu", pages=10).packed_lane_max()
        is None
    )
    fp8 = ScoreWorkspace(_geometry(False), 40, 9, "cpu", lane_max_elems=sizing)
    assert fp8.packed_lane_max() is None
