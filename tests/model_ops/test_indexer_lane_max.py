# SPDX-License-Identifier: MIT
"""Lane maxima: allocated where aiter says the OPUS launches write them, held
beside the packed FP4 logits, handed only to a band of at least 128 rows, and
written only by an FP4 band launch whose config writes them."""

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


@pytest.mark.parametrize("count", [1, 127, 128, 129])
def test_a_band_below_128_rows_hands_lane_max_to_neither_kernel(
    score, monkeypatch, count
):
    """A packed FP4 band of fewer than 128 rows passes no lane maxima to its
    scorer or its top-k; from 128 rows on both get the workspace's buffer.
    The band's rows alone decide it, so a captured batch decides it once."""
    lane_max = torch.zeros(16)
    workspace = SimpleNamespace(
        packed_logits=lambda: torch.zeros(4 * count),
        packed_lane_max=lambda: lane_max,
    )
    rows = SimpleNamespace(
        workspace=workspace,
        bands=[(0, count)],
        next_n=1,
        packed_offsets=torch.arange(count, dtype=torch.int32),
        row_ends=torch.ones(count, dtype=torch.int32),
        width=4,
    )
    seen = {}

    def scorer(*args):
        seen["scorer"] = args[-1]

    def top_k(*args, **kwargs):
        seen["top_k"] = kwargs["lane_max"]

    monkeypatch.setattr(score, "_fp4_band_logits", scorer)
    monkeypatch.setattr(score, "top_k_per_row_decode", top_k)
    query = (torch.zeros(count, 1, 2), torch.zeros(count, 1, 1))
    score._score_paged(
        query,
        torch.zeros(count, 2),
        SimpleNamespace(fp4=True),
        rows,
        8,
        1.0,
        True,
        None,
        torch.empty(count, 8, dtype=torch.int32),
    )
    expected = lane_max if count >= 128 else None
    assert seen["scorer"] is expected
    assert seen["top_k"] is expected


@pytest.fixture(scope="module")
def backend():
    with atom_under_stubs() as load:
        yield load("atom.model_ops.attentions.deepseek_v41.backend")


@pytest.mark.parametrize("writes", [True, False])
@pytest.mark.parametrize("layout", ["natural", "mfma", "preshuffle", None])
def test_the_workspace_sizes_lane_max_from_aiters_query(
    backend, monkeypatch, writes, layout
):
    """The buffer is sized only where aiter says the OPUS launches on this
    device with these index heads write lane maxima, and only on an
    OPUS-scored FP4 plane: the device and head count go to aiter as they
    are, and ATOM keeps no arch rule of its own."""
    asked = []

    def query(device, heads):
        asked.append((device, heads))
        return writes

    monkeypatch.setattr(backend, "mqa_logits_mxfp4_writes_lane_max", query)
    sizing = backend._lane_max_sizing("cuda:1", 32, layout)
    opus = layout in (backend.NATURAL, backend.MFMA)
    if opus and writes:
        assert sizing is backend.mqa_logits_mxfp4_lane_max_elems
    else:
        assert sizing is None
    assert asked == ([("cuda:1", 32)] if opus else [])
