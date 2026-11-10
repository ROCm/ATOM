# SPDX-License-Identifier: MIT
"""Lane maxima reach only an FP4 band launch that writes them."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs


@pytest.fixture(scope="module")
def score():
    with atom_under_stubs() as load:
        yield load("atom.model_ops.indexer.score")


def _launch(writes=None, schedule=None):
    cfg = (
        SimpleNamespace() if writes is None else SimpleNamespace(writes_lane_max=writes)
    )
    return {"schedule": schedule, "cfg": cfg}


def test_lane_max_reaches_only_a_launch_that_writes_it(score):
    """The scorer and top-k get `lane_max` only for an OPUS, schedule-free FP4
    band whose config writes it: aiter refuses a stray one rather than ignore
    it, so every other band, plane and workspace passes none."""
    buffer = torch.empty(4)
    workspace = SimpleNamespace(packed_lane_max=lambda: buffer)
    rows = SimpleNamespace(
        fp4_bands=[
            _launch(True),
            _launch(False),
            _launch(None),  # a config from an aiter that writes none
            _launch(True, schedule=object()),
        ]
    )
    for layout in (score.NATURAL, score.MFMA):
        cache = SimpleNamespace(fp4=True, layout=layout)
        assert score._band_lane_max(workspace, cache, rows, 0) is buffer
        for band in (1, 2, 3):
            assert score._band_lane_max(workspace, cache, rows, band) is None
    natural = SimpleNamespace(fp4=True, layout=score.NATURAL)
    other = SimpleNamespace(fp4=True, layout="preshuffle")
    assert score._band_lane_max(workspace, other, rows, 0) is None
    fp8 = SimpleNamespace(fp4=False, layout=score.NATURAL)
    assert score._band_lane_max(workspace, fp8, rows, 0) is None
    unbanded = SimpleNamespace(fp4_bands=None)
    assert score._band_lane_max(workspace, natural, unbanded, 0) is None
    # A workspace without lane maxima: a stand-in, or an aiter without them.
    assert score._band_lane_max(SimpleNamespace(), natural, rows, 0) is None
    empty = SimpleNamespace(packed_lane_max=lambda: None)
    assert score._band_lane_max(empty, natural, rows, 0) is None
