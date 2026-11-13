# SPDX-License-Identifier: MIT
"""Shared packed-row planning, runnable without AITER or a GPU."""

import itertools

import numpy as np
import pytest

from atom.model_ops.indexer.workspace import ROW_ALIGN, aligned_spans, band_offsets


@pytest.mark.parametrize("seed", range(4))
def test_packed_bands_hold_each_row_aligned_and_fill_before_cutting(seed):
    """Bands partition the rows and fill without growing the workspace."""
    rng = np.random.default_rng(seed)
    visible = rng.integers(0, 700, size=50).astype(np.int32)
    visible[::7] = 0
    capacity = 2048
    offsets = np.full(50, -1, dtype=np.int32)
    bands = band_offsets(visible, capacity, offsets)
    spans = aligned_spans(visible.astype(np.int64))
    assert bands[0][0] == 0 and bands[-1][1] == 50
    assert all(a[1] == b[0] for a, b in itertools.pairwise(bands))
    for first, end in bands:
        within = spans[first:end]
        assert np.array_equal(offsets[first:end], np.cumsum(within) - within)
        assert int(within.sum()) <= capacity
        if end < 50:
            assert int(within.sum() + spans[end]) > capacity
    assert not (offsets % ROW_ALIGN).any()
    with pytest.raises(ValueError, match="exceeds"):
        band_offsets(np.array([capacity + 1], np.int32), capacity, offsets)


@pytest.mark.parametrize("visible,expected", [([], ()), ([0, 0, 0], ((0, 3),))])
def test_empty_packed_rows_need_no_storage(visible, expected):
    offsets = np.full(len(visible), -1, dtype=np.int32)
    assert band_offsets(np.array(visible, dtype=np.int32), 0, offsets) == expected
    assert not offsets.any()
