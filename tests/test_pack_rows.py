# SPDX-License-Identifier: MIT

"""`atom.utils.pack_rows` on its own terms.

Its callers live in attention backends, whose modules import AITER at load and
so cannot be collected on a plain runner. The helper does not, and it is the
one piece the two marshals share, so its contract is pinned here where CI can
execute it: which rows it clears, which destinations it refuses, and the two
edges a flat write does not have.

The callers' side -- that each marshal still writes what the per-row loop it
replaced wrote -- is in `test_block_table_marshal.py`.
"""

from __future__ import annotations

import array

import numpy as np
import pytest

from atom.model_engine.sequence import new_block_table
from atom.utils import pack_rows

COLS = 40
# Neither 0 nor a plausible block id, so a cell that should have been cleared
# and a cell that should have been filled are both distinguishable from one
# that was never reached.
POISON = -7


def _rows(n: int) -> list[array.array]:
    """`n` rows of varying length, none empty, none full-width."""
    return [
        new_block_table(range(100 * (i + 1), 100 * (i + 1) + 1 + (i % (COLS - 2))))
        for i in range(n)
    ]


def test_rows_land_left_aligned_with_their_tails_cleared():
    """Callers hand it an uncleared destination -- `np.empty` at the TBO site
    -- and rely on it to zero-fill each row's tail. What it must not do is
    clear rows nobody scheduled."""
    dst = np.full((6, COLS), POISON, dtype=np.int32)
    rows = _rows(3)

    pack_rows(dst, rows)

    for i, row in enumerate(rows):
        assert list(dst[i, : len(row)]) == list(row)
        assert (dst[i, len(row) :] == 0).all(), "row tail not cleared"
    assert (dst[len(rows) :] == POISON).all(), "cleared rows it was not given"


def test_an_over_wide_row_raises_instead_of_reaching_the_next_one():
    """A flat write has no row edge: the overflow would land in the next
    request's row, leaving it a block table pointing at someone else's KV."""
    dst = np.full((4, COLS), POISON, dtype=np.int32)
    rows = _rows(3)
    rows[1] = new_block_table(range(COLS + 1))

    with pytest.raises(ValueError, match="exceeds"):
        pack_rows(dst, rows)


def test_more_rows_than_the_destination_holds_raises():
    """The other edge. Unchecked, this surfaces as a memoryview structure
    error naming neither the row nor the destination."""
    dst = np.full((3, COLS), POISON, dtype=np.int32)

    with pytest.raises(ValueError, match="rows exceed"):
        pack_rows(dst, _rows(4))

    pack_rows(dst, _rows(3))  # control: exactly full is fine


def test_a_destination_whose_bits_it_would_reinterpret_is_refused():
    """`.cast("i")` reinterprets rather than converts, so a destination of any
    other dtype would take the block ids as raw bits."""
    dst = np.zeros((4, COLS), dtype=np.float32)

    with pytest.raises(TypeError, match="int32"):
        pack_rows(dst, _rows(2))

    # Control: nothing about the cast itself would have complained.
    dst[0, 0] = 1.0
    assert memoryview(dst).cast("B").cast("i")[0] == 1065353216


def test_a_non_contiguous_destination_raises_rather_than_dropping_writes():
    """Flattening a non-contiguous buffer with `reshape(-1)` yields a copy, so
    every row would be written into a temporary and lost with it. `.cast`
    refuses instead, and unlike an assert it still refuses under `python -O`.
    """
    backing = np.full((4, COLS * 2), POISON, dtype=np.int32)
    dst = backing[:, :COLS]  # a view, so rows are not adjacent
    assert not dst.flags.c_contiguous

    with pytest.raises(TypeError, match="C-contiguous"):
        pack_rows(dst, _rows(2))

    # Positive control: the failure mode being guarded is real and silent.
    assert not np.shares_memory(dst.reshape(-1), dst)


def test_no_rows_leaves_the_destination_alone():
    """A warmup batch carries none."""
    dst = np.full((4, COLS), POISON, dtype=np.int32)

    pack_rows(dst, [])

    assert (dst == POISON).all()


def test_the_rows_are_copied_out_of_not_aliased():
    """An aliasing read would pin the `array("i")` and surface as a
    `BufferError` from `BlockManager`'s next append, a step or more later."""
    rows = _rows(4)
    dst = np.full((4, COLS), POISON, dtype=np.int32)

    pack_rows(dst, rows)

    for row in rows:
        row.append(999)  # BufferError here means someone kept a view
