# SPDX-License-Identifier: MIT

"""Marshalling `Sequence.block_table` rows into the int32 `block_tables` buffer.

Two properties have to hold, and neither shows up in a throughput number:

* the destination is written exactly as the pre-`array("i")` loop wrote it.
  The marshal hoists the per-row `[i] = 0` into one span memset, which is
  only equivalent because the old loop cleared precisely the rows it filled.
* nobody may retain a zero-copy int32 view of a block_table. `array("i")`
  refuses to resize while exporting a buffer, so such a view turns the next
  `BlockManager` append into a `BufferError`, far from the code that took it.

The marshal is exercised as an unbound method on a stub that supplies what it
reads, so the arithmetic under test is the shipped arithmetic. Every builder
here lives in a module that imports AITER at load, so this file skips whole on
a plain runner; the contract of the `pack_rows` helper it uses is in
`test_pack_rows.py`, which does not, and so runs in CI.
"""

from __future__ import annotations

import array
from types import SimpleNamespace

import numpy as np
import pytest

from atom.model_engine.sequence import new_block_table

CommonAttentionBuilder = pytest.importorskip(
    "atom.model_ops.attentions.backends",
    reason="the common builder's module imports aiter at load",
    exc_type=ImportError,
).CommonAttentionBuilder

AiterBuilder = pytest.importorskip(
    "atom.model_ops.attentions.aiter_attention",
    reason="the MHA builder's module imports aiter at load",
    exc_type=ImportError,
).AiterAttentionMetadataBuilder

MAX_BS = 300
MAX_COLS = 40
# Neither 0 nor a plausible block id, so a cell that should have been cleared
# and a cell that should have been filled are both distinguishable from one
# that was never reached.
POISON = -7


def _rows(bs: int) -> list[array.array]:
    """`bs` block tables of varying length, none empty, none full-width."""
    return [
        new_block_table(range(100 * (i + 1), 100 * (i + 1) + 1 + (i % (MAX_COLS - 2))))
        for i in range(bs)
    ]


def _golden(rows: list[array.array]) -> np.ndarray:
    """The destination as the pre-hoist per-row loop would have left it."""
    dst = np.full((MAX_BS, MAX_COLS), POISON, dtype=np.int32)
    for i, row in enumerate(rows):
        dst[i] = 0
        dst[i, : len(row)] = row
    return dst


def _stub(dst: np.ndarray):
    """A builder stub whose `forward_vars["block_tables"]` wraps `dst`."""
    buf = SimpleNamespace(np=dst, copy_to_gpu=lambda n: ("gpu", n))
    return SimpleNamespace(
        model_runner=SimpleNamespace(forward_vars={"block_tables": buf})
    )


@pytest.mark.parametrize("bs", [0, 1, 2, 50, 64, 256])
def test_marshal_is_bit_exact_and_stays_in_its_rows(bs):
    rows = _rows(bs)
    dst = np.full((MAX_BS, MAX_COLS), POISON, dtype=np.int32)

    CommonAttentionBuilder.prepare_block_tables(
        _stub(dst), SimpleNamespace(block_tables=rows)
    )

    assert np.array_equal(dst, _golden(rows))
    assert (dst[bs:] == POISON).all(), "wrote past the batch's rows"


def test_empty_batch_writes_nothing():
    """A warmup batch carries no block tables and must leave the buffer alone."""
    dst = np.full((MAX_BS, MAX_COLS), POISON, dtype=np.int32)
    batch = SimpleNamespace(block_tables=[])

    CommonAttentionBuilder.prepare_block_tables(_stub(dst), batch)

    assert (dst == POISON).all()
    assert batch.block_tables == []


def test_marshal_leaves_no_buffer_export_on_the_row():
    """The marshal must copy out of the row, not alias it.

    An aliasing read would pin the array and surface as a `BufferError` from
    `BlockManager`'s next append, a step or more later.
    """
    rows = _rows(4)
    dst = np.full((MAX_BS, MAX_COLS), POISON, dtype=np.int32)

    CommonAttentionBuilder.prepare_block_tables(
        _stub(dst), SimpleNamespace(block_tables=rows)
    )

    for row in rows:
        row.append(999)  # BufferError here means someone kept a view


def test_tbo_prefill_stash_does_not_pin_the_rows_it_keeps():
    """The one place that keeps block tables past the call that read them.

    `_stash_tbo_token_split_prefill_state` holds its rows until a later ubatch
    rebuilds the straddled request's prefix, so an int32 view taken here is
    still live when `BlockManager` next grows the sequence.
    """
    bs = 4
    rows = _rows(bs)
    stub = SimpleNamespace(
        _tbo_prefill_state=None,
        model_runner=SimpleNamespace(
            forward_vars={
                "cu_seqlens_q": SimpleNamespace(np=np.arange(bs + 1, dtype=np.int32))
            }
        ),
    )
    batch = SimpleNamespace(
        block_tables=rows,
        total_seqs_num_prefill=bs,
        num_cached_tokens=[0] * bs,
    )

    AiterBuilder._stash_tbo_token_split_prefill_state(stub, batch)

    assert stub._tbo_prefill_state is not None
    kept = stub._tbo_prefill_state.block_tables
    assert [len(r) for r in kept] == [len(r) for r in rows]

    for row in rows:
        row.append(999)  # BufferError here means the stash kept a view

    # And what it kept still serves its consumer, which only ever needs a
    # length and a slice assignment into an int32 destination.
    dst = np.zeros((bs, MAX_COLS), dtype=np.int32)
    for i, row in enumerate(kept):
        dst[i, : len(row)] = row
    assert (dst[0, : len(rows[0])] == np.asarray(rows[0][: len(rows[0])])).all()


def test_int32_asarray_over_a_block_table_is_the_hazard_being_guarded():
    """Positive control: the failure mode above is real, and int64 escapes it.

    Without this, the test above passes just as well against a marshal that
    could never have aliased anything.
    """
    aliased = array.array("i", [1, 2, 3])
    view = np.asarray(aliased, dtype=np.int32)
    assert np.shares_memory(view, np.frombuffer(aliased, dtype=np.int32))
    with pytest.raises(BufferError):
        aliased.append(4)
    del view

    # `np.array` copies, and so does `asarray` at a width that is not the
    # array's own -- which is why the two `aiter_mla` int64 reads are safe.
    for taken in (
        np.array(aliased, dtype=np.int32),
        np.asarray(aliased, dtype=np.int64),
    ):
        assert taken is not None
        aliased.append(4)
