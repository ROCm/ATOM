"""Turning vLLM block ids into native checkpoint ops, and what it refuses.

One vLLM block is one PAGE unit on V4.1 -- both 739,840 B -- so the
translation is a rename. The work is refusing the shapes that are not an
image, and the reason each one needs refusing here rather than downstream:

  * the wrong count and a negative id reach the copier, which validates and
    raises in a worker, after the pool is allocated;
  * a REPEATED id passes every size check the copier makes. Two units would
    address one block and a third would keep whatever it held, which is a
    wrong restore with nothing raised anywhere.
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.v41_state_image import (
    V41ImageExporter,
    restore_op,
    sizing_from_geometry,
    store_op,
)

UNIT, SLOT = 739840, 5276672
LAYOUT = "dsv41-bf16-test"


def sizing():
    return sizing_from_geometry(SimpleNamespace(paged_bytes=UNIT, state_bytes=SLOT))


def blocks(n=8, start=100):
    return list(range(start, start + n))


def test_a_store_op_names_the_units_in_image_order():
    op = store_op(3, blocks(), sizing(), LAYOUT)
    assert op.src_slot == 3
    assert op.unit_ids == tuple(blocks())
    assert op.total_bytes == SLOT
    assert op.layout_id == LAYOUT


def test_a_restore_op_is_the_same_translation_the_other_way():
    """They share it on purpose: a store and a restore that disagreed about
    unit order would read the image back shuffled, byte-exactly wrong."""
    s, r = store_op(3, blocks(), sizing(), LAYOUT), restore_op(3, blocks(), sizing(), LAYOUT)
    assert s.unit_ids == r.unit_ids
    assert (s.total_bytes, s.layout_id) == (r.total_bytes, r.layout_id)


@pytest.mark.parametrize("n", [7, 9, 0, 1])
def test_a_wrong_unit_count_is_refused(n):
    with pytest.raises(ValueError, match="is 8 units"):
        store_op(0, blocks(n), sizing(), LAYOUT)


def test_a_repeated_block_id_is_refused():
    """The one the copier cannot catch: sizes all check out, two units land
    on one block, and a third unit keeps its previous contents."""
    ids = blocks()
    ids[5] = ids[2]
    with pytest.raises(ValueError, match="repeat"):
        store_op(0, ids, sizing(), LAYOUT)


def test_a_negative_block_id_is_refused():
    ids = blocks()
    ids[0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        store_op(0, ids, sizing(), LAYOUT)


class _Copier:
    def __init__(self):
        self.calls = []

    def __call__(self, stores, restores):
        self.calls.append((list(stores), list(restores)))


def exporter(copier=None):
    return V41ImageExporter(sizing(), LAYOUT, copier or _Copier())


def test_export_moves_the_rows_vllm_asked_for():
    c = _Copier()
    e = exporter(c)
    e.export(None, rows=[(1, blocks()), (2, blocks(start=200))])
    (stores, restores), = c.calls
    assert restores == []
    assert [op.src_slot for op in stores] == [1, 2]
    assert e.stored == 2


def test_a_null_row_is_declined_not_written():
    """vLLM's NULL_BLOCK_ID is 0 and a column of -1 means "no checkpoint this
    step". Writing anyway would scribble on block 0, which belongs to
    somebody."""
    c = _Copier()
    e = exporter(c)
    e.export(None, rows=[(1, [0] * 8), (2, [-1] * 8)])
    assert c.calls == []
    assert e.declined_null == 2 and e.stored == 0


def test_a_step_with_nothing_to_store_does_not_call_the_copier():
    """The common case. A copier call with no ops still uploads a descriptor."""
    c = _Copier()
    exporter(c).export(None, rows=[])
    assert c.calls == []


def test_export_without_rows_says_what_is_missing():
    """The checkpoint metadata names block-table columns, not slots, so the
    caller has to supply the pairing; failing loudly beats storing nothing."""
    with pytest.raises(TypeError, match="rows="):
        exporter().export(None)


def test_restore_is_the_mirror_and_counts_what_it_read():
    c = _Copier()
    e = exporter(c)
    assert e.restore([(4, blocks()), (5, [0] * 8)]) == 1
    (stores, restores), = c.calls
    assert stores == []
    assert [op.dst_slot for op in restores] == [4]
    assert e.declined_null == 1


def test_a_partly_null_row_is_declined_whole():
    """Half an image is not a smaller image -- the units are positional, so
    the bytes behind a null one are not the bytes that unit holds."""
    ids = blocks()
    ids[3] = 0
    c = _Copier()
    e = exporter(c)
    e.export(None, rows=[(1, ids)])
    assert c.calls == [] and e.declined_null == 1
