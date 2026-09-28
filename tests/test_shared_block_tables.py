# SPDX-License-Identifier: MIT
"""Page-map reuse is tied to contents, publication success and physical slots."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_engine.block_table_codec import (
    BlockTableDeltaDecoder,
    BlockTableDeltaEncoder,
)
from atom.model_engine.sequence import BlockTable, new_block_table
from atom.utils import CpuGpuBuffer
from atom.utils.block_tables import block_table_state


def buffer():
    return CpuGpuBuffer(4, 8, dtype=torch.int32, device="cpu", pin_memory=False)


def test_versioned_hits_do_not_read_page_payload_or_write_sources(monkeypatch):
    buf = buffer()
    row = new_block_table([3, 5])
    state = block_table_state(buf).prepare([row], pad_to=4, page_limit=8)
    state.publish(4)
    import atom.utils.block_tables as module

    def unexpected(*args, **kwargs):
        raise AssertionError("an unchanged page mapping was read or uploaded")

    monkeypatch.setattr(module, "_int32_row", unexpected)
    monkeypatch.setattr(buf, "copy_to_gpu", unexpected)
    buf.np.flags.writeable = False
    state.prepare([row], pad_to=4, page_limit=8).publish(4)
    assert buf.gpu[0, :2].tolist() == [3, 5]


@pytest.mark.parametrize("versioned", [True, False])
def test_reorder_append_replace_shrink_and_padding(versioned, monkeypatch):
    make = (
        new_block_table
        if versioned
        else lambda values: np.asarray(values, dtype=np.int32)
    )
    a, b = make([3, 5]), make([2, 6])
    buf = buffer()
    copies = []
    original = buf.copy_to_gpu

    def copy(count):
        copies.append(count)
        return original(count)

    monkeypatch.setattr(buf, "copy_to_gpu", copy)
    state = block_table_state(buf)
    for rows, count, dirty in (
        ([a, b], 4, True),
        ([a, b], 4, False),
        ([b, a], 4, True),
        ([a], 4, True),
        ([a], 2, True),
        ([], 2, True),
        ([], 2, False),
    ):
        before = len(copies)
        state.prepare(rows, pad_to=count, page_limit=8).publish(count)
        assert len(copies) - before == dirty
        expected = np.zeros((count, 8), np.int32)
        for i, row in enumerate(rows):
            expected[i, : len(row)] = row
        np.testing.assert_array_equal(buf.gpu[:count].numpy(), expected)
    state.prepare([a], pad_to=4, page_limit=8).publish(4)
    a[0] = 7
    state.prepare([a], pad_to=4, page_limit=8).publish(4)
    assert buf.gpu[0, 0] == 7


def test_decoder_carries_append_lineage_without_mutating_older_batches():
    encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
    row = new_block_table([3, 5])

    def receive():
        return decoder.decode(
            encoder.encode(SimpleNamespace(req_ids=[1], block_tables=[row]))
        ).block_tables[0]

    first = receive()
    assert isinstance(first, BlockTable)
    assert receive() is first
    row.append(7)
    second = receive()
    assert second is not first and second.version == first.version
    assert list(first) == [3, 5] and list(second) == [3, 5, 7]
    row[0] = 2
    third = receive()
    assert third.version != second.version
    assert list(second) == [3, 5, 7]


@pytest.mark.parametrize("appended", [[7], [6, 7]])
def test_append_validates_only_new_ids_and_keeps_source_resizable(appended):
    import array

    buf = buffer()
    row = new_block_table([3, 5])
    state = block_table_state(buf).prepare([row], page_limit=8)
    row.extend(appended)  # No surviving buffer export may prevent this append.
    # Deliberately bypass version tracking to make an already validated prefix
    # unreadable as valid page ids. This is instrumentation, not a legal caller:
    # append preparation must trust that prefix and only read the new suffix.
    array.array.__setitem__(row, 0, -1)
    state.prepare([row], page_limit=8)
    assert buf.cpu[0, : len(row)].tolist() == [3, 5] + appended
    array.array.__setitem__(row, 0, 3)
    before = buf.cpu.clone()
    row.append(8)
    with pytest.raises(ValueError, match="out of range"):
        state.prepare([row], page_limit=8)
    assert torch.equal(buf.cpu, before)
    row.pop()  # Failed validation must also release every export.
    state.prepare([row], page_limit=8)
    with pytest.raises(ValueError, match="out of range"):
        state.prepare([row], page_limit=7)
    row.append(6)


def test_validate_whole_batch_before_writing_and_preserve_previous_snapshot():
    buf = buffer()
    state = block_table_state(buf).prepare([[1], [2]], page_limit=8)
    state.publish(2)
    before = buf.cpu.clone()
    for invalid in ([9], [1] * 9, np.array([2**32 + 1], dtype=np.int64)):
        with pytest.raises(ValueError):
            state.prepare([[3], invalid], page_limit=8)
        assert torch.equal(buf.cpu, before)
        assert torch.equal(buf.gpu[:2], before[:2])


def test_failed_publish_retries_and_destination_replacement_invalidates(monkeypatch):
    buf = buffer()
    state = block_table_state(buf).prepare([[3]], pad_to=2)
    original = buf.copy_to_gpu

    def fail(count):
        raise RuntimeError("enqueue failed")

    monkeypatch.setattr(buf, "copy_to_gpu", fail)
    with pytest.raises(RuntimeError, match="enqueue failed"):
        state.publish(2)
    assert state.published is None
    monkeypatch.setattr(buf, "copy_to_gpu", original)
    state.prepare([[3]], pad_to=2).publish(2)
    buf.gpu = torch.full_like(buf.gpu, -1)
    state.publish(2)
    assert buf.gpu[0].tolist() == [3] + [0] * 7


def test_cpu_preparation_is_not_gpu_publication():
    buf = buffer()
    state = block_table_state(buf)
    state.prepare([[3]], pad_to=2).publish(2)
    state.prepare([[5]], pad_to=2)  # prefill with no GPU table consumer
    assert buf.gpu[0, 0] == 3
    state.prepare([[5]], pad_to=2).publish(2)
    assert buf.gpu[0, 0] == 5


def test_tbo_slices_reuse_row_revisions_and_recover_after_capture(monkeypatch):
    source, left, right = buffer(), buffer(), buffer()
    rows = [new_block_table([i + 1]) for i in range(4)]
    state = block_table_state(source).prepare(rows)
    for dst, start in ((left, 0), (right, 2)):
        state.slice_to(dst, start, 2, pad_to=3).publish(3)

    def unexpected():
        raise AssertionError("unchanged TBO rows must not acquire a source for writing")

    with monkeypatch.context() as patch:
        patch.setattr(block_table_state(left), "_acquire", unexpected)
        patch.setattr(block_table_state(right), "_acquire", unexpected)
        state.prepare(rows)
        state.slice_to(left, 0, 2, pad_to=3).publish(3)
        state.slice_to(right, 2, 2, pad_to=3).publish(3)
    # Capture writes the same physical destination through the common entry.
    block_table_state(left).prepare(np.zeros((3, 8), np.int32)).publish(3)
    state.slice_to(left, 0, 2, pad_to=3).publish(3)
    assert left.gpu[:, 0].tolist() == [1, 2, 0, 0]
    assert right.gpu[:, 0].tolist() == [3, 4, 0, 0]


def test_slot_clones_start_with_independent_cache_state():
    original = buffer()
    state = block_table_state(original).prepare([[3]], pad_to=2)
    state.publish(2)
    clone = original.clone()
    other = block_table_state(clone)
    assert other is not state and other.published is None
    other.prepare([[5]], pad_to=2).publish(2)
    assert original.gpu[0, 0] == 3 and clone.gpu[0, 0] == 5


def test_failed_host_write_cannot_leave_a_false_empty_batch_hit(monkeypatch):
    buf = buffer()
    state = block_table_state(buf).prepare([new_block_table([1])], pad_to=4)

    def partial_write(*args):
        buf.np[0] = 77
        raise RuntimeError("host write failed")

    with monkeypatch.context() as patch:
        patch.setattr(state, "_copy_changes", partial_write)
        with pytest.raises(RuntimeError, match="host write failed"):
            state.prepare([new_block_table([2]), new_block_table([3])], pad_to=4)
    state.prepare([], pad_to=4).publish(4)
    assert buf.gpu.count_nonzero() == 0


def test_group_validation_failure_does_not_publish_a_revision():
    from atom.utils.h2d import PublicationOwner

    buf, other = buffer(), buffer()
    owner = PublicationOwner("cpu")
    table_binding = owner.bind(buf, "block_tables")
    other_binding = owner.bind(other, "other")
    group = owner.group("metadata", [table_binding, other_binding])
    owner.begin()
    row = new_block_table([3])
    state = block_table_state(buf).prepare([row], pad_to=2)
    group.set_count(other, 5)  # exceeds capacity, after the table's valid count
    with pytest.raises(ValueError):
        state.publish(2, group=group)
    assert state.published is None and buf.gpu.count_nonzero() == 0
    group.set_count(other, 2)
    state.publish(2, group=group)
    owner.finish()
    owner.begin()
    state.prepare([row], pad_to=2).publish(2, group=group)
    assert group.counts[group.indices["block_tables"]] is None
    assert buf.gpu[0, 0] == 3
    owner.finish()


@pytest.mark.parametrize("versioned", [True, False])
def test_full_replacement_clears_only_exposed_tails_and_padded_rows(versioned):
    make = new_block_table if versioned else lambda values: np.array(values, np.int32)
    buf = buffer()
    buf.np[:] = -7  # No prior snapshot: even unseen row tails need initialization.
    state = block_table_state(buf)
    for rows, padding in (
        ([make([1] * 8), make([2] * 3)], 2),
        ([make([3] * 2), make([4] * 7)], 4),
        ([make([]), make([5])], 3),
        ([make([6] * 8)], 4),
    ):
        state.prepare(rows, pad_to=padding, page_limit=8)
        expected = np.zeros((padding, 8), np.int32)
        for i, row in enumerate(rows):
            expected[i, : len(row)] = row
        np.testing.assert_array_equal(buf.np[:padding], expected)


def test_late_invalid_versioned_row_and_rejected_acquisition_never_write(monkeypatch):
    buf = buffer()
    state = block_table_state(buf).prepare([new_block_table([1])], pad_to=4)
    before, keys, revision = (
        buf.cpu.clone(),
        (state.versions, state.lengths),
        state.revision,
    )
    rows = [new_block_table([2]), new_block_table([9])]

    def reject():
        assert torch.equal(buf.cpu, before)
        raise RuntimeError("source in flight")

    monkeypatch.setattr(state, "_acquire", reject)
    with pytest.raises(ValueError, match="out of range"):
        state.prepare(rows, pad_to=4, page_limit=8)
    rows[-1][0] = 3
    with pytest.raises(RuntimeError, match="source in flight"):
        state.prepare(rows, pad_to=4, page_limit=8)
    assert (state.versions, state.lengths) == keys and state.revision == revision
    assert torch.equal(buf.cpu, before)
    for row in rows:
        row.append(4)


def test_tbo_inherits_appends_and_tracks_unversioned_content_changes():
    source, target = buffer(), buffer()
    versioned = new_block_table([1])
    unversioned = np.array([2, 3], np.int32)
    state = block_table_state(source).prepare([versioned, unversioned])
    state.slice_to(target, 0, 2, pad_to=4).publish(4)
    versioned.append(4)
    unversioned[0] = 5
    state.prepare([versioned, unversioned])
    state.slice_to(target, 0, 2, pad_to=4).publish(4)
    assert target.gpu[:2, :2].tolist() == [[1, 4], [5, 3]]
    revision = block_table_state(target).revision
    state.prepare([versioned, unversioned]).slice_to(target, 0, 2, pad_to=4)
    assert block_table_state(target).revision == revision


def test_sparse_replacement_leaves_unchanged_rows_untouched():
    # Instrument the destination with sentinels to detect redundant writes.
    # Actual callers must not write behind BlockTableState's snapshot.
    buf = buffer()
    rows = [new_block_table([i + 1] * 8) for i in range(4)]
    state = block_table_state(buf).prepare(rows)
    buf.np[0] = 71
    buf.np[2:] = 72
    rows[1] = new_block_table([6, 7])
    state.prepare(rows)
    assert np.all(buf.np[0] == 71)
    assert np.all(buf.np[2:] == 72)
    assert buf.np[1].tolist() == [6, 7] + [0] * 6


def test_reorder_reuses_validation_by_lineage(monkeypatch):
    import atom.utils.block_tables as module

    buf = buffer()
    a, b = new_block_table([1, 2]), new_block_table([3, 4])
    state = block_table_state(buf).prepare([a, b], page_limit=8)

    def unexpected(*args):
        raise AssertionError("known row contents were scanned after reordering")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_int32_row", unexpected)
        state.prepare([b, a], page_limit=8)
    assert buf.np[:2, :2].tolist() == [[3, 4], [1, 2]]
    before = buf.cpu.clone()
    with pytest.raises(ValueError, match="out of range"):
        state.prepare([b, a], page_limit=4)
    assert torch.equal(buf.cpu, before)


@pytest.mark.parametrize("page_limit", [8, 1 << 40])
def test_negative_page_ids_fail_before_any_write(page_limit):
    buf = buffer()
    state = block_table_state(buf).prepare([new_block_table([1, 2])])
    before = buf.cpu.clone()
    with pytest.raises(ValueError, match="out of range"):
        state.prepare([new_block_table([3, -1])], page_limit=page_limit)
    assert torch.equal(buf.cpu, before)


def test_zero_width_rows_and_padding():
    buf = CpuGpuBuffer(4, 0, dtype=torch.int32, device="cpu", pin_memory=False)
    state = block_table_state(buf)
    state.prepare([new_block_table()], pad_to=4, page_limit=0)
    state.prepare([], pad_to=4).publish(4)
    assert buf.gpu.shape == (4, 0)
