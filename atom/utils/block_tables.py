# SPDX-License-Identifier: MIT
"""A forward slot's CPU page-table snapshot and its published GPU revision."""

import numpy as np

from atom.model_engine.sequence import BlockTable


def _int32_row(row):
    values = np.asarray(row)
    if values.ndim != 1:
        raise ValueError("Block-table rows must be one-dimensional")
    if values.dtype != np.int32:
        if values.size and (
            values.dtype.kind not in "iu"
            or values.min() < 0
            or values.max() > np.iinfo(np.int32).max
        ):
            raise ValueError("PAGE ids must be nonnegative int32 integers")
        values = values.astype(np.int32)
    return np.ascontiguousarray(values)


class BlockTableState:
    """Keep row versions and lengths beside the pinned snapshot, not its payload.

    All writers, including capture and padding, must use this object. Versions
    and lengths identify rows without payload snapshots or per-row key tuples.
    """

    def __init__(self, buffer):
        self.buffer = buffer
        self.cpu = buffer.np
        if (
            self.cpu.dtype != np.int32
            or self.cpu.ndim != 2
            or not self.cpu.flags.c_contiguous
        ):
            raise TypeError(
                "Block tables require a contiguous two-dimensional int32 buffer"
            )
        self.bytes = self.cpu.view(np.uint8)
        self.flat = memoryview(self.cpu).cast("B").cast("i") if self.cpu.size else ()
        self.versions = []
        self.lengths = []
        self.zero_to = 0
        self.revision = 0
        self.published = None
        self.page_limit = None
        self.checked = None

    def _acquire(self):
        binding = getattr(self.buffer, "_publication", None)
        if binding is not None:
            binding.acquire_write()

    def _begin_write(self):
        self._acquire()
        self.revision += 1
        self.published = None
        # A failed host write must not leave reusable CPU keys or padding.
        self.versions = []
        self.lengths = []
        self.zero_to = 0

    def _normalize(self, rows, versions, lengths):
        rows = list(rows)
        for i, (row, version, length) in enumerate(zip(rows, versions, lengths)):
            if version is not None:
                continue
            rows[i] = values = _int32_row(row)
            unchanged = (
                i < len(self.lengths)
                and length == self.lengths[i]
                and np.array_equal(values, self.cpu[i, :length])
            )
            versions[i] = self.versions[i] if unchanged else object()
        return rows

    def _validate_range(self, rows, versions, lengths, page_limit):
        # Validation belongs to an append lineage, regardless of its row index.
        # Reordering requests must not re-read their already checked pages.
        checked = self.checked if page_limit == self.page_limit else {}
        limit = min(page_limit, 1 << 31)
        for row, version, length in zip(rows, versions, lengths):
            start = checked.get(version, 0)
            if length <= start:
                continue
            if length - start == 1:
                invalid = not 0 <= row[start] < page_limit
            else:
                tail = _int32_row(row)[start:]
                # Signed negatives sort after every valid PAGE id as uint32,
                # so one reduction checks both bounds without a payload copy.
                invalid = tail.view(np.uint32).max() >= limit
            if invalid:
                raise ValueError("Request PAGE table is incomplete or out of range")
        return dict(zip(versions, lengths))

    def prepare(self, rows, *, pad_to=None, page_limit=None, _keys=None):
        n = len(rows)
        capacity, columns = self.cpu.shape
        end = n if pad_to is None else pad_to
        if not 0 <= n <= end <= capacity:
            raise ValueError("Block-table rows exceed the declared capacity")
        if _keys is None:
            versions = [
                row.version if isinstance(row, BlockTable) else None for row in rows
            ]
            lengths = list(map(len, rows))
        else:
            versions, lengths = _keys
        if (
            versions == self.versions
            and lengths == self.lengths
            and end <= self.zero_to
            and page_limit == self.page_limit
        ):
            return self
        if max(lengths, default=0) > columns:
            raise ValueError("Block-table row exceeds the destination columns")
        if None in versions:
            rows = self._normalize(rows, versions, lengths)
        checked = (
            self._validate_range(rows, versions, lengths, page_limit)
            if page_limit is not None
            else None
        )

        old_versions, old_lengths = self.versions, self.lengths
        old_n = len(old_lengths)
        if n > old_n:
            old_versions = old_versions + [None] * (n - old_n)
            old_lengths = old_lengths + [columns] * (n - old_n)
        changes = [
            (i, length, old if version == previous and length >= old else 0, old)
            for i, (version, previous, length, old) in enumerate(
                zip(versions, old_versions, lengths, old_lengths)
            )
            if version != previous or length != old
        ]
        clear_tail = end > n and (n != old_n or end > self.zero_to)
        zero_to = max(end, self.zero_to) if n == old_n else end
        if changes or clear_tail:
            self._begin_write()  # all validation precedes the first write
            # A fully replaced batch can clear exposed tails in one pass.
            # Partial updates retain every unaffected row and copied prefix.
            bulk_clear = (
                len(changes) == n
                and lengths != old_lengths
                and min(lengths, default=columns) < columns
                and not any(start for _, _, start, _ in changes)
            )
            if bulk_clear:
                self.bytes[:n] = 0
            self._copy_changes(rows, changes, columns, bulk_clear)
            if clear_tail:
                self.bytes[n:end] = 0
        self.versions, self.lengths = versions, lengths
        self.zero_to = zero_to
        self.page_limit = page_limit
        self.checked = checked
        return self

    def _copy_changes(self, rows, changes, columns, bulk_clear):
        flat = self.flat
        for i, length, start, old in changes:
            if not bulk_clear and length < old:
                self.bytes[i, length * 4 : old * 4] = 0
            base = i * columns
            if length - start == 1:
                flat[base + start] = rows[i][start]
            elif length > start:
                flat[base + start : base + length] = (
                    rows[i][start:] if start else rows[i]
                )

    def pad(self, scheduled_bs, running_bs):
        """Finalize the padded range before publication, including draft rows."""
        if not 0 <= scheduled_bs <= running_bs <= self.cpu.shape[0]:
            raise ValueError("Invalid block-table padding range")
        if scheduled_bs != len(self.lengths) or running_bs > self.zero_to:
            versions = self.versions[:scheduled_bs]
            lengths = self.lengths[:scheduled_bs]
            self._begin_write()
            self.bytes[scheduled_bs:running_bs] = 0
            self.versions, self.lengths = versions, lengths
            self.zero_to = running_bs
        return self

    def slice_to(self, buffer, start, count, *, pad_to):
        """Prepare a TBO destination using the source snapshot's row revisions."""
        stop = start + count
        if stop > len(self.lengths):
            # Independent callers can supply an already packed CPU table.
            return block_table_state(buffer).prepare(
                self.cpu[start:stop], pad_to=pad_to
            )
        rows = [self.cpu[i, : self.lengths[i]] for i in range(start, stop)]
        keys = self.versions[start:stop], self.lengths[start:stop]
        return block_table_state(buffer).prepare(rows, pad_to=pad_to, _keys=keys)

    def publish(self, count, *, group=None):
        """Publish a changed table, optionally with the caller's metadata group.

        Omitting a table from a group does not omit its GPU view from metadata.
        Record the revision only after the entire publication succeeds.
        """
        if count is None:
            if group is not None:
                group.publish(group.counts)
            return None
        key = (self.revision, count)
        previous = self.published
        dirty = (
            previous is None or previous[0] is not self.buffer.gpu or previous[1] != key
        )
        if group is not None:
            group.set_count(self.buffer, count if dirty else None)
            group.publish(group.counts)
        elif dirty:
            self.buffer.copy_to_gpu(count)
        if dirty:
            self.published = (self.buffer.gpu, key)
        return self.buffer.gpu[:count]


def block_table_state(buffer):
    """Get the state belonging to this physical forward buffer/PP slot."""
    state = getattr(buffer, "_block_table", None)
    if state is None or state.cpu is not buffer.np:
        state = buffer._block_table = BlockTableState(buffer)
    return state
