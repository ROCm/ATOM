# SPDX-License-Identifier: MIT
"""A forward slot's CPU page-table snapshot and its published GPU revision."""

import numpy as np

from atom.model_engine.sequence import BlockTable
from atom.utils.h2d import RowPlan


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
        self.buffer.acquire_write()

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

    def _dirty(self, count):
        key = (self.revision, count)
        previous = self.published
        dirty = (
            previous is None or previous[0] is not self.buffer.gpu or previous[1] != key
        )
        return dirty, key

    def stage(self, count, group):
        """Count a changed table into ``group``, whose owner publishes it.

        Omitting a table from a group does not omit its GPU view from metadata.
        The revision is recorded once that publish has succeeded.
        """
        dirty, key = self._dirty(count)
        group.set_count(self.buffer, count if dirty else None)
        if dirty:
            gpu = self.buffer.gpu
            group.after_publish(lambda: setattr(self, "published", (gpu, key)))
        return self.buffer.gpu[:count]

    def publish(self, count):
        """Upload a changed table on its own."""
        dirty, key = self._dirty(count)
        if dirty:
            self.buffer.copy_to_gpu(count)
            self.published = (self.buffer.gpu, key)
        return self.buffer.gpu[:count]


# A table whose pinned buffer is at most this many bytes (or whose rows are not
# whole 16-byte words) is a plain `BlockTableState`: the whole-prefix copy,
# unchanged. A larger buffer gets a `TrackedBlockTableState`, which runs that
# same code while it is not tracked. Plans can pay from about 1 MiB; this bound
# is where they pay clearly. With columns = max_model_len / page size, a
# long-context model's decode table passes it at a modest batch (128K context,
# 16-token pages: above 128 rows).
WHOLE_PREFIX_BYTES = 4 << 20
# A step whose `prepare` changed more than this fraction of the published rows
# is busy: a tracked table goes up whole on it (the rows stay recorded), an
# untracked one does not enter the tracking.
PLAN_MAX_CHANGED_FRACTION = 0.5
# A tracked table leaves the tracking after this many busy steps in a row (more
# than half its rows change every step) or this many stages in a row at a count
# within `WHOLE_PREFIX_BYTES` (a batch that shrank for good). Fewer of them,
# such as a prefill staging the same buffer between decode steps, stay
# recorded.
BUSY_STEPS_TO_LEAVE = 8
SMALL_STAGES_TO_LEAVE = 8


class TrackedBlockTableState(BlockTableState):
    """A block table whose buffer can outgrow `WHOLE_PREFIX_BYTES`. On top of
    the plain state it can track, per row, what the destination holds, so a
    packed publish sends only the rows the destination lacks, each only as
    wide as needed (`RowPlan`).

    Untracked, it runs `BlockTableState`'s own `prepare`, `publish` and
    `_copy_changes`; `stage` alone decides, and at a count within the bound it
    goes straight to the plain one. Above the bound it counts the rows each
    `prepare` writes (`_counting_copy_changes`), to tell a busy step. Tracked,
    the instance carries the tracked `prepare`, `publish` and `_copy_changes`
    (`_track` installs them, `_untrack` removes them).

    Per row: the write that last changed it on the host (`row_writes`), the
    write the destination last received (`gpu_writes`, -1: unknown) and a
    bound past which the destination row holds zeros (`gpu_bound`).
    """

    def __init__(self, buffer):
        super().__init__(buffer)
        rows, columns = self.cpu.shape
        self.plain_rows = WHOLE_PREFIX_BYTES // (columns * self.cpu.itemsize)
        self.row_writes = np.zeros(rows, dtype=np.int64)
        self.writes = 0
        self.gpu_writes = np.full(rows, -1, dtype=np.int64)
        self.gpu_bound = np.full(rows, columns, dtype=np.int64)
        self.gpu_tracked = None
        self.counting = False
        # Stages and direct publishes: only the latest stage's publish callback
        # records rows as received (`_stage_tracked`).
        self.stages = 0
        self.changed_rows = 0  # rows the last `prepare` wrote, while counted
        self.busy_steps = 0
        self.small_stages = 0
        self.planned_group = None  # the group holding this table's pending plan

    def _counting_copy_changes(self, rows, changes, columns, bulk_clear):
        """`BlockTableState._copy_changes`, counting the rows written."""
        self.changed_rows = len(changes)
        BlockTableState._copy_changes(self, rows, changes, columns, bulk_clear)

    def _tracked_copy_changes(self, rows, changes, columns, bulk_clear):
        """`BlockTableState._copy_changes`, counting the rows written and
        marking them before the write, so a failed write stays dirty."""
        self.changed_rows = len(changes)
        self.writes += 1
        self.row_writes[[i for i, _, _, _ in changes]] = self.writes
        BlockTableState._copy_changes(self, rows, changes, columns, bulk_clear)

    def _tracked_prepare(self, rows, *, pad_to=None, page_limit=None, _keys=None):
        """`BlockTableState.prepare` while tracked: the written rows are marked
        in `_tracked_copy_changes`; then the tail rows it cleared, and the bound
        of rows whose upload is unconfirmed widens to the new host bytes."""
        old_n, old_zero_to, revision = len(self.lengths), self.zero_to, self.revision
        BlockTableState.prepare(
            self, rows, pad_to=pad_to, page_limit=page_limit, _keys=_keys
        )
        if self.revision == revision:
            return self  # nothing written
        n = len(self.lengths)
        end = n if pad_to is None else pad_to
        if end > n and (n != old_n or end > old_zero_to):
            self.row_writes[n:end] = self.writes  # the tail it cleared
        # A row whose upload is not confirmed (`_sending`) may still take these
        # host bytes: a staged publish reads the host when it runs, and a
        # failed one may have written part of them. Its destination bound
        # covers them.
        unknown = self.gpu_writes[:end] < 0
        if unknown.any():
            np.maximum(
                self.gpu_bound[:end],
                np.where(unknown, self._row_bounds(end), 0),
                out=self.gpu_bound[:end],
            )
        return self

    def _tracked_publish(self, count):
        """A direct `publish` writes the whole prefix behind the tracking's
        back: stop the tracking, then the plain publish."""
        self._untrack()
        return BlockTableState.publish(self, count)

    def _count_writes(self, on):
        """Count the rows each `prepare` writes (`_counting_copy_changes`)."""
        if on:
            self._copy_changes = self._counting_copy_changes
        else:
            del self._copy_changes
        self.counting = on
        self.changed_rows = 0

    def _row_bounds(self, count):
        """Per row below ``count``: a bound past which the host row is zero.
        Prepared rows hold their pages, the padding rows zeros; any other
        row (never prepared) could hold anything."""
        bounds = np.full(count, self.cpu.shape[1], dtype=np.int64)
        n = min(len(self.lengths), count)
        bounds[:n] = self.lengths[:n]
        bounds[n : min(self.zero_to, count)] = 0
        return bounds

    def _track(self, gpu):
        if self.gpu_tracked is not gpu:
            # A new destination holds nothing known.
            self.gpu_writes.fill(-1)
            self.gpu_bound.fill(self.cpu.shape[1])
            self.gpu_tracked = gpu
            self.prepare = self._tracked_prepare
            self.publish = self._tracked_publish
            self._copy_changes = self._tracked_copy_changes
            self.counting = True

    def _untrack(self):
        """Stop the tracking: back to the plain methods, and a later tracked
        stage starts from unknown (`_track`). Supersedes a pending stage's
        record, and withdraws its plan from the group."""
        self.gpu_tracked = None
        del self.prepare, self.publish, self._copy_changes
        self.counting = False
        self.busy_steps = self.small_stages = 0
        self.stages += 1
        if self.planned_group is not None:
            self.planned_group.set_row_plan(self.buffer, None)
            self.planned_group = None

    def _row_plan(self, count, bounds):
        """The rows below ``count`` the destination does not hold yet, and the
        leading columns of each that differ from it.

        A row the destination holds is identical in every byte. Sending a
        changed row's first ``width`` columns makes it so too: past ``width``
        both sides are zero -- the host row past its pages, the destination
        past its bound. Rows above ``count`` stay as they are, as with the
        whole-prefix copy. ``bounds`` is `_row_bounds(count)`."""
        rows = np.flatnonzero(self.gpu_writes[:count] != self.row_writes[:count])
        if not len(rows):
            return rows, 0
        width = max(int(bounds[rows].max()), int(self.gpu_bound[rows].max()))
        if width == 0:
            return rows[:0], 0  # zeros on both sides: nothing differs
        # Whole 16-byte words, as the packed word copy moves them.
        return rows.astype(np.int32), min(-(-width // 4) * 4, self.cpu.shape[1])

    def _sending(self, gpu, count, bounds):
        """Before an upload of rows below ``count``: until it has succeeded
        (`_published_rows`), its rows are unknown and may hold the host's
        bytes, so their destination bound widens to the host's. After a failed
        or partial upload every row below ``count`` is resent, as wide as the
        larger of the host's bound and that widened bound; only a destination
        that is not tracked at all gets every row at full width (`_track`).
        ``bounds`` is `_row_bounds(count)`."""
        self._track(gpu)
        self.gpu_writes[:count] = -1
        np.maximum(self.gpu_bound[:count], bounds, out=self.gpu_bound[:count])

    def _published_rows(self, gpu, count, bounds):
        """The destination took every row below ``count`` (planned or whole);
        ``bounds`` is `_row_bounds(count)` of the table it took."""
        self._track(gpu)
        self.gpu_writes[:count] = self.row_writes[:count]
        self.gpu_bound[:count] = bounds

    def stage(self, count, group):
        """`BlockTableState.stage`, tracked where that pays.

        Untracked, a count within `WHOLE_PREFIX_BYTES` runs the plain stage
        straight away (and stops counting written rows). Above it the table
        counts the rows each `prepare` writes; a step with no count yet or
        none written, a busy step (`PLAN_MAX_CHANGED_FRACTION`) and a group
        that is not packed stay plain, any other step enters the tracking
        (from unknown: every row at full width once).

        Tracked, a step plans unless it is busy or at a small count; those go
        up whole and stay recorded, and `BUSY_STEPS_TO_LEAVE` busy steps or
        `SMALL_STAGES_TO_LEAVE` small stages in a row leave the tracking (only
        steps that count as busy or small, not a whole copy of a wide plan). A
        stage into a group that is not packed, or a direct `publish`, stops
        the tracking at once.

        Contract: the host table is not prepared again between this stage and
        its group's publish (the upload reads the host when it runs; only then
        does the destination end byte-identical to the whole-prefix copy), and
        a stage's group must publish before any other stage or publish of this
        table.
        """
        if count <= self.plain_rows and self.gpu_tracked is None:
            if self.counting:
                self._count_writes(False)
            return BlockTableState.stage(self, count, group)
        changed, self.changed_rows = self.changed_rows, 0
        if self.gpu_tracked is None:
            if not self.counting:
                self._count_writes(True)  # from the next prepare on
                return BlockTableState.stage(self, count, group)
            if (
                not changed
                or changed > PLAN_MAX_CHANGED_FRACTION * count
                or group.transport != "packed"
            ):
                return BlockTableState.stage(self, count, group)
            return self._stage_tracked(count, group, whole=True)
        if group.transport != "packed":
            self._untrack()
            return BlockTableState.stage(self, count, group)
        if count <= self.plain_rows:
            self.small_stages += 1
            if self.small_stages >= SMALL_STAGES_TO_LEAVE:
                self._untrack()
                return BlockTableState.stage(self, count, group)
            return self._stage_tracked(count, group, whole=True)
        self.small_stages = 0
        if changed > PLAN_MAX_CHANGED_FRACTION * count:
            self.busy_steps += 1
            if self.busy_steps >= BUSY_STEPS_TO_LEAVE:
                self._untrack()
                return BlockTableState.stage(self, count, group)
            return self._stage_tracked(count, group, whole=True)
        self.busy_steps = 0
        return self._stage_tracked(count, group, whole=False)

    def _stage_tracked(self, count, group, *, whole):
        """A tracked stage into a packed group: a plan unless ``whole`` (entering
        the tracking, a busy or small step) or the plan would cover more than
        `PLAN_MAX_CHANGED_FRACTION` of the rows, else the whole prefix; either
        way the rows sent are recorded as received once the publish succeeds."""
        dirty, key = self._dirty(count)
        self.stages += 1
        stage = self.stages
        plan = None
        if dirty:
            gpu = self.buffer.gpu
            bounds = self._row_bounds(count)
            if not whole:
                rows, width = self._row_plan(count, bounds)
                if not len(rows):
                    dirty = False  # the destination already holds every row
                elif len(rows) <= PLAN_MAX_CHANGED_FRACTION * count:
                    plan = RowPlan(rows, width * self.cpu.itemsize)
            if dirty:
                self._sending(gpu, count, bounds)

            def published():
                self.planned_group = None
                if self.revision != key[0] or self.stages != stage:
                    # The host table was prepared again, or staged or published
                    # again, after this stage: the rows stay unknown
                    # (`_sending`) and are sent again. `prepare` already
                    # widened their destination bound to the host bytes.
                    return
                self._published_rows(gpu, count, bounds)
                self.published = (gpu, key)

            group.after_publish(published)
        group.set_count(self.buffer, count if dirty else None)
        if plan is not None or self.planned_group is group:
            group.set_row_plan(self.buffer, plan)
        self.planned_group = group if plan is not None else None
        return self.buffer.gpu[:count]


def block_table_state(buffer):
    """Get the state belonging to this physical forward buffer/PP slot."""
    state = getattr(buffer, "_block_table", None)
    if state is None or state.cpu is not buffer.np:
        # A buffer that cannot outgrow the plain bound, or whose rows are not
        # whole 16-byte words (which a plan moves), never tracks.
        table = buffer.np
        plain = (
            table.nbytes <= WHOLE_PREFIX_BYTES or table.shape[-1] * table.itemsize % 16
        )
        cls = BlockTableState if plain else TrackedBlockTableState
        state = buffer._block_table = cls(buffer)
    return state
