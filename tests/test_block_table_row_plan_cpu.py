# SPDX-License-Identifier: MIT
"""Row tracking of packed block-table publishes, without a GPU.

A fake group stands in for the packed transport: at publish it applies the
counted table to a numpy "device" the way `PackedCopy` would (the planned
rows' leading bytes, or the whole prefix), reading the host table at publish
time, and runs the `after_publish` callbacks only when it succeeds. The device
prefix must equal the host's after every successful publish."""

import functools
import random

import numpy as np
import pytest
import torch

from atom.model_engine.sequence import new_block_table
from atom.utils import CpuGpuBuffer
from atom.utils.block_tables import block_table_state
from atom.utils.h2d import PublicationOwner, RowPlan, gather_rows
from tests.block_table_churn import decode_churn, ids


@pytest.fixture(autouse=True)
def _plan_every_table(monkeypatch):
    """These tables are small and often change most rows; plan them anyway
    (`WHOLE_PREFIX_BYTES`, `PLAN_MAX_CHANGED_FRACTION`)."""
    from atom.utils import block_tables

    monkeypatch.setattr(block_tables, "WHOLE_PREFIX_BYTES", 0)
    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 1.0)


class _Group:
    """Records what `stage` counts; `publish` emulates the packed transport."""

    transport = "packed"

    def __init__(self, table, device):
        self.table, self.device = table, device
        self.count = self.plan = None
        self.callbacks = []
        self.sent = 0

    def set_count(self, buffer, count):
        assert buffer is self.table
        self.count = count

    def set_row_plan(self, buffer, row_plan):
        assert buffer is self.table
        self.plan = row_plan

    def after_publish(self, callback):
        self.callbacks.append(callback)

    def publish(self, mode="ok", rng=None):
        count, plan = self.count, self.plan
        host = self.table.np
        if count and mode != "fail":
            if plan is None:
                rows = np.arange(count)
                width = host.shape[1]
            else:
                rows, width = plan.rows, plan.row_bytes // host.itemsize
            if mode == "half":  # an interrupted upload: some rows landed
                rows = rows[rng.random(len(rows)) < 0.5]
            self.device[rows, :width] = host[rows, :width]
            self.sent += len(rows) * width
        self.count = self.plan = None
        callbacks, self.callbacks = self.callbacks, []
        if mode != "ok":
            return False
        for callback in callbacks:
            callback()
        return True


def _setup(rows, columns):
    table = CpuGpuBuffer(
        rows, columns, dtype=torch.int32, device="cpu", pin_memory=False
    )
    device = np.full((rows, columns), -7, dtype=np.int32)  # unknown contents
    state = block_table_state(table)
    if hasattr(state, "_count_writes"):
        state._count_writes(True)  # counted from the first prepare
    return table, device, _Group(table, device), state


@pytest.mark.parametrize("real_rules", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_churn_matches_the_host_after_every_successful_publish(
    monkeypatch, seed, real_rules
):
    """Appends, finishes, admissions, preemption, rollback, reorders, prefill
    prepares that are never published, prepares between stage and publish, and
    failed or half-applied publishes."""
    columns, capacity = 96, 48
    if real_rules:  # the counts cross the whole-prefix size; changed-row rule on
        from atom.utils import block_tables

        monkeypatch.setattr(block_tables, "WHOLE_PREFIX_BYTES", 20 * columns * 4)
        monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    table, device, group, state = _setup(capacity, columns)
    running = [
        new_block_table(ids(rng, pyrng.randint(1, 20), 1 << 20)) for _ in range(20)
    ]
    buckets = [8, 16, 24, 32, 40, 48]
    full = 0
    for step in range(600):
        kind = pyrng.random()
        if kind < 0.05:  # a prefill prepares the same buffer and never publishes
            batch = [new_block_table(ids(rng, pyrng.randint(1, columns), 1 << 20))]
            state.prepare(batch, pad_to=1)
            continue
        decode_churn(
            running,
            rng,
            pyrng,
            step,
            columns=columns,
            max_running=40,
            page_limit=1 << 20,
        )
        count = next(b for b in buckets if b >= len(running))
        state.prepare(running, pad_to=count)
        dirty = state._dirty(count)[0]
        state.stage(count, group)
        # Prepared again before the publish: that upload may leave rows it did
        # not plan stale; they must not be recorded as held, so the check
        # below holds again from the next publish on.
        raced = pyrng.random() < 0.05
        if raced:
            t = pyrng.choice(running)
            if len(t) < columns:
                t.append(int(rng.integers(0, 1 << 20)))
            state.prepare(running, pad_to=count)
        mode = pyrng.choices(["ok", "fail", "half"], [0.9, 0.05, 0.05])[0]
        if group.publish(mode, rng) and not raced:
            np.testing.assert_array_equal(
                device[:count], table.np[:count], err_msg=f"{seed}/{step}"
            )
            full += dirty * count * columns
    assert group.sent < full


@pytest.mark.parametrize("outcome", ["ok", "fail", "half"])
def test_a_prepare_between_stage_and_publish_is_not_recorded_as_received(outcome):
    """The upload reads the host at publish time. A row prepared after the plan
    was built and not in it must not be marked as held, and the next publish
    sends it. A planned row that grows in between lands on the device past
    its stage-time bound, whether the publish succeeds, fails or is cut short;
    when that row later holds a shorter request at a smaller count, the plan
    must still clear those pages."""
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([1, 2, 3]), new_block_table([4, 5])]
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.publish()
    rows[0].append(9)
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.plan.rows.tolist() == [0]
    rows[1][0] = 77  # a second prepare before the publish
    state.prepare(rows, pad_to=4)
    assert group.publish()
    assert device[1, 0] == 4  # the plan did not cover row 1
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.count == 4 and 1 in group.plan.rows.tolist()
    assert group.publish()
    np.testing.assert_array_equal(device[:4], table.np[:4])

    # A planned row grows during the race, then the count narrows.
    rows = [new_block_table([1, 2]), new_block_table(list(range(100, 120)))]
    state.prepare(rows, pad_to=2).stage(2, group)
    assert group.publish()
    rows[0].append(3)
    rows[1].append(120)
    state.prepare(rows, pad_to=2).stage(2, group)
    plan = group.plan
    assert plan.rows.tolist() == [0, 1] and plan.row_bytes == 24 * 4
    rows[0].extend(range(4, 12))  # row 0 grows to 11 pages before the publish
    state.prepare(rows, pad_to=2)
    if outcome == "half":  # the cut-short upload landed row 0 only
        device[0, :24] = table.np[0, :24]
        group.publish("fail")
    else:
        assert group.publish(outcome) == (outcome == "ok")
    if outcome != "fail":
        assert device[0, :11].tolist() == list(range(1, 12))
    state.prepare([new_block_table([500])], pad_to=1).stage(1, group)
    assert group.publish()
    np.testing.assert_array_equal(device[:1], table.np[:1])


@pytest.mark.parametrize("second", ["narrower", "nothing", "direct"])
def test_a_superseded_stage_records_nothing(second):
    """Staging the same table again (a narrower count, or one with nothing to
    send) or publishing it directly before the first stage's group publishes:
    the first stage's callback must not record rows the device never got."""
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([1, 2]) for _ in range(4)]
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.publish()
    for r in rows:
        r.append(9)
    state.prepare(rows, pad_to=4).stage(4, group)
    if second == "direct":
        state.publish(2)  # whole prefix [:2] straight to `table.gpu`
        device[:2] = table.gpu[:2].numpy()
    elif second == "nothing":
        state.stage(4, group)  # nothing new to send on top of the first
    else:
        state.stage(2, group)
    assert group.publish()
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.publish()
    np.testing.assert_array_equal(device[:4], table.np[:4])


def test_gather_rows_reads_only_the_planned_rows():
    """The planned rows of a strided view are gathered without a temporary
    copy of the whole table (what `np.take(..., out=)` makes)."""
    import tracemalloc

    rng = np.random.default_rng(0)
    table = rng.integers(0, 255, size=(4096, 1024), dtype=np.uint8)
    source = table[:, :512]  # a plan narrower than the row: strided
    rows = np.sort(rng.choice(4096, size=12, replace=False)).astype(np.int32)
    out = np.empty((len(rows), 512), dtype=np.uint8)
    tracemalloc.start()
    gather_rows(out, source, rows)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    np.testing.assert_array_equal(out, table[rows, :512])
    assert peak < source.nbytes // 16
    run = np.arange(100, 140, dtype=np.int32)
    out = np.empty((len(run), 512), dtype=np.uint8)
    gather_rows(out, source, run)
    np.testing.assert_array_equal(out, table[100:140, :512])


def test_small_tables_go_up_whole_and_restart_tracking(monkeypatch):
    """Untracked, a prefix of at most `WHOLE_PREFIX_BYTES` takes the plain
    path; the first stage past it enters the tracking from unknown (every row,
    full width), later ones plan only changed rows; back at a small count the
    table goes up whole but stays recorded until `SMALL_STAGES_TO_LEAVE`
    such stages in a row leave the tracking. The device always matches the
    host."""
    from atom.utils import block_tables

    columns = 64
    monkeypatch.setattr(block_tables, "WHOLE_PREFIX_BYTES", 8 * columns * 4)
    table, device, group, state = _setup(32, columns)
    rows = [new_block_table([i + 1, i + 2]) for i in range(8)]
    for _ in range(2):  # small: the whole prefix, no plan, untracked
        state.prepare(rows, pad_to=8).stage(8, group)
        assert group.count == 8 and group.plan is None
        assert state.gpu_tracked is None
        assert group.publish()
        np.testing.assert_array_equal(device[:8], table.np[:8])
        rows[0].append(50)
    rows += [new_block_table([100 + i]) for i in range(16)]
    state.prepare(rows, pad_to=24).stage(24, group)  # large: counting starts
    assert group.count == 24 and state.gpu_tracked is None
    assert group.publish()
    rows[2].append(66)
    state.prepare(rows, pad_to=24).stage(24, group)  # tracking starts
    assert group.count == 24 and group.plan is None  # every row, full width
    assert state.gpu_tracked is table.gpu
    assert group.publish()
    np.testing.assert_array_equal(device[:24], table.np[:24])
    rows[3].append(77)
    state.prepare(rows, pad_to=24).stage(24, group)
    assert group.plan is not None and group.plan.rows.tolist() == [3]
    assert group.publish()
    np.testing.assert_array_equal(device[:24], table.np[:24])
    rows = rows[:8]  # back to small: whole but recorded, then leave
    monkeypatch.setattr(block_tables, "SMALL_STAGES_TO_LEAVE", 2)
    for tracked in (True, False):
        rows[5].append(88)
        state.prepare(rows, pad_to=8).stage(8, group)
        assert group.plan is None
        assert (state.gpu_tracked is table.gpu) == tracked
        assert group.publish()
        np.testing.assert_array_equal(device[:8], table.np[:8])


def test_a_table_whose_columns_are_not_whole_words_gets_no_plan():
    table, device, group, state = _setup(4, 6)
    rows = [new_block_table([1, 2]), new_block_table([3])]
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.publish()
    rows[0].extend([4, 5, 6])  # five pages: the width rounds past six columns
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.count == 4 and group.plan is None
    assert group.publish()
    np.testing.assert_array_equal(device[:4], table.np[:4])
    rows[1].append(7)  # not whole 16-byte rows: never a plan
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.count == 4 and group.plan is None
    assert type(state).__name__ == "BlockTableState"  # never tracks
    assert group.publish()
    np.testing.assert_array_equal(device[:4], table.np[:4])


def test_a_step_that_changes_most_rows_goes_up_whole(monkeypatch):
    """With the real rules: a step that changed more than half of the rows
    goes up whole but stays recorded, so the next quieter step plans (a
    changed fraction swinging around one half keeps planning); two such steps
    in a row leave the tracking, and a step that changed at most half of the
    rows enters it again (from unknown), then plans single rows."""
    from atom.utils import block_tables

    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    monkeypatch.setattr(block_tables, "BUSY_STEPS_TO_LEAVE", 2)
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([i + 1]) for i in range(8)]

    def step(changed):
        for i in changed:
            rows[i].append(100 + i)
        state.prepare(rows, pad_to=8).stage(8, group)
        plan = group.plan
        assert group.publish()
        np.testing.assert_array_equal(device[:8], table.np[:8])
        return plan

    assert step([]) is None and state.gpu_tracked is None  # 8 new rows
    assert step([0]) is None and state.gpu_tracked is table.gpu  # enter: whole
    assert step([1]).rows.tolist() == [1]
    for _ in range(3):  # 6 of 8, then 1 of 8: whole but recorded, then a plan
        assert step(range(6)) is None and state.gpu_tracked is table.gpu
        assert step([2]).rows.tolist() == [2]
    assert step(range(6)) is None and state.gpu_tracked is table.gpu
    assert step(range(6)) is None and state.gpu_tracked is None  # 2nd: leave
    assert step(range(6)) is None and state.gpu_tracked is None  # stays out
    assert step([3, 4, 5]) is None and state.gpu_tracked is table.gpu  # enter
    assert step([5]).rows.tolist() == [5]


def _tracked_step(state, group, table, device, rows, changed, count):
    for i in changed:
        rows[i].append(100 + i)
    state.prepare(rows, pad_to=count).stage(count, group)
    plan = group.plan
    assert group.publish()
    np.testing.assert_array_equal(device[:count], table.np[:count])
    return plan


def test_an_idle_step_resets_the_changed_row_count(monkeypatch):
    """A busy step, then a step that writes nothing, then a one-row change:
    the idle step must not count as busy, so the table stays tracked and the
    one-row change is a one-row plan."""
    from atom.utils import block_tables

    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    monkeypatch.setattr(block_tables, "BUSY_STEPS_TO_LEAVE", 2)
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([i + 1]) for i in range(8)]
    step = functools.partial(_tracked_step, state, group, table, device, rows)
    step([], 8)  # new rows: busy, untracked
    step([0], 8)  # enter
    assert step(range(6), 8) is None and state.gpu_tracked is table.gpu
    assert step([], 8) is None  # idle: nothing written
    assert state.gpu_tracked is table.gpu
    assert step([4], 8).rows.tolist() == [4]


def test_a_busy_busy_quiet_cycle_keeps_planning(monkeypatch):
    """Two busy steps then a quiet one, repeated: with the default
    `BUSY_STEPS_TO_LEAVE` the table stays tracked and every quiet step plans;
    a long run of busy steps still leaves the tracking."""
    from atom.utils import block_tables

    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    table, device, group, state = _setup(10, 64)
    rows = [new_block_table([i + 1]) for i in range(10)]
    step = functools.partial(_tracked_step, state, group, table, device, rows)
    step([], 10)
    step([0], 10)  # enter
    for k in range(4):
        assert step(range(6), 10) is None
        assert step(range(1, 7), 10) is None
        assert step([k], 10).rows.tolist() == [k]
    for _ in range(block_tables.BUSY_STEPS_TO_LEAVE):
        step(range(10), 10)
    assert state.gpu_tracked is None


def test_a_small_prefill_stage_between_decode_steps_keeps_the_tracking(
    monkeypatch,
):
    """With the real changed-row rule: a prefill staging the same buffer at a
    small count between decode steps goes up whole but recorded; the decode
    step right after it rewrites every row (busy, whole, still recorded) and
    the next one plans again. Only `SMALL_STAGES_TO_LEAVE` small stages in a
    row leave the tracking."""
    from atom.utils import block_tables

    columns = 64
    monkeypatch.setattr(block_tables, "WHOLE_PREFIX_BYTES", 4 * columns * 4)
    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    monkeypatch.setattr(block_tables, "SMALL_STAGES_TO_LEAVE", 3)
    table, device, group, state = _setup(16, columns)
    rows = [new_block_table([i + 1]) for i in range(12)]
    step = functools.partial(_tracked_step, state, group, table, device, rows)
    step([], 12)  # new rows, then enter
    step([0], 12)
    for k in range(4):
        prefill = [new_block_table([200 + k, 201]), rows[3]]
        state.prepare(prefill, pad_to=2).stage(2, group)  # a small stage
        assert group.publish()
        np.testing.assert_array_equal(device[:2], table.np[:2])
        assert state.gpu_tracked is table.gpu
        assert step([5 + k], 12) is None  # every row rewritten: busy
        assert state.gpu_tracked is table.gpu
        assert step([6 + k], 12).rows.tolist() == [6 + k]
    for tracked in (True, True, False):  # a batch that shrank for good
        state.prepare(rows[:2], pad_to=2).stage(2, group)
        assert group.publish()
        assert (state.gpu_tracked is table.gpu) == tracked


def test_a_direct_publish_stops_the_tracking():
    """A direct `publish` writes the whole prefix behind the tracking's back:
    a row that grew before it and shrinks after it must still be cleared."""
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([1, 2]), new_block_table(list(range(100, 120)))]
    state.prepare(rows, pad_to=2).stage(2, group)
    assert group.publish()
    rows[1].append(120)  # a quiet step: enter the tracking
    state.prepare(rows, pad_to=2).stage(2, group)
    assert group.publish() and state.gpu_tracked is table.gpu
    rows[0].extend(range(3, 12))  # 11 pages
    state.prepare(rows, pad_to=2).publish(2)
    device[:2] = table.gpu[:2].numpy()
    assert state.gpu_tracked is None
    rows[0] = new_block_table([500])
    state.prepare(rows, pad_to=2).stage(2, group)
    assert group.publish()
    np.testing.assert_array_equal(device[:2], table.np[:2])


def test_a_cut_short_publish_widens_the_bound_it_may_have_written():
    """A planned row that grew is cut short mid-upload (`_sending` widened its
    bound before the upload); after it shrinks, the next plan clears it. The
    other row stays short, so only that widening covers the stale pages."""
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([1, 2]), new_block_table([100, 101])]
    state.prepare(rows, pad_to=2).stage(2, group)
    assert group.publish()
    rows[1].append(120)  # a quiet step: enter the tracking
    state.prepare(rows, pad_to=2).stage(2, group)
    assert group.publish() and state.gpu_tracked is table.gpu
    rows[0].extend(range(3, 12))  # 11 pages
    state.prepare(rows, pad_to=2).stage(2, group)
    assert group.plan.rows.tolist() == [0]
    device[0, :12] = table.np[0, :12]  # the cut-short upload landed row 0
    group.publish("fail")
    rows[0] = new_block_table([500])
    state.prepare(rows, pad_to=2).stage(2, group)
    assert group.publish()
    np.testing.assert_array_equal(device[:2], table.np[:2])


class _CpuTorch:
    """`torch` for `packed_h2d` on a CPU-only build: no pinned memory, every
    tensor on the CPU."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def empty(*args, pin_memory=False, device=None, **kwargs):
        return torch.empty(*args, **kwargs)

    @staticmethod
    def tensor(*args, device=None, **kwargs):
        return torch.tensor(*args, **kwargs)


class _PackedDevice:
    """The device `PackedCopy.create` packs for (it requires a GPU one)."""

    type = "cuda"


class _ScatterEmulator:
    """Both scatter kernels in numpy: read the uploaded header (and, on a
    planned publish, the row fields and row map) from the device arena and
    write every member's destination."""

    def __init__(self, packed):
        self.packed = packed
        self.calls = {"plain": 0, "planned": 0}

    def kernel(self, planned):
        emulator = self

        class _Launch:
            def __getitem__(self, grid):
                return self

            def __call__(self, arena, destinations, wide, *args, **kwargs):
                emulator.run(arena, planned)
                return self

        return _Launch()

    def run(self, arena, planned):
        if self.packed is None:
            return  # the compile launch in `PackedCopy.create`: an empty header
        self.calls["planned" if planned else "plain"] += 1
        a = arena.numpy()
        members = self.packed.members
        m = len(members)
        pairs = a[: 16 * m].view(np.int64)
        fields = a[16 * m : 32 * m].view(np.int64) if planned else None
        for i, binding in enumerate(members):
            start, length = int(pairs[2 * i]), int(pairs[2 * i + 1])
            target = binding.destination.view(torch.uint8).reshape(-1).numpy()
            row_bytes = int(fields[2 * i]) if planned else 0
            if not row_bytes:
                target[:length] = a[start : start + length]
                continue
            rows = a[int(fields[2 * i + 1]) :][: 4 * (length // row_bytes)]
            pitch = int(self.packed.pitches[i])
            for j, row in enumerate(rows.view(np.int32)):
                block = a[start + j * row_bytes : start + (j + 1) * row_bytes]
                target[row * pitch : row * pitch + row_bytes] = block


def test_the_real_packed_transport_carries_the_plans(monkeypatch):
    """The real `PublicationGroup` and `PackedCopy` (Triton stubbed, both
    scatter kernels emulated from the uploaded arena) under scheduler churn
    with the real changed-row rule: `set_row_plan`, `plan_rows`,
    `_submit_planned` with its header and row map, and the plain `submit`
    restored after each planned publish. The destination equals the host
    after every publish."""
    from aiter_stub import atom_under_stubs

    from atom.utils import block_tables

    with atom_under_stubs() as load:
        packed_h2d = load("atom.utils.packed_h2d")
    monkeypatch.setattr(packed_h2d, "torch", _CpuTorch())
    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    columns, capacity = 96, 48
    table = CpuGpuBuffer(
        capacity, columns, dtype=torch.int32, device="cpu", pin_memory=False
    )
    others = [
        CpuGpuBuffer(97, dtype=torch.int32, device="cpu", pin_memory=False)
        for _ in range(2)
    ]
    owner = PublicationOwner("cpu")
    members = [owner.bind(table, "block_tables")]
    members += [owner.bind(b, f"o{i}") for i, b in enumerate(others)]
    group = owner.group("step", members)
    scatter = _ScatterEmulator(None)
    monkeypatch.setattr(packed_h2d, "scatter_packed_words", scatter.kernel(False))
    monkeypatch.setattr(packed_h2d, "scatter_packed_rows", scatter.kernel(True))
    packed, why = packed_h2d.PackedCopy.create(group.members, _PackedDevice())
    assert packed is not None, why
    assert packed.row_kernel is not None  # compiled when the group is created
    scatter.packed = packed
    group._backend, group.transport = packed, "packed"
    state = block_table_state(table)
    rng = np.random.default_rng(7)
    pyrng = random.Random(7)
    running = [
        new_block_table(ids(rng, pyrng.randint(1, 20), 1 << 20)) for _ in range(20)
    ]
    buckets = [8, 16, 24, 32, 40, 48]
    for step in range(300):
        owner.begin()
        decode_churn(
            running,
            rng,
            pyrng,
            step,
            columns=columns,
            max_running=40,
            page_limit=1 << 20,
        )
        count = next(b for b in buckets if b >= len(running))
        state.prepare(running, pad_to=count).stage(count, group)
        sizes = []
        for b in others:
            b.np[:] = rng.integers(0, 1 << 30, size=b.np.shape, dtype=np.int32)
            sizes.append(int(rng.integers(1, 98)))
            group.set_count(b, sizes[-1])
        group.publish()
        owner.finish()
        assert "submit" not in vars(packed)  # the plain path is back
        assert packed.plans == [None] * 3  # the plans were consumed
        for b, size in zip(others, sizes):
            np.testing.assert_array_equal(b.gpu[:size].numpy(), b.np[:size])
        np.testing.assert_array_equal(
            table.gpu[:count].numpy(), table.np[:count], err_msg=str(step)
        )
    assert scatter.calls["planned"] and scatter.calls["plain"]


def test_a_pending_plan_is_withdrawn_when_the_tracking_stops():
    """A tracked stage leaves a plan with its group; a direct `publish` of the
    same table before that group publishes stops the tracking and withdraws
    the plan, so the group copies the whole prefix."""
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([i + 1]) for i in range(6)]
    state.prepare(rows, pad_to=6).stage(6, group)  # enter
    assert group.publish()
    rows[2].append(9)
    state.prepare(rows, pad_to=6).stage(6, group)
    assert group.plan.rows.tolist() == [2]
    state.publish(6)  # behind the group's back
    assert group.plan is None and state.gpu_tracked is None
    assert group.publish()
    np.testing.assert_array_equal(device[:6], table.np[:6])


def test_a_changed_row_that_is_zeros_on_both_sides_sends_nothing():
    """A row replaced by another empty one changes its write number, but both
    sides are zero: the plan is empty and nothing goes up."""
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([1, 2]), new_block_table([]), new_block_table([3])]
    state.prepare(rows, pad_to=3).stage(3, group)  # enter
    assert group.publish()
    rows[1] = new_block_table([])
    state.prepare(rows, pad_to=3).stage(3, group)
    assert group.count is None
    assert group.publish()
    np.testing.assert_array_equal(device[:3], table.np[:3])


def test_plan_rows_routes_one_publish_and_withdraws(monkeypatch):
    """`PackedCopy.plan_rows` routes the next publish through
    `_submit_planned`; withdrawing the last plan restores the plain
    `submit`."""
    from aiter_stub import atom_under_stubs

    with atom_under_stubs() as load:
        packed_h2d = load("atom.utils.packed_h2d")
    monkeypatch.setattr(packed_h2d, "torch", _CpuTorch())
    buffers = [
        CpuGpuBuffer(16, 64, dtype=torch.int32, device="cpu", pin_memory=False),
        CpuGpuBuffer(97, dtype=torch.int32, device="cpu", pin_memory=False),
    ]
    owner = PublicationOwner("cpu")
    members = [owner.bind(buffers[0], "t"), owner.bind(buffers[1], "o")]
    monkeypatch.setattr(
        packed_h2d, "scatter_packed_rows", _ScatterEmulator(None).kernel(True)
    )
    packed, why = packed_h2d.PackedCopy.create(members, _PackedDevice())
    assert packed is not None, why
    plan = RowPlan(np.asarray([3], dtype=np.int32), 16)
    packed.plan_rows(0, plan)
    assert vars(packed)["submit"] == packed._submit_planned
    packed.plan_rows(0, None)
    assert "submit" not in vars(packed) and packed.plans == [None, None]


def test_a_whole_stage_withdraws_a_plan_a_failed_publish_left(monkeypatch):
    """A publish that raises before `submit` leaves the stage's plan with the
    group. If the next stage of that group goes up whole (here a busy step),
    it must withdraw that plan, or the group would send the stale rows
    instead of the whole prefix."""
    from atom.utils import block_tables

    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([i + 1]) for i in range(8)]
    step = functools.partial(_tracked_step, state, group, table, device, rows)
    step([], 8)  # new rows: busy, untracked
    step([0], 8)  # enter
    rows[2].append(9)
    state.prepare(rows, pad_to=8).stage(8, group)
    assert group.plan.rows.tolist() == [2]
    # The publish raises before `submit`: nothing is sent, the plan stays.
    for r in rows[:6]:
        r.append(70)
    state.prepare(rows, pad_to=8).stage(8, group)  # busy: the whole prefix
    assert group.plan is None
    assert group.publish()
    np.testing.assert_array_equal(device[:8], table.np[:8])


def test_the_real_transport_drops_a_plan_a_failed_publish_left(monkeypatch):
    """The same with the real `PublicationGroup` and `PackedCopy`: the whole
    stage clears `PackedCopy.plans` and restores the plain `submit`, and the
    destination equals the host after the next publish."""
    from aiter_stub import atom_under_stubs

    from atom.utils import block_tables

    with atom_under_stubs() as load:
        packed_h2d = load("atom.utils.packed_h2d")
    monkeypatch.setattr(packed_h2d, "torch", _CpuTorch())
    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    table = CpuGpuBuffer(8, 64, dtype=torch.int32, device="cpu", pin_memory=False)
    other = CpuGpuBuffer(97, dtype=torch.int32, device="cpu", pin_memory=False)
    owner = PublicationOwner("cpu")
    members = [owner.bind(table, "block_tables"), owner.bind(other, "o")]
    group = owner.group("step", members)
    scatter = _ScatterEmulator(None)
    monkeypatch.setattr(packed_h2d, "scatter_packed_words", scatter.kernel(False))
    monkeypatch.setattr(packed_h2d, "scatter_packed_rows", scatter.kernel(True))
    packed, why = packed_h2d.PackedCopy.create(group.members, _PackedDevice())
    assert packed is not None, why
    scatter.packed = packed
    group._backend, group.transport = packed, "packed"
    state = block_table_state(table)
    state._count_writes(True)
    rows = [new_block_table([i + 1]) for i in range(8)]

    def step(changed, publish=True):
        owner.begin()
        for i in changed:
            rows[i].append(100 + i)
        state.prepare(rows, pad_to=8).stage(8, group)
        group.set_count(other, 5)
        if publish:
            group.publish()
        owner.finish()

    step([])  # new rows: busy, untracked
    step([0])  # enter
    step([2], publish=False)  # a plan, then a publish that never submits
    assert packed.plans[0] is not None
    step(range(6))  # busy: the whole prefix withdraws the plan
    assert packed.plans == [None, None] and "submit" not in vars(packed)
    np.testing.assert_array_equal(table.gpu.numpy(), table.np)
