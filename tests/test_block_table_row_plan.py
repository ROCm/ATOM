# SPDX-License-Identifier: MIT
"""Packed block-table publishes send only the rows the destination lacks, and
only as wide as needed, yet leave the destination byte-identical to a
whole-prefix copy of every staged table."""

import os
import random

import numpy as np
import pytest
import torch

from atom.model_engine.sequence import new_block_table
from atom.utils import CpuGpuBuffer
from atom.utils.block_tables import block_table_state
from atom.utils.h2d import PublicationOwner, RowPlan
from tests.block_table_churn import decode_churn, ids


@pytest.fixture(autouse=True)
def _plan_every_table(monkeypatch):
    """These tables are small and often change most rows; plan them anyway
    (`WHOLE_PREFIX_BYTES`, `PLAN_MAX_CHANGED_FRACTION`)."""
    from atom.utils import block_tables

    monkeypatch.setattr(block_tables, "WHOLE_PREFIX_BYTES", 0)
    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 1.0)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)


def _buffers(rows, columns, extra=2):
    table = CpuGpuBuffer(rows, columns, dtype=torch.int32, device="cuda")
    others = [
        CpuGpuBuffer(97 + i, dtype=torch.int32, device="cuda") for i in range(extra)
    ]
    return table, others


def _owner(table, others):
    owner = PublicationOwner("cuda", torch.cuda.Event())
    members = [owner.bind(table, "block_tables")]
    members += [owner.bind(b, f"o{i}") for i, b in enumerate(others)]
    group = owner.group("step", members)
    assert group.use_packed() == "packed"
    return owner, group


def _state(table):
    """The table's state, counting written rows from the first prepare."""
    state = block_table_state(table)
    state._count_writes(True)
    return state


def test_scatter_rows_kernel_matches_row_writes():
    """The row kernel's row path: arbitrary increasing rows, any 16-byte width."""
    rng = np.random.default_rng(0)
    table, others = _buffers(64, 1024)
    owner, group = _owner(table, others)
    reference = torch.zeros_like(table.gpu)
    for step in range(60):
        owner.begin()
        table.np[:] = rng.integers(0, 1 << 30, size=table.np.shape, dtype=np.int32)
        for b in others:
            b.np[:] = step
        count = int(rng.integers(1, 65))
        k = int(rng.integers(1, count + 1))
        rows = np.sort(rng.choice(count, size=k, replace=False)).astype(np.int32)
        width = int(rng.integers(1, 257)) * 4
        if step % 7 == 0:
            rows, width = np.arange(count, dtype=np.int32), 1024
        group.set_count(table, count)
        group.set_row_plan(table, RowPlan(rows, width * 4))
        for b in others:
            group.set_count(b, int(rng.integers(1, 98)))
        group.publish()
        owner.finish()
        reference[torch.from_numpy(rows).long().cuda(), :width] = torch.from_numpy(
            table.np[rows, :width]
        ).cuda()
        owner.completion.synchronize()
        assert torch.equal(table.gpu, reference), step


def _simulate(seed, steps, columns, capacity, packed=True):
    """A scheduler's churn against the full-copy reference: appends, finishes
    (later rows shift up), admissions (with prefix hits), preemption,
    speculative rollback, bucket changes, prefill interleaving, unversioned
    and dummy rows, rows of exactly ``columns`` pages."""
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    page_limit = 1 << 20
    table, others = _buffers(capacity, columns)
    owner, group = _owner(table, others)
    if not packed:
        group._backend, group.transport = None, "direct"
    state = _state(table)
    reference = table.gpu.clone()
    running = [
        new_block_table(ids(rng, int(rng.integers(1, 64)), page_limit))
        for _ in range(40)
    ]
    sent = full = 0
    buckets = [b for b in (8, 16, 24, 32, 40, 48, 56, 64, 96, 128) if b <= capacity]
    for step in range(steps):
        owner.begin()
        kind = pyrng.random()
        if kind < 0.08:  # a prefill step: another batch in the same buffer
            rows = [
                new_block_table(ids(rng, int(rng.integers(1, columns + 1)), page_limit))
                for _ in range(pyrng.randint(1, 6))
            ]
            if pyrng.random() < 0.5:  # a prefix hit shares a running request's pages
                src = pyrng.choice(running)
                rows[0] = new_block_table(list(src[: pyrng.randint(1, len(src))]) + [7])
            count = len(rows)
        elif kind < 0.1:  # a dummy run: private arange rows
            n = pyrng.randint(1, 8)
            rows = [np.arange(i * 3, i * 3 + 3, dtype=np.int32) for i in range(n)]
            count = pyrng.choice([b for b in buckets if b >= n])
        else:
            decode_churn(
                running,
                rng,
                pyrng,
                step,
                columns=columns,
                max_running=capacity,
                page_limit=page_limit,
            )
            rows = (
                running
                if pyrng.random() > 0.05
                else [np.array(t, np.int32) for t in running]
            )
            count = next((b for b in buckets if b >= len(rows)), len(rows))
        state.prepare(rows, pad_to=count, page_limit=page_limit)
        published = state.published
        dirty = state._dirty(count)[0]
        state.stage(count, group)
        plan = group._backend.plans[0] if group._backend is not None else None
        if plan is not None:
            sent += len(plan.rows) * plan.row_bytes
        elif group.counts[0]:
            sent += count * columns * 4
        for b in others:
            group.set_count(b, 5)
        group.publish()
        owner.finish()
        if dirty:  # what the whole-prefix copy would have done
            reference[:count] = torch.from_numpy(table.np[:count]).cuda()
            full += count * columns * 4
        owner.completion.synchronize()
        assert torch.equal(table.gpu, reference), (seed, step)
        assert published is None or state.published is not None
    return sent, full


@pytest.mark.parametrize("seed", range(6))
def test_churn_is_byte_identical_to_full_copies(seed):
    sent, full = _simulate(seed, 400, columns=512, capacity=128)
    assert sent < full


def test_direct_transport_ignores_plans():
    _simulate(11, 150, columns=256, capacity=64, packed=False)


def test_reentering_the_tracking_resends_every_row_at_full_width():
    table, others = _buffers(8, 64)
    owner, group = _owner(table, others)
    state = _state(table)
    rows = [new_block_table([1, 2, 3]), new_block_table([4])]
    owner.begin()
    state.prepare(rows, pad_to=4)
    state.stage(4, group)
    group.publish()
    owner.finish()
    rows[0].append(9)
    owner.begin()
    state.prepare(rows, pad_to=4)
    state.stage(4, group)
    plan = group._backend.plans[0]
    assert plan is not None and plan.rows.tolist() == [0] and plan.row_bytes == 16
    group.publish()
    owner.finish()
    owner.completion.synchronize()
    assert table.gpu[:2, :5].tolist() == [[1, 2, 3, 9, 0], [4, 0, 0, 0, 0]]
    # Out of the tracking and back in: the destination is unknown again, so
    # every row goes up at full width.
    state._untrack()
    rows[1].append(5)
    owner.begin()
    state.prepare(rows, pad_to=4)
    state.stage(4, group)
    assert group.counts[0] == 4 and group._backend.plans[0] is None
    group.publish()
    owner.finish()
    owner.completion.synchronize()
    assert torch.equal(table.gpu[:4].cpu(), torch.from_numpy(table.np[:4]))


@pytest.mark.parametrize("applied", ["none", "half"])
def test_a_failed_or_partial_publish_resends_its_rows(monkeypatch, applied):
    """A publish that raises (nothing or only some rows reached the
    destination) leaves every row it covered unknown: the next publish, on a
    rebuilt owner, sends each of them again, at the host's bound widened by
    what the failed upload may have written, and the destination ends equal
    to the host."""
    columns = 64
    table, others = _buffers(8, columns)
    owner, group = _owner(table, others)
    state = _state(table)
    rows = [new_block_table(list(range(1, 4 + 3 * i))) for i in range(6)]
    owner.begin()
    state.prepare(rows, pad_to=8)
    state.stage(8, group)
    group.publish()
    owner.finish()
    owner.completion.synchronize()
    # Rows 1 and 4 change; their pages grow past the old bounds.
    rows[1].extend([40, 41, 42, 43, 44])
    rows[4].append(77)
    owner.begin()
    state.prepare(rows, pad_to=8)
    state.stage(8, group)
    plan = group._backend.plans[0]
    assert plan is not None and plan.rows.tolist() == [1, 4]

    def failing_submit(counts, plans=None):
        if applied == "half":  # the first planned row lands, the second not
            table.gpu[1, : plan.row_bytes // 4] = torch.from_numpy(
                table.np[1, : plan.row_bytes // 4]
            ).cuda()
        raise RuntimeError("injected enqueue failure")

    monkeypatch.setattr(group._backend, "submit", failing_submit)
    with pytest.raises(RuntimeError, match="injected"):
        group.publish()
    assert owner._state == "failed"
    owner.drain()

    # The runner rebuilds a failed owner over the same buffers.
    table._publication = None
    for b in others:
        b._publication = None
    owner, group = _owner(table, others)
    rows[2].append(91)
    owner.begin()
    state.prepare(rows, pad_to=8)
    state.stage(8, group)
    plan = group._backend.plans[0]
    widest = max(len(r) for r in rows)
    assert plan is not None
    assert plan.rows.tolist() == list(range(8))  # every row of the failed count
    assert plan.row_bytes == -(-widest // 4) * 16 < columns * 4
    group.publish()
    owner.finish()
    owner.completion.synchronize()
    assert torch.equal(table.gpu[:8].cpu(), torch.from_numpy(table.np[:8]))
