# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler._early_copy_shared_prefixes.

A waiting request whose prefix another rank's pool holds has its copy started
on any tick -- delayer HOLD or FIRE -- and is not prefillable work while the
copy is in flight, so no FIRE is spent starting it.
"""

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.prefill_delayer import PrefillDecision
from atom.model_engine.scheduler import Scheduler


class _Delayer:
    is_local = False
    max_queue_ms = None
    kv_high_watermark = 0.9
    rank_prefill_costs: tuple = ()

    def __init__(self):
        self.allow = False
        self.seen: list[dict] = []

    def protects_decode(self, running_decode_batch):
        return False

    def should_allow_prefill(self, **kw):
        self.seen.append(kw)
        return PrefillDecision(allow=self.allow, prefillable_ranks=1)

    def notify_prefill_executed(self): ...


class _SharedPool:
    """What `BlockManager` answers about copies from other ranks' pools."""

    def __init__(self):
        self.asked: list[int] = []
        self.in_flight: set[int] = set()

    def start(self, seq) -> bool:
        if seq.id in self.asked:
            return False
        self.asked.append(seq.id)
        self.in_flight.add(seq.id)
        return True

    def pending(self, seq) -> bool:
        self.start(seq)
        return seq.id in self.in_flight


def _make(monkeypatch, seq_factory, n=1, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    sched = Scheduler(
        MockConfig(max_num_seqs=8, max_num_batched_tokens=64, num_kvcache_blocks=100)
    )
    delayer = _Delayer()
    sched.set_prefill_delayer(delayer)
    pool = _SharedPool()
    bm = sched.block_manager
    monkeypatch.setattr(
        BlockManager, "copies_shared_prefixes", property(lambda self: True)
    )
    monkeypatch.setattr(bm, "start_shared_prefix_copy", pool.start)
    monkeypatch.setattr(bm, "shared_prefix_pending", pool.pending)
    monkeypatch.setattr(
        bm, "shared_prefix_in_flight", lambda seq: seq.id in pool.in_flight
    )
    seqs = [seq_factory(list(range(16))) for _ in range(n)]
    for seq in seqs:
        sched.add(seq)
    return sched, delayer, pool, seqs


def _prefills(result) -> int:
    return 0 if result is None else result[0].total_seqs_num_prefill


def test_hold_tick_starts_the_copy_of_every_waiting_request(monkeypatch, seq_factory):
    sched, _, pool, seqs = _make(monkeypatch, seq_factory, n=3)

    assert _prefills(sched.schedule()) == 0

    assert pool.asked == [seq.id for seq in seqs]
    assert list(sched.waiting) == seqs


def test_request_mid_copy_is_not_prefillable_and_the_group_holds_for_it(
    monkeypatch, seq_factory
):
    sched, delayer, _, _ = _make(monkeypatch, seq_factory, n=2)

    sched.schedule()

    assert delayer.seen[-1]["prefillable"] is False
    assert delayer.seen[-1]["prefill_arriving"] is True


def test_landed_copy_makes_the_rank_prefillable_and_the_fire_admits_it(
    monkeypatch, seq_factory
):
    sched, delayer, pool, (first, second) = _make(monkeypatch, seq_factory, n=2)
    sched.schedule()  # HOLD: both copies start
    pool.in_flight.discard(second.id)

    delayer.allow = True
    result = sched.schedule()

    assert delayer.seen[-1]["prefillable"] is True
    assert delayer.seen[-1]["prefill_arriving"] is False
    # The one still mid-copy keeps its place; the landed one behind it prefills.
    assert _prefills(result) == 1 and second.id in result[1]
    assert list(sched.waiting) == [first]


def test_one_tick_starts_a_bounded_number_of_copies(monkeypatch, seq_factory):
    monkeypatch.setattr(Scheduler, "_EARLY_COPY_PER_TICK", 2)
    sched, _, pool, seqs = _make(monkeypatch, seq_factory, n=5)

    assert sched._early_copy_shared_prefixes() == 2
    assert sched._early_copy_shared_prefixes() == 2
    assert sched._early_copy_shared_prefixes() == 1
    assert sched._early_copy_shared_prefixes() == 0
    assert pool.asked == [seq.id for seq in seqs]


def test_hold_can_be_switched_off_and_the_rank_then_counts_as_blocked(
    monkeypatch, seq_factory
):
    sched, delayer, _, _ = _make(
        monkeypatch, seq_factory, ATOM_POOL_COPY_HOLDS_PREFILL=0
    )

    sched.schedule()

    assert delayer.seen[-1]["prefillable"] is False
    assert delayer.seen[-1]["prefill_arriving"] is False
    assert delayer.seen[-1]["waiting_requests"] == 1


@pytest.mark.parametrize("fire", [False, True])
def test_switched_off_the_copy_starts_where_it_did_before(
    monkeypatch, seq_factory, fire
):
    sched, delayer, pool, (seq,) = _make(
        monkeypatch, seq_factory, ATOM_POOL_EARLY_COPY=0
    )
    delayer.allow = fire

    assert sched._early_copy_shared_prefixes() == 0
    sched.schedule()

    # Only Phase 2 of a FIRE asks (the probe's walk is the block manager's own,
    # stubbed out of this test), and the probe does not look at copies.
    assert pool.asked == ([seq.id] if fire else [])
    assert delayer.seen[-1]["prefill_arriving"] is False
