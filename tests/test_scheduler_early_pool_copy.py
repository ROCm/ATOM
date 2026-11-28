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
        # What the group's reduce says beside the verdict: ranks with prefill
        # to admit, and requests decoding anywhere.
        self.prefillable_ranks = 1
        self.running_decodes = 0
        self.seen: list[dict] = []

    def protects_decode(self, running_decode_batch):
        return False

    def should_allow_prefill(self, **kw):
        self.seen.append(kw)
        return PrefillDecision(
            allow=self.allow,
            prefillable_ranks=self.prefillable_ranks,
            running_decodes=self.running_decodes,
        )

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
    # A copy indexed at the top of the next pass, as these tests were written
    # for; the wait within a pass has its own, below, with it on.
    env.setdefault("ATOM_POOL_COPY_INPASS_WAIT_MS", 0)
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
    monkeypatch.setattr(bm.kv, "fetch_pending", lambda owner: owner in pool.in_flight)
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


def test_group_stops_holding_for_a_copy_that_does_not_land(monkeypatch, seq_factory):
    monkeypatch.setattr(Scheduler, "_ARRIVING_MAX_TICKS", 2)
    sched, delayer, _, _ = _make(monkeypatch, seq_factory)

    held = []
    for _ in range(4):
        sched.schedule()
        held.append(delayer.seen[-1]["prefill_arriving"])

    assert held == [True, True, False, False]


@pytest.mark.parametrize("probed", [0, 1])
def test_pending_tokens_count_a_waiting_request_past_its_prefix_hit(
    monkeypatch, seq_factory, probed
):
    sched, delayer, pool, _ = _make(
        monkeypatch, seq_factory, n=0, ATOM_POOL_PROBED_PENDING=probed
    )
    bm = sched.block_manager
    block = bm.hash_block_size
    landed, mid_copy = (seq_factory(list(range(4 * block))) for _ in range(2))
    for seq in (landed, mid_copy):
        sched.add(seq)
    sched.schedule()  # HOLD: both copies start
    pool.in_flight.discard(landed.id)
    # Three of the landed request's four blocks are in this rank's pool now.
    monkeypatch.setattr(bm, "can_allocate_memoized", lambda seq: 3)

    sched.schedule()

    # Probed: one block to compute, and nothing yet for the one mid-copy. The
    # upper bound counts both whole prompts, up to the budget.
    assert delayer.seen[-1]["pending_tokens"] == (
        block if probed else min(8 * block, sched.max_num_batched_tokens)
    )


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

    # The prefillable probe asks, as it did before early copies, on a HOLD as
    # on a FIRE; the rank it leaves mid-copy counts as blocked, not arriving.
    assert pool.asked == [seq.id]
    assert delayer.seen[-1]["prefillable"] is False
    assert delayer.seen[-1]["prefill_arriving"] is False


# --- `_await_arriving_copies`: no dummy step between a copy and its request ---


def _idle_group(delayer):
    """What the delayer answers a rank whose only work is mid-copy, in a group
    with nothing else to run: nothing to admit anywhere, so the FIRE is vacuous."""
    delayer.allow = True
    delayer.prefillable_ranks = 0
    delayer.running_decodes = 0


def _waits(monkeypatch, sched, pool, lands=True):
    """Record the waits on the pool; a wait that `lands` ends with its copy in."""
    waits = []

    def wait_fetch(owner, timeout_s):
        waits.append((owner, timeout_s))
        if lands:
            pool.in_flight.discard(owner)
        return lands

    monkeypatch.setattr(sched.block_manager.kv, "wait_fetch", wait_fetch)
    return waits


def test_idle_group_waits_for_the_copy_and_admits_its_request_in_the_same_pass(
    monkeypatch, seq_factory
):
    sched, delayer, pool, (seq,) = _make(monkeypatch, seq_factory)
    _idle_group(delayer)
    waits = _waits(monkeypatch, sched, pool)

    result = sched.schedule()

    # The pass that started the copy is the one that prefills the request: no
    # empty pass, hence no dummy step, between the two.
    assert pool.asked == [seq.id]
    assert _prefills(result) == 1 and seq.id in result[1]
    assert [owner for owner, _ in waits] == [seq.id]
    # Bounded: the default covers the slowest copy seen, with a margin.
    assert 0 < waits[0][1] <= 0.3
    assert delayer.seen[-1]["prefill_arriving"] is True


def test_every_copy_in_flight_shares_the_one_bound(monkeypatch, seq_factory):
    sched, delayer, pool, seqs = _make(monkeypatch, seq_factory, n=3)
    _idle_group(delayer)
    waits = _waits(monkeypatch, sched, pool)

    result = sched.schedule()

    assert [owner for owner, _ in waits] == [seq.id for seq in seqs]
    assert all(0 < timeout <= 0.3 for _, timeout in waits)
    assert _prefills(result) == 3


@pytest.mark.parametrize(
    "peers",
    [
        {"running_decodes": 3},  # a peer decodes: the step is its forward
        {"prefillable_ranks": 1},  # a peer prefills: likewise
        {"allow": False},  # the group holds its prefill
    ],
)
def test_no_wait_while_any_rank_has_a_step_of_its_own(monkeypatch, seq_factory, peers):
    sched, delayer, pool, (seq,) = _make(monkeypatch, seq_factory)
    _idle_group(delayer)
    for name, value in peers.items():
        setattr(delayer, name, value)
    waits = _waits(monkeypatch, sched, pool)

    assert _prefills(sched.schedule()) == 0

    # As before: the copy runs beside the step and the next pass sees it.
    assert waits == [] and list(sched.waiting) == [seq]


def test_no_wait_while_this_rank_decodes(monkeypatch, seq_factory):
    sched, delayer, pool, _ = _make(monkeypatch, seq_factory)
    _idle_group(delayer)
    waits = _waits(monkeypatch, sched, pool)
    sched.running.append(seq_factory(list(range(4))))

    sched._await_arriving_copies(
        PrefillDecision(allow=True, prefillable_ranks=0), mid_copy_only=True
    )

    assert waits == []


def test_the_wait_can_be_switched_off(monkeypatch, seq_factory):
    sched, delayer, pool, (seq,) = _make(
        monkeypatch, seq_factory, ATOM_POOL_COPY_WAIT_MS=0
    )
    _idle_group(delayer)
    waits = _waits(monkeypatch, sched, pool)

    assert _prefills(sched.schedule()) == 0
    assert waits == [] and list(sched.waiting) == [seq]


def test_a_copy_that_misses_the_bound_is_not_waited_for_again(monkeypatch, seq_factory):
    sched, delayer, pool, (seq,) = _make(monkeypatch, seq_factory)
    _idle_group(delayer)
    waits = _waits(monkeypatch, sched, pool, lands=False)

    for _ in range(3):
        assert _prefills(sched.schedule()) == 0
    # One bound for the stalled copy, not one a pass.
    assert len(waits) == 1

    # It lands on its own and is admitted like any landed copy; the next
    # request's copy is a new one, and is waited for.
    pool.in_flight.discard(seq.id)
    assert _prefills(sched.schedule()) == 1
    sched.running.remove(seq)  # handed off: the rank has nothing to run again
    later = seq_factory(list(range(16)))
    sched.add(later)
    sched.schedule()
    assert [owner for owner, _ in waits] == [seq.id, later.id]


def test_a_rank_with_nothing_mid_copy_never_waits(monkeypatch, seq_factory):
    sched, delayer, pool, _ = _make(monkeypatch, seq_factory)
    _idle_group(delayer)
    sched.schedule()  # HOLD-free pass: its copy starts and lands meanwhile
    pool.in_flight.clear()
    waits = _waits(monkeypatch, sched, pool)

    # Prefillable now: the delayer is told so and admission goes ahead as ever.
    delayer.prefillable_ranks = 1
    assert _prefills(sched.schedule()) == 1
    assert waits == []


# --- `_land_arriving_copies`: a copy is indexed in the pass that started it ---

INPASS = {"ATOM_POOL_COPY_INPASS_WAIT_MS": 20}


def _busy_group(delayer):
    """A group under load: this rank's peers prefill and decode, so the idle
    group's wait (`_await_arriving_copies`) never applies."""
    delayer.allow = True
    delayer.prefillable_ranks = 3
    delayer.running_decodes = 5


def test_copy_started_by_a_pass_is_waited_for_and_its_request_prefills_in_it(
    monkeypatch, seq_factory
):
    sched, delayer, pool, (seq,) = _make(monkeypatch, seq_factory, **INPASS)
    _busy_group(delayer)
    waits = _waits(monkeypatch, sched, pool)

    result = sched.schedule()

    assert pool.asked == [seq.id]
    assert [owner for owner, _ in waits] == [seq.id]
    assert 0 < waits[0][1] <= 0.02
    # Landed before the delayer's probe: the rank reports the prefill work the
    # request now is, not a copy the group might hold for.
    assert delayer.seen[-1]["prefillable"] is True
    assert delayer.seen[-1]["prefill_arriving"] is False
    assert _prefills(result) == 1 and seq.id in result[1]


def test_switched_off_the_copy_waits_for_the_next_pass_as_before(
    monkeypatch, seq_factory
):
    sched, delayer, pool, (seq,) = _make(monkeypatch, seq_factory)
    _busy_group(delayer)
    waits = _waits(monkeypatch, sched, pool)

    assert _prefills(sched.schedule()) == 0
    assert waits == [] and list(sched.waiting) == [seq]
    assert delayer.seen[-1]["prefillable"] is False

    pool.in_flight.discard(seq.id)  # landed during the step
    assert _prefills(sched.schedule()) == 1
    assert waits == []


def test_one_pass_has_one_budget_for_all_its_copies(monkeypatch, seq_factory):
    sched, delayer, pool, seqs = _make(monkeypatch, seq_factory, n=3, **INPASS)
    _busy_group(delayer)
    clock = [100.0]
    monkeypatch.setattr("atom.model_engine.scheduler.time.monotonic", lambda: clock[0])
    waits = []

    def wait_fetch(owner, timeout_s):
        waits.append((owner, round(timeout_s, 6)))
        # The first copy takes 15 of the pass's 20 ms, the second the rest and
        # more, the third is over by then.
        took = {seqs[0].id: 0.015, seqs[1].id: 0.030}.get(owner, 0.0)
        clock[0] += min(took, max(timeout_s, 0.0))
        landed = took <= max(timeout_s, 0.0)
        if landed:
            pool.in_flight.discard(owner)
        return landed

    monkeypatch.setattr(sched.block_manager.kv, "wait_fetch", wait_fetch)

    result = sched.schedule()

    first, second, third = (seq.id for seq in seqs)
    assert waits[:3] == [(first, 0.02), (second, 0.005), (third, 0.0)]
    # The one that outlasted the budget stays held back, in its place.
    assert _prefills(result) == 2 and set(result[1]) == {first, third}
    assert list(sched.waiting) == [seqs[1]]


def test_copy_that_outlasts_the_budget_is_left_as_before_and_not_waited_for_again(
    monkeypatch, seq_factory
):
    sched, delayer, pool, (seq,) = _make(monkeypatch, seq_factory, **INPASS)
    _busy_group(delayer)
    waits = _waits(monkeypatch, sched, pool, lands=False)

    for _ in range(3):
        assert _prefills(sched.schedule()) == 0
        assert list(sched.waiting) == [seq] and not seq.block_table
    # Waited for once, at the pass that started it; afterwards looked at only
    # (no time left to give), at the top of each pass and where Phase 2 would
    # hold the request back.
    assert waits[0] == (seq.id, pytest.approx(0.02, abs=0.005))
    assert all(timeout <= 0 for _, timeout in waits[1:])

    # It lands during a step and the next pass's top indexes it, as before.
    pool.in_flight.discard(seq.id)
    assert _prefills(sched.schedule()) == 1


def test_copy_started_in_phase_two_gets_its_wait_there(monkeypatch, seq_factory):
    # Early copies off: the copy starts where admission first reaches the
    # request, after the delayer's decision.
    sched, delayer, pool, (seq,) = _make(
        monkeypatch, seq_factory, ATOM_POOL_EARLY_COPY=0, **INPASS
    )
    _busy_group(delayer)
    waits = _waits(monkeypatch, sched, pool)

    result = sched.schedule()

    assert pool.asked == [seq.id]
    assert [owner for owner, _ in waits] == [seq.id]
    assert _prefills(result) == 1 and seq.id in result[1]


def test_a_new_pass_gets_a_new_budget(monkeypatch, seq_factory):
    sched, delayer, pool, (first,) = _make(monkeypatch, seq_factory, **INPASS)
    _busy_group(delayer)
    waits = _waits(monkeypatch, sched, pool, lands=False)
    sched.schedule()

    later = seq_factory(list(range(16)))
    sched.add(later)
    sched.schedule()

    # The stalled copy is not waited for again; the new request's copy is.
    assert waits[0][0] == first.id and waits[0][1] > 0
    fresh = [(owner, timeout) for owner, timeout in waits[1:] if timeout > 0]
    assert [owner for owner, _ in fresh] == [later.id]


def test_no_pool_no_wait(monkeypatch, seq_factory):
    monkeypatch.setenv("ATOM_POOL_COPY_INPASS_WAIT_MS", "20")
    sched = Scheduler(
        MockConfig(max_num_seqs=8, max_num_batched_tokens=64, num_kvcache_blocks=100)
    )
    delayer = _Delayer()
    _busy_group(delayer)
    sched.set_prefill_delayer(delayer)

    def never(*args):
        raise AssertionError("an engine without a shared pool waits for no copy")

    monkeypatch.setattr(sched.block_manager.kv, "wait_fetch", never)
    seq = seq_factory(list(range(16)))
    sched.add(seq)

    assert _prefills(sched.schedule()) == 1
