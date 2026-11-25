# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ATOM_DP_IDLE_WHEN_BLOCKED: a DP rank votes "unfinished" only when a pass
could forward something (`Scheduler.has_forward_work`).

CPU only. The scheduler side runs on a real `Scheduler` with an offload
connector double whose lookup clock follows the offload tier's
(`ChunkedOffloadSchedulerBase.lookup_pending` / `lookup_would_defer`). The
busy loop runs on a bare `DPEngineCoreProc` whose collective, engine step,
dummy forward and KV drain are recorded instead of run.
"""

import itertools
import time
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import stubbed_aiter
from conftest import MockConfig
from test_scheduler_early_load import _Connector

with stubbed_aiter():
    from atom.model_engine import engine_core as ec
    from atom.model_engine.engine_core import DPEngineCoreProc

from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus
from atom.utils import envs


class _LookupConnector(_Connector):
    """Async tier lookups. `in_flight` ids are submitted and unanswered;
    `prefetch_lookups` submits. `lookup_pending` starts a request's defer
    window and `lookup_would_defer` honours it without starting it, as the
    offload tier does."""

    def __init__(self):
        super().__init__()
        self.match = (0, False)  # an answered lookup has no load: it prefills
        self.in_flight = set()
        self.submitted = []
        self.answer_on_prefetch = False
        self.defer_since = {}

    def prefetch_lookups(self, seqs):
        for seq in seqs:
            if seq.id not in self.submitted:
                self.submitted.append(seq.id)
                if not self.answer_on_prefetch:
                    self.in_flight.add(seq.id)

    def answer(self, seq):
        self.in_flight.discard(seq.id)

    def lookup_ready(self, seq):
        return seq.id not in self.in_flight

    def lookup_pending(self, seq):
        if seq.id not in self.in_flight:
            self.defer_since.pop(seq.id, None)
            return False
        now = time.monotonic()
        first = self.defer_since.setdefault(seq.id, now)
        return now - first <= envs.OFFLOAD_LOOKUP_DEFER_S

    def lookup_would_defer(self, seq):
        if seq.id not in self.in_flight:
            return False
        first = self.defer_since.get(seq.id)
        return first is None or time.monotonic() - first <= envs.OFFLOAD_LOOKUP_DEFER_S


@pytest.fixture
def make(monkeypatch, seq_factory):
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "0")

    def build(n_waiting=0, conn=True, status=SequenceStatus.WAITING):
        sched = Scheduler(
            MockConfig(
                max_num_seqs=8, max_num_batched_tokens=64, num_kvcache_blocks=100
            )
        )
        if conn:
            sched.kv_connector = _LookupConnector()
        seqs = [seq_factory(list(range(16))) for _ in range(n_waiting)]
        for seq in seqs:
            seq.status = status
            sched.waiting.append(seq)
        return sched, seqs

    return build


class _Clock:
    def __init__(self, monkeypatch, start=100.0):
        self.now = start
        monkeypatch.setattr(time, "monotonic", lambda: self.now)


# ------------------------------ scheduler ---------------------------------- #


def test_deferred_frees_alone_are_not_forward_work(make):
    s, _ = make(conn=False)
    s.deferred_free_blocks[7] = object()
    assert not s.is_finished()  # the old vote: a dummy forward every pass
    assert not s.has_forward_work()


def test_idle_scheduler_has_no_forward_work(make):
    s, _ = make()
    assert s.is_finished() and not s.has_forward_work()


def test_running_and_rejected_are_work(make, seq_factory):
    s, _ = make()
    s.running.append(seq_factory([1, 2, 3]))
    assert s.has_forward_work()
    s, _ = make()
    s._rejected.append(seq_factory([1, 2, 3]))
    assert s.has_forward_work()


def test_waiting_without_connector_is_work(make):
    s, _ = make(1, conn=False)
    assert s.has_forward_work()


def test_fresh_request_is_prefetched_and_blocks_on_its_lookup(make):
    s, (seq,) = make(1)
    conn = s.kv_connector
    assert not s.has_forward_work()
    assert conn.submitted == [seq.id]
    conn.answer(seq)
    assert s.has_forward_work()
    # And a pass now admits it: the vote and Phase 2 agree.
    _, scheduled = s.schedule()
    assert list(scheduled) == [seq.id]


def test_lookup_answered_at_once_is_work(make):
    s, _ = make(1)
    s.kv_connector.answer_on_prefetch = True
    assert s.has_forward_work()


def test_load_in_flight_blocks_until_it_reports(make):
    s, (seq,) = make(1, status=SequenceStatus.WAITING_FOR_REMOTE_KVS)
    assert not s.has_forward_work()
    s.finished_recving_kv_req_ids.append(seq.id)
    assert s.has_forward_work()
    s.finished_recving_kv_req_ids.clear()
    s.failed_recving_kv_req_ids.append(str(seq.id))  # a failure recomputes
    assert s.has_forward_work()


def test_one_unblocked_request_among_blocked_ones_is_work(make):
    s, (a, b, _c) = make(3)
    b.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    conn = s.kv_connector
    conn.prefetch_lookups([a])  # a's lookup is in flight
    conn.answer_on_prefetch = True  # c's answers when sent
    assert s.has_forward_work()


def test_aborted_waiting_request_needs_a_pass(make):
    s, _ = make(1, status=SequenceStatus.ABORTED)
    assert s.has_forward_work()


def test_unschedulable_request_needs_a_pass(make, seq_factory):
    # Longer than max_model_len: a pass rejects it, whatever its lookup does.
    s, _ = make()
    s.waiting.append(seq_factory(list(range(100))))
    assert s._unschedulable_reason(s.waiting[0]) is not None
    assert s.has_forward_work()


def test_detachable_lookup_is_not_blocked(make, monkeypatch):
    # Phase 2 admits a request whose lookup it can detach (an HBM hit the
    # tier agrees to skip): the vote must not hold it back.
    s, _ = make(1)
    monkeypatch.setattr(s, "_lookup_detach_enabled", lambda q: True)
    monkeypatch.setattr(s, "_can_detach_tier_lookup", lambda q, hit: True)
    assert s.has_forward_work()


def test_lookup_block_is_bounded_by_the_defer_window(make, monkeypatch):
    monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "0.5")
    clock = _Clock(monkeypatch)
    s, _ = make(1)
    assert not s.has_forward_work()  # t=100.0, first seen
    clock.now = 100.3
    assert not s.has_forward_work()
    clock.now = 100.6
    assert s.has_forward_work()  # past the window: a pass decides


def test_window_expiry_admits_on_the_next_pass(make, monkeypatch):
    """The vote's window and Phase 2's run together. Before, the connector's
    window started only at the first pass, i.e. after the vote's window had
    run out, so a stuck lookup held the request for two windows."""
    monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "0.5")
    clock = _Clock(monkeypatch)
    s, (seq,) = make(1)
    assert not s.has_forward_work()
    clock.now = 100.6
    assert s.has_forward_work()
    _, scheduled = s.schedule()
    assert list(scheduled) == [seq.id]  # admitted without the tier answer


def test_blocked_since_is_forgotten_for_requests_that_left(make):
    s, (seq,) = make(1)
    assert not s.has_forward_work()
    assert seq.id in s._forward_blocked_since
    s.waiting.clear()
    assert not s.has_forward_work()
    assert not s._forward_blocked_since


def test_blocked_since_stays_bounded_while_runnable_work_remains(make, seq_factory):
    """A vote that returns early (runnable work) skips the end-of-scan
    cleanup; entries of requests that left `waiting` meanwhile must still go,
    or the map grows with request history under sustained traffic."""
    s, _ = make()
    for _ in range(50):
        blocked = seq_factory(list(range(16)))
        runnable = seq_factory(list(range(16)))
        s.waiting.extend([blocked, runnable])
        s.kv_connector.prefetch_lookups([blocked])  # in flight
        s.kv_connector.answer_on_prefetch = True  # `runnable` answers at once
        assert s.has_forward_work()  # returns at `runnable`
        assert blocked.id in s._forward_blocked_since
        s.kv_connector.answer_on_prefetch = False
        s.waiting.clear()  # both admitted / finished by later passes
        s.running.append(runnable)  # the running fast path
        assert s.has_forward_work()
        s.running.clear()
        assert len(s._forward_blocked_since) <= len(s.waiting)
    assert not s._forward_blocked_since


class _StickyLookupConnector(_LookupConnector):
    """A tier that keeps deferring a request past `OFFLOAD_LOOKUP_DEFER_S`
    (e.g. the native restore-slot hold): only the vote's own bound ends it."""

    def lookup_would_defer(self, seq):
        return seq.id in self.in_flight


def test_vote_bounds_a_lookup_the_tier_keeps_deferring(make, monkeypatch):
    monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "0.5")
    clock = _Clock(monkeypatch)
    s, _ = make(1)
    s.kv_connector = _StickyLookupConnector()
    assert not s.has_forward_work()  # t=100.0, first seen
    clock.now = 100.6
    assert s.kv_connector.lookup_would_defer(s.waiting[0])  # still deferring
    assert s.has_forward_work()  # the vote's window has run out: a pass


def test_request_held_at_the_pd_admission_gate_is_blocked(make):
    """A P request waiting for D's reservation (chunked transfer, early
    handoff): Phase 2 skips it, so the vote must not ask for a pass."""
    s, (seq,) = make(1)
    conn = s.kv_connector
    conn.answer_on_prefetch = True  # its lookup is no obstacle
    ready = {"d": False}
    conn.admission_ready = lambda q: ready["d"]
    assert not s.has_forward_work()
    _, scheduled = s.schedule()
    assert scheduled == {}  # Phase 2 agrees: nothing to forward
    ready["d"] = True  # D reserved its pages
    assert s.has_forward_work()
    _, scheduled = s.schedule()
    assert list(scheduled) == [seq.id]


def test_rejection_is_checked_before_the_load(make, seq_factory):
    """Phase 2 rejects an unschedulable request at once, load in flight or
    not; the vote must not wait for the load first."""
    s, _ = make()
    seq = seq_factory(list(range(100)))  # longer than max_model_len
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    s.waiting.append(seq)
    assert s._still_loading(seq)
    assert s._unschedulable_reason(seq) is not None
    assert s.has_forward_work()


# ------------------------------ busy loop ---------------------------------- #


class Loop:
    """A one-rank DPEngineCoreProc busy loop with its side effects recorded.

    `script` maps pass index -> callable() run at the start of that pass, so a
    test can make requests arrive, lookups answer, loads land or peers go
    quiet between passes. `state.peers` is the MAX the other ranks add to the
    vote. At pass `shutdown_at` the input pull asks for shutdown, which the
    loop honours once the group has no work.

    The engine step admits every waiting request that is not blocked (a real
    forward, recorded as "schedule") and otherwise lets the loop run a DP
    dummy ("dummy"), as `_process_engine_step` returning False does.
    """

    def __init__(self, scheduler, *, enabled, shutdown_at, script=None, peers=0):
        p = DPEngineCoreProc.__new__(DPEngineCoreProc)
        p.label = "dp0"
        p.scheduler = scheduler
        p._shutting_down = False
        p._report_backlog = False
        p._is_rl_weights_offloaded = False
        p._weight_sync_fenced = False
        p._generation_paused = False
        p._idle_when_blocked = enabled
        p._dp_group_hidden_work = False
        p.dp_group = None  # a single rank: the interval agreement is local
        self.events = []
        self.votes = []
        self.passes = 0
        self.state = SimpleNamespace(peers=peers, peer_hidden=False)
        it = itertools.count()
        script = script or {}

        def tick(*_):
            # The first call of every pass. Not the input pull: once it has
            # asked for shutdown the loop stops pulling.
            i = next(it)
            self.passes = i + 1
            assert i < shutdown_at + 50, "busy loop did not converge"
            if i in script:
                script[i]()

        def pull():
            return self.passes - 1 >= shutdown_at

        p.utility_queue = None
        p.utility_handler = SimpleNamespace(
            process_queue=tick, push_metrics=lambda: None
        )

        def sync(local, shutdown=False, held=False, local_hidden=False):
            # The real one reduces `local_hidden` into `_dp_group_hidden_work`.
            self.votes.append(local)
            p._dp_group_hidden_work = local_hidden or self.state.peer_hidden
            return (local or bool(self.state.peers)), shutdown, held

        def blocked(seq):
            if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS:
                return scheduler._still_loading(seq)
            return scheduler._head_waits_on_tier_lookup(seq)

        def step():
            ready = [q for q in scheduler.waiting if not blocked(q)]
            for q in ready:
                scheduler.waiting.remove(q)
            if ready:
                self.events.append("schedule")
            return bool(ready)

        p.pull_and_process_input_queue = pull
        p._sync_dp_state = sync
        p._process_engine_step = step
        p._execute_dummy_batch = lambda: self.events.append("dummy")
        p.has_pending_kv_work = lambda: bool(scheduler.deferred_free_blocks)
        p._advance_idle_kv_transfer = lambda *a, **k: self.events.append("drain")
        p._drain_kv_work_at_exit = lambda: self.events.append("exit")
        p._report_held_kv_work = lambda: None
        scheduler.heartbeat_throughput = lambda now: None
        scheduler.engine_stats = SimpleNamespace(
            note_idle=lambda now: None,
            note_busy=lambda now: None,
            bump_step=lambda: None,
        )
        scheduler.prefill_delayer = None
        scheduler.publish_kv_events = lambda: None
        scheduler.shutdown_kv_events = lambda: None
        scheduler.close_kv_connector = lambda: None
        self.proc = p

    def run(self, monkeypatch):
        monkeypatch.setattr(ec.tracing, "shutdown", lambda: None)
        monkeypatch.setattr(ec, "BLOCKED_IDLE_SLEEP_S", 0.0)
        self.proc.busy_loop()
        return self.events


@pytest.mark.parametrize("enabled", [False, True])
def test_deferred_frees_alone(make, monkeypatch, enabled):
    # A finished request waits for D to pull its KV for passes 0-4.
    s, _ = make(conn=False)
    s.deferred_free_blocks[1] = object()
    events = Loop(
        s, enabled=enabled, shutdown_at=5, script={5: s.deferred_free_blocks.clear}
    ).run(monkeypatch)
    if enabled:
        assert events == ["drain"] * 5 + ["exit"]  # polled, no forward
    else:
        assert events == ["dummy"] * 5 + ["exit"]  # a DP dummy per pass


def test_on_still_runs_dummies_for_a_busy_peer(make, monkeypatch):
    # Lockstep: a peer's forward needs this rank's dummy whatever it votes.
    s, _ = make(conn=False)
    s.deferred_free_blocks[1] = object()
    peers = SimpleNamespace(loop=None)

    def quiet():
        peers.loop.state.peers = 0
        s.deferred_free_blocks.clear()

    loop = peers.loop = Loop(s, enabled=True, shutdown_at=3, script={3: quiet}, peers=1)
    events = loop.run(monkeypatch)
    assert events == ["dummy"] * 3 + ["exit"]


@pytest.mark.parametrize("enabled", [False, True])
def test_request_waiting_on_its_lookup(make, monkeypatch, seq_factory, enabled):
    s, _ = make()
    conn = s.kv_connector
    seq = seq_factory(list(range(16)))
    script = {
        1: lambda: s.waiting.append(seq),  # arrives
        4: lambda: conn.answer(seq),  # its lookup answers
    }
    if not enabled:
        # Off, the first pass is what sends the lookup.
        script[1] = lambda: (s.waiting.append(seq), conn.prefetch_lookups([seq]))
    loop = Loop(s, enabled=enabled, shutdown_at=6, script=script)
    events = loop.run(monkeypatch)
    if enabled:
        # Blocked passes 1-3 poll instead of forwarding; pass 4 admits it.
        assert loop.votes == [False] * 4 + [True] + [False] * 2
        assert events == ["drain"] * 3 + ["schedule", "exit"]
    else:
        assert events == ["dummy"] * 3 + ["schedule", "exit"]


@pytest.mark.parametrize("blocked", [False, True])
def test_idle_loop_sleeps_only_for_blocked_requests(
    make, monkeypatch, seq_factory, blocked
):
    """The 0.5 ms sleep is for requests blocked on a lookup or load (their
    threads need the GIL); deferred frees alone poll as with the flag off."""
    if blocked:
        s, (seq,) = make(1, status=SequenceStatus.WAITING_FOR_REMOTE_KVS)
        script = {3: lambda: s.finished_recving_kv_req_ids.append(seq.id)}
    else:
        s, _ = make(conn=False)
        s.deferred_free_blocks[1] = object()
        script = {3: s.deferred_free_blocks.clear}
    sleeps = []
    loop = Loop(s, enabled=True, shutdown_at=4, script=script)
    monkeypatch.setattr(ec.time, "sleep", sleeps.append)
    loop.run(monkeypatch)
    assert sleeps == ([ec.BLOCKED_IDLE_SLEEP_S] * 3 if blocked else [])


def test_vote_runs_only_on_rounds_that_sync(make, monkeypatch):
    """`ATOM_DP_SYNC_INTERVAL` > 1: rounds that skip the sync run paired and
    must not call `has_forward_work` (no lookups sent, no windows started).
    After the sync that saw work, interval - 1 paired rounds still run before
    the next sync lets the group idle."""
    monkeypatch.setenv("ATOM_DP_SYNC_INTERVAL", "3")
    s, _ = make(conn=False)
    s.deferred_free_blocks[1] = object()
    calls = []
    vote = s.has_forward_work
    monkeypatch.setattr(s, "has_forward_work", lambda: calls.append(1) or vote())
    holder = SimpleNamespace(loop=None)

    def peer_quiet():
        holder.loop.state.peers = 0

    # Script keys count syncs (utility commands run only on rounds that sync).
    loop = holder.loop = Loop(
        s,
        enabled=True,
        shutdown_at=3,
        script={1: peer_quiet, 3: s.deferred_free_blocks.clear},
        peers=1,
    )
    events = loop.run(monkeypatch)
    # Sync 0 sees the busy peer: its dummy, then 2 rounds without a sync.
    # Syncs 1-2 idle-poll the deferred free; sync 3 settles and exits.
    assert events == ["dummy"] * 3 + ["drain"] * 2 + ["exit"]
    assert len(calls) == len(loop.votes) == 4


def test_on_lets_a_landed_load_through(make, monkeypatch):
    s, (seq,) = make(1, status=SequenceStatus.WAITING_FOR_REMOTE_KVS)
    script = {2: lambda: s.finished_recving_kv_req_ids.append(seq.id)}
    loop = Loop(s, enabled=True, shutdown_at=4, script=script)
    events = loop.run(monkeypatch)
    assert loop.votes == [False, False, True, False, False]
    # The blocked passes poll the load (drain) with nothing deferred.
    assert events == ["drain", "drain", "schedule", "exit"]


@pytest.mark.parametrize("enabled", [False, True])
def test_shutdown_waits_for_blocked_work_as_with_the_flag_off(
    make, monkeypatch, enabled
):
    """Shutdown asked at pass 1 while a deferred free is pending until pass 6.
    Off, the loop runs dummies until it settles and then exits. On, the
    group's "no work" vote would let it exit at pass 1 with the free still
    pending; instead the ranks switch back to the off vote once the group
    agreed to shut down, and exit only when it settled."""
    s, _ = make(conn=False)
    s.deferred_free_blocks[1] = object()
    loop = Loop(
        s, enabled=enabled, shutdown_at=1, script={6: s.deferred_free_blocks.clear}
    )
    events = loop.run(monkeypatch)
    assert events[-1] == "exit"
    assert not s.deferred_free_blocks  # settled before the exit
    assert loop.passes == 7  # exits on the pass where it settled
    if enabled:
        # Pass 0 polls; pass 1 sees the shutdown and re-votes; 2-5 run the
        # dummies the off vote runs.
        assert events == ["drain"] + ["dummy"] * 4 + ["exit"]
    else:
        assert events == ["dummy"] * 6 + ["exit"]


@pytest.mark.parametrize("enabled", [False, True])
def test_shutdown_with_nothing_pending_exits_at_once(make, monkeypatch, enabled):
    # Nothing hidden anywhere: no extra pass, on or off (off is main's loop).
    s, _ = make()
    loop = Loop(s, enabled=enabled, shutdown_at=0)
    assert loop.run(monkeypatch) == ["exit"]
    assert loop.passes == 1
    assert loop.votes == [False]


@pytest.mark.parametrize("enabled", [False, True])
def test_a_peers_hidden_work_holds_every_rank_at_shutdown(make, monkeypatch, enabled):
    """The re-vote is decided on a reduced value (`_dp_group_hidden_work`), not
    on this rank's setting: an idle rank, flag on or off, takes the same extra
    pass as a peer whose blocked work the narrower vote hid, so the ranks run
    the same collectives before they exit."""
    s, _ = make()
    holder = SimpleNamespace(loop=None)

    def peer_votes_as_off():
        holder.loop.state.peer_hidden = False
        holder.loop.state.peers = 1

    def peer_settles():
        holder.loop.state.peers = 0

    loop = holder.loop = Loop(
        s,
        enabled=enabled,
        shutdown_at=0,
        script={1: peer_votes_as_off, 3: peer_settles},
    )
    loop.state.peer_hidden = True  # pass 0: the peer's work is hidden
    events = loop.run(monkeypatch)
    # Pass 0 agrees "no work" but the peer's work is hidden: no exit. Passes
    # 1-2 the peer, now voting as off, has work: this rank runs its dummies.
    # Pass 3 everything settled: exit.
    assert events == ["dummy", "dummy", "exit"]
    assert loop.passes == 4


def test_a_held_rank_sends_no_lookup_and_starts_no_window(make, monkeypatch):
    """While held (paused, weights offloaded or fenced) no pass runs, so the
    vote must not send lookups or start their defer windows either; off never
    did, as `is_finished` sends nothing."""
    s, (seq,) = make(1)
    conn = s.kv_connector
    seen = {}
    holder = SimpleNamespace(loop=None)

    def resume():
        seen["submitted"] = list(conn.submitted)
        seen["windows"] = dict(conn.defer_since)
        holder.loop.proc._generation_paused = False
        conn.answer_on_prefetch = True  # its lookup answers once sent

    loop = holder.loop = Loop(s, enabled=True, shutdown_at=3, script={3: resume})
    loop.proc._generation_paused = True
    events = loop.run(monkeypatch)
    assert seen == {"submitted": [], "windows": {}}
    assert events == ["schedule", "exit"]
    assert conn.submitted == [seq.id]


def test_sync_reduces_hidden_work_with_the_vote(monkeypatch):
    # Every rank sends the same four flags, set or not, so the all_reduce
    # matches across ranks whatever each one's setting.
    p = DPEngineCoreProc.__new__(DPEngineCoreProc)
    p.label = "dp0"
    p._shutting_down = False
    p.dp_group = object()  # a real group: the flags go through all_reduce
    sent = []

    def all_reduce(tensor, op=None, group=None):
        sent.append(tensor.tolist())
        tensor[3] = 1  # a peer's hidden work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    assert p._sync_dp_state(False, True, False) == (False, True, False)
    assert sent == [[0, 1, 0, 0]]
    assert p._dp_group_hidden_work is True
    assert p._sync_dp_state(True, False, False, local_hidden=True) == (
        True,
        False,
        False,
    )
    assert sent[-1] == [1, 0, 0, 1]


# ------------------------------- switch ------------------------------------ #


@pytest.mark.parametrize(
    ("env", "delayer", "expected"),
    [("0", None, False), ("1", None, True), ("1", object(), False)],
)
def test_switch(make, monkeypatch, env, delayer, expected):
    monkeypatch.setenv("ATOM_DP_IDLE_WHEN_BLOCKED", env)
    p = DPEngineCoreProc.__new__(DPEngineCoreProc)
    p.label = "dp0"
    p.scheduler, _ = make()
    p.scheduler.prefill_delayer = delayer
    assert p._resolve_idle_when_blocked() is expected


def test_default_is_off(make, monkeypatch):
    monkeypatch.delenv("ATOM_DP_IDLE_WHEN_BLOCKED", raising=False)
    p = DPEngineCoreProc.__new__(DPEngineCoreProc)
    p.label = "dp0"
    p.scheduler, _ = make()
    p.scheduler.prefill_delayer = None
    assert p._resolve_idle_when_blocked() is False
