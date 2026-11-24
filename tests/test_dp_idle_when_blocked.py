# SPDX-License-Identifier: MIT
"""ATOM_DP_IDLE_WHEN_BLOCKED: a DP rank votes "unfinished" only when a pass
could forward something (`Scheduler.has_forward_work`).

CPU only. The scheduler method runs on a bare `Scheduler` with the fields it
reads; the busy loop runs on a bare `DPEngineCoreProc` whose collective,
engine step, dummy forward and KV drain are recorded instead of run.
"""

import itertools
from collections import deque
from types import SimpleNamespace

import pytest

from atom.model_engine import engine_core as ec
from atom.model_engine.engine_core import DPEngineCoreProc
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus


class FakeConnector:
    """Async tier lookups: `pending` ids are in flight; prefetch submits."""

    def __init__(self):
        self.pending = set()
        self.submitted = []
        self.answer_on_prefetch = False

    def prefetch_lookups(self, seqs):
        for seq in seqs:
            if seq.id not in self.submitted:
                self.submitted.append(seq.id)
                if not self.answer_on_prefetch:
                    self.pending.add(seq.id)

    def lookup_would_defer(self, seq):
        return seq.id in self.pending


def _seq(sid, status=SequenceStatus.WAITING):
    return SimpleNamespace(id=sid, status=status)


def _sched(waiting=(), running=(), rejected=(), deferred=None, conn=None):
    s = Scheduler.__new__(Scheduler)
    s.waiting = deque(waiting)
    s.running = list(running)
    s._rejected = list(rejected)
    s.deferred_free_blocks = dict(deferred or {})
    s.kv_connector = conn
    s.finished_recving_kv_req_ids = []
    s.failed_recving_kv_req_ids = []
    return s


# ------------------------------ scheduler ---------------------------------- #


def test_deferred_frees_alone_are_not_forward_work():
    s = _sched(deferred={7: object()})
    assert not s.is_finished()  # the old vote: a dummy forward every pass
    assert not s.has_forward_work()


def test_running_and_rejected_are_work():
    assert _sched(running=[_seq(1, SequenceStatus.RUNNING)]).has_forward_work()
    assert _sched(rejected=[_seq(2)]).has_forward_work()


def test_waiting_without_connector_is_work():
    assert _sched(waiting=[_seq(1)]).has_forward_work()


def test_fresh_request_is_prefetched_and_blocks_on_its_lookup():
    conn = FakeConnector()
    s = _sched(waiting=[_seq(1)], conn=conn)
    assert not s.has_forward_work()
    assert conn.submitted == [1]
    conn.pending.clear()  # the lookup answers
    assert s.has_forward_work()


def test_lookup_answered_at_once_is_work():
    conn = FakeConnector()
    conn.answer_on_prefetch = True
    assert _sched(waiting=[_seq(1)], conn=conn).has_forward_work()


def test_load_in_flight_blocks_until_it_reports():
    s = _sched(waiting=[_seq(3, SequenceStatus.WAITING_FOR_REMOTE_KVS)])
    assert not s.has_forward_work()
    s.finished_recving_kv_req_ids.append(3)
    assert s.has_forward_work()
    s.finished_recving_kv_req_ids.clear()
    s.failed_recving_kv_req_ids.append("3")  # a failure recomputes: work too
    assert s.has_forward_work()


def test_one_unblocked_request_among_blocked_ones_is_work():
    conn = FakeConnector()
    conn.pending.add(1)
    conn.submitted.append(1)
    s = _sched(
        waiting=[
            _seq(1),
            _seq(2, SequenceStatus.WAITING_FOR_REMOTE_KVS),
            _seq(3),
        ],
        conn=conn,
    )
    conn.answer_on_prefetch = True  # 3's lookup answers when sent
    assert s.has_forward_work()


def test_aborted_waiting_request_needs_a_pass():
    conn = FakeConnector()
    s = _sched(waiting=[_seq(1, SequenceStatus.ABORTED)], conn=conn)
    assert s.has_forward_work()


def test_lookup_block_is_bounded_by_the_defer_window(monkeypatch):
    monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "0.5")
    clock = itertools.count(100.0, 0.3)
    monkeypatch.setattr(
        "atom.model_engine.scheduler.time.monotonic", lambda: next(clock)
    )
    conn = FakeConnector()
    s = _sched(waiting=[_seq(1)], conn=conn)
    assert not s.has_forward_work()  # t=100.0, first seen
    assert not s.has_forward_work()  # t=100.3
    assert s.has_forward_work()  # t=100.6: past the window, a pass decides


def test_blocked_since_is_forgotten_for_requests_that_left():
    conn = FakeConnector()
    s = _sched(waiting=[_seq(1)], conn=conn)
    assert not s.has_forward_work()
    assert 1 in s._forward_blocked_since
    s.waiting.clear()
    assert not s.has_forward_work()
    assert not s._forward_blocked_since


# ------------------------------ busy loop ---------------------------------- #


class Loop:
    """A one-rank DPEngineCoreProc busy loop with its side effects recorded.

    `script` maps pass index -> callable() run at that pass's input pull, so a
    test can make requests arrive, lookups answer, loads land or peers go
    quiet between passes. `state.peers` is the MAX the other ranks add to the
    vote. The pull from pass `last` on asks for shutdown, which the loop
    honours once the group has no work.

    The engine step admits every waiting request that is not blocked (a real
    forward, recorded as "schedule") and otherwise lets the loop run a DP
    dummy ("dummy"), as `_process_engine_step` returning False does.
    """

    def __init__(self, scheduler, *, enabled, last, script=None, peers=0):
        p = DPEngineCoreProc.__new__(DPEngineCoreProc)
        p.label = "dp0"
        p.scheduler = scheduler
        p._shutting_down = False
        p._report_backlog = False
        p._is_rl_weights_offloaded = False
        p._weight_sync_fenced = False
        p._generation_paused = False
        p._idle_when_blocked = enabled
        p.utility_queue = None
        p.utility_handler = SimpleNamespace(
            process_queue=lambda *a: None, push_metrics=lambda: None
        )
        self.events = []
        self.votes = []
        self.state = SimpleNamespace(peers=peers)
        it = itertools.count()
        script = script or {}

        def pull():
            i = next(it)
            assert i < last + 50, "busy loop did not converge"
            if i in script:
                script[i]()
            return i >= last

        def sync(local, shutdown=False, held=False):
            self.votes.append(local)
            return (local or bool(self.state.peers)), shutdown, held

        def blocked(seq):
            conn = scheduler.kv_connector
            return scheduler._still_loading(seq) or (
                conn is not None and conn.lookup_would_defer(seq)
            )

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
        p._drain_kv_work_at_exit = lambda: None
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
        self.proc.busy_loop()
        return self.events


@pytest.mark.parametrize("enabled", [False, True])
def test_deferred_frees_alone(monkeypatch, enabled):
    # A finished request waits for D to pull its KV for passes 0-4.
    s = _sched(deferred={1: object()})
    events = Loop(
        s, enabled=enabled, last=5, script={5: s.deferred_free_blocks.clear}
    ).run(monkeypatch)
    if enabled:
        assert events == ["drain"] * 5  # polled, no forward
    else:
        assert events == ["dummy"] * 5  # the old cost: a DP dummy per pass


def test_on_still_runs_dummies_for_a_busy_peer(monkeypatch):
    # Lockstep: a peer's forward needs this rank's dummy whatever it votes.
    s = _sched(deferred={1: object()})
    peers = SimpleNamespace(loop=None)

    def quiet():
        peers.loop.state.peers = 0
        s.deferred_free_blocks.clear()

    loop = peers.loop = Loop(s, enabled=True, last=3, script={3: quiet}, peers=1)
    events = loop.run(monkeypatch)
    assert events == ["dummy"] * 3


@pytest.mark.parametrize("enabled", [False, True])
def test_request_waiting_on_its_lookup(monkeypatch, enabled):
    conn = FakeConnector()
    s = _sched(conn=conn)
    script = {
        1: lambda: s.waiting.append(_seq(9)),  # arrives
        4: conn.pending.clear,  # its lookup answers
    }
    if not enabled:
        # Off, the first pass is what sends the lookup.
        script[1] = lambda: (s.waiting.append(_seq(9)), conn.pending.add(9))
    loop = Loop(s, enabled=enabled, last=6, script=script)
    events = loop.run(monkeypatch)
    if enabled:
        # Blocked passes 1-3 poll instead of forwarding; pass 4 admits it.
        assert loop.votes == [False, False, False, False, True, False, False]
        assert events == ["drain"] * 3 + ["schedule"]
    else:
        assert events == ["dummy"] * 3 + ["schedule"]


def test_on_lets_a_landed_load_through(monkeypatch):
    s = _sched(waiting=[_seq(4, SequenceStatus.WAITING_FOR_REMOTE_KVS)])
    script = {2: lambda: s.finished_recving_kv_req_ids.append(4)}
    loop = Loop(s, enabled=True, last=4, script=script)
    events = loop.run(monkeypatch)
    assert loop.votes == [False, False, True, False, False]
    # The blocked passes poll the load (drain) with nothing deferred.
    assert events == ["drain", "drain", "schedule"]


# ------------------------------- switch ------------------------------------ #


@pytest.mark.parametrize(
    ("env", "delayer", "expected"),
    [("0", None, False), ("1", None, True), ("1", object(), False)],
)
def test_switch(monkeypatch, env, delayer, expected):
    monkeypatch.setenv("ATOM_DP_IDLE_WHEN_BLOCKED", env)
    p = DPEngineCoreProc.__new__(DPEngineCoreProc)
    p.label = "dp0"
    p.scheduler = _sched()
    p.scheduler.prefill_delayer = delayer
    assert p._resolve_idle_when_blocked() is expected


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("ATOM_DP_IDLE_WHEN_BLOCKED", raising=False)
    p = DPEngineCoreProc.__new__(DPEngineCoreProc)
    p.label = "dp0"
    p.scheduler = _sched()
    p.scheduler.prefill_delayer = None
    assert p._resolve_idle_when_blocked() is False


def test_blocked_rank_yields_the_gil_between_polls(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ec.time, "sleep", sleeps.append)
    s = _sched(waiting=[_seq(4, SequenceStatus.WAITING_FOR_REMOTE_KVS)])
    script = {2: lambda: s.finished_recving_kv_req_ids.append(4)}
    Loop(s, enabled=True, last=4, script=script).run(monkeypatch)
    assert sleeps == [ec.BLOCKED_IDLE_SLEEP_S] * 2  # the two blocked passes


def test_deferred_frees_alone_do_not_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ec.time, "sleep", sleeps.append)
    s = _sched(deferred={1: object()})
    Loop(s, enabled=True, last=3, script={3: s.deferred_free_blocks.clear}).run(
        monkeypatch
    )
    assert sleeps == []
