# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Admission does not wait for a tier lookup that cannot load anything.

The contract has two halves:

1. *Safety*: a lookup is skipped only when no answer could arm a load. Every
   load path refuses `need = lmc - hbm < OFFLOAD_MIN_LOAD_TOKENS`, and a tier
   hit never exceeds the prompt, so `num_tokens - hbm < min_load` makes the
   answer irrelevant. The threshold is the connector's own.
2. *LRU*: the lookup is still sent (fire-and-forget). On the LMCache MP server
   it is what refreshes the prefix's L1 recency and opens the session whose
   `end_session` touches the request's chunks. Its answer is discarded and its
   read locks released once it arrives.
"""

from types import SimpleNamespace

import pytest
from test_lmcache_mp import _config
from test_lmcache_mp_async_lookup import CHUNK, _Adapter, _Client

from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import LoadSpec
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler

MIN_LOAD = 32


def _scheduler(monkeypatch, *, chunks=8, lookup_done=False):
    """An MP scheduler over the async-capable fake adapter."""
    adapter = _Adapter(_Client(chunks, lookup_done=lookup_done))
    client = mp_lookup._MPLookupClient(
        adapter, config=_config(), timeout=1.0, poll_interval=0.01
    )
    sched = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
    )
    sched._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        sched, _config(), chunk_size=CHUNK, lookup_client=client
    )
    sched._min_load_tokens = MIN_LOAD
    return sched, client, adapter


def _seq(n=64, sid=7):
    return SimpleNamespace(
        id=sid,
        num_tokens=n,
        num_prompt_tokens=n,
        num_cached_tokens=0,
        token_ids=list(range(n)),
        block_table=[],
    )


# ── safety: the skip never hides a load the lookup could arm ─────────────


@pytest.mark.parametrize(
    "hbm,cannot", [(64 - MIN_LOAD + 1, True), (64 - MIN_LOAD, False), (0, False)]
)
def test_threshold_is_the_connectors_min_load(monkeypatch, hbm, cannot):
    sched, _, _ = _scheduler(monkeypatch)
    assert sched._load_ruled_out(_seq(), hbm) is cannot


@pytest.mark.parametrize("hit", [0, 40, 48, 60, 64])
def test_any_tier_hit_inside_the_skip_window_is_refused_as_a_load(monkeypatch, hit):
    """The property the skip relies on, checked against the real decision."""
    sched, _, _ = _scheduler(monkeypatch)
    seq = _seq()
    hbm = 64 - MIN_LOAD + 4  # inside the skip window
    assert sched._load_ruled_out(seq, hbm)
    seq.num_cached_tokens = hbm
    spec = LoadSpec(hbm_cached_tokens=hbm, lmcache_cached_tokens=hit, can_load=False)
    should_load, reason, *_ = sched._decide_load_after_alloc(seq, spec)
    assert not should_load, reason


def test_a_gap_the_tier_could_fill_is_not_skipped(monkeypatch):
    sched, _, adapter = _scheduler(monkeypatch)
    assert sched.detach_lookup(_seq(), 64 - MIN_LOAD) is False
    assert adapter._client.lookups == []


# ── LRU: the lookup is still sent, never waited on, then discarded ──────


def test_skip_sends_the_lookup_without_waiting_and_orphans_it(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=False)
    seq = _seq()

    assert sched.detach_lookup(seq, 60) is True
    # Sent: the server refreshes L1 recency and opens the session.
    assert len(adapter._client.lookups) == 1
    # Not waited on, not consumed: no load is armed for it.
    assert "7" not in sched._load_specs
    assert "7" in client._orphans
    # Save floor from HBM, so the resident prefix is not stored again.
    assert sched._hit_save_floors["7"] == 60


def test_orphaned_lookup_releases_its_locks_when_it_answers(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=False)
    sched.detach_lookup(_seq(), 60)
    adapter._client.lookup_future.done = True
    client.pump()
    assert adapter.freed, "read locks taken by the answered lookup must be freed"
    assert "7" not in client._orphans


def test_a_prefetched_lookup_is_not_sent_twice(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=False)
    seq = _seq()
    sched.prefetch_lookups([seq])
    assert len(adapter._client.lookups) == 1
    assert sched.detach_lookup(seq, 60) is True
    assert len(adapter._client.lookups) == 1
    assert "7" in client._orphans


def test_an_armed_load_is_finished_by_the_ordinary_path(monkeypatch):
    sched, _, _ = _scheduler(monkeypatch)
    seq = _seq()
    sched._load_specs["7"] = LoadSpec(
        hbm_cached_tokens=0, lmcache_cached_tokens=48, can_load=False
    )
    assert sched.detach_lookup(seq, 60) is False


# ── scheduler gate ───────────────────────────────────────────────────────


class _Conn:
    is_offload = True

    def __init__(self, cannot=True):
        self.cannot = cannot
        self.probed = []
        self.committed = []

    def can_detach_lookup(self, seq, hbm):
        self.probed.append(hbm)
        return self.cannot

    def detach_lookup(self, seq, hbm):
        self.committed.append(hbm)
        return True


def _gate(monkeypatch, *, conn, state_offload=None, hit_blocks=10):
    from atom.model_engine import scheduler as sched_mod
    from atom.model_engine.scheduler import Scheduler

    monkeypatch.setattr(sched_mod.envs, "OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT", True)
    s = Scheduler.__new__(Scheduler)
    s.kv_connector = conn
    s._local_prefill_coalescing = False
    s._connector_flag = lambda name: getattr(conn, name, False)
    s.block_manager = SimpleNamespace(
        state_offload=state_offload,
        hash_block_size=4,
        can_allocate=lambda seq, **kw: hit_blocks,
    )
    seq = SimpleNamespace(id=7, kv_transfer_params=None)
    if not s._lookup_detach_enabled(seq):
        return False, conn
    return s._can_detach_tier_lookup(seq, s._probe_hbm_prefix(seq)), conn


def test_scheduler_probe_is_pure_and_uses_the_hbm_frontier(monkeypatch):
    skipped, conn = _gate(monkeypatch, conn=_Conn())
    assert skipped and conn.probed == [40]
    assert conn.committed == []  # committed only after admission


def test_scheduler_keeps_the_lookup_when_the_connector_could_load(monkeypatch):
    skipped, conn = _gate(monkeypatch, conn=_Conn(cannot=False))
    assert not skipped and conn.committed == []


def test_scheduler_keeps_the_lookup_when_joint_state_offload_is_on(monkeypatch):
    # The joint path is where a tier hit can move the HBM resume point.
    skipped, conn = _gate(monkeypatch, conn=_Conn(), state_offload=object())
    assert not skipped and conn.probed == []


def test_scheduler_keeps_the_lookup_on_a_refused_probe(monkeypatch):
    skipped, conn = _gate(monkeypatch, conn=_Conn(), hit_blocks=-1)
    assert not skipped and conn.probed == []


# ── session ordering: free the locks, then end the session ───────────────


def _with_session_log(adapter):
    events = []
    orig_free = adapter.free_lookup_locks

    def free(**kw):
        events.append(("free", kw["request_id"]))
        orig_free(**kw)

    adapter.free_lookup_locks = free
    adapter.end_session = lambda rid: events.append(("end", rid))
    return events


def test_session_outlives_an_unanswered_fire_and_forget_lookup(monkeypatch):
    """`end_session` before the orphan answers would leave its state-group
    read locks pinned until the L1 read TTL (the server recreates an empty
    session for the late `free_lookup_locks`)."""
    sched, _client, adapter = _scheduler(monkeypatch, lookup_done=False)
    events = _with_session_log(adapter)
    seq = _seq()
    sched.detach_lookup(seq, 60)

    sched._pending_session_ends()["7"] = seq
    sched._end_finished_sessions()
    assert events == []  # lookup still in flight: session kept

    adapter._client.lookup_future.done = True
    sched._end_finished_sessions()
    kinds = [k for k, _ in events]
    assert kinds == ["free", "end"], events
    assert "7" not in sched._pending_session_ends()


# ── review findings (#2547) ──────────────────────────────────────────────


def test_probe_has_no_side_effects(monkeypatch):
    """A request held back after the probe must keep its prefetched lookup."""
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=False)
    seq = _seq()
    sched.prefetch_lookups([seq])
    for _ in range(3):  # three passes, never admitted
        assert sched.can_detach_lookup(seq, 60)
    assert len(adapter._client.lookups) == 1
    assert "7" not in client._orphans
    assert "7" not in sched._hit_save_floors


def test_no_skip_without_an_async_lookup_to_keep(monkeypatch):
    """A synchronous-only client keeps the ordinary path (LRU side effects)."""
    sched, client, _ = _scheduler(monkeypatch)
    client._async_supported = False
    assert sched.can_detach_lookup(_seq(), 60) is False
    assert sched.detach_lookup(_seq(), 60) is False


def test_answered_orphan_frees_locks_while_its_request_is_still_running(monkeypatch):
    """Pumped every pass, not only when a session is due to end."""
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=False)
    sched.detach_lookup(_seq(), 60)
    adapter._client.lookup_future.done = True
    assert sched._pending_session_ends() == {}  # request not finished
    sched._end_finished_sessions()  # the per-step call from build_connector_meta
    assert adapter.freed
    assert "7" not in client._orphans


class _Sub:
    def __init__(self, ok):
        self.ok = ok
        self.committed = 0

    def can_detach_lookup(self, seq, hbm):
        return self.ok

    def detach_lookup(self, seq, hbm):
        self.committed += 1
        return True


def test_composite_probe_is_unanimous_and_side_effect_free():
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorScheduler,
    )

    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    a, b = _Sub(True), _Sub(False)
    multi._connectors = [a, b]
    assert multi.can_detach_lookup(None, 60) is False
    assert a.committed == b.committed == 0  # nothing mutated by the probe
    b.ok = True
    assert multi.can_detach_lookup(None, 60) is True
    assert multi.detach_lookup(None, 60) is True
    assert a.committed == b.committed == 1


# ── save floor follows the tier, not HBM (#2547 review) ──────────────────


def _admit_and_track(sched, seq, hbm):
    sched._do_save = True
    assert sched.detach_lookup(seq, hbm)
    seq.num_cached_tokens = hbm
    sched.update_state_after_alloc(seq)
    return sched._save_tracker["7"]


def test_floor_drops_to_the_tier_hit_when_the_tier_holds_less(monkeypatch):
    # Tier answers 8 chunks = 32 tokens, HBM holds 60: chunks [32, 60) must be
    # saved again, or the tier keeps a hole that hides every later chunk.
    sched, _client, adapter = _scheduler(monkeypatch, chunks=8, lookup_done=False)
    entry = _admit_and_track(sched, _seq(), 60)
    assert entry[1] == 60
    adapter._client.lookup_future.done = True
    sched._end_finished_sessions()
    assert entry[1] == 32


def test_floor_stays_when_the_tier_holds_the_whole_hbm_prefix(monkeypatch):
    sched, _client, adapter = _scheduler(monkeypatch, chunks=16, lookup_done=False)
    entry = _admit_and_track(sched, _seq(), 60)
    adapter._client.lookup_future.done = True
    sched._end_finished_sessions()
    assert entry[1] == 60
    assert sched._detached_lookups == {}


# ── a failed status query keeps the session (#2547 review) ───────────────


class _FailingStatusOnce:
    """`query_prefetch_status` whose first answer raises, then succeeds."""

    def __init__(self, client, failures=1):
        self.client = client
        self.failures = failures

    def __call__(self, request_id):
        fut = _Future_cls(self.client.chunks)
        if self.failures:
            self.failures -= 1
            fut.raise_on_result = True
        return fut


class _Future_cls:
    def __init__(self, value):
        self.value = value
        self.raise_on_result = False

    def query(self):
        return True

    def result(self, timeout=None):
        if self.raise_on_result:
            raise RuntimeError("status query failed")
        return self.value


def test_failed_status_keeps_the_session_until_locks_are_freed(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=True)
    events = _with_session_log(adapter)
    adapter._client.query_prefetch_status = _FailingStatusOnce(adapter._client)
    seq = _seq()
    sched.detach_lookup(seq, 60)
    sched._pending_session_ends()["7"] = seq

    sched._end_finished_sessions()  # status fails once: retried, still in flight
    assert client.is_pending("7")
    assert events == []

    sched._end_finished_sessions()  # retry answers: free, then end
    assert [k for k, _ in events] == ["free", "end"]


def test_lookup_phase_failure_is_dropped_without_retry(monkeypatch):
    """Never reached the server, so it holds no locks to wait for."""
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=True)

    def boom(_rid):
        raise RuntimeError("send failed")

    adapter._client.lookup_future.result = lambda timeout=None: boom(None)
    sched.detach_lookup(_seq(), 60)
    client.pump()
    assert not client.is_pending("7")
    assert "7" not in client._orphans


# ── robustness (#2547 review) ────────────────────────────────────────────


def test_release_failure_does_not_escape_into_scheduling(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=True)
    seq = _seq()
    sched.prefetch_lookups([seq])
    client.pump()  # answered, not yet consumed

    def broken(**_kw):
        raise RuntimeError("adapter down")

    adapter.free_lookup_locks = broken
    assert _adapter_holds(client, adapter)
    assert sched.detach_lookup(seq, 60)  # discard releases at once
    assert "7" not in client.pending_ids()
    assert not _adapter_holds(client, adapter)  # cleaned despite the failure


def test_engine_keeps_polling_until_a_discarded_lookup_is_released(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=False)
    seq = _seq()
    sched.detach_lookup(seq, 60)
    sched._pending_session_ends()["7"] = seq  # request already finished
    assert sched.has_pending_work()
    adapter._client.lookup_future.done = True
    adapter.end_session = lambda rid: None
    sched._end_finished_sessions()
    assert not client.has_orphans()
    assert not sched.has_pending_work()


def _adapter_holds(client, adapter, lookup_id="7"):
    rid = mp_lookup._mp_session_id(client._config, lookup_id)
    return rid in adapter._pending_lookups or rid in adapter._lookup_results


def test_an_unanswered_orphan_is_given_up_after_its_timeout(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=True)
    pending_future = type(adapter._client.lookup_future)
    # Acked by the server, but the status never answers.
    adapter._client.query_prefetch_status = lambda _rid: pending_future(None, False)
    seq = _seq()
    sched.detach_lookup(seq, 60)
    client.pump()  # lookup phase done: the adapter now tracks the ID
    assert _adapter_holds(client, adapter)
    client._orphan_since["7"] -= mp_lookup._ORPHAN_TIMEOUT_S + 1
    client.pump()
    assert not client.has_orphans()
    assert not client.is_pending("7")
    assert not _adapter_holds(client, adapter)  # no leak
    assert client.submit(list(range(seq.num_tokens)), "7")  # ID reusable


def test_a_failing_requery_is_retried_up_to_the_bound(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=True)
    calls = {"n": 0}

    def status(_rid):
        calls["n"] += 1
        if calls["n"] == 1:  # the status future handed out on transition
            fut = _Future_cls(adapter._client.chunks)
            fut.raise_on_result = True
            return fut
        raise RuntimeError("status channel down")  # every re-query

    adapter._client.query_prefetch_status = status
    sched.detach_lookup(_seq(), 60)
    for _ in range(mp_lookup._ORPHAN_STATUS_RETRIES):
        client.pump()
        assert client.is_pending("7")  # failed re-query: still in flight
    client.pump()  # bound reached
    assert not client.is_pending("7")
    assert not client.has_orphans()
    assert not _adapter_holds(client, adapter)


def test_an_unanswerable_lookup_resets_the_save_floor(monkeypatch):
    """No answer means the tier's holdings are unknown: save the prefix again."""
    sched, client, _adapter = _scheduler(monkeypatch, lookup_done=False)
    entry = _admit_and_track(sched, _seq(), 60)
    assert entry[1] == 60
    client._orphan_since["7"] -= mp_lookup._ORPHAN_TIMEOUT_S + 1
    sched._end_finished_sessions()  # pump gives it up, then the floor resets
    assert entry[1] == 0
    assert sched._detached_lookups == {}


def test_an_unstamped_orphan_still_times_out(monkeypatch):
    """`lookup()` timeouts orphan without a stamp; it must still expire."""
    sched, client, _adapter = _scheduler(monkeypatch, lookup_done=False)
    seq = _seq()
    sched.prefetch_lookups([seq])
    client._orphans.add("7")  # as lookup()'s timeout path does
    assert "7" not in client._orphan_since
    client.pump()  # first sight: stamped, not expired
    assert client.has_orphans()
    client._orphan_since["7"] -= mp_lookup._ORPHAN_TIMEOUT_S + 1
    client.pump()
    assert not client.has_orphans()


def test_a_declined_detach_turns_the_request_back_to_the_ordinary_path(monkeypatch):
    """`allocate` disowned the probed prefix, so the detach was declined: the
    request is requeued and its next pass waits for the lookup as usual,
    where the remote-load capacity check applies."""
    from atom.model_engine import scheduler as sched_mod
    from atom.model_engine.scheduler import Scheduler

    monkeypatch.setattr(sched_mod.envs, "OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT", True)
    s = Scheduler.__new__(Scheduler)
    s.kv_connector = conn = _Conn()
    s._connector_flag = lambda name: getattr(conn, name, False)
    s.block_manager = SimpleNamespace(state_offload=None)
    seq = SimpleNamespace(id=7, kv_transfer_params=None)
    assert s._lookup_detach_enabled(seq)
    s._decline_lookup_detach(seq)
    assert not s._lookup_detach_enabled(seq)
    s._lookup_detach_declined.discard(seq.id)  # as admission does
    assert s._lookup_detach_enabled(seq)


def test_admission_walks_the_prefix_once_when_the_detach_is_probed(monkeypatch):
    """The detach decision and the admission share one `can_allocate` walk."""
    from atom.model_engine import scheduler as sched_mod
    from atom.model_engine.scheduler import Scheduler

    monkeypatch.setattr(sched_mod.envs, "OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT", True)
    walks = []

    def can_allocate(seq, *, record, block_hashes, reuse_hashes):
        walks.append(record)
        block_hashes.extend([11, 12])
        return 2

    s = Scheduler.__new__(Scheduler)
    s.kv_connector = _Conn()
    s._local_prefill_coalescing = False
    s.block_manager = SimpleNamespace(hash_block_size=4, can_allocate=can_allocate)
    probe = s._probe_hbm_prefix(SimpleNamespace())
    assert probe == (2, [11, 12]) and walks == [False]
    assert s._can_detach_tier_lookup(SimpleNamespace(), probe)
    assert walks == [False]  # the decision itself does not walk again


def test_the_detach_is_on_by_default_and_0_turns_it_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT", raising=False)
    assert envs.OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT is True
    monkeypatch.setenv("OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT", "0")
    assert envs.OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT is False
