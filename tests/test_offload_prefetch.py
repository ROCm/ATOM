# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The scheduler's offload prefetch (`OffloadPrefetch`).

A waiting request whose tier lookup answered gets its blocks and its load on any
tick -- delayer HOLD or FIRE, wherever it waits -- while its prefill still waits
for its turn and for a FIRE.
"""

from types import SimpleNamespace

import pytest
from conftest import MockConfig
from test_lmcache_mp import _config, fake_lmcache_modules  # noqa: F401
from test_lmcache_mp_async_lookup import CHUNK, _Adapter
from test_lmcache_mp_async_threads import (
    _driven_client,
    _PerRequestClient,
    _wait_until,
)

from atom.kv_transfer.offload._offload_common import OffloadSchedulerMixin
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.model_engine.offload_prefetch import OffloadPrefetch
from atom.model_engine.prefill_delayer import PrefillDecision
from atom.model_engine.scheduler import Admit, Scheduler
from atom.model_engine.sequence import SequenceStatus

META = object()


class _Connector(OffloadSchedulerMixin):
    """Offload connector double: every request has an answered lookup and a
    tier hit that needs a load, unless a test says otherwise."""

    is_producer = False
    is_offload = True
    chunk_size = 4
    supports_async_lookup = True

    def __init__(self):
        self.ready = True
        self.not_ready_ids = set()
        self.match = (8, True)
        self.parks = True
        self.allocs = []
        self.cancelled = []
        self.pending_cancelled = []
        self.released = []

    def prefetch_lookups(self, seqs):
        list(seqs)

    def lookup_pending(self, seq):
        return False

    def lookup_ready(self, seq):
        return self.ready and seq.id not in self.not_ready_ids

    def take_ready_lookup(self, seq):
        return self.lookup_ready(seq)

    def get_num_new_matched_tokens(self, seq):
        return self.match

    def update_state_after_alloc(self, seq):
        self.allocs.append(seq.id)

    def should_park_for_load_after_alloc(self, seq):
        return self.parks

    def cancel_preadmission(self, seq):
        self.cancelled.append(seq.id)

    def release_taken_lookup(self, seq):
        self.released.append(seq.id)

    def build_connector_meta(self):
        return META

    def save_finished(self, req_id): ...
    def abandon_save(self, req_id): ...
    def release_stalled_save(self, seq): ...
    def source_blocks_released(self, seq): ...

    def cancel_pending_load(self, seq):
        self.pending_cancelled.append(seq.id)

    def request_finished(self, seq): ...

    def should_defer_free(self, seq):
        return False

    def load_failed(self, req_id):
        return False

    def load_finished(self, req_id):
        return True


class _Delayer:
    is_local = False
    max_queue_ms = None
    kv_high_watermark = 0.9

    def __init__(self):
        self.rank_prefill_costs = []
        self.allow = False
        self.calls = 0

    def protects_decode(self, running_decode_batch):
        return False

    def should_allow_prefill(self, **_):
        self.calls += 1
        return PrefillDecision(allow=self.allow, prefillable_ranks=1)

    def notify_prefill_executed(self): ...


def _make(monkeypatch, seq_factory, n=1, blocks=100, **env):
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    sched = Scheduler(
        MockConfig(
            max_num_seqs=8,
            max_num_batched_tokens=64,
            num_kvcache_blocks=blocks,
            offload_prefetch=True,
        )
    )
    sched.kv_connector = _Connector()
    # Built here: `__init__` saw MockConfig's connector, which answers no
    # async lookup.
    sched._prefetch = OffloadPrefetch(sched, 0.8, min_load_tokens=0)
    delayer = _Delayer()
    sched.set_prefill_delayer(delayer)
    seqs = [seq_factory(list(range(16))) for _ in range(n)]
    for seq in seqs:
        sched.add(seq)
    return sched, delayer, seqs


@pytest.fixture
def setup(monkeypatch, seq_factory):
    sched, delayer, (seq,) = _make(monkeypatch, seq_factory)
    return sched, delayer, seq


def _free(sched):
    return sched.block_manager.pool_occupancy()["free"]


def test_hold_tick_parks_a_request_and_ships_its_load(setup):
    sched, _, seq = setup

    batch, scheduled = sched.schedule()

    assert batch.total_seqs_num_prefill == 0 and scheduled == {}
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert len(seq.block_table) > 0
    assert sched.kv_connector.allocs == [seq.id]
    assert sched._num_parked_remote_kv == 1
    # The load leaves with this HOLD tick's (empty) batch.
    assert batch.connector_meta_output is META
    assert seq in sched.waiting
    assert seq.id in sched._prefetch.entries


def test_request_behind_the_head_is_loaded_and_the_order_is_unchanged(
    monkeypatch, seq_factory
):
    sched, _, (a, b, c) = _make(monkeypatch, seq_factory, n=3)
    sched.kv_connector.not_ready_ids = {a.id, c.id}  # only b's lookup answered

    sched.schedule()

    assert b.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert a.status == c.status == SequenceStatus.WAITING
    assert list(sched.waiting) == [a, b, c]


def test_loaded_request_is_not_promoted_ahead_of_earlier_arrivals(
    monkeypatch, seq_factory
):
    sched, _, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id}
    sched.schedule()  # b parks, a waits for its lookup
    assert b.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    sched.finished_recving_kv_req_ids.append(b.id)

    sched._promote_ready_remote_kv_requests()

    assert list(sched.waiting) == [a, b]  # a parked-and-loaded old-path request
    # would have moved up; a prefetched one keeps its place


def _fire_blocked_on(sched, delayer, seq, monkeypatch):
    """A FIRE whose admission stops at `seq` for want of KV blocks."""
    real = sched.block_manager.can_allocate

    def can_allocate(s, *a, **k):
        return -1 if s is seq else real(s, *a, **k)

    monkeypatch.setattr(sched.block_manager, "can_allocate", can_allocate)
    delayer.allow = True
    sched.schedule()
    delayer.allow = False
    monkeypatch.setattr(sched.block_manager, "can_allocate", real)


def test_loaded_request_passes_admission_stuck_on_resources(monkeypatch, seq_factory):
    """Finished prefetches move up only when admission last stopped for KV
    blocks or a parked-load slot (Phase 2's own record), which they hold and
    release only by running. Otherwise they keep their place."""
    sched, delayer, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id}
    sched.schedule()  # b parks with its prefetch, a waits
    sched.finished_recving_kv_req_ids.append(b.id)
    sched._promote_ready_remote_kv_requests()
    assert list(sched.waiting) == [a, b]  # nothing stuck: order kept

    sched.kv_connector.not_ready_ids = set()
    sched.kv_connector.match = (0, False)  # a prefills locally
    _fire_blocked_on(sched, delayer, a, monkeypatch)
    assert sched._blocked_head
    assert list(sched.waiting) == [a, b]  # recorded at this FIRE, not yet used
    sched._promote_ready_remote_kv_requests()
    assert list(sched.waiting) == [b, a]


def test_a_failed_prefetch_back_to_waiting_is_forgotten(monkeypatch, seq_factory):
    """A prefetched joint load that fails and cannot be disowned goes back to
    plain `waiting` holding nothing: it is no longer a prefetch, and an
    admission stuck on it lets the finished prefetches behind it run."""
    sched, delayer, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.schedule()  # both park
    bm = sched.block_manager
    a.offload_joint.boundary_tokens = 8
    monkeypatch.setattr(bm, "retreat_joint_to_hbm", lambda s: False)
    monkeypatch.setattr(bm, "disown_claimed_prefix", lambda s: False)
    sched.failed_recving_kv_req_ids.append(a.id)
    sched.finished_recving_kv_req_ids.append(b.id)

    sched._promote_ready_remote_kv_requests()

    assert a.status == SequenceStatus.WAITING and len(a.block_table) == 0
    assert not sched._prefetch.holds(a)
    # It waits for no load now: its parked slot is given back (b keeps one).
    assert sched._num_parked_remote_kv == 1
    assert sched._prefetch.holds(b) and b.offload_loaded
    assert list(sched.waiting) == [a, b]
    _fire_blocked_on(sched, delayer, a, monkeypatch)
    assert sched._blocked_head
    sched._promote_ready_remote_kv_requests()
    assert list(sched.waiting) == [b, a]


def test_promotion_never_probes(monkeypatch, seq_factory):
    sched, _, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id}
    sched.schedule()
    sched.finished_recving_kv_req_ids.append(b.id)
    sched._blocked_head = True

    def no_probe(*_a, **_k):
        raise AssertionError("promotion must not probe")

    monkeypatch.setattr(sched.block_manager, "can_allocate_memoized", no_probe)
    monkeypatch.setattr(sched.block_manager, "can_allocate", no_probe)
    sched._promote_ready_remote_kv_requests()
    assert list(sched.waiting) == [b, a]


def test_a_finished_load_is_resolved_once_at_the_top_of_the_tick(setup):
    """Resolved before anything sizes the tick's prefill work: from then on
    the delayer's probes see an ordinary resume (main's rule), and a load in
    flight is no prefill work."""
    sched, _, seq = setup
    sched.schedule()  # parks with its prefetch
    assert not sched._can_admit_head_prefill()
    assert next(sched._waiting_prefills(), None) is None
    sched.finished_recving_kv_req_ids.append(seq.id)
    seq.offload_loaded_tokens = 12
    assert not sched._can_admit_head_prefill()  # not resolved yet

    sched._promote_ready_remote_kv_requests()

    assert seq.status == SequenceStatus.WAITING and seq.offload_loaded
    assert sched.finished_recving_kv_req_ids == []
    assert sched._can_admit_head_prefill()
    assert sched._waiting_new_token_count() == len(seq) - 12


def test_prefetch_off_keeps_mains_delayer_signal(monkeypatch, seq_factory):
    """Prefetch off, offload tier: a failed joint load that re-parks for its
    own state load is no prefill work, before or after it is resolved; and a
    P/D consumer's finished receive is not resolved early (it decodes)."""
    sched, delayer, (seq,) = _make(monkeypatch, seq_factory)
    sched._prefetch = None
    delayer.allow = True
    sched.schedule()  # Phase 2 parks it
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    delayer.allow = False
    seq.offload_joint.boundary_tokens = 8
    seq.offload_joint.load_hash = 7
    monkeypatch.setattr(sched.block_manager, "retreat_joint_to_hbm", lambda s: True)
    sched.failed_recving_kv_req_ids.append(seq.id)
    assert not sched._can_admit_head_prefill()
    assert next(sched._waiting_prefills(), None) is None

    sched._promote_ready_remote_kv_requests()

    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert not sched._can_admit_head_prefill()
    assert next(sched._waiting_prefills(), None) is None
    assert sched._local_prefill_pending_work()[0] is False

    sched.kv_connector.is_offload = False
    sched.finished_recving_kv_req_ids.append(seq.id)
    sched._promote_ready_remote_kv_requests()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sched.finished_recving_kv_req_ids == [seq.id]


def test_prefetch_waits_for_a_producer_only_as_phase_2_would(setup, monkeypatch):
    """Without local prefill coalescing Phase 2 never waits for a producer, so
    the prefetch must not either."""
    sched, _, seq = setup
    calls = []
    monkeypatch.setattr(
        sched, "_wait_for_inflight_prefix", lambda *a, **k: calls.append(1) or True
    )
    sched._local_prefill_coalescing = False
    sched.schedule()
    assert calls == [] and sched._prefetch.holds(seq)


def test_a_producer_wait_is_retried_later_without_touching_the_tier(setup, monkeypatch):
    sched, _, seq = setup
    sched._local_prefill_coalescing = True
    asked = []
    conn = sched.kv_connector
    take = conn.take_ready_lookup
    conn.take_ready_lookup = lambda s: asked.append(s.id) or take(s)
    monkeypatch.setattr(sched, "_wait_for_inflight_prefix", lambda *a, **k: True)
    sched.schedule()
    assert asked == [] and not sched._prefetch.holds(seq)
    retry = sched._prefetch.skip_until[seq.id]
    assert retry == sched._schedule_tick + OffloadPrefetch.RETRY_TICKS
    sched.schedule()
    assert asked == []  # not looked at again before then


def test_a_local_admission_is_kept_as_a_resume_in_place(monkeypatch, seq_factory):
    """The connector refuses the load after `allocate`: the prefetch keeps the
    admission (no take-back) and the request resumes in place at its hit."""
    sched, delayer, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id}
    sched.kv_connector.parks = False
    sched.schedule()
    assert b.status == SequenceStatus.WAITING and len(b.block_table) > 0
    assert b.offload_held_local and not getattr(b, "offload_load_failed", False)
    assert sched._is_offload_prefill_resume(b)
    assert sched._prefetch.holds(b) and sched.kv_connector.cancelled == []
    assert sched._num_parked_remote_kv == 0
    assert list(sched.waiting) == [a, b]

    sched.kv_connector.not_ready_ids = set()
    sched.kv_connector.match = (0, False)
    delayer.allow = True
    _, scheduled = sched.schedule()
    assert list(scheduled) == [a.id, b.id]
    assert sched.kv_connector.allocs.count(b.id) == 1
    assert not b.offload_held_local


def test_requeue_cancels_the_pre_admission(setup, monkeypatch):
    """A REQUEUE (a state load whose disown cannot be backed) gives the
    connector's pre-admission back, for Phase 2 as for the prefetch."""
    sched, _, seq = setup
    bm = sched.block_manager
    seq.offload_joint.load_hash = 7
    monkeypatch.setattr(bm, "cancel_state_load", lambda s: False)
    probe = sched._probe_hbm_prefix(seq)
    outcome = sched._admit_after_probe(seq, probe, True, None)
    assert outcome is Admit.REQUEUE
    assert sched.kv_connector.cancelled == [seq.id]
    assert len(seq.block_table) == 0


def test_loaded_request_prefills_only_on_a_fire(setup):
    sched, delayer, seq = setup
    sched.schedule()  # HOLD: parks and ships the load
    sched.finished_recving_kv_req_ids.append(seq.id)

    batch, _ = sched.schedule()  # still HOLD: resolved, not prefilled
    assert batch.total_seqs_num_prefill == 0
    assert seq.status == SequenceStatus.WAITING and seq.offload_loaded

    delayer.allow = True
    batch, scheduled = sched.schedule()  # FIRE: resumes as an offload prefill
    assert batch.total_seqs_num_prefill == 1
    assert seq.id in scheduled
    assert sched._num_parked_remote_kv == 0
    # One admission only: the FIRE did not allocate or notify again.
    assert sched.kv_connector.allocs == [seq.id]
    # Its prefetch record closed when it went onto the prefill batch.
    assert seq.id not in sched._prefetch.entries


def test_request_whose_load_is_in_flight_does_not_block_later_ones(
    monkeypatch, seq_factory
):
    sched, delayer, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.match = (8, True)
    sched.schedule()  # both park
    sched.finished_recving_kv_req_ids.append(b.id)  # only b has landed

    delayer.allow = True
    _, scheduled = sched.schedule()

    assert list(scheduled) == [b.id]
    assert list(sched.waiting) == [a]
    assert a.status == SequenceStatus.WAITING_FOR_REMOTE_KVS


# -- pressure bounds ----------------------------------------------------------


def test_watermark_stops_prefetch_allocation_and_the_request_waits_for_the_fire(
    monkeypatch, seq_factory
):
    sched, delayer, (seq,) = _make(monkeypatch, seq_factory)
    sched._prefetch.max_kv_usage = 0.01

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert len(seq.block_table) == 0
    assert seq.id in sched._prefetch.skip_until
    # Back to the old path: the FIRE's admission parks it as before.
    delayer.allow = True
    sched.schedule()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sched.kv_connector.allocs == [seq.id]


# -- cancellation ---------------------------------------------------------------


def test_abort_of_a_prefetched_request_releases_its_blocks(setup):
    sched, _, seq = setup
    free = _free(sched)
    sched.schedule()
    assert _free(sched) < free

    assert sched.abort_request(seq.id)

    assert seq.id not in sched._prefetch.entries
    assert seq not in sched.waiting
    # The load is still in flight: the blocks go back once it reports (the
    # existing aborted-load protocol), never before.
    assert seq._awaiting_aborted_load_cleanup
    assert _free(sched) < free
    assert sched._finish_aborted_load_cleanup(seq.id)
    assert _free(sched) == free
    assert sched._num_parked_remote_kv == 0
    assert seq.id not in sched.deferred_free_blocks


def test_probe_looks_past_prefetches_still_loading(monkeypatch, seq_factory):
    """Four in-flight prefetches ahead of a finished fifth one: the delayer
    probe must still report prefill work, as admission will prefill the fifth."""
    sched, _, seqs = _make(monkeypatch, seq_factory, n=5)
    sched.schedule()  # all five park
    assert all(s.status == SequenceStatus.WAITING_FOR_REMOTE_KVS for s in seqs)
    assert not sched._can_admit_head_prefill()
    sched.finished_recving_kv_req_ids.append(seqs[4].id)
    sched._promote_ready_remote_kv_requests()  # the top of the next tick

    assert list(sched.waiting) == seqs  # it kept its place
    assert sched._can_admit_head_prefill()


def test_the_tier_answer_is_taken_last(setup):
    """Request-only checks and the HBM probe come first; the answer is taken
    (`take_ready_lookup`) only right before it is used."""
    sched, _, _seq = setup
    events = []
    conn = sched.kv_connector
    take, match = conn.take_ready_lookup, conn.get_num_new_matched_tokens
    probe = sched._probe_hbm_prefix
    sched._probe_hbm_prefix = lambda s: events.append("probe") or probe(s)
    conn.take_ready_lookup = lambda s: events.append("take") or take(s)
    conn.get_num_new_matched_tokens = lambda s: events.append("match") or match(s)

    sched.schedule()

    i = events.index("take")
    assert events[:i] == ["probe"] and events[i : i + 2] == ["take", "match"]


# -- lookup driver thread -------------------------------------------------------


def test_lookup_ready_and_take_in_lookup_thread_mode(
    fake_lmcache_modules,  # noqa: F811
):
    """With the lookup driver thread an answered lookup reads as ready, an
    unanswered one does not, and a consumed one stops reading as ready."""
    fake = _PerRequestClient(chunks=2)
    fake.ready.clear()  # the server has not answered yet
    adapter = _Adapter(fake)
    client = _driven_client(adapter)
    sched = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
    )
    sched._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        sched, _config(), chunk_size=CHUNK, lookup_client=client
    )
    try:
        seq = SimpleNamespace(
            id=1,
            num_prompt_tokens=10,
            num_cached_tokens=0,
            token_ids=list(range(10)),
            status=SequenceStatus.WAITING,
        )
        sched.offer_lookups([seq])
        assert _wait_until(lambda: "1" in client._async_tokens)
        assert not sched.lookup_ready(seq)  # submitted, unanswered
        fake.ready.set()
        assert _wait_until(lambda: "1" in client._async_hits)
        assert client.driving
        # No scheduler pump ran: the driver thread alone answered it.
        assert sched.lookup_ready(seq)
        assert not sched.lookup_pending(seq)
        # Taken in one step under the client's lock, and then held.
        assert sched.take_ready_lookup(seq)
        assert sched._lookup_results["1"] == (seq, 2 * CHUNK)
        assert not client.has_answer("1")
        sched._lookup_results.pop("1", None)
        assert not sched.lookup_ready(seq)  # consumed
        assert not sched.take_ready_lookup(seq)
    finally:
        client.close()


def test_in_flight_prefetches_do_not_spend_the_lookup_bypass_budget(
    monkeypatch, seq_factory
):
    """A head waiting on its lookup, then 20 prefetches in flight, then a
    ready request: passing the prefetches costs none of the 16-request
    bypass budget, so the ready request is still admitted."""
    sched, delayer, seqs = _make(monkeypatch, seq_factory, n=22)
    sched.max_num_seqs = 64
    head, prefetched, ready = seqs[0], seqs[1:21], seqs[21]
    for s in prefetched:
        s.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        sched._prefetch.entries[s.id] = s
    sched._prefetch.run = lambda: 0
    sched.kv_connector.match = (0, False)
    monkeypatch.setattr(sched, "_offload_lookup_pending", lambda s: s is head)
    delayer.allow = True
    _, scheduled = sched.schedule()
    assert list(scheduled) == [ready.id]
    assert list(sched.waiting) == [head] + prefetched


def test_a_non_resource_refusal_does_not_promote(monkeypatch, seq_factory):
    """Only a break for KV blocks or a parked slot marks Phase 2 blocked: a
    refused admission for another reason (a declined lookup detach, say)
    does not let finished prefetches pass earlier arrivals; neither does a
    blocked request that was aborted."""
    from atom.model_engine.scheduler import Admit

    sched, delayer, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id}
    sched.schedule()  # b parks
    sched.finished_recving_kv_req_ids.append(b.id)
    sched.kv_connector.not_ready_ids = set()
    sched.kv_connector.match = (0, False)
    real = sched._admit_after_probe
    monkeypatch.setattr(
        sched,
        "_admit_after_probe",
        lambda s, *x, **k: Admit.REFUSED if s is a else real(s, *x, **k),
    )
    delayer.allow = True
    sched.schedule()
    assert not sched._blocked_head
    sched._blocked_head = True
    sched.abort_request(a.id)
    assert not sched._blocked_head


def test_a_taken_answer_that_is_not_admitted_is_given_back(setup):
    """Nothing to load (here: a hit shorter than the connector's minimum
    load): the answer prefetch took is released, not held until Phase 2."""
    sched, _, seq = setup
    sched._prefetch.min_load_tokens = 64
    sched.schedule()
    assert seq.status == SequenceStatus.WAITING and len(seq.block_table) == 0
    assert sched.kv_connector.released == [seq.id]
    assert sched.kv_connector.allocs == []
