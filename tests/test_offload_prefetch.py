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
from atom.model_engine.prefill_delayer import PrefillDecision
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus

META = object()


class _Connector(OffloadSchedulerMixin):
    """Offload connector double: every request has an answered lookup and a
    tier hit that needs a load, unless a test says otherwise."""

    is_producer = False
    is_offload = True
    chunk_size = 4

    def __init__(self):
        self.ready = True
        self.not_ready_ids = set()
        self.match = (8, True)
        self.parks = True
        self.allocs = []
        self.cancelled = []
        self.pending_cancelled = []

    def prefetch_lookups(self, seqs):
        list(seqs)

    def lookup_pending(self, seq):
        return False

    def lookup_ready(self, seq):
        return self.ready and seq.id not in self.not_ready_ids

    def get_num_new_matched_tokens(self, seq):
        return self.match

    def update_state_after_alloc(self, seq):
        self.allocs.append(seq.id)

    def should_park_for_load_after_alloc(self, seq):
        return self.parks

    def cancel_preadmission(self, seq):
        self.cancelled.append(seq.id)

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
    # would have moved up; an prefetched one keeps its place


def test_loaded_request_passes_a_head_that_cannot_allocate(monkeypatch, seq_factory):
    """Nothing running and a head that cannot allocate: a finished prefetch
    behind it holds blocks and must run to release them. Otherwise it keeps
    its place."""
    sched, _, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id}
    sched.schedule()  # b parks with its prefetch, a waits
    sched.finished_recving_kv_req_ids.append(b.id)

    sched._promote_ready_remote_kv_requests()
    assert list(sched.waiting) == [a, b]  # a can allocate: order kept

    monkeypatch.setattr(sched.block_manager, "can_allocate_memoized", lambda s: -1)
    sched._promote_ready_remote_kv_requests()
    assert list(sched.waiting) == [b, a]


def test_loaded_request_passes_a_head_blocked_by_the_parked_slot_cap(
    monkeypatch, seq_factory
):
    """KV fits, but every parked-load slot is held and nothing runs: the
    finished prefetch must run first to free its slot."""
    sched, _, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id}
    sched.schedule()  # b parks with its prefetch, a waits
    sched.finished_recving_kv_req_ids.append(b.id)
    sched._num_parked_remote_kv = sched.max_num_seqs

    sched._promote_ready_remote_kv_requests()

    assert list(sched.waiting) == [b, a]


def test_new_token_count_counts_a_finished_load_from_its_resume(setup):
    sched, _, seq = setup
    sched.schedule()  # parks with its prefetch
    sched.finished_recving_kv_req_ids.append(seq.id)
    seq.offload_loaded_tokens = 12
    assert sched._waiting_new_token_count() == len(seq) - 12


def test_finished_disagg_consumer_load_is_not_prefill_work(setup):
    """A disaggregation consumer's finished load goes straight to decode."""
    sched, _, seq = setup
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    sched.finished_recving_kv_req_ids.append(seq.id)
    assert next(sched._waiting_prefills(), None) is seq  # offload: prefills
    sched.kv_connector.is_offload = False
    assert next(sched._waiting_prefills(), None) is None
    sched.failed_recving_kv_req_ids.append(seq.id)
    assert next(sched._waiting_prefills(), None) is seq  # failed: recomputes


def test_loaded_request_prefills_only_on_a_fire(setup):
    sched, delayer, seq = setup
    sched.schedule()  # HOLD: parks and ships the load
    sched.finished_recving_kv_req_ids.append(seq.id)

    batch, _ = sched.schedule()  # still HOLD
    assert batch.total_seqs_num_prefill == 0
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS

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
    sched._prefetch.KV_WATERMARK = 0.01

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert len(seq.block_table) == 0
    assert seq.id in sched._prefetch.retry_at
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

    assert sched._can_admit_head_prefill()


# -- lookup driver thread -------------------------------------------------------


def test_lookup_ready_in_lookup_thread_mode(fake_lmcache_modules):  # noqa: F811
    """With the lookup driver thread an answered lookup reads as ready, an
    unanswered one does not, and a consumed one stops reading as ready."""
    fake = _PerRequestClient(chunks=2)
    fake.ready.clear()  # the server has not answered yet
    adapter = _Adapter(fake)
    client = _driven_client(adapter)
    sched = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
    )
    sched._shipped_tokens = {}
    sched._shipped_drops = []
    sched._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        sched, _config(), chunk_size=CHUNK, lookup_client=client
    )
    try:
        seq = SimpleNamespace(
            id=1,
            num_prompt_tokens=10,
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
        assert client.lookup(list(range(10)), "1") == 2 * CHUNK
        sched._lookup_results.pop("1", None)
        assert not sched.lookup_ready(seq)  # consumed
    finally:
        client.close()
