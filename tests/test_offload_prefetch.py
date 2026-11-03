# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler._offload_prefetch.

A waiting request whose tier lookup answered gets its blocks and its load on any
tick -- delayer HOLD or FIRE, wherever it waits -- while its prefill still waits
for its turn and for a FIRE.
"""

import logging
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
from atom.kv_transfer.offload.mp.offload_prefetch import OffloadPrefetchState
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
        MockConfig(max_num_seqs=8, max_num_batched_tokens=64, num_kvcache_blocks=blocks)
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
    assert sched._prefetch.submitted == 1


def test_fire_tick_also_prefetches_and_does_not_prefill_a_loading_request(setup):
    sched, delayer, seq = setup
    delayer.allow = True

    batch, scheduled = sched.schedule()

    assert batch.total_seqs_num_prefill == 0 and scheduled == {}
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert seq in sched.waiting  # kept its place, not sent to the tail


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


def test_loaded_request_passes_a_head_refused_kv_blocks(monkeypatch, seq_factory):
    """A head that cannot allocate must not wait forever on an prefetched
    request behind it that holds blocks and could run and release them."""
    sched, _, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id}
    sched.schedule()  # b parks with its prefetch, a waits
    sched.finished_recving_kv_req_ids.append(b.id)
    # The last admission pass found the head refused KV blocks.
    sched._admission_head_blocked = True

    sched._promote_ready_remote_kv_requests()

    assert list(sched.waiting) == [b, a]


def test_resource_blocked_head_sets_the_bypass_flag(monkeypatch, seq_factory):
    sched, delayer, (a,) = _make(monkeypatch, seq_factory, n=1)
    sched.kv_connector.not_ready_ids = {a.id}
    sched.kv_connector.match = (0, False)
    delayer.allow = True
    monkeypatch.setattr(
        sched.block_manager, "can_allocate_memoized", lambda seq, *a, **k: -1
    )
    monkeypatch.setattr(
        sched.block_manager, "can_allocate", lambda seq, *a, **k: -1, raising=False
    )

    sched.schedule()

    assert sched._admission_head_blocked
    assert a in sched.waiting


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


def test_old_path_parked_request_is_still_promoted(monkeypatch, seq_factory):
    sched, _, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id, b.id}
    sched.schedule()
    # b parked by the admission pass, not by the prefetch.
    b.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    sched.finished_recving_kv_req_ids.append(b.id)

    sched._promote_ready_remote_kv_requests()

    assert list(sched.waiting) == [b, a]


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
    assert sched._prefetch.wait_n == 1


def test_prefill_stays_fifo_when_a_later_request_loaded_first(monkeypatch, seq_factory):
    sched, delayer, (a, b) = _make(monkeypatch, seq_factory, n=2)
    # a's lookup is late; b's is ready.
    sched.kv_connector.not_ready_ids = {a.id}
    sched.schedule()
    assert b.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    sched.finished_recving_kv_req_ids.append(b.id)
    # a's lookup answers and its load lands as well.
    sched.kv_connector.not_ready_ids = set()
    sched.schedule()
    assert a.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    sched.finished_recving_kv_req_ids.append(a.id)

    delayer.allow = True
    _, scheduled = sched.schedule()

    assert list(scheduled) == [a.id, b.id]  # arrival order, not load order


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


@pytest.mark.parametrize("why", ["no_load", "lookup_not_ready"])
def test_nothing_is_allocated_without_an_answered_load(setup, why):
    sched, _, seq = setup
    if why == "no_load":
        sched.kv_connector.match = (0, False)
    else:
        sched.kv_connector.ready = False
    free = _free(sched)

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert len(seq.block_table) == 0
    assert _free(sched) == free
    assert sched.kv_connector.allocs == []
    expect = "no_tier_load" if why == "no_load" else "lookup_not_ready"
    assert sched._prefetch.skips[expect] == 1
    sched.schedule()
    assert sched._prefetch.skips[expect] == 1  # counted once per request


def test_a_request_below_the_minimum_load_size_is_not_looked_at(
    monkeypatch, seq_factory
):
    sched, _, (seq,) = _make(monkeypatch, seq_factory)
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "1000")

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert sched._prefetch.skips["below_min_load"] == 1


def test_a_refused_park_is_taken_back_and_not_retried(setup):
    sched, delayer, seq = setup
    sched.kv_connector.parks = False
    free = _free(sched)

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert len(seq.block_table) == 0
    assert _free(sched) == free
    assert sched.kv_connector.cancelled == [seq.id]
    assert sched._num_parked_remote_kv == 0
    assert sched._prefetch.takebacks == 1
    sched.schedule()  # next tick: not tried again
    assert sched.kv_connector.allocs == [seq.id]

    # A FIRE admits it the ordinary way (here: no load, a plain prefill).
    delayer.allow = True
    batch, scheduled = sched.schedule()
    assert seq.id in scheduled and batch.total_seqs_num_prefill == 1


def test_unaligned_hbm_hit_is_left_to_the_admission_pass_and_counted(setup):
    sched, _, seq = setup
    sched.kv_connector.chunk_size = 8  # 4-token hash blocks: a 1-block hit is odd

    class _Probe:
        num_cached_blocks = 1

    sched._probe_hbm_prefix = lambda s: _Probe()

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert sched._prefetch.skips["unaligned_hbm"] == 1


# -- pressure bounds ----------------------------------------------------------


def test_watermark_stops_prefetch_allocation_and_the_request_waits_for_the_fire(
    monkeypatch, seq_factory
):
    sched, delayer, (seq,) = _make(monkeypatch, seq_factory)
    sched._PREFETCH_KV_WATERMARK = 0.01

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert len(seq.block_table) == 0
    assert sched._prefetch.skips["watermark"] == 1
    # Back to the old path: the FIRE's admission parks it as before.
    delayer.allow = True
    sched.schedule()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sched.kv_connector.allocs == [seq.id]


def test_pressure_skip_is_retried_once_pressure_is_gone(monkeypatch, seq_factory):
    sched, _, (seq,) = _make(monkeypatch, seq_factory)
    sched._PREFETCH_KV_WATERMARK = 0.01
    sched.schedule()
    assert seq.status == SequenceStatus.WAITING
    sched._PREFETCH_KV_WATERMARK = 0.7
    for _ in range(sched._PREFETCH_RETRY_TICKS + 1):
        sched.schedule()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS


def test_no_prefetch_allocation_without_a_free_parking_slot(monkeypatch, seq_factory):
    sched, _, (seq,) = _make(monkeypatch, seq_factory)
    sched._num_parked_remote_kv = sched.max_num_seqs

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert sched._prefetch.skips["seq_slots"] == 1


# -- cancellation ---------------------------------------------------------------


def test_abort_of_an_prefetched_request_releases_its_blocks(setup):
    sched, _, seq = setup
    free = _free(sched)
    sched.schedule()
    assert _free(sched) < free

    assert sched.abort_request(seq.id)

    assert seq.id not in sched._prefetch.entries
    assert sched._prefetch.aborted == 1
    assert seq not in sched.waiting
    # The load is still in flight: the blocks go back once it reports (the
    # existing aborted-load protocol), never before.
    assert seq._awaiting_aborted_load_cleanup
    assert _free(sched) < free
    assert sched._finish_aborted_load_cleanup(seq.id)
    assert _free(sched) == free
    assert sched._num_parked_remote_kv == 0
    assert seq.id not in sched.deferred_free_blocks


def test_abort_of_a_loaded_prefetched_request_releases_its_blocks(setup):
    sched, _, seq = setup
    free = _free(sched)
    sched.schedule()
    sched.finished_recving_kv_req_ids.append(seq.id)

    assert sched.abort_request(seq.id)

    assert _free(sched) == free
    assert sched._num_parked_remote_kv == 0
    assert not sched._prefetch.entries


def test_status_abort_is_reclaimed_on_a_hold_tick(monkeypatch, seq_factory):
    sched, _, (_a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.schedule()
    b.status = SequenceStatus.ABORTED  # marked, still queued behind a

    sched.schedule()  # HOLD: reclaimed without waiting for a FIRE

    assert b not in sched.waiting
    assert b.id not in sched._prefetch.entries


# -- reporting ------------------------------------------------------------------


def test_one_info_line_per_interval(setup, caplog):
    sched, _, _seq = setup
    sched.schedule()
    state = sched._prefetch
    state.skip(77, "watermark")
    state.skip(78, "unaligned_hbm")
    state._last_log -= 61.0
    with caplog.at_level(logging.INFO, logger="atom"):
        sched.schedule()
        sched.schedule()
    lines = [
        r.getMessage() for r in caplog.records if "[OFFLOAD-PREFETCH]" in r.getMessage()
    ]
    assert len(lines) == 1
    line = lines[0]
    assert "submitted=1" in line
    assert "'watermark': 1" in line and "'unaligned_hbm': 1" in line
    assert "held_blocks=4" in line and "held_requests=1" in line
    assert "avg_wait=" in line


def test_state_counts_a_request_once_per_reason():
    state = OffloadPrefetchState()
    for _ in range(3):
        state.skip(1, "lookup_not_ready")
    state.skip(2, "lookup_not_ready")
    state.skip(1, "watermark")
    assert state.skips == {"lookup_not_ready": 2, "watermark": 1}


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
