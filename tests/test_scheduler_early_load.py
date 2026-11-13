# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler._early_load_tier_prefixes.

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

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.kv_transfer.offload._offload_common import OffloadSchedulerMixin
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.model_engine.early_load import EarlyLoadState
from atom.model_engine.prefill_delayer import PrefillDecision
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Replay, SequenceStatus

META = object()


class _Connector(OffloadSchedulerMixin, KVConnectorSchedulerBase):
    """Offload connector double: every request has an answered lookup and a
    tier hit that needs a load, unless a test says otherwise."""

    is_producer = False
    is_offload = True
    chunk_size = 4
    supports_early_load = True
    lowers_hbm_hit_to_chunk = False

    def held_tier_hit(self, seq):
        return 0

    def close(self): ...
    def send_finished(self, req_id): ...

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
    rank_prefill_costs: tuple = ()

    def __init__(self):
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
    assert sched._early_load.submitted == 1


def test_fire_tick_also_loads_early_and_does_not_prefill_a_loading_request(setup):
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

    sched._prioritize_ready_remote_kv_requests()

    assert list(sched.waiting) == [a, b]  # a parked-and-loaded old-path request
    # would have moved up; an early-loaded one keeps its place


def test_old_path_parked_request_is_still_promoted(monkeypatch, seq_factory):
    sched, _, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id, b.id}
    sched.schedule()
    # b parked by the admission pass, not by the early load; the real
    # admission owns its block table before it can park for a tier load.
    sched.block_manager.allocate(b)
    assert b.block_table
    b.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    sched.finished_recving_kv_req_ids.append(b.id)

    sched._prioritize_ready_remote_kv_requests()

    assert list(sched.waiting) == [b, a]


def test_loaded_request_prefills_only_on_a_fire(setup):
    sched, delayer, seq = setup
    sched.schedule()  # HOLD: parks and ships the load
    sched.finished_recving_kv_req_ids.append(seq.id)

    batch, _ = sched.schedule()  # still HOLD
    assert batch.total_seqs_num_prefill == 0
    # Completion resolves before the delayer sizes the work, but HOLD still
    # launches no prefill and preserves the reserved resources.
    assert seq.status == SequenceStatus.WAITING

    delayer.allow = True
    batch, scheduled = sched.schedule()  # FIRE: resumes as an offload prefill
    assert batch.total_seqs_num_prefill == 1
    assert seq.id in scheduled
    assert sched._num_parked_remote_kv == 0
    # One admission only: the FIRE did not allocate or notify again.
    assert sched.kv_connector.allocs == [seq.id]
    # Its early-load record closed when it went onto the prefill batch.
    assert seq.id not in sched._early_load.entries
    assert sched._early_load.wait_n == 1


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
    _batch, scheduled = sched.schedule()

    assert list(scheduled) == [a.id, b.id]  # arrival order, not load order


def test_request_whose_load_is_in_flight_does_not_block_later_ones(
    monkeypatch, seq_factory
):
    sched, delayer, (a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.match = (8, True)
    sched.schedule()  # both park
    sched.finished_recving_kv_req_ids.append(b.id)  # only b has landed

    delayer.allow = True
    _batch, scheduled = sched.schedule()

    assert list(scheduled) == [b.id]
    assert list(sched.waiting) == [a]
    assert a.status == SequenceStatus.WAITING_FOR_REMOTE_KVS


def test_priority_max_queue_has_no_effect_with_early_load(monkeypatch, seq_factory):
    monkeypatch.setenv("ATOM_DP_PREFIX_PRIORITY_MAX_QUEUE", "0")
    sched = Scheduler(MockConfig(num_kvcache_blocks=100))
    first, moved = seq_factory(list(range(16))), seq_factory(list(range(16)))
    moved.dp_prefetch_priority = True
    sched.add(first)
    sched.add(moved)
    assert list(sched.waiting) == [first, moved]


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
    assert sched._early_load.skips[expect] == 1
    sched.schedule()
    assert sched._early_load.skips[expect] == 1  # counted once per request


def test_a_request_below_the_minimum_load_size_is_not_looked_at(
    monkeypatch, seq_factory
):
    sched, _, (seq,) = _make(monkeypatch, seq_factory)
    sched.config.offload_min_load_tokens = 1000

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert sched._early_load.skips["below_min_load"] == 1


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
    assert sched._early_load.skips["taken_back"] == 1
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
    assert sched._early_load.skips["unaligned_hbm"] == 1


def test_early_load_waits_for_the_inflight_prefix_producer_phase_2_waits_for(setup):
    sched, _, seq = setup
    sched._local_prefill_coalescing = True
    waited = []
    sched._wait_for_inflight_prefix = lambda s, cached: waited.append(s.id) or True

    sched._early_load_tier_prefixes()

    assert waited == [seq.id]
    assert seq.status == SequenceStatus.WAITING
    assert sched._early_load.skips["inflight_prefix"] == 1
    assert seq.id not in sched._early_load.entries


# -- pressure bounds ----------------------------------------------------------


def test_watermark_stops_early_allocation_and_the_request_waits_for_the_fire(
    monkeypatch, seq_factory
):
    sched, delayer, (seq,) = _make(
        monkeypatch, seq_factory, ATOM_OFFLOAD_EARLY_LOAD_KV_WATERMARK="0.01"
    )

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert len(seq.block_table) == 0
    assert sched._early_load.skips["watermark"] == 1
    # Back to the old path: the FIRE's admission parks it as before.
    delayer.allow = True
    sched.schedule()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sched.kv_connector.allocs == [seq.id]


def test_max_fraction_bounds_the_blocks_held_by_early_loads(monkeypatch, seq_factory):
    # 16 tokens = 4 blocks each; 100 blocks; a 5% cap fits one request, not two.
    sched, _, (a, b, c) = _make(
        monkeypatch, seq_factory, n=3, ATOM_OFFLOAD_EARLY_LOAD_MAX_FRAC="0.05"
    )

    sched.schedule()

    assert a.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert b.status == c.status == SequenceStatus.WAITING
    assert sched._early_load.skips["max_frac"] == 2
    held = sched._early_load.held_blocks()
    assert held == len(a.block_table) > 0


def test_pressure_skip_is_retried_once_pressure_is_gone(monkeypatch, seq_factory):
    sched, _, (seq,) = _make(
        monkeypatch, seq_factory, ATOM_OFFLOAD_EARLY_LOAD_KV_WATERMARK="0.01"
    )
    sched.schedule()
    assert seq.status == SequenceStatus.WAITING
    monkeypatch.setenv("ATOM_OFFLOAD_EARLY_LOAD_KV_WATERMARK", "0.7")
    for _ in range(sched._EARLY_LOAD_RETRY_TICKS + 1):
        sched.schedule()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS


def test_no_early_allocation_without_a_free_parking_slot(monkeypatch, seq_factory):
    sched, _, (seq,) = _make(monkeypatch, seq_factory)
    sched._num_parked_remote_kv = sched.max_num_seqs

    sched.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert sched._early_load.skips["seq_slots"] == 1


# -- cancellation ---------------------------------------------------------------


def test_abort_of_an_early_loaded_request_releases_its_blocks(setup):
    sched, _, seq = setup
    free = _free(sched)
    sched.schedule()
    assert _free(sched) < free

    assert sched.abort_request(seq.id)

    assert seq.id not in sched._early_load.entries
    assert sched._early_load.aborted == 1
    assert seq not in sched.waiting
    # The load is still in flight: the blocks go back once it reports (the
    # existing aborted-load protocol), never before.
    assert seq._awaiting_aborted_load_cleanup
    assert _free(sched) < free
    assert sched._finish_aborted_load_cleanup(seq.id)
    assert _free(sched) == free
    assert sched._num_parked_remote_kv == 0
    assert seq.id not in sched.deferred_free_blocks


def test_abort_of_a_loaded_early_request_releases_its_blocks(setup):
    sched, _, seq = setup
    free = _free(sched)
    sched.schedule()
    sched.finished_recving_kv_req_ids.append(seq.id)

    assert sched.abort_request(seq.id)

    assert _free(sched) == free
    assert sched._num_parked_remote_kv == 0
    assert not sched._early_load.entries


def test_status_abort_is_reclaimed_on_a_hold_tick(monkeypatch, seq_factory):
    sched, _, (_a, b) = _make(monkeypatch, seq_factory, n=2)
    sched.schedule()
    b.status = SequenceStatus.ABORTED  # marked, still queued behind a

    sched.schedule()  # HOLD: reclaimed without waiting for a FIRE

    assert b not in sched.waiting
    assert b.id not in sched._early_load.entries


# -- reporting ------------------------------------------------------------------


def test_one_info_line_per_interval(setup, caplog):
    sched, _, _seq = setup
    sched.schedule()
    state = sched._early_load
    state.skip(77, "watermark")
    state.skip(78, "unaligned_hbm")
    state._last_log -= 61.0
    with caplog.at_level(logging.INFO, logger="atom"):
        sched.schedule()
        sched.schedule()
    lines = [r.getMessage() for r in caplog.records if "[EARLY-LOAD]" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "submitted=1" in line
    assert "'watermark': 1" in line and "'unaligned_hbm': 1" in line
    assert "held_blocks=4" in line and "held_requests=1" in line
    assert "avg_wait=" in line


def test_state_counts_a_request_once_per_reason():
    state = EarlyLoadState()
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


# -- a withdrawn admission gives the save frontier back ---------------------------


def _chunked_with_tracker():
    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase

    conn = ChunkedOffloadSchedulerBase.__new__(ChunkedOffloadSchedulerBase)
    conn._do_load = conn._do_save = True
    conn._save_tracker = {}
    conn._admission_undo = None
    conn._load_specs = {}
    conn._reqs_need_recv = {}
    conn._hit_save_floors = {"7": 48}
    conn._begin_load_lifecycle = lambda seq: None
    conn._forget_tier_hit = lambda sid: None
    conn._lmcache_hit_save_floor = lambda ls: 0
    conn.cancel_pending_load = lambda seq: None
    conn._lookup_results = {}
    return conn


def test_take_back_removes_the_tracker_entry_the_admission_created():
    conn = _chunked_with_tracker()
    seq = SimpleNamespace(id=7, num_cached_tokens=0)
    conn.update_state_after_alloc(seq)
    assert conn._save_tracker["7"][1] == 48
    conn.cancel_preadmission(seq)
    assert "7" not in conn._save_tracker


def test_take_back_keeps_an_older_save_generations_frontier():
    conn = _chunked_with_tracker()
    seq = SimpleNamespace(id=7, num_cached_tokens=0)
    conn._save_tracker["7"] = [seq, 16]  # a save already dispatched up to 16
    conn.update_state_after_alloc(seq)
    assert conn._save_tracker["7"][1] == 48
    conn.cancel_preadmission(seq)
    assert conn._save_tracker["7"][1] == 16


# -- the delayer's and the router's view of a tier waiter --------------------------


def _sched(monkeypatch):
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "0")
    sched = Scheduler(
        MockConfig(max_num_seqs=8, max_num_batched_tokens=64, num_kvcache_blocks=100)
    )
    sched.kv_connector = _Connector()
    return sched


def test_a_completion_reported_as_a_string_id_is_recognised(monkeypatch, seq_factory):
    sched = _sched(monkeypatch)
    seq = seq_factory(list(range(16)))
    seq.id = 123
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sched._still_loading(seq)
    sched.finished_recving_kv_req_ids = ["123"]
    assert not sched._still_loading(seq)


def test_pending_tokens_follow_the_replay_adjusted_resume_point(
    monkeypatch, seq_factory
):
    sched = _sched(monkeypatch)
    sched.block_manager.state = SimpleNamespace(transfer=SimpleNamespace(replays=True))
    seq = seq_factory(list(range(40)))
    seq.has_per_req_cache = True
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    seq.num_cached_tokens = 8
    seq.offload_loaded_tokens = 32
    hbs = sched.block_manager.hash_block_size
    # The prefill starts one hash block below the loaded boundary.
    assert sched.pending_prefill_tokens(seq) == 40 - (32 - hbs)
    # A failed load falls back to the HBM hit, replay-adjusted the same way.
    sched.failed_recving_kv_req_ids = [seq.id]
    assert sched.pending_prefill_tokens(seq) == 40 - (8 - hbs)


def test_early_loads_in_flight_do_not_use_up_the_probe_budget(monkeypatch, seq_factory):
    sched = _sched(monkeypatch)
    loading = [seq_factory(list(range(16))) for _ in range(4)]
    for seq in loading:
        seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        sched._early_load.entries[seq.id] = (seq, 0.0)
        sched.waiting.append(seq)
    sched.waiting.append(seq_factory(list(range(16))))  # a fifth, ready to admit
    assert sched._can_admit_head_prefill()


# -- slots, age and failed loads (review of #3097) -------------------------------------


def _parked(sched, seq_factory, n, *, early=True):
    seqs = []
    for _ in range(n):
        seq = seq_factory(list(range(16)))
        seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        if early:
            sched._early_load.entries[seq.id] = (seq, 0.0)
        seqs.append(seq)
    return seqs


def test_finished_early_loads_are_promoted_when_the_head_has_no_slot(
    monkeypatch, seq_factory
):
    """Every parked slot taken by early loads that finished: the head cannot park,
    Phase 2 stops there, and only promoting them frees a slot."""
    sched = _sched(monkeypatch)
    head = seq_factory(list(range(16)))
    parked = _parked(sched, seq_factory, sched.max_num_seqs)
    for seq in parked:
        sched.block_manager.allocate(seq)
    sched.waiting.extend([head, *parked])
    sched._num_parked_remote_kv = sched.max_num_seqs
    sched.finished_recving_kv_req_ids = [seq.id for seq in parked]
    sched._prioritize_ready_remote_kv_requests()
    assert list(sched.waiting)[: len(parked)] == parked
    assert sched.waiting[-1] is head


def test_early_load_leaves_the_last_slot_to_the_head(monkeypatch, seq_factory):
    sched = _sched(monkeypatch)
    sched._num_parked_remote_kv = sched.max_num_seqs - 1
    seq = seq_factory(list(range(16)))
    sched.add(seq)
    assert sched._early_load_tier_prefixes() == 0
    assert sched._early_load.skips["seq_slots"] == 1


def test_a_finished_tier_load_has_an_age_and_a_failed_one_is_not_admittable(
    monkeypatch, seq_factory
):
    sched = _sched(monkeypatch)
    ok, failed = _parked(sched, seq_factory, 2)
    ok.arrive_time = failed.arrive_time = 1.0
    sched.waiting.extend([ok, failed])
    assert sched._oldest_waiting_prefill_age_ms() == 0.0  # both still loading
    # What `_prioritize_ready_remote_kv_requests` leaves a finished load: WAITING
    # on its blocks. A failed one stays parked until admission resolves it.
    ok.status = SequenceStatus.WAITING
    ok.block_table = [0]
    ok.offload_loaded = True
    ok.num_cached_tokens = 8
    sched.failed_recving_kv_req_ids = [failed.id]
    assert list(sched._waiting_prefills()) == [ok]
    assert sched._oldest_waiting_prefill_age_ms() > 0.0
    sched.waiting.remove(ok)
    assert list(sched._waiting_prefills()) == []
    assert sched._oldest_waiting_prefill_age_ms() == 0.0


def test_a_routed_estimate_is_reported_once_with_the_reuse_found(
    monkeypatch, seq_factory
):
    """`share_cache` routing placed the request on a 48-token prefix; the rank
    reports what it really reused, once, when the request first prefills."""
    sched = _sched(monkeypatch)
    seq = seq_factory(list(range(64)))
    seq.dp_est_cached_tokens = 48
    seq.num_cached_tokens = 16
    for _ in range(2):
        sched._note_dp_reuse(seq)
    assert sched.dp_reuse_reports == [(seq.id, 16)]
    assert seq.dp_est_cached_tokens == 0
    unplaced = seq_factory(list(range(64)))
    sched._note_dp_reuse(unplaced)
    assert sched.dp_reuse_reports == [(seq.id, 16)]


def test_a_routed_estimate_reports_the_hit_not_the_replay_start(
    monkeypatch, seq_factory
):
    """A bounded replay rewinds `num_cached_tokens` below the hit; the report
    carries the hit, or the router counts a real hit as stale."""
    sched = _sched(monkeypatch)
    seq = seq_factory(list(range(64)))
    seq.dp_est_cached_tokens = 48
    seq.num_cached_tokens = 32
    seq.replay = Replay(48, False, (), 32)
    sched._note_dp_reuse(seq)
    assert sched.dp_reuse_reports == [(seq.id, 48)]


def test_settled_early_owner_keeps_fifo_until_the_head_cannot_allocate(
    monkeypatch, seq_factory
):
    sched, _, (head, owner) = _make(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {head.id}
    sched.schedule()
    assert owner.block_table and owner.id in sched._early_load.entries
    sched.finished_recving_kv_req_ids.append(owner.id)
    for _ in range(3):
        sched.schedule()
        assert owner.status == SequenceStatus.WAITING
        assert owner.id in sched._early_load.entries
        assert list(sched.waiting) == [head, owner]
    assert owner.id not in sched.finished_recving_kv_req_ids
    original = sched.block_manager.can_allocate_memoized
    monkeypatch.setattr(
        sched.block_manager,
        "can_allocate_memoized",
        lambda seq: -1 if seq is head else original(seq),
    )
    sched._prioritize_ready_remote_kv_requests()
    assert list(sched.waiting) == [owner, head]
    assert owner.block_table


def test_a_load_in_flight_ahead_of_the_blocked_head_does_not_hide_it(
    monkeypatch, seq_factory
):
    """The head is the first request Phase 2 would try, so loads in flight ahead
    of it do not stop a finished early load from being promoted past a head that
    waits for the blocks it holds."""
    sched = _sched(monkeypatch)
    head = seq_factory(list(range(16)))
    loading, done = _parked(sched, seq_factory, 2)
    sched.block_manager.allocate(done)
    original = sched.block_manager.can_allocate_memoized
    monkeypatch.setattr(
        sched.block_manager,
        "can_allocate_memoized",
        lambda seq: -1 if seq is head else original(seq),
    )
    sched.waiting.extend([loading, head, done])
    sched.finished_recving_kv_req_ids = [done.id]
    sched._prioritize_ready_remote_kv_requests()
    assert list(sched.waiting) == [done, loading, head]


@pytest.mark.parametrize("site", ["state_leg", "joint_disown", "atomic_chunk"])
def test_every_take_back_after_the_connector_was_told_cancels_the_preadmission(
    monkeypatch, seq_factory, site
):
    sched = _sched(monkeypatch)
    conn = sched.kv_connector
    conn.match = (0, False)
    seq = seq_factory(list(range(16)))
    sched.add(seq)
    if site == "state_leg":
        monkeypatch.setattr(
            sched, "_confirm_remote_load_after_alloc", lambda seq, needs: True
        )
        monkeypatch.setattr(sched, "_drop_state_leg_of_kv_load", lambda seq: False)
    elif site == "joint_disown":
        notify = sched._notify_connector_after_prefill_alloc

        def notify_with_joint_boundary(seq):
            notify(seq)
            seq.offload_joint.boundary_tokens = 4

        monkeypatch.setattr(
            sched, "_notify_connector_after_prefill_alloc", notify_with_joint_boundary
        )
        monkeypatch.setattr(
            sched.block_manager, "retreat_joint_to_hbm", lambda s: False
        )
        monkeypatch.setattr(
            sched.block_manager, "disown_claimed_prefix", lambda s: False
        )
    else:
        calls = []

        def waits_after_allocation(*args):
            calls.append(args)
            return len(calls) > 1

        monkeypatch.setattr(sched, "_waits_below_resume", waits_after_allocation)
    sched.schedule()
    assert conn.allocs == [seq.id]
    assert conn.cancelled == [seq.id]
    assert not seq.block_table and seq.status == SequenceStatus.WAITING


def test_early_loads_in_flight_do_not_spend_the_lookup_bypass_budget(
    monkeypatch, seq_factory
):
    """With a lookup waiter parked, Phase 2 walks past up to `_prefix_bypass_limit`
    further heads. Early loads in flight are walked past for free, as the
    prefillable-work probe does, so a ready request behind them is admitted
    whenever the probe says it can be."""

    class _LookupConnector(_Connector):
        waiter = None

        def lookup_pending(self, seq):
            return seq is self.waiter

        def lookup_would_defer(self, seq):
            return seq is self.waiter

    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "0")
    sched = Scheduler(
        MockConfig(max_num_seqs=64, max_num_batched_tokens=64, num_kvcache_blocks=2000)
    )
    conn = sched.kv_connector = _LookupConnector()
    conn.match = (0, False)
    waiter = conn.waiter = seq_factory(list(range(16)))
    early = _parked(sched, seq_factory, sched._prefix_bypass_limit())
    ready = seq_factory(list(range(16)))
    sched.waiting.extend([waiter, *early, ready])
    assert sched._can_admit_head_prefill()
    _, scheduled = sched.schedule()
    assert list(scheduled) == [ready.id]
