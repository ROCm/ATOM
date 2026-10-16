# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache MP lookup driver, transfer thread and same-step saves."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from test_lmcache_mp import _config, _WorkerAdapter, fake_lmcache_modules  # noqa: F401
from test_lmcache_mp_async_lookup import CHUNK, _Adapter, _Client, _Future

from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    SaveSpec,
)
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp import transfer_thread
from atom.kv_transfer.offload.mp import worker as mp_worker
from atom.kv_transfer.offload.mp.lookup_driver import _LockedAdapter
from atom.model_engine.sequence import SequenceStatus


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def _driven_client(adapter, window=4):
    lock = threading.RLock()
    client = mp_lookup._MPLookupClient(
        _LockedAdapter(adapter, lock),
        config=_config(),
        timeout=5.0,
        poll_interval=0.01,
        lock=lock,
        driver_window=window,
    )
    return client


class _PerRequestClient(_Client):
    """Acks every lookup at once; its status answers when released."""

    def __init__(self, chunks):
        super().__init__(chunks)
        self.ready = threading.Event()
        self.ready.set()

    def query_prefetch_status(self, request_id):
        self.status_queries.append(request_id)
        if not self.ready.is_set():
            return _Future(None, done=False)
        return _Future(self.chunks)


# ── lookup driver ────────────────────────────────────────────────────────


def test_driver_sends_and_answers_an_offered_lookup_without_a_pump():
    fake = _PerRequestClient(chunks=2)
    adapter = _Adapter(fake)
    client = _driven_client(adapter)
    try:
        assert client.offer("req", lambda: list(range(10)))
        # Nobody pumps: the driver sends it and chains the status query.
        assert _wait_until(lambda: "req" in client._async_hits)
        assert len(fake.lookups) == 1
        assert client.driver_stats["sent"] == 1
        # Admission consumes it with no second round trip.
        assert client.lookup(list(range(10)), "req") == 2 * CHUNK
        assert len(fake.lookups) == 1
    finally:
        client.close()


def test_driver_keeps_answered_plus_in_flight_lookups_within_the_window():
    fake = _PerRequestClient(chunks=1)
    adapter = _Adapter(fake)
    client = _driven_client(adapter, window=2)
    try:
        for i in range(5):
            client.offer(f"r{i}", lambda i=i: list(range(8 + i)))
        assert _wait_until(lambda: len(client._async_hits) == 2)
        time.sleep(0.02)
        assert len(fake.lookups) == 2  # held: answered ones keep read locks
        assert list(client._offered) == ["r2", "r3", "r4"]
        # Consuming one frees a slot for the next arrival, in order.
        assert client.lookup(list(range(8)), "r0") == CHUNK
        assert _wait_until(lambda: "r2" in client._async_hits)
        assert len(fake.lookups) == 3
    finally:
        client.close()


def test_driver_skips_a_request_that_stopped_waiting_and_discard_drops_offers():
    fake = _PerRequestClient(chunks=1)
    adapter = _Adapter(fake)
    client = _driven_client(adapter, window=1)
    try:
        client.offer("held", lambda: list(range(8)))
        assert _wait_until(lambda: "held" in client._async_hits)
        client.offer("gone", lambda: None)  # admitted or aborted meanwhile
        client.offer("dropped", lambda: list(range(8)))
        client.discard("dropped")
        client.discard("held")  # answered: its locks are released now
        assert _wait_until(lambda: not client._offered)
        time.sleep(0.02)
        assert len(fake.lookups) == 1
        assert adapter.freed and adapter.freed[0]["end"] == CHUNK
    finally:
        client.close()


def test_driver_releases_an_answer_left_unconsumed_and_asks_again(monkeypatch):
    fake = _PerRequestClient(chunks=1)
    adapter = _Adapter(fake)
    client = _driven_client(adapter)
    try:
        client.offer("req", lambda: list(range(8)))
        assert _wait_until(lambda: "req" in client._async_hits)
        with client._lock:
            client._answered_at["req"] -= mp_lookup._MAX_UNCONSUMED_HOLD_S + 1
        client._driver.notify()
        assert _wait_until(lambda: client.driver_stats["stale_released"] == 1)
        assert adapter.freed[0]["end"] == CHUNK
        assert _wait_until(lambda: len(fake.lookups) == 2)
        assert _wait_until(lambda: "req" in client._async_hits)
        # The fresh answer is not a detached lookup's: no save floor report.
        assert client.pop_discarded_hits() == {}
    finally:
        client.close()


def test_offer_is_a_noop_without_a_driver():
    client = mp_lookup._MPLookupClient(
        _Adapter(_Client(1)), config=_config(), timeout=1.0, poll_interval=0.01
    )
    assert not client.driving
    assert not client.offer("req", lambda: list(range(8)))


def test_scheduler_offers_only_waiting_requests_to_the_driver():
    adapter = _Adapter(_PerRequestClient(chunks=1))
    client = _driven_client(adapter)
    sched = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
    )
    sched._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        sched, _config(), chunk_size=CHUNK, lookup_client=client
    )
    try:
        waiting = SimpleNamespace(
            id=1,
            num_prompt_tokens=8,
            token_ids=list(range(8)),
            status=SequenceStatus.WAITING,
        )
        running = SimpleNamespace(
            id=2,
            num_prompt_tokens=8,
            token_ids=list(range(8)),
            status=SequenceStatus.RUNNING,
        )
        sched.offer_lookups([waiting, running])
        assert _wait_until(lambda: "1" in client._async_hits)
        time.sleep(0.02)
        assert [key[1] for key, _ in adapter._client.lookups] == [
            mp_lookup._mp_session_id(client._config, "1")
        ]
    finally:
        client.close()


# ── producer fences and the transfer thread ─────────────────────────────


class _FakeEvent:
    created = 0

    def __init__(self, interprocess=False):
        assert interprocess
        _FakeEvent.created += 1
        self.records = 0

    def record(self, stream=None):
        self.records += 1


def test_fences_reuse_events_and_retire_them_after_the_record_limit(monkeypatch):
    monkeypatch.setattr(transfer_thread.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(transfer_thread.torch.cuda, "current_stream", lambda: None)
    _FakeEvent.created = 0
    fences = transfer_thread._ProducerFences(slots=2)
    events = [fences.record() for _ in range(2 * transfer_thread._FENCE_MAX_RECORDS)]
    assert _FakeEvent.created == 2
    assert events[0].records == transfer_thread._FENCE_MAX_RECORDS
    fresh = fences.record()
    assert _FakeEvent.created == 3 and fresh is not events[0]
    assert fences._retired == [events[0]]
    fences.release_retired()
    assert fences._retired == []


def test_transfer_thread_runs_in_order_off_the_caller_and_returns_arguments():
    caller = threading.current_thread()
    seen = []
    thread = transfer_thread._TransferThread("mp-transfer-test")
    try:
        for i in range(20):
            thread.submit(
                lambda i, marker: seen.append((i, threading.current_thread())), i, i
            )
        assert thread.drain(5.0)
        assert [i for i, _ in seen] == list(range(20))
        assert all(t is not caller for _, t in seen)
        assert len(thread.take_spent()) == 20
        assert thread.take_spent() == []
    finally:
        thread.close(5.0)


def test_transfer_thread_reraises_a_failed_submission_on_the_caller():
    thread = transfer_thread._TransferThread("mp-transfer-test")
    try:

        def fail():
            raise RuntimeError("duplicate operation")

        thread.submit(fail)
        assert thread.drain(5.0)
        with pytest.raises(RuntimeError, match="duplicate operation"):
            thread.raise_error()
        thread.raise_error()  # reported once
    finally:
        thread.close(5.0)


# ── worker: same-step saves go out behind the forward ───────────────────


class _RecordingAdapter(_WorkerAdapter):
    ACCEPTS_RECORDED_PRODUCER_EVENTS = True

    def __init__(self):
        super().__init__()
        self.kwargs = []

    def submit_store_request(self, request_id, op, event, **kwargs):
        self.kwargs.append(kwargs)
        return super().submit_store_request(request_id, op, event)

    def submit_retrieve_request(self, request_id, op, event, **kwargs):
        self.kwargs.append(kwargs)
        return super().submit_retrieve_request(request_id, op, event)

    def submit_store_request_with_chunk_events(self, request_id, op, event, **kwargs):
        return self.submit_store_request(request_id, op, event, **kwargs)


def _save(req_id, *, after_forward):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=list(range(8)),
        block_ids=[10],
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
        after_forward=after_forward,
    )


def _meta(*requests):
    meta = LMCacheOffloadMetadata()
    for request in requests:
        meta.add_request(request)
    return meta


def _fenced_worker(adapter, *, threaded):
    worker = mp_worker.LMCacheMPConnector(_config())
    worker._adapter = adapter
    worker.chunk_size = 8
    worker.block_size = 8
    fences = iter(f"fence{i}" for i in range(100))
    worker._producer_fence = lambda: next(fences)
    if threaded:
        worker._fences = SimpleNamespace(release_retired=lambda: None)
        worker._transfer = transfer_thread._TransferThread("mp-transfer-test")
    return worker


@pytest.mark.usefixtures("fake_lmcache_modules")
@pytest.mark.parametrize("threaded", [False, True])
def test_same_step_save_waits_for_after_forward_and_its_own_fence(threaded):
    adapter = _RecordingAdapter()
    worker = _fenced_worker(adapter, threaded=threaded)
    try:
        worker.start_load_kv(
            _meta(_save(1, after_forward=False), _save(2, after_forward=True))
        )
        if threaded:
            worker._transfer.drain(5.0)
        # The earlier-step save goes out behind the pre-forward fence ...
        assert [(rid, ev) for rid, _, ev, _ in adapter.saves] == [
            ("atom-offload-dp0:1", "fence0")
        ]
        worker.after_forward()
        if threaded:
            worker._transfer.drain(5.0)
        # ... the same-step one behind a fence recorded after the forward.
        assert [(rid, ev) for rid, _, ev, _ in adapter.saves][1] == (
            "atom-offload-dp0:2",
            "fence1",
        )
        expected = {"producer_event_recorded": True} if threaded else {}
        assert adapter.kwargs == [expected, expected]
    finally:
        worker.close()


@pytest.mark.usefixtures("fake_lmcache_modules")
def test_same_step_saves_left_by_a_missing_hook_go_out_in_get_finished():
    adapter = _RecordingAdapter()
    worker = _fenced_worker(adapter, threaded=False)
    worker.start_load_kv(_meta(_save(3, after_forward=True)))
    assert adapter.saves == []
    worker.get_finished()  # runs after the forward on the same thread
    # Recorded there, so it still follows the forward.
    assert [ev for _, _, ev, _ in adapter.saves] == ["fence0"]


@pytest.mark.usefixtures("fake_lmcache_modules")
def test_worker_stays_inline_when_the_adapter_would_record_its_own_event(monkeypatch):
    monkeypatch.setattr(mp_worker.envs, "OFFLOAD_MP_TRANSFER_THREAD", True)
    adapter = _WorkerAdapter()
    adapter._producer_events = object()  # pooled, without the capability flag
    worker = mp_worker.LMCacheMPConnector(_config())
    worker._adapter = adapter
    worker._start_transfer_thread(0)
    assert worker._transfer is None and worker._fences is None


@pytest.mark.usefixtures("fake_lmcache_modules")
def test_threaded_save_failure_stops_the_engine_on_the_forward_thread():
    adapter = _RecordingAdapter()
    worker = _fenced_worker(adapter, threaded=True)
    try:
        worker._completed_save_operations.add("save:4")
        worker.start_load_kv(_meta(_save(4, after_forward=False)))
        worker._transfer.drain(5.0)
        with pytest.raises(RuntimeError, match="duplicate"):
            worker.get_finished()
    finally:
        worker.close()


# ── scheduler: emit a chunk's save in the step that computes it ─────────


def _same_step_scheduler():
    adapter = _Adapter(_Client(0))
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
    sched._same_step_save = True
    return sched


def _prefill_seq(sid=7, n=16):
    return SimpleNamespace(
        id=sid,
        num_tokens=n,
        num_prompt_tokens=n,
        num_cached_tokens=0,
        token_ids=list(range(n)),
        block_table=list(range(100, 100 + n)),
    )


def _saves(meta):
    return [req for req in meta.requests if req.save_spec is not None]


def test_save_of_a_chunk_is_emitted_in_the_step_that_computes_it():
    sched = _same_step_scheduler()
    seq = _prefill_seq()
    sched.update_state_after_alloc(seq)
    # This step's forward computes tokens [0, 10) of the 16-token prompt.
    sched.note_scheduled_tokens({seq.id: seq}, [10])
    (save,) = _saves(sched.build_connector_meta())
    assert len(save.token_ids) == 2 * CHUNK  # chunk floor of 10
    assert save.after_forward
    assert not save.is_last_prefill
    # Nothing was computed before this step: the frontier came from the note.
    assert sched._save_frontier(seq) == 0


def test_save_covering_only_earlier_steps_is_not_held_for_the_forward():
    sched = _same_step_scheduler()
    seq = _prefill_seq()
    seq.num_cached_tokens = 8
    sched.update_state_after_alloc(seq)
    # A decode-like step that computes nothing past the next chunk boundary.
    sched.note_scheduled_tokens({seq.id: seq}, [1])
    (save,) = _saves(sched.build_connector_meta())
    assert len(save.token_ids) == 8
    assert not save.after_forward


def test_idle_build_does_not_reuse_the_previous_steps_note():
    sched = _same_step_scheduler()
    seq = _prefill_seq()
    sched.update_state_after_alloc(seq)
    sched._step_compute_end = {}
    sched.note_scheduled_tokens({seq.id: seq}, [16])
    sched._same_step_save = True
    sched.build_connector_meta()  # consumes the note
    sched._save_inflight.clear()
    seq2 = _prefill_seq(sid=8)
    sched.update_state_after_alloc(seq2)
    # An idle-path build (no note, no forward) sees only computed tokens.
    assert _saves(sched.build_connector_meta()) == []


def test_note_is_ignored_when_same_step_saves_are_off():
    sched = _same_step_scheduler()
    sched._same_step_save = False
    seq = _prefill_seq()
    sched.update_state_after_alloc(seq)
    sched.note_scheduled_tokens({seq.id: seq}, [16])
    assert _saves(sched.build_connector_meta()) == []
