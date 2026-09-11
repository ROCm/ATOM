# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import threading
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload.dense.connector import (
    DENSE_PAGE_RETIRED_CHANNEL,
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
    DenseOffloadConnector,
)
from atom.kv_transfer.offload.dense.save_executor import (
    CancellableSaveExecutor,
    SaveQueueFull,
    SeenSaveGenerations,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)


def _worker(monkeypatch, *, capacity=2, admission=True, cls=DenseOffloadConnector):
    monkeypatch.setenv("OFFLOAD_COPY_WORKERS", "1")
    monkeypatch.setenv("OFFLOAD_SAVE_ADMISSION", "1" if admission else "0")
    monkeypatch.setenv("OFFLOAD_MAX_PENDING_SAVES", str(capacity))
    worker = cls(
        SimpleNamespace(
            kv_transfer_config={"kv_role": "offload"},
            kv_cache_block_size=4,
            decode_context_parallel_size=1,
        )
    )
    worker.chunk_size = 8
    return worker


def _request(generation):
    return LMCacheReqMeta(
        req_id=str(generation),
        token_ids=list(range(8)),
        block_ids=[2 * generation, 2 * generation + 1],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(str(generation), generation),
    )


def _metadata(*requests, cancellations=()):
    metadata = LMCacheOffloadMetadata()
    metadata.requests.extend(requests)
    metadata.cancel_save_operations = list(cancellations)
    return metadata


def _completion(channel, operation, succeeded=True):
    return ConnectorCompletion(channel, operation, succeeded)


def test_cancelled_tasks_release_physical_queue_and_request_metadata():
    executor = CancellableSaveExecutor(
        max_workers=1, capacity=2, thread_name_prefix="test-save"
    )
    entered, release = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(5)

    class Payload:
        pass

    try:
        running = executor.submit(block)
        assert entered.wait(5)
        assert not running.cancel()
        for _ in range(100):
            payload = Payload()
            reference = weakref.ref(payload)
            pending = executor.submit(
                lambda value: pytest.fail("cancelled task ran"), payload
            )
            with pytest.raises(SaveQueueFull):
                executor.submit(lambda: None)
            assert pending.cancel()
            assert pending.cancel()  # repeated cancellation has no extra release
            del payload
            assert reference() is None
            assert executor.counts() == (0, 1)
            assert len(executor._queue) == len(executor._queued) == 0
    finally:
        release.set()
        executor.shutdown()


def test_executor_releases_capacity_before_done_callback():
    executor = CancellableSaveExecutor(
        max_workers=1, capacity=1, thread_name_prefix="test-save"
    )
    entered, release, callback_done = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    followups = []

    def block():
        entered.set()
        assert release.wait(5)

    def completed(_future):
        followups.append(executor.submit(lambda: 42))
        callback_done.set()

    try:
        future = executor.submit(block)
        assert entered.wait(5)
        future.add_done_callback(completed)
        release.set()
        assert callback_done.wait(5)
        assert followups[0].result(timeout=5) == 42
    finally:
        release.set()
        executor.shutdown()


def test_executor_submission_failure_leaves_no_task_or_capacity_reservation():
    executor = CancellableSaveExecutor(
        max_workers=1, capacity=1, thread_name_prefix="test-save"
    )

    class RejectingQueue(deque):
        def append(self, _task):
            raise RuntimeError("cannot enqueue")

    try:
        executor._queue = RejectingQueue()
        with pytest.raises(RuntimeError, match="cannot enqueue"):
            executor.submit(lambda: pytest.fail("failed submission ran"))
        assert executor.counts() == (0, 0)
        assert not executor._queued
        executor._queue = deque()
        assert executor.submit(lambda: 42).result(timeout=5) == 42
    finally:
        executor.shutdown()


def test_seen_generation_intervals_allow_out_of_order_cancellation():
    generations = SeenSaveGenerations()
    for generation in (5, 1, 2, 4, 3):
        assert generations.remember(generation)
    assert generations._intervals == [(1, 5)]
    for generation in range(1, 6):
        assert not generations.remember(generation)
    for generation in range(6, 1000):
        assert generations.remember(generation)
    assert generations._intervals == [(1, 999)]


def test_worker_normal_return_publishes_result_and_retirement(monkeypatch):
    worker = _worker(monkeypatch)
    request = _request(1)
    observations = []

    def store(_tokens, **_kwargs):
        observations.append(worker.get_finished())

    worker._engine = SimpleNamespace(gpu_connector=None, store=store)
    try:
        worker.start_load_kv(_metadata(request))
        worker.close()
        assert observations[0].finished_saving == set()
        output = worker.get_finished()
        assert output.finished_saving == {request.save_operation}
        assert output.connector_completions == {
            _completion(DENSE_PAGE_STORE_CHANNEL, request.save_operation),
            _completion(DENSE_PAGE_RETIRED_CHANNEL, request.save_operation),
        }
        assert worker._save_futures == {}
        assert worker._save_executor.counts() == (0, 0)
    finally:
        worker.close()


def test_worker_queue_rejection_is_safe_without_cancelling_running_save(monkeypatch):
    worker = _worker(monkeypatch, capacity=1)
    entered, release = threading.Event(), threading.Event()
    first, second = _request(1), _request(2)
    calls = []

    def store(_tokens, *, req_id, **_kwargs):
        calls.append(req_id)
        entered.set()
        assert release.wait(5)

    worker._engine = SimpleNamespace(gpu_connector=None, store=store)
    try:
        worker.start_load_kv(_metadata(first))
        assert entered.wait(5)
        worker.start_load_kv(_metadata(second))
        output = worker.get_finished()
        assert output.finished_saving == {second.save_operation}
        assert output.connector_completions == {
            _completion(DENSE_PAGE_STORE_CHANNEL, second.save_operation, False),
            _completion(DENSE_PAGE_RETIRED_CHANNEL, second.save_operation),
        }
        worker.start_load_kv(_metadata(cancellations=[first.save_operation]))
        assert worker.get_finished().is_empty()
        assert worker._save_executor.counts() == (0, 1)
        worker.start_load_kv(_metadata(second))
        assert worker.get_finished().is_empty()
        assert calls == ["1"]
    finally:
        release.set()
        worker.close()
    assert worker.get_finished().finished_saving == {first.save_operation}


def test_worker_queued_cancel_removes_future_and_prevents_late_dispatch(monkeypatch):
    worker = _worker(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    first, second = _request(1), _request(2)
    calls = []

    def store(_tokens, *, req_id, **_kwargs):
        calls.append(req_id)
        entered.set()
        assert release.wait(5)

    worker._engine = SimpleNamespace(gpu_connector=None, store=store)
    try:
        worker.start_load_kv(_metadata(first))
        assert entered.wait(5)
        worker.start_load_kv(_metadata(second))
        assert worker._save_executor.counts() == (1, 1)
        worker.start_load_kv(_metadata(cancellations=[second.save_operation]))
        assert worker._save_executor.counts() == (0, 1)
        assert second.save_operation not in worker._save_futures
        output = worker.get_finished()
        assert output.finished_saving == {second.save_operation}
        assert (
            _completion(DENSE_PAGE_STORE_CHANNEL, second.save_operation, False)
            in output.connector_completions
        )
        worker.start_load_kv(_metadata(second))
    finally:
        release.set()
        worker.close()
    assert calls == ["1"]


def test_worker_cancel_before_dispatch_and_duplicate_completion(monkeypatch):
    worker = _worker(monkeypatch)
    request = _request(2)
    worker._engine = SimpleNamespace(
        gpu_connector=None,
        store=lambda *_args, **_kwargs: pytest.fail("cancelled save ran"),
    )
    try:
        worker.start_load_kv(_metadata(request, cancellations=[request.save_operation]))
        output = worker.get_finished()
        assert output.finished_saving == {request.save_operation}
        assert output.connector_completions == {
            _completion(DENSE_PAGE_STORE_CHANNEL, request.save_operation, False),
            _completion(DENSE_PAGE_RETIRED_CHANNEL, request.save_operation),
        }
        worker.start_load_kv(_metadata(request, cancellations=[request.save_operation]))
        assert worker.get_finished().is_empty()
    finally:
        worker.close()


def test_worker_duplicate_active_and_retired_dispatch_runs_once(monkeypatch):
    worker = _worker(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    request = _request(1)
    calls = []

    def store(_tokens, **_kwargs):
        calls.append(True)
        entered.set()
        assert release.wait(5)

    worker._engine = SimpleNamespace(gpu_connector=None, store=store)
    try:
        worker.start_load_kv(_metadata(request))
        assert entered.wait(5)
        worker.start_load_kv(_metadata(request))
        assert worker._save_executor.counts() == (0, 1)
        release.set()
        worker.close()
        assert worker.get_finished().finished_saving == {request.save_operation}
        worker.start_load_kv(_metadata(request))
        assert worker.get_finished().is_empty()
        assert calls == [True]
    finally:
        release.set()
        worker.close()


@pytest.mark.parametrize("fence_fails", [False, True])
def test_failed_store_only_retires_after_both_streams_fence(monkeypatch, fence_fails):
    worker = _worker(monkeypatch)
    request = _request(1)
    threads, fences = [], []

    class Stream:
        def __init__(self, name):
            self.name = name

        def synchronize(self):
            threads.append(threading.get_ident())
            fences.append(self.name)
            if fence_fails and self.name == "pack":
                raise RuntimeError("GPU fence unavailable")

    def store(_tokens, **_kwargs):
        threads.append(threading.get_ident())
        raise RuntimeError("store failed after submitting GPU reads")

    state = SimpleNamespace(pack_stream=Stream("pack"), copy_stream=Stream("copy"))
    worker._engine = SimpleNamespace(
        gpu_connector=SimpleNamespace(_thread_state=lambda: state), store=store
    )
    try:
        worker.start_load_kv(_metadata(request))
        worker.close()
        output = worker.get_finished()
        assert (
            _completion(DENSE_PAGE_STORE_CHANNEL, request.save_operation, False)
            in output.connector_completions
        )
        assert fences == ["pack", "copy"]
        assert len(set(threads)) == 1
        assert threads[0] != threading.get_ident()
        if fence_fails:
            assert output.finished_saving == set()
            assert output.connector_completions == {
                _completion(DENSE_PAGE_STORE_CHANNEL, request.save_operation, False)
            }
        else:
            assert output.finished_saving == {request.save_operation}
            assert (
                _completion(DENSE_PAGE_RETIRED_CHANNEL, request.save_operation)
                in output.connector_completions
            )
        worker.start_load_kv(_metadata(cancellations=[request.save_operation]))
        assert worker.get_finished().is_empty()
    finally:
        worker.close()


def test_incremental_source_report_precedes_store_and_retirement(monkeypatch):
    worker = _worker(monkeypatch)
    request = _request(1)
    identity = SaveSourceGroupId(request.save_operation, ((0, 8),))
    entered, release = threading.Event(), threading.Event()

    def store(_tokens, **_kwargs):
        worker._source_group_safe(identity)
        entered.set()
        assert release.wait(5)

    worker._engine = SimpleNamespace(gpu_connector=None, store=store)
    try:
        worker.start_load_kv(_metadata(request))
        assert entered.wait(5)
        output = worker.get_finished()
        assert output.finished_saving == set()
        assert output.connector_completions == {
            _completion(DENSE_PAGE_SOURCE_SAFE_CHANNEL, identity)
        }
    finally:
        release.set()
        worker.close()
    assert worker.get_finished().finished_saving == {request.save_operation}


def test_worker_submit_after_shutdown_is_never_started_rejection(monkeypatch):
    worker = _worker(monkeypatch)
    request = _request(1)
    try:
        worker._save_executor.shutdown()
        worker.start_load_kv(_metadata(request))
        output = worker.get_finished()
        assert output.finished_saving == {request.save_operation}
        assert output.connector_completions == {
            _completion(DENSE_PAGE_STORE_CHANNEL, request.save_operation, False),
            _completion(DENSE_PAGE_RETIRED_CHANNEL, request.save_operation),
        }
        assert worker._save_futures == {}
    finally:
        worker.close()


def test_load_executor_progresses_while_save_worker_is_blocked(monkeypatch):
    worker = _worker(monkeypatch)
    save_started, release, loaded = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    request = _request(1)

    def store(_tokens, **_kwargs):
        save_started.set()
        assert release.wait(5)

    def retrieve(_tokens, *, mask, **_kwargs):
        loaded.set()
        return mask.clone()

    worker._engine = SimpleNamespace(
        gpu_connector=None,
        store=store,
        retrieve=retrieve,
        lookup_unpin=lambda _req: None,
    )
    load_request = LMCacheReqMeta(
        req_id="load",
        token_ids=list(range(8)),
        block_ids=[12, 13],
        load_spec=LoadSpec(hbm_cached_tokens=0, lmcache_cached_tokens=8),
    )
    try:
        worker.start_load_kv(_metadata(request))
        assert save_started.wait(5)
        worker.start_load_kv(_metadata(load_request))
        assert loaded.wait(5)
        assert worker._save_executor.counts() == (0, 1)
    finally:
        release.set()
        worker.close()
    assert worker.get_finished().finished_loading == {"load"}


def test_admission_disabled_preserves_safe_queue_and_cancellation(monkeypatch):
    worker = _worker(monkeypatch, capacity=1, admission=False)
    entered, release = threading.Event(), threading.Event()
    requests = [_request(generation) for generation in range(1, 6)]
    calls = []

    def store(_tokens, *, req_id, **_kwargs):
        calls.append(req_id)
        entered.set()
        assert release.wait(5)

    worker._engine = SimpleNamespace(gpu_connector=None, store=store)
    try:
        worker.start_load_kv(_metadata(requests[0]))
        assert entered.wait(5)
        worker.start_load_kv(_metadata(*requests[1:]))
        assert worker._save_executor.counts() == (4, 1)
        worker.start_load_kv(
            _metadata(cancellations=[req.save_operation for req in requests[1:]])
        )
        assert worker._save_executor.counts() == (0, 1)
        assert worker.get_finished().finished_saving == {
            req.save_operation for req in requests[1:]
        }
    finally:
        release.set()
        worker.close()
    assert calls == ["1"]
    assert worker.get_finished().finished_saving == {requests[0].save_operation}


def test_non_early_release_connector_keeps_legacy_executor_and_completion(monkeypatch):
    class LegacyDense(DenseOffloadConnector):
        _supports_early_block_release = False

    worker = _worker(monkeypatch, cls=LegacyDense)
    request = _request(1)
    worker._engine = SimpleNamespace(
        gpu_connector=None, store=lambda *_args, **_kwargs: None
    )
    try:
        assert isinstance(worker._save_executor, ThreadPoolExecutor)
        assert isinstance(worker._load_executor, ThreadPoolExecutor)
        worker.start_load_kv(_metadata(request))
        worker.close()
        output = worker.get_finished()
        assert output.finished_saving == {request.save_operation}
        assert output.connector_completions == set()
    finally:
        worker.close()
