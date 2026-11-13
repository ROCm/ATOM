# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import queue
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def engine_step(monkeypatch):
    # These tests exercise scheduling only, not the aiter-backed worker RPC.
    async_proc = ModuleType("atom.model_engine.async_proc")
    async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.async_proc", async_proc)
    from atom.model_engine import engine_core

    event = Mock()
    sleep = event.wait
    monkeypatch.setattr(engine_core.tracing, "record_forward", Mock())
    batch = SimpleNamespace(
        req_ids=[], connector_meta_output=None, total_seqs_num_prefill=0
    )
    scheduler = SimpleNamespace(
        schedule=Mock(return_value=(batch, {})),
        take_rejected=Mock(return_value=[]),
        compute_detailed_aggregates=Mock(),
        metrics=SimpleNamespace(record_forward=Mock()),
        prefill_delayer=None,
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        kv_transfer_enabled=True,
        runner_mgr=SimpleNamespace(call_func=Mock(return_value="forward-result")),
        _poll_kv_transfer_progress=Mock(return_value=False),
        _kv_progress_event=event,
        input_queue=queue.Queue(),
        utility_queue=queue.Queue(),
        _postprocess=Mock(),
        label="test",
    )
    engine._wait_for_kv_progress = lambda: engine_core.EngineCore._wait_for_kv_progress(
        engine
    )
    return engine_core, engine, batch, sleep


@pytest.mark.parametrize("metadata", [None, SimpleNamespace(reqs_to_recv={})])
def test_empty_kv_batch_yields_and_polls_without_dispatch(engine_step, metadata):
    module, engine, batch, sleep = engine_step
    batch.connector_meta_output = metadata

    assert module.EngineCore._process_engine_step_inner(engine) is False

    engine.runner_mgr.call_func.assert_not_called()
    engine._poll_kv_transfer_progress.assert_called_once_with()
    engine._postprocess.assert_not_called()
    sleep.assert_called_once_with(module.KV_IDLE_DRAIN_INTERVAL_S)


def test_receive_only_batch_dispatches_before_polling_and_yielding(engine_step):
    module, engine, batch, sleep = engine_step
    metadata = SimpleNamespace(reqs_to_recv={"request": object()})
    batch.connector_meta_output = metadata
    order = Mock()
    order.attach_mock(engine.runner_mgr.call_func, "dispatch")
    order.attach_mock(engine._poll_kv_transfer_progress, "poll")
    order.attach_mock(sleep, "sleep")

    assert module.EngineCore._process_engine_step_inner(engine) is False

    assert [call[0] for call in order.mock_calls] == ["dispatch", "poll", "sleep"]
    engine.runner_mgr.call_func.assert_called_once_with(
        "process_kvconnector_output", metadata
    )


def test_metadata_work_hook_preserves_unpin_only_dispatch(engine_step):
    module, engine, batch, sleep = engine_step
    from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata

    metadata = LMCacheOffloadMetadata()
    metadata.lookup_requests_in_step = ["request"]
    batch.connector_meta_output = metadata

    assert module.EngineCore._process_engine_step_inner(engine) is False

    engine.runner_mgr.call_func.assert_called_once_with(
        "process_kvconnector_output", metadata
    )
    sleep.assert_called_once_with(module.KV_IDLE_DRAIN_INTERVAL_S)


def test_non_kv_empty_batch_does_not_sleep(engine_step):
    module, engine, _, sleep = engine_step
    engine.kv_transfer_enabled = False

    assert module.EngineCore._process_engine_step_inner(engine) is False

    sleep.assert_not_called()


def test_reported_completion_does_not_wait(engine_step):
    module, engine, _, wait = engine_step
    engine._poll_kv_transfer_progress.return_value = True

    assert module.EngineCore._process_engine_step_inner(engine) is False

    wait.assert_not_called()


@pytest.mark.parametrize("queue_name", ["input_queue", "utility_queue"])
def test_already_queued_work_does_not_wait(engine_step, queue_name):
    module, engine, _, wait = engine_step
    getattr(engine, queue_name).put_nowait(object())

    module.EngineCore._process_engine_step_inner(engine)

    wait.assert_not_called()


def test_arrival_during_poll_is_latched_until_wait(engine_step):
    module, engine, _, _ = engine_step
    event = threading.Event()
    engine._kv_progress_event = event

    def poll():
        event.set()
        return False

    engine._poll_kv_transfer_progress.side_effect = poll
    assert module.EngineCore._process_engine_step_inner(engine) is False
    assert event.is_set()


def test_arrival_between_queue_check_and_wait_is_not_lost(engine_step):
    module, engine, _, _ = engine_step
    event = threading.Event()
    engine._kv_progress_event = event

    class ArrivalQueue:
        def empty(self):
            event.set()
            return True

    engine.utility_queue = ArrivalQueue()
    module.EngineCore._wait_for_kv_progress(engine)
    assert event.is_set()


def test_wait_releases_thread_and_wakes_on_progress(engine_step):
    module, engine, _, _ = engine_step
    event = threading.Event()
    waiting = threading.Event()
    done = threading.Event()

    class ObservedEvent:
        def wait(self, timeout):
            assert timeout == module.KV_IDLE_DRAIN_INTERVAL_S
            waiting.set()
            # A longer test-only timeout avoids depending on OS scheduling
            # within the production 1 ms interval.
            assert event.wait(5)

    engine._kv_progress_event = ObservedEvent()

    def run():
        module.EngineCore._wait_for_kv_progress(engine)
        done.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert waiting.wait(5)
        event.set()
        assert done.wait(5)
    finally:
        event.set()
        thread.join(5)
    engine.runner_mgr.call_func.assert_not_called()


def test_forward_batch_keeps_forward_and_postprocess_without_sleep(engine_step):
    module, engine, batch, sleep = engine_step
    batch.req_ids = ["request"]

    assert module.EngineCore._process_engine_step_inner(engine) is True

    engine.runner_mgr.call_func.assert_called_once_with("forward", batch, wait_out=True)
    engine._poll_kv_transfer_progress.assert_called_once_with()
    engine._postprocess.assert_called_once()
    sleep.assert_not_called()
