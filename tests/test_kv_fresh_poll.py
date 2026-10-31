# SPDX-License-Identifier: MIT

"""A fresh KV completion poll after the forward (``ATOM_KV_POLL_FRESH_WAIT_MS``)."""

import queue
import threading
import time
from unittest.mock import Mock

from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.types import KVConnectorOutput
    from atom.model_engine.async_proc import AsyncIOProcManager
    from atom.model_engine.engine_core import EngineCore, FreshKVPollStats

_POLL = "async_proc_aggregation"


def _manager(world_size=2):
    manager = AsyncIOProcManager.__new__(AsyncIOProcManager)
    manager.label = "test-kv-manager"
    manager.proc_num = world_size
    manager.rpc_broadcast_mq = Mock()
    manager.kv_output_aggregator = None
    manager.kv_outputs_queues = [queue.Queue() for _ in range(world_size)]
    manager._pending_kv_aggregation = None
    return manager


def test_wait_collects_replies_that_arrive_within_it():
    manager = _manager()

    def reply_later():
        time.sleep(0.02)
        for output_queue in manager.kv_outputs_queues:
            output_queue.put(KVConnectorOutput(finished_recving={"r"}))

    thread = threading.Thread(target=reply_later)
    thread.start()
    result = manager.call_func_with_aggregation(_POLL, timeout=0, wait_s=1.0)
    thread.join()

    assert result is not None
    assert result.finished_recving == {"r"}
    assert manager.rpc_broadcast_mq.enqueue.call_count == 2


def test_wait_gives_up_after_wait_s_and_keeps_the_aggregation_open():
    manager = _manager()
    manager.kv_outputs_queues[1].put(KVConnectorOutput(finished_recving={"r"}))

    t0 = time.monotonic()
    assert manager.call_func_with_aggregation(_POLL, timeout=0, wait_s=0.02) is None
    assert time.monotonic() - t0 < 0.5
    pending = manager._pending_kv_aggregation
    assert pending is not None and pending.missing_worker_ranks() == [0]

    manager.kv_outputs_queues[0].put(KVConnectorOutput(finished_recving={"r"}))
    result = manager.call_func_with_aggregation(_POLL, timeout=0)
    assert result is not None and result.finished_recving == {"r"}
    assert manager.rpc_broadcast_mq.enqueue.call_count == 2


class _Scheduler:
    def __init__(self, parked):
        self._num_parked_remote_kv = parked
        self.outputs = []

    def _update_from_kv_xfer_finished(self, out):
        self.outputs.append(out)


class _RunnerMgr:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    def call_func_with_aggregation(self, name, wait_s=0.0):
        self.calls.append(wait_s)
        return self._outputs.pop(0) if self._outputs else None


def _engine(*, parked, wait_s, outputs):
    proc = EngineCore.__new__(EngineCore)
    proc.label = "test-engine"
    proc.kv_transfer_enabled = True
    proc._fresh_kv_poll_s = wait_s
    proc._fresh_kv_poll_stats = FreshKVPollStats()
    proc.scheduler = _Scheduler(parked)
    proc.runner_mgr = _RunnerMgr(outputs)
    return proc


def test_fresh_poll_follows_a_completed_one_while_requests_are_parked():
    stale = KVConnectorOutput(finished_recving={"a"})
    fresh = KVConnectorOutput(finished_recving={"b"})
    proc = _engine(parked=1, wait_s=0.005, outputs=[stale, fresh])

    proc._poll_kv_transfer_progress()

    assert proc.runner_mgr.calls == [0.0, 0.005]
    assert proc.scheduler.outputs == [stale, fresh]
    stats = proc._fresh_kv_poll_stats
    assert (stats.polls, stats.completed) == (1, 1)


def test_no_fresh_poll_without_parked_requests_or_when_off():
    for parked, wait_s in ((0, 0.005), (2, 0.0)):
        out = KVConnectorOutput()
        proc = _engine(parked=parked, wait_s=wait_s, outputs=[out])
        proc._poll_kv_transfer_progress()
        assert proc.runner_mgr.calls == [0.0]
        assert proc.scheduler.outputs == [out]


def test_no_fresh_poll_while_the_previous_one_is_still_missing_ranks():
    proc = _engine(parked=1, wait_s=0.005, outputs=[None])

    proc._poll_kv_transfer_progress()

    assert proc.runner_mgr.calls == [0.0]
    assert proc.scheduler.outputs == [None]
