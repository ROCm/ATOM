# SPDX-License-Identifier: MIT
# PP-stage offload KV status aggregation (GPU-free).

import sys
import types
from unittest.mock import MagicMock

import pytest

# pp_engine_core pulls in async_proc, which needs a GPU aiter build; stub the
# import chain so the status-aggregation logic can be tested on CPU.
for _name in ("aiter", "aiter.dist", "aiter.dist.shm_broadcast"):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.__getattr__ = lambda _attr: MagicMock()
        sys.modules[_name] = _mod

from atom.kv_transfer.disaggregation.pp_kv_aggregator import PPKVAggregator
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.pp_engine_core import PPEngineCoreProc


class FakeScheduler:
    def __init__(self):
        self.outputs = []

    def _update_from_kv_xfer_finished(self, out):
        self.outputs.append(out)

    def released_sending(self):
        rel = set()
        for out in self.outputs:
            rel |= set(out.finished_sending or ())
        return rel

    def released_saving(self):
        rel = set()
        for out in self.outputs:
            rel |= set(out.finished_saving or ())
        return rel


class FakeRunnerMgr:
    """Returns one queued worker-side KVConnectorOutput per poll."""

    def __init__(self, outputs):
        self._outputs = list(outputs)

    def call_func_with_aggregation(self, name):
        assert name == "async_proc_aggregation"
        return self._outputs.pop(0) if self._outputs else KVConnectorOutput()


class FakePPTransport:
    """Returns one queued list of (pp_rank, output) per poll."""

    def __init__(self, messages):
        self._messages = list(messages)

    def recv_kv_status(self, timeout_ms=0):
        return self._messages.pop(0) if self._messages else []


def _head(pp_size, local_outputs, downstream_messages=()):
    proc = PPEngineCoreProc.__new__(PPEngineCoreProc)
    proc.kv_transfer_enabled = True
    proc.pp_size = pp_size
    proc._pp_kv_aggregator = None
    proc._held_sending = {}
    proc.scheduler = FakeScheduler()
    proc.runner_mgr = FakeRunnerMgr(local_outputs)
    proc.pp_transport = FakePPTransport(downstream_messages)
    return proc


def test_send_waits_for_every_pp_stage_save():
    proc = _head(
        pp_size=3,
        local_outputs=[
            KVConnectorOutput(finished_sending={"a"}, finished_saving={"a"}),
            KVConnectorOutput(),
        ],
        downstream_messages=[
            [(1, KVConnectorOutput(finished_saving={"a"}))],
            [(2, KVConnectorOutput(finished_saving={"a"}))],
        ],
    )

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == set()  # stage 2 still saving
    assert proc._held_sending == {"a": "a"}

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {"a"}
    assert proc.scheduler.released_saving() == {"a"}
    assert proc._held_sending == {}


def test_send_without_a_save_is_not_held():
    # Once the aggregator exists, a later send-only request (prompt shorter
    # than the offload chunk, or already persisted) must still pass straight
    # through — no finished_saving is ever coming for it.
    proc = _head(
        pp_size=2,
        local_outputs=[
            KVConnectorOutput(finished_sending={"a"}, finished_saving={"a"}),
            KVConnectorOutput(finished_sending={"b"}),
        ],
        downstream_messages=[[(1, KVConnectorOutput(finished_saving={"a"}))], []],
    )

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {"a"}

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {"a", "b"}
    assert proc._held_sending == {}


def test_send_passes_through_before_any_offload_activity():
    proc = _head(pp_size=2, local_outputs=[KVConnectorOutput(finished_sending={"a"})])
    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {"a"}
    assert proc._pp_kv_aggregator is None


def test_recv_bypasses_the_aggregator():
    proc = _head(
        pp_size=2,
        local_outputs=[KVConnectorOutput(finished_recving={"a"}, failed_recving={"b"})],
    )
    proc._poll_kv_transfer_progress()
    assert proc.scheduler.outputs[0].finished_recving == {"a"}
    assert proc.scheduler.outputs[0].failed_recving == {"b"}


def test_aggregator_requires_all_stages():
    agg = PPKVAggregator(3)
    assert agg.ingest(0, KVConnectorOutput(finished_saving={"a"})).is_empty()
    assert agg.ingest(1, KVConnectorOutput(finished_saving={"a"})).is_empty()
    assert agg.ingest(2, KVConnectorOutput(finished_saving={"a"})).finished_saving == {
        "a"
    }


def test_aggregator_rejects_bad_pp_size():
    with pytest.raises(ValueError):
        PPKVAggregator(0)
