# SPDX-License-Identifier: MIT
# Idle-time KV transfer drain (GPU-free).

"""A request leaves ``running`` in the same postprocess pass that parks it in
``deferred_free_blocks``, so ``Scheduler.is_finished()`` reports idle while its
send or save is still in flight. The busy loops therefore gate on
``has_pending_kv_work()`` as well; without it the last transfer before a lull
is never polled, its blocks are never freed, and its save is never reported.
"""

import time
from types import SimpleNamespace

from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.pp_kv_aggregator import PPKVAggregator
    from atom.kv_transfer.disaggregation.types import KVConnectorOutput
    from atom.model_engine import engine_core as engine_core_mod
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.pp_engine_core import PPEngineCoreProc


class FakeConnector:
    is_offload = True

    def __init__(self, pending=False, requests=(1,)):
        self.pending = pending
        self.requests = list(requests)
        self.builds = 0

    def has_pending_work(self):
        return self.pending

    def build_connector_meta(self):
        self.builds += 1
        return SimpleNamespace(requests=list(self.requests))


class FakeScheduler:
    def __init__(self, deferred=None, connector=None):
        self.deferred_free_blocks = dict(deferred or {})
        self.kv_connector = connector
        self.outputs = []
        self.state_publishes = 0

    def _publish_state_loads(self):
        self.state_publishes += 1

    def _publish_state_stores(self):
        self.state_publishes += 1

    def _update_from_kv_xfer_finished(self, out):
        self.outputs.append(out)

    def released_sending(self):
        rel = set()
        for out in self.outputs:
            rel |= set(out.finished_sending or ())
        return rel


class FakeRunnerMgr:
    """Returns one queued worker-side output per poll; records dispatches."""

    def __init__(self, outputs=()):
        self._outputs = list(outputs)
        self.polls = 0
        self.dispatched = []

    def call_func_with_aggregation(self, name):
        assert name == "async_proc_aggregation"
        self.polls += 1
        return self._outputs.pop(0) if self._outputs else KVConnectorOutput()

    def call_func(self, name, *args, **kwargs):
        self.dispatched.append(name)


class FakePPTransport:
    """Returns one queued list of (pp_rank, output) per poll; records sends."""

    def __init__(self, messages=()):
        self._messages = list(messages)
        self.sent = []

    def recv_kv_status(self, timeout_ms=0):
        return self._messages.pop(0) if self._messages else []

    def send_metadata(self, batch):
        self.sent.append(batch)


def _engine(*, enabled=True, deferred=None, connector=None, outputs=()):
    proc = EngineCore.__new__(EngineCore)
    proc.label = "test-engine"
    proc.kv_transfer_enabled = enabled
    proc._next_idle_kv_drain = 0.0
    proc.scheduler = FakeScheduler(deferred, connector)
    proc.runner_mgr = FakeRunnerMgr(outputs)
    return proc


def _head(pp_size, *, deferred=None, connector=None, outputs=(), messages=()):
    proc = PPEngineCoreProc.__new__(PPEngineCoreProc)
    proc.label = "test-head"
    proc.kv_transfer_enabled = True
    proc.pp_size = pp_size
    proc._pp_kv_aggregator = None
    proc._held_sending = {}
    proc._next_idle_kv_drain = 0.0
    proc.scheduler = FakeScheduler(deferred, connector)
    proc.runner_mgr = FakeRunnerMgr(outputs)
    proc.pp_transport = FakePPTransport(messages)
    return proc


# -- the predicate ---------------------------------------------------------


def test_no_pending_work_when_kv_transfer_is_off():
    proc = _engine(enabled=False, deferred={7: object()})
    assert proc.has_pending_kv_work() is False


def test_deferred_blocks_are_pending_work():
    proc = _engine(deferred={7: object()})
    assert proc.has_pending_kv_work() is True


def test_connector_reports_its_own_pending_work():
    proc = _engine(connector=FakeConnector(pending=True))
    assert proc.has_pending_kv_work() is True


def test_drained_engine_reports_nothing_pending():
    proc = _engine(connector=FakeConnector(pending=False))
    assert proc.has_pending_kv_work() is False


def test_connector_without_the_hook_is_not_pending():
    proc = _engine(connector=object())
    assert proc.has_pending_kv_work() is False


# -- the PP head's extra holding state -------------------------------------


def test_held_send_keeps_the_head_alive():
    proc = _head(2)
    proc._held_sending = {"a": ("a", {"a"})}
    assert proc.has_pending_kv_work() is True


def test_partial_stage_quorum_keeps_the_head_alive():
    proc = _head(2)
    proc._pp_kv_aggregator = PPKVAggregator(2)
    proc._pp_kv_aggregator.ingest(0, KVConnectorOutput(finished_saving={"a"}))
    assert proc.has_pending_kv_work() is True


def test_aggregator_pending_clears_on_quorum():
    agg = PPKVAggregator(2)
    agg.ingest(0, KVConnectorOutput(finished_saving={"a"}))
    assert agg.has_pending() is True
    agg.ingest(1, KVConnectorOutput(finished_saving={"a"}))
    assert agg.has_pending() is False


# -- the drain itself ------------------------------------------------------


def test_idle_drain_releases_the_last_held_send():
    # The regression guard: scheduler queues are empty, so only
    # has_pending_kv_work() can keep the head polling long enough for stage 1
    # to report and free the request's blocks.
    proc = _head(2, messages=[[(1, KVConnectorOutput(finished_saving={"a"}))]])
    proc._pp_kv_aggregator = PPKVAggregator(2)
    proc._pp_kv_aggregator.ingest(0, KVConnectorOutput(finished_saving={"a"}))
    proc._held_sending = {"a": ("a", {"a"})}

    assert proc.has_pending_kv_work() is True
    proc._advance_idle_kv_transfer()

    assert proc.scheduler.released_sending() == {"a"}
    assert proc._held_sending == {}
    assert proc.has_pending_kv_work() is False


def test_idle_drain_ships_metadata_to_every_stage():
    # schedule() returns None once the queues are empty, so the head has to
    # build the metadata itself — and downstream stages must still see it or
    # they never save their layers.
    connector = FakeConnector(pending=True)
    proc = _head(2, connector=connector)
    proc._advance_idle_kv_transfer()

    assert connector.builds == 1
    assert proc.runner_mgr.dispatched == ["process_kvconnector_output"]
    assert len(proc.pp_transport.sent) == 1
    assert list(proc.pp_transport.sent[0].req_ids) == []


def test_idle_drain_skips_dispatch_when_the_connector_has_nothing():
    connector = FakeConnector(pending=True, requests=())
    proc = _head(2, connector=connector)
    proc._advance_idle_kv_transfer()

    assert proc.runner_mgr.dispatched == []
    assert proc.pp_transport.sent == []
    assert proc.runner_mgr.polls == 1  # status is still polled


def test_idle_drain_is_paced():
    # The busy loops never block, so an unpaced drain would fire one worker
    # RPC round per spin.
    proc = _engine(connector=FakeConnector(pending=True))
    proc._advance_idle_kv_transfer()
    proc._advance_idle_kv_transfer()
    assert proc.runner_mgr.polls == 1

    proc._next_idle_kv_drain = 0.0
    proc._advance_idle_kv_transfer()
    assert proc.runner_mgr.polls == 2


# -- the bounded drain at exit ---------------------------------------------


def test_exit_drain_stops_once_everything_reports():
    connector = FakeConnector(pending=True)
    proc = _engine(connector=connector)

    original = connector.has_pending_work

    def clear_after_first_poll():
        pending = original()
        connector.pending = False
        return pending

    connector.has_pending_work = clear_after_first_poll
    proc._drain_kv_work_at_exit()
    assert proc.runner_mgr.polls == 1


def test_exit_drain_gives_up_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(engine_core_mod, "KV_SHUTDOWN_DRAIN_TIMEOUT_S", 0.02)
    proc = _engine(connector=FakeConnector(pending=True))

    t0 = time.monotonic()
    proc._drain_kv_work_at_exit()
    assert time.monotonic() - t0 < 1.0


def test_exit_drain_is_a_noop_without_kv_transfer():
    proc = _engine(enabled=False, deferred={7: object()})
    proc._drain_kv_work_at_exit()
    assert proc.runner_mgr.polls == 0


# -- new state work is dispatched at idle, but NOT while draining at exit ---


def test_idle_drain_publishes_new_state_work():
    # Normal idle: publish new state loads/stores so a lull between batches
    # still makes forward progress on the offload tiers.
    proc = _engine(connector=FakeConnector(pending=True))
    proc._advance_idle_kv_transfer()
    assert proc.scheduler.state_publishes == 2  # both loads and stores


def test_exit_drain_does_not_publish_new_state_work():
    # Shutdown drain must let in-flight transfers finish and report, but must
    # not publish new state work: a fresh store keeps has_pending_kv_work True,
    # so the loop that waits on it would manufacture its own exit condition and
    # never converge. It still builds/processes the meta to flush what is in
    # flight.
    connector = FakeConnector(pending=True)

    original = connector.has_pending_work

    def clear_after_first_poll():
        pending = original()
        connector.pending = False
        return pending

    connector.has_pending_work = clear_after_first_poll
    proc = _engine(connector=connector)
    proc._drain_kv_work_at_exit()

    assert proc.scheduler.state_publishes == 0
    assert connector.builds == 1  # in-flight work still flushed
    assert proc.runner_mgr.dispatched == ["process_kvconnector_output"]
