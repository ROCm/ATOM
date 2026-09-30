# SPDX-License-Identifier: MIT
"""Admission protocol tests: real scheduler, workers, and TP/PP aggregation."""

from unittest.mock import MagicMock

import msgpack
import pytest
from conftest import MockConfig
from test_pd_chunked_transfer import (
    consumer,
    mc,
    producer,
    scheduler_connector,
)

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.chunked_prefill import ChunkedPrefill
from atom.kv_transfer.disaggregation.pd_admission import PDTimeouts
from atom.kv_transfer.disaggregation.pp_kv_aggregator import PPKVAggregator
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    ConnectorMetadata,
    KVConnectorOutput,
)
from atom.model_engine.scheduler import Scheduler


def empty_producer():
    p, _, request, _, _ = producer()
    p._chunked_prefills.clear()
    p._chunked_local_ids.clear()
    return p, request


def ready_message(request, rank=0, count=1, digest=None):
    return dict(
        request,
        consumer_tp_rank=rank,
        consumer_tp_size=count,
        notify_port=100 + rank,
        write_nonce=200 + rank,
        prompt_digest=digest,
    )


def register(p, tid="xfer-a", req_id=7, digest=None):
    meta = ConnectorMetadata()
    meta.pd_admissions[req_id] = {"transfer_id": tid, "prompt_digest": digest}
    p.start_load_kv(meta)


def pd_seq(seq_factory, tid):
    return seq_factory(
        list(range(10)),
        kv_transfer_params={
            "chunked_transfer": True,
            "do_remote_decode": True,
            "transfer_id": tid,
        },
    )


@pytest.mark.parametrize("phase", ["admission", "compute", "transfer"])
@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), "300"])
def test_timeout_config_rejects_invalid_values(phase, value):
    with pytest.raises(ValueError, match=f"{phase}_timeout_s"):
        PDTimeouts.from_config({f"{phase}_timeout_s": value})


def test_timeout_config_has_independent_defaults_and_overrides():
    assert PDTimeouts.from_config({"compute_timeout_s": 450}) == PDTimeouts(
        300, 450, 300
    )


def test_early_ready_requires_all_distinct_consumers_and_registration():
    p, req = empty_producer()
    first, last = ready_message(req, 0, 2), ready_message(req, 1, 2)
    p._record_d_ready(first)
    p._record_d_ready(first)
    assert p.get_finished().is_empty()
    register(p)
    assert p.get_finished().is_empty()
    p._record_d_ready(last)
    assert p.get_finished().connector_completions == {
        ConnectorCompletion("pd_destination_ready", 7, True)
    }
    p._record_d_ready(last)
    assert p.get_finished().is_empty()
    assert not p._chunked_prefills  # Handshake itself never allocates KV.


@pytest.mark.parametrize(
    "bad",
    [
        {"consumer_tp_rank": 2},
        {"consumer_tp_size": 3},
        {"write_nonce": 999},
        {"prompt_digest": "different"},
    ],
)
def test_conflicting_ready_fails_closed(bad):
    p, req = empty_producer()
    register(p)
    msg = ready_message(req, 0, 2)
    p._record_d_ready(msg)
    p._record_d_ready(dict(msg, **bad))
    assert p.get_finished().connector_completions == {
        ConnectorCompletion("pd_destination_ready", 7, False)
    }
    assert "xfer-a" in p._pd_tombstones


def test_cancel_before_registration_cannot_be_revived():
    p, req = empty_producer()
    meta = ConnectorMetadata()
    meta.pd_cancellations["xfer-a"] = "client disconnected"
    p.start_load_kv(meta)
    p._record_d_ready(ready_message(req))
    register(p)
    assert p.get_finished().connector_completions == {
        ConnectorCompletion("pd_destination_ready", 7, False)
    }
    p._pending_chunked_requests.append((req, mc.time.monotonic()))
    p._dispatch_ready_chunked_requests()
    assert req["failure_reason"] == "client disconnected"
    p._notify_transfer_result.assert_called_once_with(req, success=False)


def test_real_scheduler_waits_for_tp_and_pp_quorum(monkeypatch, seq_factory):
    sched = Scheduler(MockConfig(max_num_batched_tokens=4))
    connector = sched.kv_connector = scheduler_connector(monkeypatch, True, 3)
    seq = pd_seq(seq_factory, "xfer-a")
    sched.add(seq)
    batch, _ = sched.schedule()
    assert not batch.req_ids and not seq.block_table
    registration = batch.connector_meta_output
    assert registration.has_work()
    digest = seq.kv_transfer_params["prompt_digest"]
    tp = [KVOutputAggregator(2) for _ in range(3)]
    pp = PPKVAggregator(3)
    for stage in range(3):
        for rank in range(2):
            p, req = empty_producer()
            p.tp_size, p.tp_rank = 2, rank
            p.pp_size, p.pp_rank = 3, stage
            p.start_load_kv(registration)
            p._record_d_ready(ready_message(req, digest=digest))
            outputs = [KVConnectorOutput(), KVConnectorOutput()]
            outputs[rank] = p.get_finished()
            output = pp.ingest(stage, tp[stage].aggregate(outputs))
            connector.process_pd_completions(output)
            assert connector.prefill_admission_ready(seq) is (stage == 2 and rank == 1)
    batch, _ = sched.schedule()
    assert batch.req_ids == [seq.id]
    assert seq.block_table and connector.should_defer_free(seq)


def test_ready_later_request_bypasses_unready_head(monkeypatch, seq_factory):
    sched = Scheduler(MockConfig(max_num_batched_tokens=4))
    connector = sched.kv_connector = scheduler_connector(monkeypatch, True)
    head, ready = pd_seq(seq_factory, "head"), pd_seq(seq_factory, "ready")
    sched.extend([head, ready])
    connector.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion("pd_destination_ready", ready.id, True)
            }
        )
    )
    batch, _ = sched.schedule()
    assert batch.req_ids == [ready.id]
    assert not head.block_table
    assert head in sched.waiting


@pytest.mark.parametrize("is_producer", [True, False])
def test_scheduler_admission_timeout_without_allocation(
    monkeypatch, seq_factory, is_producer
):
    now = [0.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    sched = Scheduler(MockConfig())
    connector = sched.kv_connector = scheduler_connector(monkeypatch, is_producer)
    connector._pd_timeouts = PDTimeouts(admission=10, compute=20, transfer=30)
    seq = pd_seq(seq_factory, "expired")
    seq.kv_transfer_params["do_remote_prefill"] = not is_producer
    sched.add(seq)
    now[0] = 10.0
    batch, _ = sched.schedule()
    assert not seq.block_table
    assert seq in sched.take_rejected()
    assert "phase=admission_wait" in seq.kv_transfer_params["cancel_reason"]
    assert not connector._admission_waiting
    if is_producer:
        assert batch.connector_meta_output.pd_cancellations["expired"]
        assert batch.connector_meta_output.has_work()
        assert not connector.has_pending_work()


def test_worker_compute_deadline_starts_at_full_ready(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    p, req = empty_producer()
    p._pd_timeouts = PDTimeouts(admission=10, compute=20, transfer=1)
    register(p)
    now[0] = 9
    p._record_d_ready(ready_message(req))
    assert p.get_finished().connector_completions
    p._pending_chunked_requests.append((req, 0))
    now[0] = 28
    p._dispatch_ready_chunked_requests()
    assert p._pending_chunked_requests  # Admission time isn't charged to compute.
    now[0] = 29
    p._dispatch_ready_chunked_requests()
    assert not p._pending_chunked_requests
    assert "phase=compute_wait" in req["failure_reason"]
    assert "elapsed_s=20.000" in req["failure_reason"]


def test_compute_failure_cannot_free_incomplete_prefill():
    state = ChunkedPrefill(7, [0], 4, 4, compute_timeout=0.001, transfer_timeout=100)
    with pytest.raises(
        RuntimeError, match="phase=compute_wait chunk.*src_block_offset=0"
    ):
        state.wait_chunk(0)
    state.cancel("phase=compute_wait chunk timed out", producer_done=False)
    assert not state.source_safe()
    state.finish({"first_token_id": 42})
    assert state.source_safe()  # Scheduler final metadata certifies compute exit.
    assert state.cancel_reason == "phase=compute_wait chunk timed out"


def test_transfer_deadline_preserves_first_reason_and_active_reader(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    state = ChunkedPrefill(7, [0], 4, 4, compute_timeout=100, transfer_timeout=2)
    state.acquire(("consumer",), 1)
    state.finish({"first_token_id": 42})
    now[0] = 2
    assert not state.source_safe()
    assert "phase=transfer_wait" in state.cancel_reason
    state.cancel("client disconnected")
    assert "phase=transfer_wait" in state.cancel_reason
    state.release()
    assert state.source_safe()


def test_duplicate_after_worker_timeout_waits_for_original_reader():
    p, state, req, _, _ = producer()
    assert p._acquire_chunked_reader(state, req)
    with p._completed_prefills_lock:
        p._cancel_admission_locked("xfer-a", "phase=transfer_wait timed out")
    p._pending_chunked_requests.append((req.copy(), mc.time.monotonic()))
    p._dispatch_ready_chunked_requests()
    p._notify_transfer_result.assert_not_called()
    assert not state.source_safe()
    state.release()
    state.finish({"first_token_id": 42})
    assert state.source_safe()


def test_failure_reason_traverses_wire_and_decode_logs(caplog):
    p, _, req, _, _ = producer()
    p._send_on_socket = MagicMock()
    req["failure_reason"] = "phase=compute_wait chunk timed out src_block_offset=12"
    mc.MooncakeConnector._notify_transfer_result(p, req, success=False)
    _, (kind, payload) = p._send_on_socket.call_args.args
    assert kind == mc.MSG_WRITE_DONE
    data = msgpack.loads(payload)
    assert data["failure_reason"] == req["failure_reason"]
    assert data["transfer_id"] == "xfer-a"
    d = consumer(1)
    assert d._record_write_done(
        21, 0, 0, 3, success=False, failure_reason=data["failure_reason"]
    )
    assert "phase=compute_wait chunk timed out src_block_offset=12" in caplog.text
    assert d.failed_recving == {21}


def test_no_consumer_tp_rank_releases_source_after_handoff():
    p, req = empty_producer()
    p.tp_size, p.tp_rank = 2, 1
    register(p)
    p._record_d_ready(ready_message(req, count=1))
    p.get_finished()
    meta = ConnectorMetadata()
    meta.add_new_req_to_save(
        7,
        [0],
        {
            "transfer_id": "xfer-a",
            "chunked_transfer": True,
            "num_prompt_tokens": 4,
            "prefill_handoff": {"first_token_id": 42},
        },
    )
    p.start_load_kv(meta)
    assert p.get_finished().connector_completions == {
        ConnectorCompletion("pd_source_safe", 7, True)
    }


def test_external_abort_keeps_idle_control_dispatch_alive(monkeypatch, seq_factory):
    sched = Scheduler(MockConfig())
    connector = sched.kv_connector = scheduler_connector(monkeypatch, True)
    seq = pd_seq(seq_factory, "external-abort")
    sched.add(seq)
    connector.build_connector_meta()
    sched.abort_request(seq.id)
    assert not sched.waiting and not sched.running
    sched.take_rejected()
    assert sched.is_finished()
    assert connector.has_pending_work()
    assert connector.build_connector_meta().pd_cancellations["external-abort"]
    assert not connector.has_pending_work()


def test_unschedulable_prefill_retires_admission_without_waiting(
    monkeypatch, seq_factory
):
    sched = Scheduler(MockConfig(max_model_len=4))
    connector = sched.kv_connector = scheduler_connector(monkeypatch, True)
    seq = pd_seq(seq_factory, "too-large")
    sched.add(seq)
    batch, _ = sched.schedule()
    assert not batch.req_ids and not seq.block_table
    assert seq in sched.take_rejected()
    assert "unschedulable" in seq.leave_reason
    assert not connector._admission_waiting
    assert batch.connector_meta_output.pd_cancellations["too-large"]


def test_repeated_handoff_does_not_extend_transfer_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    state = ChunkedPrefill(7, [0], 4, 4, transfer_timeout=2)
    state.finish({"first_token_id": 42})
    now[0] = 1
    state.finish({"first_token_id": 42})
    now[0] = 2
    assert state.source_safe()
    assert "phase=transfer_wait" in state.cancel_reason


def test_registration_without_any_ready_times_out_on_worker(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    p, _ = empty_producer()
    p._pd_timeouts = PDTimeouts(admission=2)
    register(p)
    now[0] = 2
    assert p.get_finished().connector_completions == {
        ConnectorCompletion("pd_destination_ready", 7, False)
    }
    assert "phase=admission_wait" in p._pd_tombstones["xfer-a"]


def test_ready_received_after_deadline_cannot_rescue_expired_admission(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    p, req = empty_producer()
    p._pd_timeouts = PDTimeouts(admission=2)
    register(p)
    now[0] = 2
    p._record_d_ready(ready_message(req))
    assert p.get_finished().connector_completions == {
        ConnectorCompletion("pd_destination_ready", 7, False)
    }


def test_wrong_write_attempt_cannot_use_ready_reservation():
    p, req = empty_producer()
    register(p)
    ready = ready_message(req)
    p._record_d_ready(ready)
    state = ChunkedPrefill(7, [0], 4, 4)
    with pytest.raises(ValueError, match="write request differs from D-ready"):
        p._acquire_chunked_reader(state, dict(ready, write_nonce=999))
    assert not state.readers
