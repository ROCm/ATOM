# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.tilert.connector import (
    TileRTConnector,
    TileRTConnectorScheduler,
    TileRTMetadata,
    require_producer,
)


def config():
    return SimpleNamespace(
        kv_transfer_config={"kv_connector": "tilert", "kv_role": "kv_producer"},
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
    )


def sequence():
    return SimpleNamespace(
        id="req-1",
        kv_transfer_params={"tilert_host": "decode", "rid": "external-1"},
        prompt_token_ids=[1, 2, 3],
        block_table=[4],
        state_slot=2,
        output_tokens=[42],
        leave_reason="length",
    )


def test_metadata_retains_source_through_cancel_and_tp_completion(monkeypatch):
    # The scheduler needs only TileRT's request-id helper, not its GPU backend.
    import sys

    monkeypatch.setitem(
        sys.modules, "tilert.pd_vllm.wire", SimpleNamespace(derive_rid=str)
    )
    scheduler = TileRTConnectorScheduler(config())
    seq = sequence()
    scheduler.request_finished(seq)
    assert seq.kv_transfer_params_output == {
        "rid": "external-1", "first_token_id": 42
    }
    assert scheduler.should_defer_free(seq)
    metadata = scheduler.build_connector_meta()
    assert metadata.has_work()
    assert metadata.requests[0].blocks == [4]
    assert metadata.requests[0].slot == 2
    assert metadata.requests[0].rid == "external-1"
    seq.block_table.append(5)
    assert metadata.requests[0].blocks == [4]
    scheduler.abort_send(seq)
    assert scheduler.should_defer_free(seq)
    scheduler.send_finished(seq.id)
    assert not scheduler.should_defer_free(seq)
    assert not scheduler.build_connector_meta().has_work()


def test_aborted_before_send_does_not_claim(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules, "tilert.pd_vllm.wire", SimpleNamespace(derive_rid=str)
    )
    scheduler = TileRTConnectorScheduler(config())
    seq = sequence()
    seq.leave_reason = "aborted"
    scheduler.request_finished(seq)
    assert not scheduler.should_defer_free(seq)
    assert not scheduler.build_connector_meta().has_work()


def test_failed_write_does_not_report_source_safe():
    worker = object.__new__(TileRTConnector)
    worker.rank, worker.finished = 0, set()
    metadata = TileRTMetadata()
    metadata.requests = [SimpleNamespace(req_id="req-1")]

    def fail(req):
        raise RuntimeError("RDMA failed")

    worker._send = fail
    with pytest.raises(RuntimeError, match="RDMA failed"):
        worker.start_load_kv(metadata)
    assert worker.get_finished() == (set(), set())


def test_successful_send_reports_completion_once():
    worker = object.__new__(TileRTConnector)
    worker.rank, worker.finished = 0, set()
    metadata = TileRTMetadata()
    metadata.requests = [SimpleNamespace(req_id="req-1")]
    calls = []
    worker._send = lambda req: calls.append(req.req_id)
    worker.start_load_kv(metadata)
    assert calls == ["req-1"]
    assert worker.get_finished() == ({"req-1"}, set())
    assert worker.get_finished() == (set(), set())


def test_external_decoder_cannot_be_instantiated_as_atom_consumer():
    cfg = config()
    cfg.kv_transfer_config["kv_role"] = "kv_consumer"
    with pytest.raises(ValueError, match="external decoder"):
        require_producer(cfg)
