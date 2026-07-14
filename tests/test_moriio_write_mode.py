# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import threading
from collections import deque
from unittest.mock import MagicMock, patch

import msgspec

from atom.kv_transfer.disaggregation.moriio import moriio_connector as mc
from atom.kv_transfer.disaggregation.moriio.moriio_common import (
    MoRIIOConstants,
    MoRIIOWriteDone,
    MoRIIOWriteRegion,
    MoRIIOWriteRequest,
    TransferMode,
    get_port_offset,
)
from atom.kv_transfer.disaggregation.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    KVConnectorOutput,
    KVTransferRegion,
    ReqMeta,
)
from atom.model_engine.sequence import Sequence


class _FakeWrapper:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.registered: list[tuple[int, int, int]] = []

    def register_local_buffer(self, ptr: int, size: int, device_id: int) -> bytes:
        self.registered.append((ptr, size, device_id))
        return f"{ptr}:{size}:{device_id}".encode()

    def get_agent_metadata(self) -> bytes:
        return b"engine-desc"


def _bare_connector() -> MoRIIOConnector:
    conn = object.__new__(MoRIIOConnector)
    conn.moriio_wrapper = _FakeWrapper()
    conn._staging_cv = threading.Condition(threading.Lock())
    conn._staging_free = []
    conn._fence_lock = threading.Lock()
    conn._blocks_pending_fence = []
    conn.done_sending = set()
    conn.done_recving = set()
    conn.failed_recving = set()
    conn._deferred_write_recvs = {}
    conn._deferred_write_recv_deadlines = {}
    conn._pending_write_recv = set()
    conn._pending_write_recv_blocks = {}
    conn._pending_write_recv_slots = {}
    conn._pending_write_recv_deadlines = {}
    conn._write_scatter_slot = None
    conn._terminal_write_transfers = set()
    conn._terminal_write_transfer_order = deque()
    conn._completed_prefills = {}
    conn._completed_prefill_deadlines = {}
    conn._inflight_write_transfers = set()
    conn._completed_prefills_cv = threading.Condition(threading.Lock())
    conn.transfer_mode = TransferMode.WRITE_PUSH
    conn.is_producer = False
    conn._write_recv_timeout_s = 60.0
    conn._write_prefill_orphan_timeout_s = 60.0
    conn._assert_region_on_device = lambda region, device_id: None
    return conn


def _make_scheduler(*, role: str, transfer_mode: str = "write"):
    cfg = MagicMock()
    cfg.kv_transfer_config = {
        "kv_role": role,
        "transfer_mode": transfer_mode,
        "handshake_port": 6301,
    }
    cfg.tensor_parallel_size = 4
    cfg.parallel_config.data_parallel_size = 2
    cfg.parallel_config.data_parallel_rank = 1
    with (
        patch(
            "atom.kv_transfer.disaggregation.moriio.moriio_connector.get_open_port",
            return_value=9999,
        ),
        patch(
            "atom.kv_transfer.disaggregation.moriio.moriio_connector.get_ip",
            return_value="127.0.0.1",
        ),
    ):
        return MoRIIOConnectorScheduler(cfg)


def _write_req_meta(*, slot: int = 2) -> ReqMeta:
    return ReqMeta(
        local_block_ids=[8, 9],
        remote_block_ids=[100, 101],
        remote_host="10.0.0.1",
        remote_port=7000,
        remote_handshake_port=6301,
        remote_engine_id="prefill",
        tp_size=4,
        remote_dp_size=1,
        remote_dp_rank=0,
        transfer_id=0,
        local_slot_index=slot,
    )


def test_write_protocol_constants_and_transfer_mode():
    assert TransferMode("read") is TransferMode.READ_PULL
    assert TransferMode("write") is TransferMode.WRITE_PUSH
    assert MoRIIOConstants.WRITE_REQUEST == b"write_request"
    assert MoRIIOConstants.WRITE_DONE == b"write_done"


def test_write_request_schema_preserves_req_id_types():
    region = MoRIIOWriteRegion(
        kind="block",
        chunks=[b"desc0"],
        unit_bytes=128,
        units_per_chunk=16,
        total_units=64,
    )
    request = MoRIIOWriteRequest(
        decode_req_id="decode-req",
        transfer_id=0,
        consumer_engine_desc=b"engine-desc",
        consumer_regions=[region],
        dst_block_ids=[1, 2],
        notify_host="127.0.0.1",
        notify_port=6301,
        consumer_tp_size=4,
    )

    encoded = msgspec.msgpack.encode(request)
    decoded = msgspec.msgpack.decode(encoded, type=MoRIIOWriteRequest)

    assert decoded.decode_req_id == "decode-req"
    assert decoded.transfer_id == 0
    assert decoded.consumer_regions[0].kind == "block"


def test_get_port_offset_uses_tp_size_for_dp_tp_grid():
    assert get_port_offset(dp_rank=0, tp_rank=3, tp_size=8) == 3
    assert get_port_offset(dp_rank=1, tp_rank=3, tp_size=8) == 11
    assert get_port_offset(dp_rank=2, tp_rank=1, tp_size=4) == 9


def test_pack_write_region_chunks_on_unit_boundaries(monkeypatch):
    conn = _bare_connector()
    monkeypatch.setattr(mc, "MAX_RDMA_CHUNK_BYTES", 32)

    packed = conn._pack_write_region(
        "block",
        KVTransferRegion(base_addr=1000, total_bytes=100, unit_bytes=10),
        device_id=3,
    )

    assert packed is not None
    assert packed.units_per_chunk == 3
    assert packed.total_units == 10
    assert conn.moriio_wrapper.registered == [
        (1000, 30, 3),
        (1030, 30, 3),
        (1060, 30, 3),
        (1090, 10, 3),
    ]


def test_handle_write_done_success_scatter_fence_and_output():
    conn = _bare_connector()
    scatter = MagicMock()
    conn._write_scatter_slot = scatter
    conn._pending_write_recv.add("r1")
    conn._pending_write_recv_blocks["r1"] = [10, 11]
    conn._pending_write_recv_slots["r1"] = (7, 2)

    conn._handle_write_done(MoRIIOWriteDone(decode_req_id="r1", status="ok"))

    scatter.assert_called_once_with(7, 2)
    assert conn.get_finished_recv_blocks() == [10, 11]
    out = conn.get_finished()
    assert isinstance(out, KVConnectorOutput)
    assert out.finished_recving == {"r1"}
    assert out.failed_recving == set()
    assert conn._staging_free == [2]


def test_handle_write_done_failure_reports_failed_without_scatter():
    conn = _bare_connector()
    scatter = MagicMock()
    conn._write_scatter_slot = scatter
    conn._pending_write_recv.add("r2")
    conn._pending_write_recv_blocks["r2"] = [12]
    conn._pending_write_recv_slots["r2"] = (8, 3)

    conn._handle_write_done(
        MoRIIOWriteDone(decode_req_id="r2", status="failed", reason="boom")
    )

    scatter.assert_not_called()
    assert conn.get_finished_recv_blocks() == []
    out = conn.get_finished()
    assert isinstance(out, KVConnectorOutput)
    assert out.finished_recving == set()
    assert out.failed_recving == {"r2"}
    assert conn._staging_free == [3]


def test_write_request_without_staging_slot_is_deferred_not_pending():
    conn = _bare_connector()
    conn.tp_rank = 0
    conn.tp_size = 4
    conn.local_ip = "127.0.0.1"
    conn.side_channel_port = 6301
    conn._write_has_staging_region = True
    conn._write_local_regions = [
        MoRIIOWriteRegion(
            kind="block",
            chunks=[b"desc0"],
            unit_bytes=128,
            units_per_chunk=16,
            total_units=64,
        )
    ]

    sent = conn._send_write_request_for_req("r-staged", _write_req_meta(slot=3))

    assert sent is False
    assert "r-staged" not in conn._pending_write_recv
    assert conn._staging_free == []


def test_start_write_mode_retries_deferred_recvs_locally():
    conn = _bare_connector()
    conn.tp_rank = 0
    conn.tp_size = 4
    conn._write_has_staging_region = True
    conn._write_local_regions = [
        MoRIIOWriteRegion(
            kind="block",
            chunks=[b"desc0"],
            unit_bytes=128,
            units_per_chunk=16,
            total_units=64,
        )
    ]

    meta = ConnectorMetadata()
    meta.reqs_to_recv["r-deferred"] = _write_req_meta(slot=3)

    conn._start_write_mode(meta)

    assert "r-deferred" in conn._deferred_write_recvs
    assert "r-deferred" in conn._deferred_write_recv_deadlines
    assert "r-deferred" not in conn._pending_write_recv


def test_pending_write_recv_timeout_releases_staging_and_fails():
    conn = _bare_connector()
    conn._pending_write_recv.add("r-timeout")
    conn._pending_write_recv_blocks["r-timeout"] = [10, 11]
    conn._pending_write_recv_slots["r-timeout"] = (7, 2)
    conn._pending_write_recv_deadlines["r-timeout"] = 0.0

    out = conn.get_finished()

    assert isinstance(out, KVConnectorOutput)
    assert out.finished_recving == set()
    assert out.failed_recving == {"r-timeout"}
    assert conn._staging_free == [2]
    assert "r-timeout" not in conn._pending_write_recv


def test_late_write_done_after_timeout_is_ignored():
    conn = _bare_connector()
    conn._pending_write_recv.add("r-late")
    conn._pending_write_recv_blocks["r-late"] = [12]
    conn._pending_write_recv_deadlines["r-late"] = 0.0

    timed_out = conn.get_finished()
    assert timed_out.failed_recving == {"r-late"}

    conn._handle_write_done(MoRIIOWriteDone(decode_req_id="r-late", status="ok"))
    late = conn.get_finished()

    assert late.finished_recving == set()
    assert late.failed_recving == set()
    assert conn.get_finished_recv_blocks() == []


def test_execute_write_task_timeout_marks_transfer_terminal():
    conn = _bare_connector()
    conn.is_producer = True
    conn._wait_for_prefill_data = MagicMock(return_value=None)
    conn._enqueue_write_done = MagicMock()
    request = MoRIIOWriteRequest(
        decode_req_id="decode-1",
        transfer_id=123,
        consumer_engine_desc=b"engine-desc",
        consumer_regions=[],
        dst_block_ids=[],
        notify_host="127.0.0.1",
        notify_port=6301,
        consumer_tp_size=1,
    )

    conn.tp_size = 1
    conn._execute_write_task(mc._WriteTask(request, "remote-engine"))

    assert conn.done_sending == {123}
    assert conn._terminal_write_transfers == {123}
    conn._enqueue_write_done.assert_called_once()


def test_producer_orphaned_completed_prefill_times_out_and_frees_blocks():
    conn = _bare_connector()
    conn.is_producer = True
    conn._completed_prefills[321] = {"block_ids": [1, 2], "slot_index": -1}
    conn._completed_prefill_deadlines[321] = 0.0

    out = conn.get_finished()

    assert isinstance(out, KVConnectorOutput)
    assert out.finished_sending == {321}
    assert conn._completed_prefills == {}
    assert conn._completed_prefill_deadlines == {}
    assert conn._inflight_write_transfers == set()


def test_producer_inflight_prefill_is_not_orphan_swept():
    conn = _bare_connector()
    conn.is_producer = True
    conn._completed_prefills[322] = {"block_ids": [1, 2], "slot_index": -1}
    conn._completed_prefill_deadlines[322] = 0.0
    conn._inflight_write_transfers.add(322)

    out = conn.get_finished()

    assert isinstance(out, KVConnectorOutput)
    assert out.finished_sending == set()
    assert 322 in conn._completed_prefills
    assert 322 in conn._completed_prefill_deadlines
    assert 322 in conn._inflight_write_transfers


def test_scheduler_write_mode_queues_producer_save_and_preserves_zero_transfer_id():
    sched = _make_scheduler(role="kv_producer")
    seq = Sequence(
        [1, 2, 3],
        block_size=16,
        kv_transfer_params={"do_remote_decode": True, "transfer_id": 0},
    )
    seq.block_table = [4, 5]
    seq.per_req_cache_group = 6

    sched.update_state_after_alloc(seq)
    meta = sched.build_connector_meta()

    assert seq.id in meta.reqs_to_save
    req_meta = meta.reqs_to_save[seq.id]
    assert req_meta.local_block_ids == [4, 5]
    assert req_meta.transfer_id == 0
    assert req_meta.local_slot_index == 6


def test_scheduler_write_mode_queues_consumer_recv_slot_index():
    sched = _make_scheduler(role="kv_consumer")
    seq = Sequence(
        [1, 2, 3],
        block_size=16,
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_block_ids": [100, 101],
            "remote_engine_id": "prefill",
            "remote_host": "10.0.0.1",
            "remote_port": 7000,
            "remote_handshake_port": 6301,
            "transfer_id": 0,
        },
    )
    seq.block_table = [8, 9]
    seq.per_req_cache_group = 2

    sched.update_state_after_alloc(seq)
    meta = sched.build_connector_meta()

    assert seq.id in meta.reqs_to_recv
    req_meta = meta.reqs_to_recv[seq.id]
    assert req_meta.local_block_ids == [8, 9]
    assert req_meta.remote_block_ids == [100, 101]
    assert req_meta.transfer_id == 0
    assert req_meta.local_slot_index == 2
