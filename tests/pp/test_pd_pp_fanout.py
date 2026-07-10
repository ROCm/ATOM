# SPDX-License-Identifier: MIT
# CPP+PD J2 — consumer per-stage fanout + write-done completion counting.

import threading
from unittest.mock import MagicMock

import pytest

from atom.kv_transfer.disaggregation.port_offset import side_channel_port_offset
from atom.kv_transfer.disaggregation.types import ConnectorMetadata


def test_consumer_targets_every_producer_stage_port():
    # The ports a consumer computes for stages 0..pp-1 must equal the ports each
    # producer stage binds — otherwise a stage never receives its write_request.
    base, remote_dp_rank, remote_tp_rank = 6301, 0, 0
    remote_pp_size, remote_dp_size, remote_tp_size = 4, 1, 1

    consumer_ports = {
        base
        + side_channel_port_offset(
            remote_dp_rank,
            remote_tp_rank,
            remote_tp_size,
            stage,
            remote_pp_size,
            remote_dp_size,
        )
        for stage in range(remote_pp_size)
    }
    # Each producer stage binds with its own pp_rank and the same topology.
    producer_ports = {
        base
        + side_channel_port_offset(
            remote_dp_rank,
            remote_tp_rank,
            remote_tp_size,
            pp_rank,
            remote_pp_size,
            remote_dp_size,
        )
        for pp_rank in range(remote_pp_size)
    }
    assert consumer_ports == producer_ports
    assert len(consumer_ports) == remote_pp_size  # distinct per stage


# --- Behavioural tests need the worker connector (torch/aiter); CI/GPU only.
#     importorskip lives inside the helper so the pure test above still runs. ---


def _make_consumer(pp_size):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    c = object.__new__(mc.MooncakeConnector)
    c.is_producer = False
    c.tp_rank = 0
    c.tp_size = 1
    c.local_ip = "10.0.0.9"
    c.rpc_port = 5555
    c._notification_port = 6000
    c.num_blocks = 8
    c._has_slot_regions = False
    c.kv_caches_base_addr = [0x1000, 0x2000]
    c._per_block_bytes_list = [128, 128]
    c._notify_sockets = {}
    c._notify_sockets_lock = threading.Lock()
    c._completion_lock = threading.Lock()
    c._fence_lock = threading.Lock()
    c._pending_recv = set()
    c._pending_recv_blocks = {}
    c._pending_recv_slots = {}
    c._pending_recv_expected = {}
    c._release_targets = {}
    c._release_count = {}
    c._completed_prefills = {}
    c._completed_prefills_lock = threading.Lock()
    c.done_sending = set()
    c.pp_size = pp_size
    c._blocks_pending_fence = []
    c.done_recving = set()
    c._scatter_slot = None
    c.zmq_context = MagicMock()
    c.zmq_context.socket.return_value = MagicMock()
    return c


def _meta(pp_size):
    return ConnectorMetadata._build_req_meta(
        req_id="r0",
        local_block_ids=[10, 11, 12],
        kv_transfer_params={
            "remote_block_ids": [20, 21, 22],
            "remote_host": "10.0.0.1",
            "remote_handshake_port": 6301,
            "tp_size": 1,
            "remote_dp_size": 1,
            "remote_dp_rank": 0,
            "remote_pp_size": pp_size,
            "transfer_id": 7,
        },
    )


def test_fanout_sends_one_request_per_stage_with_src_blocks():
    import msgpack

    c = _make_consumer(4)
    meta = ConnectorMetadata()
    meta.request_id_to_transfer_id = {}
    meta.reqs_to_recv = {"r0": _meta(4)}

    c.start_load_kv(meta)

    # One socket per distinct stage port, each sent exactly one write_request.
    assert len(c._notify_sockets) == 4
    payloads = []
    for sock in c._notify_sockets.values():
        assert sock.send_multipart.call_count == 1
        _mtype, body = sock.send_multipart.call_args[0][0]
        payloads.append(msgpack.loads(body))
    for p in payloads:
        assert p["src_block_ids"] == [20, 21, 22]
        assert p["dst_block_ids"] == [10, 11, 12]
    assert c._pending_recv_expected["r0"] == 4
    assert "r0" in c._pending_recv


def test_completion_requires_all_stage_write_dones():
    c = _make_consumer(4)
    c._pending_recv_expected["r0"] = 4
    c._pending_recv.add("r0")
    c._pending_recv_blocks["r0"] = [10, 11, 12]

    assert c._record_write_done("r0") is False  # stage 1
    assert c._record_write_done("r0") is False  # stage 2
    assert c._record_write_done("r0") is False  # stage 3
    assert c.done_recving == set()

    assert c._record_write_done("r0") is True  # stage 4 → complete
    assert "r0" in c.done_recving
    assert "r0" not in c._pending_recv
    assert c._blocks_pending_fence == [10, 11, 12]


def test_release_frees_only_after_all_decode_ranks():
    # stage-0 must not mark done_sending until it has one release per decode rank
    # (all stage×rank writes complete), so it can't reuse the shared page early.
    c = _make_consumer(4)  # reuse helper to get a connector instance
    c.done_sending = set()
    c._release_count = {}
    # 3 decode ranks (consumer_tp_size=3) → need 3 releases for transfer_id=7.
    c._record_release(7, 3)
    assert c.done_sending == set()
    c._record_release(7, 3)
    assert c.done_sending == set()
    c._record_release(7, 3)
    assert 7 in c.done_sending  # all ranks released → page freed


def test_completion_pp1_single_write_done():
    c = _make_consumer(1)
    c._pending_recv_expected["r0"] = 1
    c._pending_recv.add("r0")
    c._pending_recv_blocks["r0"] = [1]
    assert c._record_write_done("r0") is True
    assert "r0" in c.done_recving


def test_duplicate_write_done_ignored():
    c = _make_consumer(2)
    c._pending_recv_expected["r0"] = 2
    c._pending_recv.add("r0")
    c._pending_recv_blocks["r0"] = [1]
    assert c._record_write_done("r0") is False  # stage 1
    assert c._record_write_done("r0") is True  # stage 2 → complete
    c.done_recving.clear()  # simulate get_finished() draining it
    # A late/duplicate write-done for the same req must not re-finalize.
    assert c._record_write_done("r0") is False
    assert c.done_recving == set()
