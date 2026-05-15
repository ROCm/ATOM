# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Advanced tests for the MoRIIO-based KV connector.

Covers:
- MoRIIOWrapper: dispatch_message, send_notify, pop_finished, shutdown
- ZMQ handshake roundtrip between listener and client
- KVConnectorScheduler full lifecycle (multi-request flow)
- MoRIIOConstants protocol values

References vLLM's test_moriio_connector.py patterns.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import msgpack
import pytest
import zmq

from atom.kv_transfer.disaggregation.kv_transfer_engine import (
    KVConnectorScheduler,
    MoRIIOConstants,
    MoRIIOWrapper,
    Role,
    _RoleManager,
    set_role,
)
from atom.model_engine.sequence import Sequence
from atom.utils import get_open_port, make_zmq_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*, kv_role="kv_consumer", tp_size=1, dp_size=1, dp_rank=0):
    cfg = MagicMock()
    cfg.kv_transfer_config = {"kv_role": kv_role}
    cfg.tensor_parallel_size = tp_size
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.data_parallel_rank = dp_rank
    return cfg


def _make_seq(*, token_ids=None, kv_transfer_params=None, block_table=None):
    if token_ids is None:
        token_ids = list(range(10))
    seq = Sequence(token_ids, block_size=16, kv_transfer_params=kv_transfer_params)
    if block_table is not None:
        seq.block_table = block_table
    return seq


_KV_PARAMS = {
    "do_remote_prefill": True,
    "remote_block_ids": [10, 11],
    "remote_engine_id": "engine-p",
    "remote_host": "10.0.0.1",
    "remote_port": 5000,
    "remote_handshake_port": 5001,
    "transfer_id": 100,
}


# ---------------------------------------------------------------------------
# MoRIIOConstants
# ---------------------------------------------------------------------------


class TestMoRIIOConstants:
    def test_get_meta_msg_is_bytes(self):
        assert isinstance(MoRIIOConstants.GET_META_MSG, bytes)

    def test_protocol_values_exist(self):
        assert hasattr(MoRIIOConstants, "POP_DONE_RECV")
        assert hasattr(MoRIIOConstants, "OVER")
        assert hasattr(MoRIIOConstants, "COMPLETION_PREFIX")
        assert hasattr(MoRIIOConstants, "DEFAULT_HANDSHAKE_PORT")
        assert hasattr(MoRIIOConstants, "PING_INTERVAL_SECONDS")


# ---------------------------------------------------------------------------
# MoRIIOWrapper — unit tests (no RDMA, just internal state + ZMQ)
# ---------------------------------------------------------------------------


class TestMoRIIOWrapper:
    def test_init_defaults(self):
        w = MoRIIOWrapper()
        assert w.tp_rank == 0
        assert w.dp_rank == 0
        assert w.moriio_engine is None
        assert w.done_req_ids == []
        assert w.done_write_cache_req_ids == []
        assert w.local_memory_registered is False

    def test_set_moriio_engine_rejects_none(self):
        w = MoRIIOWrapper()
        with pytest.raises(ValueError, match="None"):
            w.set_moriio_engine(None)

    def test_set_moriio_engine_accepts_valid(self):
        w = MoRIIOWrapper()
        engine = MagicMock()
        w.set_moriio_engine(engine)
        assert w.moriio_engine is engine

    def test_dispatch_completion_message(self):
        """String prefixed with 'cmpl' is handled as completion."""
        _RoleManager._instance = None
        set_role(Role.PRODUCER)
        w = MoRIIOWrapper()
        w.done_remote_allocate_req_dict = {}
        msg = f"{MoRIIOConstants.COMPLETION_PREFIX}:req-42".encode("utf-8")
        w._dispatch_message(msg)
        assert f"{MoRIIOConstants.COMPLETION_PREFIX}:req-42" in w.done_req_ids

    def test_dispatch_block_alloc_message(self):
        """Msgpack dict with req_id is handled as block allocation."""
        _RoleManager._instance = None
        set_role(Role.PRODUCER)
        w = MoRIIOWrapper()
        w.done_remote_allocate_req_dict = {}
        data = {"req_id": "r1", "block_notify_list": [100, 101], "decode_rank": 0}
        msg = msgpack.dumps(data)
        w._dispatch_message(msg)
        assert "r1" in w.done_remote_allocate_req_dict
        assert w.done_remote_allocate_req_dict["r1"].block_ids == [100, 101]

    def test_dispatch_unrecognized_raises(self):
        w = MoRIIOWrapper()
        w.done_remote_allocate_req_dict = {}
        with pytest.raises(ValueError, match="Unrecognized"):
            w._dispatch_message(b"some_random_text")

    def test_pop_finished_req_ids(self):
        w = MoRIIOWrapper()
        w.done_req_ids = ["a", "b", "a"]
        result = w.pop_finished_req_ids()
        assert result == {"a", "b"}
        assert w.done_req_ids == []

    def test_pop_finished_write_req_ids(self):
        w = MoRIIOWrapper()
        w.done_write_cache_req_ids = ["x"]
        result = w.pop_finished_write_req_ids()
        assert result == {"x"}
        assert w.done_write_cache_req_ids == []

    def test_shutdown_clears_sockets(self):
        w = MoRIIOWrapper()
        mock_sock = MagicMock()
        w._sockets["tcp://1.2.3.4:5000"] = mock_sock
        w.shutdown()
        mock_sock.close.assert_called_once_with(linger=0)
        assert w._sockets == {}


# ---------------------------------------------------------------------------
# ZMQ handshake roundtrip — tests the handshake_listener <-> client protocol
# ---------------------------------------------------------------------------


class TestZMQHandshake:
    def test_handshake_returns_metadata(self):
        """Simulate the GET_META_MSG handshake and verify metadata exchange."""
        port = get_open_port()
        engine_metadata = b"fake-engine-metadata-bytes"
        layer_meta = {"layer0": [b"meta0"], "layer1": [b"meta1"]}

        metadata_dict = {
            "engine_id": "test-engine",
            "agent_metadata": engine_metadata,
        }
        encoded_metadata = msgpack.dumps(metadata_dict)

        ready_event = threading.Event()

        def listener():
            path = make_zmq_path("tcp", "*", port)
            ctx = zmq.Context()
            sock = ctx.socket(zmq.ROUTER)
            sock.bind(path)
            ready_event.set()
            try:
                parts = sock.recv_multipart()
                identity, msg = parts[0], parts[1]
                assert msg == MoRIIOConstants.GET_META_MSG

                # Phase 1: engine metadata
                sock.send_multipart((identity, b"", encoded_metadata))
                # Phase 2: layer KV cache metadata
                buf = msgpack.dumps(layer_meta)
                sock.send_multipart((identity, b"", buf))
            finally:
                sock.close(linger=0)
                ctx.term()

        t = threading.Thread(target=listener, daemon=True)
        t.start()
        ready_event.wait(timeout=5)

        # Client side
        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        path = make_zmq_path("tcp", "127.0.0.1", port)
        sock.connect(path)
        try:
            sock.send(MoRIIOConstants.GET_META_MSG)

            # Phase 1: engine metadata
            frame1 = sock.recv_multipart()
            assert len(frame1) == 2
            assert frame1[0] == b""
            decoded = msgpack.loads(frame1[1])
            key = "engine_id" if "engine_id" in decoded else b"engine_id"
            assert decoded[key] == "test-engine" or decoded[key] == b"test-engine"

            # Phase 2: layer KV cache metadata
            frame2 = sock.recv_multipart()
            assert len(frame2) == 2
            assert frame2[0] == b""
            received_layer_meta = msgpack.loads(frame2[1])
            expected_keys = {"layer0", "layer1"}
            actual_keys = {
                k if isinstance(k, str) else k.decode()
                for k in received_layer_meta.keys()
            }
            assert actual_keys == expected_keys
        finally:
            sock.close(linger=0)
            ctx.term()
            t.join(timeout=2)

    def test_pop_done_recv_roundtrip(self):
        """POP_DONE_RECV message should be received by the handshake listener."""
        port = get_open_port()
        received_ids: list[int] = []
        ready_event = threading.Event()

        def listener():
            ctx = zmq.Context()
            path = make_zmq_path("tcp", "*", port)
            sock = ctx.socket(zmq.ROUTER)
            sock.bind(path)
            ready_event.set()
            try:
                for _ in range(2):
                    parts = sock.recv_multipart()
                    if parts[1] == MoRIIOConstants.POP_DONE_RECV:
                        received_ids.append(int(parts[2]))
            finally:
                sock.close(linger=0)
                ctx.term()

        t = threading.Thread(target=listener, daemon=True)
        t.start()
        ready_event.wait(timeout=5)

        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.connect(make_zmq_path("tcp", "127.0.0.1", port))
        try:
            sock.send_multipart([MoRIIOConstants.POP_DONE_RECV, b"42"])
            sock.send_multipart([MoRIIOConstants.POP_DONE_RECV, b"99"])
            time.sleep(0.3)
        finally:
            sock.close(linger=0)
            ctx.term()
            t.join(timeout=2)

        assert sorted(received_ids) == [42, 99]

    def test_send_notify_via_wrapper(self):
        """MoRIIOWrapper.send_notify sends POP_DONE_RECV to the remote side."""
        port = get_open_port()
        received = []
        ready_event = threading.Event()

        def listener():
            ctx = zmq.Context()
            sock = ctx.socket(zmq.ROUTER)
            sock.bind(make_zmq_path("tcp", "*", port))
            ready_event.set()
            try:
                for _ in range(2):
                    parts = sock.recv_multipart()
                    if parts[1] == MoRIIOConstants.POP_DONE_RECV:
                        received.append(parts[2].decode())
            finally:
                sock.close(linger=0)
                ctx.term()

        t = threading.Thread(target=listener, daemon=True)
        t.start()
        ready_event.wait(timeout=5)

        wrapper = MoRIIOWrapper()
        try:
            wrapper.send_notify(["req-1", "req-2"], "127.0.0.1", port)
            time.sleep(0.3)
        finally:
            wrapper.shutdown()
            t.join(timeout=2)

        assert sorted(received) == ["req-1", "req-2"]


# ---------------------------------------------------------------------------
# KVConnectorScheduler — full lifecycle integration
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    """End-to-end scheduler-side connector flow: arrive → alloc → meta → finish."""

    @pytest.fixture()
    def sched(self):
        with (
            patch(
                "atom.kv_transfer.disaggregation.kv_transfer_engine.get_open_port",
                return_value=9999,
            ),
            patch(
                "atom.kv_transfer.disaggregation.kv_transfer_engine.get_ip",
                return_value="127.0.0.1",
            ),
        ):
            return KVConnectorScheduler(_make_config(kv_role="kv_consumer"))

    @pytest.fixture()
    def producer(self):
        with (
            patch(
                "atom.kv_transfer.disaggregation.kv_transfer_engine.get_open_port",
                return_value=8888,
            ),
            patch(
                "atom.kv_transfer.disaggregation.kv_transfer_engine.get_ip",
                return_value="127.0.0.1",
            ),
        ):
            return KVConnectorScheduler(_make_config(kv_role="kv_producer"))

    def test_consumer_full_flow(self, sched):
        """Request → detect → alloc → build_meta → finish → cleanup."""
        seq = _make_seq(
            token_ids=[0] * 64,
            kv_transfer_params={**_KV_PARAMS},
            block_table=[0, 1, 2, 3],
        )

        # Step 1: detect remote prefill
        num_tokens, needs_load = sched.get_num_new_matched_tokens(seq)
        assert num_tokens == 64
        assert needs_load is True

        # Step 2: allocate → records mapping, queues for recv
        sched.update_state_after_alloc(seq)
        assert seq.id in sched.request_id_to_transfer_id
        assert sched.request_id_to_transfer_id[seq.id] == 100
        assert seq.kv_transfer_params["do_remote_prefill"] is False

        # Step 3: build connector metadata
        meta = sched.build_connector_meta()
        assert seq.id in meta.reqs_to_recv
        assert meta.reqs_to_recv[seq.id].remote_block_ids == [10, 11]
        assert meta.request_id_to_transfer_id[seq.id] == 100
        assert sched._reqs_need_recv == {}

        # Step 4: request finishes
        sched.request_finished(seq)
        assert seq.kv_transfer_params_output is not None
        assert 100 not in sched.transfer_id_to_request_id
        assert seq.id not in sched.request_id_to_transfer_id

    def test_producer_full_flow(self, producer):
        """Producer side: allocate does NOT queue; finish populates output."""
        seq = _make_seq(
            token_ids=[0] * 32,
            kv_transfer_params={"transfer_id": 50},
            block_table=[10, 11, 12],
        )

        producer.update_state_after_alloc(seq)
        assert len(producer._reqs_need_recv) == 0
        assert 50 not in producer.transfer_id_to_request_id

        producer.request_finished(seq)
        out = seq.kv_transfer_params_output
        assert out["remote_block_ids"] == [10, 11, 12]
        assert out["remote_host"] == "127.0.0.1"
        assert out["remote_port"] == 8888
        assert out["transfer_id"] == seq.id

    def test_multiple_requests_concurrent(self, sched):
        """Multiple requests go through the pipeline without interfering."""
        seqs = []
        for i in range(3):
            params = {**_KV_PARAMS, "transfer_id": 200 + i}
            seq = _make_seq(
                token_ids=[0] * (10 + i * 10),
                kv_transfer_params=params,
                block_table=list(range(i * 4, (i + 1) * 4)),
            )
            seqs.append(seq)

        for seq in seqs:
            n, load = sched.get_num_new_matched_tokens(seq)
            assert load is True

        for seq in seqs:
            sched.update_state_after_alloc(seq)

        assert len(sched._reqs_need_recv) == 3

        meta = sched.build_connector_meta()
        assert len(meta.reqs_to_recv) == 3
        assert sched._reqs_need_recv == {}

        for seq in seqs:
            sched.request_finished(seq)
            assert seq.kv_transfer_params_output is not None

        assert len(sched.transfer_id_to_request_id) == 0
        assert len(sched.request_id_to_transfer_id) == 0

    def test_idempotent_detection(self, sched):
        """get_num_new_matched_tokens is idempotent after first tagging."""
        seq = _make_seq(
            token_ids=[0] * 20,
            kv_transfer_params={**_KV_PARAMS},
        )
        n1, l1 = sched.get_num_new_matched_tokens(seq)
        n2, l2 = sched.get_num_new_matched_tokens(seq)
        n3, l3 = sched.get_num_new_matched_tokens(seq)
        assert (n1, l1) == (20, True)
        assert (n2, l2) == (0, False)
        assert (n3, l3) == (0, False)

    def test_build_meta_empty_twice(self, sched):
        """Calling build_connector_meta on empty queue doesn't crash."""
        m1 = sched.build_connector_meta()
        m2 = sched.build_connector_meta()
        assert m1.reqs_to_recv == {}
        assert m2.reqs_to_recv == {}
