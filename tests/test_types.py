# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for atom.kv_transfer.disaggregation.types dataclasses and ConnectorMetadata."""

import pytest

from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    RemoteAllocInfo,
    RemoteMeta,
    ReqMeta,
)

# ---------------------------------------------------------------------------
# ReqMeta
# ---------------------------------------------------------------------------


class TestReqMeta:
    def test_fields(self):
        rm = ReqMeta(
            local_block_ids=[0, 1],
            remote_block_ids=[10, 11],
            remote_host="10.0.0.1",
            remote_port=5000,
            remote_handshake_port=5001,
            remote_engine_id="engine-0",
            tp_size=8,
            remote_dp_size=2,
        )
        assert rm.local_block_ids == [0, 1]
        assert rm.remote_port == 5000
        assert rm.remote_dp_rank == 0  # default
        assert rm.transfer_id == 0  # default


# ---------------------------------------------------------------------------
# RemoteAllocInfo
# ---------------------------------------------------------------------------


class TestRemoteAllocInfo:
    def test_defaults(self):
        info = RemoteAllocInfo()
        assert info.block_ids == []
        assert info.writes_done == 0
        assert info.decode_dp_rank == 0
        assert info.transfer_offset is None


# ---------------------------------------------------------------------------
# RemoteMeta
# ---------------------------------------------------------------------------


class TestRemoteMeta:
    def test_all_fields(self):
        rm = RemoteMeta(
            block_ids=[1, 2, 3],
            host="10.0.0.2",
            port=9000,
            engine_id="e1",
            request_id="req-42",
        )
        assert rm.request_id == "req-42"
        assert rm.block_ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# ConnectorMetadata
# ---------------------------------------------------------------------------

_SAMPLE_KV_PARAMS = {
    "remote_block_ids": [10, 11, 12],
    "remote_engine_id": "engine-prefill",
    "remote_host": "10.0.0.1",
    "remote_port": 5000,
    "remote_handshake_port": 5001,
    "remote_dp_size": 2,
    "remote_dp_rank": 1,
    "tp_size": 8,
    "transfer_id": 42,
}


class TestConnectorMetadata:
    def test_init_empty(self):
        meta = ConnectorMetadata()
        assert meta.reqs_to_recv == {}
        assert meta.reqs_to_save == {}
        assert meta.reqs_to_send == {}
        assert meta.reqs_in_batch == set()
        assert meta.reqs_not_processed == set()
        assert meta.request_id_to_transfer_id == {}

    def test_add_new_req_to_recv(self):
        meta = ConnectorMetadata()
        meta.add_new_req_to_recv("req-1", [0, 1, 2], _SAMPLE_KV_PARAMS)

        assert "req-1" in meta.reqs_to_recv
        rm: ReqMeta = meta.reqs_to_recv["req-1"]
        assert rm.local_block_ids == [0, 1, 2]
        assert rm.remote_block_ids == [10, 11, 12]
        assert rm.remote_host == "10.0.0.1"
        assert rm.transfer_id == 42

    def test_add_new_req_to_save(self):
        meta = ConnectorMetadata()
        meta.add_new_req_to_save("req-2", [3, 4], _SAMPLE_KV_PARAMS)

        assert "req-2" in meta.reqs_to_save
        rm: ReqMeta = meta.reqs_to_save["req-2"]
        assert rm.local_block_ids == [3, 4]
        assert rm.remote_engine_id == "engine-prefill"

    def test_multiple_reqs_no_clobber(self):
        meta = ConnectorMetadata()
        params_a = {**_SAMPLE_KV_PARAMS, "remote_engine_id": "engine-a"}
        params_b = {**_SAMPLE_KV_PARAMS, "remote_engine_id": "engine-b"}

        meta.add_new_req_to_recv("req-a", [0], params_a)
        meta.add_new_req_to_recv("req-b", [1], params_b)

        assert meta.reqs_to_recv["req-a"].remote_engine_id == "engine-a"
        assert meta.reqs_to_recv["req-b"].remote_engine_id == "engine-b"

    def test_missing_required_param_raises(self):
        meta = ConnectorMetadata()
        incomplete = {"remote_block_ids": [1]}  # missing many required fields
        with pytest.raises(KeyError):
            meta.add_new_req_to_recv("req-x", [0], incomplete)

    def test_defaults_for_optional_params(self):
        minimal = {
            "remote_block_ids": [1],
            "remote_engine_id": "e",
            "remote_host": "h",
            "remote_port": 1,
            "remote_handshake_port": 2,
        }
        meta = ConnectorMetadata()
        meta.add_new_req_to_recv("req-y", [0], minimal)
        rm = meta.reqs_to_recv["req-y"]
        assert rm.remote_dp_size == 1
        assert rm.remote_dp_rank == 0
        assert rm.tp_size == 1
        assert rm.transfer_id == 0
