# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for KVConnectorScheduler (scheduler-side connector)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from atom.kv_transfer.disaggregation.kv_transfer_engine import (
    KVConnectorScheduler,
    Role,
    convert_virtual_to_physical_pages,
    get_port_offset,
    get_role,
    set_role,
    _RoleManager,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata
from atom.model_engine.sequence import Sequence

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    *,
    kv_role: str = "kv_consumer",
    tp_size: int = 1,
    dp_size: int = 1,
    dp_rank: int = 0,
) -> MagicMock:
    """Return a minimal Config-like object for KVConnectorScheduler."""
    cfg = MagicMock()
    cfg.kv_transfer_config = {"kv_role": kv_role}
    cfg.tensor_parallel_size = tp_size
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.data_parallel_rank = dp_rank
    return cfg


def _make_seq(
    *,
    token_ids: list[int] | None = None,
    kv_transfer_params: dict | None = None,
    block_table: list[int] | None = None,
) -> Sequence:
    """Create a lightweight Sequence for testing."""
    if token_ids is None:
        token_ids = list(range(10))
    seq = Sequence(token_ids, block_size=16, kv_transfer_params=kv_transfer_params)
    if block_table is not None:
        seq.block_table = block_table
    return seq


@pytest.fixture()
def consumer_sched() -> KVConnectorScheduler:
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
def producer_sched() -> KVConnectorScheduler:
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


# ---------------------------------------------------------------------------
# Tests: get_num_new_matched_tokens
# ---------------------------------------------------------------------------


class TestGetNumNewMatchedTokens:
    def test_remote_prefill_returns_token_count(self, consumer_sched):
        seq = _make_seq(
            token_ids=[0] * 100,
            kv_transfer_params={"do_remote_prefill": True},
        )
        num_tokens, needs_load = consumer_sched.get_num_new_matched_tokens(seq)
        assert num_tokens == 100
        assert needs_load is True

    def test_second_call_idempotent(self, consumer_sched):
        seq = _make_seq(
            token_ids=[0] * 50,
            kv_transfer_params={"do_remote_prefill": True},
        )
        consumer_sched.get_num_new_matched_tokens(seq)
        num_tokens, needs_load = consumer_sched.get_num_new_matched_tokens(seq)
        assert num_tokens == 0
        assert needs_load is False

    def test_no_kv_transfer_params(self, consumer_sched):
        seq = _make_seq(kv_transfer_params=None)
        num_tokens, needs_load = consumer_sched.get_num_new_matched_tokens(seq)
        assert num_tokens == 0
        assert needs_load is False

    def test_no_do_remote_prefill(self, consumer_sched):
        seq = _make_seq(kv_transfer_params={"something_else": True})
        num_tokens, needs_load = consumer_sched.get_num_new_matched_tokens(seq)
        assert num_tokens == 0
        assert needs_load is False


# ---------------------------------------------------------------------------
# Tests: update_state_after_alloc
# ---------------------------------------------------------------------------


class TestUpdateStateAfterAlloc:
    def test_consumer_records_transfer_id(self, consumer_sched):
        seq = _make_seq(
            kv_transfer_params={"transfer_id": 42},
            block_table=[0, 1, 2],
        )
        consumer_sched.update_state_after_alloc(seq)
        assert consumer_sched.transfer_id_to_request_id[42] == seq.id
        assert consumer_sched.request_id_to_transfer_id[seq.id] == 42

    def test_consumer_queues_for_recv(self, consumer_sched):
        seq = _make_seq(
            kv_transfer_params={"do_remote_prefill": True},
            block_table=[10, 11],
        )
        consumer_sched.update_state_after_alloc(seq)
        assert seq.id in consumer_sched._reqs_need_recv
        assert seq.kv_transfer_params["do_remote_prefill"] is False

    def test_producer_does_not_queue(self, producer_sched):
        seq = _make_seq(kv_transfer_params={"transfer_id": 7})
        producer_sched.update_state_after_alloc(seq)
        assert len(producer_sched._reqs_need_recv) == 0
        assert 7 not in producer_sched.transfer_id_to_request_id


# ---------------------------------------------------------------------------
# Tests: build_connector_meta
# ---------------------------------------------------------------------------


class TestBuildConnectorMeta:
    def test_empty_queue(self, consumer_sched):
        meta = consumer_sched.build_connector_meta()
        assert isinstance(meta, ConnectorMetadata)
        assert meta.reqs_to_recv == {}

    def test_drains_pending_queue(self, consumer_sched):
        kv_params = {
            "do_remote_prefill": True,
            "remote_block_ids": [10],
            "remote_engine_id": "e1",
            "remote_host": "10.0.0.1",
            "remote_port": 5000,
            "remote_handshake_port": 5001,
        }
        seq = _make_seq(kv_transfer_params=kv_params, block_table=[0, 1])
        consumer_sched.update_state_after_alloc(seq)
        assert len(consumer_sched._reqs_need_recv) == 1

        meta = consumer_sched.build_connector_meta()
        assert seq.id in meta.reqs_to_recv
        assert consumer_sched._reqs_need_recv == {}


# ---------------------------------------------------------------------------
# Tests: request_finished
# ---------------------------------------------------------------------------


class TestRequestFinished:
    def test_producer_populates_output(self, producer_sched):
        seq = _make_seq(block_table=[5, 6, 7])
        producer_sched.request_finished(seq)

        out = seq.kv_transfer_params_output
        assert out is not None
        assert out["remote_block_ids"] == [5, 6, 7]
        assert out["remote_engine_id"] == "None"
        assert out["remote_host"] == "127.0.0.1"
        assert out["do_remote_prefill"] is True

    def test_consumer_cleans_up_mapping(self, consumer_sched):
        seq = _make_seq(
            kv_transfer_params={"transfer_id": 99},
            block_table=[1],
        )
        consumer_sched.update_state_after_alloc(seq)
        assert 99 in consumer_sched.transfer_id_to_request_id

        consumer_sched.request_finished(seq)
        assert 99 not in consumer_sched.transfer_id_to_request_id
        assert seq.id not in consumer_sched.request_id_to_transfer_id

    def test_transfer_id_bidirectional_consistency(self, consumer_sched):
        seq = _make_seq(
            kv_transfer_params={"transfer_id": 77},
            block_table=[2, 3],
        )
        consumer_sched.update_state_after_alloc(seq)
        tid = consumer_sched.request_id_to_transfer_id[seq.id]
        assert consumer_sched.transfer_id_to_request_id[tid] == seq.id


# ---------------------------------------------------------------------------
# Tests: convert_virtual_to_physical_pages
# ---------------------------------------------------------------------------


class TestConvertVirtualToPhysicalPages:
    def test_default_16_to_1(self):
        result = convert_virtual_to_physical_pages([0, 1])
        assert result == list(range(32))

    def test_same_block_size(self):
        result = convert_virtual_to_physical_pages(
            [3], virtual_block_size=1, physical_block_size=1
        )
        assert result == [3]

    def test_custom_ratio(self):
        result = convert_virtual_to_physical_pages(
            [2], virtual_block_size=8, physical_block_size=2
        )
        assert result == [8, 9, 10, 11]

    def test_empty_input(self):
        assert convert_virtual_to_physical_pages([]) == []


# ---------------------------------------------------------------------------
# Tests: get_port_offset
# ---------------------------------------------------------------------------


class TestGetPortOffset:
    def test_formula(self):
        assert get_port_offset(dp_rank=0, tp_rank=0) == 0
        assert get_port_offset(dp_rank=1, tp_rank=0, tp_size=8) == 8
        assert get_port_offset(dp_rank=2, tp_rank=3, tp_size=4) == 11


# ---------------------------------------------------------------------------
# Tests: Role / set_role / get_role
# ---------------------------------------------------------------------------


class TestRoleManager:
    def test_initial_role(self):
        _RoleManager._instance = None  # reset singleton
        assert get_role() == Role.NOT_INITIALIZED

    def test_set_and_get(self):
        _RoleManager._instance = None
        set_role(Role.PRODUCER)
        assert get_role() == Role.PRODUCER

    def test_overwrite(self):
        _RoleManager._instance = None
        set_role(Role.PRODUCER)
        set_role(Role.CONSUMER)
        assert get_role() == Role.CONSUMER
