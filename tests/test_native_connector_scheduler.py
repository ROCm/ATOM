# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""GPU-free unit tests for the native KV connector scheduler side.

Covers the request-routing logic of ``NativeConnectorScheduler``
(do_remote_prefill / do_remote_decode bookkeeping, transfer-id mapping,
metadata snapshot/clear, finish handling) and the ``"native"`` backend
registration in :class:`KVConnectorFactory`. Worker-side VMM transport is
covered by ``tests/test_native_vmm_transfer.py`` instead.
"""

from __future__ import annotations

import types

import pytest

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.native.native_connector import (
    NativeConnectorScheduler,
)


def _config(role: str = "kv_producer", **extra):
    kv_cfg = {"kv_connector": "native", "kv_role": role}
    kv_cfg.update(extra)
    return types.SimpleNamespace(kv_transfer_config=kv_cfg)


def _seq(req_id, params=None, block_ids=(), slot=3):
    return types.SimpleNamespace(
        id=req_id,
        kv_transfer_params=dict(params or {}),
        kv_transfer_params_output=None,
        block_ids=list(block_ids),
        per_req_cache_group=slot,
    )


_REMOTE = {
    "remote_block_ids": [40, 41],
    "remote_host": "10.0.0.2",
    "remote_port": 8004,
    "remote_handshake_port": 6501,
    "remote_engine_id": "engine-d",
    "tp_size": 4,
    "remote_dp_size": 1,
    "remote_dp_rank": 0,
}


# --- factory ---------------------------------------------------------------


def test_factory_registers_native():
    entry = KVConnectorFactory._registry["native"]
    assert entry["worker_class"] == "NativeConnector"
    assert entry["scheduler_class"] == "NativeConnectorScheduler"
    assert entry["worker_module"].endswith("native.native_connector")


def test_factory_creates_native_scheduler():
    sched = KVConnectorFactory.create_connector(_config(), role="scheduler")
    assert isinstance(sched, NativeConnectorScheduler)


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown KV connector backend"):
        KVConnectorFactory.create_connector(
            types.SimpleNamespace(kv_transfer_config={"kv_connector": "nope"}),
            role="scheduler",
        )


# --- get_num_new_matched_tokens --------------------------------------------


def test_matched_tokens_consumer_remote_prefill():
    sched = NativeConnectorScheduler(_config("kv_consumer"))
    seq = _seq("r1", {"do_remote_prefill": True})
    assert sched.get_num_new_matched_tokens(seq) == (0, True)


def test_matched_tokens_consumer_local():
    sched = NativeConnectorScheduler(_config("kv_consumer"))
    assert sched.get_num_new_matched_tokens(_seq("r1")) == (0, False)


def test_matched_tokens_producer_ignores_remote_prefill():
    sched = NativeConnectorScheduler(_config("kv_producer"))
    seq = _seq("r1", {"do_remote_prefill": True})
    assert sched.get_num_new_matched_tokens(seq) == (0, False)


# --- update_state_after_alloc ----------------------------------------------


def test_alloc_remote_prefill_goes_to_recv():
    sched = NativeConnectorScheduler(_config("kv_consumer"))
    seq = _seq(
        "r1",
        {"do_remote_prefill": True, "transfer_id": 77, **_REMOTE},
        block_ids=[5, 6],
        slot=9,
    )
    sched.update_state_after_alloc(seq)

    assert "r1" in sched._reqs_to_recv
    meta = sched._reqs_to_recv["r1"]
    assert meta.local_block_ids == [5, 6]
    assert meta.remote_block_ids == [40, 41]
    assert meta.remote_host == "10.0.0.2"
    assert meta.remote_handshake_port == 6501
    assert meta.tp_size == 4
    assert meta.transfer_id == 77
    assert meta.local_slot_index == 9
    # the one-shot trigger must be consumed so later steps don't re-recv
    assert seq.kv_transfer_params["do_remote_prefill"] is False
    # transfer-id <-> request-id mapping for finish tracking
    assert sched.request_id_to_transfer_id["r1"] == 77
    assert sched.transfer_id_to_request_id[77] == "r1"


def test_alloc_remote_decode_goes_to_save_with_own_req_id():
    sched = NativeConnectorScheduler(_config("kv_producer"))
    seq = _seq(
        "r2",
        {"do_remote_decode": True, "transfer_id": 55, **_REMOTE},
        block_ids=[7],
        slot=4,
    )
    sched.update_state_after_alloc(seq)

    assert "r2" in sched._reqs_to_save
    meta = sched._reqs_to_save["r2"]
    # the consumer later requests the transfer by the producer's OWN request
    # id, not by params["transfer_id"]
    assert meta.transfer_id == "r2"
    assert meta.local_block_ids == [7]
    assert meta.local_slot_index == 4


def test_alloc_plain_request_not_tracked():
    sched = NativeConnectorScheduler(_config("kv_producer"))
    sched.update_state_after_alloc(_seq("r3"))
    assert not sched._reqs_to_save
    assert not sched._reqs_to_recv


# --- build_connector_meta ---------------------------------------------------


def test_build_connector_meta_snapshots_and_clears():
    sched = NativeConnectorScheduler(_config("kv_consumer"))
    sched.update_state_after_alloc(
        _seq("r1", {"do_remote_prefill": True, "transfer_id": 1, **_REMOTE})
    )
    meta = sched.build_connector_meta()
    assert set(meta.reqs_to_recv) == {"r1"}
    assert meta.request_id_to_transfer_id == {"r1": 1}
    # internal queues drained; id map kept for finish correlation
    assert not sched._reqs_to_recv
    assert sched.request_id_to_transfer_id == {"r1": 1}
    meta2 = sched.build_connector_meta()
    assert not meta2.reqs_to_recv and not meta2.reqs_to_save


def test_build_connector_meta_returns_copies():
    sched = NativeConnectorScheduler(_config("kv_consumer"))
    sched.update_state_after_alloc(
        _seq("r1", {"do_remote_prefill": True, "transfer_id": 1, **_REMOTE})
    )
    meta = sched.build_connector_meta()
    meta.reqs_to_recv.clear()
    meta.request_id_to_transfer_id.clear()
    # mutating the snapshot must not corrupt the scheduler's id map
    assert sched.request_id_to_transfer_id == {"r1": 1}


# --- request_finished -------------------------------------------------------


def test_request_finished_producer_emits_output_params():
    sched = NativeConnectorScheduler(_config("kv_producer"))
    seq = _seq("r1", {"transfer_id": 42})
    sched.update_state_after_alloc(seq)
    sched.request_finished(seq)
    assert seq.kv_transfer_params_output == {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "transfer_id": "r1",
    }
    assert "r1" not in sched.request_id_to_transfer_id
    assert 42 not in sched.transfer_id_to_request_id


def test_request_finished_consumer_silent():
    sched = NativeConnectorScheduler(_config("kv_consumer"))
    seq = _seq("r1", {"transfer_id": 42})
    sched.update_state_after_alloc(seq)
    sched.request_finished(seq)
    assert seq.kv_transfer_params_output is None
    assert "r1" not in sched.request_id_to_transfer_id


def test_request_finished_unknown_request_tolerated():
    sched = NativeConnectorScheduler(_config("kv_producer"))
    sched.request_finished(_seq("ghost"))  # must not raise
