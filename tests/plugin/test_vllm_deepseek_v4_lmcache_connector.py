# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    SaveOperationId,
)
from atom.plugin.vllm import deepseek_v4_bridge as bridge
from atom.plugin.vllm.deepseek_v4_lmcache_connector import (
    DSV4LMCacheConnector,
    DSV4WorkerMeta,
    _SequenceView,
    _atom_config,
)


def _vllm_config(extra: dict | None = None):
    return SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "LMCacheConnectorV1",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra or {},
        },
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
        ),
        cache_config=SimpleNamespace(block_size=128, cache_dtype="fp8"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            model="deepseek-v4",
        ),
        speculative_config=None,
    )


def test_atom_config_enables_prompt_boundary_slot_checkpoints_by_default():
    config = _atom_config(_vllm_config())

    assert config.kv_transfer_config["kv_connector"] == "lmcache_offload"
    assert config.kv_transfer_config["kv_role"] == "offload"
    assert config.state_checkpoint_interval_tokens == -1


def test_atom_config_preserves_explicit_slot_checkpoint_interval():
    config = _atom_config(
        _vllm_config({"state_checkpoint_interval_tokens": 16384})
    )

    assert config.state_checkpoint_interval_tokens == 16384


def test_worker_meta_preserves_rank_order_for_atom_tp_aggregation():
    operation = SaveOperationId("request", 7)
    outputs = [
        KVConnectorOutput(finished_saving={operation}),
        KVConnectorOutput(finished_saving={operation}),
    ]

    aggregated_meta = DSV4WorkerMeta([outputs[0]]).aggregate(
        DSV4WorkerMeta([outputs[1]])
    )
    aggregated = KVOutputAggregator(world_size=2).aggregate(
        aggregated_meta.worker_outputs
    )

    assert aggregated_meta.worker_outputs == outputs
    assert aggregated.finished_saving == {operation}


def test_reserved_slot_is_not_reset_when_request_first_reaches_model(monkeypatch):
    allocator = bridge._V4StateSlotAllocator(4)
    monkeypatch.setattr(
        bridge, "_V4_SLOT_ALLOCATORS_BY_PROXY_PTR", {1234: allocator}
    )

    bridge.reserve_deepseek_v4_state_slot(1234, "restored-request", 2)
    slots, reset = allocator.assign(["restored-request"], [8192])

    assert slots.tolist() == [2]
    assert reset == set()


def test_reserving_slot_requires_bound_model_allocator(monkeypatch):
    monkeypatch.setattr(bridge, "_V4_SLOT_ALLOCATORS_BY_PROXY_PTR", {})

    with pytest.raises(RuntimeError, match="not bound"):
        bridge.reserve_deepseek_v4_state_slot(1234, "request", 0)


def test_scheduler_sync_does_not_publish_tokens_before_forward():
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=list(range(16384)),
        all_token_ids=list(range(16384)),
        num_computed_tokens=8192,
    )
    connector = object.__new__(DSV4LMCacheConnector)
    connector._sequences = {"request": _SequenceView(request, 0)}
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request"],
            resumed_req_ids=set(),
            new_block_ids=[([64],)],
            num_computed_tokens=[8192],
        ),
        num_scheduled_tokens={"request": 8192},
    )

    connector._sync_from_scheduler_output(scheduler_output)

    seq = connector._sequences["request"]
    assert seq.num_cached_tokens == 8192
    assert seq.block_table == [64]
