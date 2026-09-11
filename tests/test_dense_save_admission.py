# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Admission protects foreground capacity without weakening source ownership."""

import time
from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import ConnectorCompletion, SaveOperationId
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.connector import (
    DENSE_PAGE_RETIRED_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.dense.save_admission import build_save_budget
from atom.model_engine.scheduler import Scheduler


def scheduler(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(
        offcfg, "build_lmcache_config", lambda _: SimpleNamespace(chunk_size=8)
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_: object())
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "kv_producer"},
        kv_cache_block_size=4,
        tensor_parallel_size=1,
        num_kvcache_blocks=1000,
        kv_cache_block_bytes=128,
    )
    return DenseOffloadScheduler(config)


def seq(req_id, tokens=8, first=0):
    return SimpleNamespace(
        id=req_id,
        num_cached_tokens=tokens,
        num_prompt_tokens=tokens,
        token_ids=list(range(tokens)),
        block_table=list(range(first, first + tokens // 4)),
    )


def outcome(s, op, ok=True):
    s.connector_completion(ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, op, ok))


def retired(s, op):
    s.connector_completion(ConnectorCompletion(DENSE_PAGE_RETIRED_CHANNEL, op, True))


def test_ten_finishes_admit_two_and_never_lease_the_other_eight(monkeypatch):
    s = scheduler(monkeypatch)
    requests = [seq(i, first=2 * i) for i in range(10)]
    for request in requests:
        s.update_state_after_alloc(request)
        s.request_finished(request)
        s.activate_block_leases(request, s.protected_block_ids(request))
    metadata = s.build_connector_meta()
    assert [r.req_id for r in metadata.requests] == [0, 1]
    assert [len(s.protected_block_ids(q)) for q in requests] == [2, 2] + [0] * 8
    stats = s.get_statistics()
    assert stats["save_ops_admitted"] == 2
    assert stats["save_ops_dropped"] == 8
    assert stats["save_tokens_dropped"] == 64
    assert stats["source_blocks_reserved"] == stats["source_blocks_leased"] == 4
    assert stats["save_pending_bytes"] == 512
    for request in metadata.requests:
        outcome(s, request.save_operation)
        retired(s, request.save_operation)
    assert s.take_source_safe_releases() == [frozenset({0, 1}), frozenset({2, 3})]
    assert not s.has_pending_work()


def test_source_budget_counts_live_requests_before_any_lease(monkeypatch):
    s = scheduler(monkeypatch, OFFLOAD_MAX_RESERVED_SOURCE_BLOCKS=100)
    for i in range(2):
        s.update_state_after_alloc(seq(i, tokens=240, first=i * 60))
    metadata = s.build_connector_meta()
    assert len(metadata.requests) == 1
    assert s.get_statistics()["source_blocks_reserved"] == 60
    assert s.get_statistics()["source_blocks_leased"] == 0
    assert s.get_statistics()["save_tokens_dropped"] == 240


@pytest.mark.parametrize("first", ["outcome", "retired"])
@pytest.mark.parametrize("ok", [False, True])
def test_credit_requires_both_outcome_and_source_retirement(monkeypatch, first, ok):
    s = scheduler(monkeypatch, OFFLOAD_MAX_PENDING_SAVES=1)
    q = seq(1)
    s.update_state_after_alloc(q)
    op = s.build_connector_meta().requests[0].save_operation
    if first == "outcome":
        outcome(s, op, ok)
        assert s.get_statistics()["source_blocks_reserved"] == 2
    else:
        retired(s, op)
        assert s.get_statistics()["source_blocks_reserved"] == 0
    assert s.get_statistics()["save_ops_unretired"] == 1
    assert s.get_statistics()["save_pending_bytes"] == 256
    assert s.total_save_requests == 0
    if first == "outcome":
        retired(s, op)
    else:
        outcome(s, op, ok)
    assert s.get_statistics()["save_ops_unretired"] == 0
    assert s.get_statistics()["save_pending_bytes"] == 0
    assert s.total_save_requests == int(ok)
    # Duplicate reports cannot return a second operation credit.
    retired(s, op)
    outcome(s, op, ok)
    assert s.total_save_requests == int(ok)


def test_final_tail_is_dropped_while_a_prior_save_owns_sources(monkeypatch):
    s = scheduler(monkeypatch)
    q = seq(1, tokens=32)
    q.num_cached_tokens = 8
    s.update_state_after_alloc(q)
    op = s.build_connector_meta().requests[0].save_operation
    q.num_cached_tokens = 32
    s.request_finished(q)
    s.request_finished(q)  # streaming and teardown both invoke this hook
    assert s.protected_block_ids(q) == frozenset({0, 1})
    assert s.get_statistics()["save_tokens_dropped"] == 24
    q.block_table.clear()
    outcome(s, op)
    retired(s, op)
    assert s.build_connector_meta().requests == []
    assert not s.has_pending_work()


def test_drop_survives_realloc_and_failed_load_but_not_request_id_reuse(monkeypatch):
    s = scheduler(monkeypatch, OFFLOAD_MAX_PENDING_SAVES=1)
    holding = seq(0)
    old = seq(1, tokens=16, first=10)
    old.num_cached_tokens = 8
    for q in (holding, old):
        s.update_state_after_alloc(q)
    op = s.build_connector_meta().requests[0].save_operation
    outcome(s, op)
    retired(s, op)
    old.num_cached_tokens = 16
    s.update_state_after_alloc(old)
    s._load_save_floors["1"] = 0
    assert s.load_failed(1)
    assert s.build_connector_meta().requests == []
    assert s.get_statistics()["save_tokens_dropped"] == 16
    new = seq(1, tokens=8, first=20)
    s.update_state_after_alloc(new)
    assert s.build_connector_meta().requests[0].block_ids == [20, 21]


def test_reused_id_admits_a_new_lifecycle_without_releasing_the_old_one(monkeypatch):
    s = scheduler(monkeypatch)
    old, new = seq(1), seq(1, first=10)
    s.update_state_after_alloc(old)
    old_op = s.build_connector_meta().requests[0].save_operation
    s.request_finished(old)
    s.activate_block_leases(old, s.protected_block_ids(old))
    outcome(s, old_op, False)
    s.update_state_after_alloc(new)
    new_op = s.build_connector_meta().requests[0].save_operation
    retired(s, old_op)
    assert s._save_inflight["1"] == new_op
    assert s.take_source_safe_releases() == [frozenset({0, 1})]
    assert s.protected_block_ids(new) == frozenset({10, 11})
    assert s._save_tracker["1"][0] is new


def test_age_requests_cancel_once_and_never_frees_sources(monkeypatch):
    s = scheduler(monkeypatch)
    q = seq(1)
    s.update_state_after_alloc(q)
    op = s.build_connector_meta().requests[0].save_operation
    s.request_finished(q)
    s.activate_block_leases(q, s.protected_block_ids(q))
    s._save_budget.operations[op].created_at -= 10
    metadata = s.build_connector_meta()
    assert metadata.requests == []
    assert metadata.cancel_save_operations == [op]
    assert metadata.has_work()
    assert s.build_connector_meta().cancel_save_operations == []
    assert s.take_source_safe_releases() == []
    assert s.get_statistics()["save_cancel_requests"] == 1
    assert s.get_statistics()["source_blocks_leased"] == 2
    outcome(s, op, False)
    assert s.take_source_safe_releases() == []
    retired(s, op)
    assert s.take_source_safe_releases() == [frozenset({0, 1})]


def test_engine_timeout_cannot_bypass_dense_retirement(monkeypatch):
    s = scheduler(monkeypatch)
    q = seq(1)
    s.update_state_after_alloc(q)
    s.build_connector_meta()
    q._deferred_save_at = time.monotonic() - 1000
    engine = Scheduler.__new__(Scheduler)
    engine.kv_connector = s
    engine.deferred_free_blocks = {q.id: q}
    engine._next_save_reconcile_at = 0
    engine._abandoned_saves = 0
    engine.block_manager = SimpleNamespace(
        deallocate=lambda _: pytest.fail("unsafe block free")
    )
    assert engine._reconcile_stalled_deferred_saves() == 0
    assert engine.deferred_free_blocks == {q.id: q}
    assert s.get_statistics()["save_cancel_requests"] == 1


def test_byte_budget_remains_after_source_safe_until_backend_retires(monkeypatch):
    s = scheduler(monkeypatch, OFFLOAD_MAX_PENDING_SAVE_BYTES=256)
    first = seq(1)
    s.update_state_after_alloc(first)
    op = s.build_connector_meta().requests[0].save_operation
    retired(s, op)
    s.update_state_after_alloc(seq(2, first=10))
    assert s.build_connector_meta().requests == []
    assert s.get_statistics()["source_blocks_reserved"] == 0
    assert s.get_statistics()["save_pending_bytes"] == 256
    assert s.get_statistics()["save_ops_dropped"] == 1


def test_geometry_sets_byte_budget_and_control_preserves_accounting(monkeypatch):
    s = scheduler(monkeypatch)
    assert s._save_budget.max_source_blocks == 100
    assert s._save_budget.max_pending_bytes == 12800
    monkeypatch.setenv("OFFLOAD_SAVE_ADMISSION", "0")
    budget = build_save_budget(s._config, 4)
    for generation in range(10):
        assert (
            budget.reserve(SaveOperationId(generation, generation), range(1000), 4000)
            is None
        )
    assert budget.source_blocks == 10000
    assert budget.pending_bytes == 1280000


def test_metadata_construction_failure_does_not_reserve_or_advance(monkeypatch):
    from atom.kv_transfer.offload.dense import connector

    s = scheduler(monkeypatch)
    q = seq(1)
    s.update_state_after_alloc(q)

    def fail(**kwargs):
        raise RuntimeError("metadata construction failed")

    monkeypatch.setattr(connector, "LMCacheReqMeta", fail)
    with pytest.raises(RuntimeError, match="metadata construction failed"):
        s.build_connector_meta()
    assert s._save_budget.operations == {}
    assert s._save_operation_blocks == {}
    assert s._save_tracker["1"][1] == 0
    assert s.get_statistics()["save_ops_admitted"] == 0


@pytest.mark.parametrize(
    "env",
    [
        "OFFLOAD_MAX_PENDING_SAVE_BYTES",
        "OFFLOAD_MAX_RESERVED_SOURCE_BLOCKS",
        "OFFLOAD_MAX_PENDING_SAVE_TOKENS",
    ],
)
def test_invalid_budget_fails_at_startup(monkeypatch, env):
    with pytest.raises(ValueError, match="positive integer"):
        scheduler(monkeypatch, **{env: 0})
