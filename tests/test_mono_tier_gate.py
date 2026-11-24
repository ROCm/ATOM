# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The mono_blocked gate: a decode step stays off Mono while the GPU guard sees a
foreign process or a CPU-tier load / save is not yet terminal on every TP rank."""

from types import SimpleNamespace

import pytest
from conftest import with_min_load

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadOperationId,
    SaveOperationId,
    mono_offload_hazard,
)
from atom.kv_transfer.offload import chunked_scheduler
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.metadata import (
    LoadSpec,
)
from atom.model_engine.engine_core import EngineCore


def _config(role):
    return with_min_load(
        SimpleNamespace(
            kv_transfer_config={"kv_role": role},
            kv_cache_block_size=4,
            decode_context_parallel_size=2,
            tensor_parallel_size=1,
        )
    )


@pytest.fixture
def scheduler_of(monkeypatch):
    monkeypatch.setattr(
        offcfg, "build_lmcache_config", lambda _config=None: SimpleNamespace(chunk_size=8)
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    return lambda role="offload": DenseOffloadScheduler(_config(role))


def _seq(req_id, n=16):
    return SimpleNamespace(
        id=req_id,
        num_cached_tokens=0,
        num_prompt_tokens=n,
        token_ids=list(range(n)),
        block_table=list(range(n // 8)),
        offload_loaded=False,
        offload_load_failed=False,
    )


def _dispatch_load(scheduler, seq):
    sid = str(seq.id)
    scheduler._min_load_tokens = 0
    scheduler._load_specs[sid] = LoadSpec(
        hbm_cached_tokens=0, lmcache_cached_tokens=8, can_load=True
    )
    scheduler._reqs_need_recv[sid] = seq
    scheduler._lookup_results[sid] = (seq, 8)
    scheduler._load_lifecycles[sid] = seq
    return scheduler.build_connector_meta().requests[0].load_operation


def _dispatch_save(scheduler, seq):
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 8
    return scheduler.build_connector_meta().requests[0].save_operation


def test_load_blocks_mono_until_every_rank_reports(scheduler_of):
    scheduler = scheduler_of("kv_consumer")
    seq = _seq(1)
    assert not scheduler.mono_hazard()
    op = _dispatch_load(scheduler, seq)
    assert isinstance(op, LoadOperationId)
    # dispatched: the old predicate already let go of it
    assert not scheduler._reqs_need_recv
    assert scheduler.mono_hazard()

    agg = KVOutputAggregator(world_size=2)
    one_rank = agg.aggregate([KVConnectorOutput(finished_loading={op}), KVConnectorOutput()])
    scheduler.process_completions(one_rank)
    assert scheduler.mono_hazard(), "rank 1 may still run the unpack kernels"
    both = agg.aggregate([KVConnectorOutput(), KVConnectorOutput(finished_loading={op})])
    scheduler.process_completions(both)
    assert not scheduler.mono_hazard()


def test_cancelled_load_still_blocks_until_its_terminal(scheduler_of):
    scheduler = scheduler_of("kv_consumer")
    seq = _seq(2)
    op = _dispatch_load(scheduler, seq)
    scheduler.cancel_pending_load(seq)
    assert str(seq.id) not in scheduler._active_load_operations
    assert scheduler.mono_hazard(), "the worker's load runs on after a cancel"
    scheduler.process_completions(KVConnectorOutput(failed_loading={op}))
    assert not scheduler.mono_hazard()


def test_save_blocks_mono_until_reported(scheduler_of):
    scheduler = scheduler_of("kv_producer")
    seq = _seq(3)
    op = _dispatch_save(scheduler, seq)
    assert isinstance(op, SaveOperationId)
    assert scheduler.mono_hazard()
    scheduler.process_completions(KVConnectorOutput(finished_saving={op}))
    assert not scheduler.mono_hazard()


def test_raw_request_report_settles_that_request_only(scheduler_of):
    scheduler = scheduler_of("kv_consumer")
    a, b = _seq(4), _seq(5)
    _dispatch_load(scheduler, a)
    op_b = _dispatch_load(scheduler, b)
    scheduler._settle_worker_ops(KVConnectorOutput(finished_loading={4}))
    assert set(scheduler._worker_ops) == {op_b}


def test_lost_report_stops_blocking_after_the_limit(scheduler_of, monkeypatch):
    scheduler = scheduler_of("kv_consumer")
    op = _dispatch_load(scheduler, _seq(6))
    scheduler._active_load_operations.clear()
    monkeypatch.setattr(chunked_scheduler, "MONO_TIER_GATE_MAX_S", 30.0)
    assert scheduler.mono_hazard()
    scheduler._worker_ops[op] -= 31.0
    assert not scheduler.mono_hazard()


def test_hazard_predicate_falls_back_to_pending_work():
    assert not mono_offload_hazard(None)
    assert mono_offload_hazard(SimpleNamespace(has_pending_work=lambda: True))
    assert not mono_offload_hazard(SimpleNamespace(has_pending_work=lambda: False))
    assert mono_offload_hazard(
        SimpleNamespace(mono_hazard=lambda: True, has_pending_work=lambda: False)
    )


def _engine(guard_present, connector, kv=True):
    core = EngineCore.__new__(EngineCore)
    core.label = "EngineCore"
    core.gpu_guard = (
        None if guard_present is None else SimpleNamespace(present=lambda: guard_present)
    )
    core.kv_transfer_enabled = kv
    core.scheduler = SimpleNamespace(kv_connector=connector)
    return core


@pytest.mark.parametrize(
    "guard, hazard, kv, why",
    [
        (None, False, True, None),
        (False, False, True, None),
        (True, False, True, "GPU guard"),
        (True, True, True, "GPU guard"),
        (False, True, True, "KV tier transfer"),
        (None, True, True, "KV tier transfer"),
        (False, True, False, None),
    ],
)
def test_engine_decides_once_per_step(guard, hazard, kv, why):
    core = _engine(guard, SimpleNamespace(mono_hazard=lambda: hazard), kv)
    assert core._mono_block_reason() == why
