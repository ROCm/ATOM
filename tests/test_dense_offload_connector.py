# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec
from atom.model_engine.scheduler import Scheduler


def _config(role="offload"):
    return SimpleNamespace(
        kv_transfer_config={"kv_role": role},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        tensor_parallel_size=1,
    )


def _scheduler(monkeypatch, role="offload"):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=8),
    )
    return DenseOffloadScheduler(_config(role))


def _load_seq(req_id, *, num_prompt_tokens=8):
    return SimpleNamespace(
        id=req_id,
        num_cached_tokens=0,
        num_prompt_tokens=num_prompt_tokens,
        token_ids=list(range(num_prompt_tokens)),
        block_table=list(range(num_prompt_tokens // 8)),
    )


def _arm_load(scheduler, seq, *, hbm=0, lmcache=8):
    sid = str(seq.id)
    scheduler._min_load_tokens = 0
    scheduler._load_specs[sid] = LoadSpec(
        hbm_cached_tokens=hbm,
        lmcache_cached_tokens=lmcache,
        can_load=True,
    )
    scheduler._reqs_need_recv[sid] = seq


def _engine_scheduler(connector):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_connector = connector
    scheduler.finished_recving_kv_req_ids = []
    scheduler.failed_recving_kv_req_ids = []
    scheduler.deferred_free_blocks = {}
    return scheduler


@pytest.mark.parametrize(
    "connector_cls", [DenseOffloadConnector, DenseOffloadScheduler]
)
def test_dense_backend_rejects_unknown_role(monkeypatch, connector_cls):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=8),
    )

    with pytest.raises(ValueError, match="invalid kv_role"):
        connector_cls(_config("invalid"))


def test_dense_worker_resolves_virtual_dcp_block_size():
    worker = DenseOffloadConnector(_config())
    try:
        assert worker.block_size == 4
        assert worker.virtual_block_size == 8
    finally:
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


def test_dense_producer_role_tracks_only_saves(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_producer")
    seq = SimpleNamespace(
        id=1,
        num_cached_tokens=0,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        block_table=[3],
    )

    scheduler.update_state_after_alloc(seq)

    assert scheduler._do_save is True
    assert scheduler._do_load is False
    assert scheduler._save_tracker["1"] == [seq, 0]
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)


def test_dense_consumer_role_does_not_defer_for_saves(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    seq = SimpleNamespace(
        id=2,
        num_cached_tokens=8,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        block_table=[4],
    )

    scheduler.update_state_after_alloc(seq)

    assert scheduler._do_save is False
    assert scheduler._save_tracker == {}
    assert scheduler.should_defer_free(seq) is False


def test_dense_reused_request_id_resets_save_frontier(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    first = SimpleNamespace(
        id=3,
        num_cached_tokens=8,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        block_table=[5],
    )
    replacement = SimpleNamespace(
        id=3,
        num_cached_tokens=0,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        block_table=[6],
    )
    scheduler._save_tracker["3"] = [first, 8]

    scheduler.update_state_after_alloc(replacement)

    assert scheduler._save_tracker["3"] == [replacement, 0]


def test_dense_build_load_metadata_uses_increasing_exact_generations(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    seq = _load_seq(11)

    _arm_load(scheduler, seq)
    first = scheduler.build_connector_meta().requests[0].load_operation
    _arm_load(scheduler, seq)
    second = scheduler.build_connector_meta().requests[0].load_operation

    assert first == LoadOperationId(req_id=11, generation=0)
    assert second == LoadOperationId(req_id=11, generation=1)
    assert scheduler._active_load_operations["11"] == (seq, second)


@pytest.mark.parametrize("outcome", ["exception", "miss"])
def test_dense_worker_load_failure_reports_exact_operation(outcome):
    worker = DenseOffloadConnector(_config("kv_consumer"))
    operation = LoadOperationId(req_id=21, generation=4)
    request = LMCacheReqMeta(
        req_id=21,
        token_ids=list(range(8)),
        block_ids=[3],
        load_spec=LoadSpec(
            hbm_cached_tokens=0,
            lmcache_cached_tokens=8,
            can_load=True,
        ),
        load_operation=operation,
    )
    worker.chunk_size = 8

    try:
        if outcome == "exception":

            def fail_load(_request):
                raise RuntimeError("synthetic load failure")

            worker._guard("load", fail_load, request)
        else:

            class MissEngine:
                gpu_connector = None

                @staticmethod
                def retrieve(_tokens, *, mask, **_kwargs):
                    return mask.clone().fill_(False)

                @staticmethod
                def lookup_unpin(_lookup_id):
                    pass

            worker._engine = MissEngine()
            worker._do_load_req(request)

        result = worker.get_finished()

        assert result.finished_loading == set()
        assert result.failed_loading == {operation}
    finally:
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


def test_dense_exact_load_failure_rolls_back_save_frontier(monkeypatch):
    connector = _scheduler(monkeypatch)
    seq = _load_seq(31, num_prompt_tokens=16)
    operation = LoadOperationId(req_id=31, generation=2)
    connector._save_tracker["31"] = [seq, 16]
    connector._load_save_floors["31"] = 8
    connector._active_load_operations["31"] = (seq, operation)
    scheduler = _engine_scheduler(connector)

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(failed_loading={operation})
    )

    assert connector._save_tracker["31"] == [seq, 8]
    assert "31" not in connector._active_load_operations
    assert "31" not in connector._load_save_floors
    assert scheduler.failed_recving_kv_req_ids == [operation]


def test_dense_stale_load_generation_does_not_clear_active_operation(monkeypatch):
    connector = _scheduler(monkeypatch)
    seq = _load_seq(41, num_prompt_tokens=16)
    stale = LoadOperationId(req_id=41, generation=6)
    active = LoadOperationId(req_id=41, generation=7)
    connector._save_tracker["41"] = [seq, 16]
    connector._load_save_floors["41"] = 8
    connector._active_load_operations["41"] = (seq, active)
    scheduler = _engine_scheduler(connector)

    scheduler._update_from_kv_xfer_finished(KVConnectorOutput(failed_loading={stale}))

    assert connector._active_load_operations["41"] == (seq, active)
    assert connector._load_save_floors["41"] == 8
    assert connector._save_tracker["41"] == [seq, 16]
    assert scheduler.failed_recving_kv_req_ids == []


def test_dense_lookup_unpin_passes_one_string_id():
    worker = DenseOffloadConnector(_config("kv_consumer"))
    received = []
    worker._engine = SimpleNamespace(lookup_unpin=received.append)

    try:
        worker._lookup_unpin(51)

        assert received == ["51"]
    finally:
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)
