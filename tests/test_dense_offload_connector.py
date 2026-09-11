# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadOperationId,
    SaveOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus


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
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_metadata",
        lambda *_args: object(),
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


def test_dense_scheduler_invalid_lmcache_config_fails_fast(monkeypatch):
    def invalid_config(_config=None):
        raise ValueError("invalid LMCache storage")

    monkeypatch.setattr(offcfg, "build_lmcache_config", invalid_config)

    with pytest.raises(ValueError, match="invalid LMCache storage"):
        DenseOffloadScheduler(_config())


def test_dense_scheduler_invalid_lmcache_metadata_fails_fast(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=8),
    )

    def invalid_metadata(*_args):
        raise ValueError("invalid LMCache layer geometry")

    monkeypatch.setattr(offcfg, "build_lmcache_metadata", invalid_metadata)

    with pytest.raises(ValueError, match="invalid LMCache layer geometry"):
        DenseOffloadScheduler(_config())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kv_cache_block_size", 4.5, "Dense block size must be an integer"),
        ("chunk_size", 8.5, "LMCache chunk size must be an integer"),
    ],
)
def test_dense_scheduler_rejects_coerced_geometry(
    monkeypatch,
    field,
    value,
    message,
):
    config = _config()
    if field == "kv_cache_block_size":
        config.kv_cache_block_size = value
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(
            chunk_size=value if field == "chunk_size" else 8
        ),
    )

    with pytest.raises(ValueError, match=message):
        DenseOffloadScheduler(config)


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


def test_dense_active_load_defers_only_its_concrete_lifecycle(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    seq = _load_seq(20)
    replacement = _load_seq(20)
    operation = LoadOperationId(seq.id, 3)
    scheduler._active_load_operations[str(seq.id)] = (seq, operation)

    assert scheduler.should_defer_free(seq) is True
    assert scheduler.should_defer_free(replacement) is False

    assert scheduler.load_finished(operation) is True
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


def test_dense_build_save_metadata_uses_increasing_exact_generations(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_producer")
    seq = _load_seq(12, num_prompt_tokens=16)
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 8

    first_meta = scheduler.build_connector_meta()
    first = first_meta.requests[0].save_operation
    scheduler.save_finished(first)
    seq.num_cached_tokens = 16
    second_meta = scheduler.build_connector_meta()
    second = second_meta.requests[0].save_operation

    assert first == SaveOperationId(req_id=12, generation=0)
    assert second == SaveOperationId(req_id=12, generation=1)
    assert scheduler._save_inflight["12"] == second


def test_dense_stale_or_raw_save_completion_cannot_clear_exact_lifecycle(
    monkeypatch,
):
    scheduler = _scheduler(monkeypatch, "kv_producer")
    seq = _load_seq(13, num_prompt_tokens=16)
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 8
    stale = scheduler.build_connector_meta().requests[0].save_operation
    scheduler.save_finished(stale)

    seq.num_cached_tokens = 16
    current = scheduler.build_connector_meta().requests[0].save_operation
    scheduler.save_finished(stale)
    scheduler.save_finished(seq.id)

    assert scheduler._save_inflight["13"] == current

    scheduler.save_finished(current)
    assert "13" not in scheduler._save_inflight

    # A raw completion remains compatible with an explicitly legacy lifecycle.
    scheduler._save_inflight["legacy"] = "legacy"
    scheduler.save_finished("legacy")
    assert "legacy" not in scheduler._save_inflight


def test_dense_worker_exact_save_generations_do_not_form_cross_tp_quorum():
    workers = [DenseOffloadConnector(_config("kv_producer")) for _ in range(2)]
    operations = [
        SaveOperationId(req_id=14, generation=6),
        SaveOperationId(req_id=14, generation=7),
    ]

    try:
        outputs = []
        for worker_idx, (worker, operation) in enumerate(zip(workers, operations)):
            worker.chunk_size = 8
            skip = 8
            if worker_idx:
                skip = 0
                worker._engine = SimpleNamespace(
                    gpu_connector=None,
                    store=lambda _tokens, **_kwargs: None,
                )
            worker._do_save_req(
                LMCacheReqMeta(
                    req_id=14,
                    token_ids=list(range(8)),
                    block_ids=[3],
                    save_spec=SaveSpec(skip_leading_tokens=skip),
                    save_operation=operation,
                )
            )
            outputs.append(worker.get_finished())

        assert outputs[0].finished_saving == {operations[0]}
        assert outputs[1].finished_saving == {operations[1]}
        assert (
            KVOutputAggregator(world_size=2).aggregate(outputs).finished_saving == set()
        )
    finally:
        for worker in workers:
            worker._save_executor.shutdown(wait=True)
            worker._load_executor.shutdown(wait=True)


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
    assert scheduler.failed_recving_kv_req_ids == [31]


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


@pytest.mark.parametrize("outcome", ["hbm_hit", "small_hit", "cancel", "miss", "error"])
def test_unused_lookup_releases_worker_pin(monkeypatch, outcome):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    worker = DenseOffloadConnector(_config("kv_consumer"))
    pins = set()
    calls = []

    def lookup(_tokens, lookup_id):
        pins.add(lookup_id)
        calls.append(lookup_id)
        if outcome == "error":
            raise RuntimeError("lookup transport failed after a worker pinned")
        return 0 if outcome == "miss" else 16

    scheduler._lookup_client = SimpleNamespace(
        lookup=lookup, clear_lookup_status=lambda _sid: None
    )
    worker._engine = SimpleNamespace(lookup_unpin=pins.discard)
    seq = _load_seq(52, num_prompt_tokens=24)
    if outcome == "hbm_hit":
        seq.num_cached_tokens = 16
    try:
        scheduler.get_num_new_matched_tokens(seq)
        if outcome == "cancel":
            scheduler.cancel_pending_load(seq)
        elif outcome == "small_hit":
            scheduler.update_state_after_alloc(seq)
        else:
            # Repeated scheduler probes must not acquire another pin lease.
            scheduler.get_num_new_matched_tokens(seq)
        assert calls == ["52"]
        assert scheduler.has_pending_work()
        metadata = scheduler.build_connector_meta()
        assert metadata.lookup_requests_in_step == ["52"]
        worker.start_load_kv(metadata)
        assert pins == set()
        assert not scheduler.has_pending_work()
    finally:
        worker.close()


def test_pending_load_keeps_pin_until_cancelled(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    scheduler._lookup_client = SimpleNamespace(
        lookup=lambda *_args, **_kwargs: 16,
        clear_lookup_status=lambda _sid: None,
    )
    seq = _load_seq(53, num_prompt_tokens=24)
    assert scheduler.get_num_new_matched_tokens(seq) == (16, True)
    # A pre-allocation lookup is owned by the waiting request, not dispatchable
    # idle work. It must retain its pin without keeping the idle drain alive.
    assert not scheduler.has_pending_work()
    assert scheduler.build_connector_meta().lookup_requests_in_step == []
    scheduler.cancel_pending_load(seq)
    assert scheduler.has_pending_work()
    assert scheduler.build_connector_meta().lookup_requests_in_step == ["53"]
    assert scheduler.build_connector_meta().lookup_requests_in_step == []


def test_scheduler_abort_before_allocation_releases_lookup_pin(
    monkeypatch, scheduler, seq_factory
):
    connector = _scheduler(monkeypatch, "kv_consumer")
    worker = DenseOffloadConnector(_config("kv_consumer"))
    pins = set()
    calls = []

    def lookup(_tokens, lookup_id):
        calls.append(lookup_id)
        pins.add(lookup_id)
        return 16

    connector._lookup_client = SimpleNamespace(
        lookup=lookup, clear_lookup_status=lambda _sid: None
    )
    worker._engine = SimpleNamespace(lookup_unpin=pins.discard)
    scheduler.kv_connector = connector
    seq = seq_factory(list(range(24)))
    scheduler.add(seq)
    allocation_attempts = []

    def cannot_allocate(value):
        allocation_attempts.append(value.id)
        return -1

    monkeypatch.setattr(scheduler.block_manager, "can_allocate", cannot_allocate)
    sid = str(seq.id)
    try:
        scheduler.schedule()
        assert list(scheduler.waiting) == [seq]
        assert calls == [sid]
        assert allocation_attempts == [seq.id]
        assert pins == {sid}
        assert sid in connector._load_specs
        assert not getattr(seq, "_counted_as_inflight_load", False)
        assert connector.build_connector_meta().lookup_requests_in_step == []

        seq.status = SequenceStatus.ABORTED
        batch, scheduled = scheduler.schedule()
        assert seq.status == SequenceStatus.FINISHED
        assert scheduler._num_parked_remote_kv == 0
        assert sid not in connector._load_specs
        assert sid not in connector._load_lifecycles
        assert scheduler.deferred_free_blocks == {}
        assert scheduled == {}
        # schedule() already dispatched cleanup into this empty batch.
        meta = batch.connector_meta_output
        assert meta.requests == []
        assert meta.lookup_requests_in_step == [sid]
        worker.start_load_kv(meta)
        assert pins == set()
        assert connector._lookup_results == {}
        assert not connector.has_pending_work()
        assert connector.build_connector_meta().lookup_requests_in_step == []
    finally:
        worker.close()


def test_lookup_id_reuse_does_not_consume_an_old_pin(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    calls = []
    scheduler._lookup_client = SimpleNamespace(
        lookup=lambda _tokens, lookup_id: calls.append(lookup_id) or 16,
        clear_lookup_status=lambda _sid: None,
    )
    old = _load_seq(54, num_prompt_tokens=24)
    new = _load_seq(54, num_prompt_tokens=32)
    scheduler.get_num_new_matched_tokens(old)
    scheduler.cancel_pending_load(old)
    assert scheduler.get_num_new_matched_tokens(new) == (0, False)
    assert calls == ["54"]
    assert scheduler.build_connector_meta().lookup_requests_in_step == ["54"]
    assert scheduler.get_num_new_matched_tokens(new) == (16, True)
    assert calls == ["54", "54"]


def test_hbm_catches_up_after_a_pending_cpu_lookup(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    scheduler._lookup_client = SimpleNamespace(
        lookup=lambda *_args, **_kwargs: 16,
        clear_lookup_status=lambda _sid: None,
    )
    seq = _load_seq(55, num_prompt_tokens=24)
    assert scheduler.get_num_new_matched_tokens(seq) == (16, True)
    seq.num_cached_tokens = 16
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)
    metadata = scheduler.build_connector_meta()
    assert metadata.lookup_requests_in_step == ["55"]
    assert metadata.requests == []


def test_cpu_pin_is_retained_until_retrieve_finishes(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    scheduler._min_load_tokens = 0
    scheduler._lookup_client = SimpleNamespace(
        lookup=lambda *_args, **_kwargs: 16,
        clear_lookup_status=lambda _sid: None,
    )
    seq = _load_seq(56, num_prompt_tokens=24)
    scheduler.get_num_new_matched_tokens(seq)
    scheduler.update_state_after_alloc(seq)
    metadata = scheduler.build_connector_meta()
    pins = {"56"}

    def retrieve(_tokens, *, mask, **_kwargs):
        assert pins == {"56"}
        return mask.clone()

    worker = DenseOffloadConnector(_config("kv_consumer"))
    worker.chunk_size = 8
    worker._engine = SimpleNamespace(retrieve=retrieve, lookup_unpin=pins.discard)
    try:
        worker.start_load_kv(metadata)
        worker.close()
        assert pins == set()
        assert worker.get_finished().finished_loading == {
            metadata.requests[0].load_operation
        }
    finally:
        worker.close()
