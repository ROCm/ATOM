# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    LoadOperationId,
    SaveOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
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
    # An armed load always comes from a lookup, and the pin it took is what the
    # dispatch confirms before handing the retrieve to the worker.
    scheduler._lookup_results[sid] = (seq, lmcache)


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


def test_dense_save_passes_one_step_producer_event_to_every_store(monkeypatch):
    trace = []
    rpc_stream = object()
    rpc_thread = threading.get_ident()
    events = []

    class Event:
        def __init__(self):
            events.append(self)

        def record(self, stream):
            # The fence is recorded once, on the dispatching RPC thread, so a
            # save thread never records its own (unordered) event.
            assert threading.get_ident() == rpc_thread
            assert stream is rpc_stream
            trace.append(("record", self))

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: rpc_stream)

    class Engine:
        gpu_connector = object()

        @staticmethod
        def store(_tokens, **kwargs):
            trace.append(("store", kwargs["producer_event"]))

    worker = DenseOffloadConnector(_config("kv_producer"))
    worker.chunk_size = 8
    worker._engine = Engine()
    metadata = LMCacheOffloadMetadata()
    metadata.requests.extend(
        [
            LMCacheReqMeta(
                req_id=req_id,
                token_ids=list(range(8)),
                block_ids=[req_id],
                save_spec=SaveSpec(skip_leading_tokens=0),
            )
            for req_id in (31, 32)
        ]
    )

    try:
        worker.start_load_kv(metadata)
        worker.close()

        assert len(events) == 1
        event = events[0]
        assert trace[0] == ("record", event)
        assert trace[1:] == [
            ("store", event),
            ("store", event),
        ]
    finally:
        worker.close()


def test_guard_forwards_keyword_arguments_and_logs_function_name(caplog):
    worker = DenseOffloadConnector(_config("kv_producer"))
    seen = []

    def fail_save(req, *, producer_event):
        seen.append((req.req_id, producer_event))
        raise RuntimeError("boom")

    request = LMCacheReqMeta(
        req_id=41,
        token_ids=list(range(8)),
        block_ids=[1],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(req_id=41, generation=0),
    )
    try:
        with caplog.at_level("ERROR", logger="atom"):
            worker._guard("save", fail_save, request, producer_event="fence")
        assert seen == [(41, "fence")]
        assert "fail_save failed for 41" in caplog.text
        assert worker.get_finished().finished_saving == {request.save_operation}
    finally:
        worker.close()


def test_dense_save_fence_failure_does_not_drop_the_step_load(monkeypatch):
    load_operation = LoadOperationId(req_id=33, generation=1)
    save_operation = SaveOperationId(req_id=33, generation=1)
    second_save_operation = SaveOperationId(req_id=34, generation=2)
    record_calls = []

    class Event:
        @staticmethod
        def record(_stream):
            record_calls.append(1)
            raise RuntimeError("sticky HIP error")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", object)

    worker = DenseOffloadConnector(_config("kv_both"))
    worker.chunk_size = 8
    worker._engine = SimpleNamespace(
        gpu_connector=object(),
        lookup_unpin=lambda _req_id: None,
        store=lambda *_args, **_kwargs: pytest.fail(
            "an unfenced save must not be submitted"
        ),
    )
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(
        LMCacheReqMeta(
            req_id=33,
            token_ids=list(range(8)),
            block_ids=[3],
            load_spec=LoadSpec(
                hbm_cached_tokens=0,
                lmcache_cached_tokens=0,
                can_load=True,
            ),
            save_spec=SaveSpec(skip_leading_tokens=0),
            load_operation=load_operation,
            save_operation=save_operation,
        )
    )
    metadata.add_request(
        LMCacheReqMeta(
            req_id=34,
            token_ids=list(range(8)),
            block_ids=[4],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=second_save_operation,
        )
    )

    try:
        worker.start_load_kv(metadata)
        worker.close()
        output = worker.get_finished()

        assert output.finished_loading == {load_operation}
        assert output.failed_loading == set()
        assert record_calls == [1]
        assert output.finished_saving == {save_operation, second_save_operation}
        assert (
            ConnectorCompletion("dense.page.store", save_operation, False)
            in output.connector_completions
        )
        assert (
            ConnectorCompletion("dense.page.source_quiescent", save_operation, True)
            in output.connector_completions
        )
        assert (
            ConnectorCompletion(
                "dense.page.source_quiescent", second_save_operation, True
            )
            in output.connector_completions
        )
    finally:
        worker.close()


def test_block_gpu_connector_waits_on_actual_pack_stream_without_host_sync():
    trace = []
    stats = {}
    producer_event = SimpleNamespace(
        synchronize=lambda: pytest.fail("producer event must not host-synchronize")
    )
    pack_stream = SimpleNamespace(
        wait_event=lambda event: trace.append(("wait", event))
    )
    state = SimpleNamespace(pack_stream=pack_stream, copy_stream=object())
    connector = BlockGPUConnector.__new__(BlockGPUConnector)
    connector._capture_transfer_stats = lambda: nullcontext(stats)
    connector._prepare_transfer = lambda *_args, **_kwargs: (state, [object()])
    connector._record_transfer_shape = lambda *_args: None

    def prepare_block_id_stage(*_args):
        # The block-ID upload is the first pack-stream work of a transfer; it
        # must already be ordered behind the producer.
        trace.append(("block_ids", None))
        return object(), None, False

    connector._prepare_block_id_stage = prepare_block_id_stage
    connector._run_staged_pipeline = lambda *_args, **_kwargs: trace.append(
        ("pipeline", None)
    )

    connector.batched_from_gpu(
        [object()],
        [0],
        [8],
        producer_event=producer_event,
    )

    assert trace == [
        ("wait", producer_event),
        ("block_ids", None),
        ("pipeline", None),
    ]
    assert stats["producer_fenced"] == 1


def test_block_gpu_connector_producer_event_is_keyword_only():
    connector = BlockGPUConnector.__new__(BlockGPUConnector)
    with pytest.raises(TypeError):
        connector.batched_from_gpu([object()], [0], [8], object())


def test_block_gpu_connector_refuses_producer_fence_without_pack_stream():
    producer_event = SimpleNamespace(
        synchronize=lambda: pytest.fail("must not fall back to host sync")
    )
    state = SimpleNamespace(pack_stream=None, copy_stream=None)
    with pytest.raises(RuntimeError, match="pack stream"):
        BlockGPUConnector._wait_for_save_source(state, producer_event)


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

    def cannot_allocate(value, **_kwargs):
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


def _counting_lookup(calls, hit):
    def lookup(_tokens, lookup_id):
        calls.append(lookup_id)
        return hit

    return SimpleNamespace(lookup=lookup, clear_lookup_status=lambda _sid: None)


def test_native_retry_on_a_full_kv_cache_reuses_the_lookup(
    monkeypatch, scheduler, seq_factory
):
    """One lookup per frontier, not one per scheduler step.

    ATOM's own scheduler runs the external-tier lookup at the top of its waiting
    loop, ahead of `can_allocate`, and every failure path puts the sequence back
    at the head of `waiting` with its frontier untouched -- so the next step
    asks the identical question. A complete miss pins nothing, so
    `build_connector_meta` dispatches the lookup's cleanup and the step after
    that hashes the whole prompt again. With the KV cache full that is every
    step, on the scheduler thread, which is how the engine livelocks.
    """

    connector = _scheduler(monkeypatch, "kv_consumer")
    calls = []
    connector._lookup_client = _counting_lookup(calls, 0)
    scheduler.kv_connector = connector
    seq = seq_factory(list(range(24)))
    scheduler.add(seq)
    monkeypatch.setattr(
        scheduler.block_manager, "can_allocate", lambda seq, **_kwargs: -1
    )

    for _ in range(8):
        scheduler.schedule()

    assert list(scheduler.waiting) == [seq]
    assert calls == [str(seq.id)]
    assert connector.total_lookups_skipped_by_memo == 7


def test_native_admission_spends_the_memo(monkeypatch, scheduler, seq_factory):
    """Admission spends the remembered hit.

    Once the request is scheduled its load spec has been dispatched and its
    frontier moves, so the next question about this id is a different one and
    has to reach the tier. A preempted request therefore pays for one lookup,
    not for a replay of an answer that no longer describes anything.
    """

    connector = _scheduler(monkeypatch, "kv_consumer")
    calls = []
    connector._lookup_client = _counting_lookup(calls, 0)
    scheduler.kv_connector = connector
    seq = seq_factory(list(range(24)))
    scheduler.add(seq)
    kv_full = {"yes": True}
    can_allocate = scheduler.block_manager.can_allocate
    monkeypatch.setattr(
        scheduler.block_manager,
        "can_allocate",
        lambda seq, **kwargs: -1 if kv_full["yes"] else can_allocate(seq, **kwargs),
    )

    scheduler.schedule()
    assert calls == [str(seq.id)]
    assert str(seq.id) in connector._tier_hit_memo

    kv_full["yes"] = False
    scheduler.schedule()

    assert list(scheduler.waiting) == []
    assert connector._tier_hit_memo == {}


def test_a_declined_load_is_not_looked_up_again_every_step(monkeypatch):
    """The plugin's shape: the decline is what releases the lookup.

    vLLM hands ATOM the real HBM frontier, so `should_park_for_load_after_alloc`
    routinely declines a hit -- already covered by HBM, unaligned, or below the
    transfer floor. Declining clears the pending load, which makes the lookup
    dispatchable, so the pin is gone by the next step. The request is still at
    the head of the waiting queue, so without the remembered hit the next step
    pays the tier again. A declined hit parks nothing, so the remembered answer
    is the whole answer and no lookup is needed to escalate it.
    """

    sched = _scheduler(monkeypatch, "kv_consumer")
    sched._min_load_tokens = 8192  # every hit here is "too small" to transfer
    calls = []
    sched._lookup_client = _counting_lookup(calls, 16)
    seq = _load_seq(60, num_prompt_tokens=24)

    for _ in range(8):
        need, _park = sched.get_num_new_matched_tokens(seq)
        assert not (need > 0 and sched.should_park_for_load_after_alloc(seq))
        sched.build_connector_meta()

    assert calls == ["60"]
    assert sched.total_lookups_skipped_by_memo == 7
    assert "60" not in sched._load_specs


def test_a_moved_frontier_is_re_derived_from_the_same_hit(monkeypatch):
    """The frontier moves; the tier's answer about the prompt does not.

    How much of the prompt the tier holds is a property of the prompt, so the
    HBM frontier is not part of the question -- it only decides how much of
    that hit is still worth transferring. Re-asking the tier because the
    frontier moved would pay for the one expensive input again to learn
    something already known.
    """

    sched = _scheduler(monkeypatch, "kv_consumer")
    sched._min_load_tokens = 0
    calls = []
    sched._lookup_client = _counting_lookup(calls, 16)
    seq = _load_seq(61, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    sched.build_connector_meta()
    seq.num_cached_tokens = 8

    assert sched.get_num_new_matched_tokens(seq) == (8, True)
    assert calls == ["61"]


def test_a_failed_lookup_backs_off_before_it_is_retried(monkeypatch):
    """A timeout is not an answer, but retrying it every step is what hurts.

    Each retry costs `lmcache.mp.lookup_timeout` seconds *of the scheduler
    thread*, so a tier that has stopped replying would stall the engine harder
    than the hashing did. The non-answer is remembered too, for
    `OFFLOAD_LOOKUP_RETRY_STEPS` steps, and then the tier is asked again -- a
    dropped reply must not become a permanent miss for the rest of the
    request's life.
    """

    sched = _scheduler(monkeypatch, "kv_consumer")
    sched._min_load_tokens = 0
    sched._tier_retry_steps = 1
    replies = [None, 16]
    calls = []

    def lookup(_tokens, lookup_id):
        calls.append(lookup_id)
        return replies.pop(0)

    sched._lookup_client = SimpleNamespace(
        lookup=lookup, clear_lookup_status=lambda _sid: None
    )
    seq = _load_seq(62, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    sched.build_connector_meta()

    # Inside the backoff: answered from the remembered non-answer.
    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert calls == ["62"]
    sched.build_connector_meta()

    # Budget spent: the tier is asked again.
    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    assert calls == ["62", "62"]


def test_a_stale_memo_expires_and_asks_the_tier_again(monkeypatch):
    """The remembered hit describes a tier that keeps moving.

    Nothing tells this connector that another engine stored a longer prefix, or
    that this one evicted the prefix it matched. So the memo is a bounded
    optimisation, not a cache: after `OFFLOAD_LOOKUP_MEMO_STEPS` replays the
    question is put to the tier again.
    """

    sched = _scheduler(monkeypatch, "kv_consumer")
    sched._min_load_tokens = 8192  # every hit here is declined after alloc
    sched._tier_memo_steps = 2
    calls = []
    sched._lookup_client = _counting_lookup(calls, 16)
    seq = _load_seq(65, num_prompt_tokens=24)

    for _ in range(4):
        need, _park = sched.get_num_new_matched_tokens(seq)
        if need > 0:
            sched.should_park_for_load_after_alloc(seq)
        sched.build_connector_meta()

    assert calls == ["65", "65"]


def test_a_multi_connector_cancel_re_arms_honestly(monkeypatch):
    """Losing to another sub must not cost a fresh hash every step.

    `MultiConnectorScheduler` queries *every* sub on every scheduler pass and
    cancels the pending load of everyone but the winner, so under the supported
    `[moriio, lmcache_offload]` ordering the offload sub arms a load and has it
    taken away. The cancel does not forget the hit: the tier did not stop
    holding the prefix because another connector won this step. So when moriio
    stops matching -- its `kv_async_tagged` is one-shot -- the offload sub
    answers from what it already knows, and only the dispatch pays the tier.
    """

    sched = _scheduler(monkeypatch, "kv_consumer")
    sched._min_load_tokens = 0
    calls = []
    sched._lookup_client = _counting_lookup(calls, 16)
    moriio_matched = {"yes": False}

    def moriio_match(_seq):
        if moriio_matched["yes"]:
            return 0, False
        moriio_matched["yes"] = True
        return 16, True

    moriio = SimpleNamespace(get_num_new_matched_tokens=moriio_match)
    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = [moriio, sched]
    multi._load_winner = {}
    seq = _load_seq(64, num_prompt_tokens=24)

    for _ in range(5):
        assert multi.get_num_new_matched_tokens(seq) == (16, True)
        sched.build_connector_meta()

    assert calls == ["64"]
    assert sched._load_specs["64"].lmcache_cached_tokens == 16


def test_a_finished_request_leaves_no_memo_behind(monkeypatch):
    sched = _scheduler(monkeypatch, "kv_consumer")
    calls = []
    sched._lookup_client = _counting_lookup(calls, 0)
    seq = _load_seq(63, num_prompt_tokens=24)

    sched.get_num_new_matched_tokens(seq)
    sched.request_finished(seq)

    assert sched._tier_hit_memo == {}


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


def test_dense_worker_records_the_blocks_a_failed_load_left_unfilled():
    """The id alone does not truncate anything.

    vLLM caches the whole external prefix unless the failure also names blocks:
    `_update_requests_with_invalid_blocks` cuts `num_computed_tokens` at the
    first block reported here. Blocks below the HBM frontier are deliberately
    excluded -- they hold valid KV and may be shared with another request.

    The grid is the virtual block size (block_size 4 x dcp 2 = 8), matching the
    one `BlockGPUConnector` maps chunks with; the physical size would index
    different entries of the very same table.
    """
    worker = DenseOffloadConnector(_config("kv_consumer"))
    request = LMCacheReqMeta(
        req_id=61,
        token_ids=list(range(16)),
        block_ids=[10, 11, 12, 13],
        load_spec=LoadSpec(
            hbm_cached_tokens=4,
            lmcache_cached_tokens=12,
            can_load=True,
        ),
        load_operation=LoadOperationId(req_id=61, generation=1),
    )

    try:
        with worker._lock:
            worker._record_load_error_blocks(request)

        # Tokens [4, 12) on a grid of 8 are entries 0 and 1 of the table.
        assert worker.take_load_error_blocks() == {10, 11}
        # Drained, so the next step does not truncate a request all over again.
        assert worker.take_load_error_blocks() == set()
    finally:
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


def test_dense_worker_fences_in_flight_jobs_for_a_preempted_request():
    """Preemption hands the blocks to somebody else in the same step.

    A save still gathering from them stores the new occupant's bytes under the
    preempted request's key -- a poisoned cache entry, not a lost one. The fence
    is the only thing standing between the two.
    """
    import threading

    worker = DenseOffloadConnector(_config("kv_consumer"))
    release = threading.Event()
    finished = []

    def slow_job(_request):
        release.wait(timeout=5)
        finished.append(True)

    try:
        worker._track_job(
            71,
            worker._save_executor.submit(
                worker._guard, "save", slow_job, SimpleNamespace(req_id=71)
            ),
        )
        assert worker._inflight_jobs["71"]

        # Another request's ids are not this request's business.
        worker.wait_for_requests(["72"])
        assert finished == []

        release.set()
        worker.wait_for_requests(["71"])

        assert finished == [True]
        assert "71" not in worker._inflight_jobs
    finally:
        release.set()
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


def test_dense_load_failure_by_request_resolves_the_parked_generation(monkeypatch):
    """vLLM hands back plain strings; `load_failed` refuses a raw id.

    Routing the failure through `load_finished` instead would pop the floor
    recording that the [HBM, LMCache) range is NOT persisted, so the recomputed
    chunks would never be saved.
    """
    connector = _scheduler(monkeypatch)
    seq = _load_seq(81, num_prompt_tokens=16)
    operation = LoadOperationId(req_id=81, generation=3)
    connector._save_tracker["81"] = [seq, 16]
    connector._load_save_floors["81"] = 8
    connector._active_load_operations["81"] = (seq, operation)

    assert connector.load_failed_by_request("81") is True

    assert connector._save_tracker["81"] == [seq, 8]
    assert "81" not in connector._active_load_operations


def test_dense_worker_pool_widths_follow_env(monkeypatch):
    """A single load thread saturates once the CPU tier serves real traffic.

    Measured on the radix workload with the HBM pool squeezed to 7900 blocks:
    88% duty cycle inside `retrieve` on every rank, which turned a +57.8pp
    hit-rate win into a throughput loss. Both pools must be tunable, and both
    must keep their one-thread default so existing deployments are unchanged.
    """

    worker = DenseOffloadConnector(_config())
    try:
        assert (worker.save_workers, worker.load_workers) == (1, 1)
        assert worker._save_executor._max_workers == 1
        assert worker._load_executor._max_workers == 1
    finally:
        worker.close()

    monkeypatch.setenv("OFFLOAD_COPY_WORKERS", "4")
    monkeypatch.setenv("OFFLOAD_LOAD_WORKERS", "3")
    worker = DenseOffloadConnector(_config())
    try:
        assert (worker.save_workers, worker.load_workers) == (4, 3)
        assert worker._save_executor._max_workers == 4
        assert worker._load_executor._max_workers == 3
    finally:
        worker.close()


@pytest.mark.parametrize("var", ["OFFLOAD_COPY_WORKERS", "OFFLOAD_LOAD_WORKERS"])
def test_dense_worker_rejects_non_positive_pool_width(monkeypatch, var):
    monkeypatch.setenv(var, "0")
    with pytest.raises(ValueError, match="worker count must be positive"):
        DenseOffloadConnector(_config())


class _StubLookupClient:
    """Lookup client that always reports the same hit length."""

    def __init__(self, hit):
        self.hit = hit
        self.calls = 0
        self.cleared = []

    def lookup(self, token_ids, lookup_id):
        self.calls += 1
        return self.hit

    def clear_lookup_status(self, lookup_id):
        self.cleared.append(lookup_id)


def _lookup_scheduler(monkeypatch, hit, *, role="offload"):
    sched = _scheduler(monkeypatch, role)
    sched._lookup_client = _StubLookupClient(hit)
    sched._min_load_tokens = 0  # these prompts are far below the 8192 default
    return sched


def _dispatch_and_fail_load(sched, seq):
    """Run the alloc + metadata step, then fail the load it dispatched.

    The metadata build is what releases the lookup memo, so a later pass is a
    fresh lookup rather than the "older lifecycle still owns the pin" deferral.
    """

    sched.update_state_after_alloc(seq)
    meta = sched.build_connector_meta()
    (load,) = [req for req in meta.requests if req.load_spec is not None]
    assert sched.load_failed(load.load_operation) is True


def test_dense_full_prompt_hit_is_floored_to_a_loadable_chunk(monkeypatch):
    # A hit covering the whole prompt is decremented so something is left to
    # compute. With a prompt length that is an exact multiple of the chunk
    # size, that decrement lands off the chunk boundary -- and LMCache resolves
    # at chunk granularity, so the resulting spec asks for tokens the tier can
    # never return and the load fails every time. The floor is what keeps the
    # spec satisfiable. Scaled-down mirror of the production shape observed on
    # GLM-5.2 (chunk 64, prompt 32768: 32767 requested, 32704 available).
    sched = _lookup_scheduler(monkeypatch, hit=16)
    assert sched.chunk_size == 8
    seq = _load_seq(940, num_prompt_tokens=16)

    assert sched.get_num_new_matched_tokens(seq) == (8, True)
    assert sched._load_specs["940"].lmcache_cached_tokens == 8


def test_dense_partial_hit_off_a_chunk_boundary_is_floored(monkeypatch):
    # The same arithmetic with no decrement involved: any unaligned hit names
    # tokens the tier does not hold at chunk granularity.
    sched = _lookup_scheduler(monkeypatch, hit=13)
    seq = _load_seq(941, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (8, True)
    assert sched._load_specs["941"].lmcache_cached_tokens == 8


def test_dense_failed_load_is_not_retried_for_the_same_request(monkeypatch):
    # `load_failed` clears the pending load and the lookup memo, so without a
    # record of the attempt the next scheduler pass looks up, hits, parks the
    # request in WAITING_FOR_REMOTE_KVS and fails again -- forever, holding its
    # KV blocks and concurrency slot. One attempt per request; after that the
    # request prefills normally.
    sched = _lookup_scheduler(monkeypatch, hit=16)
    seq = _load_seq(942, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    _dispatch_and_fail_load(sched, seq)

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert sched.total_suppressed_load_retries == 1
    assert sched._load_specs == {}
    assert sched._lookup_client.calls == 1


def test_dense_new_sequence_reusing_request_id_gets_a_fresh_attempt(monkeypatch):
    # The mark is against the sequence, not the ID: a request ID leased to a
    # new sequence has spent nothing.
    sched = _lookup_scheduler(monkeypatch, hit=16)
    seq = _load_seq(943, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    _dispatch_and_fail_load(sched, seq)

    reused = _load_seq(943, num_prompt_tokens=24)
    assert sched.get_num_new_matched_tokens(reused) == (16, True)
    assert sched.total_suppressed_load_retries == 0


def test_dense_request_finished_releases_the_failed_load_mark(monkeypatch):
    sched = _lookup_scheduler(monkeypatch, hit=16)
    seq = _load_seq(944, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    _dispatch_and_fail_load(sched, seq)
    assert sched._load_failed_seqs == {"944": seq}

    sched.request_finished(seq)
    assert sched._load_failed_seqs == {}
