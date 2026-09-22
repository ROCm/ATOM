# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker contracts for reusable native-state LMCache MP transfers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVTransferRegion,
    KVTransferTensors,
    LoadOperationId,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload.chunked_scheduler import DENSE_PAGE_SOURCE_SAFE_CHANNEL
from atom.kv_transfer.offload.metadata import (
    LMCacheReqMeta,
    LoadSpec,
    NativeStateTransfer,
    SaveSpec,
)
from atom.kv_transfer.offload.mp.backend import _model_namespace, _tp_replication_factor
from atom.kv_transfer.offload.mp.native_state_layout import (
    build_native_state_mp_layout,
)
from atom.kv_transfer.offload.mp.native_state_worker import (
    NATIVE_STATE_MP_SOURCE_SAFE_CHANNEL,
    NATIVE_STATE_MP_STORE_CHANNEL,
    NativeStateLMCacheMPConnector,
    require_native_state_server,
)
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec


class Future:
    def __init__(self, value=True, ready=False, source_ranges=()):
        self.value, self.ready = value, ready
        self.source_ranges = list(source_ranges)

    def query(self):
        return self.ready

    def result(self, timeout=0):
        return self.value

    def take_completed_ranges(self):
        ranges, self.source_ranges = self.source_ranges, []
        return tuple(ranges)


def config(**extra):
    return SimpleNamespace(
        kv_cache_block_size=4,
        tensor_parallel_size=2,
        hf_config=SimpleNamespace(model_type="reusable_test_model", kv_lora_rank=512),
        kv_transfer_config={
            "kv_connector": "lmcache_mp",
            "kv_role": "offload",
            "kv_connector_extra_config": extra,
        },
    )


@pytest.fixture
def worker():
    instance = NativeStateLMCacheMPConnector(config())
    page = torch.zeros((32, 1, 32), dtype=torch.uint8)
    spec = PagedStateCheckpointSpec(32, 128, "native-test-v1", 80)
    tensors = KVTransferTensors(
        block_regions=[
            KVTransferRegion(
                base_addr=page.data_ptr(), unit_bytes=32, total_bytes=page.numel()
            )
        ],
        slot_regions=[],
        block_tensor_views=[page],
        paged_state_checkpoint_spec=spec,
        execute_paged_state_copies=lambda *_: None,
    )
    tensors.set_block_count(32)
    instance._native_layout = build_native_state_mp_layout(
        tensors, block_size=4, chunk_size=8
    )
    instance.chunk_size = 8
    instance.submitted = []
    instance.submitted_request_ids = []
    instance.future = Future()

    def submit(request_id, op, event):
        instance.submitted_request_ids.append(request_id)
        instance.submitted.append(op)
        return instance.future

    instance._adapter = SimpleNamespace(
        submit_store_request=submit, submit_retrieve_request=submit
    )
    return instance


def request(*, loading=False, units=(0, 25, 31), generation=1, hbm=0):
    return LMCacheReqMeta(
        req_id=7,
        token_ids=list(range(16)),
        block_ids=[1, 2, 3, 4],
        load_spec=LoadSpec(hbm, 16) if loading else None,
        save_spec=None if loading else SaveSpec(0),
        save_operation=None if loading else SaveOperationId(7, generation),
        load_operation=LoadOperationId(7, generation) if loading else None,
        native_state=NativeStateTransfer(units, 16, 998, 2 if loading else None),
    )


def test_store_transmits_page_zero_as_real_native_unit(worker):
    req = request()
    worker._submit_save(req, object())
    assert worker.submitted_request_ids == ["atom-offload-dp0:7"]
    assert worker.submitted[0].block_ids == [
        [1, 2, 3, 4],
        [-1, 0],
        [-1, 25],
        [-1, 31],
    ]
    assert not worker.get_finished().connector_completions
    worker.future.ready = True
    finished = worker.get_finished()
    assert {completion.channel for completion in finished.connector_completions} == {
        DENSE_PAGE_SOURCE_SAFE_CHANNEL,
        NATIVE_STATE_MP_SOURCE_SAFE_CHANNEL,
        NATIVE_STATE_MP_STORE_CHANNEL,
    }
    terminals = [
        completion
        for completion in finished.connector_completions
        if completion.channel == NATIVE_STATE_MP_STORE_CHANNEL
    ]
    assert terminals == [
        ConnectorCompletion(NATIVE_STATE_MP_STORE_CHANNEL, req.save_operation, True)
    ]
    page_ranges = {
        completion.operation_id.ranges
        for completion in finished.connector_completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    }
    assert page_ranges == {((0, 8),), ((8, 16),)}
    assert not finished.finished_saving  # one quorum channel for the entire pair


def test_native_state_groups_share_one_presence_mask(worker):
    req = replace(
        request(),
        token_ids=list(range(32)),
        block_ids=list(range(1, 9)),
        native_state=NativeStateTransfer((0, 25, 31), 32, 998, None),
    )

    block_groups = worker._native_block_ids(req, 0, 32, loading=False)
    state_presence = [
        [block_id != -1 for block_id in group] for group in block_groups[1:]
    ]

    assert state_presence == [[False, False, False, True]] * 3


@pytest.mark.parametrize(
    "units", [(0, -1, 31), (0, 31), (0, 0, 31), (0, 25, 32), (0, 2, 31)]
)
def test_invalid_native_source_fails_before_transport(worker, units):
    worker._submit_save(request(units=units), object())
    assert not worker.submitted
    completions = worker.get_finished().connector_completions
    [terminal] = [
        completion
        for completion in completions
        if completion.channel == NATIVE_STATE_MP_STORE_CHANNEL
    ]
    assert not terminal.succeeded


def test_uncertain_remote_submission_retains_lease(worker):
    def uncertain(*_):
        raise ConnectionError("server may have received request")

    worker._adapter.submit_store_request = uncertain
    worker._submit_save(request(), object())
    assert not worker.get_finished().connector_completions
    assert len(worker._native_saves) == 1


def test_failed_retrieve_does_not_restore_or_report_success(worker, monkeypatch):
    monkeypatch.setattr(
        worker, "_begin_restore", lambda _: pytest.fail("unexpected restore")
    )
    worker.future.value, worker.future.ready = False, True
    req = request(loading=True)
    worker._submit_load(req, object())
    finished = worker.get_finished()
    assert finished.failed_loading == {req.load_operation}
    assert not finished.finished_loading


def test_load_completion_waits_for_native_restore(worker, monkeypatch):
    restore_event = Future(ready=False)
    restored = []

    def restore(pending):
        restored.append(pending.request.native_state.destination_slot)
        pending.restore_event = restore_event
        pending.restore_succeeded = True

    monkeypatch.setattr(worker, "_begin_restore", restore)
    req = request(loading=True)
    worker._submit_load(req, object())
    assert not worker.get_finished().finished_loading
    assert restored == []
    worker.future.ready = True
    assert not worker.get_finished().finished_loading
    assert restored == [2]
    restore_event.ready = True
    assert worker.get_finished().finished_loading == {req.load_operation}
    assert restored == [2]


def test_query_exception_never_releases_dma_source(worker):
    def broken():
        raise RuntimeError("IPC event unavailable")

    worker.future.query = broken
    worker._submit_save(request(), object())
    assert not worker.get_finished().connector_completions
    assert worker._native_saves


def test_worker_admission_bound_rejects_before_dma(worker):
    worker._max_pending_saves = 1
    worker._submit_save(request(), object())
    worker._submit_save(request(generation=2), object())
    assert len(worker.submitted) == 1
    completions = worker.get_finished().connector_completions
    [terminal] = [
        completion
        for completion in completions
        if completion.channel == NATIVE_STATE_MP_STORE_CHANNEL
    ]
    assert terminal.operation_id == SaveOperationId(7, 2)
    assert not terminal.succeeded


def test_exact_completed_generation_cannot_replay(worker):
    worker.future.ready = True
    worker._submit_save(request(), object())
    worker.get_finished()
    with pytest.raises(RuntimeError, match="duplicate"):
        worker._submit_save(request(), object())


def test_native_state_uses_same_tp_rank_collapse_as_page():
    assert _tp_replication_factor(config()) == 2
    assert _tp_replication_factor(config(), native_state=True) == 2
    assert (
        _tp_replication_factor(
            config(**{"lmcache.mp.tp_rank_collapse": True}), native_state=True
        )
        == 2
    )


def test_incremental_load_transfers_only_page_suffix_but_full_native_image(worker):
    req = request(loading=True, hbm=8)
    worker._submit_load(req, object())
    [submitted] = worker.submitted
    assert submitted.start == 8
    assert submitted.end == 16
    assert submitted.block_ids == [[3, 4], [0], [25], [31]]


def test_source_safe_ranges_are_reported_before_terminal(worker):
    req = request()
    worker.future.source_ranges = [(0, 8)]
    worker._submit_save(req, object())

    first = worker.get_finished()
    assert first.connector_completions == {
        ConnectorCompletion(
            DENSE_PAGE_SOURCE_SAFE_CHANNEL,
            SaveSourceGroupId(req.save_operation, ((0, 8),)),
            True,
        )
    }
    assert worker._native_saves

    worker.future.source_ranges = [(8, 16)]
    second = worker.get_finished()
    assert (
        ConnectorCompletion(
            DENSE_PAGE_SOURCE_SAFE_CHANNEL,
            SaveSourceGroupId(req.save_operation, ((8, 16),)),
            True,
        )
        in second.connector_completions
    )
    assert (
        ConnectorCompletion(
            NATIVE_STATE_MP_SOURCE_SAFE_CHANNEL, req.save_operation, True
        )
        in second.connector_completions
    )
    assert all(
        completion.channel != NATIVE_STATE_MP_STORE_CHANNEL
        for completion in second.connector_completions
    )


def test_collapsed_tp_non_writer_skips_transport_and_reports_safe_success(worker):
    worker._is_kv_writer = False
    req = request()
    worker._submit_save(req, object())
    assert worker.submitted == []

    completions = worker.get_finished().connector_completions
    assert (
        ConnectorCompletion(
            NATIVE_STATE_MP_SOURCE_SAFE_CHANNEL, req.save_operation, True
        )
        in completions
    )
    assert (
        ConnectorCompletion(NATIVE_STATE_MP_STORE_CHANNEL, req.save_operation, True)
        in completions
    )
    assert {
        completion.operation_id.ranges
        for completion in completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((0, 8),), ((8, 16),)}


def test_restore_uses_independent_descriptor_slot_without_stream_sync(
    worker, monkeypatch
):
    from atom.kv_transfer.offload.mp import native_state_worker

    class Event:
        def __init__(self):
            self.recorded_on = None

        def record(self, stream):
            self.recorded_on = stream

    class StreamContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    restore_stream = object()
    event = Event()
    copied = []
    worker._restore_stream = restore_stream
    worker._restore_descriptor_slots = [3]
    worker._native_copy = lambda stores, restores, descriptor_slot=0: copied.append(
        (stores, restores, descriptor_slot)
    )
    monkeypatch.setattr(native_state_worker.torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(
        native_state_worker.torch.cuda, "stream", lambda stream: StreamContext()
    )

    req = request(loading=True)
    worker._submit_load(req, object())
    pending = next(iter(worker._native_loads.values()))
    assert worker._begin_restore(pending)
    assert pending.descriptor_slot == 3
    assert worker._restore_descriptor_slots == []
    assert copied[0][2] == 3
    assert event.recorded_on is restore_stream


def test_native_server_chunk_mismatch_fails_before_registration(monkeypatch):
    from atom.kv_transfer.offload.mp import native_state_worker

    monkeypatch.setattr(
        native_state_worker.offcfg,
        "build_lmcache_config",
        lambda _: SimpleNamespace(chunk_size=256),
    )
    adapter = SimpleNamespace(lmcache_tokens_per_chunk=512)
    with pytest.raises(ValueError, match="must match"):
        require_native_state_server(adapter, config())


def test_native_namespace_changes_with_image_codec(monkeypatch):
    from atom.kv_transfer.offload.mp import backend

    monkeypatch.setattr(backend.offcfg, "build_lmcache_config", lambda _: object())
    monkeypatch.setattr(backend.offcfg, "lmcache_replica_world_size", lambda _: 2)
    monkeypatch.setattr(
        backend.offcfg, "build_page_namespace", lambda *_: "page-config"
    )
    first = PagedStateCheckpointSpec(32, 128, "native-test-v1", 80)
    second = replace(first, image_bytes=81)
    assert _model_namespace(config(), checkpoint_spec=first) != _model_namespace(
        config(), checkpoint_spec=second
    )
