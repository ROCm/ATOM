# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker contracts for reusable native-state LMCache MP transfers."""

import sys
import types
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVTransferRegion,
    KVTransferTensors,
    LoadOperationId,
    PageRegion,
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
from atom.kv_transfer.offload.mp.deployment import (
    _model_namespace,
    _tp_replication_factor,
)
from atom.kv_transfer.offload.mp.native_state_layout import (
    build_native_state_mp_layout,
)
from atom.kv_transfer.offload.mp.native_state_worker import (
    NATIVE_STATE_MP_STORE_CHANNEL,
    NativeStateLMCacheMPConnector,
    require_native_state_server,
)
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec


@dataclass
class _TransferSpec:
    token_ids: list[int]
    block_ids: list[list[int]]
    start: int = 0
    end: int = 0


@pytest.fixture(autouse=True)
def fake_lmcache_transfer_spec(monkeypatch):
    """Keep transport-contract tests independent of optional LMCache installs."""
    lmcache = types.ModuleType("lmcache")
    integration = types.ModuleType("lmcache.integration")
    atom = types.ModuleType("lmcache.integration.atom")
    atom.AtomMPTransferSpec = _TransferSpec
    lmcache.integration = integration
    integration.atom = atom
    monkeypatch.setitem(sys.modules, "lmcache", lmcache)
    monkeypatch.setitem(sys.modules, "lmcache.integration", integration)
    monkeypatch.setitem(sys.modules, "lmcache.integration.atom", atom)


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
        pages=[
            PageRegion(
                KVTransferRegion(
                    base_addr=page.data_ptr(), unit_bytes=32, total_bytes=page.numel()
                ),
                page,
            )
        ],
        paged_state_checkpoint_spec=spec,
        execute_paged_state_copies=lambda stores, restores, descriptor_slot=0: None,
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


@pytest.mark.parametrize("loading", [False, True])
def test_unprovable_submission_retains_lease_until_the_deadline_stops_the_engine(
    worker, monkeypatch, loading
):
    """No clock releases a lease the server may still be using: the operation
    stays pending, and past the transfer deadline the worker fails stop."""
    from atom.kv_transfer.offload.mp import transfer

    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])

    def unprovable(*_):
        raise ConnectionError("server may have received request")

    worker._adapter.submit_store_request = unprovable
    worker._adapter.submit_retrieve_request = unprovable
    pending = worker._native_loads if loading else worker._native_saves
    if loading:
        worker._submit_load(request(loading=True), object())
    else:
        worker._submit_save(request(), object())
    now[0] += worker._transfer_deadline_s - 1
    output = worker.get_finished()
    assert not output.connector_completions
    assert not output.finished_loading and not output.failed_loading
    assert len(pending) == 1

    now[0] += 2
    with pytest.raises(transfer.LMCacheTransferUnprovable):
        worker.get_finished()
    assert len(pending) == 1


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
    assert _tp_replication_factor(config(**{"lmcache.mp.tp_rank_collapse": True})) == 2


def test_incremental_load_transfers_only_page_suffix_but_full_native_image(worker):
    req = request(loading=True, hbm=8)
    worker._submit_load(req, object())
    [submitted] = worker.submitted
    assert submitted.start == 8
    assert submitted.end == 16
    assert submitted.block_ids == [[3, 4], [0], [25], [31]]


def test_page_ranges_are_source_safe_before_terminal_but_state_is_not(worker):
    """Chunk milestones cover PAGE only: the final chunk being done says
    nothing about the STATE groups, so nothing about the image is reported
    before the STORE terminal."""
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
    assert second.connector_completions == {
        ConnectorCompletion(
            DENSE_PAGE_SOURCE_SAFE_CHANNEL,
            SaveSourceGroupId(req.save_operation, ((8, 16),)),
            True,
        )
    }


def test_collapsed_tp_non_writer_skips_transport_and_reports_safe_success(worker):
    worker._is_kv_writer = False
    req = request()
    worker._submit_save(req, object())
    assert worker.submitted == []

    completions = worker.get_finished().connector_completions
    assert (
        ConnectorCompletion(NATIVE_STATE_MP_STORE_CHANNEL, req.save_operation, True)
        in completions
    )
    assert {
        completion.operation_id.ranges
        for completion in completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((0, 8),), ((8, 16),)}


class _Event:
    def __init__(self):
        self.recorded_on = None
        self.done = False

    def record(self, stream):
        self.recorded_on = stream

    def query(self):
        return self.done


class _Stream:
    def __init__(self, name):
        self.name = name
        self.waited_streams = []
        self.waited_events = []

    def wait_stream(self, other):
        self.waited_streams.append(other)

    def wait_event(self, event):
        self.waited_events.append(event)


class _StreamContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.fixture
def cuda(worker, monkeypatch):
    """CPU stand-ins for the restore stream, the compute stream and events."""
    from atom.kv_transfer.offload.mp import native_state_worker

    stubs = SimpleNamespace(
        compute=_Stream("compute"), restore=_Stream("restore"), events=[]
    )

    def event():
        stubs.events.append(_Event())
        return stubs.events[-1]

    worker._restore_stream = stubs.restore
    monkeypatch.setattr(native_state_worker.torch.cuda, "Event", event)
    monkeypatch.setattr(
        native_state_worker.torch.cuda, "stream", lambda stream: _StreamContext()
    )
    monkeypatch.setattr(
        native_state_worker.torch.cuda, "current_stream", lambda: stubs.compute
    )
    return stubs


def test_restore_is_fenced_against_the_compute_stream_both_ways(worker, cuda):
    """The restore uses its own descriptor slot and stream, and orders against
    compute both ways: it waits for work already on the compute stream (the
    SLOT's previous occupant), and the next step's compute waits for it."""
    copied = []
    worker._restore_descriptor_slots = [3]
    worker._native_copy = lambda stores, restores, descriptor_slot=0: copied.append(
        (stores, restores, descriptor_slot)
    )

    req = request(loading=True)
    worker._submit_load(req, object())
    pending = next(iter(worker._native_loads.values()))
    assert worker._begin_restore(pending)
    [event] = cuda.events
    assert pending.descriptor_slot == 3
    assert worker._restore_descriptor_slots == []
    assert copied[0][2] == 3
    assert event.recorded_on is cuda.restore
    assert cuda.restore.waited_streams == [cuda.compute]

    worker.start_load_kv(object())
    assert cuda.compute.waited_events == [event]


def test_native_restore_reports_success_and_returns_its_slot(worker, cuda):
    """End to end through `get_finished`: a terminal retrieve starts the real
    restore with the production copy signature, and only once its event is
    done does the load finish and the descriptor slot come back."""
    copied = []
    worker._restore_descriptor_slots = [3]
    worker._native_copy = lambda stores, restores, descriptor_slot=0: copied.append(
        (stores, restores, descriptor_slot)
    )
    worker.future.value, worker.future.ready = True, True
    req = request(loading=True)
    worker._submit_load(req, object())

    first = worker.get_finished()
    assert not first.finished_loading and not first.failed_loading
    [(stores, (restore,), slot)] = copied
    assert stores == () and slot == 3
    assert restore.dst_slot == 2
    assert tuple(restore.unit_ids) == (0, 25, 31)
    assert worker._restore_descriptor_slots == []

    cuda.events[-1].done = True
    second = worker.get_finished()
    assert second.finished_loading == {req.load_operation}
    assert not second.failed_loading
    assert worker._restore_descriptor_slots == [3]
    assert worker._native_loads == {}


def test_registration_reserves_every_restore_descriptor_slot(monkeypatch):
    """Pinned staging allocation synchronizes; the restore slots are reserved
    while registering, not on the connector thread mid-serving."""
    import sys
    import types

    from atom.kv_transfer.offload.mp import native_state_layout, native_state_worker

    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: SimpleNamespace(rank_in_group=0)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)
    adapter = SimpleNamespace(
        lmcache_tokens_per_chunk=8,
        register_kv_caches=lambda tensors, engine_group_infos: None,
        shutdown=lambda: None,
    )
    monkeypatch.setattr(
        native_state_worker, "_make_worker_adapter", lambda *_a, **_k: adapter
    )
    monkeypatch.setattr(
        native_state_worker, "require_native_state_server", lambda *_a: None
    )
    monkeypatch.setattr(
        native_state_layout.NativeStateMPLayout, "engine_group_infos", lambda _s: []
    )
    monkeypatch.setattr(native_state_worker.torch.cuda, "Stream", lambda **_k: None)
    monkeypatch.setattr(native_state_worker.torch.cuda, "current_device", lambda: 0)

    reserved = []
    page = torch.zeros((32, 1, 32), dtype=torch.uint8)
    tensors = KVTransferTensors(
        pages=[
            PageRegion(
                KVTransferRegion(
                    base_addr=page.data_ptr(), unit_bytes=32, total_bytes=page.numel()
                ),
                page,
            )
        ],
        paged_state_checkpoint_spec=PagedStateCheckpointSpec(
            32, 128, "native-test-v1", 80
        ),
        execute_paged_state_copies=lambda stores, restores, descriptor_slot=0: None,
    )
    tensors.set_block_count(32)
    tensors.state_backend = SimpleNamespace(
        reserve_checkpoint_descriptors=lambda slots: reserved.extend(slots)
    )
    connector = NativeStateLMCacheMPConnector(
        config(**{"lmcache.mp.tp_rank_collapse": False})
    )
    connector.register_kv_caches({}, tensors, 32)

    assert reserved == connector._restore_descriptor_slots
    assert reserved and 0 not in reserved


def test_raising_restore_keeps_its_descriptor_slot_until_the_deadline(
    worker, monkeypatch
):
    """A restore that raises after taking a descriptor slot may have queued a
    copy that still reads the slot's staging buffer: the slot and lease stay
    held, and the transfer deadline stops the engine instead of recycling."""
    from atom.kv_transfer.offload.mp import native_state_worker, transfer

    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])
    # A real Event cannot be built on a CPU runner; fail deterministically
    # after the slot is taken instead, at the stream fence.
    monkeypatch.setattr(
        native_state_worker.torch.cuda, "Event", lambda: SimpleNamespace()
    )
    worker._restore_stream = object()  # no wait_stream: raises inside
    worker._restore_descriptor_slots = [3]
    worker.future.value, worker.future.ready = True, True
    req = request(loading=True)
    worker._submit_load(req, object())

    first = worker.get_finished()
    assert not first.finished_loading and not first.failed_loading
    assert worker._restore_descriptor_slots == []

    now[0] += worker._transfer_deadline_s
    with pytest.raises(transfer.LMCacheTransferUnprovable):
        worker.get_finished()
    assert worker._restore_descriptor_slots == []
    assert len(worker._native_loads) == 1


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
    from atom.kv_transfer.offload.mp import deployment

    monkeypatch.setattr(deployment.offcfg, "build_lmcache_config", lambda _: object())
    monkeypatch.setattr(deployment.offcfg, "lmcache_replica_world_size", lambda _: 2)
    monkeypatch.setattr(
        deployment.offcfg, "build_page_namespace", lambda *_: "page-config"
    )
    first = PagedStateCheckpointSpec(32, 128, "native-test-v1", 80)
    second = replace(first, image_bytes=81)
    assert _model_namespace(config(), checkpoint_spec=first) != _model_namespace(
        config(), checkpoint_spec=second
    )
