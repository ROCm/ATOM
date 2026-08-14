# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the composite ``multi`` KV connector.

Pure-Python: sub-connectors are mocked, so no GPU / lmcache / moriio runtime is
needed. Covers the merge strategy (first-hit-wins, fan-out, metadata routing,
completion union) and the send/save pairing that protects a producer node's
blocks from being freed while a transfer is still reading them.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnector,
    MultiConnectorMetadata,
    MultiConnectorScheduler,
    MultiSaveOperationId,
)
from atom.kv_transfer.disaggregation.types import (
    STATE_CHECKPOINT_STAGING_CHANNEL,
    ConnectorCompletion,
    ConnectorMetadata,
    KVConnectorOutput,
    LoadOperationId,
    SaveOperationId,
    SendOperationId,
)
from atom.kv_transfer.offload.hybrid.dsv4.connector import (
    DSV4_CHECKPOINT_SAVE_CHANNEL,
)
from atom.model_engine.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Mock sub-connectors
# ---------------------------------------------------------------------------


class FakeSchedSub:
    """Scheduler-side sub-connector mock."""

    def __init__(
        self,
        *,
        match=(0, False),
        is_producer=False,
        is_offload=False,
        offload_methods=False,
        completion_channels=(),
    ):
        self._match = match
        self.is_producer = is_producer
        if is_offload:
            self.is_offload = True
        self.completion_channels = frozenset(completion_channels)
        self.alloc_calls = []
        self.finished_calls = []
        self.completion_calls = []
        self.meta = ConnectorMetadata()
        self._offload = offload_methods

        if offload_methods:
            self.park = False
            self.partial_park = False
            self.defer = False
            self.chunk_ret = None
            self.saved = []
            self.load_finished_ids = []
            self.load_failed_ids = []

    def get_num_new_matched_tokens(self, seq):
        return self._match

    def build_connector_meta(self):
        return self.meta

    def update_state_after_alloc(self, seq):
        self.alloc_calls.append(seq)

    def request_finished(self, seq):
        self.finished_calls.append(seq)

    # offload-specific (only present when offload_methods=True)
    def should_park_for_load_after_alloc(self, seq):
        return self.park

    def adjust_prefill_chunk_after_alloc(self, seq, chunk):
        return self.chunk_ret if self.chunk_ret is not None else chunk

    def should_park_partial_prefill_for_load(self, seq):
        return self.partial_park

    def should_defer_free(self, seq):
        return self.defer

    def save_finished(self, req_id):
        self.saved.append(req_id)

    def connector_completion(self, completion):
        self.completion_calls.append(completion)
        return True

    def load_failed(self, req_id):
        self.load_failed_ids.append(req_id)

    def load_finished(self, req_id):
        self.load_finished_ids.append(req_id)

    def __getattribute__(self, name):
        # Hide offload-specific methods unless this mock opts in, so
        # MultiConnector's hasattr() guards are exercised realistically.
        offload_api = {
            "should_park_for_load_after_alloc",
            "adjust_prefill_chunk_after_alloc",
            "should_park_partial_prefill_for_load",
            "should_defer_free",
            "save_finished",
            "load_finished",
            "load_failed",
        }
        if name in offload_api and not object.__getattribute__(self, "_offload"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)


class FakeWorkerSub:
    """Worker-side sub-connector mock."""

    def __init__(
        self,
        *,
        is_producer=False,
        finished=None,
        recv_blocks=None,
        completion_channels=(),
    ):
        self.is_producer = is_producer
        self.completion_channels = frozenset(completion_channels)
        self._finished = finished if finished is not None else KVConnectorOutput()
        self._recv_blocks = recv_blocks or []
        self.registered = None
        self.loaded_meta = None

    def register_kv_caches(self, kv_caches, transfer_tensors=None, num_blocks=None):
        self.registered = (kv_caches, transfer_tensors, num_blocks)

    def start_load_kv(self, metadata):
        self.loaded_meta = metadata

    def get_finished(self):
        return self._finished

    def get_finished_recv_blocks(self):
        return self._recv_blocks


class FakeCheckpointWorkerSub(FakeWorkerSub):
    """Worker mock that consumes native checkpoint records via the new hook."""

    state_checkpoint_completion_channel = STATE_CHECKPOINT_STAGING_CHANNEL

    def __init__(self, **kwargs):
        super().__init__(
            completion_channels={STATE_CHECKPOINT_STAGING_CHANNEL},
            **kwargs,
        )

    def start_load_kv_with_state_checkpoints(self, metadata, copies):
        self.loaded_meta = metadata
        self.checkpoint_copies = copies


def _sched(connectors):
    with patch(
        "atom.kv_transfer.disaggregation.multi.multi_connector._build_subconnectors",
        return_value=connectors,
    ):
        return MultiConnectorScheduler(SimpleNamespace())


def _worker(connectors):
    with patch(
        "atom.kv_transfer.disaggregation.multi.multi_connector._build_subconnectors",
        return_value=connectors,
    ):
        return MultiConnector(SimpleNamespace())


def _save_meta(*req_ids):
    """Metadata opting its raw request IDs into async save pairing."""
    meta = ConnectorMetadata()
    meta.requests = [
        SimpleNamespace(
            req_id=r,
            save_spec=object(),
            slot_save_spec=None,
            load_spec=None,
        )
        for r in req_ids
    ]
    meta.iter_async_save_operations = lambda: tuple((r, r) for r in req_ids)
    return meta


def _sidecar_save_meta(*req_ids):
    meta = ConnectorMetadata()
    meta.requests = [
        SimpleNamespace(
            req_id=r,
            save_spec=None,
            slot_save_spec=object(),
            load_spec=None,
        )
        for r in req_ids
    ]
    meta.iter_async_save_operations = lambda: tuple((r, r) for r in req_ids)
    return meta


def _operation_save_meta(operation, *, page=True, sidecar=False):
    meta = ConnectorMetadata()
    meta.requests = [
        SimpleNamespace(
            req_id=operation.req_id,
            save_spec=object() if page else None,
            slot_save_spec=object() if sidecar else None,
            save_operation=operation,
            load_spec=None,
        )
    ]
    meta.iter_async_save_operations = lambda: ((operation.req_id, operation),)
    return meta


def _multi_save(operation: SaveOperationId, connector_idx: int):
    return MultiSaveOperationId(
        operation.req_id,
        operation.generation,
        connector_idx,
    )


def _checkpoint_copy(
    *,
    copy_id=41,
    req_id=9,
    boundary_tokens=8192,
    boundary_block_hash=0x1234,
    source_group=2,
    destination_group=7,
):
    return SimpleNamespace(
        copy_id=copy_id,
        request_id=req_id,
        boundary_tokens=boundary_tokens,
        boundary_block_hash=boundary_block_hash,
        source_group=source_group,
        destination_group=destination_group,
    )


def _checkpoint_save_meta(copy_record):
    meta = ConnectorMetadata()
    meta.state_checkpoint_completion_channel = STATE_CHECKPOINT_STAGING_CHANNEL
    meta.requests = [
        SimpleNamespace(
            req_id=copy_record.request_id,
            save_spec=None,
            slot_save_spec=SimpleNamespace(
                boundary_tokens=copy_record.boundary_tokens,
                boundary_block_hash=copy_record.boundary_block_hash,
                source_group=copy_record.source_group,
            ),
            load_spec=None,
        )
    ]
    meta.iter_async_save_operations = lambda: (
        (copy_record.request_id, copy_record.request_id),
    )
    return meta


def _completion(channel, operation_id, *, succeeded=True):
    return ConnectorCompletion(
        channel=channel,
        operation_id=operation_id,
        succeeded=succeeded,
    )


# ---------------------------------------------------------------------------
# Scheduler-side
# ---------------------------------------------------------------------------


def test_matched_tokens_first_hit_wins():
    a = FakeSchedSub(match=(0, False))
    b = FakeSchedSub(match=(5, True))
    sched = _sched([a, b])
    assert sched.get_num_new_matched_tokens(object()) == (5, True)


def test_matched_tokens_earlier_connector_wins_over_later():
    a = FakeSchedSub(match=(3, True))
    b = FakeSchedSub(match=(5, True))
    sched = _sched([a, b])
    assert sched.get_num_new_matched_tokens(object()) == (3, True)


def test_first_hit_exclusively_owns_load_metadata_and_terminal():
    seq = SimpleNamespace(id=41)

    class _LoadSub:
        is_producer = False
        is_offload = True

        def __init__(self, index, *, stale=False, reactivate_on_update=False):
            self.index = index
            self.queries = 0
            self.pending = stale
            self.reactivate_on_update = reactivate_on_update
            self.cancelled = []
            self.terminals = []
            self.operation = LoadOperationId(seq.id, index)

        def get_num_new_matched_tokens(self, _seq):
            self.queries += 1
            self.pending = True
            return 8, True

        def update_state_after_alloc(self, _seq):
            if self.reactivate_on_update:
                self.pending = True

        def cancel_pending_load(self, value):
            self.cancelled.append(value)
            self.pending = False

        def build_connector_meta(self):
            meta = ConnectorMetadata()
            if self.pending:
                seq._load_operation = self.operation
            meta.requests = (
                [
                    SimpleNamespace(
                        req_id=seq.id,
                        load_spec=object(),
                        slot_load_spec=None,
                        load_operation=self.operation,
                    )
                ]
                if self.pending
                else []
            )
            return meta

        def should_park_for_load_after_alloc(self, _seq):
            return self.pending

        def load_finished(self, operation):
            self.terminals.append(operation)
            return operation == self.operation

        def request_finished(self, _seq):
            self.pending = False

    first = _LoadSub(0)
    second = _LoadSub(1, stale=True, reactivate_on_update=True)
    sched = _sched([first, second])

    assert sched.get_num_new_matched_tokens(seq) == (8, True)
    assert first.queries == 1
    assert second.queries == 0

    sched.update_state_after_alloc(seq)
    meta = sched.build_connector_meta()

    assert len(meta.metas[0].requests) == 1
    assert all(
        req.load_spec is None and req.slot_load_spec is None
        for req in meta.metas[1].requests
    )
    assert second.cancelled == [seq, seq]
    assert seq._load_operation == first.operation

    class _LoadWorker(FakeWorkerSub):
        def __init__(self):
            super().__init__()
            self.writes = []

        def start_load_kv(self, sub_meta):
            self.writes.extend(
                req.load_operation
                for req in getattr(sub_meta, "requests", ())
                if req.load_spec is not None or req.slot_load_spec is not None
            )

    winner_worker = _LoadWorker()
    loser_worker = _LoadWorker()
    worker = _worker([winner_worker, loser_worker])
    worker.start_load_kv(meta)

    assert winner_worker.writes == [first.operation]
    assert loser_worker.writes == []
    assert sched.load_finished(first.operation) is True
    assert first.terminals == [first.operation]
    assert second.terminals == []


def test_reused_request_id_gets_fresh_load_owner():
    first = FakeSchedSub(match=(4, True), is_offload=True, offload_methods=True)
    second = FakeSchedSub(match=(0, False), is_offload=True, offload_methods=True)
    sched = _sched([first, second])
    old = SimpleNamespace(id=52)

    assert sched.get_num_new_matched_tokens(old) == (4, True)
    sched.request_finished(old)

    first._match = (0, False)
    second._match = (6, True)
    new = SimpleNamespace(id=52)

    assert sched.get_num_new_matched_tokens(new) == (6, True)
    second.park = True
    assert sched.should_park_for_load_after_alloc(new) is True


@pytest.mark.parametrize("offload_first", [True, False])
def test_heterogeneous_multi_filters_loser_load_metadata(offload_first):
    seq = SimpleNamespace(id=53)

    class _GenericPD:
        is_producer = False
        is_offload = False

        def __init__(self, hit):
            self.hit = hit
            self.pending = False

        def get_num_new_matched_tokens(self, _seq):
            self.pending = True
            return (4, True) if self.hit else (0, False)

        def update_state_after_alloc(self, _seq):
            self.pending = True

        def build_connector_meta(self):
            meta = ConnectorMetadata()
            if self.pending:
                meta.reqs_to_recv[seq.id] = object()
                meta.reqs_not_processed.add(seq.id)
            return meta

        def request_finished(self, _seq):
            self.pending = False

    class _Offload:
        is_producer = False
        is_offload = True

        def __init__(self, hit):
            self.hit = hit
            self.pending = False
            self.operation = LoadOperationId(seq.id, 20)

        def get_num_new_matched_tokens(self, _seq):
            self.pending = True
            return (8, True) if self.hit else (0, False)

        def update_state_after_alloc(self, _seq):
            self.pending = True

        def cancel_pending_load(self, _seq):
            self.pending = False

        def build_connector_meta(self):
            meta = ConnectorMetadata()
            meta.requests = (
                [
                    SimpleNamespace(
                        req_id=seq.id,
                        load_spec=object(),
                        slot_load_spec=None,
                        load_operation=self.operation,
                    )
                ]
                if self.pending
                else []
            )
            return meta

        def request_finished(self, _seq):
            self.pending = False

    pd = _GenericPD(hit=not offload_first)
    offload = _Offload(hit=offload_first)
    connectors = [offload, pd] if offload_first else [pd, offload]
    sched = _sched(connectors)

    expected = (8, True) if offload_first else (4, True)
    assert sched.get_num_new_matched_tokens(seq) == expected
    sched.update_state_after_alloc(seq)
    meta = sched.build_connector_meta()

    pd_meta = meta.metas[connectors.index(pd)]
    offload_meta = meta.metas[connectors.index(offload)]

    class _Worker(FakeWorkerSub):
        def __init__(self):
            super().__init__()
            self.writes = []

        def start_load_kv(self, sub_meta):
            self.writes.extend(getattr(sub_meta, "reqs_to_recv", {}))
            self.writes.extend(
                req.req_id
                for req in getattr(sub_meta, "requests", ())
                if req.load_spec is not None or req.slot_load_spec is not None
            )

    workers = [_Worker(), _Worker()]
    worker = _worker(workers)
    worker.start_load_kv(meta)
    pd_worker = workers[connectors.index(pd)]
    offload_worker = workers[connectors.index(offload)]
    if offload_first:
        assert pd_meta.reqs_to_recv == {}
        assert pd_meta.reqs_not_processed == set()
        assert len(offload_meta.requests) == 1
        assert pd_worker.writes == []
        assert offload_worker.writes == [seq.id]
        assert seq._remote_load_is_offload is True
    else:
        assert list(pd_meta.reqs_to_recv) == [seq.id]
        assert all(req.load_spec is None for req in offload_meta.requests)
        assert pd_worker.writes == [seq.id]
        assert offload_worker.writes == []
        assert seq._remote_load_is_offload is False


def test_no_match_returns_zero():
    sched = _sched([FakeSchedSub(), FakeSchedSub()])
    assert sched.get_num_new_matched_tokens(object()) == (0, False)


def test_update_and_finished_fan_out_to_all():
    a, b = FakeSchedSub(), FakeSchedSub()
    sched = _sched([a, b])
    seq = object()
    sched.update_state_after_alloc(seq)
    sched.request_finished(seq)
    assert a.alloc_calls == [seq] and b.alloc_calls == [seq]
    assert a.finished_calls == [seq] and b.finished_calls == [seq]


def test_build_connector_meta_wraps_subs_in_order():
    a, b = FakeSchedSub(), FakeSchedSub()
    sched = _sched([a, b])
    meta = sched.build_connector_meta()
    assert isinstance(meta, MultiConnectorMetadata)
    assert meta.metas == [a.meta, b.meta]
    assert meta.completion_channel_owners == {}


def test_build_connector_meta_records_completion_channel_owner():
    unrelated = FakeSchedSub()
    owner = FakeSchedSub(
        completion_channels={STATE_CHECKPOINT_STAGING_CHANNEL},
    )
    owner.meta.requests = [object()]
    sched = _sched([unrelated, owner])

    meta = sched.build_connector_meta()

    assert meta.completion_channel_owners == {
        STATE_CHECKPOINT_STAGING_CHANNEL: 1,
    }
    assert (
        meta.requests_for_completion_channel(STATE_CHECKPOINT_STAGING_CHANNEL)
        == owner.meta.requests
    )
    assert meta.requests_for_completion_channel("unknown") == []


def test_multi_metadata_delegates_checkpoint_selection_to_protocol_owner():
    class _CheckpointMetadata(ConnectorMetadata):
        state_checkpoint_completion_channel = STATE_CHECKPOINT_STAGING_CHANNEL

        def select_state_checkpoint_copies(self, checkpoints):
            return checkpoints[1:]

    checkpoints = [object(), object()]
    meta = MultiConnectorMetadata(
        [ConnectorMetadata(), _CheckpointMetadata()],
        completion_channel_owners={STATE_CHECKPOINT_STAGING_CHANNEL: 1},
    )

    assert meta.state_checkpoint_owner == 1
    assert meta.select_state_checkpoint_copies(checkpoints) == checkpoints[1:]


def test_scheduler_rejects_duplicate_completion_channel_owners():
    first = FakeSchedSub(
        completion_channels={STATE_CHECKPOINT_STAGING_CHANNEL},
    )
    second = FakeSchedSub(
        completion_channels={STATE_CHECKPOINT_STAGING_CHANNEL},
    )
    with pytest.raises(ValueError, match="must have one owner"):
        _sched([first, second])


def test_role_attrs_aggregate():
    sched = _sched(
        [
            FakeSchedSub(is_producer=True),
            FakeSchedSub(is_offload=True, offload_methods=True),
        ]
    )
    assert sched.is_producer is True
    assert sched.is_offload is True


def test_offload_and_completion_methods_forwarded_to_owning_sub():
    moriio = FakeSchedSub(is_producer=True)  # no offload methods
    off = FakeSchedSub(
        match=(5, True),
        is_offload=True,
        offload_methods=True,
        completion_channels={DSV4_CHECKPOINT_SAVE_CHANNEL},
    )
    off.park = True
    off.partial_park = True
    off.should_pause_partial_prefill_for_save = lambda _seq: True
    off.defer = True
    off.chunk_ret = 7
    sched = _sched([moriio, off])
    seq = SimpleNamespace(id="r1")
    assert sched.get_num_new_matched_tokens(seq) == (5, True)
    assert sched.should_park_for_load_after_alloc(seq) is True
    assert sched.should_park_partial_prefill_for_load(seq) is True
    assert sched.should_pause_partial_prefill_for_save(seq) is True
    assert sched.should_defer_free(seq) is True
    assert sched.adjust_prefill_chunk_after_alloc(seq, 10) == 7
    sched.save_finished("r1")
    succeeded = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, "r2")
    failed = _completion(
        DSV4_CHECKPOINT_SAVE_CHANNEL,
        "r3",
        succeeded=False,
    )
    assert sched.connector_completion(succeeded) is True
    assert sched.connector_completion(failed) is True
    assert sched.connector_completion(_completion("unknown", "r4")) is False
    sched.load_finished("r1")
    seq2 = SimpleNamespace(id="r2")
    assert sched.get_num_new_matched_tokens(seq2) == (5, True)
    sched.load_failed("r2")
    assert off.saved == ["r1"]
    assert moriio.completion_calls == []
    assert off.completion_calls == [succeeded, failed]
    assert off.load_finished_ids == ["r1"]
    assert off.load_failed_ids == ["r2"]


def test_offload_methods_default_when_no_sub_implements():
    sched = _sched([FakeSchedSub(is_producer=True), FakeSchedSub()])
    seq = object()
    assert sched.should_park_for_load_after_alloc(seq) is False
    assert sched.should_park_partial_prefill_for_load(seq) is False
    assert sched.should_pause_partial_prefill_for_save(seq) is False
    assert sched.should_defer_free(seq) is False
    assert sched.adjust_prefill_chunk_after_alloc(seq, 10) == 10  # unchanged
    assert sched.connector_completion(_completion("unknown", "r1")) is False


# ---------------------------------------------------------------------------
# Worker-side
# ---------------------------------------------------------------------------


def test_register_kv_caches_fans_out():
    a, b = FakeWorkerSub(), FakeWorkerSub()
    w = _worker([a, b])
    kv = {"layer_0": object()}
    w.register_kv_caches(kv, transfer_tensors="tt", num_blocks=42)
    assert a.registered == (kv, "tt", 42)
    assert b.registered == (kv, "tt", 42)


def test_start_load_kv_routes_by_index_and_records_saves():
    a, b = FakeWorkerSub(is_producer=True), FakeWorkerSub()
    w = _worker([a, b])
    m0 = ConnectorMetadata()  # moriio sub-meta (no .requests)
    m1 = _save_meta(101, 102)  # offload sub-meta with two saves
    w.start_load_kv(MultiConnectorMetadata([m0, m1]))
    assert a.loaded_meta is m0
    assert b.loaded_meta is m1
    assert w.pairing_state_count == (2, 0)


def test_worker_rejects_duplicate_completion_channel_owners():
    with pytest.raises(ValueError, match="must have one owner"):
        _worker(
            [
                FakeCheckpointWorkerSub(),
                FakeCheckpointWorkerSub(),
            ]
        )


def test_worker_rejects_scheduler_worker_completion_owner_mismatch():
    worker = _worker([FakeWorkerSub(), FakeCheckpointWorkerSub()])
    meta = MultiConnectorMetadata(
        [ConnectorMetadata(), ConnectorMetadata()],
        completion_channel_owners={STATE_CHECKPOINT_STAGING_CHANNEL: 0},
    )

    with pytest.raises(RuntimeError, match="ownership differs"):
        worker.start_load_kv(meta)


def test_worker_rejects_checkpoint_channel_mismatch_with_same_owner():
    scheduler_channel = "test.scheduler.checkpoint"
    worker_channel = "test.worker.checkpoint"
    owner = FakeCheckpointWorkerSub()
    owner.completion_channels = frozenset({scheduler_channel, worker_channel})
    owner.state_checkpoint_completion_channel = worker_channel
    worker = _worker([owner])
    owner_meta = ConnectorMetadata()
    owner_meta.state_checkpoint_completion_channel = scheduler_channel
    meta = MultiConnectorMetadata(
        [owner_meta],
        completion_channel_owners={
            scheduler_channel: 0,
            worker_channel: 0,
        },
    )

    with pytest.raises(RuntimeError, match="channel differs"):
        worker.start_load_kv(meta)


def test_get_finished_unions_and_normalizes_tuple():
    # moriio returns a legacy tuple; offload returns KVConnectorOutput.
    moriio = FakeWorkerSub(finished=(set(), {"d1"}))  # recving d1
    off = FakeWorkerSub(
        finished=KVConnectorOutput(finished_recving={"d2"}, failed_recving={"f1"})
    )
    w = _worker([moriio, off])  # not producer
    out = w.get_finished()
    assert out.finished_recving == {"d1", "d2"}
    assert out.failed_recving == {"f1"}


def test_get_finished_ors_child_pending_work():
    worker = _worker(
        [
            FakeWorkerSub(finished=KVConnectorOutput()),
            FakeWorkerSub(finished=KVConnectorOutput(pending_work=True)),
        ]
    )

    assert worker.get_finished().pending_work is True


def test_has_pending_work_ors_child_liveness_hooks():
    first = FakeWorkerSub()
    second = FakeWorkerSub()
    first.has_pending_work = lambda: False
    second.has_pending_work = lambda: True
    worker = _worker([first, second])

    assert worker.has_pending_work() is True


def test_producer_offload_load_completion_uses_loading_state():
    moriio = FakeWorkerSub(is_producer=True, finished=(set(), set()))
    off = FakeWorkerSub(
        finished=KVConnectorOutput(finished_loading={"l1"}, failed_loading={"f1"})
    )
    w = _worker([moriio, off])

    out = w.get_finished()

    assert out.finished_recving == set()
    assert out.failed_recving == set()
    assert out.finished_loading == {"l1"}
    assert out.failed_loading == {"f1"}


def test_load_completion_preserves_exact_generation():
    operation = LoadOperationId("l1", 7)
    worker = _worker(
        [FakeWorkerSub(finished=KVConnectorOutput(finished_loading={operation}))]
    )

    out = worker.get_finished()

    assert out.finished_loading == {operation}


def test_recv_blocks_concat():
    w = _worker([FakeWorkerSub(recv_blocks=[1, 2]), FakeWorkerSub(recv_blocks=[3])])
    assert w.get_finished_recv_blocks() == [1, 2, 3]


def test_non_producer_passes_saving_through():
    off = FakeWorkerSub(finished=KVConnectorOutput(finished_saving={"s1"}))
    w = _worker([off])  # is_producer False
    out = w.get_finished()
    assert out.finished_saving == {"s1"}


def test_connector_completions_bypass_producer_send_save_pairing():
    succeeded = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, 9)
    failed = _completion(
        DSV4_CHECKPOINT_SAVE_CHANNEL,
        10,
        succeeded=False,
    )
    moriio = FakeWorkerSub(is_producer=True, finished=({9}, set()))
    off = FakeWorkerSub(
        finished=KVConnectorOutput(connector_completions={succeeded, failed}),
        completion_channels={DSV4_CHECKPOINT_SAVE_CHANNEL},
    )
    w = _worker([moriio, off])
    w.start_load_kv(
        MultiConnectorMetadata(
            [ConnectorMetadata(), _save_meta(9)],
            completion_channel_owners={DSV4_CHECKPOINT_SAVE_CHANNEL: 1},
        )
    )

    out = w.get_finished()

    assert out.finished_sending == set()
    assert out.finished_saving == set()
    assert out.connector_completions == {succeeded, failed}


def test_checkpoint_staging_completions_bypass_send_save_pairing():
    staged = _completion(STATE_CHECKPOINT_STAGING_CHANNEL, 41)
    aborted = _completion(
        STATE_CHECKPOINT_STAGING_CHANNEL,
        42,
        succeeded=False,
    )
    moriio = FakeWorkerSub(is_producer=True, finished=({9}, set()))
    off = FakeCheckpointWorkerSub(
        finished=KVConnectorOutput(connector_completions={staged, aborted}),
    )
    w = _worker([moriio, off])
    off_meta = _save_meta(9)
    off_meta.state_checkpoint_completion_channel = STATE_CHECKPOINT_STAGING_CHANNEL
    w.start_load_kv(
        MultiConnectorMetadata(
            [ConnectorMetadata(), off_meta],
            completion_channel_owners={STATE_CHECKPOINT_STAGING_CHANNEL: 1},
        )
    )

    out = w.get_finished()

    assert out.finished_sending == set()
    assert out.connector_completions == {staged, aborted}


def test_completion_from_non_owner_child_is_dropped():
    completion = _completion(STATE_CHECKPOINT_STAGING_CHANNEL, 51)
    non_owner = FakeWorkerSub(
        finished=KVConnectorOutput(connector_completions={completion})
    )
    owner = FakeCheckpointWorkerSub()
    worker = _worker([non_owner, owner])

    assert worker.get_finished().connector_completions == set()

    non_owner._finished = KVConnectorOutput()
    owner._finished = KVConnectorOutput(connector_completions={completion})

    assert worker.get_finished().connector_completions == {completion}


def test_checkpoint_staging_owner_completion_is_forwarded_immediately():
    checkpoint = _checkpoint_copy(copy_id=53)
    completion = _completion(STATE_CHECKPOINT_STAGING_CHANNEL, 53)
    producer = FakeWorkerSub(is_producer=True)
    offload = FakeCheckpointWorkerSub()
    worker = _worker([producer, offload])
    meta = MultiConnectorMetadata(
        [ConnectorMetadata(), _checkpoint_save_meta(checkpoint)],
        completion_channel_owners={STATE_CHECKPOINT_STAGING_CHANNEL: 1},
    )
    meta.state_checkpoint_copies = [checkpoint]
    worker.start_load_kv(meta)

    offload._finished = KVConnectorOutput(connector_completions={completion})
    terminal = worker.get_finished()

    assert terminal.connector_completions == {completion}


def test_send_without_pending_save_is_released_immediately():
    moriio = FakeWorkerSub(is_producer=True, finished=({"r1"}, set()))
    w = _worker([moriio])
    out = w.get_finished()
    assert out.finished_sending == {"r1"}


def test_send_is_withheld_until_save_completes():
    # One producer (moriio) + one offload sub, sharing req "r9".
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub()
    w = _worker([moriio, off])

    # offload will save r9
    w.start_load_kv(MultiConnectorMetadata([ConnectorMetadata(), _save_meta(9)]))
    assert w.pairing_state_count == (1, 0)

    # Step 1: moriio reports send done, offload's save still in flight.
    moriio._finished = ({9}, set())
    off._finished = KVConnectorOutput()
    out1 = w.get_finished()
    assert out1.finished_sending == set()  # withheld
    assert out1.finished_saving == set()

    # Step 2: offload reports save done -> both released together.
    moriio._finished = (set(), set())
    off._finished = KVConnectorOutput(finished_saving={9})
    out2 = w.get_finished()
    assert out2.finished_sending == {9}
    assert out2.finished_saving == {9}
    assert w.pairing_state_count == (0, 0)


def test_page_then_sidecar_only_save_keeps_send_paired_until_both_finish():
    operation = SaveOperationId(9, 1)
    sidecar_completion = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, operation)
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub(
        completion_channels={DSV4_CHECKPOINT_SAVE_CHANNEL},
    )
    w = _worker([moriio, off])
    w.start_load_kv(
        MultiConnectorMetadata(
            [ConnectorMetadata(), _operation_save_meta(operation, page=True)],
            completion_channel_owners={DSV4_CHECKPOINT_SAVE_CHANNEL: 1},
        )
    )
    w.start_load_kv(
        MultiConnectorMetadata(
            [
                ConnectorMetadata(),
                _operation_save_meta(operation, page=False, sidecar=True),
            ],
            completion_channel_owners={DSV4_CHECKPOINT_SAVE_CHANNEL: 1},
        )
    )

    assert w.pairing_state_count == (1, 0)

    moriio._finished = ({9}, set())
    off._finished = KVConnectorOutput(connector_completions={sidecar_completion})
    page_done = w.get_finished()

    assert page_done.finished_sending == set()
    assert page_done.finished_saving == set()
    assert page_done.connector_completions == {sidecar_completion}

    moriio._finished = (set(), set())
    off._finished = KVConnectorOutput(finished_saving={operation})
    sidecar_done = w.get_finished()

    assert sidecar_done.finished_sending == {9}
    assert sidecar_done.finished_saving == {_multi_save(operation, 1)}
    assert w.pairing_state_count == (0, 0)


def test_generation_saves_pair_send_only_after_every_exact_operation():
    gen0 = SaveOperationId(9, 0)
    gen1 = SaveOperationId(9, 1)
    failed_sidecar = _completion(
        DSV4_CHECKPOINT_SAVE_CHANNEL,
        gen1,
        succeeded=False,
    )
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub(
        completion_channels={DSV4_CHECKPOINT_SAVE_CHANNEL},
    )
    w = _worker([moriio, off])
    w.start_load_kv(
        MultiConnectorMetadata(
            [ConnectorMetadata(), _operation_save_meta(gen0, page=True)],
            completion_channel_owners={DSV4_CHECKPOINT_SAVE_CHANNEL: 1},
        )
    )
    w.start_load_kv(
        MultiConnectorMetadata(
            [
                ConnectorMetadata(),
                _operation_save_meta(gen1, page=True, sidecar=True),
            ],
            completion_channel_owners={DSV4_CHECKPOINT_SAVE_CHANNEL: 1},
        )
    )

    moriio._finished = ({9}, set())
    off._finished = KVConnectorOutput(finished_saving={gen0})
    first = w.get_finished()

    assert first.finished_sending == set()
    assert first.finished_saving == {_multi_save(gen0, 1)}

    moriio._finished = (set(), set())
    off._finished = KVConnectorOutput(
        finished_saving={gen1},
        connector_completions={failed_sidecar},
    )
    terminal = w.get_finished()

    assert terminal.finished_sending == {9}
    assert terminal.finished_saving == {_multi_save(gen1, 1)}
    assert terminal.connector_completions == {failed_sidecar}


def test_exact_send_generation_survives_multi_save_pairing():
    save = SaveOperationId(9, 8)
    send = SendOperationId(9, 4)
    moriio = FakeWorkerSub(is_producer=True)
    offload = FakeWorkerSub()
    worker = _worker([moriio, offload])
    worker.start_load_kv(
        MultiConnectorMetadata(
            [ConnectorMetadata(), _operation_save_meta(save, page=True)]
        )
    )

    moriio._finished = KVConnectorOutput(finished_sending={send})
    assert worker.get_finished().finished_sending == set()

    moriio._finished = KVConnectorOutput()
    offload._finished = KVConnectorOutput(finished_saving={save})
    terminal = worker.get_finished()

    assert terminal.finished_sending == {send}
    assert terminal.finished_saving == {_multi_save(save, 1)}


def test_late_exact_send_generation_cannot_overwrite_current_generation():
    save = SaveOperationId(9, 9)
    stale_send = SendOperationId(9, 3)
    current_send = SendOperationId(9, 4)
    producer = FakeWorkerSub(is_producer=True)
    saver = FakeWorkerSub()
    worker = _worker([producer, saver])
    worker.start_load_kv(
        MultiConnectorMetadata(
            [ConnectorMetadata(), _operation_save_meta(save, page=True)]
        )
    )

    producer._finished = KVConnectorOutput(finished_sending={current_send})
    assert worker.get_finished().finished_sending == set()

    # The older notification arrives later.  The former scalar table replaced
    # current_send here and released only stale_send when the save completed.
    producer._finished = KVConnectorOutput(finished_sending={stale_send})
    assert worker.get_finished().finished_sending == set()

    producer._finished = KVConnectorOutput()
    saver._finished = KVConnectorOutput(finished_saving={save})
    terminal = worker.get_finished()

    assert terminal.finished_sending == {stale_send, current_send}
    assert terminal.finished_saving == {_multi_save(save, 1)}


def test_exact_held_send_suppresses_legacy_raw_id_for_same_request():
    save = SaveOperationId(9, 10)
    exact_send = SendOperationId(9, 5)
    producer = FakeWorkerSub(is_producer=True)
    saver = FakeWorkerSub()
    worker = _worker([producer, saver])
    worker.start_load_kv(
        MultiConnectorMetadata(
            [ConnectorMetadata(), _operation_save_meta(save, page=True)]
        )
    )

    producer._finished = KVConnectorOutput(finished_sending={9})
    assert worker.get_finished().finished_sending == set()
    producer._finished = KVConnectorOutput(finished_sending={exact_send})
    assert worker.get_finished().finished_sending == set()
    producer._finished = KVConnectorOutput(finished_sending={9})
    assert worker.get_finished().finished_sending == set()

    producer._finished = KVConnectorOutput()
    saver._finished = KVConnectorOutput(finished_saving={save})

    assert worker.get_finished().finished_sending == {exact_send}


def test_two_saving_connectors_emit_child_namespaced_operations():
    operation = SaveOperationId(9, 20)
    first = FakeWorkerSub()
    second = FakeWorkerSub()
    worker = _worker([first, second])
    worker.start_load_kv(
        MultiConnectorMetadata(
            [
                _operation_save_meta(operation),
                _operation_save_meta(operation),
            ]
        )
    )

    first._finished = KVConnectorOutput(finished_saving={operation})
    assert worker.get_finished().finished_saving == {_multi_save(operation, 0)}

    assert worker.get_finished().finished_saving == set()

    first._finished = KVConnectorOutput()
    second._finished = KVConnectorOutput(finished_saving={operation})
    assert worker.get_finished().finished_saving == {_multi_save(operation, 1)}


def test_completed_generation_in_one_child_does_not_tombstone_another_child():
    operation = SaveOperationId(9, 0)
    first = FakeWorkerSub()
    second = FakeWorkerSub()
    worker = _worker([first, second])

    worker.start_load_kv(
        MultiConnectorMetadata([_operation_save_meta(operation), None])
    )
    first._finished = KVConnectorOutput(finished_saving={operation})
    assert worker.get_finished().finished_saving == {_multi_save(operation, 0)}

    first._finished = KVConnectorOutput()
    worker.start_load_kv(
        MultiConnectorMetadata([None, _operation_save_meta(operation)])
    )
    assert worker.pairing_state_count == (1, 0)
    second._finished = KVConnectorOutput(finished_saving={operation})

    assert worker.get_finished().finished_saving == {_multi_save(operation, 1)}


def test_namespaced_save_completion_routes_only_to_owning_scheduler_child():
    operation = SaveOperationId(9, 3)
    first = FakeSchedSub(offload_methods=True)
    second = FakeSchedSub(offload_methods=True)
    scheduler = _sched([first, second])

    scheduler.save_finished(_multi_save(operation, 1))

    assert first.saved == []
    assert second.saved == [operation]


def test_legacy_child_completion_maps_oldest_operation_for_its_connector():
    first = SaveOperationId(9, 21)
    second = SaveOperationId(9, 22)
    child = FakeWorkerSub()
    worker = _worker([child])
    worker.start_load_kv(MultiConnectorMetadata([_operation_save_meta(first)]))
    worker.start_load_kv(MultiConnectorMetadata([_operation_save_meta(second)]))

    child._finished = KVConnectorOutput(finished_saving={9})
    oldest = worker.get_finished()
    child._finished = KVConnectorOutput(finished_saving={second})
    newest = worker.get_finished()

    assert oldest.finished_saving == {_multi_save(first, 0)}
    assert newest.finished_saving == {_multi_save(second, 0)}


def test_send_released_before_late_save_registration_still_emits_save():
    operation = SaveOperationId(9, 23)
    producer = FakeWorkerSub(is_producer=True, finished=({9}, set()))
    saver = FakeWorkerSub()
    worker = _worker([producer, saver])

    assert worker.get_finished().finished_sending == {9}

    producer._finished = (set(), set())
    worker.start_load_kv(
        MultiConnectorMetadata([ConnectorMetadata(), _operation_save_meta(operation)])
    )
    saver._finished = KVConnectorOutput(finished_saving={operation})
    late_save = worker.get_finished()

    assert late_save.finished_sending == set()
    assert late_save.finished_saving == {_multi_save(operation, 1)}
    assert worker.pairing_state_count == (0, 0)


def test_reused_req_id_late_generation_duplicate_cannot_complete_new_save():
    first = SaveOperationId(9, 24)
    second = SaveOperationId(9, 25)
    producer = FakeWorkerSub(is_producer=True)
    saver = FakeWorkerSub()
    worker = _worker([producer, saver])

    worker.start_load_kv(
        MultiConnectorMetadata([ConnectorMetadata(), _operation_save_meta(first)])
    )
    producer._finished = ({9}, set())
    saver._finished = KVConnectorOutput(finished_saving={first})
    assert worker.get_finished().finished_saving == {_multi_save(first, 1)}

    worker.start_load_kv(
        MultiConnectorMetadata([ConnectorMetadata(), _operation_save_meta(second)])
    )
    producer._finished = ({9}, set())
    saver._finished = KVConnectorOutput(finished_saving={first})
    duplicate = worker.get_finished()
    assert duplicate.finished_sending == set()
    assert duplicate.finished_saving == set()

    producer._finished = (set(), set())
    saver._finished = KVConnectorOutput(finished_saving={second})
    terminal = worker.get_finished()
    assert terminal.finished_sending == {9}
    assert terminal.finished_saving == {_multi_save(second, 1)}


def test_independent_send_release_retains_no_raw_request_state():
    producers = FakeWorkerSub(is_producer=True)
    worker = _worker([producers])

    for req_id in range(4):
        producers._finished = ({req_id}, set())
        assert worker.get_finished().finished_sending == {req_id}

    assert worker.pairing_state_count == (0, 0)


def test_completed_save_tombstones_are_bounded():
    saver = FakeWorkerSub()
    worker = _worker([saver])
    worker._completed_save_operations.limit = 2

    for generation in range(4):
        operation = SaveOperationId(9, 30 + generation)
        worker.start_load_kv(MultiConnectorMetadata([_operation_save_meta(operation)]))
        saver._finished = KVConnectorOutput(finished_saving={operation})
        assert worker.get_finished().finished_saving == {_multi_save(operation, 0)}

    assert worker.completed_save_tombstone_count == 2


@pytest.mark.parametrize("succeeded", [True, False])
def test_connector_completion_cannot_release_send_before_save_completion(
    succeeded,
):
    operation = SaveOperationId(9, 2)
    completion = _completion(
        DSV4_CHECKPOINT_SAVE_CHANNEL,
        operation,
        succeeded=succeeded,
    )
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub(
        completion_channels={DSV4_CHECKPOINT_SAVE_CHANNEL},
    )
    w = _worker([moriio, off])
    w.start_load_kv(
        MultiConnectorMetadata(
            [
                ConnectorMetadata(),
                _operation_save_meta(operation, page=False, sidecar=True),
            ],
            completion_channel_owners={DSV4_CHECKPOINT_SAVE_CHANNEL: 1},
        )
    )

    moriio._finished = ({9}, set())
    off._finished = KVConnectorOutput(connector_completions={completion})
    sidecar_first = w.get_finished()

    assert sidecar_first.finished_sending == set()
    assert sidecar_first.finished_saving == set()
    assert sidecar_first.connector_completions == {completion}
    assert w.pairing_state_count == (1, 1)

    moriio._finished = (set(), set())
    off._finished = KVConnectorOutput(finished_saving={operation})
    save_terminal = w.get_finished()

    assert save_terminal.finished_sending == {9}
    assert save_terminal.finished_saving == {_multi_save(operation, 1)}
    assert w._pending_save == {}
    assert w._sent == {}


def test_paired_send_completion_cleans_connector_before_deallocation():
    producer = FakeSchedSub(is_producer=True)
    off = FakeSchedSub(is_offload=True, offload_methods=True)
    multi = _sched([producer, off])
    seq = SimpleNamespace(id=9)
    host = Scheduler.__new__(Scheduler)
    host.kv_connector = multi
    host.deferred_free_blocks = {9: seq}
    host.finished_recving_kv_req_ids = []
    host.failed_recving_kv_req_ids = []

    class _BlockManager:
        def __init__(self):
            self.deallocated = []

        def deallocate(self, released):
            assert producer.finished_calls == [seq]
            assert off.finished_calls == [seq]
            self.deallocated.append(released)

    host.block_manager = _BlockManager()

    host._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_sending={9},
            finished_saving={9},
        )
    )

    assert host.block_manager.deallocated == [seq]
    assert host.deferred_free_blocks == {}


def test_scheduler_routes_generic_completion_to_unique_multi_owner():
    completion = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, "r1")
    undeclared = _completion("unknown", "r2")
    unrelated = FakeSchedSub()
    owner = FakeSchedSub(
        completion_channels={DSV4_CHECKPOINT_SAVE_CHANNEL},
    )
    multi = _sched([unrelated, owner])
    host = Scheduler.__new__(Scheduler)
    host.kv_connector = multi
    host.deferred_free_blocks = {}
    host.finished_recving_kv_req_ids = []
    host.failed_recving_kv_req_ids = []

    host._update_from_kv_xfer_finished(
        KVConnectorOutput(connector_completions={completion, undeclared})
    )

    assert unrelated.completion_calls == []
    assert owner.completion_calls == [completion]


def test_save_then_send_also_pairs():
    moriio = FakeWorkerSub(is_producer=True)
    off = FakeWorkerSub()
    w = _worker([moriio, off])
    w.start_load_kv(MultiConnectorMetadata([ConnectorMetadata(), _save_meta(9)]))

    # Step 1: save completes independently; Scheduler owns block lifetime.
    off._finished = KVConnectorOutput(finished_saving={9})
    out1 = w.get_finished()
    assert out1.finished_sending == set()
    assert out1.finished_saving == {9}

    # Step 2: send completes without replaying the prior save.
    off._finished = KVConnectorOutput()
    moriio._finished = ({9}, set())
    out2 = w.get_finished()
    assert out2.finished_sending == {9}
    assert out2.finished_saving == set()
