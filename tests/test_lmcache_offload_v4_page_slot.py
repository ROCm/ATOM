# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.types import (
    KVTransferRegion,
    SaveOperationId,
)
from atom.kv_transfer.offload.hybrid.dsv4 import connector as connector_module
from atom.kv_transfer.offload.hybrid.dsv4.codec import (
    HEADER_BYTES,
    DSV4CheckpointCodec,
    DSV4CheckpointCorruptionError,
    DSV4CheckpointError,
    DSV4CheckpointHeader,
    DSV4CheckpointKey,
    DSV4PageSlotCodec,
    decode_checkpoint,
    encode_checkpoint,
)
from atom.kv_transfer.offload.hybrid.dsv4.connector import (
    DSV4_CHECKPOINT_SAVE_CHANNEL,
    _env_nonnegative_float,
    _env_positive_float,
    _wait_for_publication,
)
from atom.kv_transfer.offload.hybrid.dsv4.connector import (
    DSV4OffloadConnector as LMCacheOffloadConnector,
)
from atom.kv_transfer.offload.hybrid.dsv4.policy import (
    _compute_slot_fingerprint,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
    SlotLoadSpec,
    SlotSaveSpec,
)

_FINGERPRINT = bytes.fromhex("00112233445566778899aabbccddeeff")
_PAYLOAD = b"\x07\x08\x09\xff"


def _completion_ids(output, channel: str, *, succeeded: bool) -> set:
    return {
        completion.operation_id
        for completion in output.connector_completions
        if completion.channel == channel and completion.succeeded is succeeded
    }


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        assert duration > 0
        self.sleeps.append(duration)
        self.now += duration


class _OrderedSet(set):
    def __init__(self, order: list[str], label: str) -> None:
        super().__init__()
        self._order = order
        self._label = label

    def add(self, value) -> None:
        self._order.append(self._label)
        super().add(value)


class _FakeEvent:
    def __init__(
        self,
        order: list[str],
        *,
        record_error: Exception | None = None,
        synchronize_error: Exception | None = None,
        query_result: bool = True,
        query_error: Exception | None = None,
    ) -> None:
        self._order = order
        self.record_error = record_error
        self.synchronize_error = synchronize_error
        self.query_result = query_result
        self.query_error = query_error

    def record(self, stream) -> None:
        self._order.append("event-record")
        if self.record_error is not None:
            raise self.record_error

    def synchronize(self) -> None:
        self._order.append("event-sync")
        if self.synchronize_error is not None:
            raise self.synchronize_error

    def query(self) -> bool:
        self._order.append("event-query")
        if self.query_error is not None:
            raise self.query_error
        return self.query_result


class _FakeRPCStream:
    def __init__(
        self,
        order: list[str] | None = None,
        *,
        synchronize_error: Exception | None = None,
    ) -> None:
        self._order = order
        self.synchronize_error = synchronize_error

    def synchronize(self) -> None:
        if self._order is not None:
            self._order.append("rpc-stream-sync")
        if self.synchronize_error is not None:
            raise self.synchronize_error


class _FakeTransferStream:
    def __init__(
        self,
        order: list[str],
        *,
        synchronize_error: Exception | None = None,
    ) -> None:
        self._order = order
        self.synchronize_error = synchronize_error

    def synchronize(self) -> None:
        self._order.append("sync")
        if self.synchronize_error is not None:
            raise self.synchronize_error


class _FakeRow:
    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.device = torch.device("cuda:0")
        self.copied = None

    def copy_(self, source, *, non_blocking=False):
        self._order.append("h2d")
        self.copied = bytes(source.tolist())
        return self


class _FakeSlotCodec:
    slot_bytes = len(_PAYLOAD)
    device = torch.device("cuda:0")

    def __init__(self, order: list[str], staging_slots: int) -> None:
        self.order = order
        self.snapshot_error: Exception | None = None
        self.restore_error: Exception | None = None
        self.snapshot_groups: list[int] = []
        self.rows = [_FakeRow(order) for _ in range(staging_slots)]
        self.row = self.rows[0]

    def gather_slot(self, group, dst, *, stream=None) -> None:
        self.order.append("snapshot")
        self.snapshot_groups.append(group)
        assert dst in self.rows
        assert isinstance(stream, _FakeRPCStream)
        if self.snapshot_error is not None:
            raise self.snapshot_error

    def scatter_slot(self, src, group, *, stream=None) -> None:
        self.order.append("restore")
        assert group == 3
        assert src in self.rows
        assert isinstance(stream, _FakeTransferStream)
        if self.restore_error is not None:
            raise self.restore_error


class _FakeSlotStaging:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self._rows = rows
        self.shape = (len(rows), len(_PAYLOAD))

    def __getitem__(self, staging_id: int) -> _FakeRow:
        return self._rows[staging_id]


class _FakeUnifiedCodec:
    slot_bytes = len(_PAYLOAD)
    payload_bytes = slot_bytes
    device = torch.device("cuda:0")

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.gathered: list[tuple[int, torch.Tensor, object]] = []
        self.scattered: list[tuple[torch.Tensor, int, object]] = []

    def gather_slot(self, group, dst, *, stream=None) -> None:
        self.order.append("unified-gather")
        self.gathered.append((group, dst, stream))

    def scatter_slot(self, src, group, *, stream=None) -> None:
        self.order.append("unified-scatter")
        self.scattered.append((src, group, stream))


class _FakeAdmission:
    def __init__(
        self,
        order: list[str],
        *,
        available: bool = True,
        capacity: int = 1,
    ) -> None:
        self.order = order
        self.available = available
        self.capacity = capacity
        self._free = list(range(capacity))
        self._acquired: set[int] = set()
        self.quarantined: list[int] = []
        self.release_error: Exception | None = None
        self.released: list[int] = []

    @property
    def acquired(self) -> bool:
        return bool(self._acquired)

    @property
    def num_free(self) -> int:
        return len(self._free)

    def try_acquire(self):
        self.order.append("acquire")
        if not self.available or not self._free:
            return None
        staging_id = self._free.pop(0)
        self._acquired.add(staging_id)
        return staging_id

    def release(self, staging_id) -> None:
        self.order.append("release")
        assert staging_id in self._acquired
        self._acquired.remove(staging_id)
        self._free.append(staging_id)
        self._free.sort()
        self.released.append(staging_id)
        if self.release_error is not None:
            raise self.release_error

    def quarantine(self, staging_id) -> None:
        self.order.append("quarantine")
        assert staging_id in self._acquired
        self._acquired.remove(staging_id)
        self.quarantined.append(staging_id)


class _FakeEngine:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.store_error: Exception | None = None
        self.lookup_error: Exception | None = None
        self.lookup_hit: int | None = None
        self.lookup_hits: list[int | None] = []
        self.retrieve_complete = True
        self.store_calls = []
        self.lookup_calls = []
        self.retrieve_calls = []
        self.unpinned = []

    def store(self, tokens, mask=None, **kwargs) -> None:
        self.order.append("store")
        self.store_calls.append((tokens.tolist(), mask.tolist(), kwargs))
        if self.store_error is not None:
            raise self.store_error

    def lookup(self, tokens, **kwargs) -> int:
        self.order.append("lookup")
        self.lookup_calls.append((list(tokens), kwargs))
        if self.lookup_error is not None:
            raise self.lookup_error
        if self.lookup_hits:
            hit = self.lookup_hits.pop(0)
            return len(tokens) if hit is None else hit
        return len(tokens) if self.lookup_hit is None else self.lookup_hit

    def retrieve(self, tokens, mask=None, **kwargs):
        self.order.append("retrieve")
        self.retrieve_calls.append((tokens.tolist(), mask.tolist(), kwargs))
        result = mask.clone()
        if not self.retrieve_complete:
            result[-1] = False
        return result

    def lookup_unpin(self, lookup_id) -> None:
        self.order.append("unpin")
        self.unpinned.append(lookup_id)


class _FakeStore:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.put_result = True
        self.put_error: Exception | None = None
        self.get_result = None
        self.contains_result = True
        self.contains_results: list[bool] = []
        self.contains_error: Exception | None = None
        self.invalidate_result = True
        self.invalidate_results: list[bool] = []
        self.put_calls = []
        self.get_calls = []
        self.contains_calls = []
        self.invalidate_calls = []
        self.unresolved_corruption = set()

    def put(self, key, blob) -> bool:
        if key in self.unresolved_corruption and not self.invalidate(key):
            return False
        self.order.append("put")
        self.put_calls.append((key, blob))
        if self.put_error is not None:
            raise self.put_error
        return self.put_result

    def get(self, key):
        self.order.append("get")
        self.get_calls.append(key)
        return self.get_result

    @contextmanager
    def borrow(self, key):
        result = self.get(key)
        if result is not None and not isinstance(result, torch.Tensor):
            result = torch.frombuffer(bytearray(result), dtype=torch.uint8)
        try:
            yield result
        finally:
            if result is not None:
                self.order.append("store-release")

    def contains(self, key) -> bool:
        self.order.append("contains")
        self.contains_calls.append(key)
        if self.contains_error is not None:
            raise self.contains_error
        if self.contains_results:
            return self.contains_results.pop(0)
        return self.contains_result

    def invalidate(self, key) -> bool:
        self.order.append("invalidate")
        self.invalidate_calls.append(key)
        if self.invalidate_results:
            result = self.invalidate_results.pop(0)
        else:
            result = self.invalidate_result
        if result:
            self.unresolved_corruption.discard(key)
        else:
            self.unresolved_corruption.add(key)
        return result


class _InlineExecutor:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def submit(self, fn, *args):
        self.order.append("submit")
        fn(*args)


class _DeferredExecutor:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls = []

    def submit(self, fn, *args):
        self.order.append("submit")
        self.calls.append((fn, args))


class _RejectingExecutor:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def submit(self, fn, *args):
        self.order.append("submit")
        raise RuntimeError("executor unavailable")


class _FirstDeferredThenRejectExecutor:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls = []

    def submit(self, fn, *args):
        self.order.append("submit")
        if self.calls:
            raise RuntimeError("executor unavailable")
        self.calls.append((fn, args))


class _ConcurrentExecutor:
    def __init__(self, workers: int = 2) -> None:
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures = []
        self.started = []
        self._lock = threading.Lock()

    def submit(self, fn, *args):
        started = threading.Event()
        with self._lock:
            self.started.append(started)

        def invoke():
            started.set()
            return fn(*args)

        future = self.pool.submit(invoke)
        self.futures.append(future)
        return future

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


def _sidecar_blob(
    *,
    boundary_tokens: int = 8,
    boundary_block_hash: int = 0x1234,
    payload: bytes = _PAYLOAD,
) -> bytes:
    return encode_checkpoint(
        DSV4CheckpointHeader(
            boundary_tokens=boundary_tokens,
            boundary_block_hash=boundary_block_hash,
            payload_bytes=None,
            payload_crc32=None,
            fingerprint=_FINGERPRINT,
            tp_size=2,
            tp_rank=1,
        ),
        payload,
    )


def _crc_corrupt_sidecar_blob() -> bytes:
    blob = bytearray(_sidecar_blob())
    blob[-1] ^= 1
    return bytes(blob)


def _worker(
    order: list[str],
    *,
    admission_available: bool = True,
    staging_slots: int = 1,
) -> LMCacheOffloadConnector:
    connector = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    connector._lock = threading.Lock()
    connector._done_load = _OrderedSet(order, "done")
    connector._failed_load = _OrderedSet(order, "failed")
    connector._done_save = _OrderedSet(order, "done")
    connector._done_sidecar_save = _OrderedSet(order, "sidecar-done")
    connector._failed_sidecar_save = _OrderedSet(order, "sidecar-failed")
    connector._pending_save_ops = {}
    connector._pending_legacy_save_ops = {}
    connector._save_req_locks = {}
    connector._engine = _FakeEngine(order)
    connector._codec = _FakeSlotCodec(order, staging_slots)
    connector._checkpoint_codec = DSV4CheckpointCodec(
        fingerprint=_FINGERPRINT,
        tp_size=2,
        tp_rank=1,
    )
    connector._slot_staging = _FakeSlotStaging(connector._codec.rows)
    connector._slot_store = _FakeStore(order)
    connector._slot_admission = _FakeAdmission(
        order,
        available=admission_available,
        capacity=staging_slots,
    )
    connector._rank = 1
    connector.chunk_size = 4
    connector._do_save = True
    connector._do_load = True
    connector._save_executor = _InlineExecutor(order)
    connector._max_pending_saves = 100
    connector._save_admission = threading.BoundedSemaphore(100)
    connector._load_executor = _InlineExecutor(order)
    connector._lookup_server = None
    connector._publication_timeout_s = 0.0
    connector._publication_poll_interval_s = 0.01
    connector._publication_clock = connector_module.time.monotonic
    connector._publication_sleep = connector_module.time.sleep

    def _copy_slot_staging_to_cpu(staging_id):
        order.append("d2h")
        framed = torch.empty(HEADER_BYTES + len(_PAYLOAD), dtype=torch.uint8)
        framed[HEADER_BYTES:] = torch.tensor(list(_PAYLOAD), dtype=torch.uint8)
        return framed

    connector._copy_slot_staging_to_cpu = _copy_slot_staging_to_cpu
    connector._create_slot_stream = lambda: _FakeTransferStream(order)
    connector._slot_stream_context = lambda stream: nullcontext()
    return connector


def _inject_fake_publication_clock(
    connector: LMCacheOffloadConnector,
) -> _FakeClock:
    clock = _FakeClock()
    connector._publication_clock = clock.monotonic
    connector._publication_sleep = clock.sleep
    connector._publication_timeout_s = 0.3
    connector._publication_poll_interval_s = 0.1
    return clock


def _patch_rpc_cuda(monkeypatch, order: list[str]) -> _FakeRPCStream:
    stream = _FakeRPCStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: _FakeEvent(order))
    return stream


def _save_request(*, page: bool = True) -> LMCacheReqMeta:
    return LMCacheReqMeta(
        req_id=17,
        token_ids=list(range(8)),
        block_ids=[10, 11],
        save_spec=SaveSpec(skip_leading_tokens=0) if page else None,
        slot_save_spec=SlotSaveSpec(
            boundary_tokens=8,
            boundary_block_hash=0x1234,
            source_group=2,
        ),
    )


def _load_request(*, page: bool = True, req_id: int = 23) -> LMCacheReqMeta:
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=list(range(8)),
        block_ids=[20, 21],
        load_spec=(
            LoadSpec(hbm_cached_tokens=4, lmcache_cached_tokens=8, can_load=True)
            if page
            else None
        ),
        slot_load_spec=SlotLoadSpec(
            boundary_tokens=8,
            boundary_block_hash=0x1234,
            destination_group=3,
        ),
    )


def _metadata(req: LMCacheReqMeta) -> LMCacheOffloadMetadata:
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(req)
    metadata.lookup_requests_in_step = [str(req.req_id)]
    return metadata


def test_slot_request_specs_are_immutable_and_optional():
    save = SlotSaveSpec(8, 0x1234, 2)
    load = SlotLoadSpec(8, 0x1234, 3)
    request = LMCacheReqMeta(req_id=1, token_ids=[], block_ids=[])

    assert request.slot_save_spec is None
    assert request.slot_load_spec is None
    with pytest.raises(FrozenInstanceError):
        save.source_group = 0
    with pytest.raises(FrozenInstanceError):
        load.destination_group = 0


def test_get_finished_atomically_drains_sidecar_save_results():
    connector = _worker([])
    connector._done_load.add(1)
    connector._done_save.add(2)
    connector._failed_load.add(3)
    connector._done_sidecar_save.add(4)
    connector._failed_sidecar_save.add(5)

    result = connector.get_finished()

    assert result.finished_loading == {1}
    assert result.finished_saving == {2}
    assert result.failed_loading == {3}
    assert _completion_ids(result, DSV4_CHECKPOINT_SAVE_CHANNEL, succeeded=True) == {4}
    assert _completion_ids(result, DSV4_CHECKPOINT_SAVE_CHANNEL, succeeded=False) == {5}
    assert connector.get_finished().is_empty()


def test_unexpected_slot_save_failure_reports_page_and_sidecar_terminals():
    connector = _worker([])
    request = _save_request(page=False)

    def fail(_request, *args):
        raise RuntimeError("unexpected save failure")

    connector._do_save_req = fail
    req_lock = connector._begin_save_operation(request.req_id)
    connector._run_save_req(request, None, None, req_lock)
    result = connector.get_finished()

    assert result.finished_saving == {request.req_id}
    assert not _completion_ids(result, DSV4_CHECKPOINT_SAVE_CHANNEL, succeeded=True)
    assert _completion_ids(result, DSV4_CHECKPOINT_SAVE_CHANNEL, succeeded=False) == {
        request.req_id
    }
    assert connector._pending_save_ops == {}
    assert connector._save_req_locks == {}


def test_unexpected_save_failure_reports_exact_operation_generation():
    connector = _worker([])
    request = _save_request(page=False)
    request.save_operation = SaveOperationId(request.req_id, 3)

    def fail(_request, *args):
        raise RuntimeError("unexpected save failure")

    connector._do_save_req = fail
    req_lock = connector._begin_save_operation(
        request.req_id,
        request.save_operation,
    )
    connector._run_save_req(request, None, None, req_lock)
    result = connector.get_finished()

    assert result.finished_saving == {request.save_operation}
    assert _completion_ids(result, DSV4_CHECKPOINT_SAVE_CHANNEL, succeeded=False) == {
        request.save_operation
    }
    assert connector._pending_save_ops == {}
    assert connector._save_req_locks == {}


def test_request_metadata_preserves_old_positional_field_order():
    load_spec = LoadSpec(4, 8, True)
    save_spec = SaveSpec(0, True)

    request = LMCacheReqMeta(
        1,
        list(range(8)),
        [10, 11],
        load_spec,
        save_spec,
        False,
    )

    assert request.load_spec is load_spec
    assert request.save_spec is save_spec
    assert request.is_last_prefill is False
    assert request.slot_load_spec is None
    assert request.slot_save_spec is None
    assert request.save_operation is None


def test_slot_fingerprint_is_stable_and_covers_required_geometry():
    regions = [
        SimpleNamespace(
            base_addr=0x1000,
            unit_bytes=16,
            total_bytes=64,
            reverse_indexed=True,
            semantic_role="dsv4.main_kv.nope",
        ),
        SimpleNamespace(
            base_addr=0x2000,
            unit_bytes=24,
            total_bytes=96,
            reverse_indexed=True,
            semantic_role="dsv4.main_kv.rope",
        ),
    ]
    kwargs = {
        "model_tag": "org/model",
        "page_namespace": "org/model::atom-page-v1-current",
        "kv_dtype": "fp8",
        "compress_ratios": [4, 128, 0],
        "block_size": 64,
        "kv_head_dim": 512,
        "index_head_dim": 128,
        "num_slots": 4,
        "slot_regions": regions,
        "tp_size": 2,
        "tp_rank": 1,
    }

    fingerprint = _compute_slot_fingerprint(**kwargs)
    assert len(fingerprint) == 16
    assert fingerprint == _compute_slot_fingerprint(**kwargs)

    moved_regions = [
        SimpleNamespace(**{**vars(region), "base_addr": region.base_addr + 0x10000})
        for region in regions
    ]
    assert fingerprint == _compute_slot_fingerprint(
        **{**kwargs, "slot_regions": moved_regions}
    )

    changed = [
        {**kwargs, "model_tag": "other/model"},
        {**kwargs, "page_namespace": "org/model::atom-page-v0-stale"},
        {**kwargs, "kv_dtype": "bf16"},
        {**kwargs, "compress_ratios": [4, 64, 0]},
        {**kwargs, "block_size": 32},
        {**kwargs, "kv_head_dim": 576},
        {**kwargs, "index_head_dim": 160},
        {**kwargs, "num_slots": 3},
        {**kwargs, "tp_size": 4},
        {**kwargs, "tp_rank": 0},
        {
            **kwargs,
            "slot_regions": [
                SimpleNamespace(**{**vars(regions[0]), "unit_bytes": 17}),
                regions[1],
            ],
        },
    ]
    assert all(
        _compute_slot_fingerprint(**variant) != fingerprint for variant in changed
    )

    stale_fingerprint = _compute_slot_fingerprint(
        **{**kwargs, "page_namespace": "org/model"}
    )
    stale_blob = encode_checkpoint(
        DSV4CheckpointHeader(
            boundary_tokens=8,
            boundary_block_hash=0x1234,
            payload_bytes=None,
            payload_crc32=None,
            fingerprint=stale_fingerprint,
            tp_size=2,
            tp_rank=1,
        ),
        _PAYLOAD,
    )
    with pytest.raises(DSV4CheckpointError, match="fingerprint mismatch"):
        decode_checkpoint(
            stale_blob,
            expected_fingerprint=fingerprint,
            expected_tp_size=2,
            expected_tp_rank=1,
            expected_boundary_tokens=8,
            expected_boundary_block_hash=0x1234,
            expected_payload_bytes=len(_PAYLOAD),
        )


def test_slot_fingerprint_tracks_equal_geometry_plane_roles_from_codec_snapshot():
    page_regions = [
        KVTransferRegion(0x1000, 64, 16),
        KVTransferRegion(0x2000, 64, 16),
    ]
    slot_a = KVTransferRegion(
        0x1040,
        32,
        8,
        reverse_indexed=True,
        semantic_role="dsv4.main_kv.nope",
    )
    slot_b = KVTransferRegion(
        0x2040,
        32,
        8,
        reverse_indexed=True,
        semantic_role="dsv4.main_kv.rope",
    )
    canonical = DSV4PageSlotCodec(
        page_regions,
        [slot_a, slot_b],
        num_blocks=4,
        num_slots=4,
        device="cpu",
    )
    reordered = DSV4PageSlotCodec(
        page_regions,
        [slot_b, slot_a],
        num_blocks=4,
        num_slots=4,
        device="cpu",
    )
    kwargs = {
        "model_tag": "org/model",
        "page_namespace": "org/model::atom-page-v2-current",
        "kv_dtype": "fp8",
        "compress_ratios": [4, 128, 0],
        "block_size": 64,
        "kv_head_dim": 512,
        "index_head_dim": 128,
        "num_slots": 4,
        "tp_size": 2,
        "tp_rank": 1,
    }

    canonical_fingerprint = _compute_slot_fingerprint(
        **kwargs,
        slot_regions=canonical.slot_regions,
    )
    reordered_fingerprint = _compute_slot_fingerprint(
        **kwargs,
        slot_regions=reordered.slot_regions,
    )

    assert [region.semantic_role for region in canonical.slot_regions] == [
        "dsv4.main_kv.nope",
        "dsv4.main_kv.rope",
    ]
    assert [region.semantic_role for region in reordered.slot_regions] == [
        "dsv4.main_kv.rope",
        "dsv4.main_kv.nope",
    ]
    assert canonical_fingerprint != reordered_fingerprint

    # The fingerprint input is the immutable codec snapshot, not the mutable
    # transfer descriptor that happened to create it.
    slot_a.unit_bytes = 7
    assert (
        _compute_slot_fingerprint(
            **kwargs,
            slot_regions=canonical.slot_regions,
        )
        == canonical_fingerprint
    )


def test_stateful_page_initialization_builds_all_slot_components_after_engine(
    monkeypatch,
    caplog,
):
    order = []
    captured = {}
    sensitive_model_name = "private-org/sensitive-model"
    caplog.set_level(logging.INFO, logger="atom")

    class _Store:
        def __init__(self, engine, **kwargs) -> None:
            order.append("store-adapter")
            assert engine is connector._engine
            captured["store"] = (engine, kwargs)

    monkeypatch.setattr(connector_module, "DSV4CheckpointStore", _Store)
    monkeypatch.setenv("OFFLOAD_SLOT_STAGING_SLOTS", "3")

    regions = [
        KVTransferRegion(
            base_addr=0x1000,
            total_bytes=64,
            unit_bytes=16,
            reverse_indexed=True,
            semantic_role="dsv4.main_kv.nope",
        )
    ]
    connector = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    connector._config = SimpleNamespace(
        model=sensitive_model_name,
        kv_cache_dtype="fp8",
        kv_cache_block_size=64,
        kv_transfer_config={},
        hf_config=SimpleNamespace(compress_ratios=[4, 128, 0]),
    )
    connector.block_size = 64
    connector.profile = SimpleNamespace(kv_head_dim=512, index_head_dim=128)
    connector._engine = object()
    connector._codec = DSV4PageSlotCodec(
        page_regions=[KVTransferRegion(0x2000, 384, 96)],
        slot_regions=regions,
        num_blocks=4,
        num_slots=4,
        device="cpu",
    )
    connector._checkpoint_codec = None
    connector._slot_staging = None
    connector._slot_store = None
    connector._slot_admission = None
    transfer_tensors = SimpleNamespace(
        swa_block_regions=regions,
        num_slots=4,
        expected_full_slot_region_count=1,
    )

    connector._initialize_slot_sidecar(
        transfer_tensors,
        model_name=sensitive_model_name,
        tp_size=2,
        tp_rank=1,
    )

    assert order == ["store-adapter"]
    assert isinstance(connector._checkpoint_codec, DSV4CheckpointCodec)
    assert connector._slot_staging.shape == (3, 16)
    assert connector._slot_admission.capacity == 3
    assert connector._slot_store is not None
    assert connector._checkpoint_codec.fingerprint == _compute_slot_fingerprint(
        model_tag=sensitive_model_name,
        page_namespace=sensitive_model_name,
        kv_dtype="fp8",
        compress_ratios=[4, 128, 0],
        block_size=64,
        kv_head_dim=512,
        index_head_dim=128,
        num_slots=4,
        slot_regions=connector._codec.slot_regions,
        tp_size=2,
        tp_rank=1,
    )
    registration_logs = [
        record.message
        for record in caplog.records
        if "PAGE+SLOT registered" in record.message
    ]
    assert len(registration_logs) == 1
    registration_log = registration_logs[0]
    assert "page_bytes_per_block=96" in registration_log
    assert "slot_bytes=16" in registration_log
    assert "slot_staging_slots=3" in registration_log
    fingerprint = connector._checkpoint_codec.fingerprint
    assert f"fingerprint={fingerprint.hex()[:12]}" in registration_log
    assert fingerprint.hex() not in registration_log
    assert sensitive_model_name not in registration_log


@pytest.mark.parametrize("invalid", [True, 2.0, "2"])
def test_explicit_slot_staging_geometry_rejects_integer_coercion(invalid):
    connector = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    connector._config = SimpleNamespace(
        kv_transfer_config={
            "kv_connector_extra_config": {
                "slot_sidecar_staging_slots": invalid,
            }
        }
    )

    with pytest.raises(ValueError, match="staging count must be an integer"):
        connector._slot_staging_slots()


@pytest.mark.parametrize(
    "transfer_tensors",
    [
        SimpleNamespace(
            swa_block_regions=[],
            num_slots=4,
            expected_full_slot_region_count=1,
        ),
        SimpleNamespace(
            swa_block_regions=[KVTransferRegion(0x1000, 64, 16, reverse_indexed=True)],
            num_slots=0,
            expected_full_slot_region_count=1,
        ),
    ],
)
def test_stateful_page_initialization_rejects_incomplete_slot_geometry(
    transfer_tensors,
):
    connector = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    connector._config = SimpleNamespace(
        model="org/model",
        kv_cache_dtype="fp8",
        kv_transfer_config={},
        hf_config=SimpleNamespace(compress_ratios=[4]),
    )
    connector.block_size = 64
    connector._engine = object()

    with pytest.raises(ValueError, match="SLOT|slot"):
        connector._initialize_slot_sidecar(
            transfer_tensors,
            model_name="org/model",
            tp_size=2,
            tp_rank=1,
        )


def test_start_load_kv_issues_slot_snapshot_on_rpc_stream_before_return(
    monkeypatch,
):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._save_executor = _DeferredExecutor(order)

    connector.start_load_kv(_metadata(_save_request(page=False)))

    assert order.index("snapshot") < order.index("event-record") < order.index("submit")
    assert len(connector._save_executor.calls) == 1
    assert connector._done_save == set()


def test_unified_codec_gathers_into_connector_owned_slot_row(monkeypatch):
    order = []
    stream = _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    codec = _FakeUnifiedCodec(order)
    connector._codec = codec
    connector._slot_staging = torch.empty((1, len(_PAYLOAD)), dtype=torch.uint8)

    snapshot = connector._snapshot_reserved_slot_save(
        _save_request(page=False),
        source_group=7,
        staging_id=0,
    )

    assert snapshot.snapshot_ok is True
    assert len(codec.gathered) == 1
    group, row, used_stream = codec.gathered[0]
    assert group == 7
    assert row.data_ptr() == connector._slot_staging[0].data_ptr()
    assert used_stream is stream


def test_unified_codec_page_restore_passes_buffer_before_plan(monkeypatch):
    codec = DSV4PageSlotCodec(
        page_regions=[KVTransferRegion(0x2000, 384, 96)],
        slot_regions=[],
        num_blocks=4,
        num_slots=0,
        device="cpu",
    )
    device_buf = torch.empty(2 * codec.page_bytes_per_block, dtype=torch.uint8)
    stream = object()
    calls = []

    def _scatter(src, plan, *, stream=None):
        calls.append((src, plan, stream))

    monkeypatch.setattr(codec, "scatter", _scatter)

    codec.chunk_major_device_buffer_to_gpu(
        device_buf,
        [[1], [3]],
        stream=stream,
    )

    assert calls == [(device_buf, codec.page_plan((1, 3)), stream)]


def test_unified_codec_scatters_from_connector_owned_slot_row():
    order = []
    connector = _worker(order)
    codec = _FakeUnifiedCodec(order)
    connector._codec = codec
    connector._slot_staging = torch.empty((1, len(_PAYLOAD)), dtype=torch.uint8)
    payload = torch.tensor(list(_PAYLOAD), dtype=torch.uint8)

    connector._restore_slot_payload(payload, 0, destination_group=3)

    assert connector._slot_staging[0].tolist() == list(_PAYLOAD)
    assert len(codec.scattered) == 1
    row, group, stream = codec.scattered[0]
    assert row.data_ptr() == connector._slot_staging[0].data_ptr()
    assert group == 3
    assert isinstance(stream, _FakeTransferStream)


def test_slot_snapshot_uses_live_source_group_and_staging_row(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._save_executor = _DeferredExecutor(order)
    request = _save_request(page=False)

    connector.start_load_kv(_metadata(request))

    assert connector._codec.snapshot_groups == [request.slot_save_spec.source_group]
    assert order.index("snapshot") < order.index("event-record") < order.index("submit")
    assert connector._slot_admission.num_free == 0

    fn, args = connector._save_executor.calls[0]
    fn(*args)
    assert order.index("d2h") < order.index("release")
    assert order.index("release") < order.index("put")
    assert connector._slot_admission.released == [0]


def test_slot_background_save_waits_for_snapshot_event_before_d2h(monkeypatch):
    order = []
    event = _FakeEvent(order)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *args, **kwargs: _FakeRPCStream(order),
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: event)
    connector = _worker(order)
    connector._save_executor = _DeferredExecutor(order)
    connector.start_load_kv(_metadata(_save_request(page=False)))

    fn, args = connector._save_executor.calls[0]
    fn(*args)

    assert order.index("event-sync") < order.index("d2h")


def test_slot_staging_exhaustion_keeps_page_save_and_fails_sidecar(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order, admission_available=False)
    request = _save_request(page=True)

    connector.start_load_kv(_metadata(request))

    assert connector._codec.snapshot_groups == []
    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    result = connector.get_finished()
    assert result.finished_saving == {request.req_id}
    assert _completion_ids(result, DSV4_CHECKPOINT_SAVE_CHANNEL, succeeded=False) == {
        request.req_id
    }


def test_save_submission_failure_releases_snapshot_and_completes_save(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._save_executor = _RejectingExecutor(order)

    connector.start_load_kv(_metadata(_save_request()))

    assert order.index("snapshot") < order.index("submit") < order.index("event-sync")
    assert connector._slot_admission.released == [0]
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}


def test_rejected_save_sync_failure_quarantines_snapshot_row(monkeypatch):
    order = []
    connector = _worker(order)
    connector._save_executor = _RejectingExecutor(order)
    event = _FakeEvent(
        order,
        synchronize_error=RuntimeError("event synchronize failed"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *args, **kwargs: _FakeRPCStream(order),
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: event)

    connector.start_load_kv(_metadata(_save_request()))

    assert connector._done_save == {17}
    assert connector._failed_sidecar_save == {17}
    assert connector._slot_admission.released == []
    assert connector._slot_admission.quarantined == [0]
    assert connector._slot_admission.num_free == 0


def test_rejected_boundary_save_does_not_finish_prior_page_operation(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._save_executor = _FirstDeferredThenRejectExecutor(order)
    page = LMCacheReqMeta(
        req_id=17,
        token_ids=list(range(4)),
        block_ids=[10],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(17, 0),
    )
    boundary = _save_request()
    boundary.save_operation = SaveOperationId(17, 1)

    connector.start_load_kv(_metadata(page))
    connector.start_load_kv(_metadata(boundary))

    assert connector._pending_save_ops == {17: 1}
    rejected = connector.get_finished()
    assert rejected.finished_saving == {SaveOperationId(17, 1)}
    assert _completion_ids(rejected, DSV4_CHECKPOINT_SAVE_CHANNEL, succeeded=False) == {
        SaveOperationId(17, 1)
    }

    fn, args = connector._save_executor.calls[0]
    fn(*args)

    assert connector._done_save == {SaveOperationId(17, 0)}
    assert connector._pending_save_ops == {}
    assert connector._save_req_locks == {}


def test_max_pending_saves_bounds_running_plus_queued_before_snapshot(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._max_pending_saves = 1
    connector._save_admission = threading.BoundedSemaphore(1)
    connector._save_executor = _DeferredExecutor(order)
    page = LMCacheReqMeta(
        req_id=17,
        token_ids=list(range(4)),
        block_ids=[10],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(17, 0),
    )
    boundary = _save_request()
    boundary.save_operation = SaveOperationId(17, 1)

    connector.start_load_kv(_metadata(page))
    connector.start_load_kv(_metadata(boundary))

    assert len(connector._save_executor.calls) == 1
    assert "snapshot" not in order
    assert connector._slot_admission.num_free == 1
    assert connector._pending_save_ops == {17: 1}
    rejected = connector.get_finished()
    assert rejected.finished_saving == {SaveOperationId(17, 1)}
    assert _completion_ids(rejected, DSV4_CHECKPOINT_SAVE_CHANNEL, succeeded=False) == {
        SaveOperationId(17, 1)
    }

    fn, args = connector._save_executor.calls[0]
    fn(*args)
    third = LMCacheReqMeta(
        req_id=18,
        token_ids=list(range(4)),
        block_ids=[11],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(18, 0),
    )
    connector.start_load_kv(_metadata(third))

    assert len(connector._save_executor.calls) == 2
    assert connector._pending_save_ops == {18: 1}


def test_save_workers_parallelize_different_requests_but_complete_once(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    executor = _ConcurrentExecutor(workers=2)
    connector._save_executor = executor
    first_started = threading.Event()
    release_first = threading.Event()
    other_started = threading.Event()

    def controlled_save(req, producer_event=None, slot_snapshot=None):
        if req.req_id == 17:
            first_started.set()
            assert release_first.wait(timeout=2)
        else:
            other_started.set()

    connector._do_save_req = controlled_save
    first = LMCacheReqMeta(
        req_id=17,
        token_ids=list(range(4)),
        block_ids=[10],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(17, 0),
    )
    other = LMCacheReqMeta(
        req_id=18,
        token_ids=list(range(4)),
        block_ids=[11],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(18, 0),
    )

    try:
        connector.start_load_kv(_metadata(first))
        assert first_started.wait(timeout=2)
        connector.start_load_kv(_metadata(other))
        assert executor.started[1].wait(timeout=2)
        assert other_started.wait(timeout=2)
        release_first.set()
        for future in executor.futures:
            future.result(timeout=2)
    finally:
        release_first.set()
        executor.shutdown()

    assert connector.get_finished().finished_saving == {
        SaveOperationId(17, 0),
        SaveOperationId(18, 0),
    }
    assert connector._pending_save_ops == {}
    assert connector._save_req_locks == {}


def test_fallback_page_event_constructor_failure_is_terminal(monkeypatch):
    order = []
    connector = _worker(order, admission_available=False)

    def _event_failure(*args, **kwargs):
        raise RuntimeError("event constructor failed")

    monkeypatch.setattr(torch.cuda, "Event", _event_failure)

    connector.start_load_kv(_metadata(_save_request()))

    assert connector._engine.store_calls == []
    assert connector._done_save == {17}
    assert connector._failed_sidecar_save == {17}


def test_fallback_page_event_record_failure_is_terminal(monkeypatch):
    order = []
    connector = _worker(order, admission_available=False)
    event = _FakeEvent(order, record_error=RuntimeError("event record failed"))
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: event)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *args, **kwargs: _FakeRPCStream(order),
    )

    connector.start_load_kv(_metadata(_save_request()))

    assert connector._engine.store_calls == []
    assert connector._done_save == {17}
    assert connector._failed_sidecar_save == {17}


def test_load_submission_failure_unpins_once_and_marks_failed():
    order = []
    connector = _worker(order)
    connector._load_executor = _RejectingExecutor(order)

    connector.start_load_kv(_metadata(_load_request()))

    assert connector._engine.unpinned == ["23"]
    assert connector._done_load == set()
    assert connector._failed_load == {23}
    assert connector._slot_admission.released == [0]


def test_page_only_load_needs_no_slot_reservation():
    order = []
    connector = _worker(order, admission_available=False)
    request = _load_request()
    request.slot_load_spec = None

    connector.start_load_kv(_metadata(request))

    assert "acquire" not in order
    assert connector._done_load == {23}
    assert connector._failed_load == set()


def test_mixed_batch_reserves_slot_load_before_preparing_save(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._load_executor = _DeferredExecutor(order)
    connector._slot_store.get_result = _sidecar_blob()
    save_request = _save_request()
    load_request = _load_request()
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(save_request)
    metadata.add_request(load_request)
    metadata.lookup_requests_in_step = [str(load_request.req_id)]

    connector.start_load_kv(metadata)

    assert connector._slot_admission.acquired
    assert len(connector._load_executor.calls) == 1
    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._done_load == set()

    fn, args = connector._load_executor.calls[0]
    fn(*args)

    assert connector._done_load == {23}
    assert connector._failed_load == set()
    assert connector._slot_admission.released == [0]


def test_two_slot_loads_share_one_batch_reservation_sequentially():
    order = []
    connector = _worker(order)
    connector._slot_store.get_result = _sidecar_blob()
    first = _load_request(req_id=23)
    second = _load_request(req_id=24)
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(first)
    metadata.add_request(second)
    metadata.lookup_requests_in_step = ["23", "24"]

    connector.start_load_kv(metadata)

    assert order.count("restore") == 2
    assert connector._done_load == {23, 24}
    assert connector._failed_load == set()
    assert connector._slot_admission.released == [0]
    assert connector._slot_admission.num_free == 1


def test_extra_staging_row_allows_save_after_batch_load_reservation(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order, staging_slots=2)
    connector._load_executor = _DeferredExecutor(order)
    connector._slot_store.get_result = _sidecar_blob()
    load_request = _load_request()
    save_request = _save_request()
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(save_request)
    metadata.add_request(load_request)
    metadata.lookup_requests_in_step = [str(load_request.req_id)]

    connector.start_load_kv(metadata)

    assert connector._done_sidecar_save == {17}
    assert connector._failed_sidecar_save == set()
    assert connector._slot_admission.released == [1]
    assert connector._slot_admission.num_free == 1

    fn, args = connector._load_executor.calls[0]
    fn(*args)

    assert connector._done_load == {23}
    assert connector._slot_admission.released == [1, 0]
    assert connector._slot_admission.num_free == 2


def test_wait_for_publication_polls_until_visible_without_busy_spin():
    clock = _FakeClock()
    observations = iter([False, False, True])

    assert _wait_for_publication(
        lambda: next(observations),
        timeout_s=1.0,
        poll_interval_s=0.1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    assert clock.sleeps == [0.1, 0.1]


def test_wait_for_publication_immediate_hit_does_not_sleep():
    clock = _FakeClock()

    assert _wait_for_publication(
        lambda: True,
        timeout_s=5.0,
        poll_interval_s=0.1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    assert clock.sleeps == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        (
            "publication timeout",
            {"timeout_s": None, "poll_interval_s": 0.1},
        ),
        (
            "publication poll interval",
            {"timeout_s": 1.0, "poll_interval_s": None},
        ),
    ],
)
def test_wait_for_publication_rejects_nonfinite_bounds_before_probe(
    value,
    field,
    kwargs,
):
    probe_calls = 0

    def _probe() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return True

    kwargs = {
        name: value if configured is None else configured
        for name, configured in kwargs.items()
    }
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        _wait_for_publication(
            _probe,
            **kwargs,
            clock=lambda: 0.0,
            sleep=lambda duration: None,
        )
    assert probe_calls == 0


@pytest.mark.parametrize("visible", [False, True])
def test_wait_for_publication_zero_timeout_probes_once_without_sleeping(visible):
    clock = _FakeClock()
    probe_calls = 0

    def _probe() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return visible

    assert (
        _wait_for_publication(
            _probe,
            timeout_s=0.0,
            poll_interval_s=0.1,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        is visible
    )
    assert probe_calls == 1
    assert clock.sleeps == []


@pytest.mark.parametrize(
    ("name", "parser", "value"),
    [
        pytest.param(
            "OFFLOAD_PUBLICATION_TIMEOUT_S",
            _env_nonnegative_float,
            value,
            id=f"timeout-{label}",
        )
        for label, value in (
            ("nan", "nan"),
            ("positive-inf", "inf"),
            ("negative-inf", "-inf"),
        )
    ]
    + [
        pytest.param(
            "OFFLOAD_PUBLICATION_POLL_INTERVAL_S",
            _env_positive_float,
            value,
            id=f"interval-{label}",
        )
        for label, value in (
            ("nan", "nan"),
            ("positive-inf", "inf"),
            ("negative-inf", "-inf"),
        )
    ],
)
def test_publication_env_parsers_reject_nonfinite_values(
    monkeypatch,
    name,
    parser,
    value,
):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=rf"{name} must be finite"):
        parser(name, 1.0)


def test_publication_timeout_env_parser_accepts_zero(monkeypatch):
    monkeypatch.setenv("OFFLOAD_PUBLICATION_TIMEOUT_S", "0")

    assert _env_nonnegative_float("OFFLOAD_PUBLICATION_TIMEOUT_S", 5.0) == 0.0


@pytest.mark.parametrize(
    "name",
    [
        "OFFLOAD_PUBLICATION_TIMEOUT_S",
        "OFFLOAD_PUBLICATION_POLL_INTERVAL_S",
    ],
)
def test_connector_init_rejects_nonfinite_config_before_starting_executors(
    monkeypatch,
    name,
):
    monkeypatch.setenv(name, "nan")

    def _unexpected_executor(*args, **kwargs):
        raise AssertionError("executor started before config validation")

    monkeypatch.setattr(
        "atom.kv_transfer.offload._offload_common.ThreadPoolExecutor",
        _unexpected_executor,
    )
    config = SimpleNamespace(
        kv_transfer_config={},
        kv_cache_block_size=64,
    )

    with pytest.raises(ValueError, match=rf"{name} must be finite"):
        LMCacheOffloadConnector(config)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1.5", "bad"])
def test_max_pending_saves_rejects_invalid_connector_extra(value):
    with pytest.raises(ValueError, match="max pending saves"):
        connector_module.max_pending_saves(
            {
                "kv_connector_extra_config": {
                    "max_pending_saves": value,
                }
            },
            2,
        )


def test_max_pending_saves_precedence_and_worker_default(monkeypatch):
    monkeypatch.setenv("OFFLOAD_MAX_PENDING_SAVES", "7")
    assert connector_module.max_pending_saves({}, 3) == 7
    assert (
        connector_module.max_pending_saves(
            {"kv_connector_extra_config": {"max_pending_saves": 5}},
            3,
        )
        == 5
    )
    monkeypatch.delenv("OFFLOAD_MAX_PENDING_SAVES")
    assert connector_module.max_pending_saves({}, 3) == 6
    assert connector_module.max_pending_saves({}, 1) == 2


def test_wait_for_publication_times_out_with_bounded_positive_sleeps():
    clock = _FakeClock()

    assert not _wait_for_publication(
        lambda: False,
        timeout_s=0.3,
        poll_interval_s=0.1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    assert len(clock.sleeps) == 3
    assert all(duration > 0 for duration in clock.sleeps)
    assert sum(clock.sleeps) == pytest.approx(0.3)


def test_wait_for_publication_propagates_probe_exception_without_sleeping():
    clock = _FakeClock()

    def _broken_probe() -> bool:
        raise RuntimeError("probe failed")

    with pytest.raises(RuntimeError, match="probe failed"):
        _wait_for_publication(
            _broken_probe,
            timeout_s=1.0,
            poll_interval_s=0.1,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
    assert clock.sleeps == []


def test_page_and_slot_save_polls_partial_page_and_sidecar_visibility(
    monkeypatch,
    caplog,
):
    order = []
    caplog.set_level(logging.INFO, logger="atom")
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    clock = _inject_fake_publication_clock(connector)
    connector._engine.lookup_hits = [4, 8]
    connector._slot_store.contains_results = [False, True]

    connector.start_load_kv(_metadata(_save_request()))

    assert (
        order.index("snapshot")
        < order.index("store")
        < order.index("lookup")
        < order.index("put")
        < order.index("contains")
        < order.index("done")
    )
    assert (
        order.index("event-sync")
        < order.index("d2h")
        < order.index("release")
        < order.index("store")
        < order.index("put")
    )
    assert order.index("release") < min(
        i for i, item in enumerate(order) if item == "contains"
    )
    assert clock.sleeps == [0.1, 0.1]
    assert connector._done_save == {17}
    assert connector._done_sidecar_save == {17}
    assert connector._failed_sidecar_save == set()
    assert connector._slot_admission.released == [0]

    key, blob = connector._slot_store.put_calls[0]
    assert key == DSV4CheckpointKey(0x1234, _FINGERPRINT, 2, 1)
    assert isinstance(blob, torch.Tensor)
    assert blob.numel() == HEADER_BYTES + len(_PAYLOAD)
    header, payload = decode_checkpoint(
        memoryview(blob.numpy()),
        expected_fingerprint=_FINGERPRINT,
        expected_tp_size=2,
        expected_tp_rank=1,
        expected_boundary_tokens=8,
        expected_boundary_block_hash=0x1234,
        expected_payload_bytes=len(_PAYLOAD),
    )
    assert header.boundary_tokens == 8
    assert payload == _PAYLOAD
    assert connector._engine.lookup_calls == [
        (list(range(8)), {"pin": False}),
        (list(range(8)), {"pin": False}),
    ]
    terminal_logs = [
        record.message
        for record in caplog.records
        if "SLOT sidecar save " in record.message
    ]
    assert len(terminal_logs) == 1
    assert "SLOT sidecar save published" in terminal_logs[0]
    assert "boundary=8" in terminal_logs[0]
    rendered_logs = "\n".join(record.message for record in caplog.records)
    assert _PAYLOAD.hex() not in rendered_logs
    assert str(key) not in rendered_logs
    assert str(list(range(8))) not in rendered_logs
    assert "private-sensitive-model" not in rendered_logs


@pytest.mark.parametrize("blocked_operation", ["put", "contains"])
def test_slot_temp_row_is_free_while_checkpoint_storage_blocks(
    monkeypatch,
    blocked_operation,
):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    reached_storage = threading.Event()
    unblock_storage = threading.Event()
    original = getattr(connector._slot_store, blocked_operation)

    def blocked(*args, **kwargs):
        order.append(blocked_operation)
        assert connector._slot_admission.num_free == 1
        assert connector._slot_admission.released == [0]
        reached_storage.set()
        if not unblock_storage.wait(timeout=5):
            raise TimeoutError("test did not release blocked checkpoint storage")
        return original(*args, **kwargs)

    setattr(connector._slot_store, blocked_operation, blocked)
    executor = ThreadPoolExecutor(max_workers=1)
    connector._save_executor = executor
    try:
        connector.start_load_kv(_metadata(_save_request()))
        assert reached_storage.wait(timeout=5)
        assert order.index("d2h") < order.index("release")
        assert order.index("release") < order.index(blocked_operation)
    finally:
        unblock_storage.set()
        executor.shutdown(wait=True)

    assert connector._done_sidecar_save == {17}
    assert connector._failed_sidecar_save == set()


def test_post_d2h_release_failure_quarantines_row_and_keeps_page_save(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)

    class _ReleaseFailsBeforeMutation(_FakeAdmission):
        def release(self, staging_id) -> None:
            self.order.append("release")
            assert staging_id in self._acquired
            raise RuntimeError("release failed before ownership transition")

    admission = _ReleaseFailsBeforeMutation(order)
    connector._slot_admission = admission

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    assert admission.quarantined == [0]
    assert connector._done_sidecar_save == set()
    assert connector._failed_sidecar_save == {17}


def test_sidecar_submitted_but_not_visible_times_out_without_commit(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    clock = _inject_fake_publication_clock(connector)
    connector._slot_store.contains_result = False

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._slot_store.put_calls) == 1
    assert len(connector._slot_store.contains_calls) == 4
    assert len(clock.sleeps) == 3
    assert connector._done_sidecar_save == set()
    assert connector._failed_sidecar_save == {17}
    assert order.index("release") < min(
        i for i, item in enumerate(order) if item == "contains"
    )


def test_sidecar_visibility_exception_fails_once_without_retrying(monkeypatch, caplog):
    order = []
    secret = "private-key=full-storage-key"
    caplog.set_level(logging.WARNING, logger="atom")
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    clock = _inject_fake_publication_clock(connector)
    connector._slot_store.contains_error = RuntimeError(secret)

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._slot_store.contains_calls) == 1
    assert clock.sleeps == []
    assert connector._done_sidecar_save == set()
    assert connector._failed_sidecar_save == {17}
    terminal_logs = [
        record.message
        for record in caplog.records
        if "SLOT sidecar save " in record.message
    ]
    assert len(terminal_logs) == 1
    assert secret not in "\n".join(record.message for record in caplog.records)


def test_sidecar_only_save_skips_page_store(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)

    connector.start_load_kv(_metadata(_save_request(page=False)))

    assert connector._engine.store_calls == []
    assert connector._engine.lookup_calls == [(list(range(8)), {"pin": False})]
    assert len(connector._slot_store.put_calls) == 1
    assert connector._done_sidecar_save == {17}
    assert connector._done_save == {17}


def test_page_store_failure_prevents_sidecar_publish_and_releases_staging(
    monkeypatch,
):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._engine.store_error = RuntimeError("page store failed")

    connector.start_load_kv(_metadata(_save_request()))

    assert connector._slot_store.put_calls == []
    assert connector._done_sidecar_save == set()
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == [0]


@pytest.mark.parametrize("lookup_hit", [0, 4])
def test_nonthrowing_page_store_requires_full_lookup_coverage(
    monkeypatch,
    lookup_hit,
):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._engine.lookup_hit = lookup_hit

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._engine.store_calls) == 1
    assert connector._engine.lookup_calls == [(list(range(8)), {"pin": False})]
    assert connector._slot_store.put_calls == []
    assert connector._done_sidecar_save == set()
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == [0]


def test_unhealthy_frozen_like_store_skip_cannot_authorize_sidecar(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._engine.lookup_hit = 0

    def _silently_skipped_store(tokens, mask=None, **kwargs):
        order.append("store-skipped")

    connector._engine.store = _silently_skipped_store

    connector.start_load_kv(_metadata(_save_request()))

    assert order.index("store-skipped") < order.index("lookup")
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}


def test_page_coverage_lookup_error_prevents_sidecar_put(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._engine.lookup_error = RuntimeError("lookup unavailable")

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == [0]


def test_sidecar_only_save_requires_existing_full_page_coverage(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._engine.lookup_hit = 4

    connector.start_load_kv(_metadata(_save_request(page=False)))

    assert connector._engine.store_calls == []
    assert connector._engine.lookup_calls == [(list(range(8)), {"pin": False})]
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}


def test_boundary_page_suffix_can_establish_full_sidecar_coverage(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    request = _save_request()
    request.save_spec = SaveSpec(skip_leading_tokens=4)

    connector.start_load_kv(_metadata(request))

    assert len(connector._engine.store_calls) == 1
    assert connector._engine.lookup_calls == [(list(range(8)), {"pin": False})]
    assert len(connector._slot_store.put_calls) == 1
    assert connector._done_sidecar_save == {17}


def test_page_coverage_checks_only_tokens_through_slot_boundary(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    request = _save_request()
    request.token_ids = list(range(12))
    connector._engine.lookup_hit = 8

    connector.start_load_kv(_metadata(request))

    assert connector._engine.lookup_calls == [(list(range(8)), {"pin": False})]
    assert len(connector._slot_store.put_calls) == 1
    assert connector._done_sidecar_save == {17}


def test_save_staging_exhaustion_has_one_terminal_sidecar_log(monkeypatch, caplog):
    order = []
    caplog.set_level(logging.WARNING, logger="atom")
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order, admission_available=False)

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == []
    assert sum("SLOT sidecar save " in record.message for record in caplog.records) == 1


def test_save_admission_error_fails_sidecar_but_still_saves_page(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)

    class _BrokenAdmission:
        def try_acquire(self):
            order.append("acquire")
            raise RuntimeError("admission failed")

    connector._slot_admission = _BrokenAdmission()

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}


def test_snapshot_failure_fails_sidecar_but_still_saves_page_and_releases(
    monkeypatch,
):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._codec.snapshot_error = RuntimeError("snapshot failed")

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == [0]


def test_snapshot_cleanup_sync_error_is_total_and_page_save_proceeds(monkeypatch):
    order = []
    connector = _worker(order)
    connector._codec.snapshot_error = RuntimeError("snapshot failed")
    stream = _FakeRPCStream(
        order,
        synchronize_error=RuntimeError("stream synchronize failed"),
    )
    events = iter(
        [
            _FakeEvent(order, record_error=RuntimeError("event record failed")),
            _FakeEvent(order),
        ]
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: next(events))

    connector.start_load_kv(_metadata(_save_request()))

    assert "rpc-stream-sync" in order
    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == []
    assert connector._slot_admission.quarantined == [0]
    assert connector._slot_admission.num_free == 0


def test_snapshot_cleanup_release_error_is_total_and_page_save_proceeds(monkeypatch):
    order = []
    connector = _worker(order)
    connector._codec.snapshot_error = RuntimeError("snapshot failed")
    connector._slot_admission.release_error = RuntimeError("release failed")
    stream = _FakeRPCStream(order)
    events = iter(
        [
            _FakeEvent(order, record_error=RuntimeError("event record failed")),
            _FakeEvent(order),
        ]
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: next(events))

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == [0]


def test_save_event_synchronize_error_does_not_escape_cleanup(monkeypatch):
    order = []
    connector = _worker(order)
    event = _FakeEvent(
        order,
        synchronize_error=RuntimeError("event synchronize failed"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *args, **kwargs: _FakeRPCStream(order),
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: event)
    request = _save_request()
    snapshot = connector._prepare_slot_save(request)

    req_lock = connector._begin_save_operation(request.req_id)
    connector._run_save_req(request, event, snapshot, req_lock)

    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == []
    assert connector._slot_admission.quarantined == [0]
    assert connector._slot_admission.num_free == 0


def test_start_load_kv_event_synchronize_error_is_terminal(monkeypatch):
    order = []
    connector = _worker(order)
    event = _FakeEvent(
        order,
        synchronize_error=RuntimeError("event synchronize failed"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *args, **kwargs: _FakeRPCStream(order),
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: event)

    connector.start_load_kv(_metadata(_save_request()))

    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == []
    assert connector._slot_admission.quarantined == [0]
    assert connector._slot_admission.num_free == 0


def test_invalid_page_plan_still_releases_acquired_slot_snapshot(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    request = _save_request()
    request.save_spec = SimpleNamespace(skip_leading_tokens=object(), can_save=True)

    connector.start_load_kv(_metadata(request))

    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == [0]


def test_sidecar_put_rejection_marks_failure_and_releases_staging(
    monkeypatch,
    caplog,
):
    order = []
    caplog.set_level(logging.WARNING, logger="atom")
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._slot_store.put_result = False

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._slot_store.put_calls) == 1
    assert connector._done_sidecar_save == set()
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == [0]
    assert any(
        "SLOT sidecar save failed" in record.message and "boundary=8" in record.message
        for record in caplog.records
    )


def test_sidecar_save_failure_emits_one_sanitized_terminal_log(
    monkeypatch,
    caplog,
):
    order = []
    secret = "private-model:key=full-storage-key:tokens=[0,1,2]:payload=070809ff"
    caplog.set_level(logging.WARNING, logger="atom")
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._slot_store.put_error = RuntimeError(secret)

    connector.start_load_kv(_metadata(_save_request()))

    terminal_logs = [
        record.message
        for record in caplog.records
        if "SLOT sidecar save " in record.message
    ]
    assert len(terminal_logs) == 1
    assert "SLOT sidecar save failed" in terminal_logs[0]
    assert secret not in "\n".join(record.message for record in caplog.records)


def test_sidecar_d2h_failure_releases_staging(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)

    def _fail_d2h(staging_id):
        order.append("d2h")
        raise RuntimeError("D2H failed")

    connector._copy_slot_staging_to_cpu = _fail_d2h

    connector.start_load_kv(_metadata(_save_request()))

    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == [0]


def test_sidecar_d2h_sync_failure_quarantines_staging(monkeypatch):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    del connector._copy_slot_staging_to_cpu
    connector._create_slot_stream = lambda: _FakeTransferStream(
        order,
        synchronize_error=RuntimeError("D2H synchronize failed"),
    )

    class _FakeHost:
        def __getitem__(self, key):
            return self

        def copy_(self, source, *, non_blocking=False):
            order.append("d2h")
            return self

    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: _FakeHost())

    connector.start_load_kv(_metadata(_save_request()))

    assert len(connector._engine.store_calls) == 1
    assert connector._slot_store.put_calls == []
    assert connector._failed_sidecar_save == {17}
    assert connector._done_save == {17}
    assert connector._slot_admission.released == []
    assert connector._slot_admission.quarantined == [0]
    assert connector._slot_admission.num_free == 0


def test_page_and_slot_load_orders_retrieve_get_decode_restore_sync_then_done(
    monkeypatch,
    caplog,
):
    order = []
    caplog.set_level(logging.INFO, logger="atom")
    connector = _worker(order)
    connector._slot_store.get_result = torch.tensor(
        list(_sidecar_blob()),
        dtype=torch.uint8,
    )
    real_decode = connector._checkpoint_codec.decode_tensor

    def _record_decode(*args, **kwargs):
        order.append("decode")
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(connector._checkpoint_codec, "decode_tensor", _record_decode)

    connector.start_load_kv(_metadata(_load_request()))

    assert (
        order.index("retrieve")
        < order.index("get")
        < order.index("decode")
        < order.index("h2d")
        < order.index("restore")
        < order.index("sync")
        < order.index("store-release")
        < order.index("release")
        < order.index("done")
    )
    assert connector._done_load == {23}
    assert connector._failed_load == set()
    assert connector._engine.unpinned == ["23"]
    assert connector._slot_admission.released == [0]
    assert connector._codec.row.copied == _PAYLOAD
    terminal_logs = [
        record.message
        for record in caplog.records
        if "SLOT sidecar load " in record.message
    ]
    assert len(terminal_logs) == 1
    assert "SLOT sidecar load restored" in terminal_logs[0]
    assert "boundary=8" in terminal_logs[0]


def test_slot_only_load_skips_page_retrieve_and_marks_done():
    order = []
    connector = _worker(order)
    connector._slot_store.get_result = _sidecar_blob()

    connector.start_load_kv(_metadata(_load_request(page=False)))

    assert connector._engine.retrieve_calls == []
    assert connector._done_load == {23}
    assert connector._failed_load == set()
    assert connector._engine.unpinned == ["23"]
    assert connector._slot_admission.released == [0]


@pytest.mark.parametrize(
    "blob",
    [
        pytest.param(None, id="missing"),
        pytest.param(_crc_corrupt_sidecar_blob(), id="crc-mismatch"),
    ],
)
def test_missing_or_corrupt_sidecar_fails_whole_load_and_unpins_once(blob, caplog):
    order = []
    caplog.set_level(logging.WARNING, logger="atom")
    connector = _worker(order)
    connector._slot_store.get_result = blob

    connector.start_load_kv(_metadata(_load_request()))

    # PAGE retrieval succeeded first; the missing/corrupt SLOT must still make
    # the composite checkpoint load fail closed rather than publishing PAGE B.
    assert order.index("retrieve") < order.index("get")
    assert len(connector._engine.retrieve_calls) == 1
    assert connector._done_load == set()
    assert connector._failed_load == {23}
    assert connector._engine.unpinned == ["23"]
    assert connector._slot_admission.released == [0]
    assert len(connector._slot_store.invalidate_calls) == (0 if blob is None else 1)
    terminal_logs = [
        record.message
        for record in caplog.records
        if "SLOT sidecar load " in record.message
    ]
    assert len(terminal_logs) == 1
    assert "SLOT sidecar load failed" in terminal_logs[0]
    rendered_logs = "\n".join(record.message for record in caplog.records)
    assert _PAYLOAD.hex() not in rendered_logs
    assert str(list(range(8))) not in rendered_logs


def test_malformed_storage_object_is_invalidated_like_corrupt_aos1():
    order = []
    connector = _worker(order)

    @contextmanager
    def _malformed_borrow(key):
        connector._slot_store.get_calls.append(key)
        try:
            raise DSV4CheckpointCorruptionError(
                "LMCache sidecar object did not expose a tensor"
            )
            yield None
        finally:
            order.append("store-release")

    connector._slot_store.borrow = _malformed_borrow

    connector.start_load_kv(_metadata(_load_request()))

    assert connector._done_load == set()
    assert connector._failed_load == {23}
    assert connector._engine.unpinned == ["23"]
    assert connector._slot_store.invalidate_calls == [
        DSV4CheckpointKey(0x1234, _FINGERPRINT, 2, 1)
    ]
    assert connector._slot_admission.released == [0]


@pytest.mark.parametrize("retry_succeeds", [False, True])
def test_failed_corruption_invalidation_cannot_fake_later_commit(
    monkeypatch,
    retry_succeeds,
):
    order = []
    _patch_rpc_cuda(monkeypatch, order)
    connector = _worker(order)
    connector._slot_store.get_result = _crc_corrupt_sidecar_blob()
    connector._slot_store.invalidate_results = [False, retry_succeeds]

    connector.start_load_kv(_metadata(_load_request()))
    connector.start_load_kv(_metadata(_save_request()))

    key = DSV4CheckpointKey(0x1234, _FINGERPRINT, 2, 1)
    assert connector._slot_store.invalidate_calls == [key, key]
    if retry_succeeds:
        assert len(connector._slot_store.put_calls) == 1
        assert connector._done_sidecar_save == {17}
        assert connector._failed_sidecar_save == set()
        assert connector._slot_store.unresolved_corruption == set()
    else:
        assert connector._slot_store.put_calls == []
        assert connector._done_sidecar_save == set()
        assert connector._failed_sidecar_save == {17}
        assert connector._slot_store.unresolved_corruption == {key}


def test_load_staging_exhaustion_has_one_terminal_sidecar_log(monkeypatch, caplog):
    order = []
    caplog.set_level(logging.WARNING, logger="atom")
    connector = _worker(order, admission_available=False)
    connector._slot_store.get_result = _sidecar_blob()
    real_decode = connector._checkpoint_codec.decode_tensor

    def _record_decode(*args, **kwargs):
        order.append("decode")
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(connector._checkpoint_codec, "decode_tensor", _record_decode)

    connector.start_load_kv(_metadata(_load_request()))

    assert "retrieve" not in order
    assert "get" not in order
    assert "decode" not in order
    assert "restore" not in order
    assert connector._done_load == set()
    assert connector._failed_load == {23}
    assert connector._engine.unpinned == ["23"]
    assert sum("SLOT sidecar load " in record.message for record in caplog.records) == 1


def test_page_retrieve_failure_prevents_sidecar_fetch_and_unpins_once():
    order = []
    connector = _worker(order)
    connector._engine.retrieve_complete = False
    connector._slot_store.get_result = _sidecar_blob()

    connector.start_load_kv(_metadata(_load_request()))

    assert connector._slot_store.get_calls == []
    assert connector._done_load == set()
    assert connector._failed_load == {23}
    assert connector._engine.unpinned == ["23"]
    assert connector._slot_admission.released == [0]


def test_restore_failure_synchronizes_and_releases_admission():
    order = []
    connector = _worker(order)
    connector._slot_store.get_result = _sidecar_blob()
    connector._codec.restore_error = RuntimeError("restore failed")

    connector.start_load_kv(_metadata(_load_request()))

    assert order.index("restore") < order.index("sync") < order.index("release")
    assert connector._slot_admission.released == [0]
    assert connector._done_load == set()
    assert connector._failed_load == {23}
    assert connector._engine.unpinned == ["23"]


def test_restore_sync_failure_quarantines_batch_staging():
    order = []
    connector = _worker(order)
    connector._slot_store.get_result = _sidecar_blob()
    connector._create_slot_stream = lambda: _FakeTransferStream(
        order,
        synchronize_error=RuntimeError("restore synchronize failed"),
    )

    connector.start_load_kv(_metadata(_load_request()))

    assert connector._done_load == set()
    assert connector._failed_load == {23}
    assert connector._engine.unpinned == ["23"]
    assert connector._slot_admission.released == []
    assert connector._slot_admission.quarantined == [0]
    assert connector._slot_admission.num_free == 0
