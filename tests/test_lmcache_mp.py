# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import sys
import types
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import (
    KVTransferRegion,
    KVTransferTensors,
    LoadOperationId,
    SaveOperationId,
)
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.kv_transfer.offload.mp import backend as mp_connector


def _config(
    *,
    model_type: str = "test_model",
    tp: int = 2,
    pp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    dp: int = 1,
    enable_dp_attention: bool = False,
    role: str = "offload",
    extra: dict | None = None,
    kv_lora_rank: int | None = None,
) -> SimpleNamespace:
    hf_config = SimpleNamespace(
        model_type=model_type,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_size=2048,
        head_dim=128,
        kv_lora_rank=kv_lora_rank,
    )
    return SimpleNamespace(
        hf_config=hf_config,
        model="test/model",
        model_tag="test/model",
        kv_cache_block_size=4,
        kv_cache_dtype="fp8",
        index_cache_dtype="fp8",
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        decode_context_parallel_size=dcp,
        prefill_context_parallel_size=pcp,
        enable_dp_attention=enable_dp_attention,
        speculative_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        kv_transfer_config={
            "kv_connector": "lmcache_mp",
            "kv_role": role,
            "kv_connector_extra_config": extra or {},
        },
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pp": 2}, "does not support PP"),
        ({"dcp": 2}, "does not support DCP"),
        ({"pcp": 2}, "does not support PCP"),
        ({"dp": 2}, "TP-only"),
        ({"enable_dp_attention": True}, "TP-only"),
        ({"tp": 1.5}, "tensor_parallel_size must be an integer"),
    ],
)
def test_mp_config_rejects_unsupported_topologies(kwargs, message):
    with pytest.raises((NotImplementedError, ValueError), match=message):
        mp_connector._validate_mp_config(_config(**kwargs))


def test_mp_config_accepts_arbitrary_model_type():
    assert mp_connector._validate_mp_config(_config(model_type="ordinary_mha")) == (
        2,
        1,
    )


def test_mp_scheduler_is_not_a_dense_transport_scheduler():
    from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler

    assert issubclass(
        mp_connector.LMCacheMPConnectorScheduler,
        ChunkedOffloadSchedulerBase,
    )
    assert not issubclass(
        mp_connector.LMCacheMPConnectorScheduler,
        DenseOffloadScheduler,
    )


def test_mp_config_rejects_engine_driven_transfer(monkeypatch):
    monkeypatch.setenv("LMCACHE_MP_TRANSFER_MODE", " EnGiNe_DrIvEn ")
    with pytest.raises(NotImplementedError, match="multiple physical"):
        mp_connector._validate_mp_config(_config())

    monkeypatch.setenv("LMCACHE_MP_TRANSFER_MODE", "auto")
    with pytest.raises(NotImplementedError, match="multiple physical"):
        mp_connector._validate_mp_config(
            _config(extra={"lmcache.mp.mp_transfer_mode": " EnGiNe_DrIvEn "})
        )


@pytest.mark.parametrize(
    ("extra", "environment_mode", "expected"),
    [
        (
            {"lmcache.mp.mp_transfer_mode": " LmCaChe_DrIvEn "},
            "engine_driven",
            "lmcache_driven",
        ),
        ({}, " AuTo ", "auto"),
    ],
)
def test_worker_adapter_normalizes_transfer_mode(
    monkeypatch,
    extra,
    environment_mode,
    expected,
):
    class AtomMPParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class AtomMPWorkerAdapter:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    adapter_module = types.ModuleType("lmcache.integration.atom")
    adapter_module.AtomMPParallelConfig = AtomMPParallelConfig
    adapter_module.AtomMPWorkerAdapter = AtomMPWorkerAdapter
    monkeypatch.setitem(sys.modules, "lmcache.integration.atom", adapter_module)
    monkeypatch.setenv("LMCACHE_MP_TRANSFER_MODE", environment_mode)
    monkeypatch.setattr(mp_connector, "_model_namespace", lambda _config: "test")

    adapter = mp_connector._make_worker_adapter(_config(extra=extra), rank=1)

    assert adapter.transfer_mode == expected


def test_parallel_strategy_keeps_every_tp_rank(monkeypatch):
    class AtomMPParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    adapter_module = types.ModuleType("lmcache.integration.atom")
    adapter_module.AtomMPParallelConfig = AtomMPParallelConfig
    monkeypatch.setitem(
        sys.modules,
        "lmcache.integration.atom",
        adapter_module,
    )

    strategies = [
        mp_connector._parallel_strategy(_config(tp=8), rank) for rank in range(8)
    ]

    assert {strategy.world_size for strategy in strategies} == {8}
    assert {strategy.worker_id for strategy in strategies} == set(range(8))
    assert {strategy.tp_size for strategy in strategies} == {8}


def test_parallel_strategy_collapses_fully_replicated_mla(monkeypatch):
    class AtomMPParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    adapter_module = types.ModuleType("lmcache.integration.atom")
    adapter_module.AtomMPParallelConfig = AtomMPParallelConfig
    monkeypatch.setitem(
        sys.modules,
        "lmcache.integration.atom",
        adapter_module,
    )

    strategies = [
        mp_connector._parallel_strategy(_config(tp=8, kv_lora_rank=512), rank)
        for rank in range(8)
    ]

    assert {strategy.world_size for strategy in strategies} == {1}
    assert {strategy.worker_id for strategy in strategies} == {0}
    assert {strategy.tp_size for strategy in strategies} == {8}


def test_tp_rank_collapse_can_be_disabled_and_rejects_bad_values():
    assert (
        mp_connector._tp_replication_factor(
            _config(
                tp=8,
                kv_lora_rank=512,
                extra={"lmcache.mp.tp_rank_collapse": False},
            )
        )
        == 1
    )
    with pytest.raises(TypeError, match="true, false, or 'auto'"):
        mp_connector._tp_replication_factor(
            _config(extra={"lmcache.mp.tp_rank_collapse": 1})
        )


@pytest.mark.parametrize(
    ("kv_lora_rank", "expected_readers"),
    [(None, 1), (512, 8)],
)
def test_scheduler_reserves_locks_for_every_collapsed_tp_reader(
    monkeypatch,
    kv_lora_rank,
    expected_readers,
):
    @dataclass(frozen=True)
    class Key:
        num_kv_readers: int = 1

    class AtomMPParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class AtomMPSchedulerAdapter:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def _create_key(self, *_args, **_kwargs):
            return Key()

    adapter_module = types.ModuleType("lmcache.integration.atom")
    adapter_module.AtomMPParallelConfig = AtomMPParallelConfig
    adapter_module.AtomMPSchedulerAdapter = AtomMPSchedulerAdapter
    monkeypatch.setitem(sys.modules, "lmcache.integration.atom", adapter_module)
    monkeypatch.setattr(mp_connector, "_model_namespace", lambda _config: "test")

    adapter = mp_connector._make_scheduler_adapter(
        _config(tp=8, kv_lora_rank=kv_lora_rank)
    )
    key = adapter._create_key([], 0, 0, "req", None)

    assert key.num_kv_readers == expected_readers


def test_server_url_normalization_and_single_server_limit():
    assert mp_connector._server_urls(_config()) == ["tcp://localhost:5555"]
    assert mp_connector._server_urls(
        _config(extra={"lmcache.mp.host": "cache-host", "lmcache.mp.port": 6555})
    ) == ["tcp://cache-host:6555"]
    assert mp_connector._server_urls(
        _config(extra={"lmcache.mp.server_urls": "tcp://cache-host:6555"})
    ) == ["tcp://cache-host:6555"]

    with pytest.raises(NotImplementedError, match="exactly one"):
        mp_connector._server_urls(
            _config(extra={"lmcache.mp.server_urls": "host-a:1,host-b:2"})
        )
    with pytest.raises(NotImplementedError, match="exactly one"):
        mp_connector._server_urls(_config(extra={"lmcache.mp.server_urls": []}))
    with pytest.raises(ValueError, match=r"\[1, 65535\]"):
        mp_connector._server_urls(_config(extra={"lmcache.mp.port": 70000}))


def test_model_namespace_reuses_generic_page_namespace(monkeypatch):
    calls = []
    cfg = object()
    monkeypatch.setattr(mp_connector.offcfg, "build_lmcache_config", lambda _kvc: cfg)
    monkeypatch.setattr(
        mp_connector.offcfg,
        "build_page_namespace",
        lambda config, lmcache_cfg, world: (
            calls.append((config, lmcache_cfg, world)) or "model::atom-page-v2-layout"
        ),
    )

    config = _config(model_type="ordinary_mha", tp=4)
    assert mp_connector._model_namespace(config) == (
        "model::atom-page-v2-layout::lmcache-mp-v2"
    )
    assert calls == [(config, cfg, 4)]


def test_scheduler_validates_role_before_connecting(monkeypatch):
    config = _config(role="not-a-role")
    connected = False

    def connect(_config):
        nonlocal connected
        connected = True
        raise AssertionError("must not connect")

    monkeypatch.setattr(mp_connector, "_make_scheduler_adapter", connect)
    with pytest.raises(ValueError, match="invalid kv_role"):
        mp_connector.LMCacheMPConnectorScheduler(config)
    assert connected is False


def test_scheduler_closes_adapter_if_local_initialization_fails(monkeypatch):
    class Adapter:
        lmcache_tokens_per_chunk = 0

        def __init__(self):
            self.closed = False

        def shutdown(self):
            self.closed = True

    adapter = Adapter()
    monkeypatch.setattr(
        mp_connector,
        "_make_scheduler_adapter",
        lambda _config: adapter,
    )

    with pytest.raises(ValueError, match="LMCache chunk size"):
        mp_connector.LMCacheMPConnectorScheduler(_config())
    assert adapter.closed is True


def _transfer_tensors(*, tp_replication_factor: int = 1) -> KVTransferTensors:
    tensors = [
        torch.zeros(2, 4, 32, dtype=torch.float16),
        torch.zeros(2, 4, 32, dtype=torch.float16),
        torch.zeros(2, 4, 16, dtype=torch.uint8),
        torch.zeros(2, 4, 16, dtype=torch.uint8),
    ]
    roles = ["primary.0", "primary.1", "sidecar.0", "sidecar.1"]
    regions = [
        KVTransferRegion(
            base_addr=tensor.data_ptr(),
            total_bytes=tensor.numel() * tensor.element_size(),
            unit_bytes=tensor[0].numel() * tensor.element_size(),
            semantic_role=role,
        )
        for tensor, role in zip(tensors, roles, strict=True)
    ]
    return KVTransferTensors(
        block_regions=regions,
        slot_regions=[],
        num_blocks=2,
        block_tensor_views=tensors,
        tp_replication_factor=tp_replication_factor,
    )


def test_build_cache_views_groups_opaque_layouts():
    views = mp_connector._build_cache_views(_transfer_tensors(), num_blocks=2)

    assert list(views.tensors) == [
        "page.0.primary.0",
        "page.1.primary.1",
        "page.2.sidecar.0",
        "page.3.sidecar.1",
    ]
    assert views.layer_groups == ((0, 1), (2, 3))
    assert views.bytes_per_block == 2 * (4 * 32 * 2 + 4 * 16)


def test_build_cache_views_rejects_missing_or_bad_geometry():
    missing_view = _transfer_tensors()
    missing_view.block_tensor_views.pop()
    with pytest.raises(ValueError, match="one block_tensor_view per block region"):
        mp_connector._build_cache_views(missing_view, num_blocks=2)

    bad_geometry = _transfer_tensors()
    bad_geometry.block_regions[0].unit_bytes += 1
    with pytest.raises(ValueError, match="byte geometry mismatch"):
        mp_connector._build_cache_views(bad_geometry, num_blocks=2)

    noncontiguous = _transfer_tensors()
    noncontiguous.block_tensor_views[0] = torch.zeros(2, 32, 4).transpose(1, 2)
    assert not noncontiguous.block_tensor_views[0].is_contiguous()
    with pytest.raises(ValueError, match="non-empty and contiguous"):
        mp_connector._build_cache_views(noncontiguous, num_blocks=2)

    unsupported_rank = _transfer_tensors()
    unsupported_rank.block_tensor_views[0] = torch.zeros(
        2, 4, 4, 8, dtype=torch.float16
    )
    with pytest.raises(ValueError, match="physical_slots, opaque_width"):
        mp_connector._build_cache_views(unsupported_rank, num_blocks=2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_slots", 2),
        (
            "slot_regions",
            [KVTransferRegion(base_addr=0x1000, total_bytes=32, unit_bytes=16)],
        ),
        (
            "swa_block_regions",
            [KVTransferRegion(base_addr=0x2000, total_bytes=32, unit_bytes=16)],
        ),
        (
            "staging_region",
            KVTransferRegion(base_addr=0x3000, total_bytes=32, unit_bytes=16),
        ),
        ("gather_slot", lambda *_args: None),
        ("scatter_slot", lambda *_args: None),
        ("expected_full_slot_region_count", 1),
    ],
)
def test_build_cache_views_rejects_stateful_slot_layouts(field, value):
    transfer_tensors = _transfer_tensors()
    setattr(transfer_tensors, field, value)

    with pytest.raises(
        NotImplementedError,
        match=rf"PAGE-only layouts.*{field}",
    ):
        mp_connector._build_cache_views(transfer_tensors, num_blocks=2)


class _LookupAdapter:
    def __init__(self, results) -> None:
        self.results = deque(results)
        self.submissions = []
        self.freed = []
        self.cleaned = []
        self.ended = []

    def maybe_submit_lookup_request(self, request_id, token_ids):
        self.submissions.append((request_id, list(token_ids)))

    def check_lookup_result(self, request_id):
        if self.results:
            return self.results.popleft()
        return None

    def free_lookup_locks(self, **kwargs):
        self.freed.append(kwargs)

    def cleanup_lookup_result(self, request_id):
        self.cleaned.append(request_id)

    def end_session(self, request_id):
        self.ended.append(request_id)


def test_mp_lookup_releases_only_hbm_prefix_after_retrieve_handoff(monkeypatch):
    monkeypatch.setattr(mp_connector.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([None, 8])
    client = mp_connector._MPLookupClient(
        adapter,
        timeout=10.0,
        poll_interval=0.01,
    )

    assert client.lookup(list(range(8)), "req") == 8
    client.prepare_retrieve("req", 4)
    client.complete_retrieve("req", succeeded=False)

    # LMCache owns and releases [4, 8) once retrieve is submitted, including
    # on terminal failure. The scheduler releases only the HBM-resident prefix.
    assert [(call["start"], call["end"]) for call in adapter.freed] == [(0, 4)]
    assert client.hit_tokens("req") is None


def test_mp_lookup_releases_hit_suffix_outside_retrieve_range(monkeypatch):
    monkeypatch.setattr(mp_connector.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([12])
    client = mp_connector._MPLookupClient(
        adapter,
        timeout=10.0,
        poll_interval=0.01,
    )

    assert client.lookup(list(range(12)), "req") == 12
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(
        LMCacheReqMeta(
            req_id="req",
            token_ids=list(range(8)),
            block_ids=[1, 2],
            load_spec=LoadSpec(
                hbm_cached_tokens=2,
                lmcache_cached_tokens=12,
                can_load=True,
                transfer_end_tokens=8,
            ),
        )
    )
    scheduler = mp_connector.LMCacheMPConnectorScheduler.__new__(
        mp_connector.LMCacheMPConnectorScheduler
    )
    scheduler._lookup_client = client
    monkeypatch.setattr(
        ChunkedOffloadSchedulerBase,
        "build_connector_meta",
        lambda _self: metadata,
    )

    assert scheduler.build_connector_meta() is metadata

    # The worker owns only [2, 8). The scheduler releases both ranges that
    # will not be consumed by the retrieve.
    assert [(call["start"], call["end"]) for call in adapter.freed] == [
        (0, 2),
        (8, 12),
    ]


def test_mp_lookup_rejects_retrieve_beyond_lookup_hit(monkeypatch):
    monkeypatch.setattr(mp_connector.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([8])
    client = mp_connector._MPLookupClient(
        adapter,
        timeout=10.0,
        poll_interval=0.01,
    )

    assert client.lookup(list(range(12)), "req") == 8
    with pytest.raises(ValueError, match="retrieve end 12 exceeds lookup hit 8"):
        client.prepare_retrieve("req", 0, 12)

    assert adapter.freed == []
    assert client.hit_tokens("req") == 8


def test_mp_lookup_timeout_defers_cleanup_until_result(monkeypatch):
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(mp_connector.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(mp_connector.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([None, None])
    client = mp_connector._MPLookupClient(
        adapter,
        timeout=1.0,
        poll_interval=0.01,
    )

    assert client.lookup(list(range(8)), "req") == 0
    assert adapter.cleaned == []

    adapter.results.append(8)
    client.clear_lookup_status("req")
    assert [(call["start"], call["end"]) for call in adapter.freed] == [(0, 8)]
    assert adapter.cleaned == ["req"]


def test_mp_lookup_pending_cleanup_drops_adapter_bookkeeping(monkeypatch):
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(mp_connector.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(mp_connector.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([None, None])
    client = mp_connector._MPLookupClient(
        adapter,
        timeout=1.0,
        poll_interval=0.01,
    )

    assert client.lookup(list(range(8)), "req") == 0
    client.clear_lookup_status("req")

    assert adapter.cleaned == ["req"]
    assert client.hit_tokens("req") is None


def test_full_prompt_hit_retrieves_chunk_but_recomputes_last_token(monkeypatch):
    monkeypatch.setattr(mp_connector.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([8])
    lookup = mp_connector._MPLookupClient(
        adapter,
        timeout=1.0,
        poll_interval=0.01,
    )
    scheduler = mp_connector.LMCacheMPConnectorScheduler.__new__(
        mp_connector.LMCacheMPConnectorScheduler
    )
    scheduler._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        scheduler,
        _config(),
        chunk_size=8,
        lookup_client=lookup,
    )
    scheduler._min_load_tokens = 0
    seq = SimpleNamespace(
        id=7,
        num_prompt_tokens=8,
        num_cached_tokens=0,
        token_ids=list(range(8)),
        block_table=[10, 11],
    )

    assert scheduler.get_num_new_matched_tokens(seq) == (7, True)
    assert scheduler._load_specs["7"].lmcache_cached_tokens == 7
    assert scheduler._load_specs["7"].transfer_end_tokens == 8

    scheduler.update_state_after_alloc(seq)
    request = scheduler.build_connector_meta().requests[0]
    assert request.token_ids == list(range(8))
    assert request.load_spec.lmcache_cached_tokens == 7
    assert request.load_spec.transfer_end_tokens == 8
    assert seq.offload_loaded_tokens == 7

    assert scheduler.load_finished(request.load_operation) is True
    assert lookup.hit_tokens("7") is None


def test_stale_load_failure_does_not_release_current_generation_locks():
    adapter = _LookupAdapter([])
    lookup = mp_connector._MPLookupClient(
        adapter,
        timeout=1.0,
        poll_interval=0.01,
    )
    scheduler = mp_connector.LMCacheMPConnectorScheduler.__new__(
        mp_connector.LMCacheMPConnectorScheduler
    )
    scheduler._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        scheduler,
        _config(),
        chunk_size=8,
        lookup_client=lookup,
    )
    seq = SimpleNamespace(id=7)
    current = LoadOperationId(req_id=7, generation=2)
    stale = LoadOperationId(req_id=7, generation=1)
    scheduler._active_load_operations["7"] = (seq, current)
    lookup._lookups["7"] = mp_connector._LookupState(
        token_ids=list(range(8)),
        hit=8,
        retrieve_start=0,
    )

    assert scheduler.load_failed(stale) is False
    assert adapter.freed == []
    assert lookup.hit_tokens("7") == 8

    assert scheduler.load_failed(current) is True
    assert adapter.freed == []
    assert lookup.hit_tokens("7") is None


@dataclass
class _FakeLoadStoreOp:
    token_ids: list[int]
    block_ids: list[list[int]]
    start: int = 0
    end: int = 0


@dataclass
class _FakeParallelConfig:
    world_size: int
    worker_id: int
    tp_size: int


@dataclass
class _FakeEngineGroupInfo:
    engine_group_id: int
    layer_indices: tuple[int, ...]
    tokens_per_block: int


class _WorkerFuture:
    def __init__(self, result=True) -> None:
        self.ready = False
        self.value = result
        self.query_error = None

    def query(self):
        if self.query_error is not None:
            raise self.query_error
        return self.ready

    def result(self, timeout=None):
        if not self.ready:
            raise TimeoutError("future is not ready")
        return self.value


@pytest.fixture
def fake_lmcache_modules(monkeypatch):
    lmcache = types.ModuleType("lmcache")
    lmcache.__path__ = []
    integration = types.ModuleType("lmcache.integration")
    integration.__path__ = []
    atom = types.ModuleType("lmcache.integration.atom")
    atom.AtomMPParallelConfig = _FakeParallelConfig
    atom.AtomMPTransferSpec = _FakeLoadStoreOp
    v1 = types.ModuleType("lmcache.v1")
    v1.__path__ = []
    multiprocess = types.ModuleType("lmcache.v1.multiprocess")
    multiprocess.__path__ = []
    group_view = types.ModuleType("lmcache.v1.multiprocess.group_view")
    group_view.EngineGroupInfo = _FakeEngineGroupInfo
    modules = {
        "lmcache": lmcache,
        "lmcache.integration": integration,
        "lmcache.integration.atom": atom,
        "lmcache.v1": v1,
        "lmcache.v1.multiprocess": multiprocess,
        "lmcache.v1.multiprocess.group_view": group_view,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


class _WorkerAdapter:
    lmcache_tokens_per_chunk = 8

    def __init__(self) -> None:
        self.is_healthy = True
        self.registered = None
        self.groups = None
        self.loads = []
        self.saves = []
        self.shutdown_called = False

    def register_kv_caches(self, tensors, *, engine_group_infos):
        self.registered = tensors
        self.groups = engine_group_infos

    def submit_retrieve_request(self, request_id, op, event):
        future = _WorkerFuture()
        self.loads.append((request_id, op, event, future))
        return future

    def submit_store_request(self, request_id, op, event):
        future = _WorkerFuture()
        self.saves.append((request_id, op, event, future))
        return future

    def shutdown(self):
        self.shutdown_called = True


def _worker(adapter: _WorkerAdapter) -> mp_connector.LMCacheMPConnector:
    worker = mp_connector.LMCacheMPConnector(_config())
    worker._adapter = adapter
    worker.chunk_size = 8
    return worker


def _finish_load(
    worker: mp_connector.LMCacheMPConnector,
    operation_id: str,
    *,
    result: bool = True,
) -> None:
    future = worker._pending_loads[operation_id].future
    future.value = result
    future.ready = True


def _finish_save(
    worker: mp_connector.LMCacheMPConnector,
    operation_id: str,
    *,
    result: bool = True,
) -> None:
    future = worker._pending_saves[operation_id].future
    future.value = result
    future.ready = True


def test_worker_uses_transfer_boundary_and_exact_completion(fake_lmcache_modules):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    operation = LoadOperationId(req_id=5, generation=3)
    request = LMCacheReqMeta(
        req_id=5,
        token_ids=list(range(8)),
        block_ids=[10, 11],
        load_spec=LoadSpec(
            hbm_cached_tokens=0,
            lmcache_cached_tokens=7,
            can_load=True,
            transfer_end_tokens=8,
        ),
        load_operation=operation,
    )

    worker._submit_load(request, object())
    submitted = adapter.loads[0][1]
    assert submitted.start == 0
    assert submitted.end == 8
    assert submitted.block_ids == [[10, 11]]

    _finish_load(worker, "load:5:3")
    assert worker.get_finished().finished_loading == {operation}


def test_worker_reports_failed_load_from_future(fake_lmcache_modules):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    operation = LoadOperationId(req_id=6, generation=4)
    request = LMCacheReqMeta(
        req_id=6,
        token_ids=list(range(8)),
        block_ids=[20, 21],
        load_spec=LoadSpec(0, 8, can_load=True),
        load_operation=operation,
    )
    worker._submit_load(request, object())
    _finish_load(worker, "load:6:4", result=False)

    output = worker.get_finished()
    assert output.finished_loading == set()
    assert output.failed_loading == {operation}


def test_worker_future_query_exception_preserves_inflight_transfers(
    fake_lmcache_modules,
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    load_operation = LoadOperationId(req_id=10, generation=1)
    save_operation = SaveOperationId(req_id=11, generation=2)
    worker._submit_load(
        LMCacheReqMeta(
            req_id=10,
            token_ids=list(range(8)),
            block_ids=[50, 51],
            load_spec=LoadSpec(0, 8, can_load=True),
            load_operation=load_operation,
        ),
        object(),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=11,
            token_ids=list(range(8)),
            block_ids=[60, 61],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        object(),
    )

    load_future = worker._pending_loads["load:10:1"].future
    save_future = worker._pending_saves["save:11:2"].future
    load_future.query_error = RuntimeError("load query failed")
    save_future.query_error = RuntimeError("save query failed")
    output = worker.get_finished()

    assert output.finished_loading == set()
    assert output.failed_loading == set()
    assert output.finished_saving == set()
    assert set(worker._pending_loads) == {"load:10:1"}
    assert set(worker._pending_saves) == {"save:11:2"}

    load_future.query_error = None
    save_future.query_error = None
    _finish_load(worker, "load:10:1")
    _finish_save(worker, "save:11:2", result=False)
    output = worker.get_finished()

    assert output.finished_loading == {load_operation}
    assert output.failed_loading == set()
    # A failed store loses this cache opportunity but is terminal and safe to
    # release, matching the legacy connector's save-failure semantics.
    assert output.finished_saving == {save_operation}


def test_worker_unhealthy_preserves_pending_until_device_futures_are_terminal(
    fake_lmcache_modules,
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    load_operation = LoadOperationId(req_id=12, generation=1)
    save_operation = SaveOperationId(req_id=13, generation=1)
    worker._submit_load(
        LMCacheReqMeta(
            req_id=12,
            token_ids=list(range(8)),
            block_ids=[70, 71],
            load_spec=LoadSpec(0, 8, can_load=True),
            load_operation=load_operation,
        ),
        object(),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=13,
            token_ids=list(range(8)),
            block_ids=[80, 81],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        object(),
    )

    adapter.is_healthy = False
    output = worker.get_finished()

    assert output.finished_loading == set()
    assert output.failed_loading == set()
    assert output.finished_saving == set()
    assert set(worker._pending_loads) == {"load:12:1"}
    assert set(worker._pending_saves) == {"save:13:1"}

    _finish_load(worker, "load:12:1", result=False)
    _finish_save(worker, "save:13:1", result=False)
    output = worker.get_finished()

    assert output.failed_loading == {load_operation}
    assert output.finished_saving == {save_operation}
    assert worker._pending_loads == {}
    assert worker._pending_saves == {}


def test_worker_pre_submit_drops_are_immediately_terminal(
    fake_lmcache_modules,
    monkeypatch,
):
    adapter = _WorkerAdapter()
    monkeypatch.setattr(
        adapter,
        "submit_retrieve_request",
        lambda _request_id, _op, _event: None,
    )
    monkeypatch.setattr(
        adapter,
        "submit_store_request",
        lambda _request_id, _op, _event: None,
    )
    worker = _worker(adapter)
    load_operation = LoadOperationId(req_id=14, generation=1)
    save_operation = SaveOperationId(req_id=15, generation=1)
    worker._submit_load(
        LMCacheReqMeta(
            req_id=14,
            token_ids=list(range(8)),
            block_ids=[90, 91],
            load_spec=LoadSpec(0, 8, can_load=True),
            load_operation=load_operation,
        ),
        object(),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=15,
            token_ids=list(range(8)),
            block_ids=[100, 101],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        object(),
    )

    output = worker.get_finished()

    assert output.failed_loading == {load_operation}
    assert output.finished_saving == {save_operation}
    assert worker._pending_loads == {}
    assert worker._pending_saves == {}


def test_worker_save_slices_chunk_blocks_and_preserves_operation(
    fake_lmcache_modules,
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    operation = SaveOperationId(req_id=8, generation=2)
    request = LMCacheReqMeta(
        req_id=8,
        token_ids=list(range(16)),
        block_ids=[30, 31, 32, 33],
        save_spec=SaveSpec(skip_leading_tokens=8),
        save_operation=operation,
    )

    worker._submit_save(request, object())
    submitted = adapter.saves[0][1]
    assert submitted.start == 8
    assert submitted.end == 16
    assert submitted.block_ids == [[32, 33]]

    _finish_save(worker, "save:8:2")
    assert worker.get_finished().finished_saving == {operation}


def test_non_writer_completes_save_without_submitting(fake_lmcache_modules):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    worker._is_kv_writer = False
    operation = SaveOperationId(req_id=9, generation=2)
    request = LMCacheReqMeta(
        req_id=9,
        token_ids=list(range(8)),
        block_ids=[40, 41],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=operation,
    )

    worker._submit_save(request, object())

    assert adapter.saves == []
    assert worker.get_finished().finished_saving == {operation}
    with pytest.raises(RuntimeError, match="duplicate LMCache MP save"):
        worker._submit_save(request, object())


def test_worker_tracks_two_load_generations_for_one_raw_request(
    fake_lmcache_modules,
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    operations = [
        LoadOperationId(req_id=20, generation=1),
        LoadOperationId(req_id=20, generation=2),
    ]
    for operation in operations:
        worker._submit_load(
            LMCacheReqMeta(
                req_id=20,
                token_ids=list(range(8)),
                block_ids=[70, 71],
                load_spec=LoadSpec(0, 8, can_load=True),
                load_operation=operation,
            ),
            object(),
        )

    assert set(worker._pending_loads) == {"load:20:1", "load:20:2"}

    _finish_load(worker, "load:20:2")
    assert worker.get_finished().finished_loading == {operations[1]}
    assert set(worker._pending_loads) == {"load:20:1"}
    _finish_load(worker, "load:20:1")
    assert worker.get_finished().finished_loading == {operations[0]}


def test_worker_tracks_two_save_generations_for_one_raw_request(
    fake_lmcache_modules,
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    operations = [
        SaveOperationId(req_id=21, generation=1),
        SaveOperationId(req_id=21, generation=2),
    ]
    for operation in operations:
        worker._submit_save(
            LMCacheReqMeta(
                req_id=21,
                token_ids=list(range(8)),
                block_ids=[80, 81],
                save_spec=SaveSpec(skip_leading_tokens=0),
                save_operation=operation,
            ),
            object(),
        )

    assert set(worker._pending_saves) == {"save:21:1", "save:21:2"}

    _finish_save(worker, "save:21:1")
    assert worker.get_finished().finished_saving == {operations[0]}
    assert set(worker._pending_saves) == {"save:21:2"}
    _finish_save(worker, "save:21:2")
    assert worker.get_finished().finished_saving == {operations[1]}


def test_worker_load_and_save_coexist_for_same_raw_request(fake_lmcache_modules):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    load_operation = LoadOperationId(req_id=22, generation=4)
    save_operation = SaveOperationId(req_id=22, generation=5)
    worker._submit_load(
        LMCacheReqMeta(
            req_id=22,
            token_ids=list(range(8)),
            block_ids=[90, 91],
            load_spec=LoadSpec(0, 8, can_load=True),
            load_operation=load_operation,
        ),
        object(),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=22,
            token_ids=list(range(8)),
            block_ids=[90, 91],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        object(),
    )

    assert set(worker._pending_loads) == {"load:22:4"}
    assert set(worker._pending_saves) == {"save:22:5"}
    _finish_load(worker, "load:22:4")
    _finish_save(worker, "save:22:5")
    output = worker.get_finished()
    assert output.finished_loading == {load_operation}
    assert output.finished_saving == {save_operation}


@pytest.mark.parametrize("kind", ["load", "save"])
def test_worker_rejects_exact_operation_replay(fake_lmcache_modules, kind):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    operation = (
        LoadOperationId(req_id=23, generation=6)
        if kind == "load"
        else SaveOperationId(req_id=23, generation=6)
    )
    request = LMCacheReqMeta(
        req_id=23,
        token_ids=list(range(8)),
        block_ids=[100, 101],
        load_spec=LoadSpec(0, 8, can_load=True) if kind == "load" else None,
        save_spec=SaveSpec(skip_leading_tokens=0) if kind == "save" else None,
        load_operation=operation if kind == "load" else None,
        save_operation=operation if kind == "save" else None,
    )
    submit = worker._submit_load if kind == "load" else worker._submit_save
    operation_id = f"{kind}:23:6"

    submit(request, object())
    completed = (
        worker._completed_load_operations
        if kind == "load"
        else worker._completed_save_operations
    )
    assert operation_id not in completed
    with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
        submit(request, object())

    if kind == "load":
        _finish_load(worker, operation_id)
    else:
        _finish_save(worker, operation_id)
    worker.get_finished()
    with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
        submit(request, object())


@pytest.mark.parametrize("kind", ["load", "save"])
def test_worker_rejects_duplicate_while_operation_is_submitting(
    fake_lmcache_modules,
    monkeypatch,
    kind,
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    operation = (
        LoadOperationId(req_id=27, generation=1)
        if kind == "load"
        else SaveOperationId(req_id=27, generation=1)
    )
    request = LMCacheReqMeta(
        req_id=27,
        token_ids=list(range(8)),
        block_ids=[140, 141],
        load_spec=LoadSpec(0, 8, can_load=True) if kind == "load" else None,
        save_spec=SaveSpec(skip_leading_tokens=0) if kind == "save" else None,
        load_operation=operation if kind == "load" else None,
        save_operation=operation if kind == "save" else None,
    )
    submit = worker._submit_load if kind == "load" else worker._submit_save
    adapter_method_name = (
        "submit_retrieve_request" if kind == "load" else "submit_store_request"
    )
    original_adapter_submit = getattr(adapter, adapter_method_name)

    def submit_with_replay(*args):
        with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
            submit(request, object())
        return original_adapter_submit(*args)

    monkeypatch.setattr(adapter, adapter_method_name, submit_with_replay)

    submit(request, object())

    submitting = (
        worker._submitting_loads if kind == "load" else worker._submitting_saves
    )
    pending = worker._pending_loads if kind == "load" else worker._pending_saves
    completed = (
        worker._completed_load_operations
        if kind == "load"
        else worker._completed_save_operations
    )
    assert submitting == set()
    assert set(pending) == {f"{kind}:27:1"}
    assert completed == set()


def test_worker_operation_tombstone_limit_is_4096():
    assert mp_connector._OPERATION_TOMBSTONE_LIMIT == 4096


@pytest.mark.parametrize("kind", ["load", "save"])
def test_worker_terminal_operation_tombstones_are_bounded(
    fake_lmcache_modules,
    monkeypatch,
    kind,
):
    monkeypatch.setattr(mp_connector, "_OPERATION_TOMBSTONE_LIMIT", 2)
    adapter = _WorkerAdapter()
    worker = _worker(adapter)

    def request(generation):
        operation = (
            LoadOperationId(req_id=24, generation=generation)
            if kind == "load"
            else SaveOperationId(req_id=24, generation=generation)
        )
        return LMCacheReqMeta(
            req_id=24,
            token_ids=list(range(8)),
            block_ids=[110, 111],
            load_spec=LoadSpec(0, 8, can_load=True) if kind == "load" else None,
            save_spec=SaveSpec(skip_leading_tokens=0) if kind == "save" else None,
            load_operation=operation if kind == "load" else None,
            save_operation=operation if kind == "save" else None,
        )

    submit = worker._submit_load if kind == "load" else worker._submit_save
    for generation in (1, 2, 3):
        submit(request(generation), object())
        operation_id = f"{kind}:24:{generation}"
        if kind == "load":
            _finish_load(worker, operation_id)
        else:
            _finish_save(worker, operation_id)
        worker.get_finished()

    seen = (
        worker._completed_load_operations
        if kind == "load"
        else worker._completed_save_operations
    )
    order = (
        worker._completed_load_operation_order
        if kind == "load"
        else worker._completed_save_operation_order
    )
    assert mp_connector._OPERATION_TOMBSTONE_LIMIT == 2
    assert seen == {f"{kind}:24:2", f"{kind}:24:3"}
    assert list(order) == [f"{kind}:24:2", f"{kind}:24:3"]

    with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
        submit(request(3), object())
    submit(request(1), object())
    pending = worker._pending_loads if kind == "load" else worker._pending_saves
    assert set(pending) == {f"{kind}:24:1"}


def test_worker_immediate_operation_tombstones_reject_replay(
    fake_lmcache_modules,
    monkeypatch,
):
    adapter = _WorkerAdapter()

    def fail_load(_request_id, _op, _event):
        raise RuntimeError("submission failed")

    monkeypatch.setattr(adapter, "submit_retrieve_request", fail_load)
    worker = _worker(adapter)
    load_operation = LoadOperationId(req_id=25, generation=1)
    save_operation = SaveOperationId(req_id=26, generation=1)
    load_request = LMCacheReqMeta(
        req_id=25,
        token_ids=list(range(8)),
        block_ids=[120, 121],
        load_spec=LoadSpec(0, 8, can_load=True),
        load_operation=load_operation,
    )
    save_request = LMCacheReqMeta(
        req_id=26,
        token_ids=list(range(4)),
        block_ids=[130],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=save_operation,
    )

    worker._submit_load(load_request, object())
    worker._submit_save(save_request, object())

    assert worker._completed_load_operations == {"load:25:1"}
    assert list(worker._completed_load_operation_order) == ["load:25:1"]
    assert worker._completed_save_operations == {"save:26:1"}
    assert list(worker._completed_save_operation_order) == ["save:26:1"]
    output = worker.get_finished()
    assert output.failed_loading == {load_operation}
    assert output.finished_saving == {save_operation}

    with pytest.raises(RuntimeError, match="duplicate LMCache MP load"):
        worker._submit_load(load_request, object())
    with pytest.raises(RuntimeError, match="duplicate LMCache MP save"):
        worker._submit_save(save_request, object())


def test_registers_multiple_layouts_as_views_of_one_engine_group(
    fake_lmcache_modules,
    monkeypatch,
):
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    dist = types.ModuleType("aiter.dist")
    dist.__path__ = []
    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: SimpleNamespace(rank_in_group=0)
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.dist", dist)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)

    adapter = _WorkerAdapter()
    monkeypatch.setattr(
        mp_connector,
        "_make_worker_adapter",
        lambda _config, _rank: adapter,
    )
    worker = mp_connector.LMCacheMPConnector(_config(model_type="ordinary_mha"))
    transfer_tensors = _transfer_tensors()
    worker.register_kv_caches(
        {},
        transfer_tensors=transfer_tensors,
        num_blocks=2,
    )

    assert list(adapter.registered) == [
        "page.0.primary.0",
        "page.1.primary.1",
        "page.2.sidecar.0",
        "page.3.sidecar.1",
    ]
    assert [group.engine_group_id for group in adapter.groups] == [0, 0]
    assert [group.layer_indices for group in adapter.groups] == [(0, 1), (2, 3)]
    assert [group.tokens_per_block for group in adapter.groups] == [4, 4]


@pytest.mark.parametrize(("rank", "is_writer"), [(0, True), (3, False)])
def test_registration_collapses_only_backend_declared_tp_replicas(
    fake_lmcache_modules,
    monkeypatch,
    rank,
    is_writer,
):
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    dist = types.ModuleType("aiter.dist")
    dist.__path__ = []
    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: SimpleNamespace(rank_in_group=rank)
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.dist", dist)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)

    adapter = _WorkerAdapter()
    monkeypatch.setattr(
        mp_connector,
        "_make_worker_adapter",
        lambda _config, _rank: adapter,
    )
    worker = mp_connector.LMCacheMPConnector(_config(tp=8, kv_lora_rank=512))
    worker.register_kv_caches(
        {},
        transfer_tensors=_transfer_tensors(tp_replication_factor=8),
        num_blocks=2,
    )

    assert worker._is_kv_writer is is_writer
    assert adapter.registered is not None


def test_registration_rejects_rank_collapse_without_backend_declaration(
    fake_lmcache_modules,
    monkeypatch,
):
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    dist = types.ModuleType("aiter.dist")
    dist.__path__ = []
    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: SimpleNamespace(rank_in_group=0)
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.dist", dist)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)

    worker = mp_connector.LMCacheMPConnector(
        _config(
            tp=8,
            extra={"lmcache.mp.tp_rank_collapse": True},
        )
    )
    with pytest.raises(ValueError, match="did not declare.*fully replicated"):
        worker.register_kv_caches(
            {},
            transfer_tensors=_transfer_tensors(),
            num_blocks=2,
        )


def test_factory_registers_lmcache_mp_alias_without_pd_staging():
    assert KVConnectorFactory.canonical_name("LMCacheMPConnector") == "lmcache_mp"
    assert (
        KVConnectorFactory.topology_uses_pd_staging(
            {"kv_connector": "LMCacheMPConnector", "kv_role": "offload"}
        )
        is False
    )
