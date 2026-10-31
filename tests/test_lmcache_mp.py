# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import sys
import threading
import types
from collections import deque
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVTransferRegion,
    KVTransferTensors,
    LoadOperationId,
    PageRegion,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
    ChunkedOffloadSchedulerBase,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    NativeStateTransfer,
    SaveSpec,
)
from atom.kv_transfer.offload.mp import deployment, page_views, transfer
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp import worker as mp_worker


def _step_event():
    """The step's recorded event, as `start_load_kv` hands to a submit."""

    return object()


def _config(
    *,
    model_type: str = "test_model",
    tp: int = 2,
    pp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    dp: int = 1,
    dp_local: int | None = None,
    dp_rank: int = 0,
    enable_dp_attention: bool = False,
    enable_expert_parallel: bool = False,
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
        enable_expert_parallel=enable_expert_parallel,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            data_parallel_size_local=dp if dp_local is None else dp_local,
            data_parallel_rank=dp_rank,
        ),
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
        ({"dp": 2, "dp_local": 1}, "only within one host"),
        ({"tp": 1.5}, "tensor_parallel_size must be an integer"),
    ],
)
def test_mp_config_rejects_unsupported_topologies(kwargs, message):
    with pytest.raises((NotImplementedError, ValueError), match=message):
        deployment._validate_mp_config(_config(**kwargs))


def test_mp_config_accepts_arbitrary_model_type():
    assert deployment._validate_mp_config(_config(model_type="ordinary_mha")) == (
        2,
        1,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dp": 2, "dp_rank": 1},
        {"dp": 2, "enable_dp_attention": True},
        {
            "tp": 1,
            "dp": 8,
            "enable_dp_attention": True,
            "enable_expert_parallel": True,
        },
    ],
)
def test_mp_config_accepts_single_host_dp_and_dpa(kwargs):
    assert deployment._validate_mp_config(_config(**kwargs))[1] == 1


def test_mp_scheduler_is_not_a_dense_transport_scheduler():
    from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler

    assert issubclass(
        mp_scheduler.LMCacheMPConnectorScheduler,
        ChunkedOffloadSchedulerBase,
    )
    assert not issubclass(
        mp_scheduler.LMCacheMPConnectorScheduler,
        DenseOffloadScheduler,
    )


def test_mp_config_rejects_engine_driven_transfer(monkeypatch):
    monkeypatch.setenv("LMCACHE_MP_TRANSFER_MODE", " EnGiNe_DrIvEn ")
    with pytest.raises(NotImplementedError, match="multiple physical"):
        deployment._validate_mp_config(_config())

    monkeypatch.setenv("LMCACHE_MP_TRANSFER_MODE", "auto")
    with pytest.raises(NotImplementedError, match="multiple physical"):
        deployment._validate_mp_config(
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
    monkeypatch.setattr(
        deployment, "_model_namespace", lambda _config, **_kwargs: "test"
    )

    adapter = deployment._make_worker_adapter(_config(extra=extra), rank=1)

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
        deployment._parallel_strategy(_config(tp=8, dp=2), rank) for rank in range(8)
    ]

    assert {strategy.world_size for strategy in strategies} == {8}
    assert {strategy.worker_id for strategy in strategies} == set(range(8))
    assert {strategy.tp_size for strategy in strategies} == {8}


def test_parallel_strategy_uses_one_worker_per_dpa_engine(monkeypatch):
    class AtomMPParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    adapter_module = types.ModuleType("lmcache.integration.atom")
    adapter_module.AtomMPParallelConfig = AtomMPParallelConfig
    monkeypatch.setitem(sys.modules, "lmcache.integration.atom", adapter_module)

    strategy = deployment._parallel_strategy(
        _config(
            tp=1,
            dp=8,
            dp_rank=5,
            enable_dp_attention=True,
            enable_expert_parallel=True,
        ),
        0,
    )

    assert strategy.world_size == 1
    assert strategy.worker_id == 0
    assert strategy.tp_size == 1


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
        deployment._parallel_strategy(_config(tp=8, kv_lora_rank=512), rank)
        for rank in range(8)
    ]

    assert {strategy.world_size for strategy in strategies} == {1}
    assert {strategy.worker_id for strategy in strategies} == {0}
    assert {strategy.tp_size for strategy in strategies} == {8}


def test_auto_rank_collapse_is_off_for_a_draft_with_its_own_pool(monkeypatch):
    """A DSpark draft whose backend owns a KV pool appends per-rank PAGE
    regions, so a replicated MLA target no longer collapses."""
    from atom.utils import selector

    owns_pool = {"value": True}
    monkeypatch.setattr(selector, "attn_family", lambda _hf: "draft-family")
    monkeypatch.setattr(
        selector,
        "get_attn_backend",
        lambda _family: SimpleNamespace(DRAFT_OWNS_KV_POOL=owns_pool["value"]),
    )
    config = _config(tp=8, kv_lora_rank=512)
    assert deployment._tp_replication_factor(config) == 8

    config.speculative_config = SimpleNamespace(
        method="dspark", draft_model_hf_config=SimpleNamespace()
    )
    assert deployment._tp_replication_factor(config) == 1

    owns_pool["value"] = False
    assert deployment._tp_replication_factor(config) == 8
    config.speculative_config.method = "mtp"
    owns_pool["value"] = True
    assert deployment._tp_replication_factor(config) == 8


def test_auto_rank_collapse_distinguishes_glm52_mla_from_minimax_m3_gqa():
    glm52 = _config(model_type="glm_moe_dsa", tp=8, kv_lora_rank=512)
    minimax = _config(model_type="minimax_m3_vl", tp=8)
    minimax.hf_config.architectures = ["MiniMaxM3SparseForConditionalGeneration"]
    minimax.hf_config.text_config = SimpleNamespace(
        num_key_value_heads=4,
        # Stay fail-closed for this model family even if a wrapper grows an
        # unrelated field with the same name in the future.
        kv_lora_rank=512,
    )

    assert deployment._tp_replication_factor(glm52) == 8
    assert deployment._tp_replication_factor(minimax) == 1


def _kimi_k3_config(**kwargs):
    k3 = _config(model_type="kimi_k3", tp=8, **kwargs)
    k3.hf_config.text_config = SimpleNamespace(
        model_type="kimi_linear",
        kv_lora_rank=512,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_hidden_layers=2,
        hidden_size=2048,
        head_dim=128,
    )
    return k3


def test_auto_rank_collapse_keeps_kimi_k3_per_rank():
    # K3's MLA KV is replicated, but the KDA checkpoints ride in the same PAGE
    # units and hold TP-sharded heads, so the unit as a whole is not.
    assert deployment._tp_replication_factor(_kimi_k3_config()) == 1


def test_tp_rank_collapse_can_be_disabled_and_rejects_bad_values():
    assert (
        deployment._tp_replication_factor(
            _config(
                tp=8,
                kv_lora_rank=512,
                extra={"lmcache.mp.tp_rank_collapse": False},
            )
        )
        == 1
    )
    with pytest.raises(TypeError, match="true, false, or 'auto'"):
        deployment._tp_replication_factor(
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
    monkeypatch.setattr(
        deployment, "_model_namespace", lambda _config, **_kwargs: "test"
    )

    adapter = mp_scheduler._make_scheduler_adapter(
        _config(tp=8, kv_lora_rank=kv_lora_rank)
    )
    key = adapter._create_key([], 0, 0, "req", None)

    assert key.num_kv_readers == expected_readers


def test_server_url_normalization_and_single_server_limit():
    assert deployment._server_urls(_config()) == ["tcp://localhost:5555"]
    assert deployment._server_urls(
        _config(extra={"lmcache.mp.host": "cache-host", "lmcache.mp.port": 6555})
    ) == ["tcp://cache-host:6555"]
    assert deployment._server_urls(
        _config(extra={"lmcache.mp.server_urls": "tcp://cache-host:6555"})
    ) == ["tcp://cache-host:6555"]

    with pytest.raises(NotImplementedError, match="exactly one"):
        deployment._server_urls(
            _config(extra={"lmcache.mp.server_urls": "host-a:1,host-b:2"})
        )
    with pytest.raises(NotImplementedError, match="exactly one"):
        deployment._server_urls(_config(extra={"lmcache.mp.server_urls": []}))
    with pytest.raises(ValueError, match=r"\[1, 65535\]"):
        deployment._server_urls(_config(extra={"lmcache.mp.port": 70000}))


def test_model_namespace_reuses_generic_page_namespace(monkeypatch):
    calls = []
    cfg = object()
    monkeypatch.setattr(deployment.offcfg, "build_lmcache_config", lambda _kvc: cfg)
    monkeypatch.setattr(
        deployment.offcfg,
        "build_page_namespace",
        lambda config, lmcache_cfg, world: (
            calls.append((config, lmcache_cfg, world)) or "model::atom-page-v2-layout"
        ),
    )

    config = _config(model_type="ordinary_mha", tp=4)
    assert deployment._model_namespace(config) == (
        "model::atom-page-v2-layout::lmcache-mp-v3"
    )
    assert calls == [(config, cfg, 4)]


def test_dp_replicas_share_model_namespace_but_not_request_sessions(monkeypatch):
    cfg = object()
    monkeypatch.setattr(deployment.offcfg, "build_lmcache_config", lambda _kvc: cfg)
    monkeypatch.setattr(
        deployment.offcfg,
        "build_page_namespace",
        lambda _config, _lmcache_cfg, world: f"page-tp{world}",
    )
    rank0 = _config(tp=4, dp=2, dp_rank=0)
    rank1 = _config(tp=4, dp=2, dp_rank=1)

    assert deployment._model_namespace(rank0) == deployment._model_namespace(rank1)
    assert deployment._mp_session_id(rank0, 0) == "atom-offload-dp0:0"
    assert deployment._mp_session_id(rank1, 0) == "atom-offload-dp1:0"


def test_scheduler_validates_role_before_connecting(monkeypatch):
    config = _config(role="not-a-role")
    connected = False

    def connect(_config, **_kwargs):
        nonlocal connected
        connected = True
        raise AssertionError("must not connect")

    monkeypatch.setattr(mp_scheduler, "_make_scheduler_adapter", connect)
    with pytest.raises(ValueError, match="invalid kv_role"):
        mp_scheduler.LMCacheMPConnectorScheduler(config)
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
        mp_scheduler,
        "_make_scheduler_adapter",
        lambda _config, **_kwargs: adapter,
    )

    with pytest.raises(ValueError, match="LMCache chunk size"):
        mp_scheduler.LMCacheMPConnectorScheduler(_config())
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
    transfer = KVTransferTensors(
        pages=[
            PageRegion(region, tensor)
            for region, tensor in zip(regions, tensors, strict=True)
        ],
        tp_replication_factor=tp_replication_factor,
    )
    transfer.set_block_count(2)
    return transfer


def test_build_cache_views_groups_opaque_layouts():
    transfer_tensors = _transfer_tensors()
    views = page_views._build_cache_views(transfer_tensors, num_blocks=2)

    assert list(views.tensors) == [
        "page.0.primary.0",
        "page.1.primary.1",
        "page.2.sidecar.0",
        "page.3.sidecar.1",
    ]
    assert views.layer_groups == ((0, 1), (2, 3))
    assert views.bytes_per_block == 2 * (4 * 32 * 2 + 4 * 16)
    assert all(tensor.dtype == torch.uint8 for tensor in views.tensors.values())
    assert tuple(views.tensors["page.0.primary.0"].shape) == (2, 4, 64)
    assert tuple(views.tensors["page.2.sidecar.0"].shape) == (2, 4, 16)
    assert views.tensors["page.0.primary.0"].data_ptr() == (
        transfer_tensors.block_tensor_views[0].data_ptr()
    )


def test_build_cache_views_publishes_float8_pages_as_raw_bytes():
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        pytest.skip("torch build has no float8_e4m3fn")

    page = torch.empty((2, 4, 32), dtype=fp8)
    page.view(torch.uint8).copy_(
        torch.arange(page.numel(), dtype=torch.uint8).reshape(page.shape)
    )
    region = KVTransferRegion(
        base_addr=page.data_ptr(),
        total_bytes=page.numel(),
        unit_bytes=page[0].numel(),
        semantic_role="latent",
    )
    transfer = KVTransferTensors(pages=[PageRegion(region, page)])
    transfer.set_block_count(2)

    views = page_views._build_cache_views(transfer, num_blocks=2)
    published = views.tensors["page.0.latent"]

    assert published.dtype == torch.uint8
    assert published.shape == page.shape
    assert published.data_ptr() == page.data_ptr()
    assert torch.equal(published, page.view(torch.uint8))


def test_build_cache_views_rejects_missing_or_bad_geometry():
    missing_view = _transfer_tensors()
    missing_view.pages[-1] = PageRegion(missing_view.pages[-1].region)
    with pytest.raises(ValueError, match="one block_tensor_view per block region"):
        page_views._build_cache_views(missing_view, num_blocks=2)

    bad_geometry = _transfer_tensors()
    bad_geometry.block_regions[0].unit_bytes += 1
    with pytest.raises(ValueError, match="byte geometry mismatch"):
        page_views._build_cache_views(bad_geometry, num_blocks=2)

    noncontiguous = _transfer_tensors()
    noncontiguous.pages[0] = replace(
        noncontiguous.pages[0], view=torch.zeros(2, 32, 4).transpose(1, 2)
    )
    assert not noncontiguous.block_tensor_views[0].is_contiguous()
    with pytest.raises(ValueError, match="non-empty and contiguous"):
        page_views._build_cache_views(noncontiguous, num_blocks=2)

    unsupported_rank = _transfer_tensors()
    unsupported_rank.pages[0] = replace(
        unsupported_rank.pages[0], view=torch.zeros(2, 4, 4, 8, dtype=torch.float16)
    )
    with pytest.raises(ValueError, match="physical_slots, opaque_width"):
        page_views._build_cache_views(unsupported_rank, num_blocks=2)


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
        page_views._build_cache_views(transfer_tensors, num_blocks=2)


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
    monkeypatch.setattr(transfer.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([None, 8])
    client = mp_lookup._MPLookupClient(
        adapter,
        config=_config(),
        timeout=10.0,
        poll_interval=0.01,
    )

    assert client.lookup(list(range(8)), "req") == 8
    client.prepare_retrieve("req", 4)
    client.complete_retrieve("req", succeeded=False)

    # LMCache owns and releases [4, 8) once retrieve is submitted, including
    # on terminal failure. The scheduler releases only the HBM-resident prefix.
    assert adapter.submissions == [("atom-offload-dp0:req", list(range(8)))]
    assert [(call["start"], call["end"]) for call in adapter.freed] == [(0, 4)]
    assert {call["request_id"] for call in adapter.freed} == {"atom-offload-dp0:req"}
    assert adapter.cleaned == ["atom-offload-dp0:req", "atom-offload-dp0:req"]
    assert client.hit_tokens("req") is None


def test_mp_lookup_releases_hit_suffix_outside_retrieve_range(monkeypatch):
    monkeypatch.setattr(transfer.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([12])
    client = mp_lookup._MPLookupClient(
        adapter,
        config=_config(),
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
    scheduler = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
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
    monkeypatch.setattr(transfer.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([8])
    client = mp_lookup._MPLookupClient(
        adapter,
        config=_config(),
        timeout=10.0,
        poll_interval=0.01,
    )

    assert client.lookup(list(range(12)), "req") == 8
    with pytest.raises(ValueError, match="retrieve end 12 exceeds lookup hit 8"):
        client.prepare_retrieve("req", 0, 12)

    assert adapter.freed == []
    assert client.hit_tokens("req") == 8


def _main_thread_clock(values):
    """`time.monotonic` stepping through `values` on this thread only.

    The patch is process-wide, and drain threads that earlier tests left
    running call the clock too; they get the real time, so they cannot use up
    the sequence this test drives.
    """
    ticks = iter(values)
    real = transfer.time.monotonic

    def clock():
        if threading.current_thread() is threading.main_thread():
            return next(ticks)
        return real()

    return clock


def test_mp_lookup_timeout_defers_cleanup_until_result(monkeypatch):
    monkeypatch.setattr(transfer.time, "monotonic", _main_thread_clock([0.0, 0.0, 2.0]))
    monkeypatch.setattr(transfer.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([None, None])
    client = mp_lookup._MPLookupClient(
        adapter,
        config=_config(),
        timeout=1.0,
        poll_interval=0.01,
    )

    assert client.lookup(list(range(8)), "req") is None
    assert adapter.cleaned == []

    adapter.results.append(8)
    client.clear_lookup_status("req")
    assert [(call["start"], call["end"]) for call in adapter.freed] == [(0, 8)]
    assert adapter.cleaned == ["atom-offload-dp0:req"]


def test_mp_lookup_pending_cleanup_drops_adapter_bookkeeping(monkeypatch):
    monkeypatch.setattr(transfer.time, "monotonic", _main_thread_clock([0.0, 0.0, 2.0]))
    monkeypatch.setattr(transfer.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([None, None])
    client = mp_lookup._MPLookupClient(
        adapter,
        config=_config(),
        timeout=1.0,
        poll_interval=0.01,
    )

    assert client.lookup(list(range(8)), "req") is None
    client.clear_lookup_status("req")

    assert adapter.cleaned == ["atom-offload-dp0:req"]
    assert client.hit_tokens("req") is None


def _full_prompt_hit_scheduler(monkeypatch, chunk_size, hit=8):
    monkeypatch.setattr(transfer.time, "sleep", lambda _seconds: None)
    adapter = _LookupAdapter([hit])
    lookup = mp_lookup._MPLookupClient(
        adapter,
        config=_config(),
        timeout=1.0,
        poll_interval=0.01,
    )
    scheduler = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
    )
    scheduler._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        scheduler,
        _config(),
        chunk_size=chunk_size,
        lookup_client=lookup,
    )
    scheduler._min_load_tokens = 0
    return scheduler, lookup


def test_full_prompt_hit_loads_to_the_chunk_boundary_below_the_last_token(monkeypatch):
    # A full-prompt hit must leave a token to compute, and the tier resolves at
    # chunk granularity, so the load floors to the chunk boundary below
    # ``num_prompt - 1``: 8 tokens hit, 7 wanted, 4 loadable.
    scheduler, lookup = _full_prompt_hit_scheduler(monkeypatch, chunk_size=4)
    seq = SimpleNamespace(
        id=7,
        num_prompt_tokens=8,
        num_cached_tokens=0,
        token_ids=list(range(8)),
        block_table=[10, 11],
    )

    assert scheduler.get_num_new_matched_tokens(seq) == (4, True)
    assert scheduler._load_specs["7"].lmcache_cached_tokens == 4
    assert scheduler._load_specs["7"].transfer_end_tokens == 8

    scheduler.update_state_after_alloc(seq)
    request = scheduler.build_connector_meta().requests[0]
    assert request.token_ids == list(range(8))
    assert request.load_spec.lmcache_cached_tokens == 4
    assert request.load_spec.transfer_end_tokens == 8
    assert seq.offload_loaded_tokens == 4

    assert scheduler.load_finished(request.load_operation) is True
    assert lookup.hit_tokens("7") is None


def test_dispatched_load_names_its_start_so_the_loaded_prefix_is_published(
    monkeypatch,
):
    # The scheduler publishes a loaded PAGE prefix into the GPU index only when
    # the load names where it starts. The start is the post-allocate HBM
    # frontier: 4 tokens resident, 12 hit, so the load covers [4, 12).
    scheduler, _ = _full_prompt_hit_scheduler(monkeypatch, chunk_size=4, hit=12)
    seq = SimpleNamespace(
        id=9,
        num_prompt_tokens=16,
        num_cached_tokens=0,
        token_ids=list(range(16)),
        block_table=[10, 11, 12, 13],
    )

    assert scheduler.get_num_new_matched_tokens(seq) == (12, True)
    seq.num_cached_tokens = 4
    scheduler.update_state_after_alloc(seq)
    request = scheduler.build_connector_meta().requests[0]

    assert request.load_spec.hbm_cached_tokens == 4
    assert seq.offload_load_start_tokens == 4
    assert seq.offload_loaded_tokens == 12


def test_full_prompt_hit_of_exactly_one_chunk_asks_for_no_load_at_all(monkeypatch):
    # Same floor, but here it lands on zero: the decremented hit walks off the
    # only chunk boundary there is. Asking for 7 of 8 tokens would name a
    # partial chunk the tier cannot serve, and the load could never finish.
    scheduler, lookup = _full_prompt_hit_scheduler(monkeypatch, chunk_size=8)
    seq = SimpleNamespace(
        id=7,
        num_prompt_tokens=8,
        num_cached_tokens=0,
        token_ids=list(range(8)),
        block_table=[10, 11],
    )

    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)
    assert "7" not in scheduler._load_specs
    assert lookup.hit_tokens("7") is None


def _hit_seq(req_id=7, num_prompt=8):
    return SimpleNamespace(
        id=req_id,
        num_prompt_tokens=num_prompt,
        num_cached_tokens=0,
        token_ids=list(range(num_prompt)),
        block_table=[10, 11],
    )


def test_mp_lookup_timeout_is_not_recorded_as_a_tier_miss(monkeypatch):
    """A deadline that expires is a non-answer, not an empty answer.

    The scheduler remembers the tier hit for as long as the request waits, so a
    timeout recorded as "this tier holds nothing" would stand for the rest of
    that wait -- including, as here, after the result the lookup was waiting
    for has arrived. A non-answer is remembered as one (`None`), and is only
    held long enough to keep a dead tier from costing a timeout every step.
    """

    monkeypatch.setattr(
        transfer.time, "monotonic", _main_thread_clock([0.0, 0.0, 2.0, 0.0])
    )
    scheduler, lookup = _full_prompt_hit_scheduler(monkeypatch, chunk_size=4)
    adapter = lookup._adapter
    adapter.results.clear()
    adapter.results.extend([None, None, 8])
    seq = _hit_seq()

    scheduler._tier_retry_steps = 0  # no backoff, so the retry is the next call

    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)
    # Remembered as "did not answer", never as a hit of 0.
    assert scheduler._tier_hit_memo["7"][1] is None

    assert scheduler.get_num_new_matched_tokens(seq) == (4, True)
    assert [request_id for request_id, _tokens in adapter.submissions] == [
        "atom-offload-dp0:7",
        "atom-offload-dp0:7",
    ]


def test_block_size_one_full_prompt_hit_stays_declined_when_asked_again(monkeypatch):
    """The subclass has the last word on every call, including replayed ones.

    One-token blocks cannot host the final LMCache chunk, so a whole-prompt hit
    is refused here and the armed load is cleared. Only the hit is remembered,
    so the refusal is re-derived on the next call rather than replayed from it
    -- which is what keeps the second call from parking the request for a
    transfer nobody dispatches.
    """

    scheduler, lookup = _full_prompt_hit_scheduler(monkeypatch, chunk_size=4)
    scheduler.block_size = 1
    scheduler.virtual_block_size = 1
    seq = _hit_seq()

    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)
    assert "7" not in scheduler._load_specs
    assert len(lookup._adapter.submissions) == 1


def test_stale_load_failure_does_not_release_current_generation_locks():
    adapter = _LookupAdapter([])
    lookup = mp_lookup._MPLookupClient(
        adapter,
        config=_config(),
        timeout=1.0,
        poll_interval=0.01,
    )
    scheduler = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
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
    lookup._lookups["7"] = mp_lookup._LookupState(
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
    # Mirrors the real `EngineGroupInfo`'s defaults so a PAGE-only
    # registration still reads back the values the server would apply.
    sw_size_tokens: int = -1
    recurrent_state: bool = False
    extra_object_group_tag: int = 0


class _WorkerFuture:
    def __init__(self, result=True) -> None:
        self.ready = False
        self.value = result
        self.query_error = None
        self.completed_ranges = []

    def query(self):
        if self.query_error is not None:
            raise self.query_error
        return self.ready

    def result(self, timeout=None):
        if not self.ready:
            raise TimeoutError("future is not ready")
        return self.value

    def take_completed_ranges(self):
        ranges = self.completed_ranges
        self.completed_ranges = []
        return ranges


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

    def submit_store_request_with_chunk_events(self, request_id, op, event):
        return self.submit_store_request(request_id, op, event)

    def shutdown(self):
        self.shutdown_called = True


def _worker(adapter: _WorkerAdapter) -> mp_worker.LMCacheMPConnector:
    worker = mp_worker.LMCacheMPConnector(_config())
    worker._adapter = adapter
    worker.chunk_size = 8
    # Both legs are submitted and polled on the connector's drain thread, off
    # the model step. These tests are about the bookkeeping that submit
    # produces, not about the hand-off, so settle each one before returning;
    # the hand-off itself is covered by
    # `test_worker_save_settles_off_the_step_thread`.
    submit_save = worker._submit_save
    submit_load = worker._submit_load

    def settling_submit_save(*args, **kwargs):
        submit_save(*args, **kwargs)
        worker._settle_transfers()

    def settling_submit_load(*args, **kwargs):
        submit_load(*args, **kwargs)
        worker._settle_transfers()

    worker._submit_save = settling_submit_save  # type: ignore[method-assign]
    worker._submit_load = settling_submit_load  # type: ignore[method-assign]
    return worker


def test_close_unregisters_the_worker_once(monkeypatch):
    """`ModelRunner.exit` closes the connector before the KV pool is freed.

    The adapter's shutdown is what unregisters the pool from the MP server;
    skipping it leaves the server holding the pool's GPU memory until its
    reaper fires, which a restarted engine on the same GPUs cannot survive.
    """
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    worker.close()
    assert adapter.shutdown_called
    assert worker._adapter is None
    adapter.shutdown_called = False
    worker.close()
    assert not adapter.shutdown_called

    class Failing(_WorkerAdapter):
        def shutdown(self):
            raise RuntimeError("server gone")

    _worker(Failing()).close()  # teardown must not raise


def test_shell_forwards_close_and_tolerates_an_unregistered_worker():
    from atom.kv_transfer.offload.mp.connector import LMCacheMPConnector

    shell = LMCacheMPConnector.__new__(LMCacheMPConnector)
    shell._impl = None
    shell.close()
    adapter = _WorkerAdapter()
    shell._impl = _worker(adapter)
    shell.close()
    assert adapter.shutdown_called


def _await_drain_stop(
    worker: mp_worker.LMCacheMPConnector, timeout: float = 5.0
) -> None:
    """Wait out the drain thread after a fail-stop.

    The transfer deadline is noticed on the save drain thread, which cannot stop
    the engine, so it records the failure and exits; the next step is what
    raises. Loads are still submitted and polled inline and have no thread.
    """
    drain = worker._drain_thread
    if drain is not None:
        drain.join(timeout=timeout)


def _finish_load(
    worker: mp_worker.LMCacheMPConnector,
    operation_id: str,
    *,
    result: bool = True,
) -> None:
    future = worker._pending_loads[operation_id].future
    future.value = result
    future.ready = True
    # The drain thread is what notices; give it a round before the caller asks.
    worker._settle_transfers()


def _finish_save(
    worker: mp_worker.LMCacheMPConnector,
    operation_id: str,
    *,
    result: bool = True,
) -> None:
    future = worker._pending_saves[operation_id].future
    future.value = result
    future.ready = True
    # The drain thread is what notices; give it a round before the caller asks.
    worker._settle_transfers()


def test_worker_save_settles_off_the_step_thread(fake_lmcache_modules):
    """Submitting and polling a save must not block the model step.

    Both ends are HIP IPC calls -- `export_event` on submit, `from_ipc_handle`
    on the poll that completes the future. Measured on an idle GPU they cost
    microseconds, but under the rank's own kernel-launch pressure they cost
    milliseconds each, and under TP rank collapse only the writer pays them, so
    the whole TP group waits at the next collective.
    """
    released = threading.Event()
    entered = threading.Event()
    adapter = _WorkerAdapter()
    real_submit = adapter.submit_store_request

    def blocking_submit(request_id, op, event):
        entered.set()
        released.wait(5.0)
        return real_submit(request_id, op, event)

    adapter.submit_store_request = blocking_submit
    # Deliberately not `_worker`: that wrapper settles each save, which is the
    # very thing under test here.
    worker = mp_worker.LMCacheMPConnector(_config())
    worker._adapter = adapter
    worker.chunk_size = 8
    operation = SaveOperationId(req_id=77, generation=1)
    worker._submit_save(
        LMCacheReqMeta(
            req_id=77,
            token_ids=list(range(8)),
            block_ids=[70, 71],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=operation,
        ),
        _step_event(),
    )

    assert entered.wait(5.0), "the drain thread never picked the save up"
    # The step returned while the submit is still in flight on the drain thread.
    assert worker._pending_saves == {}
    assert worker.get_finished().finished_saving == set()

    released.set()
    worker._settle_transfers()
    assert set(worker._pending_saves) == {"save:77:1"}
    _finish_save(worker, "save:77:1")
    assert worker.get_finished().finished_saving == {operation}


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

    worker._submit_load(request, _step_event())
    assert adapter.loads[0][0] == "atom-offload-dp0:5"
    submitted = adapter.loads[0][1]
    assert submitted.start == 0
    assert submitted.end == 8
    assert submitted.block_ids == [[10, 11]]

    _finish_load(worker, "load:5:3")
    assert worker.get_finished().finished_loading == {operation}


def test_a_load_landing_after_the_drain_round_is_reported_that_step(
    fake_lmcache_modules,
):
    """The step's drain round ran before the forward; `get_finished` looks again.

    Otherwise the request waits a whole step in WAITING_FOR_REMOTE_KVS, which
    base, polling there, did not.
    """
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    operation = LoadOperationId(req_id=7, generation=1)
    request = LMCacheReqMeta(
        req_id=7,
        token_ids=list(range(8)),
        block_ids=[30, 31],
        load_spec=LoadSpec(0, 8, can_load=True),
        load_operation=operation,
    )
    worker._submit_load(request, _step_event())
    worker._settle_transfers()
    future = worker._pending_loads["load:7:1"].future
    future.value, future.ready = True, True
    # No drain round in between.
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
    worker._submit_load(request, _step_event())
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
        _step_event(),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=11,
            token_ids=list(range(8)),
            block_ids=[60, 61],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        _step_event(),
    )

    worker._settle_transfers()
    load_future = worker._pending_loads["load:10:1"].future
    save_future = worker._pending_saves["save:11:2"].future
    load_future.query_error = RuntimeError("load query failed")
    save_future.query_error = RuntimeError("save query failed")
    # Whichever poll meets its failure first stops the engine: the load's, on
    # the step thread in `get_finished`, or the save's on the drain round that
    # call wakes. Neither releases what it holds, and nothing resumes after a
    # fail-stop, so both stay pending.
    with pytest.raises(transfer.LMCacheTransferUnprovable, match="query failed"):
        worker.get_finished()
        _await_drain_stop(worker)
        worker.get_finished()
    assert set(worker._pending_loads) == {"load:10:1"}
    assert set(worker._pending_saves) == {"save:11:2"}


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
        _step_event(),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=13,
            token_ids=list(range(8)),
            block_ids=[80, 81],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        _step_event(),
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
        _step_event(),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=15,
            token_ids=list(range(8)),
            block_ids=[100, 101],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        _step_event(),
    )

    output = worker.get_finished()

    assert output.failed_loading == {load_operation}
    assert output.finished_saving == {save_operation}
    assert worker._pending_loads == {}
    assert worker._pending_saves == {}


def _unprovable_request(kind: str) -> LMCacheReqMeta:
    if kind == "load":
        return LMCacheReqMeta(
            req_id=16,
            token_ids=list(range(16)),
            block_ids=[100, 101, 102, 103],
            load_spec=LoadSpec(0, 16, can_load=True),
            load_operation=LoadOperationId(req_id=16, generation=1),
        )
    return LMCacheReqMeta(
        req_id=16,
        token_ids=list(range(16)),
        block_ids=[100, 101, 102, 103],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(req_id=16, generation=1),
    )


@pytest.mark.parametrize("kind", ["load", "save"])
def test_worker_unprovable_submission_is_never_released_on_a_clock(
    fake_lmcache_modules,
    monkeypatch,
    kind,
):
    """A raising submit may have reached the server, which may still read the
    source or write the destination. Nothing settles, so nothing is released;
    past the transfer deadline the worker stops the engine instead."""
    adapter = _WorkerAdapter()

    def unprovable(_request_id, _op, _event):
        raise ConnectionError("server may have received request")

    for name in (
        "submit_retrieve_request",
        "submit_store_request",
        "submit_store_request_with_chunk_events",
    ):
        monkeypatch.setattr(adapter, name, unprovable)
    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])
    worker = _worker(adapter)
    request = _unprovable_request(kind)
    if kind == "load":
        worker._submit_load(request, _step_event())
        pending = worker._pending_loads
    else:
        worker._submit_save(request, _step_event())
        pending = worker._pending_saves

    now[0] += worker._transfer_deadline_s - 1
    output = worker.get_finished()
    assert not output.connector_completions
    assert not output.failed_loading and not output.finished_saving
    assert len(pending) == 1

    now[0] += 2
    _await_drain_stop(worker)
    with pytest.raises(transfer.LMCacheTransferUnprovable):
        worker.get_finished()
    assert len(pending) == 1


@pytest.mark.parametrize("kind", ["load", "save"])
def test_worker_failed_future_stops_immediately_without_releasing_lease(
    fake_lmcache_modules,
    monkeypatch,
    kind,
):
    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])
    worker = _worker(_WorkerAdapter())
    getattr(worker, f"_submit_{kind}")(_unprovable_request(kind), _step_event())
    # Submitted and polled once by the drain thread; not terminal yet.
    worker._settle_transfers()
    operations = getattr(worker, f"_pending_{kind}s")
    [pending] = operations.values()
    pending.future.query_error = ConnectionError("IPC context torn down")

    # A load is polled once more by `get_finished`; a save by the drain round
    # that call wakes, whose failure the next call raises.
    with pytest.raises(
        transfer.LMCacheTransferUnprovable, match="IPC context torn down"
    ):
        worker.get_finished()
        _await_drain_stop(worker)
        worker.get_finished()
    assert list(operations.values()) == [pending]
    assert now[0] == 1000.0


def test_worker_invalid_descriptor_fails_before_transport(
    fake_lmcache_modules,
):
    """A descriptor that cannot be built is provably unsent: it settles now."""
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    load = replace(_unprovable_request("load"), block_ids=[100])
    save = replace(_unprovable_request("save"), block_ids=[100])
    worker._submit_load(load, _step_event())
    worker._submit_save(save, _step_event())

    output = worker.get_finished()
    assert not adapter.loads and not adapter.saves
    assert output.failed_loading == {load.load_operation}
    assert output.finished_saving == {save.save_operation}
    [store] = [
        completion
        for completion in output.connector_completions
        if completion.channel == DENSE_PAGE_STORE_CHANNEL
    ]
    assert not store.succeeded


def test_transfer_deadline_defaults_to_thirty_seconds_and_can_be_overridden():
    assert transfer._transfer_deadline_s(_config()) == 30.0
    assert (
        transfer._transfer_deadline_s(
            _config(extra={"lmcache.mp.transfer_deadline_s": 180})
        )
        == 180.0
    )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_transfer_deadline_rejects_unbounded_or_invalid_values(value):
    with pytest.raises(ValueError, match="transfer_deadline_s"):
        transfer._transfer_deadline_s(
            _config(extra={"lmcache.mp.transfer_deadline_s": value})
        )


@pytest.mark.parametrize("kind", ["load", "save"])
@pytest.mark.parametrize("completes", [False, True])
def test_worker_silent_future_is_bounded_without_remote_errors(
    fake_lmcache_modules, monkeypatch, kind, completes
):
    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])
    worker = _worker(_WorkerAdapter())
    getattr(worker, f"_submit_{kind}")(_unprovable_request(kind), object())
    worker._settle_transfers()
    operations = getattr(worker, f"_pending_{kind}s")
    [pending] = operations.values()
    now[0] += 29
    output = worker.get_finished()
    assert not output.connector_completions
    assert list(operations.values()) == [pending]
    if completes:
        finish = _finish_load if kind == "load" else _finish_save
        finish(worker, f"{kind}:16:1")
        worker.get_finished()
        assert not operations
    now[0] += 1
    if completes:
        worker.get_finished()
    else:
        # A load's deadline is checked by `get_finished` itself, a save's by
        # the drain round that call wakes.
        with pytest.raises(transfer.LMCacheTransferUnprovable, match="30s"):
            worker.get_finished()
            _await_drain_stop(worker)
            worker.get_finished()
        assert list(operations.values()) == [pending]


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

    worker._submit_save(request, _step_event())
    assert adapter.saves[0][0] == "atom-offload-dp0:8"
    submitted = adapter.saves[0][1]
    assert submitted.start == 8
    assert submitted.end == 16
    assert submitted.block_ids == [[32, 33]]

    _finish_save(worker, "save:8:2")
    assert worker.get_finished().finished_saving == {operation}


def test_worker_reports_chunk_source_safe_before_store_terminal(fake_lmcache_modules):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    operation = SaveOperationId(req_id=81, generation=2)
    request = LMCacheReqMeta(
        req_id=81,
        token_ids=list(range(16)),
        block_ids=[30, 31, 32, 33],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=operation,
    )

    worker._submit_save(request, _step_event())
    future = worker._pending_saves["save:81:2"].future
    future.completed_ranges = [(8, 16)]
    worker._settle_transfers()
    output = worker.get_finished()

    assert output.finished_saving == set()
    assert output.connector_completions == {
        ConnectorCompletion(
            DENSE_PAGE_SOURCE_SAFE_CHANNEL,
            SaveSourceGroupId(operation, ((8, 16),)),
            True,
        )
    }


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

    worker._submit_save(request, _step_event())

    assert adapter.saves == []
    output = worker.get_finished()
    assert output.finished_saving == {operation}
    assert {
        completion.operation_id.ranges
        for completion in output.connector_completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((0, 8),)}
    with pytest.raises(RuntimeError, match="duplicate LMCache MP save"):
        worker._submit_save(request, _step_event())


def test_collapsed_tp_early_release_waits_only_for_the_single_writer_dma(
    fake_lmcache_modules,
):
    operation = SaveOperationId(req_id=91, generation=1)
    request = LMCacheReqMeta(
        req_id=91,
        token_ids=list(range(16)),
        block_ids=[40, 41, 42, 43],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=operation,
    )
    writer = _worker(_WorkerAdapter())
    non_writer = _worker(_WorkerAdapter())
    non_writer._is_kv_writer = False
    writer._submit_save(request, _step_event())
    non_writer._submit_save(request, _step_event())
    writer_future = writer._pending_saves["save:91:1"].future
    writer_future.completed_ranges = [(0, 8)]
    writer._settle_transfers()

    aggregator = KVOutputAggregator(world_size=2)
    first = aggregator.aggregate([writer.get_finished(), non_writer.get_finished()])
    assert {
        completion.operation_id.ranges
        for completion in first.connector_completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((0, 8),)}

    writer_future.completed_ranges = [(8, 16)]
    writer._settle_transfers()
    second = aggregator.aggregate([writer.get_finished(), non_writer.get_finished()])
    assert {
        completion.operation_id.ranges
        for completion in second.connector_completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((8, 16),)}


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
            _step_event(),
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
            _step_event(),
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
        _step_event(),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=22,
            token_ids=list(range(8)),
            block_ids=[90, 91],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        _step_event(),
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

    submit(request, _step_event())
    completed = (
        worker._completed_load_operations
        if kind == "load"
        else worker._completed_save_operations
    )
    assert operation_id not in completed
    with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
        submit(request, _step_event())

    if kind == "load":
        _finish_load(worker, operation_id)
    else:
        _finish_save(worker, operation_id)
    worker.get_finished()
    with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
        submit(request, _step_event())


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
            submit(request, _step_event())
        return original_adapter_submit(*args)

    monkeypatch.setattr(adapter, adapter_method_name, submit_with_replay)

    submit(request, _step_event())

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
    assert transfer._OPERATION_TOMBSTONE_LIMIT == 4096


@pytest.mark.parametrize("kind", ["load", "save"])
def test_worker_terminal_operation_tombstones_are_bounded(
    fake_lmcache_modules,
    monkeypatch,
    kind,
):
    monkeypatch.setattr(transfer, "_OPERATION_TOMBSTONE_LIMIT", 2)
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
        submit(request(generation), _step_event())
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
    assert transfer._OPERATION_TOMBSTONE_LIMIT == 2
    assert seen == {f"{kind}:24:2", f"{kind}:24:3"}
    assert list(order) == [f"{kind}:24:2", f"{kind}:24:3"]

    with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
        submit(request(3), _step_event())
    submit(request(1), _step_event())
    pending = worker._pending_loads if kind == "load" else worker._pending_saves
    assert set(pending) == {f"{kind}:24:1"}


def test_worker_immediate_operation_tombstones_reject_replay(
    fake_lmcache_modules,
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    load_operation = LoadOperationId(req_id=25, generation=1)
    save_operation = SaveOperationId(req_id=26, generation=1)
    # Too few blocks for the range: provably never sent, so immediately terminal.
    load_request = LMCacheReqMeta(
        req_id=25,
        token_ids=list(range(8)),
        block_ids=[120],
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

    worker._submit_load(load_request, _step_event())
    worker._submit_save(save_request, _step_event())

    assert worker._completed_load_operations == {"load:25:1"}
    assert list(worker._completed_load_operation_order) == ["load:25:1"]
    assert worker._completed_save_operations == {"save:26:1"}
    assert list(worker._completed_save_operation_order) == ["save:26:1"]
    output = worker.get_finished()
    assert output.failed_loading == {load_operation}
    assert output.finished_saving == {save_operation}

    with pytest.raises(RuntimeError, match="duplicate LMCache MP load"):
        worker._submit_load(load_request, _step_event())
    with pytest.raises(RuntimeError, match="duplicate LMCache MP save"):
        worker._submit_save(save_request, _step_event())


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
        mp_worker,
        "_make_worker_adapter",
        lambda _config, _rank: adapter,
    )
    worker = mp_worker.LMCacheMPConnector(_config(model_type="ordinary_mha"))
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
        mp_worker,
        "_make_worker_adapter",
        lambda _config, _rank: adapter,
    )
    worker = mp_worker.LMCacheMPConnector(_config(tp=8, kv_lora_rank=512))
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

    worker = mp_worker.LMCacheMPConnector(
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


def test_page_region_is_a_region_and_its_byte_view_built_together():
    """One call yields both halves, zero-copy, over exactly the PAGE bytes:
    a larger allocation (DSV4's planes also hold SLOT rows) is cut to them."""
    arena = torch.arange(3 * 8 + 4, dtype=torch.int16)  # 3 blocks + a tail
    fp16 = torch.zeros(3, 4, 2, dtype=torch.float16)
    transfer = KVTransferTensors(
        pages=[
            page_region(arena, semantic_role="plane", unit_bytes=16, total_bytes=48),
            page_region(fp16, semantic_role="rows"),
        ]
    )

    plane, rows = transfer.block_regions
    assert (plane.base_addr, plane.total_bytes, plane.unit_bytes) == (
        arena.data_ptr(),
        48,
        16,
    )
    assert (rows.total_bytes, rows.unit_bytes) == (48, 16)
    for region, view in zip(
        transfer.block_regions, transfer.block_tensor_views, strict=True
    ):
        assert view.dtype == torch.uint8
        assert tuple(view.shape) == (3, 1, 16)
        assert view.data_ptr() == region.base_addr
    transfer.block_tensor_views[1][2].fill_(1)
    assert torch.all(fp16[2].view(torch.uint8) == 1)
    transfer.set_block_count(3)
    assert page_views._build_cache_views(transfer, num_blocks=3).bytes_per_block == 32


def test_page_region_rejects_what_it_cannot_alias():
    with pytest.raises(ValueError, match="contiguous"):
        page_region(torch.zeros(4, 3).t(), semantic_role="strided")
    with pytest.raises(ValueError, match="cannot publish"):
        page_region(
            torch.zeros(8, dtype=torch.uint8),
            semantic_role="short",
            unit_bytes=4,
            total_bytes=12,
        )


def test_pages_are_read_only_through_the_derived_lists():
    """`block_regions` / `block_tensor_views` are views of `pages`, so nothing
    can append to one without the other."""
    transfer = KVTransferTensors(
        pages=[page_region(torch.zeros(2, 4, dtype=torch.uint8), semantic_role="p")]
    )
    with pytest.raises(AttributeError):
        transfer.block_regions.append(transfer.block_regions[0])
    with pytest.raises(AttributeError):
        transfer.block_tensor_views.append(transfer.block_tensor_views[0])


def test_merge_pages_appends_a_draft_and_takes_the_gcd_replication():
    target = KVTransferTensors(
        pages=[page_region(torch.zeros(2, 4, dtype=torch.uint8), semantic_role="t")],
        tp_replication_factor=8,
    )
    target.merge_pages(KVTransferTensors(tp_replication_factor=1))
    assert target.tp_replication_factor == 8  # nothing merged, nothing changes
    draft = KVTransferTensors(
        pages=[page_region(torch.zeros(2, 2, dtype=torch.uint8), semantic_role="d")],
        tp_replication_factor=1,
    )
    target.merge_pages(draft)
    assert [r.semantic_role for r in target.block_regions] == ["t", "d"]
    assert len(target.block_tensor_views) == 2
    assert target.tp_replication_factor == 1


def _checkpointed_transfer_tensors(*, num_blocks: int = 16) -> KVTransferTensors:
    """Two 16-byte PAGE planes whose units also hold a 40-byte checkpoint image
    (two units), declared the way a backend publishes native state."""
    from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec

    planes = [torch.zeros(num_blocks, 16, dtype=torch.uint8) for _ in range(2)]
    transfer = KVTransferTensors(
        pages=[
            page_region(plane, semantic_role=f"plane.{index}")
            for index, plane in enumerate(planes)
        ],
        paged_state_checkpoint_spec=PagedStateCheckpointSpec(
            page_unit_bytes=32,
            slot_bytes=64,
            layout_id="checkpointed-test",
            image_bytes=40,
        ),
        execute_paged_state_copies=lambda stores, restores, descriptor_slot=0: None,
        paged_state_region_count=2,
    )
    transfer.set_block_count(num_blocks)
    # Keeps the planes alive: the transfer container does not own them.
    transfer._test_planes = planes
    return transfer


def _register_checkpointed(monkeypatch, *, null_block_id=0):
    """A base worker registered on a layout that publishes checkpoint images."""
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
    namespaces = []

    def connect(_config, _rank, **kwargs):
        namespaces.append(kwargs)
        return adapter

    monkeypatch.setattr(mp_worker, "_make_worker_adapter", connect)
    monkeypatch.setattr(mp_worker, "require_native_state_server", lambda *_a: None)
    worker = mp_worker.LMCacheMPConnector(_config(), null_block_id=null_block_id)
    worker.register_kv_caches({}, _checkpointed_transfer_tensors(), 16)
    adapter.namespaces = namespaces
    return worker, adapter


def test_a_published_checkpoint_spec_registers_through_the_native_layout(
    fake_lmcache_modules, monkeypatch
):
    """The base worker registers the native layout, under its namespace.

    PAGE is engine group 0; image ordinal `j` is group `1 + j`, one chunk wide
    and marked recurrent -- the same registration the native-state worker
    makes, so the two cannot drift apart.
    """
    worker, adapter = _register_checkpointed(monkeypatch)
    spec = worker._native_layout.checkpoint_spec
    assert adapter.namespaces == [{"checkpoint_spec": spec, "images": True}]
    assert worker._native_layout.null_block_id == 0
    assert worker.chunk_size == 8
    assert list(adapter.registered) == [f"native_state.{i}" for i in range(5)]
    assert [g.engine_group_id for g in adapter.groups] == [0, 1, 2]
    assert [g.recurrent_state for g in adapter.groups] == [False, True, True]
    assert [g.tokens_per_block for g in adapter.groups] == [4, 8, 8]
    assert [g.sw_size_tokens for g in adapter.groups] == [-1, 8, 8]


def _settled(worker, submit, req):
    submit(req, _step_event())
    worker._settle_transfers()


def test_the_base_worker_carries_a_checkpoint_image_with_its_page_kv(
    fake_lmcache_modules, monkeypatch
):
    """Only the boundary chunk names the image's units; every earlier chunk
    names the server's null block, here 0. A load names the same lists as
    the save that wrote them."""
    worker, adapter = _register_checkpointed(monkeypatch)
    _settled(
        worker,
        worker._submit_save,
        LMCacheReqMeta(
            req_id=8,
            token_ids=list(range(24)),
            block_ids=[1, 2, 3, 4, 5, 6],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=SaveOperationId(req_id=8, generation=2),
            native_state=NativeStateTransfer((7, 9), 24, 99),
        ),
    )
    assert adapter.saves[0][1].block_ids == [[1, 2, 3, 4, 5, 6], [0, 0, 7], [0, 0, 9]]

    _settled(
        worker,
        worker._submit_load,
        LMCacheReqMeta(
            req_id=5,
            token_ids=list(range(24)),
            block_ids=[1, 2, 3, 4, 5, 6],
            load_spec=LoadSpec(
                hbm_cached_tokens=0, lmcache_cached_tokens=24, can_load=True
            ),
            load_operation=LoadOperationId(req_id=5, generation=3),
            native_state=NativeStateTransfer((7, 9), 24, 99),
        ),
    )
    assert adapter.loads[0][1].block_ids == [[1, 2, 3, 4, 5, 6], [0, 0, 7], [0, 0, 9]]


def _checkpointed_load(state):
    return LMCacheReqMeta(
        req_id=5,
        token_ids=list(range(16)),
        block_ids=[1, 2, 3, 4],
        load_spec=LoadSpec(
            hbm_cached_tokens=0, lmcache_cached_tokens=16, can_load=True
        ),
        load_operation=LoadOperationId(req_id=5, generation=3),
        native_state=state,
    )


@pytest.mark.parametrize(
    "state",
    [
        # This worker restores no SLOT: the image stays in the PAGE units.
        NativeStateTransfer((7, 9), 16, 99, destination_slot=2),
        # PAGE alone on a layout with images: someone else's state.
        None,
        # An image from another boundary would not continue this KV.
        NativeStateTransfer((7, 9), 8, 99),
        # The null block cannot hold an image.
        NativeStateTransfer((0, 9), 16, 99),
        # Image units that are also KV blocks of the same transfer.
        NativeStateTransfer((3, 9), 16, 99),
    ],
)
def test_the_base_worker_refuses_an_image_transfer_it_cannot_carry(
    fake_lmcache_modules, monkeypatch, state
):
    """Refused before anything is sent: a terminal failure, not a timeout."""
    worker, adapter = _register_checkpointed(monkeypatch)
    req = _checkpointed_load(state)
    _settled(worker, worker._submit_load, req)
    assert adapter.loads == []
    assert worker.get_finished().failed_loading == {req.load_operation}


def test_a_page_only_worker_refuses_an_image_transfer(fake_lmcache_modules):
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    req = _checkpointed_load(NativeStateTransfer((7, 9), 16, 99))
    worker._submit_load(req, _step_event())
    assert adapter.loads == []
    assert worker.get_finished().failed_loading == {req.load_operation}


def _load_step(req_id: int) -> LMCacheOffloadMetadata:
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(
        LMCacheReqMeta(
            req_id=req_id,
            token_ids=list(range(16)),
            block_ids=[100, 101, 102, 103],
            load_spec=LoadSpec(0, 16, can_load=True),
            load_operation=LoadOperationId(req_id=req_id, generation=1),
        )
    )
    return metadata


def test_a_rank_whose_saves_settle_without_the_server_records_no_event(
    fake_lmcache_modules, monkeypatch
):
    """Under TP rank collapse a non-writer's saves never reach the server.

    Recording the step event anyway, step after step, and exporting it first
    at a later load failed the server's wait with `HIP error: invalid
    argument` on every such rank.
    """
    created = []
    monkeypatch.setattr(
        mp_worker.torch.cuda,
        "Event",
        lambda **_kw: created.append(1) or SimpleNamespace(record=lambda _s: None),
    )
    monkeypatch.setattr(mp_worker.torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(mp_worker, "_stream_idle", lambda: True)
    worker = _worker(_WorkerAdapter())
    worker._is_kv_writer = False
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(
        LMCacheReqMeta(
            req_id=30,
            token_ids=list(range(16)),
            block_ids=[100, 101],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=SaveOperationId(req_id=30, generation=1),
        )
    )
    worker.start_load_kv(metadata)
    assert created == [] and len(worker._step_events) == 0
    worker.start_load_kv(_load_step(31))
    assert created == [1]


def test_an_event_recorded_but_never_exported_is_not_reused(monkeypatch):
    monkeypatch.setattr(
        mp_worker.torch.cuda,
        "Event",
        lambda **_kw: SimpleNamespace(record=lambda _s: None),
    )
    monkeypatch.setattr(mp_worker.torch.cuda, "current_stream", lambda: None)
    lock = threading.Lock()
    ring = mp_worker._StepEventRing()
    first, _ = ring.record(lock)
    second, _ = ring.record(lock)
    assert second != first
    ring.hold(second)
    ring.release(second)
    assert ring.record(lock)[0] == second


def test_a_step_event_is_reused_only_once_nothing_waits_on_it(
    fake_lmcache_modules, monkeypatch
):
    """The server waits on the step's event before it moves the bytes.

    Re-recording an event it still waits on raises `HIP error: invalid
    argument` there, so a step whose earlier event has a pending transfer
    takes another; once every transfer on it is terminal it comes back.
    """
    created = []

    class _FakeEvent:
        def __init__(self, interprocess=False):
            self.records = 0
            created.append(self)

        def record(self, stream=None):
            self.records += 1

    monkeypatch.setattr(mp_worker.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(mp_worker.torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(mp_worker, "_stream_idle", lambda: True)
    adapter = _WorkerAdapter()
    worker = _worker(adapter)

    worker.start_load_kv(_load_step(20))
    worker._settle_transfers()
    worker.start_load_kv(_load_step(21))
    assert len(created) == 2, "an event with a pending transfer is not re-recorded"

    for _, _, _, future in adapter.loads:
        future.ready = True
    worker._settle_transfers()
    worker.get_finished()
    worker.start_load_kv(_load_step(22))
    assert len(created) == 2, "a settled event comes back"
    assert [event.records for event in created] == [2, 1]


def test_preemption_fence_covers_queued_transfers_and_fails_stop(monkeypatch):
    """A queued save reaches the server after the fence; it must hold it."""
    import threading

    from atom.plugin.vllm.kv_transfer import mp_worker_adapter

    # Bounded by the worker's own transfer deadline, not a constant of its own.
    monkeypatch.setattr(mp_worker_adapter, "_FENCE_TIMEOUT_S", 3600.0)
    inner = SimpleNamespace(
        _lock=threading.Lock(),
        _pending_saves={},
        _pending_loads={},
        _submitting_owner={"save:7:1": "7"},
        _drain_wake=threading.Event(),
        _drain_thread=SimpleNamespace(is_alive=lambda: True),
        _drain_error=None,
        _transfer_deadline_s=0.01,
    )
    fence = mp_worker_adapter.MPOffloadWorkerAdapter(inner)
    fence.wait_for_requests(["8"])  # someone else's transfer: no wait
    with pytest.raises(RuntimeError, match="timed out after 0s, the transfer deadline"):
        fence.wait_for_requests(["7"])
    assert inner._drain_wake.is_set()


def test_the_preemption_fence_names_what_waiting_cannot_clear(monkeypatch):
    """A dead drain, its error, or an unprovable submit: raised at once."""
    import threading

    from atom.kv_transfer.offload.mp.transfer import _UnprovableSubmission
    from atom.plugin.vllm.kv_transfer import mp_worker_adapter

    monkeypatch.setattr(mp_worker_adapter, "_FENCE_TIMEOUT_S", 3600.0)
    inner = SimpleNamespace(
        _lock=threading.Lock(),
        _pending_saves={},
        _pending_loads={},
        _submitting_owner={"save:7:1": "7"},
        _drain_wake=threading.Event(),
        _drain_thread=None,
        _drain_error=None,
    )
    fence = mp_worker_adapter.MPOffloadWorkerAdapter(inner)
    with pytest.raises(RuntimeError, match="no drain thread"):
        fence.wait_for_requests(["7"])
    inner._drain_error = ConnectionError("server went away")
    with pytest.raises(RuntimeError, match="drain failed"):
        fence.wait_for_requests(["7"])
    inner._drain_error, inner._submitting_owner = None, {}
    inner._pending_saves = {
        "save:7:1": SimpleNamespace(completion="7", future=_UnprovableSubmission())
    }
    with pytest.raises(RuntimeError, match="submit raised"):
        fence.wait_for_requests(["7"])


def test_get_finished_wakes_the_drain_while_transfers_are_pending(
    fake_lmcache_modules,
):
    """An idle engine queues nothing, so only this call can wake the poll."""
    worker = _worker(_WorkerAdapter())
    worker._drain_wake.clear()
    worker.get_finished()
    assert not worker._drain_wake.is_set()
    worker._pending_loads["load:5:1"] = mp_worker._PendingLoad(
        completion=LoadOperationId(req_id=5, generation=1), future=_WorkerFuture()
    )
    worker.get_finished()
    assert worker._drain_wake.is_set()


def test_ipc_resources_are_destroyed_on_the_step_thread(
    fake_lmcache_modules, monkeypatch
):
    """No step event is destroyed, and no finished future on the drain thread.

    On ROCm 7.2.x destroying an interprocess event synchronises the device and
    blocks creating one on any other thread. Step events are reused instead.
    Like LMCache, this adapter lets go of a future at its next submission, on
    the drain thread, so the step thread must still be holding it then; it
    destroys futures only while its stream is idle.
    """
    import itertools
    import threading

    destroyed = {}  # name -> destroyed on the step thread
    names = itertools.count(1)

    class _Destroyed:
        def __del__(self):
            destroyed[self.name] = threading.current_thread() is threading.main_thread()

    class _Event(_Destroyed):
        def __init__(self, interprocess=False):
            self.name = f"event{next(names)}"

        def record(self, stream=None):
            pass

    class _Future(_WorkerFuture, _Destroyed):
        pass

    class _Adapter(_WorkerAdapter):
        def __init__(self):
            super().__init__()
            self.tracked = []
            self.exported = deque(maxlen=2)

        def submit_retrieve_request(self, request_id, op, event):
            self.tracked = [f for f in self.tracked if not f.query()]
            self.exported.append(event)
            if request_id.endswith(":2"):
                return None
            if request_id.endswith(":3"):
                raise ConnectionError("server went away mid-submit")
            future = _Future()
            future.name = f"future{request_id[-1]}"
            future.ready = True
            future.retained = [event]
            self.tracked.append(future)
            return future

    monkeypatch.setattr(mp_worker.torch.cuda, "Event", _Event)
    monkeypatch.setattr(mp_worker.torch.cuda, "current_stream", lambda: None)
    idle = [False]
    monkeypatch.setattr(mp_worker, "_stream_idle", lambda: idle[0])
    monkeypatch.setattr(mp_worker, "_mp_session_id", lambda _c, rid: f"s:{rid}")
    worker = mp_worker.LMCacheMPConnector(_config())
    worker._adapter = _Adapter()
    worker.chunk_size = 8

    def step(metadata):
        worker.start_load_kv(metadata)
        worker._settle_transfers()
        worker.get_finished()

    for req_id in range(1, 4):
        step(_load_step(req_id))
    assert destroyed == {}  # stream busy: everything still held

    # Idle from here: the step thread could let go first, the adapter after.
    idle[0] = True
    for req_id in range(4, 9):
        step(_load_step(req_id))
    step(_load_step(9))
    assert all(destroyed.values())
    # The adapter still holds the last future.
    assert {f"future{i}" for i in (1, 4, 5, 6, 7)} <= destroyed.keys()
    assert "future8" not in destroyed
    assert not any(name.startswith("event") for name in destroyed)
    # One more for the step whose unprovable submit still holds its event.
    assert len(worker._step_events) == 2
    worker.close()


def test_only_recent_save_leases_count_as_blocks_about_to_return():
    """A lease past the horizon is a stuck save, which must not stop preemption."""
    import time

    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase

    s = ChunkedOffloadSchedulerBase.__new__(ChunkedOffloadSchedulerBase)
    s._save_lease_blocks = {1: {10}}
    s._save_lease_at = {1: time.monotonic() - 60.0}
    assert s.has_active_save_leases()
    assert not s.has_active_save_leases(max_age_s=10.0)
    s._save_lease_blocks[2] = {11}
    s._save_lease_at[2] = time.monotonic()
    assert s.has_active_save_leases(max_age_s=10.0)


def test_a_declined_save_retires_only_where_the_decline_is_permanent():
    """NativeState declines while its pin budget is full: that clears."""
    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
    from atom.kv_transfer.offload.mp.native_state_scheduler import (
        NativeStateLMCacheMPConnectorScheduler,
    )

    assert ChunkedOffloadSchedulerBase._unbuilt_save_is_final(None)
    assert not NativeStateLMCacheMPConnectorScheduler._unbuilt_save_is_final(None)


def test_a_failed_load_reports_only_the_blocks_it_writes():
    """The HBM prefix below the load may be shared with a running request."""
    from atom.plugin.vllm.kv_transfer import mp_worker_adapter

    inner = SimpleNamespace(block_size=4, start_load_kv=lambda metadata: None)
    adapter = mp_worker_adapter.MPOffloadWorkerAdapter(inner)
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(
        LMCacheReqMeta(
            req_id=7,
            token_ids=list(range(16)),
            block_ids=[100, 101, 102, 103],
            load_spec=LoadSpec(8, 14, can_load=True),
        )
    )
    adapter.start_load_kv(metadata)
    assert adapter._load_blocks == {"7": {102, 103}}


def test_the_v41_checkpoint_switch_parses_like_other_switches(monkeypatch):
    from atom.utils import envs

    for value, on in (("TRUE", True), ("1", True), ("true", True), ("0", False)):
        monkeypatch.setenv("ATOM_V41_STATE_CHECKPOINTS", value)
        assert envs.ATOM_V41_STATE_CHECKPOINTS is on
    monkeypatch.delenv("ATOM_V41_STATE_CHECKPOINTS")
    assert envs.ATOM_V41_STATE_CHECKPOINTS is False


def test_a_busy_stream_past_the_cap_releases_a_batch_per_step():
    """Async scheduling rarely idles; the backlog is paid off, not dumped."""
    entries = deque(object() for _ in range(5))
    first = mp_worker._take_unshared(entries, limit=2)
    assert len(first) == 2 and len(entries) == 3
    assert len(mp_worker._take_unshared(entries)) == 3 and not entries


def test_mp_bounds_a_stalled_save_by_its_deadline_not_by_abandoning_it():
    """MP cannot release a save's blocks without its report, so the plugin
    keeps the quorum and drives the deadline from its own per-step hook."""
    sched = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
    )
    assert sched.abandons_saves is False
    called = []
    sched._enforce_transfer_deadlines = lambda: called.append(True)
    sched.enforce_transfer_deadlines()
    assert called == [True]


def test_the_release_cap_stops_scanning_once_it_is_met():
    """The unchecked rest stay queued in order; a held entry goes to the back."""
    held = object()
    entries = deque([held] + [object() for _ in range(4)])
    rest = [id(entry) for entry in list(entries)[3:]]
    taken = mp_worker._take_unshared(entries, limit=2)
    assert len(taken) == 2
    assert [id(entry) for entry in entries] == rest + [id(held)]


def test_the_last_drain_round_submits_what_close_found_queued():
    """A save queued before `close` is submitted and polled, not dropped."""
    import queue

    fake = SimpleNamespace(
        _drain_stop=threading.Event(),
        _drain_wake=threading.Event(),
        _transfer_queue=queue.Queue(),
        _perform_save_submit=Mock(),
        _perform_load_submit=Mock(),
        _poll_pending_loads=Mock(),
        _poll_pending_saves=Mock(),
        _drain_rounds=0,
        _drain_error=None,
    )
    fake._drain_round_cv = threading.Condition(threading.Lock())
    fake._transfer_queue.put(("save", "a", "b"))
    fake._drain_stop.set()
    mp_worker.LMCacheMPConnector._drain_loop(fake)
    fake._perform_save_submit.assert_called_once_with("a", "b")
    fake._poll_pending_saves.assert_called_once()
    assert fake._drain_rounds == 1


def test_close_leaves_a_stuck_drains_client_alone():
    """The client is not thread-safe; a drain still in a submit keeps it."""
    adapter = Mock()
    fake = SimpleNamespace(
        _drain_stop=threading.Event(),
        _drain_wake=threading.Event(),
        _drain_thread=SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True),
        _adapter=adapter,
    )
    mp_worker.LMCacheMPConnector.close(fake)
    adapter.shutdown.assert_not_called()
    assert fake._adapter is adapter
