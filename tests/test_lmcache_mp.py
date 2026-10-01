# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import itertools
import sys
import types
from collections import deque
from dataclasses import dataclass, replace
from types import SimpleNamespace

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
    SaveSpec,
)
from atom.kv_transfer.offload.mp import deployment, page_views, transfer
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp import stage_servers as mp_stage_servers
from atom.kv_transfer.offload.mp import worker as mp_worker


def _config(
    *,
    model_type: str = "test_model",
    tp: int = 2,
    pp: int = 1,
    pp_rank: int = 0,
    layers: int = 2,
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
        num_hidden_layers=layers,
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
            pipeline_parallel_rank=pp_rank,
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

    adapter = deployment._make_worker_adapter(_config(extra=extra), tp_rank=1)

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


def test_auto_rank_collapse_keeps_kimi_k3_per_rank():
    # MLA KV is replicated, but the KDA checkpoint images stored in the same
    # PAGE units hold TP-sharded heads.
    k3 = _config(model_type="kimi_k3", tp=8)
    k3.hf_config.text_config = SimpleNamespace(
        model_type="kimi_linear", kv_lora_rank=512
    )

    assert deployment._tp_replication_factor(k3) == 1


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


def test_mp_lookup_timeout_defers_cleanup_until_result(monkeypatch):
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(transfer.time, "monotonic", lambda: next(ticks))
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
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(transfer.time, "monotonic", lambda: next(ticks))
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

    ticks = iter([0.0, 0.0, 2.0, 0.0])
    monkeypatch.setattr(transfer.time, "monotonic", lambda: next(ticks))
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


def _finish_load(
    worker: mp_worker.LMCacheMPConnector,
    operation_id: str,
    *,
    result: bool = True,
) -> None:
    future = worker._pending_loads[operation_id].future
    future.value = result
    future.ready = True


def _finish_save(
    worker: mp_worker.LMCacheMPConnector,
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
    assert adapter.loads[0][0] == "atom-offload-dp0:5"
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
        worker._submit_load(request, object())
        pending = worker._pending_loads
    else:
        worker._submit_save(request, object())
        pending = worker._pending_saves

    now[0] += worker._transfer_deadline_s - 1
    output = worker.get_finished()
    assert not output.connector_completions
    assert not output.failed_loading and not output.finished_saving
    assert len(pending) == 1

    now[0] += 2
    with pytest.raises(transfer.LMCacheTransferUnprovable):
        worker.get_finished()
    assert len(pending) == 1


def test_worker_future_that_keeps_raising_is_bounded_by_the_deadline(
    fake_lmcache_modules,
    monkeypatch,
):
    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])
    worker = _worker(_WorkerAdapter())
    worker._submit_save(_unprovable_request("save"), object())
    [pending] = worker._pending_saves.values()
    pending.future.query_error = ConnectionError("IPC context torn down")

    assert not worker.get_finished().finished_saving
    now[0] += worker._transfer_deadline_s
    with pytest.raises(transfer.LMCacheTransferUnprovable):
        worker.get_finished()


def test_worker_invalid_descriptor_fails_before_transport(
    fake_lmcache_modules,
):
    """A descriptor that cannot be built is provably unsent: it settles now."""
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    load = replace(_unprovable_request("load"), block_ids=[100])
    save = replace(_unprovable_request("save"), block_ids=[100])
    worker._submit_load(load, object())
    worker._submit_save(save, object())

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


def test_transfer_deadline_defaults_to_twenty_minutes():
    assert transfer._transfer_deadline_s(_config()) == 1200.0
    assert (
        transfer._transfer_deadline_s(
            _config(extra={"lmcache.mp.transfer_deadline_s": 30})
        )
        == 30.0
    )
    with pytest.raises(ValueError, match="transfer_deadline_s"):
        transfer._transfer_deadline_s(
            _config(extra={"lmcache.mp.transfer_deadline_s": 0})
        )


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

    worker._submit_save(request, object())
    future = worker._pending_saves["save:81:2"].future
    future.completed_ranges = [(8, 16)]
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

    worker._submit_save(request, object())

    assert adapter.saves == []
    output = worker.get_finished()
    assert output.finished_saving == {operation}
    assert {
        completion.operation_id.ranges
        for completion in output.connector_completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((0, 8),)}
    with pytest.raises(RuntimeError, match="duplicate LMCache MP save"):
        worker._submit_save(request, object())


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
    writer._submit_save(request, object())
    non_writer._submit_save(request, object())
    writer_future = writer._pending_saves["save:91:1"].future
    writer_future.completed_ranges = [(0, 8)]

    aggregator = KVOutputAggregator(world_size=2)
    first = aggregator.aggregate([writer.get_finished(), non_writer.get_finished()])
    assert {
        completion.operation_id.ranges
        for completion in first.connector_completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((0, 8),)}

    writer_future.completed_ranges = [(8, 16)]
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
    assert transfer._OPERATION_TOMBSTONE_LIMIT == 2
    assert seen == {f"{kind}:24:2", f"{kind}:24:3"}
    assert list(order) == [f"{kind}:24:2", f"{kind}:24:3"]

    with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
        submit(request(3), object())
    submit(request(1), object())
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


# ---------------------------------------------------------------------------
# Pipeline parallelism: one LMCache kv rank group per PP stage
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_parallel_config(monkeypatch):
    class AtomMPParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    adapter_module = types.ModuleType("lmcache.integration.atom")
    adapter_module.AtomMPParallelConfig = AtomMPParallelConfig
    monkeypatch.setitem(sys.modules, "lmcache.integration.atom", adapter_module)
    return adapter_module


def _with_draft(config, layers: int = 1):
    config.speculative_config = SimpleNamespace(
        method="mtp",
        draft_model_hf_config=SimpleNamespace(num_nextn_predict_layers=layers),
    )
    return config


def _stage_strategies(*, pp, tp, **kwargs):
    """Every (pp_rank, tp_rank) worker strategy of one replica."""
    return {
        (pp_rank, tp_rank): deployment._parallel_strategy(
            _config(pp=pp, pp_rank=pp_rank, tp=tp, layers=8, **kwargs), tp_rank
        )
        for pp_rank in range(pp)
        for tp_rank in range(tp)
    }


def test_mp_config_accepts_pp():
    assert deployment._validate_mp_config(_config(tp=1, pp=4, layers=8)) == (1, 4)


def test_parallel_strategy_gives_each_pp4_tp1_stage_its_own_worker(
    fake_parallel_config,
):
    strategies = _stage_strategies(pp=4, tp=1, kv_lora_rank=512)

    assert {key[0]: s.worker_id for key, s in strategies.items()} == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }
    assert {s.world_size for s in strategies.values()} == {4}
    assert {s.tp_size for s in strategies.values()} == {1}


def test_parallel_strategy_collapses_mla_tp_ranks_inside_each_pp_stage(
    fake_parallel_config,
):
    strategies = _stage_strategies(pp=2, tp=4, kv_lora_rank=512)

    assert {key: s.worker_id for key, s in strategies.items()} == {
        (pp_rank, tp_rank): pp_rank for pp_rank in range(2) for tp_rank in range(4)
    }
    assert {s.world_size for s in strategies.values()} == {2}
    assert {s.tp_size for s in strategies.values()} == {4}


def test_parallel_strategy_keeps_sharded_tp_ranks_inside_each_pp_stage(
    fake_parallel_config,
):
    strategies = _stage_strategies(pp=2, tp=2)

    assert {key: s.worker_id for key, s in strategies.items()} == {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    assert {s.world_size for s in strategies.values()} == {4}


@pytest.mark.parametrize("pp_rank", [-1, 2])
def test_parallel_strategy_rejects_pp_rank_outside_the_pipeline(
    fake_parallel_config, pp_rank
):
    with pytest.raises(ValueError, match="PP rank|pipeline_parallel_rank"):
        deployment._parallel_strategy(_config(pp=2, pp_rank=pp_rank, layers=8), 0)


def test_parallel_strategy_rejects_tp_rank_outside_the_stage(fake_parallel_config):
    with pytest.raises(ValueError, match=r"TP rank 2 is outside \[0, 2\)"):
        deployment._parallel_strategy(_config(pp=2, pp_rank=1, layers=8), 2)


@pytest.mark.parametrize(
    ("pp", "tp", "kv_lora_rank"),
    [(4, 1, 512), (2, 4, 512), (2, 2, None)],
)
def test_scheduler_adapter_world_size_matches_every_worker(
    fake_parallel_config, monkeypatch, pp, tp, kv_lora_rank
):
    """The server resolves a lookup's layout by ``(model_name, world_size)``."""

    class AtomMPSchedulerAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_parallel_config.AtomMPSchedulerAdapter = AtomMPSchedulerAdapter
    monkeypatch.setattr(
        deployment, "_model_namespace", lambda _config, **_kwargs: "test"
    )
    workers = _stage_strategies(pp=pp, tp=tp, kv_lora_rank=kv_lora_rank)
    world_size = {s.world_size for s in workers.values()}
    # Every stage builds a scheduler; only the head's looks up, with no worker
    # id, so only its world size reaches the server.
    schedulers = {
        deployment._make_scheduler_adapter(
            _config(
                pp=pp,
                pp_rank=pp_rank,
                tp=tp,
                layers=8,
                kv_lora_rank=kv_lora_rank,
                extra={"lmcache.mp.l2": "none"},
            )
        )
        .kwargs["parallel_config"]
        .world_size
        for pp_rank in range(pp)
    }

    assert schedulers == world_size
    assert {s.worker_id for s in workers.values()} == set(range(world_size.pop()))


@pytest.fixture
def stage_namespace(monkeypatch):
    """Namespace of a config whose PAGE namespace sees only the PP x TP world."""
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    monkeypatch.setattr(
        deployment.offcfg, "build_lmcache_config", lambda _kvc: object()
    )
    monkeypatch.setattr(
        deployment.offcfg,
        "build_page_namespace",
        lambda _config, _lmcache_cfg, world: f"page-w{world}",
    )
    return deployment._model_namespace


def test_pp4_tp1_and_pp1_tp4_do_not_share_a_namespace(stage_namespace):
    pp4 = stage_namespace(_config(tp=1, pp=4, layers=8))
    tp4 = stage_namespace(_config(tp=4, pp=1, layers=8))

    # Both PAGE namespaces see world 4; only the stage layout tells them apart.
    assert pp4.startswith("page-w4::lmcache-mp-v3::pp-")
    assert tp4 == "page-w4::lmcache-mp-v3"
    assert pp4 != tp4


def test_layer_partition_changes_the_namespace(stage_namespace, monkeypatch):
    config = _config(tp=1, pp=4, layers=8)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,2,2,2")
    even = stage_namespace(config)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "3,2,2,1")
    skewed = stage_namespace(config)

    assert even != skewed


def test_draft_layers_on_the_last_stage_change_the_namespace(stage_namespace):
    plain = stage_namespace(_config(tp=1, pp=4, layers=8))
    with_draft = stage_namespace(_with_draft(_config(tp=1, pp=4, layers=8)))

    assert plain != with_draft


def test_every_pp_stage_shares_one_namespace(stage_namespace):
    namespaces = {
        stage_namespace(_with_draft(_config(tp=1, pp=4, pp_rank=rank, layers=8)))
        for rank in range(4)
    }

    assert len(namespaces) == 1


def test_pp_stage_layout_counts_draft_layers_on_the_last_stage(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "3,2,2,1")
    layout = deployment._pp_stage_layout(
        _with_draft(_config(tp=1, pp=4, layers=8, kv_lora_rank=512))
    )

    assert layout == {
        "pp_size": 4,
        "spans": [[0, 3], [3, 5], [5, 7], [7, 8]],
        "draft_layers": 1,
        "draft_shares_target_pool": True,
        "stage_layers": [3, 2, 2, 2],
    }


def _pp_with_draft(**extra):
    # 2/2/2/2 target layers plus one draft layer on the last stage.
    return _with_draft(_config(tp=1, pp=4, layers=8, extra=extra))


@pytest.fixture
def even_partition(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,2,2,2")


def _no_probe(_url, **_kwargs):
    raise AssertionError("must not probe the LMCache HTTP frontend")


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            lambda: _pp_with_draft(**{"lmcache.mp.l2": "present"}), id="uneven-layers"
        ),
        # Equal layer counts do not prove equal PAGE layouts: GLM-5.2's index
        # cache has rows only for non-"shared" indexer layers.
        pytest.param(
            lambda: _config(tp=1, pp=4, layers=8, extra={"lmcache.mp.l2": "present"}),
            id="even-layers",
        ),
    ],
)
def test_l2_guard_refuses_pp_with_an_l2(even_partition, monkeypatch, config):
    monkeypatch.setattr(deployment, "_fetch_l2_adapters", _no_probe)
    with pytest.raises(NotImplementedError, match="4 PP stages cannot use"):
        deployment._validate_pp_l2_layouts(config())


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            lambda: _pp_with_draft(**{"lmcache.mp.l2": " None "}), id="l1-only-server"
        ),
        pytest.param(
            lambda: _pp_with_draft(
                **{
                    "lmcache.mp.l2": "present",
                    "lmcache.mp.server_per_rank_layouts": True,
                }
            ),
            id="per-rank-layout-server",
        ),
        pytest.param(
            lambda: _with_draft(_config(extra={"lmcache.mp.l2": "present"})),
            id="no-pp",
        ),
    ],
)
def test_l2_guard_allows_layouts_the_server_cannot_confuse(
    even_partition, monkeypatch, config
):
    monkeypatch.setattr(deployment, "_fetch_l2_adapters", _no_probe)
    deployment._validate_pp_l2_layouts(config())


class _FakeOpener:
    """Stands in for ``urllib.request.build_opener`` and records its use."""

    def __init__(self, respond):
        self.respond = respond
        self.handlers = []
        self.opened = []

    def build_opener(self, *handlers):
        self.handlers.extend(handlers)
        return self

    def open(self, url, timeout):
        self.opened.append((url, timeout))
        return self.respond(url)


def test_l2_guard_auto_refuses_an_unreachable_server(even_partition, monkeypatch):
    import urllib.error
    import urllib.request

    def unreachable(_url):
        raise urllib.error.URLError("connection refused")

    opener = _FakeOpener(unreachable)
    monkeypatch.setattr(urllib.request, "build_opener", opener.build_opener)
    config = _pp_with_draft(**{"lmcache.mp.host": "tcp://cache-host"})

    with pytest.raises(NotImplementedError, match="cannot list the server's L2"):
        deployment._validate_pp_l2_layouts(config)
    assert opener.opened == [("http://cache-host:8080/config/adapters", 5.0)]


def test_l2_guard_probe_bypasses_environment_proxies(even_partition, monkeypatch):
    import io
    import json
    import urllib.request

    monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")
    opener = _FakeOpener(lambda _url: io.BytesIO(json.dumps({"adapters": []}).encode()))
    monkeypatch.setattr(urllib.request, "build_opener", opener.build_opener)

    deployment._validate_pp_l2_layouts(_pp_with_draft())

    (handler,) = opener.handlers
    assert isinstance(handler, urllib.request.ProxyHandler)
    assert handler.proxies == {}


@pytest.mark.parametrize(
    ("adapters", "refused"),
    [([], False), ([{"type_name": "mooncake_store", "primary": True}], True)],
)
def test_l2_guard_auto_reads_the_server_adapter_list(
    even_partition, monkeypatch, adapters, refused
):
    import io
    import json
    import urllib.request

    def respond(url):
        assert url == "http://cache-http:9090/config/adapters"
        return io.BytesIO(json.dumps({"adapters": adapters}).encode())

    opener = _FakeOpener(respond)
    monkeypatch.setattr(urllib.request, "build_opener", opener.build_opener)
    config = _pp_with_draft(**{"lmcache.mp.http_url": "cache-http:9090/"})

    if refused:
        with pytest.raises(NotImplementedError, match="mooncake_store"):
            deployment._validate_pp_l2_layouts(config)
    else:
        deployment._validate_pp_l2_layouts(config)


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        ({"lmcache.mp.l2": "maybe"}, ValueError),
        ({"lmcache.mp.server_per_rank_layouts": "yes"}, TypeError),
    ],
)
def test_l2_guard_rejects_bad_settings(even_partition, extra, error):
    with pytest.raises(error, match=next(iter(extra))):
        deployment._validate_pp_l2_layouts(_pp_with_draft(**extra))


def test_adapter_factories_apply_the_l2_guard_before_connecting(
    even_partition, fake_parallel_config, monkeypatch
):
    def connect(**_kwargs):
        raise AssertionError("must not connect")

    fake_parallel_config.AtomMPSchedulerAdapter = connect
    fake_parallel_config.AtomMPWorkerAdapter = connect
    monkeypatch.setattr(
        deployment, "_model_namespace", lambda _config, **_kwargs: "test"
    )
    config = _pp_with_draft(**{"lmcache.mp.l2": "present"})

    with pytest.raises(NotImplementedError, match="cannot use an LMCache L2"):
        deployment._make_scheduler_adapter(config)
    with pytest.raises(NotImplementedError, match="cannot use an LMCache L2"):
        deployment._make_worker_adapter(config, 0)


def test_worker_registers_with_its_pp_stage_worker_id(
    fake_lmcache_modules, monkeypatch
):
    class AtomMPWorkerAdapter(_WorkerAdapter):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    sys.modules["lmcache.integration.atom"].AtomMPWorkerAdapter = AtomMPWorkerAdapter
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    dist = types.ModuleType("aiter.dist")
    dist.__path__ = []
    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: SimpleNamespace(rank_in_group=0)
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.dist", dist)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    monkeypatch.setattr(
        deployment, "_model_namespace", lambda _config, **_kwargs: "test"
    )
    logged = []
    monkeypatch.setattr(
        mp_worker.logger, "info", lambda message, *args: logged.append(message % args)
    )

    worker = mp_worker.LMCacheMPConnector(
        _config(
            model_type="ordinary_mha",
            tp=1,
            pp=4,
            pp_rank=2,
            layers=8,
            extra={"lmcache.mp.l2": "none"},
        )
    )
    worker.register_kv_caches({}, transfer_tensors=_transfer_tensors(), num_blocks=2)

    parallel = worker._adapter.kwargs["parallel_config"]
    assert (parallel.worker_id, parallel.world_size) == (2, 4)
    assert worker._adapter.registered is not None
    assert "pp_rank=2 kv_worker_id=2/4" in logged[-1]


# ---------------------------------------------------------------------------
# lmcache.mp.stage_servers: one LMCache server per group of PP stages
# ---------------------------------------------------------------------------

_URL_A = "tcp://127.0.0.1:25555"
_URL_B = "tcp://127.0.0.1:25556"
_TWO_GROUPS = [
    {"url": "127.0.0.1:25555", "pp_ranks": [0, 1]},
    {"url": _URL_B, "pp_ranks": [2, 3]},
]


def _staged(*, groups=None, pp=4, pp_rank=0, tp=1, extra=None, **kwargs):
    return _config(
        pp=pp,
        pp_rank=pp_rank,
        tp=tp,
        layers=8,
        extra={
            "lmcache.mp.stage_servers": _TWO_GROUPS if groups is None else groups,
            "lmcache.mp.l2": "none",
            **(extra or {}),
        },
        **kwargs,
    )


def test_stage_servers_parse_two_numa_groups():
    servers = deployment._stage_servers(_staged())

    assert servers == (
        deployment._StageServer(_URL_A, 0, 1),
        deployment._StageServer(_URL_B, 2, 3),
    )
    assert [server.num_stages for server in servers] == [2, 2]
    assert deployment._stage_server_for(_staged(), 3) == servers[1]


def test_stage_servers_absent_keeps_the_single_server():
    config = _config(tp=1, pp=4, layers=8)

    assert deployment._stage_servers(config) is None
    assert deployment._stage_server_for(config, 2) is None
    assert deployment._server_urls(config) == ["tcp://localhost:5555"]


@pytest.mark.parametrize(
    ("groups", "extra", "kwargs", "error", "message"),
    [
        ([{"url": "a:1", "pp_ranks": [0, 1, 2, 3]}], {}, {}, ValueError, "at least 2"),
        ("a:1,b:2", {}, {}, ValueError, "at least 2"),
        (
            [{"url": "a:1", "pp_ranks": [0]}, {"url": "b:2", "pp_ranks": [2, 3]}],
            {},
            {},
            ValueError,
            "starting at 1",
        ),
        (
            [{"url": "a:1", "pp_ranks": [0, 1]}, {"url": "b:2", "pp_ranks": [1, 2, 3]}],
            {},
            {},
            ValueError,
            "starting at 2",
        ),
        (
            [{"url": "a:1", "pp_ranks": [2, 3]}, {"url": "b:2", "pp_ranks": [0, 1]}],
            {},
            {},
            ValueError,
            "starting at 0",
        ),
        (
            [{"url": "a:1", "pp_ranks": [0, 2]}, {"url": "b:2", "pp_ranks": [1, 3]}],
            {},
            {},
            ValueError,
            "consecutive",
        ),
        (
            [{"url": "a:1", "pp_ranks": [0, 1]}, {"url": "b:2", "pp_ranks": [2, 3, 4]}],
            {},
            {},
            ValueError,
            r"outside \[0, 4\)",
        ),
        (
            [{"url": "a:1", "pp_ranks": [0, 1]}, {"url": "b:2", "pp_ranks": [2]}],
            {},
            {},
            ValueError,
            r"covers PP stages \[0, 3\)",
        ),
        (
            [{"url": "a:1", "pp_ranks": []}, {"url": "b:2", "pp_ranks": [0, 1, 2, 3]}],
            {},
            {},
            ValueError,
            "non-empty list of ints",
        ),
        (
            [
                {"url": "a:1", "pp_ranks": [False]},
                {"url": "b:2", "pp_ranks": [1, 2, 3]},
            ],
            {},
            {},
            ValueError,
            "non-empty list of ints",
        ),
        (
            [
                {"url": "a:1", "pp_ranks": [0, 1]},
                {"url": "tcp://a:1", "pp_ranks": [2, 3]},
            ],
            {},
            {},
            ValueError,
            "repeats a url",
        ),
        (
            [
                {"url": "a:1", "pp_ranks": [0, 1], "numa": 0},
                {"url": "b:2", "pp_ranks": [2, 3]},
            ],
            {},
            {},
            ValueError,
            r"unknown keys \['numa'\]",
        ),
        (
            [{"url": " ", "pp_ranks": [0, 1]}, {"url": "b:2", "pp_ranks": [2, 3]}],
            {},
            {},
            ValueError,
            "url must be non-empty",
        ),
        (["a:1", "b:2"], {}, {}, TypeError, "must be an object"),
        (None, {"lmcache.mp.host": "tcp://x"}, {}, ValueError, "lmcache.mp.host"),
        (None, {"lmcache.mp.port": 1}, {}, ValueError, "lmcache.mp.port"),
        (None, {"lmcache.mp.server_urls": "x:1"}, {}, ValueError, "server_urls"),
        (None, {"lmcache.mp.http_url": "x:8080"}, {}, ValueError, "http_url"),
        (None, {}, {"dp": 2}, NotImplementedError, "does not support DP"),
    ],
)
def test_stage_servers_reject_invalid_layouts(groups, extra, kwargs, error, message):
    with pytest.raises(error, match=message):
        deployment._stage_servers(_staged(groups=groups, extra=extra, **kwargs))


def _staged_strategies(*, pp, tp, groups, **kwargs):
    return {
        (pp_rank, tp_rank): deployment._parallel_strategy(
            _staged(pp=pp, pp_rank=pp_rank, tp=tp, groups=groups, **kwargs), tp_rank
        )
        for pp_rank in range(pp)
        for tp_rank in range(tp)
    }


def test_stage_servers_give_each_group_its_own_kv_world(fake_parallel_config):
    strategies = _staged_strategies(pp=4, tp=1, groups=_TWO_GROUPS, kv_lora_rank=512)

    assert {key[0]: (s.worker_id, s.world_size) for key, s in strategies.items()} == {
        0: (0, 2),
        1: (1, 2),
        2: (0, 2),
        3: (1, 2),
    }


def test_stage_servers_collapse_mla_tp_ranks_per_one_stage_group(
    fake_parallel_config,
):
    groups = [{"url": "a:1", "pp_ranks": [0]}, {"url": "b:2", "pp_ranks": [1]}]
    strategies = _staged_strategies(pp=2, tp=8, groups=groups, kv_lora_rank=512)

    assert {(s.worker_id, s.world_size) for s in strategies.values()} == {(0, 1)}


def test_stage_servers_keep_sharded_tp_ranks_inside_each_group(
    fake_parallel_config,
):
    strategies = _staged_strategies(pp=4, tp=4, groups=_TWO_GROUPS)

    for stages in ((0, 1), (2, 3)):
        group = [s for key, s in strategies.items() if key[0] in stages]
        assert sorted(s.worker_id for s in group) == list(range(8))
        assert {s.world_size for s in group} == {8}


def test_stage_server_namespaces_split_by_group(stage_namespace):
    def namespace(pp_rank, groups=None):
        config = _with_draft(_staged(pp_rank=pp_rank, groups=groups))
        return stage_namespace(
            config,
            stage_server=deployment._stage_server_for(config, pp_rank),
        )

    group_a = {namespace(0), namespace(1)}
    group_b = {namespace(2), namespace(3)}
    single = stage_namespace(_with_draft(_config(tp=1, pp=4, layers=8)))
    other_split = namespace(
        1,
        [{"url": "a:1", "pp_ranks": [0]}, {"url": "b:2", "pp_ranks": [1, 2, 3]}],
    )

    assert len(group_a) == len(group_b) == 1
    (a,) = group_a
    (b,) = group_b
    assert a == f"{single}::stages-0-1"
    assert b == f"{single}::stages-2-3"
    assert len({a, b, single, other_split}) == 4
    assert "@" not in a + b


def test_worker_routes_to_its_stage_group_server(fake_parallel_config, monkeypatch):
    class AtomMPWorkerAdapter:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_parallel_config.AtomMPWorkerAdapter = AtomMPWorkerAdapter
    monkeypatch.setattr(
        deployment,
        "_model_namespace",
        lambda _config, *, checkpoint_spec=None, stage_server=None: (
            f"ns-{stage_server.first_pp_rank}-{stage_server.last_pp_rank}"
        ),
    )

    adapter = deployment._make_worker_adapter(_staged(pp_rank=2), 0)

    assert adapter.server_url == _URL_B
    assert adapter.model_name == "ns-2-3"
    assert (adapter.parallel_config.worker_id, adapter.parallel_config.world_size) == (
        0,
        2,
    )
    assert adapter.mq_timeout == 300.0


@dataclass(frozen=True)
class _LookupKey:
    num_kv_readers: int = 1


@pytest.fixture
def scheduler_adapter_spy(fake_parallel_config, monkeypatch):
    """``AtomMPSchedulerAdapter`` stand-in recording construction and shutdown."""

    class Spy:
        opened = []  # noqa: RUF012
        chunk_by_url = {}  # noqa: RUF012
        unreachable = set()  # noqa: RUF012

        def __init__(self, **kwargs):
            if kwargs["server_url"] in self.unreachable:
                raise ConnectionError(kwargs["server_url"])
            self.kwargs = kwargs
            self.lmcache_tokens_per_chunk = self.chunk_by_url.get(
                kwargs["server_url"], 256
            )
            self.closed = False
            self.opened.append(self)

        def _create_key(self, *_args, **_kwargs):
            return _LookupKey()

        def shutdown(self):
            self.closed = True

    fake_parallel_config.AtomMPSchedulerAdapter = Spy
    monkeypatch.setattr(
        deployment,
        "_model_namespace",
        lambda _config, *, checkpoint_spec=None, stage_server=None: (
            f"ns-{stage_server.first_pp_rank}-{stage_server.last_pp_rank}"
        ),
    )
    return Spy


def test_scheduler_opens_one_lookup_adapter_per_stage_server(scheduler_adapter_spy):
    adapter = deployment._make_scheduler_adapter(_staged(tp=4, kv_lora_rank=512))

    assert isinstance(adapter, mp_stage_servers._StageServersSchedulerAdapter)
    assert adapter.lmcache_tokens_per_chunk == 256
    assert [
        (
            spy.kwargs["server_url"],
            spy.kwargs["model_name"],
            spy.kwargs["parallel_config"].world_size,
            spy.kwargs["mq_timeout"],
        )
        for spy in scheduler_adapter_spy.opened
    ] == [(_URL_A, "ns-0-1", 2, 30.0), (_URL_B, "ns-2-3", 2, 30.0)]
    # MLA collapse: one read lock per TP consumer, on every server.
    assert {
        spy._create_key([], 0, 0, "r", None).num_kv_readers
        for spy in scheduler_adapter_spy.opened
    } == {4}


def test_scheduler_stage_adapters_share_one_chunk_size(scheduler_adapter_spy):
    scheduler_adapter_spy.chunk_by_url[_URL_B] = 512

    with pytest.raises(ValueError, match="share one chunk size"):
        deployment._make_scheduler_adapter(_staged())
    assert [spy.closed for spy in scheduler_adapter_spy.opened] == [True, True]


def test_scheduler_closes_opened_stage_adapters_when_one_fails(
    scheduler_adapter_spy,
):
    scheduler_adapter_spy.unreachable.add(_URL_B)

    with pytest.raises(ConnectionError):
        deployment._make_scheduler_adapter(_staged())
    assert [spy.closed for spy in scheduler_adapter_spy.opened] == [True]


class _StageServer:
    """One stage server's scheduler adapter."""

    def __init__(self, *results, chunk=4):
        self.lmcache_tokens_per_chunk = chunk
        self.results = deque(results)
        self.submissions = []
        self.polls = 0
        self.freed = []
        self.cleaned = []
        self.ended = []
        self.closed = False
        self.submit_error = None
        self.poll_error = None
        self.end_error = None
        self.shutdown_error = None

    def maybe_submit_lookup_request(self, request_id, token_ids):
        self.submissions.append(request_id)
        if self.submit_error is not None:
            raise self.submit_error

    def check_lookup_result(self, request_id):
        self.polls += 1
        if self.poll_error is not None:
            raise self.poll_error
        return self.results.popleft() if self.results else None

    def free_lookup_locks(self, token_ids, start, end, request_id):
        self.freed.append((start, end))

    def cleanup_lookup_result(self, request_id):
        self.cleaned.append(request_id)

    def end_session(self, request_id):
        self.ended.append(request_id)
        if self.end_error is not None:
            raise self.end_error

    def shutdown(self):
        self.closed = True
        if self.shutdown_error is not None:
            raise self.shutdown_error


@pytest.fixture
def stage_clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(mp_stage_servers.time, "monotonic", lambda: clock.now)
    return clock


@pytest.fixture
def stage_warnings(monkeypatch):
    logged = []
    monkeypatch.setattr(
        mp_stage_servers.logger,
        "warning",
        lambda message, *args, **_kwargs: logged.append(message % args),
    )
    return logged


def _fan_out(*servers):
    return mp_stage_servers._StageServersSchedulerAdapter(
        list(servers), [_URL_A, _URL_B][: len(servers)]
    )


def test_fan_out_equal_hits_free_nothing(stage_clock):
    a, b = _StageServer(8), _StageServer(8)
    adapter = _fan_out(a, b)

    adapter.maybe_submit_lookup_request("r", list(range(12)))

    assert adapter.check_lookup_result("r") == 8
    assert a.freed == b.freed == []
    assert a.submissions == b.submissions == ["r"]


def test_fan_out_answers_the_shortest_hit_and_frees_longer_tails(stage_clock):
    a, b = _StageServer(12), _StageServer(4)
    adapter = _fan_out(a, b)
    adapter.maybe_submit_lookup_request("r", list(range(16)))

    assert adapter.check_lookup_result("r") == 4
    assert a.freed == [(4, 12)]
    assert b.freed == []
    # Cached: neither server is asked again and nothing is freed twice.
    assert adapter.check_lookup_result("r") == 4
    assert (a.polls, b.polls, a.freed) == (1, 1, [(4, 12)])


def test_fan_out_waits_for_every_server(stage_clock):
    a, b = _StageServer(12), _StageServer(None, 4)
    adapter = _fan_out(a, b)
    adapter.maybe_submit_lookup_request("r", list(range(16)))

    assert adapter.check_lookup_result("r") is None
    assert a.freed == b.freed == []
    assert adapter.check_lookup_result("r") == 4
    # A answered on the first poll and is not asked again.
    assert (a.polls, b.polls) == (1, 2)
    assert a.freed == [(4, 12)]


def test_fan_out_unknown_request_is_a_miss(stage_clock):
    assert _fan_out(_StageServer(), _StageServer()).check_lookup_result("x") == 0


def test_fan_out_submit_failure_is_a_miss_until_the_retry_window_ends(
    stage_clock, stage_warnings
):
    a, b = _StageServer(12), _StageServer()
    b.submit_error = ConnectionError("B down")
    adapter = _fan_out(a, b)

    adapter.maybe_submit_lookup_request("r1", list(range(16)))
    assert adapter.check_lookup_result("r1") == 0
    assert a.freed == [(0, 12)]
    assert any(_URL_B in message for message in stage_warnings)

    # Inside the window: answered at once, no RPC to either server.
    stage_clock.now += mp_stage_servers._STAGE_SERVER_RETRY_S - 1
    adapter.maybe_submit_lookup_request("r2", list(range(16)))
    assert adapter.check_lookup_result("r2") == 0
    assert (a.submissions, b.submissions) == (["r1"], ["r1"])
    adapter.cleanup_lookup_result("r2")
    assert a.freed == [(0, 12)]

    # After it, B is contacted again.
    stage_clock.now += 2
    b.submit_error = None
    a.results.append(8)
    b.results.append(8)
    adapter.maybe_submit_lookup_request("r3", list(range(16)))
    assert adapter.check_lookup_result("r3") == 8
    assert b.submissions == ["r1", "r3"]


def test_fan_out_retry_window_doubles_while_a_server_keeps_failing(
    stage_clock, stage_warnings
):
    a, b = _StageServer(), _StageServer()
    b.submit_error = TimeoutError("B hung")
    adapter = _fan_out(a, b)
    base = mp_stage_servers._STAGE_SERVER_RETRY_S
    cap = mp_stage_servers._STAGE_SERVER_MAX_RETRY_S

    def probe(request_id):
        adapter.maybe_submit_lookup_request(request_id, list(range(16)))
        adapter.cleanup_lookup_result(request_id)
        return b.submissions.count(request_id)

    # Every probe that fails doubles the next window, up to the cap.
    window, probes = base, 0
    while True:
        probes += 1
        assert probe(f"p{probes}") == 1
        stage_clock.now += window - 1
        assert probe(f"skip{probes}") == 0
        stage_clock.now += 2
        if window == cap:
            break
        window = min(window * 2, cap)
    assert probes == 5  # 30, 60, 120, 240, 300

    # One answered lookup resets the window to the base.
    b.submit_error = None
    a.results.append(4)
    b.results.append(4)
    adapter.maybe_submit_lookup_request("ok", list(range(16)))
    assert adapter.check_lookup_result("ok") == 4
    b.submit_error = TimeoutError("B hung again")
    assert probe("again") == 1
    stage_clock.now += base + 1
    assert probe("after-base") == 1


def test_fan_out_poll_failure_is_a_miss_and_skips_the_server(
    stage_clock, stage_warnings
):
    a, b = _StageServer(12), _StageServer()
    b.poll_error = TimeoutError("B hung")
    adapter = _fan_out(a, b)
    adapter.maybe_submit_lookup_request("r", list(range(16)))

    assert adapter.check_lookup_result("r") == 0
    assert a.freed == [(0, 12)]
    assert any(_URL_B in message for message in stage_warnings)

    # The request's later lock releases reach only the live server.
    adapter.free_lookup_locks(list(range(16)), 0, 4, "r")
    assert b.freed == []
    adapter.maybe_submit_lookup_request("r2", list(range(16)))
    assert adapter.check_lookup_result("r2") == 0
    assert b.submissions == ["r"]


def test_fan_out_cleanup_of_an_unanswered_lookup_frees_the_answered_servers(
    stage_clock,
):
    a, b = _StageServer(12), _StageServer()
    adapter = _fan_out(a, b)
    adapter.maybe_submit_lookup_request("r", list(range(16)))
    assert adapter.check_lookup_result("r") is None

    adapter.cleanup_lookup_result("r")

    assert a.freed == [(0, 12)]
    assert b.freed == []
    assert a.cleaned == b.cleaned == ["r"]
    assert adapter.check_lookup_result("r") == 0


def test_fan_out_cleanup_of_a_reconciled_lookup_frees_nothing(stage_clock):
    a, b = _StageServer(12), _StageServer(8)
    adapter = _fan_out(a, b)
    adapter.maybe_submit_lookup_request("r", list(range(16)))
    assert adapter.check_lookup_result("r") == 8

    adapter.cleanup_lookup_result("r")

    assert a.freed == [(8, 12)]
    assert b.freed == []


def test_fan_out_end_session_and_shutdown_reach_every_server(
    stage_clock, stage_warnings
):
    a, b = _StageServer(), _StageServer()
    a.end_error = ConnectionError("A down")
    a.shutdown_error = ConnectionError("A down")
    adapter = _fan_out(a, b)

    adapter.end_session("r")
    adapter.shutdown()

    assert a.ended == b.ended == ["r"]
    assert a.closed and b.closed
    assert len(stage_warnings) == 2


def test_fan_out_rejects_mismatched_chunk_sizes():
    with pytest.raises(ValueError, match="share one chunk size"):
        _fan_out(_StageServer(chunk=4), _StageServer(chunk=8))


def test_stage_server_locks_through_the_scheduler_are_released_once(
    stage_clock, monkeypatch
):
    monkeypatch.setattr(transfer.time, "sleep", lambda _seconds: None)
    a, b = _StageServer(16), _StageServer(12)
    adapter = _fan_out(a, b)
    client = mp_lookup._MPLookupClient(
        adapter, config=_config(), timeout=10.0, poll_interval=0.01
    )

    assert client.lookup(list(range(16)), "req") == 12
    assert a.freed == [(12, 16)]
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(
        LMCacheReqMeta(
            req_id="req",
            token_ids=list(range(16)),
            block_ids=[1, 2],
            load_spec=LoadSpec(
                hbm_cached_tokens=4,
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
        ChunkedOffloadSchedulerBase, "build_connector_meta", lambda _self: metadata
    )

    scheduler.build_connector_meta()
    client.clear_lookup_status("req")

    assert a.freed == [(12, 16), (0, 4), (8, 12)]
    assert b.freed == [(0, 4), (8, 12)]
    # With the worker's own [4, 8), each server's hit is released exactly once.
    for server, server_hit in ((a, 16), (b, 12)):
        covered = sorted([*server.freed, (4, 8)])
        assert all(prev[1] == nxt[0] for prev, nxt in itertools.pairwise(covered))
        assert (covered[0][0], covered[-1][1]) == (0, server_hit)


def test_l2_guard_skips_single_stage_groups(monkeypatch):
    monkeypatch.setattr(deployment, "_fetch_l2_adapters", _no_probe)
    groups = [{"url": "a:1", "pp_ranks": [0]}, {"url": "b:2", "pp_ranks": [1]}]

    deployment._validate_pp_l2_layouts(
        _staged(pp=2, groups=groups, extra={"lmcache.mp.l2": "present"})
    )


def test_l2_guard_refuses_a_multi_stage_group_with_an_l2(monkeypatch):
    monkeypatch.setattr(deployment, "_fetch_l2_adapters", _no_probe)
    groups = [{"url": "a:1", "pp_ranks": [0]}, {"url": "b:2", "pp_ranks": [1, 2, 3]}]

    with pytest.raises(NotImplementedError, match="3 PP stages on tcp://b:2"):
        deployment._validate_pp_l2_layouts(
            _staged(groups=groups, extra={"lmcache.mp.l2": "present"})
        )


def test_l2_guard_auto_needs_each_multi_stage_entry_http_url(monkeypatch):
    monkeypatch.setattr(deployment, "_fetch_l2_adapters", _no_probe)

    with pytest.raises(ValueError, match="entry's http_url"):
        deployment._validate_pp_l2_layouts(_staged(extra={"lmcache.mp.l2": "auto"}))


def test_l2_guard_auto_probes_every_multi_stage_entry(monkeypatch):
    probed = []
    monkeypatch.setattr(
        deployment,
        "_fetch_l2_adapters",
        lambda url, **_kwargs: probed.append(url) or [],
    )
    groups = [
        {"url": "a:1", "pp_ranks": [0, 1], "http_url": "a:8080/"},
        {"url": "b:2", "pp_ranks": [2, 3], "http_url": "http://b:8081"},
    ]

    deployment._validate_pp_l2_layouts(
        _staged(groups=groups, extra={"lmcache.mp.l2": "auto"})
    )

    assert probed == [
        "http://a:8080/config/adapters",
        "http://b:8081/config/adapters",
    ]
