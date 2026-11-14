# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import sys
import threading
import time
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
    NativeStateTransfer,
    SaveSpec,
)
from atom.kv_transfer.offload.mp import (
    deployment,
    page_views,
    transfer,
    transfer_thread,
)
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp import worker as mp_worker
from atom.kv_transfer.offload.mp.token_shipping import token_list
from atom.model_engine.sequence import SequenceStatus
from tests.lmcache_mp_support import bare_mp_scheduler as _bare_mp_scheduler
from tests.lmcache_mp_support import close_workers, cpu_step_events  # noqa: F401
from tests.lmcache_mp_support import settle as _settle
from tests.lmcache_mp_support import shipped_tokens as _shipped_tokens
from tests.lmcache_mp_support import start_transfers as _start_transfers
from tests.lmcache_mp_support import step_event as _step_event


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


class _FakeParallelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _OlderWorkerAdapter:
    """An LMCache ATOM adapter from before the optional arguments."""

    def __init__(
        self,
        server_url,
        context,
        model_name,
        block_size,
        parallel_config,
        *,
        mq_timeout=300.0,
        heartbeat_interval=10.0,
        transfer_mode=None,
    ):
        self.kwargs = {}


class _NewerWorkerAdapter(_OlderWorkerAdapter):
    def __init__(
        self,
        *args,
        pooled_ipc_events=False,
        submit_after_producer=False,
        client_chunk_hashes=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kwargs = {
            "pooled_ipc_events": pooled_ipc_events,
            "submit_after_producer": submit_after_producer,
            "client_chunk_hashes": client_chunk_hashes,
        }


class _OlderSchedulerAdapter:
    def __init__(
        self,
        server_url,
        context,
        model_name,
        block_size,
        parallel_config,
        *,
        mq_timeout=300.0,
    ):
        self.kwargs = {}


class _NewerSchedulerAdapter(_OlderSchedulerAdapter):
    def __init__(self, *args, client_chunk_hashes=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = {"client_chunk_hashes": client_chunk_hashes}


def _fake_adapter_module(monkeypatch, *, newer: bool) -> None:
    adapter_module = types.ModuleType("lmcache.integration.atom")
    adapter_module.AtomMPParallelConfig = _FakeParallelConfig
    adapter_module.AtomMPWorkerAdapter = (
        _NewerWorkerAdapter if newer else _OlderWorkerAdapter
    )
    adapter_module.AtomMPSchedulerAdapter = (
        _NewerSchedulerAdapter if newer else _OlderSchedulerAdapter
    )
    monkeypatch.setitem(sys.modules, "lmcache.integration.atom", adapter_module)
    monkeypatch.setattr(
        deployment, "_model_namespace", lambda _config, **_kwargs: "test"
    )


def test_adapter_options_default_when_supported(monkeypatch):
    _fake_adapter_module(monkeypatch, newer=True)

    worker = deployment._make_worker_adapter(_config(), rank=0)
    scheduler = mp_scheduler._make_scheduler_adapter(_config())

    assert worker.kwargs == {
        "pooled_ipc_events": True,
        "submit_after_producer": True,
        "client_chunk_hashes": False,
    }
    assert scheduler.kwargs == {"client_chunk_hashes": False}


def test_adapter_options_follow_extra_config(monkeypatch):
    _fake_adapter_module(monkeypatch, newer=True)
    config = _config(
        extra={
            "lmcache.mp.pooled_ipc_events": False,
            "lmcache.mp.submit_after_producer": False,
            "lmcache.mp.client_chunk_hashes": True,
        }
    )

    worker = deployment._make_worker_adapter(config, rank=0)
    scheduler = mp_scheduler._make_scheduler_adapter(config)

    assert worker.kwargs == {
        "pooled_ipc_events": False,
        "submit_after_producer": False,
        "client_chunk_hashes": True,
    }
    assert scheduler.kwargs == {"client_chunk_hashes": True}


def test_adapter_options_skipped_by_older_lmcache(monkeypatch, caplog):
    _fake_adapter_module(monkeypatch, newer=False)
    config = _config(extra={"lmcache.mp.client_chunk_hashes": True})

    with caplog.at_level("WARNING", logger="atom"):
        worker = deployment._make_worker_adapter(config, rank=0)
        scheduler = mp_scheduler._make_scheduler_adapter(config)

    assert worker.kwargs == {} and scheduler.kwargs == {}
    # Only an option set to true is worth a warning, once per adapter half.
    warned = [
        r.getMessage() for r in caplog.records if "does not support" in r.getMessage()
    ]
    assert len(warned) == 2
    assert all("lmcache.mp.client_chunk_hashes" in m for m in warned)


def test_adapter_options_must_be_booleans(monkeypatch):
    _fake_adapter_module(monkeypatch, newer=True)

    with pytest.raises(TypeError, match="lmcache.mp.pooled_ipc_events"):
        deployment._make_worker_adapter(
            _config(extra={"lmcache.mp.pooled_ipc_events": "yes"}), rank=0
        )


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
    monkeypatch.setattr(
        selector,
        "attn_family",
        lambda _hf: selector.Family.MHA if owns_pool["value"] else selector.Family.MLA,
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
        request_id: str = "req"
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


@dataclass(frozen=True)
class _SaltKey:
    request_id: str
    num_kv_readers: int = 1
    cache_salt: str = ""


def _install_salt_adapters(monkeypatch, scheduler_base=None):
    """LMCache's ATOM adapters as far as keys go: one per call, unsalted.
    `scheduler_base` replaces the scheduler adapter's."""

    class AtomMPParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class AtomMPSchedulerAdapter(_LookupAdapter):
        def __init__(self, **_kwargs):
            super().__init__([8, 8])
            self.keys = []

        def _create_key(self, token_ids, start, end, request_id, worker_id):
            return _SaltKey(request_id)

        def maybe_submit_lookup_request(self, request_id, token_ids):
            super().maybe_submit_lookup_request(request_id, token_ids)
            self.keys.append(
                self._create_key(
                    token_ids,
                    start=0,
                    end=len(token_ids),
                    request_id=request_id,
                    worker_id=None,
                )
            )

    class AtomMPWorkerAdapter:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def _create_key(self, request_id, spec):
            return _SaltKey(request_id)

    adapter_module = types.ModuleType("lmcache.integration.atom")
    adapter_module.AtomMPTransferSpec = _FakeLoadStoreOp
    adapter_module.AtomMPParallelConfig = AtomMPParallelConfig
    adapter_module.AtomMPSchedulerAdapter = scheduler_base or AtomMPSchedulerAdapter
    adapter_module.AtomMPWorkerAdapter = AtomMPWorkerAdapter
    monkeypatch.setitem(sys.modules, "lmcache.integration.atom", adapter_module)
    monkeypatch.setattr(
        deployment, "_model_namespace", lambda _config, **_kwargs: "test"
    )


@pytest.mark.parametrize("cache_seed, salt", [(11, "11"), (12, "12"), (-1, "")])
def test_mp_keys_carry_the_prompts_salt(monkeypatch, cache_seed, salt):
    """Image placeholder ids are the same for every picture, so each MP key of
    a seeded prompt carries its `cache_seed`; a text prompt's carries none."""
    monkeypatch.setattr(transfer.time, "sleep", lambda _seconds: None)
    _install_salt_adapters(monkeypatch)
    adapter = mp_scheduler._make_scheduler_adapter(_config())
    client = mp_lookup._MPLookupClient(
        adapter, config=_config(), timeout=10.0, poll_interval=0.01
    )
    sched = _bare_mp_scheduler()
    ChunkedOffloadSchedulerBase.__init__(
        sched, _config(), chunk_size=8, lookup_client=client
    )
    seq = SimpleNamespace(
        id=7,
        num_prompt_tokens=16,
        num_tokens=16,
        num_cached_tokens=0,
        token_ids=list(range(16)),
        block_table=[1, 2, 3, 4],
        cache_seed=cache_seed,
        status=SequenceStatus.WAITING,
    )

    sched.get_num_new_matched_tokens(seq)
    assert [key.cache_salt for key in adapter.keys] == [salt]
    adapter.end_session(mp_lookup._mp_session_id(_config(), "7"))
    assert adapter.cache_salts == {}

    sent = []
    sched._lookup_client = SimpleNamespace(
        driving=True,
        offer=lambda _sid, _tokens, **kwargs: sent.append(kwargs),
        submit=lambda _tokens, _sid, **kwargs: sent.append(kwargs) or True,
        pump=lambda: None,
        pending_ids=set,
        supports_async=lambda: True,
        discard=lambda _sid: None,
    )
    sched.offer_lookups([seq])
    fresh = SimpleNamespace(**{**vars(seq), "id": 8})
    sched.prefetch_lookups([fresh])
    assert sched.detach_lookup(fresh, 0)
    expected = {"request_configs": {"lmcache.tag.cache_salt": salt}} if salt else {}
    assert sent == [expected] * 3

    worker_adapter = deployment._make_worker_adapter(_config(), rank=0)
    spec = deployment.transfer_spec(
        token_ids=list(range(8)), block_ids=[[1]], start=0, end=8, cache_salt=salt
    )
    assert isinstance(spec, _FakeLoadStoreOp)
    assert worker_adapter._create_key("load:7", spec).cache_salt == salt


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
    scheduler = _bare_mp_scheduler()
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
    scheduler = _bare_mp_scheduler()
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
    assert _shipped_tokens(request) == list(range(8))
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


def _hit_of_twelve(monkeypatch):
    scheduler, lookup = _full_prompt_hit_scheduler(monkeypatch, chunk_size=4, hit=12)
    seq = SimpleNamespace(
        id=9,
        num_prompt_tokens=16,
        num_cached_tokens=0,
        token_ids=list(range(16)),
        block_table=[10, 11, 12, 13],
    )
    assert scheduler.get_num_new_matched_tokens(seq) == (12, True)
    return scheduler, lookup, seq


def test_a_load_the_tier_lost_stores_the_recomputed_span_back(monkeypatch):
    """Evicted between the lookup and the dispatch: the request recomputes
    `[HBM, hit)`, the span the tier no longer holds, so it is saved again
    from the HBM frontier rather than counted as persisted up to the hit."""
    scheduler, _, seq = _hit_of_twelve(monkeypatch)
    seq.num_cached_tokens = 4
    scheduler.update_state_after_alloc(seq)
    assert scheduler._save_tracker["9"][1] == 12
    monkeypatch.setattr(scheduler, "_ensure_lookup_pin", lambda *_a: False)

    meta = scheduler.build_connector_meta()

    assert [r for r in meta.requests if r.load_spec is not None] == []
    assert scheduler._save_tracker["9"][1] == 4


def test_a_withdrawn_promise_frees_its_lookup_locks(monkeypatch):
    """Never dispatched, so no retrieve took over the hit's read locks: the
    lookup cleanup frees them. `complete_retrieve`, for a submitted retrieve,
    would forget the lookup before that cleanup ran, and leave them held."""
    scheduler, _, seq = _hit_of_twelve(monkeypatch)
    scheduler.update_state_after_alloc(seq)

    assert scheduler.load_failed(seq.id) is True

    freed = scheduler._mp_adapter.freed
    assert [(call["start"], call["end"]) for call in freed] == [(0, 12)]
    # Never dispatched: the tier still holds what the request recomputes.
    assert scheduler._save_tracker["9"][1] == 12


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
    scheduler = _bare_mp_scheduler()
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


def _worker(
    adapter: _WorkerAdapter, *, off_step: bool = True
) -> mp_worker.LMCacheMPConnector:
    """A worker with ``adapter`` registered.

    ``off_step`` is the vLLM plugin's: its transfer thread submits and polls.
    Without it, native ATOM's: transfers run on the step thread. These tests
    call `_submit_*` directly, as either thread does, and settle a round before
    reading what a polling thread polled.
    """
    worker = mp_worker.LMCacheMPConnector(_config(), transfers_off_step=off_step)
    worker._adapter = adapter
    worker.chunk_size = 8
    _start_transfers(worker)
    return worker


@pytest.fixture(params=[False, True], ids=["inline", "off_step"])
def off_step(request):
    """Run a test on native ATOM's step-thread worker and on the plugin's."""
    return request.param


def test_close_unregisters_the_worker_once(monkeypatch, off_step):
    """`ModelRunner.exit` closes the connector before the KV pool is freed.

    The adapter's shutdown is what unregisters the pool from the MP server;
    skipping it leaves the server holding the pool's GPU memory until its
    reaper fires, which a restarted engine on the same GPUs cannot survive.
    """
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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
    """Let the transfer thread poll after a fail-stop.

    It notices the transfer deadline but cannot stop the engine, so it records
    the failure for the forward thread's next call to raise.
    """
    _settle(worker)


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
    _settle(worker)


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
    _settle(worker)


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
    worker = _worker(adapter)
    operation = SaveOperationId(req_id=77, generation=1)
    # Queued as `start_load_kv` queues it.
    worker._dispatch(
        worker._submit_save,
        LMCacheReqMeta(
            req_id=77,
            token_ids=list(range(8)),
            block_ids=[70, 71],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=operation,
        ),
        _step_event(worker),
    )

    assert entered.wait(5.0), "the transfer thread never picked the save up"
    # The step returned while the submit is still in flight on that thread.
    assert worker._pending_saves == {}
    assert worker.get_finished().finished_saving == set()

    released.set()
    _settle(worker)
    assert set(worker._pending_saves) == {"save:77:1"}
    _finish_save(worker, "save:77:1")
    assert worker.get_finished().finished_saving == {operation}


@pytest.mark.parametrize("kind", ["load", "save"])
def test_inline_transfers_settle_on_the_step_thread(fake_lmcache_modules, kind):
    """Native ATOM submits before the forward and polls at the report, as it
    always has: its save admission is sized to a save reported the step it
    lands. No drain thread starts."""
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=False)
    threads = threading.active_count()
    request = _unprovable_request(kind)
    if kind == "load":
        worker._submit_load(request, _step_event(worker))
        pending, (*_, future) = worker._pending_loads, adapter.loads[0]
    else:
        worker._submit_save(request, _step_event(worker))
        pending, (*_, future) = worker._pending_saves, adapter.saves[0]
    assert len(pending) == 1, "submitted before the call returned"
    future.ready = True
    output = worker.get_finished()
    if kind == "load":
        assert output.finished_loading == {request.load_operation}
    else:
        assert output.finished_saving == {request.save_operation}
    assert not pending
    assert threading.active_count() == threads


def test_inline_transfer_deadline_raises_from_the_report(
    fake_lmcache_modules, monkeypatch
):
    """Inline, the step's own poll notices the deadline and stops the engine."""
    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])
    worker = _worker(_WorkerAdapter(), off_step=False)
    worker._submit_save(_unprovable_request("save"), _step_event(worker))
    now[0] += worker._transfer_deadline_s + 1
    with pytest.raises(transfer.LMCacheTransferUnprovable):
        worker.get_finished()
    assert len(worker._pending_saves) == 1


def test_inline_transfers_have_no_fence(fake_lmcache_modules):
    """Only the vLLM plugin hands a preempted request's blocks on before its
    transfers end, and its transfer thread polls them."""
    worker = _worker(_WorkerAdapter(), off_step=False)
    with pytest.raises(NotImplementedError, match="transfers_off_step"):
        worker.fence(["16"])


def test_worker_uses_transfer_boundary_and_exact_completion(
    fake_lmcache_modules, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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
        cache_salt="11",
    )

    worker._submit_load(request, _step_event(worker))
    assert adapter.loads[0][0] == "atom-offload-dp0:5"
    submitted = adapter.loads[0][1]
    assert submitted.cache_salt == "11"
    assert submitted.start == 0
    assert submitted.end == 8
    assert submitted.block_ids == [[10, 11]]

    _finish_load(worker, "load:5:3")
    assert worker.get_finished().finished_loading == {operation}


def test_a_load_landing_during_the_forward_is_reported_that_step(
    fake_lmcache_modules, monkeypatch
):
    """With the drain idle, `get_finished` waits for one pure poll round, so
    the request does not wait a whole step in WAITING_FOR_REMOTE_KVS."""
    monkeypatch.setattr(mp_worker, "_LAST_LOOK_WAIT_S", 5.0)
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
    worker._submit_load(request, _step_event(worker))
    future = worker._pending_loads["load:7:1"].future
    future.value, future.ready = True, True
    assert worker.get_finished().finished_loading == {operation}


def test_the_last_look_never_waits_behind_a_submit(fake_lmcache_modules, monkeypatch):
    """A round that has to submit first outlasts any wait short enough for a
    step, so `get_finished` does not wait on one."""
    monkeypatch.setattr(mp_worker, "_LAST_LOOK_WAIT_S", 5.0)
    released, entered = threading.Event(), threading.Event()
    adapter = _WorkerAdapter()
    real_submit = adapter.submit_store_request

    def blocking_submit(request_id, op, event):
        entered.set()
        released.wait(5.0)
        return real_submit(request_id, op, event)

    adapter.submit_store_request = blocking_submit
    worker = _worker(adapter)
    worker._pending_loads["load:5:1"] = mp_worker._PendingLoad(
        completion=LoadOperationId(req_id=5, generation=1), future=_WorkerFuture()
    )
    worker._dispatch(
        worker._submit_save, _unprovable_request("save"), _step_event(worker)
    )
    assert entered.wait(5.0), "the transfer thread never picked the save up"
    started = time.monotonic()
    worker.get_finished()
    waited = time.monotonic() - started
    released.set()
    assert waited < 1.0


def test_worker_reports_failed_load_from_future(fake_lmcache_modules, off_step):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
    operation = LoadOperationId(req_id=6, generation=4)
    request = LMCacheReqMeta(
        req_id=6,
        token_ids=list(range(8)),
        block_ids=[20, 21],
        load_spec=LoadSpec(0, 8, can_load=True),
        load_operation=operation,
    )
    worker._submit_load(request, _step_event(worker))
    _finish_load(worker, "load:6:4", result=False)

    output = worker.get_finished()
    assert output.finished_loading == set()
    assert output.failed_loading == {operation}


def test_worker_future_query_exception_preserves_inflight_transfers(
    fake_lmcache_modules, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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
        _step_event(worker),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=11,
            token_ids=list(range(8)),
            block_ids=[60, 61],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        _step_event(worker),
    )

    _settle(worker)
    load_future = worker._pending_loads["load:10:1"].future
    save_future = worker._pending_saves["save:11:2"].future
    load_future.query_error = RuntimeError("load query failed")
    save_future.query_error = RuntimeError("save query failed")
    # The first poll to meet its failure stops the engine, on the step thread
    # or from the transfer thread's round at the next call. Neither releases
    # what it holds.
    with pytest.raises(transfer.LMCacheTransferUnprovable, match="query failed"):
        worker.get_finished()
        _await_drain_stop(worker)
        worker.get_finished()
    assert set(worker._pending_loads) == {"load:10:1"}
    assert set(worker._pending_saves) == {"save:11:2"}
    if off_step:
        # A polling thread stops at its first failed round; the engine stops.
        assert worker._transfer._poll_failed
        return

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
    fake_lmcache_modules, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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
        _step_event(worker),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=13,
            token_ids=list(range(8)),
            block_ids=[80, 81],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        _step_event(worker),
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
    fake_lmcache_modules, monkeypatch, off_step
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
    worker = _worker(adapter, off_step=off_step)
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
        _step_event(worker),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=15,
            token_ids=list(range(8)),
            block_ids=[100, 101],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        _step_event(worker),
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
    fake_lmcache_modules, monkeypatch, kind, off_step
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
    worker = _worker(adapter, off_step=off_step)
    request = _unprovable_request(kind)
    if kind == "load":
        worker._submit_load(request, _step_event(worker))
        pending = worker._pending_loads
    else:
        worker._submit_save(request, _step_event(worker))
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
    fake_lmcache_modules, monkeypatch, kind, off_step
):
    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])
    worker = _worker(_WorkerAdapter(), off_step=off_step)
    getattr(worker, f"_submit_{kind}")(_unprovable_request(kind), _step_event(worker))
    # Submitted and polled once by the drain thread; not terminal yet.
    _settle(worker)
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
    fake_lmcache_modules, off_step
):
    """A descriptor that cannot be built is provably unsent: it settles now."""
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
    load = replace(_unprovable_request("load"), block_ids=[100])
    save = replace(_unprovable_request("save"), block_ids=[100])
    worker._submit_load(load, _step_event(worker))
    worker._submit_save(save, _step_event(worker))
    _settle(worker)  # the transfer thread's next round reports them

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
    fake_lmcache_modules, monkeypatch, kind, completes, off_step
):
    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])
    worker = _worker(_WorkerAdapter(), off_step=off_step)
    getattr(worker, f"_submit_{kind}")(_unprovable_request(kind), _step_event(worker))
    _settle(worker)
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
        # Both legs' deadlines are checked by the drain round that call wakes;
        # the step raises what the drain recorded.
        with pytest.raises(transfer.LMCacheTransferUnprovable, match="30s"):
            worker.get_finished()
            _await_drain_stop(worker)
            worker.get_finished()
        assert list(operations.values()) == [pending]


def test_worker_save_slices_chunk_blocks_and_preserves_operation(
    fake_lmcache_modules, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
    operation = SaveOperationId(req_id=8, generation=2)
    request = LMCacheReqMeta(
        req_id=8,
        token_ids=list(range(16)),
        block_ids=[30, 31, 32, 33],
        save_spec=SaveSpec(skip_leading_tokens=8),
        save_operation=operation,
        cache_salt="11",
    )

    worker._submit_save(request, _step_event(worker))
    assert adapter.saves[0][0] == "atom-offload-dp0:8"
    submitted = adapter.saves[0][1]
    assert submitted.cache_salt == "11"
    assert submitted.start == 8
    assert submitted.end == 16
    assert submitted.block_ids == [[32, 33]]

    _finish_save(worker, "save:8:2")
    assert worker.get_finished().finished_saving == {operation}


def test_worker_reports_chunk_source_safe_before_store_terminal(
    fake_lmcache_modules, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
    operation = SaveOperationId(req_id=81, generation=2)
    request = LMCacheReqMeta(
        req_id=81,
        token_ids=list(range(16)),
        block_ids=[30, 31, 32, 33],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=operation,
    )

    worker._submit_save(request, _step_event(worker))
    future = worker._pending_saves["save:81:2"].future
    future.completed_ranges = [(8, 16)]
    _settle(worker)
    output = worker.get_finished()

    assert output.finished_saving == set()
    assert output.connector_completions == {
        ConnectorCompletion(
            DENSE_PAGE_SOURCE_SAFE_CHANNEL,
            SaveSourceGroupId(operation, ((8, 16),)),
            True,
        )
    }


def test_non_writer_completes_save_without_submitting(fake_lmcache_modules, off_step):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
    worker._is_kv_writer = False
    operation = SaveOperationId(req_id=9, generation=2)
    request = LMCacheReqMeta(
        req_id=9,
        token_ids=list(range(8)),
        block_ids=[40, 41],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=operation,
    )

    worker._submit_save(request, _step_event(worker))

    assert adapter.saves == []
    output = worker.get_finished()
    assert output.finished_saving == {operation}
    assert {
        completion.operation_id.ranges
        for completion in output.connector_completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((0, 8),)}
    with pytest.raises(RuntimeError, match="duplicate LMCache MP save"):
        worker._submit_save(request, _step_event(worker))


def test_collapsed_tp_early_release_waits_only_for_the_single_writer_dma(
    fake_lmcache_modules, off_step
):
    operation = SaveOperationId(req_id=91, generation=1)
    request = LMCacheReqMeta(
        req_id=91,
        token_ids=list(range(16)),
        block_ids=[40, 41, 42, 43],
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=operation,
    )
    writer = _worker(_WorkerAdapter(), off_step=off_step)
    non_writer = _worker(_WorkerAdapter(), off_step=off_step)
    non_writer._is_kv_writer = False
    writer._submit_save(request, _step_event(writer))
    non_writer._submit_save(request, _step_event(non_writer))
    writer_future = writer._pending_saves["save:91:1"].future
    writer_future.completed_ranges = [(0, 8)]
    _settle(writer)

    aggregator = KVOutputAggregator(world_size=2)
    first = aggregator.aggregate([writer.get_finished(), non_writer.get_finished()])
    assert {
        completion.operation_id.ranges
        for completion in first.connector_completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((0, 8),)}

    writer_future.completed_ranges = [(8, 16)]
    _settle(writer)
    second = aggregator.aggregate([writer.get_finished(), non_writer.get_finished()])
    assert {
        completion.operation_id.ranges
        for completion in second.connector_completions
        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL
    } == {((8, 16),)}


def test_worker_tracks_two_load_generations_for_one_raw_request(
    fake_lmcache_modules, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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
            _step_event(worker),
        )

    assert set(worker._pending_loads) == {"load:20:1", "load:20:2"}

    _finish_load(worker, "load:20:2")
    assert worker.get_finished().finished_loading == {operations[1]}
    assert set(worker._pending_loads) == {"load:20:1"}
    _finish_load(worker, "load:20:1")
    assert worker.get_finished().finished_loading == {operations[0]}


def test_worker_tracks_two_save_generations_for_one_raw_request(
    fake_lmcache_modules, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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
            _step_event(worker),
        )

    assert set(worker._pending_saves) == {"save:21:1", "save:21:2"}

    _finish_save(worker, "save:21:1")
    assert worker.get_finished().finished_saving == {operations[0]}
    assert set(worker._pending_saves) == {"save:21:2"}
    _finish_save(worker, "save:21:2")
    assert worker.get_finished().finished_saving == {operations[1]}


def test_worker_load_and_save_coexist_for_same_raw_request(
    fake_lmcache_modules, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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
        _step_event(worker),
    )
    worker._submit_save(
        LMCacheReqMeta(
            req_id=22,
            token_ids=list(range(8)),
            block_ids=[90, 91],
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=save_operation,
        ),
        _step_event(worker),
    )

    assert set(worker._pending_loads) == {"load:22:4"}
    assert set(worker._pending_saves) == {"save:22:5"}
    _finish_load(worker, "load:22:4")
    _finish_save(worker, "save:22:5")
    output = worker.get_finished()
    assert output.finished_loading == {load_operation}
    assert output.finished_saving == {save_operation}


@pytest.mark.parametrize("kind", ["load", "save"])
def test_worker_rejects_exact_operation_replay(fake_lmcache_modules, kind, off_step):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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

    submit(request, _step_event(worker))
    completed = (
        worker._completed_load_operations
        if kind == "load"
        else worker._completed_save_operations
    )
    assert operation_id not in completed
    with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
        submit(request, _step_event(worker))

    if kind == "load":
        _finish_load(worker, operation_id)
    else:
        _finish_save(worker, operation_id)
    worker.get_finished()
    with pytest.raises(RuntimeError, match="duplicate LMCache MP"):
        submit(request, _step_event(worker))


@pytest.mark.parametrize("kind", ["load", "save"])
def test_worker_rejects_duplicate_while_operation_is_submitting(
    fake_lmcache_modules, monkeypatch, kind, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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
            submit(request, _step_event(worker))
        return original_adapter_submit(*args)

    monkeypatch.setattr(adapter, adapter_method_name, submit_with_replay)

    submit(request, _step_event(worker))

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
    fake_lmcache_modules, monkeypatch, kind, off_step
):
    monkeypatch.setattr(transfer, "_OPERATION_TOMBSTONE_LIMIT", 2)
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)

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
        submit(request(generation), _step_event(worker))
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
        submit(request(3), _step_event(worker))
    submit(request(1), _step_event(worker))
    pending = worker._pending_loads if kind == "load" else worker._pending_saves
    assert set(pending) == {f"{kind}:24:1"}


def test_worker_immediate_operation_tombstones_reject_replay(
    fake_lmcache_modules, off_step
):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
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

    worker._submit_load(load_request, _step_event(worker))
    worker._submit_save(save_request, _step_event(worker))

    assert worker._completed_load_operations == {"load:25:1"}
    assert list(worker._completed_load_operation_order) == ["load:25:1"]
    assert worker._completed_save_operations == {"save:26:1"}
    assert list(worker._completed_save_operation_order) == ["save:26:1"]
    output = worker.get_finished()
    assert output.failed_loading == {load_operation}
    assert output.finished_saving == {save_operation}

    with pytest.raises(RuntimeError, match="duplicate LMCache MP load"):
        worker._submit_load(load_request, _step_event(worker))
    with pytest.raises(RuntimeError, match="duplicate LMCache MP save"):
        worker._submit_save(save_request, _step_event(worker))


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


# -- incremental token shipping (`token_shipping.py`) ------------------------


def _shipping_pair():
    """A scheduler, and the concrete MP worker every frontend reaches (the
    shell, the native-state worker and the vLLM plugin's adapter all call its
    `start_load_kv`). The worker has no adapter, so `start_load_kv` resolves the
    shipped tokens and then stops at "not registered"; `_send` collects what it
    resolved."""
    from atom.kv_transfer.offload.mp.token_shipping import WorkerTokens

    sched = _bare_mp_scheduler()
    worker = mp_worker.LMCacheMPConnector.__new__(mp_worker.LMCacheMPConnector)
    worker._tokens = WorkerTokens()
    worker._adapter = None
    worker.chunk_size = None
    received = []
    worker._received = received
    return sched, worker, received


def _start(worker, meta):
    try:
        worker.start_load_kv(meta)
    except RuntimeError as exc:
        if "not registered" not in str(exc):
            raise


def _send(worker, *reqs):
    import pickle

    from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata

    meta = LMCacheOffloadMetadata()
    meta.requests = list(reqs)
    sent = pickle.loads(pickle.dumps(meta))
    _start(worker, sent)
    worker._received.extend(token_list(r.token_ids) for r in sent.requests)


def test_incremental_tokens_round_trip_through_the_worker_shell():
    """A load, growing saves, a shorter transfer after a longer one, and an id
    reused after its drop: each transfer hands the worker exactly its prefix,
    while only tokens past what the workers hold travel, as raw bytes."""
    import array
    import pickle

    from atom.kv_transfer.offload.mp.token_shipping import ShippedTokens

    sched, shell, received = _shipping_pair()
    seq = SimpleNamespace(
        id=7, token_ids=array.array("i", range(1000)), block_table=[0]
    )
    load = sched._build_load_request(seq, LoadSpec(0, 256, can_load=True), None, 256)
    assert isinstance(load.token_ids, ShippedTokens)
    assert isinstance(load.token_ids.tokens, array.array)
    _send(shell, load)
    sizes = []
    for end in (512, 768, 512, 1000):
        save = sched._build_save_request(seq, 0, end, None, [0], False)
        sizes.append(len(save.token_ids.tokens))
        _send(shell, save)
    assert sizes == [256, 256, 0, 232]
    assert [len(t) for t in received] == [256, 512, 768, 512, 1000]
    assert all(t == list(range(len(t))) for t in received)
    assert all(type(t) is list for t in received)
    # Raw bytes on the wire, not a pickled list of ints.
    assert len(pickle.dumps(save.token_ids)) < 232 * 4 + 200

    # The session ends; a reused id starts its copy over at base 0.
    sched._token_shipper.drop(7, seq)
    other = SimpleNamespace(
        id=7, token_ids=array.array("i", range(5000, 5600)), block_table=[0]
    )
    save = sched._build_save_request(other, 0, 600, None, [0], False)
    assert save.token_ids.base == 0
    _send(shell, save)
    assert list(received[-1]) == list(range(5000, 5600))


def test_a_reused_id_restarts_its_copy_and_a_late_drop_spares_it():
    import array

    sched, _shell, _ = _shipping_pair()
    old = SimpleNamespace(id=7, token_ids=array.array("i", range(64)))
    new = SimpleNamespace(id=7, token_ids=array.array("i", range(100, 164)))
    shipper = sched._token_shipper
    assert shipper.ship(old, 32).base == 0
    shipped = shipper.ship(new, 32)  # reused id: old's copy goes first
    assert shipped.base == 0 and shipper.take_drops() == (7,)
    shipper.drop(7, old)  # old's session ends late: new's copy stays
    assert shipper.ship(new, 64).base == 32


def test_incremental_tokens_desync_raises():
    from atom.kv_transfer.offload.mp.token_shipping import ShippedTokens

    _, shell, _ = _shipping_pair()
    req = SimpleNamespace(req_id=7, token_ids=ShippedTokens(50, 52, [1, 2]))
    from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata

    meta = LMCacheOffloadMetadata()
    meta.requests = [req]
    with pytest.raises(RuntimeError, match="shipped from 50"):
        shell.start_load_kv(meta)


def test_the_session_end_drops_the_workers_copy(monkeypatch):
    """`_end_finished_sessions` is where no save can follow: it drops the
    request's shipped prefix, and only then."""
    import array

    sched = _bare_mp_scheduler()
    seq = SimpleNamespace(id=7, token_ids=array.array("i", range(64)))
    sched._token_shipper.ship(seq, 32)
    sched._lookup_client = SimpleNamespace()
    sched._save_tracker, sched._save_inflight = {}, {}
    sched._mp_adapter = SimpleNamespace(end_session=lambda rid: None)
    sched._config = SimpleNamespace()
    monkeypatch.setattr(sched, "reconcile_detached_lookups", lambda: None)
    monkeypatch.setattr(sched, "_pending_session_ends", lambda: {"7": seq})
    monkeypatch.setattr(
        "atom.kv_transfer.offload.mp.scheduler._mp_session_id", lambda c, i: str(i)
    )
    sched._save_inflight["7"] = object()  # a save is still in flight
    sched._end_finished_sessions()
    assert 7 in sched._token_shipper._shipped
    del sched._save_inflight["7"]
    sched._end_finished_sessions()
    assert 7 not in sched._token_shipper._shipped
    assert sched._token_shipper._drops == [7]


def test_drops_reach_the_workers_without_a_transfer():
    """A finished request's copy goes with the next metadata even when no
    transfer is left to carry it."""
    import array

    sched, shell, _ = _shipping_pair()
    seq = SimpleNamespace(id=7, token_ids=array.array("i", range(64)))
    _send(
        shell, SimpleNamespace(req_id=7, token_ids=sched._token_shipper.ship(seq, 32))
    )
    assert 7 in shell._tokens._held
    sched._token_shipper.drop(7, seq)
    from atom.kv_transfer.offload.mp.metadata import MPOffloadMetadata

    meta = MPOffloadMetadata()
    meta.token_drops = sched._token_shipper.take_drops()
    assert meta.has_work()
    _start(shell, meta)
    assert 7 not in shell._tokens._held


@pytest.mark.parametrize(
    "dp,tp,enable_dp_attention,requires_match",
    [
        (1, 1, False, False),
        (1, 2, False, False),
        (2, 1, False, True),
        (1, 2, True, True),
        (2, 1, True, True),
    ],
)
def test_a_chunk_configured_differently_from_the_mp_server_stops_startup(
    monkeypatch, dp, tp, enable_dp_attention, requires_match
):
    """Only `prefix_spread` reads the configured chunk (to predict loads), so
    only an active prefix router is refused when the server runs on another
    one. Ordinary DP=1 and other strategies use the server's chunk as before;
    DP-attention's TP ranks are routable engines too."""
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "256")
    server_512 = SimpleNamespace(lmcache_tokens_per_chunk=512)
    config = _config(dp=dp, tp=tp, enable_dp_attention=enable_dp_attention)
    config.dp_load_balance = "least_requests"
    deployment.require_chunk_matches_server(server_512, config)
    config.dp_load_balance = "prefix_spread"
    if requires_match:
        with pytest.raises(ValueError, match="configured chunk to match the server"):
            deployment.require_chunk_matches_server(server_512, config)
    else:
        deployment.require_chunk_matches_server(server_512, config)
    deployment.require_chunk_matches_server(
        SimpleNamespace(lmcache_tokens_per_chunk=256), config
    )
    monkeypatch.delenv("LMCACHE_CHUNK_SIZE")
    config.kv_transfer_config = {
        **config.kv_transfer_config,
        "kv_connector_extra_config": {
            **config.kv_transfer_config.get("kv_connector_extra_config", {}),
            "lmcache.chunk_size": 512,
        },
    }
    deployment.require_chunk_matches_server(server_512, config)


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


def _register_checkpointed(monkeypatch, adapter=None, *, off_step=False):
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

    adapter = _WorkerAdapter() if adapter is None else adapter
    namespaces = []

    def connect(_config, _rank, **kwargs):
        namespaces.append(kwargs)
        return adapter

    monkeypatch.setattr(mp_worker, "_make_worker_adapter", connect)
    monkeypatch.setattr(mp_worker, "require_native_state_server", lambda *_a: None)
    worker = mp_worker.LMCacheMPConnector(_config(), transfers_off_step=off_step)
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
    assert worker.chunk_size == 8
    assert list(adapter.registered) == [f"native_state.{i}" for i in range(5)]
    assert [g.engine_group_id for g in adapter.groups] == [0, 1, 2]
    assert [g.recurrent_state for g in adapter.groups] == [False, True, True]
    assert [g.tokens_per_block for g in adapter.groups] == [4, 8, 8]
    assert [g.sw_size_tokens for g in adapter.groups] == [-1, 8, 8]


def _settled(worker, submit, req):
    submit(req, _step_event(worker))
    _settle(worker)


def test_the_base_worker_carries_a_checkpoint_image_with_its_page_kv(
    fake_lmcache_modules, monkeypatch
):
    """Only the boundary chunk names the image's units; every earlier chunk
    names the server's null block, -1. A load names the same lists as the save
    that wrote them."""
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
    assert adapter.saves[0][1].block_ids == [
        [1, 2, 3, 4, 5, 6],
        [-1, -1, 7],
        [-1, -1, 9],
    ]

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
    assert adapter.loads[0][1].block_ids == [
        [1, 2, 3, 4, 5, 6],
        [-1, -1, 7],
        [-1, -1, 9],
    ]


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
        NativeStateTransfer((-1, 9), 16, 99),
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


def test_a_page_only_worker_refuses_an_image_transfer(fake_lmcache_modules, off_step):
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=off_step)
    req = _checkpointed_load(NativeStateTransfer((7, 9), 16, 99))
    worker._submit_load(req, _step_event(worker))
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


def test_off_step_fences_come_from_the_bounded_event_pool(fake_lmcache_modules):
    """The transfer thread's fences, as under `OFFLOAD_MP_TRANSFER_THREAD`: no
    event is recorded past the limit a ROCm server stream can wait on."""
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    assert isinstance(worker._fences, transfer_thread._ProducerFences)
    worker._fences = transfer_thread._ProducerFences(slots=2)
    steps = 3 * transfer_thread._FENCE_MAX_RECORDS
    for req_id in range(steps):
        worker.start_load_kv(_load_step(req_id))
    assert worker._transfer.drain(5.0)

    events = {id(event): event for _, _, event, _ in adapter.loads}
    assert len(adapter.loads) == steps
    assert len(events) > 2
    assert all(e.records <= transfer_thread._FENCE_MAX_RECORDS for e in events.values())


def test_off_step_transfers_refuse_an_adapter_that_records_its_own_event(
    fake_lmcache_modules, monkeypatch
):
    """Its pooled event would be recorded on the transfer thread's stream, not
    behind the forward; inline, nothing could fence a preempted request. Refused
    before the server registration, which is shut down again."""
    adapter = _WorkerAdapter()
    adapter._producer_events = object()
    with pytest.raises(RuntimeError, match="records producer events at submission"):
        _register_checkpointed(monkeypatch, adapter, off_step=True)
    assert adapter.registered is None
    assert adapter.shutdown_called


def test_off_step_transfers_poll_whatever_the_transfer_thread_flag_says(
    fake_lmcache_modules, monkeypatch
):
    """`OFFLOAD_MP_TRANSFER_THREAD` starts native ATOM's submit-only thread;
    a worker built to run off the step polls on its own thread either way."""
    monkeypatch.setattr(mp_worker.envs, "OFFLOAD_MP_TRANSFER_THREAD", True)
    assert _worker(_WorkerAdapter())._transfer.polls
    assert not _worker(_WorkerAdapter(), off_step=False)._transfer.polls


def test_off_step_transfers_tell_a_capable_adapter_its_fence_is_recorded(
    fake_lmcache_modules,
):
    class _Capable(_WorkerAdapter):
        ACCEPTS_RECORDED_PRODUCER_EVENTS = True

        def __init__(self):
            super().__init__()
            self._producer_events = object()
            self.kwargs = []

        def submit_retrieve_request(self, request_id, op, event, **kwargs):
            self.kwargs.append(kwargs)
            return super().submit_retrieve_request(request_id, op, event)

    adapter = _Capable()
    worker = _worker(adapter)
    worker.start_load_kv(_load_step(7))
    assert worker._transfer.drain(5.0)
    assert adapter.kwargs == [{"producer_event_recorded": True}]


def test_handed_back_futures_are_dropped_while_the_stream_is_idle(
    fake_lmcache_modules, monkeypatch
):
    """Each holds the server's imported IPC event, whose destruction
    synchronises the device: never mid-forward, unless too many are held."""
    monkeypatch.setattr(mp_worker, "_RETURNED_FUTURES_CAP", 2)
    monkeypatch.setattr(mp_worker, "_RETURNED_FUTURES_BATCH", 1)
    worker = _worker(_WorkerAdapter())
    stream = mp_worker.torch.cuda.stream
    stream.idle = False
    for _ in range(2):
        worker._transfer.hand_back(object())
    worker._settle_transfer_thread()
    assert worker._transfer.returned == 2
    worker._transfer.hand_back(object())
    worker._settle_transfer_thread()
    assert worker._transfer.returned == 2
    stream.idle = True
    worker._settle_transfer_thread()
    assert worker._transfer.returned == 0


def test_a_submit_only_transfer_thread_keeps_no_fence_bookkeeping(
    fake_lmcache_modules, monkeypatch
):
    """Native ATOM's OFFLOAD_MP_TRANSFER_THREAD submits as main does: nothing
    counts its queued requests, which only the polling thread's fence reads,
    and it has no fence."""
    monkeypatch.setattr(mp_worker.envs, "OFFLOAD_MP_TRANSFER_THREAD", True)
    adapter = _WorkerAdapter()
    worker = _worker(adapter, off_step=False)
    counted = []
    monkeypatch.setattr(
        worker, "_submit_queued", lambda *a: counted.append(a), raising=False
    )
    worker.start_load_kv(_load_step(7))
    assert worker._transfer.drain(5.0)
    assert len(adapter.loads) == 1 and not counted
    assert worker._queued_requests == {}
    with pytest.raises(NotImplementedError, match="transfers_off_step"):
        worker.fence(["7"])


def test_a_round_during_close_settles_without_the_worker_s_handle(
    fake_lmcache_modules, monkeypatch
):
    """`close` drains the thread it is shutting down, and a round can still run
    meanwhile: it hands a finished future back to the thread it runs on."""
    worker = _worker(_WorkerAdapter())
    request = _unprovable_request("save")
    worker._submit_save(request, _step_event(worker))
    _settle(worker)
    worker._pending_saves["save:16:1"].future.ready = True
    transfer = worker._transfer
    real_close = transfer.close

    def close_after_a_round(timeout=None):
        assert transfer.wait_round(transfer.request_round(), 5.0)
        real_close(timeout)

    monkeypatch.setattr(transfer, "close", close_after_a_round)
    worker.close()

    assert transfer._error is None
    assert worker._polled.finished_saving == {request.save_operation}
    assert worker._transfer is None


def _hold_the_transfer_thread(worker):
    """Park the transfer thread in a submission; returns the release."""
    entered, release = threading.Event(), threading.Event()

    def slow_submit():
        entered.set()
        release.wait(5.0)

    worker._transfer.submit(slow_submit)
    assert entered.wait(5.0)
    return release


@pytest.mark.parametrize("kind", ["load", "save"])
def test_a_transfer_that_finished_is_not_failed_while_rounds_are_held(
    fake_lmcache_modules, monkeypatch, kind
):
    """A slow submit holds back the round that would settle a finished
    transfer; past its deadline, nothing on the forward thread fails it, and
    the next round reports it."""
    now = [1000.0]
    monkeypatch.setattr(transfer.time, "monotonic", lambda: now[0])
    worker = _worker(_WorkerAdapter())
    worker._transfer_deadline_s = 30.0
    request = _unprovable_request(kind)
    getattr(worker, f"_submit_{kind}")(request, _step_event(worker))
    _settle(worker)
    pending = getattr(worker, f"_pending_{kind}s")
    release = _hold_the_transfer_thread(worker)
    try:
        [entry] = pending.values()
        entry.future.ready = True
        now[0] = 1031.0
        worker._settle_transfer_thread()
    finally:
        release.set()
    _settle(worker)
    finished = worker.get_finished()
    operation = request.load_operation if kind == "load" else request.save_operation
    assert operation in (
        finished.finished_loading if kind == "load" else finished.finished_saving
    )


def test_close_leaves_the_adapter_to_a_thread_that_did_not_stop(
    fake_lmcache_modules, monkeypatch
):
    """A thread still inside a submit or a poll may be using the adapter's
    futures: it is not shut down under it."""
    adapter = _WorkerAdapter()
    worker = _worker(adapter)
    transfer_obj = worker._transfer
    real_close = transfer_obj.close
    monkeypatch.setattr(transfer_obj, "close", lambda timeout=None: False)
    worker.close()
    assert not adapter.shutdown_called
    assert worker._transfer is None
    real_close(5.0)


def test_the_fence_covers_queued_transfers_and_fails_stop(fake_lmcache_modules):
    """A queued save reaches the server after the fence; it must hold it.

    Bounded by the worker's own transfer deadline.
    """
    worker = _worker(_WorkerAdapter())
    worker._transfer_deadline_s = 0.05
    worker._queued_requests["7"] = 1
    worker.fence(["8"])  # someone else's transfer: no wait
    with pytest.raises(RuntimeError, match="timed out after 0s, the transfer deadline"):
        worker.fence(["7"])


def test_the_fence_names_what_waiting_cannot_clear(fake_lmcache_modules):
    """No polling thread, its error, or an unprovable submit: raised at once."""
    from atom.kv_transfer.offload.mp.transfer import _UnprovableSubmission

    inline = mp_worker.LMCacheMPConnector(_config())
    inline._adapter = _WorkerAdapter()
    with pytest.raises(NotImplementedError, match="transfers_off_step"):
        inline.fence(["7"])
    worker = _worker(_WorkerAdapter())
    worker._queued_requests["7"] = 1
    worker._transfer._error = ConnectionError("server went away")
    with pytest.raises(ConnectionError, match="server went away"):
        worker.fence(["7"])
    worker._queued_requests.clear()
    worker._pending_saves["save:7:1"] = mp_worker._PendingSave(
        completion="7", future=_UnprovableSubmission(), start=0, end=8
    )
    with pytest.raises(RuntimeError, match="submit of save:7:1 raised"):
        worker.fence(["7"])


def test_get_finished_wakes_the_transfer_thread_while_transfers_are_pending(
    fake_lmcache_modules, monkeypatch
):
    """An idle engine queues nothing, so only this call can wake the poll."""
    worker = _worker(_WorkerAdapter())
    looks = []
    monkeypatch.setattr(worker._transfer, "look", looks.append)
    worker.get_finished()
    assert looks == []
    worker._pending_loads["load:5:1"] = mp_worker._PendingLoad(
        completion=LoadOperationId(req_id=5, generation=1), future=_WorkerFuture()
    )
    worker.get_finished()
    assert looks == [mp_worker._LAST_LOOK_WAIT_S]


def test_queued_token_drops_keep_the_connector_stepping(monkeypatch):
    """The last request retiring queues the drop of its worker token copy; with
    nothing else running, only `has_pending_work` makes the engine dispatch the
    metadata that carries it."""
    import array

    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase

    sched, _worker, _ = _shipping_pair()
    monkeypatch.setattr(
        ChunkedOffloadSchedulerBase, "has_pending_work", lambda s: False
    )
    sched._lookup_client = SimpleNamespace()
    monkeypatch.setattr(sched, "_pending_session_ends", dict)
    assert not sched.has_pending_work()
    seq = SimpleNamespace(id=7, token_ids=array.array("i", range(64)))
    sched._token_shipper.ship(seq, 32)
    sched._token_shipper.drop(7, seq)
    assert sched.has_pending_work()
    sched._token_shipper.take_drops()
    assert not sched.has_pending_work()


def test_the_vllm_plugin_path_resolves_shipped_tokens():
    """The plugin builds the concrete worker without the shell; the worker's own
    `start_load_kv` must resolve what the scheduler shipped."""
    import array

    sched, worker, received = _shipping_pair()
    seq = SimpleNamespace(id=9, token_ids=array.array("i", range(300)))
    save = sched._build_save_request(seq, 0, 256, None, [0], False)
    _send(worker, save)
    assert received == [list(range(256))]


def test_resolve_does_not_copy_the_prefix():
    """`resolve` only extends the worker copy; the list is built where a
    transfer is submitted, and the prefix it names survives later shipments."""
    import array

    from atom.kv_transfer.offload.mp.token_shipping import TokenPrefix

    sched, worker, _ = _shipping_pair()
    seq = SimpleNamespace(id=7, token_ids=array.array("i", range(600)))
    first = SimpleNamespace(req_id=7, token_ids=sched._token_shipper.ship(seq, 256))
    meta = SimpleNamespace(requests=[first])
    worker._tokens.resolve(meta)
    assert isinstance(first.token_ids, TokenPrefix) and len(first.token_ids) == 256
    later = SimpleNamespace(req_id=7, token_ids=sched._token_shipper.ship(seq, 600))
    worker._tokens.resolve(SimpleNamespace(requests=[later]))
    assert token_list(first.token_ids) == list(range(256))
    assert token_list(later.token_ids) == list(range(600))
