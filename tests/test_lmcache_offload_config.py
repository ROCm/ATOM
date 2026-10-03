# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import sys
import types
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload import config as offcfg


def _config():
    return SimpleNamespace(
        model="org/model",
        model_tag="org/model",
        kv_cache_dtype="fp8",
        index_cache_dtype="fp8",
        kv_cache_block_size=256,
        tensor_parallel_size=4,
        decode_context_parallel_size=2,
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
        ),
        hf_config=SimpleNamespace(
            num_hidden_layers=61,
            num_attention_heads=128,
            num_key_value_heads=16,
            hidden_size=7168,
            head_dim=128,
            kv_head_dim=512,
            index_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            compress_ratios=[4, 128, 0],
            indexer_dtype="fp8",
        ),
    )


def _lmcache_config():
    return SimpleNamespace(chunk_size=8192)


def test_page_namespace_is_stable_for_equivalent_config():
    first = offcfg.build_page_namespace(_config(), _lmcache_config(), 4)
    second = offcfg.build_page_namespace(
        deepcopy(_config()),
        deepcopy(_lmcache_config()),
        4,
    )

    assert first == second
    assert first.startswith("org/model::atom-page-v")


def test_page_namespace_supports_torch_dtype_in_speculative_config():
    torch = pytest.importorskip("torch")
    config = _config()
    config.speculative_config.draft_model_hf_config = SimpleNamespace(
        torch_dtype=torch.bfloat16
    )

    bf16_namespace = offcfg.build_page_namespace(config, _lmcache_config(), 4)
    config.speculative_config.draft_model_hf_config.torch_dtype = torch.float16
    fp16_namespace = offcfg.build_page_namespace(config, _lmcache_config(), 4)

    assert bf16_namespace != fp16_namespace


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config, cfg: setattr(config, "kv_cache_dtype", "bf16"),
        lambda config, cfg: setattr(config, "index_cache_dtype", "fp4"),
        lambda config, cfg: setattr(config.hf_config, "indexer_dtype", "bf16"),
        lambda config, cfg: setattr(config.hf_config, "kv_head_dim", 576),
        lambda config, cfg: setattr(config.hf_config, "index_head_dim", 160),
        lambda config, cfg: setattr(
            config.hf_config,
            "compress_ratios",
            [4, 64, 0],
        ),
        lambda config, cfg: setattr(config, "kv_cache_block_size", 128),
        lambda config, cfg: setattr(cfg, "chunk_size", 4096),
        lambda config, cfg: setattr(config, "decode_context_parallel_size", 1),
        lambda config, cfg: setattr(
            config.speculative_config,
            "num_speculative_tokens",
            4,
        ),
    ],
    ids=[
        "kv-dtype",
        "effective-index-dtype",
        "index-dtype",
        "kv-head-dim",
        "index-head-dim",
        "compression",
        "block",
        "chunk",
        "dcp",
        "speculative",
    ],
)
def test_page_namespace_changes_for_meaningful_geometry(mutate):
    config = _config()
    cfg = _lmcache_config()
    original = offcfg.build_page_namespace(config, cfg, 4)

    mutate(config, cfg)

    assert offcfg.build_page_namespace(config, cfg, 4) != original


def test_page_namespace_changes_when_code_layout_version_changes():
    current = offcfg.build_page_namespace(_config(), _lmcache_config(), 4)
    future = offcfg.build_page_namespace(
        _config(),
        _lmcache_config(),
        4,
        layout_version=offcfg.PAGE_LAYOUT_VERSION + 1,
    )

    assert future != current


def test_page_namespace_separates_explicit_offload_layout_families():
    dense = _config()
    hybrid = deepcopy(dense)
    dense.kv_transfer_config = {"offload_layout": "dense"}
    hybrid.kv_transfer_config = {"offload_layout": "hybrid"}

    assert offcfg.build_page_namespace(
        dense, _lmcache_config(), 4
    ) != offcfg.build_page_namespace(hybrid, _lmcache_config(), 4)


@pytest.mark.parametrize("override", ["typo", True, 1])
def test_unknown_explicit_offload_layout_is_rejected(override):
    config = _config()
    config.kv_transfer_config = {"offload_layout": override}

    with pytest.raises(ValueError, match="unknown offload_layout"):
        offcfg.select_offload_layout(config)


@pytest.mark.parametrize(
    "model_type",
    ["qwen3_next", "qwen3_next_mtp", "qwen3_5_text", "qwen3_5_moe_text"],
)
def test_gdn_linear_model_offload_is_refused(model_type):
    # A GDN model must be turned away at config resolution, not fall through to
    # `dense` and restore a KV prefix over its stale recurrent state. Clear
    # `compress_ratios` so the hybrid branch does not claim it first.
    config = _config()
    config.hf_config.compress_ratios = None
    config.hf_config.model_type = model_type

    with pytest.raises(ValueError, match="does not support GDN"):
        offcfg.select_offload_layout(config)


@pytest.mark.parametrize(
    "model_type",
    ["qwen3_next", "qwen3_next_mtp", "qwen3_5_text", "qwen3_5_moe_text"],
)
def test_gdn_refusal_is_not_bypassed_by_offload_layout_override(model_type):
    # The refusal must run *before* an explicit override is honoured. Otherwise
    # `offload_layout: dense` on a GDN checkpoint returns `dense` early and
    # restores a KV prefix over stale recurrent state (silent wrong output).
    config = _config()
    config.hf_config.compress_ratios = None
    config.hf_config.model_type = model_type
    config.kv_transfer_config = {"offload_layout": "dense"}

    with pytest.raises(ValueError, match="does not support GDN"):
        offcfg.select_offload_layout(config)


def test_offload_layout_override_cannot_downgrade_kimi_k3_to_dense():
    # kimi_linear's KDA per-request state is owned only by the kimi_k3 layout;
    # an override to a layout with no tier for it is silent wrong output, so it
    # must be refused rather than silently honoured.
    config = _config()
    config.hf_config.compress_ratios = None
    config.hf_config.model_type = "kimi_linear"
    config.kv_transfer_config = {"offload_layout": "dense"}

    with pytest.raises(ValueError, match="owns no tier"):
        offcfg.select_offload_layout(config)


def test_offload_layout_override_still_allows_dense_hybrid_choice():
    # The downgrade guard is narrow: dense<->hybrid is a legitimate operator
    # choice (namespace separation), not a recurrent-state downgrade.
    config = _config()  # compress_ratios set -> natural "hybrid"
    config.kv_transfer_config = {"offload_layout": "dense"}
    assert offcfg.select_offload_layout(config) == "dense"


def test_minimax_m2_and_dense_model_types_still_route_to_dense():
    # The refusal must be narrow: a non-GDN model with no compress_ratios is
    # ordinary dense. MiniMax-M2 (standard attention, no NSA index cache) is
    # plain dense; only M3, whose NSA index cache needs its own tier, is carved
    # out to the PAGE-only "m3" layout (see the two tests below).
    for model_type in ("minimax_m2", "llama", None):
        config = _config()
        config.hf_config.compress_ratios = None
        config.hf_config.model_type = model_type
        assert offcfg.select_offload_layout(config) == "dense"


def test_minimax_m3_model_type_routes_to_m3():
    # M3's NSA index cache rides in transfer_tensors.block_regions, which only
    # the PAGE-only "m3" layout sources; `dense` would offload K/V and silently
    # drop the index cache, corrupting restored prefixes.
    config = _config()
    config.hf_config.compress_ratios = None
    config.hf_config.model_type = "minimax_m3"
    assert offcfg.select_offload_layout(config) == "m3"


@pytest.mark.parametrize("override", ["dense", "hybrid", "kimi_k3"])
def test_minimax_m3_rejects_incompatible_layout_override(override):
    config = _config()
    config.hf_config.compress_ratios = None
    config.hf_config.model_type = "minimax_m3"
    config.kv_transfer_config = {"offload_layout": override}

    with pytest.raises(ValueError, match="MiniMax-M3 requires offload_layout='m3'"):
        offcfg.select_offload_layout(config)


def test_minimax_m3_accepts_explicit_m3_layout():
    config = _config()
    config.hf_config.compress_ratios = None
    config.hf_config.model_type = "minimax_m3"
    config.kv_transfer_config = {"offload_layout": "m3"}

    assert offcfg.select_offload_layout(config) == "m3"


def test_minimax_m3_is_detected_by_nested_text_config_model_type():
    config = _config()
    config.hf_config.compress_ratios = None
    config.hf_config.model_type = "multimodal_wrapper"
    config.hf_config.text_config = SimpleNamespace(model_type="minimax_m3")

    assert offcfg.select_offload_layout(config) == "m3"


def test_minimax_m3_is_detected_by_nested_text_config_architecture():
    config = _config()
    config.hf_config.compress_ratios = None
    config.hf_config.model_type = "multimodal_wrapper"
    config.hf_config.text_config = SimpleNamespace(
        model_type="minimax_text_01",
        architectures=["MiniMaxM3ForCausalLM"],
    )

    assert offcfg.select_offload_layout(config) == "m3"


def test_minimax_m3_is_detected_by_architecture_name():
    # M3 is identified by architecture name, not a model_type family, so the
    # arch-name path must route to "m3" from the architectures list even when
    # the model_type string does not name M3.
    config = _config()
    config.hf_config.compress_ratios = None
    config.hf_config.model_type = "minimax_text_01"
    config.hf_config.architectures = ["MiniMaxM3ForCausalLM"]
    assert offcfg.select_offload_layout(config) == "m3"


@pytest.mark.parametrize("invalid", [True, 256.0, "256"])
@pytest.mark.parametrize("field", ["block", "chunk", "world", "hf"])
def test_page_namespace_rejects_coerced_integer_geometry(field, invalid):
    config = _config()
    cfg = _lmcache_config()
    world_size = 4
    if field == "block":
        config.kv_cache_block_size = invalid
    elif field == "chunk":
        cfg.chunk_size = invalid
    elif field == "world":
        world_size = invalid
    else:
        config.hf_config.kv_head_dim = invalid

    with pytest.raises(ValueError, match="must be an integer"):
        offcfg.build_page_namespace(config, cfg, world_size)


def test_page_namespace_rejects_numpy_boolean_geometry():
    np = pytest.importorskip("numpy")
    config = _config()
    config.hf_config.kv_head_dim = np.bool_(True)

    with pytest.raises(ValueError, match="must be an integer"):
        offcfg.build_page_namespace(config, _lmcache_config(), 4)


def test_page_namespace_accepts_numpy_integer_geometry():
    np = pytest.importorskip("numpy")
    config = _config()
    config.hf_config.kv_head_dim = np.int64(512)

    assert offcfg.build_page_namespace(config, _lmcache_config(), 4)


def test_unknown_lmcache_override_is_rejected():
    cfg = SimpleNamespace(chunk_size=8192)

    with pytest.raises(ValueError, match="unknown LMCache override"):
        offcfg.apply_extra_overrides(
            cfg,
            {"kv_connector_extra_config": {"lmcache.chunk_szie": 4096}},
        )


def _install_metadata_modules(monkeypatch):
    @dataclass
    class _Metadata:
        model_name: str
        world_size: int
        local_world_size: int
        worker_id: int
        local_worker_id: int
        kv_dtype: object
        kv_shape: tuple
        use_mla: bool
        chunk_size: int
        engine_id: str

    aiter_module = types.ModuleType("aiter")
    aiter_module.dtypes = SimpleNamespace(d_dtypes={"fp8": "torch-fp8"})
    metadata_module = types.ModuleType("lmcache.v1.metadata")
    metadata_module.LMCacheMetadata = _Metadata
    monkeypatch.setitem(sys.modules, "aiter", aiter_module)
    monkeypatch.setitem(sys.modules, "lmcache", types.ModuleType("lmcache"))
    monkeypatch.setitem(sys.modules, "lmcache.v1", types.ModuleType("lmcache.v1"))
    monkeypatch.setitem(sys.modules, "lmcache.v1.metadata", metadata_module)
    return _Metadata


def test_scheduler_and_worker_metadata_share_page_namespace(monkeypatch):
    _install_metadata_modules(monkeypatch)

    scheduler = offcfg.build_lmcache_metadata(_config(), _lmcache_config(), 4, 0)
    worker = offcfg.build_lmcache_metadata(_config(), _lmcache_config(), 4, 3)

    assert scheduler.model_name == worker.model_name
    assert scheduler.worker_id == 0
    assert worker.worker_id == 3


@pytest.mark.parametrize(
    ("world_size", "worker_ids"),
    [(1, [1]), (1, [0, 1]), (4, [4])],
)
def test_lmcache_metadata_rejects_lookup_scope_outside_replica_world(
    monkeypatch,
    world_size,
    worker_ids,
):
    _install_metadata_modules(monkeypatch)
    cfg = _lmcache_config()
    cfg.lookup_server_worker_ids = worker_ids

    with pytest.raises(ValueError) as exc_info:
        offcfg.build_lmcache_metadata(_config(), cfg, world_size, 0)
    expected = (
        "lookup_server_worker_ids must be within the replica-local world "
        f"[0, {world_size})"
    )
    assert expected in str(exc_info.value)


@pytest.mark.parametrize(
    ("world_size", "worker_ids"),
    [(1, []), (1, [0]), (4, [0, 3])],
)
def test_lmcache_metadata_accepts_lookup_scope_within_replica_world(
    monkeypatch,
    world_size,
    worker_ids,
):
    metadata_type = _install_metadata_modules(monkeypatch)
    cfg = _lmcache_config()
    cfg.lookup_server_worker_ids = worker_ids

    metadata = offcfg.build_lmcache_metadata(_config(), cfg, world_size, 0)

    assert isinstance(metadata, metadata_type)
    assert metadata.world_size == world_size
    assert metadata.worker_id == 0


@pytest.mark.parametrize("worker_ids", [0, 3])
def test_lmcache_metadata_rejects_scalar_lookup_scope(monkeypatch, worker_ids):
    _install_metadata_modules(monkeypatch)
    cfg = _lmcache_config()
    cfg.lookup_server_worker_ids = worker_ids

    with pytest.raises(ValueError, match="must be a list or tuple"):
        offcfg.build_lmcache_metadata(_config(), cfg, 4, 0)


def test_lmcache_metadata_rejects_duplicate_lookup_scope(monkeypatch):
    _install_metadata_modules(monkeypatch)
    cfg = _lmcache_config()
    cfg.lookup_server_worker_ids = [0, 0]

    with pytest.raises(ValueError, match="must not contain duplicate"):
        offcfg.build_lmcache_metadata(_config(), cfg, 4, 0)


def test_lmcache_metadata_accepts_tuple_lookup_scope(monkeypatch):
    metadata_type = _install_metadata_modules(monkeypatch)
    cfg = _lmcache_config()
    cfg.lookup_server_worker_ids = (0, 3)

    metadata = offcfg.build_lmcache_metadata(_config(), cfg, 4, 0)

    assert isinstance(metadata, metadata_type)
    assert metadata.world_size == 4


def test_page_namespace_survives_a_worker_normalising_hf_config():
    """Scheduler and worker must hash the same geometry.

    A worker normalises `hf_config` while it builds the model (Kimi-K3 derives
    `head_dim`); the scheduler never builds it. Hashing the live config gave
    them different namespaces, and every `lmcache_mp` lookup missed. `Config`
    snapshots the geometry before the config reaches either process.
    """
    import copy

    shipped = _config()
    shipped.hf_config.head_dim = None
    shipped.offload_page_hf_geometry = offcfg.snapshot_page_hf_geometry(
        shipped.hf_config
    )
    scheduler, worker = copy.deepcopy(shipped), copy.deepcopy(shipped)
    worker.hf_config.head_dim = 192  # what `_normalize_kimi_config` writes

    assert offcfg.build_page_namespace(
        scheduler, _lmcache_config(), 4
    ) == offcfg.build_page_namespace(worker, _lmcache_config(), 4)

    # Without the snapshot the two diverge -- the bug this guards against.
    del scheduler.offload_page_hf_geometry, worker.offload_page_hf_geometry
    assert offcfg.build_page_namespace(
        scheduler, _lmcache_config(), 4
    ) != offcfg.build_page_namespace(worker, _lmcache_config(), 4)
