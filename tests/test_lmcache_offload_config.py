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


def test_minimax_and_dense_model_types_still_route_to_dense():
    # The refusal must be narrow: a non-GDN model with no compress_ratios is
    # ordinary dense, including MiniMax (sparse/standard attention, no state).
    for model_type in ("minimax_m2", "minimax_m3", "llama", None):
        config = _config()
        config.hf_config.compress_ratios = None
        config.hf_config.model_type = model_type
        assert offcfg.select_offload_layout(config) == "dense"


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


def test_scheduler_and_worker_metadata_share_page_namespace(monkeypatch):
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

    scheduler = offcfg.build_lmcache_metadata(_config(), _lmcache_config(), 4, 0)
    worker = offcfg.build_lmcache_metadata(_config(), _lmcache_config(), 4, 3)

    assert scheduler.model_name == worker.model_name
    assert scheduler.worker_id == 0
    assert worker.worker_id == 3
