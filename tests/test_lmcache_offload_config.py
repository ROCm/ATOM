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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config, cfg: setattr(config, "kv_cache_dtype", "bf16"),
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
