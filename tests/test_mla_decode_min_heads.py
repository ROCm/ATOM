# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The 12-head MLA decode: where MLAAttention pads a query to, and that the
metadata builder plans the step's one work plan at the same width."""

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom import config as atom_config_module
from atom.plugin import prepare


def _import_with_lent_arch(name):
    """AITER's Triton helpers read the GPU arch at import; a CPU runner has
    none. Nothing here launches a kernel, so lend one for the import."""
    arch_info = pytest.importorskip("aiter.ops.triton.utils._triton.arch_info")
    lent = arch_info._CACHED_ARCH is None
    if lent:
        arch_info._CACHED_ARCH = "gfx950"
    try:
        return pytest.importorskip(name)
    finally:
        if lent:
            arch_info._CACHED_ARCH = None


attention_mla = _import_with_lent_arch("atom.model_ops.attention_mla")
aiter_mla = _import_with_lent_arch("atom.model_ops.attentions.aiter_mla")
dtypes = attention_mla.dtypes

FP8 = torch.float8_e4m3fn
MLAAttention = attention_mla.MLAAttention


@pytest.fixture
def native(monkeypatch):
    """Native ATOM on a gfx950 rank with 12 heads (96 at tp8), an fp8 KV
    cache, no indexer, no DCP and the Kimi-K3 interactive band's DSpark 7 (an
    8-token verify): the configuration the 12-head decode is for. Tests change
    one thing each."""
    cfg = SimpleNamespace(
        kv_cache_dtype="fp8",
        speculative_config=SimpleNamespace(num_speculative_tokens=7),
        decode_context_parallel_size=1,
        kv_cache_block_size=1,
        hf_config=SimpleNamespace(num_attention_heads=96, kv_lora_rank=512),
        enable_tbo=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        dcp_config=SimpleNamespace(
            enable_query_replication=False,
            enable_project_before_merge=False,
            comm_backend="ag",
        ),
    )
    monkeypatch.setattr(atom_config_module, "_current_atom_config", cfg)
    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "atom")
    monkeypatch.setattr(attention_mla, "_gfx", lambda: "gfx950")
    monkeypatch.setattr(
        attention_mla, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(attention_mla, "dcp_persistent_supported", lambda: True)
    monkeypatch.setattr(attention_mla, "dcp_prefill_merge_bf16_ok", lambda: True)
    # A CPU runner's aiter cannot name the arch's fp8; gfx950's is e4m3fn.
    monkeypatch.setitem(dtypes.d_dtypes, "fp8", FP8)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    for name in ("ATOM_USE_TRITON_MLA", "ATOM_MLA_KV_SHUFFLED", "ATOM_MLA_PAGE_SIZE"):
        monkeypatch.delenv(name, raising=False)
    return cfg


def _layer(cfg, *, sparse=False, num_heads=12, **kwargs):
    modules = attention_mla.MLAModules(
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        rotary_emb=None,
        q_proj=nn.Linear(1, 1),
        kv_b_proj=nn.Linear(1, 1),
        o_proj=nn.Identity(),
        indexer=None,
        is_sparse=sparse,
        topk_tokens=2048 if sparse else None,
    )
    return MLAAttention(
        num_heads, 576, 1.0, 1, cfg.kv_cache_dtype, 0, modules, **kwargs
    )


class _Stop(Exception):
    pass


class _Runner(SimpleNamespace):
    @property
    def has_mla_indexer(self):
        # Read on the line after the decode width: nothing past it is needed.
        raise _Stop


def _builder_width(cfg, monkeypatch, num_heads=12):
    """Run AiterMLAMetadataBuilder.__init__ up to its decode width. The base
    builder's __init__ pins host buffers, which a CPU runner cannot."""

    def base_init(self, runner):
        self.model_runner = runner
        self.max_bs = runner.max_bs
        self.num_attention_heads = num_heads

    monkeypatch.setattr(aiter_mla.CommonAttentionBuilder, "__init__", base_init)
    builder = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    with pytest.raises(_Stop):
        builder.__init__(_Runner(config=cfg, max_bs=4, block_size=64))
    return builder.padded_num_attention_heads


@pytest.mark.parametrize(
    "arch, q_dtype, kv_dtype, qlen, persistent, expected",
    [
        ("gfx950", FP8, FP8, 1, True, 12),
        ("gfx950", torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, 1, True, 12),
        ("gfx950", FP8, FP8, 10, True, 12),
        ("gfx950", FP8, FP8, 11, True, 16),  # 12 * 11 > 128
        ("gfx950", FP8, FP8, 1, False, 16),
        ("gfx950", torch.bfloat16, FP8, 1, True, 16),
        ("gfx950", FP8, torch.bfloat16, 1, True, 16),
        # aiter serves 12 here too; ATOM mirrors only the fp8 branch.
        ("gfx950", torch.bfloat16, torch.bfloat16, 8, True, 16),
        ("gfx950", torch.uint8, torch.uint8, 1, True, 16),  # byte KV, not fp8
        ("gfx942", FP8, FP8, 1, True, 16),
        ("gfx1250", FP8, FP8, 1, True, 16),
    ],
)
def test_aiter_gate(arch, q_dtype, kv_dtype, qlen, persistent, expected):
    assert (
        attention_mla.mla_decode_min_heads(q_dtype, kv_dtype, qlen, persistent, arch)
        == expected
    )


@pytest.mark.parametrize(
    "num_heads, floor, expected",
    [(12, 12, 12), (13, 12, 16), (12, 16, 16), (24, 12, 32), (96, 12, 96)],
)
def test_kernel_width_takes_a_twelve_head_floor(num_heads, floor, expected):
    assert attention_mla.mla_kernel_num_heads(num_heads, floor) == expected


def _set(monkeypatch, cfg, change):
    if change == "gfx942":
        monkeypatch.setattr(attention_mla, "_gfx", lambda: "gfx942")
    elif change == "bf16 kv":
        cfg.kv_cache_dtype = "bf16"
    elif change == "triton mla":
        monkeypatch.setenv("ATOM_USE_TRITON_MLA", "1")
    elif change == "dp2":
        monkeypatch.setattr(
            attention_mla, "get_dp_group", lambda: SimpleNamespace(world_size=2)
        )
    elif change == "dp2, no dpa persistent":
        _set(monkeypatch, cfg, "dp2")
        monkeypatch.setattr(
            attention_mla, "supports_dpa_persistent_mode", lambda *a, **k: False
        )
    elif change == "indexer":
        cfg.hf_config.index_topk = 2048
    elif change.startswith("qlen "):
        qlen = int(change.split()[1])
        cfg.speculative_config = (
            SimpleNamespace(num_speculative_tokens=qlen - 1) if qlen > 1 else None
        )
    elif change == "vllm":
        monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "vllm")
    elif change != "none":
        raise AssertionError(change)


@pytest.mark.parametrize(
    "change, expected",
    [
        ("none", 12),
        ("gfx942", 16),
        ("bf16 kv", 16),
        ("triton mla", 16),
        ("dp2", 12),
        ("dp2, no dpa persistent", 16),
        ("indexer", 16),
        # aiter serves 12 up to qlen 10; ATOM takes it only at qlen 8.
        ("qlen 1", 16),
        ("qlen 4", 16),
        ("qlen 5", 16),
        ("qlen 6", 16),
        ("qlen 7", 16),
        ("qlen 9", 16),
        ("vllm", 16),
    ],
)
def test_layer_and_builder_pad_alike(native, monkeypatch, change, expected):
    _set(monkeypatch, native, change)
    layer = _layer(native)

    assert layer.padded_num_heads == _builder_width(native, monkeypatch) == expected


def test_model_with_an_indexer_keeps_sixteen_on_every_layer(native, monkeypatch):
    """Its dense and sparse layers share the one work_* plan, at 16."""
    _set(monkeypatch, native, "indexer")
    dense = _layer(native)
    sparse = _layer(native, sparse=True)

    assert dense.padded_num_heads == sparse.padded_num_heads == 16
    assert _builder_width(native, monkeypatch) == 16


def test_kv_shuffled_keeps_sixteen(native, monkeypatch):
    assert attention_mla.mla_dense_min_query_heads(native, 12, 1) == 12
    monkeypatch.setenv("ATOM_MLA_KV_SHUFFLED", "1")
    assert attention_mla.mla_dense_min_query_heads(native, 12, 1) == 16


def test_dcp_keeps_sixteen(native):
    assert attention_mla.mla_dense_min_query_heads(native, 12, 1) == 12
    assert attention_mla.mla_dense_min_query_heads(native, 12, 2) == 16


def test_fewer_heads_keep_sixteen(native, monkeypatch):
    layer = _layer(native, num_heads=8)

    assert layer.padded_num_heads == 16
    assert _builder_width(native, monkeypatch, num_heads=8) == 16


def test_explicit_floor_wins(native):
    """A DSpark draft passes its own floor and plans its own work."""
    assert _layer(native, min_query_heads=32).padded_num_heads == 32


def test_each_decode_width_logs_once(native, monkeypatch, caplog):
    """Logged unpadded too (12 -> 12), once per width."""
    monkeypatch.setattr(MLAAttention, "_decode_widths_logged", set())
    with caplog.at_level(logging.INFO, logger="atom"):
        _layer(native)
        _layer(native)
        _layer(native, num_heads=8)

    assert [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("MLA decode query heads")
    ] == [
        "MLA decode query heads: 12 -> 12 (floor 12)",
        "MLA decode query heads: 8 -> 16 (floor 16)",
    ]
