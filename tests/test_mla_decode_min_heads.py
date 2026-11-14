# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The 12-head MLA decode: where MLAAttention pads a query to, and that the
metadata builder plans each layer's work at the width that layer dispatches."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom import config as atom_config_module
from atom.plugin import prepare
from atom.utils.forward_context import AttentionMetaData, MlaWorkPlan


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
    """Native ATOM on a gfx950 rank with an fp8 KV cache, no DCP and the
    Kimi-K3 interactive band's DSpark 7 (an 8-token verify): the configuration
    the 12-head decode is for. Tests change one thing each."""
    cfg = SimpleNamespace(
        kv_cache_dtype="fp8",
        speculative_config=SimpleNamespace(num_speculative_tokens=7),
        decode_context_parallel_size=1,
        kv_cache_block_size=1,
        hf_config=SimpleNamespace(),
        enable_tbo=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        dcp_config=SimpleNamespace(
            enable_query_replication=False,
            enable_project_before_merge=False,
            comm_backend="ag",
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
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


def _layer(cfg, *, sparse=False, num_heads=12, layer_num=0, **kwargs):
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
    layer = MLAAttention(
        num_heads, 576, 1.0, 1, cfg.kv_cache_dtype, layer_num, modules, **kwargs
    )
    cfg.compilation_config.static_forward_context[f"layers.{layer_num}"] = (
        SimpleNamespace(impl=layer)
    )
    return layer


def _builder(cfg, *, sparse, num_heads=12):
    builder = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    builder.model_runner = SimpleNamespace(config=cfg)
    builder.num_attention_heads = num_heads
    builder.is_sparse = sparse
    builder._configure_decode_head_widths(cfg)
    return builder


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


def test_mixed_model_plans_each_layer_at_its_own_width(native):
    """Kimi-K3 with DSA layers: the DSA layers dispatch 16 heads and read
    work_*, the dense ones dispatch 12 and get a plan of their own at 12."""
    dense = _layer(native, layer_num=3)
    sparse = _layer(native, sparse=True, layer_num=7)
    builder = _builder(native, sparse=True)

    assert dense.padded_num_heads == builder.dense_work_num_heads == 12
    assert sparse.padded_num_heads == builder.padded_num_attention_heads == 16
    assert dense.head_pad == 0


def test_mixed_model_plans_the_dense_layers_apart_at_sixteen_too(native):
    """Off the 12-head gate both kinds of layer dispatch 16, but at qlen 1
    work_* is planned over the DSA top-k; the dense layers still get a plan
    of their own over the dense KV."""
    native.kv_cache_dtype = "bf16"
    dense = _layer(native, layer_num=3)
    _layer(native, sparse=True, layer_num=7)
    builder = _builder(native, sparse=True)

    assert dense.padded_num_heads == builder.dense_work_num_heads == 16


def test_mixed_model_under_dcp_plans_no_dense_width(native, monkeypatch):
    _layer(native, layer_num=3)
    monkeypatch.setattr(aiter_mla, "get_dcp_world_size", lambda: 2)
    monkeypatch.setattr(attention_mla, "get_dcp_world_size", lambda: 2)
    builder = _builder(native, sparse=True)

    assert builder.dense_work_num_heads is None


def test_dense_model_plans_one_width(native):
    layer = _layer(native)
    builder = _builder(native, sparse=False)

    assert layer.padded_num_heads == builder.padded_num_attention_heads == 12
    assert builder.dense_work_num_heads is None


def test_all_sparse_model_plans_no_dense_width(native):
    """GLM-5 / DeepSeek-V3.2: every layer sparse, so no second plan."""
    _layer(native, sparse=True)
    builder = _builder(native, sparse=True)

    assert builder.padded_num_attention_heads == 16
    assert builder.dense_work_num_heads is None


def test_force_dense_sparse_model_plans_the_dense_width(native):
    """ATOM_GLM5_FORCE_DENSE_MLA: a sparse model whose layers run dense."""
    layer = _layer(native)
    builder = _builder(native, sparse=True)

    assert layer.padded_num_heads == builder.dense_work_num_heads == 12


def test_explicit_floor_wins_and_is_not_planned_for(native):
    """A DSpark draft passes its own floor and plans its own work."""
    draft = _layer(native, min_query_heads=32)
    builder = _builder(native, sparse=True)

    assert draft.padded_num_heads == 32
    assert builder.dense_work_num_heads is None


def _set(monkeypatch, cfg, change):
    if change == "gfx942":
        monkeypatch.setattr(attention_mla, "_gfx", lambda: "gfx942")
    elif change == "bf16 kv":
        cfg.kv_cache_dtype = "bf16"
    elif change == "non-persistent":
        monkeypatch.setenv("ATOM_USE_TRITON_MLA", "1")
    elif change.startswith("qlen "):
        qlen = int(change.split()[1])
        cfg.speculative_config = (
            SimpleNamespace(num_speculative_tokens=qlen - 1) if qlen > 1 else None
        )
    elif change == "vllm":
        monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "vllm")
    else:
        raise AssertionError(change)


@pytest.mark.parametrize(
    "change, expected",
    [
        ("gfx942", 16),
        ("bf16 kv", 16),
        ("non-persistent", 16),
        # aiter serves 12 up to qlen 10; ATOM takes it where it is faster.
        ("qlen 1", 16),
        ("qlen 4", 16),
        ("qlen 5", 12),
        ("qlen 6", 12),
        ("qlen 9", 16),
        ("qlen 10", 16),
        ("qlen 11", 16),
        ("vllm", 16),
    ],
)
def test_gate_reaches_both_call_sites(native, monkeypatch, change, expected):
    _set(monkeypatch, native, change)
    layer = _layer(native)
    builder = _builder(native, sparse=False)

    assert layer.padded_num_heads == builder.padded_num_attention_heads == expected


def test_fewer_heads_keep_sixteen(native):
    layer = _layer(native, num_heads=8)
    builder = _builder(native, sparse=False, num_heads=8)

    assert layer.padded_num_heads == builder.padded_num_attention_heads == 16


def test_dcp_keeps_sixteen(native, monkeypatch):
    monkeypatch.setattr(attention_mla, "get_dcp_world_size", lambda: 2)

    assert attention_mla.mla_dense_min_query_heads(native, 12) == 16


def test_vllm_builder_plans_the_layers_padding(native, monkeypatch):
    """AttentionForVllmMLA runs MLAAttention.__init__, and the plugin builder
    reads the width off the layers it serves (`mla_layers_query_heads`)."""
    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "vllm")
    layers = [_layer(native, layer_num=n) for n in range(2)]
    heads, floor = attention_mla.mla_layers_query_heads(layers, 96)

    assert attention_mla.mla_kernel_num_heads(heads, floor) == 16
    assert all(layer.padded_num_heads == 16 for layer in layers)


def test_vllm_builder_refuses_layers_padding_differently():
    layers = [
        SimpleNamespace(num_heads=12, min_query_heads=16),
        SimpleNamespace(num_heads=12, min_query_heads=12),
    ]
    with pytest.raises(ValueError, match="different floors"):
        attention_mla.mla_layers_query_heads(layers, 12)


class _Buffer(SimpleNamespace):
    pass


def _plan_vars(prefixes):
    var = {}
    for prefix in prefixes:
        for name in aiter_mla.PERSISTENT_WORK_BUFFERS:
            var[f"{prefix}{name}"] = torch.zeros(4, dtype=torch.int32)
    for name, value in {
        "cu_seqlens_q": [0, 1, 2],
        "kv_indptr": [0, 5000, 9000],
        "kv_last_page_lens": [1, 1],
        "sparse_kv_indptr": [0, 2048, 4096],
        "sparse_kv_last_page_lens": [1, 1],
    }.items():
        var[name] = _Buffer(gpu=torch.tensor(value, dtype=torch.int32))
    return var


def test_set_buffers_plans_the_dense_layers_off_the_dense_kv(monkeypatch):
    """At qlen 1 a sparse model plans work_* over the top-k selection; the
    dense plan has to come off the dense kv_indptr, at the dense width."""
    calls = []

    def fake_plan(qo, kv_indptr, last, heads, _nk, _causal, wmd, *rest, **kw):
        calls.append((kv_indptr.tolist(), heads, wmd))

    monkeypatch.setattr(aiter_mla, "get_mla_metadata_v1", fake_plan)
    builder = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    var = _plan_vars(("", "dense_"))
    builder.model_runner = SimpleNamespace(forward_vars=var)
    builder.kv_shuffled = False
    builder.block_size = 1
    builder.is_sparse = True
    builder.persistent_num_heads = 16
    builder.dense_work_num_heads = 12
    builder.dtype_q = builder.dtype_kv = FP8

    ctx = builder.set_mla_persistent_worker_buffers(2, 1)

    assert calls == [
        ([0, 2048, 4096], 16, var["work_meta_data"]),
        ([0, 5000, 9000], 12, var["dense_work_meta_data"]),
    ]
    assert ctx["work_num_heads"] == 16
    plan = ctx["dense_work_plan"]
    assert plan.num_heads == 12 and plan.work_meta_data is var["dense_work_meta_data"]
    # The capture path builds its metadata from this dict.
    assert AttentionMetaData(**ctx).dense_work_plan is plan


def _decode(layer, monkeypatch, attn_metadata):
    seen = {}

    def fake_decode(q, _kv, _o, *args, **kwargs):
        seen["heads"] = q.shape[1]
        seen["work_meta_data"] = kwargs["work_meta_data"]
        return None, torch.zeros(q.shape[0], q.shape[1])

    monkeypatch.setattr(attention_mla, "mla_decode_fwd", fake_decode)
    q = torch.zeros(2, layer.num_heads, 576)
    layer._forward_decode(q, torch.zeros(8, 576), attn_metadata, return_lse=True)
    return seen


def _metadata(**kwargs):
    one = torch.zeros(3, dtype=torch.int32)
    return AttentionMetaData(
        cu_seqlens_q=one,
        kv_indptr=one,
        kv_indices=one,
        kv_last_page_lens=one,
        max_seqlen_q=1,
        **kwargs,
    )


def _work(tag):
    return {
        name: torch.full((1,), tag, dtype=torch.int32)
        for name in aiter_mla.PERSISTENT_WORK_BUFFERS
    }


def test_dense_layer_decodes_on_the_dense_plan(native, monkeypatch):
    layer = _layer(native)
    dense = MlaWorkPlan(12, *_work(2).values())
    seen = _decode(
        layer,
        monkeypatch,
        _metadata(work_num_heads=16, dense_work_plan=dense, **_work(1)),
    )

    assert seen["heads"] == 12
    assert seen["work_meta_data"] is dense.work_meta_data


def test_dense_layer_refuses_a_plan_of_another_width(native, monkeypatch):
    layer = _layer(native)
    with pytest.raises(AssertionError, match="for 16 query heads"):
        _decode(layer, monkeypatch, _metadata(work_num_heads=16, **_work(1)))
