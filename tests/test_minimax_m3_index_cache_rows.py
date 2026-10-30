# SPDX-License-Identifier: MIT
"""Exercise production row assignment, binding and PAGE exports on CPU.

Only GPU imports/staging are replaced: the builder methods are compiled from
source and operate on real pools and tensors, including nonuniform PP spans.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField, carve
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry, PoolRowsMixin
from atom.utils import envs

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if getattr(n, "name", None) == class_name).body
    selected = [n for n in nodes if getattr(n, "name", None) in names]
    assert len(selected) == len(names)
    for node in selected:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    exec(  # noqa: S102 - repository source only
        compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), namespace
    )


def make_builder(
    monkeypatch,
    *,
    use=True,
    freq=4,
    pattern=None,
    offset=0,
    span=range(60),
    fp4=True,
    nvfp4=True,
    sparse=None,
):
    aiter = ModuleType("aiter")
    aiter.dtypes = SimpleNamespace(d_dtypes={"fp8": torch.float8_e4m3fn})
    monkeypatch.setitem(__import__("sys").modules, "aiter", aiter)
    fp4_module = ModuleType("atom.model_ops.minimax_m3.index_fp4")
    fp4_module.FP4_GROUP = 16
    definitions(
        "atom/model_ops/minimax_m3/index_fp4.py",
        {"fp4_index_row_bytes"},
        fp4_module.__dict__,
    )
    monkeypatch.setitem(__import__("sys").modules, fp4_module.__name__, fp4_module)
    monkeypatch.setattr(envs, "ATOM_M3_SPARSE_KV_NVFP4", nvfp4)
    hf = SimpleNamespace(
        use_index_cache=use,
        index_topk_freq=freq,
        index_topk_pattern=pattern,
        index_skip_topk_offset=offset,
        sparse_attention_config={
            "sparse_index_dim": 128,
            "sparse_attention_freq": sparse or [1] * 60,
        },
    )
    ns = {"PretrainedConfig": object}
    definitions(
        "atom/models/minimax_m3/model.py",
        {
            "_sparse_attention_layer_ids",
            "_sparse_attention_layer_ordinals",
            "_should_skip_minimax_m3_index_topk",
        },
        ns,
    )

    class Layer:
        def __init__(self, i):
            self.layer_num = i
            self.base_attention = True
            self.use_mla = False
            self.num_kv_heads, self.head_dim = 1, 128
            self.k_scale = self.v_scale = None
            self.impl = SimpleNamespace(
                is_indexed_sparse_attention=bool(
                    hf.sparse_attention_config["sparse_attention_freq"][i]
                ),
                skip_index_topk=ns["_should_skip_minimax_m3_index_topk"](hf, i)[0],
                index_cache=None,
                index_scale=None,
                index_stage=None,
                index_stage_slots=None,
            )

    layers = [Layer(i) for i in span]
    scope = {
        "torch": torch,
        "logger": logging.getLogger("atom"),
        "NamedTuple": NamedTuple,
        "INDEX_ROWS": "index",
        "envs": envs,
        "KvGeometry": KvGeometry,
        "MhaKvPool": MhaKvPool,
        "EntryField": EntryField,
        "carve": carve,
    }
    source = "atom/model_ops/attentions/aiter_attention.py"
    definitions(
        source,
        {
            "_IndexCacheSpec",
            "_is_indexed_sparse_attention",
            "_resolve_index_cache_dtype",
        },
        scope,
    )
    methods = {
        "_module_kinds",
        "_declare_kv_pools",
        "_index_cache_spec",
        "_index_cache_fields",
        "_allocate_kv_pools",
        "build_kv_cache_tensor",
        "get_kv_transfer_tensors",
    }
    definitions(source, methods, scope, "AiterAttentionMetadataBuilder")
    Builder = type("Builder", (PoolRowsMixin,), {name: scope[name] for name in methods})
    builder = Builder()
    builder.block_size, builder.block_ratio = 128, 1
    builder._has_sparse_attention = True
    builder.model_runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: iter(layers)),
        draft_shares_kv_pool=lambda: False,
        config=SimpleNamespace(
            hf_config=hf,
            kv_cache_dtype="fp8",
            index_cache_dtype="fp4" if fp4 else "fp8",
            max_model_len=4096,
        ),
        device="cpu",
        _sparse_attention_topk_cache_state={},
    )
    builder._index_fp4_stage = lambda dim: (
        torch.zeros(1, 128, dim, dtype=torch.bfloat16),
        torch.arange(128),
    )
    builder._kv_nvfp4_stage = lambda nh, hd: (
        torch.zeros(1, 2, 128, nh, hd, dtype=torch.bfloat16),
        torch.arange(128),
    )
    declared = builder._declare_kv_pools()
    buf = torch.empty(
        sum(p.pool_bytes(2) for p in declared.values()), dtype=torch.uint8
    )
    builder._allocate_kv_pools(2, buf)
    for layer in layers:
        builder.build_kv_cache_tensor(layer)
    return builder, layers


@pytest.mark.parametrize("fp4,nvfp4", [(False, False), (True, False), (True, True)])
def test_only_computing_layers_own_index_bytes_and_transfer_regions(
    monkeypatch, fp4, nvfp4
):
    builder, layers = make_builder(monkeypatch, fp4=fp4, nvfp4=nvfp4)
    assert [m.layer_num for m in builder.pool_rows["index"]] == list(range(0, 60, 4))
    assert (
        sum(n for k, n in builder.row_counts().items() if isinstance(k, KvGeometry))
        == 60
    )
    for layer in layers:
        assert layer.k_cache.numel() > 0
        assert (
            layer.impl.index_topk_cache_state
            is builder.model_runner._sparse_attention_topk_cache_state
        )
        if layer.layer_num % 4:
            assert layer.impl.index_cache is None
            assert layer.impl.index_scale is None
            assert layer.impl.index_stage is None
        else:
            layer.impl.index_cache.view(torch.uint8).fill_(layer.layer_num + 1)
    regions = builder.get_kv_transfer_tensors().pages
    index_regions = [r for r in regions if ".index.layer_" in r.region.semantic_role]
    assert len(index_regions) == 15
    # Export order must point at the same nonconsecutive model layers bound above.
    for region, layer in zip(index_regions, layers[::4]):
        assert region.region.base_addr == layer.impl.index_cache.data_ptr()
    compact_bytes = sum(p.entry_bytes for p in builder.kv_pools.values())
    full, _ = make_builder(monkeypatch, use=False, fp4=fp4, nvfp4=nvfp4)
    full_bytes = sum(p.entry_bytes for p in full.kv_pools.values())
    assert full_bytes - compact_bytes == 45 * 128 * (72 if fp4 else 128)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"use": False}, list(range(60))),
        ({"freq": 1}, list(range(60))),
        ({"offset": 2}, [0, 1, 2] + list(range(6, 60, 4))),
        ({"pattern": "CSSCCSSS", "span": range(8)}, [0, 3, 4]),
        ({"span": range(15, 30)}, [16, 20, 24, 28]),
        ({"span": range(1, 4)}, []),
        ({"sparse": [0, 1] * 30, "span": range(10)}, [1, 9]),
    ],
)
def test_schedule_is_taken_from_model_including_pp_and_patterns(
    monkeypatch, kwargs, expected
):
    builder, _ = make_builder(monkeypatch, **kwargs)
    assert [m.layer_num for m in builder.pool_rows.get("index", {})] == expected
    if not expected:
        assert all(not p.extra_fields for p in builder.kv_pools.values())


def test_nvfp4_reusing_layer_writes_kv_without_index_buffers(monkeypatch):
    """Execute rope_cache with kernel-boundary spies and real staging buffers."""
    import sys
    from unittest.mock import Mock

    _builder, layers = make_builder(monkeypatch)
    impl = layers[1].impl
    impl.layer_num = 1
    impl.num_heads = impl.num_kv_heads = impl.num_idx_heads = 1
    impl.head_dim = impl.index_head_dim = impl.index_q_size = 128
    impl.kv_cache_dtype, impl.index_cache_dtype = "fp8", "fp4"
    impl.q_norm = impl.k_norm = impl.index_q_norm = impl.index_k_norm = SimpleNamespace(
        weight=torch.ones(128), variance_epsilon=1e-6
    )
    impl.rotary_emb = SimpleNamespace(rotary_dim=64)
    sparse = ModuleType("atom.model_ops.minimax_m3.sparse_attn")
    sparse.to_page16_shuffle = Mock(side_effect=AssertionError("NVFP4 is not SHUFFLE"))
    monkeypatch.setitem(sys.modules, sparse.__name__, sparse)
    model = ModuleType("atom.models.minimax_m3.model")
    model._minimax_m3_cos_sin_cache = lambda *_: torch.ones(16, 64)
    monkeypatch.setitem(sys.modules, model.__name__, model)
    quantize = Mock()
    sys.modules["atom.model_ops.minimax_m3.index_fp4"].kv_nvfp4_store = quantize
    fused = Mock()
    sys.modules["aiter"].fused_qknorm_idxrqknorm = fused
    ns = {"torch": torch, "aiter": sys.modules["aiter"], "ForwardContext": object}
    definitions(
        "atom/model_ops/attention_mha.py",
        {"rope_cache"},
        ns,
        "SparseMHAPagedAttentionImpl",
    )
    slots = torch.tensor([1, 129])
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(slot_mapping=slots),
        kv_cache_data={"layer_1": layers[1]},
    )
    qkv = torch.zeros(2, 5 * 128, dtype=torch.bfloat16)
    ns["rope_cache"](impl, None, None, None, qkv, torch.arange(2), ctx)
    args, kwargs = fused.call_args
    assert kwargs["skip_index_branch"] is True
    assert kwargs["asm_layout"] is False
    assert args[15] is None  # index cache
    assert args[18] is None  # index Q output
    assert quantize.call_count == 1
    assert quantize.call_args.args[1] is slots
    assert impl._index_q is None
    assert impl._index_q_cache_key_info == ((2, 1, 128), qkv.dtype, qkv.device)


def test_compact_page_copy_preserves_every_computing_layer(monkeypatch):
    producer, p_layers = make_builder(monkeypatch)
    consumer, d_layers = make_builder(monkeypatch)
    p_pages = producer.get_kv_transfer_tensors().pages
    d_pages = consumer.get_kv_transfer_tensors().pages
    for i, (p, d) in enumerate(zip(p_pages, d_pages)):
        assert p.region.semantic_role == d.region.semantic_role
        p.view.fill_((i + 1) % 256)
        d.view.zero_()
        # Receive page 1 while preserving page 0 (an existing cached prefix).
        d.view[1].copy_(p.view[1])
    for p, d in zip(p_layers, d_layers):
        if p.impl.index_cache is None:
            assert d.impl.index_cache is None
            continue
        for field in ("index_cache", "index_scale"):
            src, dst = getattr(p.impl, field), getattr(d.impl, field)
            assert torch.equal(src[1], dst[1])
            assert not torch.count_nonzero(dst[0])
