# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Kimi's query layout travels from its constructor through binding to MLA."""

import ast
import functools
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def _load(path, names, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if getattr(node, "name", None) in names
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        )
    ]
    if methods is not None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                node.body = [
                    item for item in node.body if getattr(item, "name", None) in methods
                ]
    tree.body = [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *body]
    exec(  # noqa: S102 - only production definitions from this checkout
        compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace
    )


@pytest.fixture
def runtime(monkeypatch):
    context = {
        "__name__": __name__,
        "torch": torch,
        "dataclass": dataclass,
        "field": field,
        "contextmanager": contextmanager,
        "_forward_context_local": threading.local(),
    }
    _load(
        "atom/utils/forward_context.py",
        {"ForwardContext", "get_forward_context", "side_stream"},
        context,
    )
    ctx = context["ForwardContext"](
        context=SimpleNamespace(
            is_prefill=True,
            is_dummy_run=False,
            scheduled_tokens=5,
            forward_mode=SimpleNamespace(prefill_token_graph=True),
        ),
        attn_metadata=SimpleNamespace(
            slot_mapping=torch.tensor([7, 3, 11, 2, 13, -1, -1, -1])
        ),
    )
    context["_forward_context"] = ctx
    ctx.tr_graph = "replay"
    graph = {
        "torch": torch,
        "NamedTuple": NamedTuple,
        "functools": functools,
        "_local": threading.local(),
        "forward_context": SimpleNamespace(side_stream=context["side_stream"]),
    }
    _load(
        "atom/utils/region_graph.py",
        {"TokenGraphQueryLayout", "_recorder", "eager_region"},
        graph,
    )
    layout = graph["TokenGraphQueryLayout"]
    mode = SimpleNamespace(plugin=False, quantized=False, sparse=True, fail=False)
    seen = []

    class MLA(nn.Module):
        def __init__(self, layer_num, sparse):
            super().__init__()
            self.layer_num, self.is_sparse_mla = layer_num, sparse
            self.split_prefill_recording = False

        def prefill_split_supported(self):
            return not self.is_sparse_mla

        def records_prefill_split(self):
            return False

        def forward(self, query, k_nope, k_rope, positions, q_scale):
            slots = ctx.attn_metadata.slot_mapping
            seen.append(
                (
                    query.shape[0],
                    k_nope.shape[0],
                    k_rope.shape[0],
                    positions.shape[-1],
                    slots,
                    positions,
                    q_scale,
                )
            )
            if mode.fail:
                raise RuntimeError("attention failed")
            return query[:, :3] + k_nope[:, :3] + k_rope[:, :3]

    class Projection(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Attention(nn.Module):
        def __init__(self, *args, layer_num, mla_modules, **kwargs):
            super().__init__()
            self.impl = MLA(layer_num, mla_modules.is_sparse)
            self.layer_num, self.base_attention, self.use_mla = layer_num, None, True
            self.indexer = mla_modules.indexer

    quant = Enum("QuantType", {"No": 0, "per_Token": 1, "per_1x128": 2})
    fp8, fp4 = torch.float8_e4m3fn, torch.uint8
    hf = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        dsa_layer_indices=(0, 2, 4),
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        num_hidden_layers=4,
    )
    config = SimpleNamespace(
        hf_config=hf,
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
        max_model_len=32,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        prefill_context_parallel_size=1,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "MLAAttention": MLA,
        "MLAModules": SimpleNamespace,
        "Attention": Attention,
        "TokenGraphQueryLayout": layout,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "_extract_layer_idx": lambda prefix: int(prefix.split(".")[1]),
        "indexer_projection_fusable": lambda *args: True,
        "_is_e8m0_blockscale": lambda *args: False,
        "qrep_tp_override": lambda *args: {},
        "QuantType": quant,
        "dtypes": SimpleNamespace(fp8=fp8, fp4x2=fp4),
        "_RMS_FUSABLE_QUANT_TYPES": (),
        "is_plugin_mode": lambda: mode.plugin,
        "effective_layer_quant": lambda config, prefix: (
            (quant.per_Token, fp8)
            if mode.quantized and prefix.endswith("q_b_proj")
            else (quant.No, None)
        ),
    }
    for name in (
        "FusedIndexerProjection",
        "MergedReplicatedLinear",
        "RMSNorm",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "NoPositionalRotaryEmbedding",
        "Indexer",
    ):
        namespace[name] = Projection
    _load(
        "atom/models/kimi_k3/model.py",
        {"_text_config", "KimiFullAttention"},
        namespace,
        {"__init__"},
    )
    receiver = {
        "torch": torch,
        "MLAAttention": MLA,
        "TokenGraphQueryLayout": layout,
        "get_forward_context": context["get_forward_context"],
        "get_current_atom_config": lambda: config,
        "is_plugin_mode": lambda: mode.plugin,
        "eager_region": graph["eager_region"],
        "mark_spliting_op": lambda **kw: lambda f: f,
        "fake_": lambda *args: None,
    }
    _load(
        "atom/model_ops/base_attention.py",
        {
            "_mla_token_graph_forward",
            "unified_attention_with_output_base",
            "_attention",
            "_attention_region",
        },
        receiver,
    )
    builder_ns = {
        "torch": torch,
        "MLAAttention": MLA,
        "TokenGraphQueryLayout": layout,
        "PageUnitGeometryMixin": type("Geometry", (), {}),
        "GDNStateMixin": type("State", (), {}),
        "MLA_ROWS": "mla",
        "is_plugin_mode": lambda: mode.plugin,
    }
    _load(
        "atom/model_ops/attentions/kimi_mla_gdn_attn.py",
        {"_KimiMLAGDNCommon"},
        builder_ns,
        {"build_kv_cache_tensor", "supports_token_graph"},
    )

    def make(layer=0):
        hf.dsa_layer_indices = (0, 2, 4) if mode.sparse else ()
        owner = namespace["KimiFullAttention"](
            config, None, prefix=f"layers.{layer}.self_attn"
        )
        config.compilation_config.static_forward_context["layer"] = owner.attn
        return owner

    def bind(owner, target=None):
        builder = builder_ns["_KimiMLAGDNCommon"]()
        builder.model_runner = SimpleNamespace(
            config=config,
            hf_text_config=hf,
            has_mla_indexer=False,
            model=nn.ModuleList([owner]) if target is None else target,
        )
        builder.pool_rows = {"mla": {owner.attn: 0}}
        builder.kv_pool = SimpleNamespace(
            entry_dim=4, layer=lambda *args: torch.zeros(2, 4)
        )
        case.builder = builder
        with monkeypatch.context() as patch:
            patch.setitem(
                sys.modules,
                "atom.config",
                SimpleNamespace(KVCacheTensor=SimpleNamespace),
            )
            return builder.build_kv_cache_tensor(owner.attn)

    args = [
        torch.arange(64, dtype=torch.bfloat16).view(8, 8),
        None,
        torch.ones(8, 8, dtype=torch.bfloat16),
        torch.ones(8, 8, dtype=torch.bfloat16),
        torch.arange(8),
        "layer",
        True,
        None,
    ]
    case = SimpleNamespace(
        make=make,
        bind=bind,
        mode=mode,
        config=config,
        ctx=ctx,
        args=args,
        call=receiver["unified_attention_with_output_base"],
        seen=seen,
        layout=layout,
        receiver=receiver,
    )
    return case


@pytest.mark.parametrize("layer", [0, 2], ids=["first-stage", "later-stage"])
@pytest.mark.parametrize("position_axes", [1, 3])
def test_constructor_builder_and_receiver_preserve_live_rows(
    runtime, layer, position_axes
):
    owner = runtime.make(layer)
    declaration = owner.attn.impl.prefill_query_layout
    assert declaration == runtime.layout(layer, torch.bfloat16)
    bound = runtime.bind(owner)
    assert bound.layer_num == layer
    assert owner.attn.impl.token_graph_query_layout is declaration
    if position_axes > 1:
        runtime.args[4] = runtime.args[4].expand(position_axes, -1)
    slots = runtime.ctx.attn_metadata.slot_mapping
    out = runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (5, 5, 5, 5)
    torch.testing.assert_close(runtime.seen[-1][4], slots[:5])
    torch.testing.assert_close(runtime.seen[-1][5], runtime.args[4][..., :5])
    assert runtime.ctx.attn_metadata.slot_mapping is slots
    torch.testing.assert_close(out[:5], runtime.args[0][:5, :3] + 2)
    assert out.shape == (8, 3) and not out[5:].any()


@pytest.mark.parametrize("case", ["quantized", "dense", "draft", "plugin"])
def test_constructor_declines_unsupported_owners(runtime, case):
    runtime.mode.quantized = case == "quantized"
    runtime.mode.sparse = case != "dense"
    runtime.mode.plugin = case == "plugin"
    owner = runtime.make(4 if case == "draft" else 0)
    assert getattr(owner.attn.impl, "prefill_query_layout", None) is None


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "wrong_layer",
        "wrong_dtype",
        "draft_identity",
        "unknown_target",
        "plugin",
        "dp",
        "pcp",
    ],
)
def test_builder_revokes_layout_without_target_provenance(runtime, case):
    owner = runtime.make()
    if case == "unknown":
        owner.attn.impl.prefill_query_layout = True
    elif case == "wrong_layer":
        owner.attn.impl.prefill_query_layout = runtime.layout(1, torch.bfloat16)
    elif case == "wrong_dtype":
        owner.attn.impl.prefill_query_layout = runtime.layout(0, torch.float8_e4m3fn)
    elif case == "plugin":
        runtime.mode.plugin = True
    elif case == "dp":
        runtime.config.parallel_config.data_parallel_size = 2
    elif case == "pcp":
        runtime.config.prefill_context_parallel_size = 2
    target = nn.ModuleList([]) if case == "draft_identity" else None
    if case == "unknown_target":
        target = SimpleNamespace()
    runtime.bind(owner, target)
    assert owner.attn.impl.token_graph_query_layout is None
    slots = runtime.ctx.attn_metadata.slot_mapping
    runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (8, 8, 8, 8)
    assert runtime.seen[-1][4] is slots


def test_receiver_requires_target_binding_before_using_declared_layout(runtime):
    owner = runtime.make()
    assert owner.attn.impl.prefill_query_layout is not None
    assert getattr(owner.attn.impl, "token_graph_query_layout", None) is None
    slots = runtime.ctx.attn_metadata.slot_mapping
    runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (8, 8, 8, 8)
    assert runtime.seen[-1][4] is slots


def test_prefill_graph_gate_accepts_bound_native_sparse_queries(runtime):
    runtime.bind(runtime.make())
    assert runtime.builder.supports_token_graph


@pytest.mark.parametrize("reason", ["quantized", "draft", "plugin", "dp", "pcp"])
def test_prefill_graph_gate_keeps_unsupported_sparse_queries_eager(runtime, reason):
    runtime.mode.quantized = reason == "quantized"
    runtime.mode.plugin = reason == "plugin"
    if reason == "dp":
        runtime.config.parallel_config.data_parallel_size = 2
    if reason == "pcp":
        runtime.config.prefill_context_parallel_size = 2
    runtime.bind(runtime.make(4 if reason == "draft" else 0))
    assert not runtime.builder.supports_token_graph


def test_plain_attention_does_not_enter_the_token_graph_helper(runtime):
    runtime.bind(runtime.make())
    runtime.ctx.tr_graph = None
    runtime.ctx.context.forward_mode.prefill_token_graph = False
    runtime.receiver["_mla_token_graph_forward"] = lambda *args: pytest.fail(
        "plain call entered graph helper"
    )
    runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (8, 8, 8, 8)


@pytest.mark.parametrize(
    "case", ["scale", "dtype", "eager", "decode", "dummy", "full", "plugin"]
)
def test_receiver_preserves_non_sliceable_or_unpadded_calls(runtime, case):
    runtime.bind(runtime.make())
    if case == "scale":
        runtime.ctx.context.forward_mode.prefill_token_graph = False
        runtime.args[1] = torch.ones(1, 8)
    elif case == "dtype":
        runtime.ctx.context.forward_mode.prefill_token_graph = False
        runtime.args[0] = runtime.args[0].float()
    elif case == "eager":
        runtime.ctx.context.forward_mode.prefill_token_graph = False
    elif case == "decode":
        runtime.ctx.context.is_prefill = False
    elif case == "dummy":
        runtime.ctx.context.is_dummy_run = True
    elif case == "full":
        runtime.ctx.context.scheduled_tokens = 8
    else:
        runtime.mode.plugin = True
    slots = runtime.ctx.attn_metadata.slot_mapping
    out = runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (8, 8, 8, 8)
    assert runtime.seen[-1][4] is slots
    assert runtime.seen[-1][6] is runtime.args[1]
    assert out.dtype == runtime.args[0].dtype


@pytest.mark.parametrize("case", ["scale", "dtype"])
def test_bound_padded_query_refuses_changed_runtime_layout(runtime, case):
    runtime.bind(runtime.make())
    if case == "scale":
        runtime.args[1] = torch.ones(1, 8)
    else:
        runtime.args[0] = runtime.args[0].float()
    slots = runtime.ctx.attn_metadata.slot_mapping
    with pytest.raises(ValueError, match="bound token-graph query layout"):
        runtime.call(*runtime.args)
    assert not runtime.seen
    assert runtime.ctx.attn_metadata.slot_mapping is slots


def test_slots_are_restored_on_failure_and_next_live_step(runtime):
    runtime.bind(runtime.make())
    slots = runtime.ctx.attn_metadata.slot_mapping
    runtime.mode.fail = True
    with pytest.raises(RuntimeError, match="attention failed"):
        runtime.call(*runtime.args)
    assert runtime.ctx.attn_metadata.slot_mapping is slots
    runtime.mode.fail = False
    runtime.ctx.context.scheduled_tokens = 2
    new_slots = torch.tensor([12, 4, -1, -1, -1, -1, -1, -1])
    runtime.ctx.attn_metadata.slot_mapping = new_slots
    out = runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (2, 2, 2, 2)
    torch.testing.assert_close(runtime.seen[-1][4], new_slots[:2])
    assert runtime.ctx.attn_metadata.slot_mapping is new_slots
    assert not out[2:].any()
