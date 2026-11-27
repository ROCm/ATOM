# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Kimi graph, residual and storage contracts across prerequisite interfaces."""

import ast
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
NAMESPACES = itertools.count()


def _load(path, names, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [
        n
        for n in tree.body
        if getattr(n, "name", None) in names
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    ]
    if methods:
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                node.body = [
                    n for n in node.body if getattr(n, "name", None) in methods
                ]
    tree.body.insert(0, ast.ImportFrom("__future__", [ast.alias("annotations")], 0))
    exec(  # noqa: S102 - production definitions from the current checkout
        compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace
    )


@pytest.fixture
def residual(monkeypatch):
    ns = {"__name__": __name__, "torch": torch, "nn": nn, "dataclass": dataclass}
    _load(
        "atom/model_ops/kimi_k3/attention_residual.py",
        {"ColumnTail", "materialize_column_output"},
        ns,
    )
    monkeypatch.setattr(
        sys.modules[__name__], "ColumnTail", ns["ColumnTail"], raising=False
    )
    _load("atom/utils/custom_register.py", {"eager_direct_call"}, ns)
    state = SimpleNamespace(plugin=False, rows=True, columns=False)
    ns.update(
        is_plugin_mode=lambda: state.plugin,
        tensor_model_parallel_all_reduce=lambda x: x.clone(),
        get_tensor_model_parallel_rank=lambda: 0,
    )
    for name in ("is_plugin_mode", "eager_direct_call", "materialize_column_output"):
        monkeypatch.setattr(sys.modules[__name__], name, ns[name], raising=False)
    _load(
        "atom/models/kimi_k3/model.py",
        {"KimiDecoderLayer"},
        ns,
        {
            "_mlp",
            "_post_attention",
            "post_attention_forward",
            "_post_attention_parts",
            "aux_hidden_state",
        },
    )
    monkeypatch.setattr(
        sys.modules[__name__], "KimiDecoderLayer", ns["KimiDecoderLayer"], raising=False
    )
    layer = object.__new__(ns["KimiDecoderLayer"])
    nn.Module.__init__(layer)
    layer.post_attention_op = True
    layer.layer_name = "layer"
    layer.block_sparse_moe = SimpleNamespace(
        tp_size=2,
        column_gather_name="",
        takes_row_tail=lambda n: state.rows,
        _can_defer_column_tail=lambda x: state.columns,
        row_sharded_moe=lambda hidden, rows, prefix: hidden + prefix,
    )
    layer.mlp_attn_res = SimpleNamespace()

    class ResidualSite:
        def window(self, bank):
            return bank

        def __call__(self, prefix, bank, add):
            value = add if prefix is None else prefix + add
            return value, value.clone()

    layer.mlp_attn_res = ResidualSite()
    layer._ffn = lambda x: (x + 1, x + 2)

    class Rows:
        def __init__(self, *args):
            pass

        def shard(self, x):
            return x

        scatter = shard
        gather = shard

    monkeypatch.setitem(
        sys.modules,
        "atom.distributed.sequence_parallel",
        SimpleNamespace(SequenceParallel=Rows),
    )
    name = f"kimi_residual_contract_{next(NAMESPACES)}"
    library = torch.library.Library(name, "DEF")
    library.define(
        "post(Tensor x, Tensor? prefix, Tensor? bank, str name) -> (Tensor, Tensor, Tensor)"
    )
    library.impl(
        "post", lambda x, p, b, n: layer.post_attention_forward(x, p, b), "CPU"
    )
    torch.library.register_fake(
        f"{name}::post",
        lambda x, p, b, n: (
            torch.empty_like(x),
            torch.empty_like(x),
            torch.empty_like(x),
        ),
    )
    library.define("reduce(Tensor x) -> Tensor")
    library.impl("reduce", lambda x: x.clone(), "CPU")
    torch.library.register_fake(f"{name}::reduce", lambda x: torch.empty_like(x))
    library.define("gather(Tensor x, int world) -> Tensor")
    library.impl("gather", lambda x, world: torch.cat([x] * world, dim=-1), "CPU")
    torch.library.register_fake(
        f"{name}::gather", lambda x, world: x.new_empty(x.shape[0], x.shape[1] * world)
    )
    ops = getattr(torch.ops, name)
    for public, op in (
        ("kimi_k3_post_attention_forward", ops.post),
        ("tbo_all_reduce", ops.reduce),
        ("kimi_k3_gather_columns", ops.gather),
    ):
        monkeypatch.setattr(torch.ops.aiter, public, op, raising=False)
    x = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    prefix, bank = torch.full_like(x, 3), torch.zeros(2, 2, 4)
    yield SimpleNamespace(
        layer=layer, state=state, ns=ns, ops=ops, x=x, prefix=prefix, bank=bank
    )


def test_native_completed_residual_has_no_zero_storage(residual, monkeypatch):
    calls = []
    original = torch.zeros_like

    def zero(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", zero)
    out = residual.layer._post_attention(residual.x, residual.prefix, residual.bank)
    assert out[1:] == (None, None)
    assert not calls
    torch.testing.assert_close(out[0], 2 * (residual.x + residual.prefix))
    assert torch.equal(residual.layer.aux_hidden_state((*out, residual.bank)), out[0])


def test_fixed_tensor_boundary_keeps_fresh_independent_zeros(residual, monkeypatch):
    empty_like = torch.empty_like

    def poisoned(*args, **kwargs):
        return empty_like(*args, **kwargs).fill_(float("nan"))

    monkeypatch.setattr(torch, "empty_like", poisoned)
    first = residual.ops.post(residual.x, residual.prefix, residual.bank, "layer")
    assert all(isinstance(x, torch.Tensor) for x in first)
    assert not first[1].any() and not first[2].any()
    saved = first[0].clone()
    first[1].fill_(17)
    assert not first[2].any() and torch.equal(first[0], saved)
    second = residual.ops.post(residual.x + 4, residual.prefix, residual.bank, "layer")
    assert not second[1].any() and not second[2].any()
    assert torch.equal(first[0], saved) and torch.all(first[1] == 17)


def test_plugin_keeps_the_fixed_boundary(residual):
    residual.state.plugin = True
    out = residual.layer._post_attention(residual.x, residual.prefix, residual.bank)
    assert all(isinstance(x, torch.Tensor) for x in out)
    assert not out[1].any() and not out[2].any()


@pytest.mark.parametrize("compiled", [False, True])
def test_typed_columns_stay_outside_the_post_attention_op(residual, compiled):
    residual.state.rows = False
    residual.state.columns = True
    residual.layer.block_sparse_moe.column_gather_name = "gather"
    column = residual.ns["ColumnTail"]
    residual.layer._ffn = lambda x: (column(x[:, :2].clone(), "gather", 2), None)

    def run(x, prefix, bank):
        out = residual.layer._post_attention(x, prefix, bank)
        return residual.ns["materialize_column_output"](out[1])

    call = torch.compile(run, backend="aot_eager", fullgraph=True) if compiled else run
    out = call(residual.x, residual.prefix, residual.bank)
    value = (residual.x + residual.prefix)[:, :2]
    torch.testing.assert_close(out, torch.cat([value, value], dim=-1))


def test_direct_fixed_op_materializes_columns_before_returning(residual):
    residual.state.rows = False
    column = residual.ns["ColumnTail"]
    residual.layer._ffn = lambda x: (column(x[:, :2].clone(), "gather", 2), None)
    out = residual.ops.post(residual.x, residual.prefix, residual.bank, "layer")
    assert all(isinstance(x, torch.Tensor) for x in out)
    value = (residual.x + residual.prefix)[:, :2]
    torch.testing.assert_close(out[1], torch.cat([value, value], dim=-1))
    assert not out[2].any()


def test_compilation_keeps_the_fixed_tensor_contract(residual):
    def run(x, prefix, bank):
        return residual.layer._post_attention(x, prefix, bank)

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
    out = compiled(residual.x, residual.prefix, residual.bank)
    assert all(isinstance(x, torch.Tensor) for x in out)
    torch.testing.assert_close(out[0], 2 * (residual.x + residual.prefix))
    assert not out[1].any() and not out[2].any()


@pytest.mark.parametrize(
    "warming,capturing", [(False, False), (True, False), (False, True), (True, True)]
)
def test_row_tail_keeps_capture_and_its_warmup_on_the_same_path(
    monkeypatch, warming, capturing
):
    namespace = {
        "nn": nn,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing)
        ),
        "get_forward_context": lambda: SimpleNamespace(in_hipgraph=warming),
        "envs": SimpleNamespace(
            ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0,
        ),
    }
    _load(
        "atom/models/kimi_k3/model.py",
        {"KimiSparseMoeBlock", "_MOE_ROW_SHARD_MIN_TOKENS"},
        namespace,
        {"takes_row_tail"},
    )
    monkeypatch.setitem(
        sys.modules,
        "atom.utils.tbo.ubatching",
        SimpleNamespace(tbo_active=lambda: False),
    )
    got = namespace["KimiSparseMoeBlock"].takes_row_tail(
        SimpleNamespace(tp_size=8), 8192
    )
    assert got is (not warming and not capturing)


@pytest.mark.parametrize("offset", [-1, 0])
def test_graph_projection_keeps_the_loaded_large_row_split(offset):
    gemms = []

    def mm(x, weight):
        gemms.append(weight)
        return torch.nn.functional.linear(x, weight)

    namespace = {"nn": nn, "torch": torch, "tgemm": SimpleNamespace(mm=mm)}
    _load(
        "atom/models/kimi_k3/model.py",
        {"KimiKDAAttention", "_KDA_IN_PROJ_SPLIT_MIN_TOKENS"},
        namespace,
        {"_project_in"},
    )
    rows = namespace["_KDA_IN_PROJ_SPLIT_MIN_TOKENS"] + offset
    random = torch.Generator().manual_seed(19)
    weight = torch.randn(20, 4, generator=random).bfloat16()
    fb = torch.randn(4, 2, generator=random).bfloat16()
    inputs = torch.randn(rows, 4, generator=random).bfloat16()

    class Projection:
        def __init__(self):
            self.weight, self.calls = weight, 0

        def __call__(self, x, x_scale=None):
            self.calls += 1
            return torch.nn.functional.linear(x, self.weight)

    layer = SimpleNamespace(
        _split_in_proj=True,
        local_proj_size=4,
        num_local_heads=2,
        head_dim=2,
        in_proj=Projection(),
        fuse_f_a_quant=False,
        f_b_proj=lambda x: torch.nn.functional.linear(x, fb),
    )
    qkv, output_gate, beta, gate = namespace["KimiKDAAttention"]._project_in(
        layer, inputs, None
    )
    assert len(gemms) == (2 if offset == 0 else 0)
    assert layer.in_proj.calls == (0 if offset == 0 else 1)
    fused = torch.nn.functional.linear(inputs, weight)
    for got, expected in (
        (qkv, fused[:, :12]),
        (output_gate, fused[:, 12:16]),
        (beta, fused[:, 16:18]),
        (gate, torch.nn.functional.linear(fused[:, 18:].contiguous(), fb)),
    ):
        torch.testing.assert_close(got, expected)
    if gemms:
        assert [w.storage_offset() for w in gemms] == [0, 64]
        assert all(
            w.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
            for w in gemms
        )
