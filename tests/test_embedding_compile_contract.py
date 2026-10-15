# SPDX-License-Identifier: MIT
"""Embedding compilation must not manufacture a weight update."""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch


@pytest.mark.parametrize(
    "name", ["_masked_embedding_read_only", "_replicated_embedding_read_only"]
)
def test_embedding_declares_read_only_inputs(name):
    source = Path(__file__).resolve().parents[1] / "atom/model_ops/embed_head.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    guard = next(
        node
        for node in function.decorator_list
        if isinstance(node, ast.Call) and node.func.id == "torch_compile_guard"
    )
    mutations = next(
        (keyword.value for keyword in guard.keywords if keyword.arg == "mutates_args"),
        None,
    )
    assert mutations is not None, "unknown mutation schema invalidates weight bindings"
    assert ast.literal_eval(mutations) == []


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compiled ROCm/CUDA regression"
)
@pytest.mark.parametrize("name", ["masked_embedding", "replicated_embedding"])
def test_compiled_embedding_preserves_bound_weight(name):
    # Import the real op on GPU; missing AITER symbols must fail, not skip.
    from atom.model_ops import embed_head
    from atom.models.deepseek_v4.mono.weights import _signature

    op = getattr(embed_head, name)
    schema = getattr(torch.ops.aiter, f"_{name}_read_only").default._schema
    assert not any(argument.alias_info for argument in schema.arguments)
    weight = torch.nn.Parameter(
        torch.randn(512, 7168, device="cuda", dtype=torch.bfloat16), requires_grad=False
    )
    original = weight.detach().clone()
    signature = _signature(weight)
    ids = torch.tensor([3, 19, 511, -1], device="cuda", dtype=torch.int32)

    def lookup(tokens, bound_weight):
        if name == "masked_embedding":
            return op(tokens, bound_weight, 0, 512)
        return op(tokens, bound_weight)

    compiled = torch.compile(lookup, fullgraph=True)
    for tokens in (ids, ids.flip(0), ids):
        expected = torch.nn.functional.embedding(tokens.clamp_min(0).long(), original)
        expected[tokens < 0] = 0
        actual = compiled(tokens, weight)
        torch.cuda.synchronize()
        assert _signature(weight) == signature
        torch.testing.assert_close(weight, original, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Production level3 uses standalone_compile and persists AOT metadata.
    # Verify both the artifact and its reload, not only torch.compile's cache.
    from torch.fx.experimental.proxy_tensor import make_fx

    graph = make_fx(lookup)(ids, weight)
    artifact = torch._inductor.standalone_compile(graph, [ids, weight])
    with TemporaryDirectory(prefix="atom-embedding-compile-") as cache:
        artifact.save(path=cache, format="unpacked")
        reloaded = torch._inductor.CompiledArtifact.load(path=cache, format="unpacked")
        for fn in (artifact, reloaded):
            actual = fn(ids, weight)
            torch.cuda.synchronize()
            assert _signature(weight) == signature
            tensors, _ = torch.utils._pytree.tree_flatten(actual)
            assert len(tensors) == 1
            torch.testing.assert_close(tensors[0], expected, rtol=0, atol=0)

    # Do not weaken the binding invariant to accommodate compilation: a real
    # in-place reload must still change the signature.
    with torch.no_grad():
        weight.add_(1)
    assert _signature(weight) != signature
