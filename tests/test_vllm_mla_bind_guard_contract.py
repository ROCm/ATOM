# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA bind point may not squeeze unconditionally.

``AttentionForVllmMLA.bind_kv_cache`` drops vLLM 0.29's head slot so consumers
below keep seeing 0.28's three-dimensional MLA page. Upstream does the same
thing with a bare ``squeeze(1)``; this layer must not, because it can also be
handed a 0.28-shaped ``[B, N, C]`` page whose ``N`` is 1 -- MLA's kernel block
size -- and a bare squeeze would eat the block dimension instead of a head slot.
The page would still be a tensor, still bind, and be silently wrong.

``tests/plugin/test_vllm_mla_bind_kv_cache.py`` tests the behaviour on real
tensors and is the better test, but importing that layer pulls in vLLM and CI's
unit-test job both lacks vLLM and passes ``--ignore=tests/plugin``, so it cannot
run there. This file reads the source statically for the same reason
``test_kv_pool_contract`` does: the check is worth less than the behavioural one
but it is the one that runs where the regression would land.

It is deliberately narrow. It does not check that the squeeze is correct -- only
that the decision to squeeze is still conditioned on both halves of the shape.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

LAYER = (
    pathlib.Path(__file__).resolve().parent.parent
    / "atom/plugin/vllm/attention/layer_mla.py"
)
CLASS, METHOD = "AttentionForVllmMLA", "bind_kv_cache"


def _bind_method() -> ast.FunctionDef:
    tree = ast.parse(LAYER.read_text(), filename=str(LAYER))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if cls.name != CLASS:
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == METHOD:
                return fn
    pytest.fail(f"{LAYER.name} no longer defines {CLASS}.{METHOD}")


def _squeeze_calls(node: ast.AST) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "squeeze"
    ]


def _guards_both_halves(test: ast.expr) -> bool:
    """A test that reads both the rank and the size of the axis it drops."""
    source = ast.dump(test)
    reads_rank = "attr='dim'" in source
    reads_axis = "attr='shape'" in source or "attr='size'" in source
    return reads_rank and reads_axis


def test_the_squeeze_is_conditioned_on_rank_and_head_count():
    method = _bind_method()
    squeezes = _squeeze_calls(method)
    assert squeezes, f"{CLASS}.{METHOD} no longer squeezes anything"

    guarded = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and _guards_both_halves(node.test)
        and _squeeze_calls(node)
    ]
    assert len(guarded) == len(squeezes), (
        f"{CLASS}.{METHOD} squeezes outside a rank-and-head-count guard. "
        "Upstream squeezes unconditionally; this layer cannot -- a 0.28-shaped "
        "[B, N=1, C] page would lose its block dimension."
    )
