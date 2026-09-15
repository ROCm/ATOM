# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA bind point may not squeeze unconditionally.

``AttentionForVllmMLA.bind_kv_cache`` drops vLLM 0.29's head slot so consumers
below keep seeing 0.28's three-dimensional MLA page. Upstream does the same
thing with a bare ``squeeze(1)``; this layer must not, because it can also be
handed a 0.28-shaped ``[B, N, C]`` page whose ``N`` is 1 -- MLA's kernel block
size -- and a bare squeeze would eat the block dimension instead of a head slot.
The page would still be a tensor, still bind, and be silently wrong.

``tests/plugin/test_vllm_mla_bind_kv_cache.py`` drives the real method and is
the better test, but importing that layer pulls in vLLM, and CI's unit-test job
both lacks vLLM and passes ``--ignore=tests/plugin`` -- so it cannot run where
the regression would land. This file covers the same contract without importing
the layer, the way ``test_kv_pool_contract`` does.

It does not pattern-match the guard's syntax, which would fail any equivalent
rewrite while passing mutations that keep the shape of the code. It lifts the
guard expression and the squeeze call out of the source and *evaluates them*
against real tensors, so what is asserted is the decision the two lines make,
not how they are spelled. ``kv_cache.size(1) == 1``, reordered conjuncts and a
nested ``if`` all read the same to it.

What it assumes about the source: the page is squeezed by a call on the
parameter itself, inside an ``if``, and the guard mentions no name other than
that parameter. Rewriting the method past those assumptions fails this test,
deliberately and with a message saying so -- the override is load-bearing
enough to be worth re-reading by hand.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

LAYER = (
    pathlib.Path(__file__).resolve().parent.parent
    / "atom/plugin/vllm/attention/layer_mla.py"
)
CLASS, METHOD = "AttentionForVllmMLA", "bind_kv_cache"

# Every shape the bind point can be handed, and the page each must yield.
# The 0.29 view is the only one that loses an axis; the 0.28 page with a single
# block row is the one upstream's unconditional squeeze would destroy.
CASES = (
    ((4, 1, 16, 8), (4, 16, 8), "0.29's [B, H=1, N, C] view: the head slot goes"),
    ((4, 2, 16, 8), (4, 2, 16, 8), "more than one head slot: not this layer's call"),
    ((4, 1, 8), (4, 1, 8), "0.28's [B, N=1, C] page: N is a block row, not a head"),
    ((4, 16, 8), (4, 16, 8), "0.28's [B, N, C] page: already what consumers want"),
)


def _bind_method() -> ast.FunctionDef:
    tree = ast.parse(LAYER.read_text(), filename=str(LAYER))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if cls.name != CLASS:
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == METHOD:
                return fn
    pytest.fail(f"{LAYER.name} no longer defines {CLASS}.{METHOD}")


def _page_squeeze(method: ast.FunctionDef) -> tuple[ast.Call, ast.If]:
    """The ``<page>.squeeze(...)`` call, and the ``if`` that decides to run it."""
    page = method.args.args[1].arg
    parents = {
        child: node for node in ast.walk(method) for child in ast.iter_child_nodes(node)
    }
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "squeeze"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == page
    ]
    if len(calls) != 1:
        pytest.fail(
            f"{CLASS}.{METHOD} makes {len(calls)} `{page}.squeeze(...)` calls, "
            "expected exactly one. Upstream squeezes the head slot once at the "
            "bind point; if this method now normalises the page some other way, "
            "re-derive this test against it."
        )
    node = calls[0]
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            return calls[0], node
    pytest.fail(
        f"{CLASS}.{METHOD} squeezes `{page}` unconditionally. Upstream can; this "
        "layer cannot -- a 0.28-shaped [B, N=1, C] page would lose its block "
        "dimension and bind as a silently wrong two-dimensional page."
    )


def _eval(node: ast.expr, page: torch.Tensor, page_name: str):
    """Run one expression from the source with the parameter bound to `page`."""
    try:
        # The expression evaluated here comes from this repo's own source.
        return eval(
            compile(ast.Expression(body=node), str(LAYER), "eval"),
            {"torch": torch},
            {page_name: page},
        )
    except NameError as exc:
        pytest.fail(
            f"{CLASS}.{METHOD} line {node.lineno} reads {exc}, which this test "
            f"cannot supply -- it evaluates the guard with only `{page_name}` "
            "bound. Re-derive this test against the new method."
        )


def test_the_bind_point_drops_only_a_head_slot():
    method = _bind_method()
    squeeze, guard = _page_squeeze(method)
    page_name = method.args.args[1].arg

    for shape, expected, why in CASES:
        page = torch.zeros(*shape)
        taken = bool(_eval(guard.test, page, page_name))
        bound = _eval(squeeze, page, page_name) if taken else page
        assert tuple(bound.shape) == expected, (
            f"binding a {shape} page yields {tuple(bound.shape)}, expected "
            f"{expected} -- {why}. The guard on line {guard.lineno} "
            f"{'fired' if taken else 'did not fire'}."
        )
