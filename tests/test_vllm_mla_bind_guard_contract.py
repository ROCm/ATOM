# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA page may be normalised only when it carries exactly one head slot.

`AttentionForVllmMLA` reconciles vLLM 0.29's `[B, H, N, C]` page with the
three-dimensional `[B, N, C]` page every consumer below it was written against,
and it does so at two points, on the same condition -- one published head slot:

* `bind_kv_cache` (the bind point) squeezes the head slot away, the way
  upstream's own `MLAAttention.bind_kv_cache` does, so the rank is normalised
  once for everything downstream.
* `_as_atom_page` folds it with `view`, and is called at each of the four
  places a page crosses out of that file into a kernel written against
  `[num_blocks, block_size, entry]`.

Neither may drop the condition to match upstream's unconditional form: a
0.28-shaped `[B, N, C]` page whose `N` is 1 -- MLA's kernel block size -- would
lose its block dimension, still be a tensor, still bind, and be silently wrong.
And the two are not equally forgiving about the head-slot half of it. Relaxing
it at the bind point is inert, because `Tensor.squeeze(dim)` does nothing when
that dimension is not 1; at the fold it is not, because `view` has no such rule
and a two-head-slot page folds into twice as many block rows -- head-major, so
the second slot's rows read as block indices past the end of the first. Such a
page must cross four-dimensional and trip the kernels' own rank assertions
instead.

`tests/plugin/test_vllm_mla_bind_kv_cache.py` drives the real bind point and is
the stronger test, but importing the layer pulls in vLLM, and CI's unit-test job
both lacks vLLM and passes `--ignore=tests/plugin` -- so it cannot run where the
regression would land. This file covers the contract without importing the
layer, the way `test_kv_pool_contract` does.

It does not pattern-match the guard's syntax, which would fail any equivalent
rewrite while passing mutations that keep the shape of the code. It lifts the
guard and the normalising expression out of the source and *evaluates them*
against real tensors, so what is asserted is the decision those lines make, not
how they are spelled: `size(1) == 1`, reordered conjuncts and a nested `if` all
read the same to it.

Evaluation cannot see anything the two expressions do not do, though -- dropping
the assignment back, or mangling the page after the guard, leaves both
expressions correct. So there is a second, deliberately syntactic check that the
normalised page reaches the method's exit. It asserts only that the value flows,
never how the guard is written.

What it assumes about each method: one `if` that reads the page's rank, holding
one assignment back to the page, and a last statement that returns the page or
hands it to a call.
Rewriting past that fails with a message saying to re-derive this test -- these
overrides are load-bearing enough to be worth a human reading a rewrite of them.
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest
import torch

LAYER = (
    pathlib.Path(__file__).resolve().parent.parent
    / "atom/plugin/vllm/attention/layer_mla.py"
)
CLASS = "AttentionForVllmMLA"
FOLD_POINT = "_as_atom_page"
POINTS = ("bind_kv_cache", FOLD_POINT)

KV_LORA_RANK, QK_ROPE_HEAD_DIM = 6, 2
ENTRY = KV_LORA_RANK + QK_ROPE_HEAD_DIM

# Every shape a normalisation point can be handed, and the page it must produce.
# Both points owe the same answers -- that is what "the same condition" means
# above, and asserting it here is what keeps them from drifting apart even
# though one squeezes and the other folds.
CASES = (
    ((4, 1, 16, ENTRY), (4, 16, ENTRY), "0.29's [B, H=1, N, C]: the head slot goes"),
    ((4, 2, 16, ENTRY), (4, 2, 16, ENTRY), "two head slots: fall through, fail loudly"),
    ((4, 1, ENTRY), (4, 1, ENTRY), "0.28's [B, N=1, C]: N is a block row, not a head"),
    ((4, 16, ENTRY), (4, 16, ENTRY), "0.28's [B, N, C]: already what consumers want"),
    # Nothing produces these three. They are here because merging four crossings
    # into one helper is where the admission set gets chosen, and the union of
    # the four is wider than the strictest of them: `dim() != 4 or shape[1] == 1`
    # answers the four rows above identically and still folds all three of these
    # into something that satisfies the kernels' checks instead of tripping them.
    ((4, ENTRY), (4, ENTRY), "rank-2: not a page at all, must fall through"),
    (
        (4, 1, 1, 16, ENTRY),
        (4, 1, 1, 16, ENTRY),
        "rank-5: likewise, not this rule's business",
    ),
    (
        (4, 16, 2 * ENTRY),
        (4, 16, 2 * ENTRY),
        (
            "wrong entry width: folding it would make [B, 2N, C] out of "
            "[B, N, 2C] -- the same silent head-major concatenation, one axis "
            "over -- instead of tripping aiter's own size(2) check"
        ),
    ),
)


def _method(name: str) -> ast.FunctionDef:
    tree = ast.parse(LAYER.read_text(), filename=str(LAYER))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if cls.name != CLASS:
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == name:
                return fn
    pytest.fail(f"{LAYER.name} no longer defines {CLASS}.{name}")


def _reads_rank(test: ast.expr, page: str) -> bool:
    """Does this `if` test ask how many dimensions the page has?

    Only used to *locate* the rank guard among a method's `if`s -- what the
    guard decides is then judged by evaluating it, not by reading its shape.
    """
    for node in ast.walk(test):
        receiver = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value if node.func.attr == "dim" else None
        elif isinstance(node, ast.Attribute) and node.attr == "ndim":
            receiver = node.value
        if isinstance(receiver, ast.Name) and receiver.id == page:
            return True
    return False


def _rank_guard(method: ast.FunctionDef, page: str) -> tuple[ast.If, ast.expr]:
    """The `if` that decides on rank, and the expression it normalises with."""
    guards = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If) and _reads_rank(node.test, page)
    ]
    if len(guards) != 1:
        pytest.fail(
            f"{CLASS}.{method.name} has {len(guards)} `if`s that read the page's "
            f"rank, expected exactly one. Re-derive this test against it."
        )
    guard = guards[0]
    rebinds = [
        stmt.value
        for stmt in guard.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == page
    ]
    if len(rebinds) != 1:
        pytest.fail(
            f"{CLASS}.{method.name} makes {len(rebinds)} assignments back to "
            f"`{page}` under its rank guard, expected exactly one. A normalising "
            "call whose result is dropped leaves the page unchanged, silently: "
            "neither `squeeze` nor `view` writes in place."
        )
    return guard, rebinds[0]


def _eval(node: ast.expr, page: torch.Tensor, page_name: str, where: str):
    """Run one expression from the source, with only the page and a stub self."""
    stub = types.SimpleNamespace(
        kv_lora_rank=KV_LORA_RANK, qk_rope_head_dim=QK_ROPE_HEAD_DIM
    )
    try:
        # The expression evaluated here comes from this repo's own source.
        return eval(
            compile(ast.Expression(body=node), str(LAYER), "eval"),
            {"torch": torch},
            {page_name: page, "self": stub},
        )
    except (NameError, AttributeError) as exc:
        pytest.fail(
            f"{where} line {node.lineno} reads {exc}, which this test cannot "
            f"supply -- it evaluates with only `{page_name}` and a stub `self` "
            "bound. Re-derive this test against the new method."
        )


@pytest.mark.parametrize("name", POINTS)
def test_the_page_is_normalised_only_for_a_single_head_slot(name: str):
    method = _method(name)
    page_name = method.args.args[1].arg
    guard, action = _rank_guard(method, page_name)
    where = f"{CLASS}.{name}"

    for shape, expected, why in CASES:
        page = torch.zeros(*shape)
        taken = bool(_eval(guard.test, page, page_name, where))
        out = _eval(action, page, page_name, where) if taken else page
        assert tuple(out.shape) == expected, (
            f"{where} turns a {shape} page into {tuple(out.shape)}, expected "
            f"{expected} -- {why}. The guard on line {guard.lineno} "
            f"{'fired' if taken else 'did not fire'}."
        )


@pytest.mark.parametrize("name", POINTS)
def test_the_normalised_page_reaches_the_exit(name: str):
    """Evaluating the guard says nothing about where its result goes."""
    method = _method(name)
    page_name = method.args.args[1].arg
    _rank_guard(method, page_name)  # also asserts the result is assigned back

    last = method.body[-1]
    bare = [
        node
        for node in ast.walk(last)
        if isinstance(node, ast.Call)
        for arg in node.args
        if isinstance(arg, ast.Name) and arg.id == page_name
    ]
    returned = isinstance(last, ast.Return) and isinstance(last.value, ast.Name)
    assert bare or (returned and last.value.id == page_name), (
        f"{CLASS}.{name} neither returns `{page_name}` nor hands it to a call "
        "in its last statement. The normalisation is only worth anything if "
        "the normalised page is what leaves the method."
    )


def test_no_page_is_folded_outside_the_one_place_that_guards_it():
    """The rule is only a rule if every crossing goes through it.

    Four call sites in `layer_mla.py` hand a page to a kernel written against
    `[num_blocks, block_size, entry]`, and before `de5ce82a1` each folded it
    inline and unconditionally. That is why the sparse cells stayed green
    without the bind-point squeeze -- for `H == 1` an unconditional fold is
    accidentally correct. A fifth site written the old way would reintroduce
    the silent case without touching anything the tests above look at.
    """
    tree = ast.parse(LAYER.read_text(), filename=str(LAYER))
    inside = {
        node
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name == FOLD_POINT
        for node in ast.walk(fn)
    }
    stray = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "view"
        and len(node.args) == 3
        and isinstance(node.args[1], ast.UnaryOp)
        and isinstance(node.args[1].op, ast.USub)
        and node not in inside
    ]
    assert not stray, (
        f"{LAYER.name} folds a page with `view(..., -1, ...)` outside "
        f"{CLASS}.{FOLD_POINT} at line(s) {stray}. Every crossing into a "
        "three-dimensional kernel must go through the one guarded fold, or a "
        "multi-head-slot page is silently concatenated head-major at that "
        "site while the guarded ones correctly pass it through."
    )
