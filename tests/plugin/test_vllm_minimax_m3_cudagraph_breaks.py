# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3's sparse attention must break the breakable cudagraph.

vLLM auto-enables ``VLLM_USE_BREAKABLE_CUDAGRAPH`` for this architecture, so
plugin mode compiles nothing and splits nothing: one stream capture drives the
whole forward, and only ops carrying ``@eager_break_during_capture`` end a
segment. Drop the decorator and 57 of M3's 60 attention layers get captured
wholesale, freezing this batch's token counts, ``block_table``, ``seq_lens``
and topk indices into every replay.

Nothing downstream notices. A cold prompt prefills past the largest captured
size and runs eagerly, so it answers correctly; reuse a prefix and the short
remainder lands inside a captured size and answers fluently from the wrong KV.
Catching that costs a two-run accuracy sweep on four GPUs, which is why this
guard is here instead.

The scan reads the source rather than importing it: the module imports ``aiter``
at module scope, so an importing test would skip on every CI runner -- exactly
where the guard is supposed to fire.
"""

import ast
from pathlib import Path

M3_ATTENTION = (
    Path(__file__).resolve().parents[2]
    / "atom"
    / "plugin"
    / "vllm"
    / "attention"
    / "minimax_m3_attnetion.py"
)

BREAK_DECORATOR = "eager_break_during_capture"
SPARSE_OP = "minimax_m3_sparse_attention"


def _module() -> ast.Module:
    return ast.parse(M3_ATTENTION.read_text())


def _decorator_names(node: ast.FunctionDef) -> set[str]:
    names = set()
    for dec in node.decorator_list:
        target = getattr(dec, "func", dec)
        name = getattr(target, "id", None) or getattr(target, "attr", None)
        if name:
            names.add(name)
    return names


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_sparse_attention_run_is_an_eager_break():
    tree = _module()
    run = _find_function(tree, "_sparse_attn_run")
    assert run is not None, (
        "MiniMaxM3SparseAttentionForVllm._sparse_attn_run is gone -- whatever "
        "replaced it still has to carry the eager break, or M3's 57 sparse "
        "layers go back inside the captured graph"
    )
    assert BREAK_DECORATOR in _decorator_names(run), (
        f"_sparse_attn_run lost @{BREAK_DECORATOR}; sparse attention would be "
        "captured with this batch's metadata frozen into every replay"
    )


def test_the_break_writes_into_a_caller_owned_buffer():
    """The decorator replays the Python kernel, so its output must be passed in.

    A tensor allocated inside lands at a new address on every replay while the
    captured segments that consume it still read the address recorded at
    capture -- the failure ``eager_break_during_capture`` documents by name.
    """
    tree = _module()

    op = _find_function(tree, SPARSE_OP)
    assert op is not None, f"{SPARSE_OP} is gone"
    assert "output" in {a.arg for a in op.args.args}, (
        f"{SPARSE_OP} no longer takes `output` from its caller; allocating it "
        "inside breaks every replay after the first"
    )

    mark = next(
        (
            dec
            for dec in op.decorator_list
            if getattr(getattr(dec, "func", dec), "id", None) == "mark_spliting_op"
        ),
        None,
    )
    assert mark is not None, f"{SPARSE_OP} is no longer registered as a custom op"
    mutates = next((kw.value for kw in mark.keywords if kw.arg == "mutates_args"), None)
    assert isinstance(mutates, ast.List), "mutates_args must be a literal list"
    assert "output" in {
        el.value for el in mutates.elts if isinstance(el, ast.Constant)
    }, (
        f"{SPARSE_OP} must declare `output` in mutates_args, or torch records "
        "it as a pure return and the schema stops marking the write"
    )

    run = _find_function(tree, "_sparse_attn_run")
    assert run is not None and "output" in {
        a.arg for a in run.args.args
    }, "the eager break must receive the buffer rather than allocate one"
