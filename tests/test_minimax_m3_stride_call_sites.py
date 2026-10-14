# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every caller of the stride-taking helpers must still bind.

`block_page_stride` was inserted into `_emit_sparse_block_table_row` *between*
`pages_per_block` and `NUM_KV_HEADS`, and appended to `_launch_select` as a
required parameter. Both are internal helpers called positionally, so adding a
parameter does not raise at import: a call that is now one argument short binds
every later argument to the wrong name. In `_emit_sparse_block_table_row` the
later arguments are `tl.constexpr`, so the damage is a silently wrong stride and
head count rather than a type error.

Two call sites were missed when the parameter was added, both on the opt-in M3
indexer context-parallel path: `indexer_candidate_exchange.py` and
`indexer_context_parallel.py`. The test that covers that path
(`tests/model_ops/test_indexer_cp_parity.py`) needs a GPU, so CI never saw it.

Argument binding does not need a GPU. This checks it with
`inspect.Signature.bind`, driven from the call sites found in the source, so it
runs in the non-GPU job where the regression actually escaped.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_M3 = _ROOT / "atom" / "model_ops" / "minimax_m3"

# Helpers that gained the parameter, and the modules that call them.
_TARGETS = {
    "_emit_sparse_block_table_row": "index_topk.py",
    "_launch_select": "index_topk.py",
}


def _signature_params(module: pathlib.Path, func: str) -> inspect.Signature:
    """Build a Signature from the source, without importing (needs triton)."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            args = node.args
            names = [a.arg for a in args.args]
            offset = len(names) - len(args.defaults)
            params = [
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    **(
                        {}
                        if i < offset
                        else {"default": ast.unparse(args.defaults[i - offset])}
                    ),
                )
                for i, name in enumerate(names)
            ]
            return inspect.Signature(params)
    raise AssertionError(f"{func} not found in {module}")


def _call_sites(func: str) -> list[tuple[pathlib.Path, ast.Call]]:
    out: list[tuple[pathlib.Path, ast.Call]] = []
    for path in sorted(_M3.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == func:
                out.append((path, node))
    return out


@pytest.mark.parametrize("func", sorted(_TARGETS))
def test_every_call_site_binds(func: str):
    signature = _signature_params(_M3 / _TARGETS[func], func)
    sites = _call_sites(func)
    assert sites, f"no call site found for {func}; this test has gone blind"
    for path, call in sites:
        positional = ["<arg>"] * len(call.args)
        keywords = {kw.arg: "<kw>" for kw in call.keywords if kw.arg}
        try:
            signature.bind(*positional, **keywords)
        except TypeError as exc:
            pytest.fail(
                f"{path.name}:{call.lineno} does not bind {func}{signature}: {exc}"
            )


def test_stride_arguments_are_named_at_every_call_site():
    """Positional is not enough: a shifted argument still binds.

    `_emit_sparse_block_table_row` takes `pages_per_block`,
    `block_page_stride`, `NUM_KV_HEADS` and `BLOCK_SIZE_T` in a row. Drop one
    from a positional call and the rest shift up by one and bind silently, which
    is the failure this file exists for. Requiring the stride to be passed by
    keyword makes that shift impossible to express.
    """
    for path, call in _call_sites("_emit_sparse_block_table_row"):
        named = {kw.arg for kw in call.keywords if kw.arg}
        assert "block_page_stride" in named, (
            f"{path.name}:{call.lineno} passes block_page_stride positionally; "
            "pass it by keyword so a missing argument cannot shift the "
            "constexprs after it"
        )
