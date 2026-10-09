# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`PAGES_PER_SPARSE_BLOCK` is defined twice and nothing enforced it.

`sparse_attn.py` derives it (`SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE`);
`index_topk.py` repeats it as a literal `8`, because `sparse_attn` imports
`index_topk` and importing back would close a cycle. The duplication was held
together by a comment saying "must match", which is not a check: the two are the
source and destination factors of the same block-table arithmetic
(`logical_id * block_page_stride` into `slot * pages_per_block`), so a silent
divergence emits a block table wrong by the difference, on every M3 sparse path,
with no exception anywhere.

The same goes for the `block_page_stride` defaults. ATOM's native engine and the
sglang plugin never pass that argument -- they allocate separate K and V caches
whose blocks are back to back -- so the default is the whole of their contract.
Only `atom/plugin/vllm/attention/minimax_m3_attnetion.py` overrides it, with its
own `BLOCK_PAGE_STRIDE`, because only that path packs K and V into one block.

Both facts are read out of the sources with `ast` rather than by importing the
modules, the way `tests/test_vllm_mla_bind_guard_contract.py` does: these
modules import `triton`, which CI's non-GPU unit job does not have, and a test
that skips there is no protection at all -- the non-GPU job is exactly where
this regression would otherwise land unnoticed.
"""

from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPARSE_ATTN = _ROOT / "atom" / "model_ops" / "minimax_m3" / "sparse_attn.py"
_INDEX_TOPK = _ROOT / "atom" / "model_ops" / "minimax_m3" / "index_topk.py"
_PLUGIN = _ROOT / "atom" / "plugin" / "vllm" / "attention" / "minimax_m3_attnetion.py"


def _as_int(node: ast.expr, path: pathlib.Path, known: dict[str, int]) -> int | None:
    """The node's integer value, or None when it is not an integer constant.

    A module's top level holds dataclasses, tuples and calls as well; those are
    not errors to skip over, they are the expected majority, so this reports
    "not an integer" as a value rather than as an exception.
    """
    expression = compile(ast.Expression(node), str(path), "eval")
    # The source is this repository's own; the namespace has no builtins and
    # only the integers resolved so far in this module.
    try:
        value = eval(expression, {"__builtins__": {}}, dict(known))
    except Exception:  # noqa: BLE001 - any non-constant expression
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _module_constants(
    path: pathlib.Path, seed: dict[str, int] | None = None
) -> dict[str, int]:
    """Evaluate the module's top-level integer assignments, without importing.

    `seed` stands in for names the module imports rather than assigns, which is
    how the plugin gets `PAGES_PER_SPARSE_BLOCK`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, int] = dict(seed or {})
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _as_int(node.value, path, values)
        if value is not None:
            values[target.id] = value
    return values


def _stride_defaults(path: pathlib.Path) -> dict[str, str]:
    """Map each function taking `block_page_stride` to its default, as source."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args
        names = [a.arg for a in args.args]
        if "block_page_stride" not in names:
            continue
        offset = len(names) - len(args.defaults)
        index = names.index("block_page_stride") - offset
        if index < 0:
            continue
        out[node.name] = ast.unparse(args.defaults[index])
    return out


def test_pages_per_sparse_block_agrees_across_its_two_definitions():
    sparse = _module_constants(_SPARSE_ATTN)
    topk = _module_constants(_INDEX_TOPK)
    assert topk["PAGES_PER_SPARSE_BLOCK"] == sparse["PAGES_PER_SPARSE_BLOCK"], (
        "index_topk repeats sparse_attn.PAGES_PER_SPARSE_BLOCK as a literal to "
        "avoid a circular import; the two have diverged"
    )


def test_pages_per_sparse_block_is_the_derived_value():
    """Pin the derivation too, so the literal cannot win by both being wrong."""
    sparse = _module_constants(_SPARSE_ATTN)
    assert sparse["PAGES_PER_SPARSE_BLOCK"] == (
        sparse["SPARSE_BLOCK_SIZE"] // sparse["ASM_PAGE_SIZE"]
    )


def test_kernel_stride_defaults_are_the_back_to_back_value():
    """The default is the native and sglang contract; it must not move."""
    defaults = _stride_defaults(_SPARSE_ATTN) | _stride_defaults(_INDEX_TOPK)
    assert defaults, "no entry point takes block_page_stride any more"
    for name, default in defaults.items():
        assert default == "PAGES_PER_SPARSE_BLOCK", (
            f"{name}'s block_page_stride default moved to {default!r}; ATOM "
            "native and the sglang plugin rely on the back-to-back value"
        )


def test_packed_plane_stride_lives_with_the_cache_it_describes():
    """`BLOCK_PAGE_STRIDE` belongs to the vLLM page view, not to the kernels."""
    sparse = _module_constants(_SPARSE_ATTN)
    plugin = _module_constants(_PLUGIN, seed=sparse)
    assert "BLOCK_PAGE_STRIDE" not in _SPARSE_ATTN.read_text(encoding="utf-8"), (
        "the packed-plane stride is a property of the vLLM plugin's cache; "
        "shared kernel code should not name it"
    )
    assert plugin["BLOCK_PAGE_STRIDE"] == 2 * sparse["PAGES_PER_SPARSE_BLOCK"]
