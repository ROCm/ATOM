# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A mixed build's host-side metadata writes must reach the device.

`prepare_decode` takes `cu_seqlens_q`'s device side as a bare `.gpu[...]` view,
on the strength of a writer having already uploaded it. `prepare_mixed`
overwrites the leading decode rows with decode-local spans -- so if it writes
only `.np`, every host-side assert still passes while the kernels segment `q`
by the *full-batch* cumsum and index thousands of rows into an n-row tensor.

These are source-level checks on purpose: they need no aiter and so run in CI,
which never executes these builders. They encode the pairing rule
("host write implies upload, at the same width") rather than any one bug.
"""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BUILDERS = [
    "atom/model_ops/attentions/deepseek_v4_attn.py",
    "atom/model_ops/attentions/aiter_mla.py",
]


def _prepare_mixed_body(rel: str) -> tuple[str, ast.FunctionDef]:
    src = (_ROOT / rel).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_mixed":
            lines = src.splitlines()[node.lineno - 1 : node.end_lineno]
            return "\n".join(lines), node
    pytest.fail(f"{rel} has no prepare_mixed")


@pytest.mark.parametrize("rel", _BUILDERS)
def test_cu_seqlens_q_host_write_is_uploaded(rel):
    """Whoever rewrites cu_seqlens_q on the host uploads it too."""
    body, _ = _prepare_mixed_body(rel)
    if 'cu_seqlens_q"].np[' not in body and "_cu.np[" not in body:
        pytest.skip(f"{rel}'s prepare_mixed does not rewrite cu_seqlens_q")
    uploaded = "_cu.copy_to_gpu(" in body or 'cu_seqlens_q"].copy_to_gpu(' in body
    assert uploaded, (
        f"{rel}: prepare_mixed rewrites cu_seqlens_q on the host but never "
        "uploads it; prepare_decode reads the device side as a bare view, so "
        "the kernels would keep segmenting by the full-batch cumsum"
    )


@pytest.mark.parametrize("rel", _BUILDERS)
def test_prefill_bank_is_swapped_through_the_shared_helper(rel):
    """No hand-rolled swap: it is where the missing cu_seqlens_q sync hid.

    `mixed_prefill_bank_active` binds the swap to the per-step sync that has
    to accompany it. A builder that assigns `forward_vars` itself has, by
    construction, skipped that sync.
    """
    body, _ = _prepare_mixed_body(rel)
    assert "mixed_prefill_bank_active()" in body, (
        f"{rel}: prepare_mixed must enter the prefill bank via "
        "mixed_prefill_bank_active()"
    )
    assert "forward_vars = pf_bank" not in body, (
        f"{rel}: prepare_mixed swaps forward_vars by hand; that is the form "
        "that silently drops the cu_seqlens_q sync"
    )


def test_the_shared_helper_syncs_cu_seqlens_q_both_sides():
    """Host AND device -- prefill reads one and the cross-check the other."""
    src = (_ROOT / "atom/model_ops/attentions/backends.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "mixed_prefill_bank_active"
        ):
            body = "\n".join(src.splitlines()[node.lineno - 1 : node.end_lineno])
            assert 'bank["cu_seqlens_q"].cpu.copy_(' in body
            assert 'bank["cu_seqlens_q"].gpu.copy_(' in body
            assert "finally:" in body, "the swap must be restored on error"
            return
    pytest.fail("backends.py has no mixed_prefill_bank_active")
