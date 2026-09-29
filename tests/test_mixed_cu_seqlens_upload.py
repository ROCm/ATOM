# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A mixed build's `cu_seqlens_q` has exactly one writer per buffer per step.

A mixed batch's attention runs as two segments that each want their own spans:
the prefill half reads them from its private bank, the decode half from the
runner's buffer. `publish_cu_seqlens_q` writes the decode segment's into the
runner's buffer and `mixed_prefill_bank_active` the prefill segment's into the
bank, so neither is published twice in one epoch -- which the H2D ledger
refuses, and which `prepare_mixed` used to do by overwriting the runner's copy
after it had been uploaded.

These are source-level checks on purpose: they need no aiter and so run in CI,
which never executes these builders.
"""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BUILDERS = [
    "atom/model_ops/attentions/deepseek_v4_attn.py",
    "atom/model_ops/attentions/aiter_mla.py",
]
_BACKENDS = "atom/model_ops/attentions/backends.py"


def _function_body(rel: str, name: str) -> str:
    src = (_ROOT / rel).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = src.splitlines()[node.lineno - 1 : node.end_lineno]
            return "\n".join(lines)
    pytest.fail(f"{rel} has no {name}")


@pytest.mark.parametrize("rel", _BUILDERS)
def test_prepare_mixed_does_not_republish_cu_seqlens_q(rel):
    """`publish_cu_seqlens_q` already published the decode spans this epoch."""
    body = _function_body(rel, "prepare_mixed")
    assert 'cu_seqlens_q"].np[' not in body or "!=" in body, (
        f"{rel}: prepare_mixed writes cu_seqlens_q on the host; the decode "
        "segment's spans are publish_cu_seqlens_q's to write"
    )
    assert 'cu_seqlens_q"].copy_to_gpu(' not in body, (
        f"{rel}: prepare_mixed re-uploads cu_seqlens_q, a second publish of it "
        "in this epoch"
    )


@pytest.mark.parametrize("rel", _BUILDERS)
def test_prefill_bank_is_entered_through_the_shared_helper(rel):
    """No hand-rolled swap: it skips the bank's owner, groups and spans."""
    body = _function_body(rel, "prepare_mixed")
    assert "mixed_prefill_bank_active(batch)" in body, (
        f"{rel}: prepare_mixed must enter the prefill bank via "
        "mixed_prefill_bank_active(batch)"
    )
    assert "forward_vars = " not in body, (
        f"{rel}: prepare_mixed swaps forward_vars by hand, leaving the "
        "runner's publication groups bound to the live buffers"
    )


def test_publish_gives_a_mixed_batch_decode_local_spans():
    body = _function_body(_BACKENDS, "prepare_cu_seqlens_q")
    assert "is_mixed" in body
    assert "total_seqs_num_prefill" in body


def test_the_bank_publishes_the_prefill_spans_and_restores_the_runner():
    active = _function_body(_BACKENDS, "mixed_prefill_bank_active")
    assert "num_scheduled_tokens[:n_p_seqs]" in active
    assert "cu.copy_to_gpu(" in active
    assert "owner.begin()" in active and "owner.finish()" in active
    selected = _function_body(_BACKENDS, "mixed_prefill_bank_selected")
    assert "h2d_groups" in selected, "groups must swap with forward_vars"
    assert "finally:" in selected, "the swap must be restored on error"
