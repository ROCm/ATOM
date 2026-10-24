# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The BLASST threshold is wired into the prefill call, and only that call.

``test_blasst_threshold.py`` covers ``resolve_threshold`` in isolation.

These tests read ``attention_mha.py`` as a SOURCE TREE rather than importing it.
That module pulls in aiter at import time, and per ``tests/conftest.py`` CI has
no aiter, so an import-based test here would be skipped in CI and only ever run
on a developer's machine -- coverage-shaped, never executed. Parsing the file
needs neither aiter nor a GPU, so these run everywhere.

The trade: this pins the call SITE, not runtime behaviour. It catches the
regression that is actually likely (the keyword quietly disappearing in a
refactor); it cannot catch a wrong value computed at runtime.
"""

import ast
from pathlib import Path

import pytest

ATTENTION_MHA = (
    Path(__file__).resolve().parents[1] / "atom" / "model_ops" / "attention_mha.py"
)
PREFILL = "prefill_attention_triton"
# Upstream renamed this from paged_attention_triton. The name is asserted to
# exist below: a rename must fail loudly, because a stale name here makes
# test_decode_omits_the_threshold iterate an empty list and pass vacuously --
# the exact failure this file exists to prevent.
DECODE = "paged_attention_unified"


def _callers_of(func_name):
    """Map {enclosing function -> [call nodes]} for calls to ``func_name``."""
    tree = ast.parse(ATTENTION_MHA.read_text())
    found = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == func_name
        ]
        if calls:
            found[fn.name] = calls
    return found


@pytest.fixture(scope="module")
def unified_attention_calls():
    calls = _callers_of("unified_attention")
    assert calls, (
        f"no call to unified_attention found in {ATTENTION_MHA.name}. The module "
        "was renamed or restructured; these guards are now vacuous and must be "
        "re-pointed rather than deleted."
    )
    return calls


def _kwargs(call):
    return {k.arg for k in call.keywords}


class TestPrefillIsWired:
    def test_prefill_passes_the_threshold(self, unified_attention_calls):
        assert PREFILL in unified_attention_calls, f"{PREFILL} no longer calls unified_attention"
        for call in unified_attention_calls[PREFILL]:
            assert "block_skip_threshold" in _kwargs(call), (
                f"{PREFILL} calls unified_attention without block_skip_threshold. "
                "BLASST is now dead code: resolve_threshold still returns a value "
                "and nothing consumes it."
            )

    def test_threshold_comes_from_the_attended_kv_length(self):
        """resolve_threshold must be fed the KV length, not a constant.

        The fit is ``alpha * exp(beta * sparsity) / seqlen``, so a fixed seqlen
        would skip an ever-larger fraction as context grows -- the failure is
        silent accuracy loss on long prompts, not a crash.
        """
        tree = ast.parse(ATTENTION_MHA.read_text())
        resolve_calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "resolve_threshold"
        ]
        assert resolve_calls, "nothing calls blasst.resolve_threshold any more"
        for call in resolve_calls:
            assert call.args, "resolve_threshold called with no sequence length"
            arg = ast.unparse(call.args[0])
            assert "max_seqlen_k" in arg, (
                f"resolve_threshold is fed {arg!r} rather than the attended KV "
                "length (max_seqlen_k); the 1/seqlen term is then wrong."
            )


class TestDecodeIsNotWired:
    def test_decode_omits_the_threshold(self, unified_attention_calls):
        """Decode has one query row and no prior running max, so nothing can be
        elided; passing a threshold would only add the check's cost."""
        assert DECODE in unified_attention_calls, (
            f"{DECODE} no longer calls unified_attention -- it was probably "
            "renamed. Re-point DECODE rather than deleting this test, which "
            "would otherwise pass over an empty list."
        )
        for call in unified_attention_calls[DECODE]:
            assert "block_skip_threshold" not in _kwargs(call), (
                f"{DECODE} now passes block_skip_threshold. BLASST is prefill-only."
            )

    def test_exactly_one_call_site_is_wired(self, unified_attention_calls):
        """A new call site that forwards the threshold is a deliberate decision,
        not something to acquire by copy-paste."""
        wired = sorted(
            name
            for name, calls in unified_attention_calls.items()
            if any("block_skip_threshold" in _kwargs(c) for c in calls)
        )
        assert wired == [PREFILL], (
            f"expected only {PREFILL} to pass block_skip_threshold, found {wired}"
        )
