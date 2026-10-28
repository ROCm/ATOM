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
    """Literal keyword names on a call, plus ``**name`` unpacks as ``**name``."""
    names = set()
    for k in call.keywords:
        names.add(k.arg if k.arg is not None else "**" + ast.unparse(k.value))
    return names


def _forwards_blasst(call):
    """True if the call forwards the threshold, directly or via ``**blasst_kwargs``.

    The threshold is forwarded conditionally (see
    ``test_prefill_forwards_the_threshold_only_when_enabled``), so at the call
    site it appears as ``**blasst_kwargs`` rather than a literal keyword. Accept
    either form so the guard survives a refactor back to an unconditional pass.
    """
    kw = _kwargs(call)
    return "block_skip_threshold" in kw or "**blasst_kwargs" in kw


def _prefill_func(tree):
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == PREFILL:
            return fn
    return None


class TestPrefillIsWired:
    def test_prefill_passes_the_threshold(self, unified_attention_calls):
        assert (
            PREFILL in unified_attention_calls
        ), f"{PREFILL} no longer calls unified_attention"
        for call in unified_attention_calls[PREFILL]:
            assert _forwards_blasst(call), (
                f"{PREFILL} calls unified_attention without forwarding "
                "block_skip_threshold. BLASST is now dead code: resolve_threshold "
                "still returns a value and nothing consumes it."
            )

    def test_prefill_forwards_the_threshold_only_when_enabled(self):
        """The keyword must be omitted when skipping is off.

        block_skip_threshold is from ROCm/aiter#5868, which is unmerged, so stock
        upstream AITER -- the default Docker dependency -- does not accept it.
        Passing it unconditionally raises TypeError on the first prefill of every
        ATOM_USE_UNIFIED_ATTN=1 deployment, including ones that never enabled
        BLASST. So the forwarding has to sit behind a ``threshold > 0`` guard; a
        literal ``block_skip_threshold=`` keyword straight on the call is the
        regression this test exists to catch.
        """
        tree = ast.parse(ATTENTION_MHA.read_text())
        fn = _prefill_func(tree)
        assert fn is not None, f"{PREFILL} no longer calls unified_attention"

        # It is not passed as an unconditional literal keyword on the call.
        for call in (
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "unified_attention"
        ):
            assert "block_skip_threshold" not in {k.arg for k in call.keywords}, (
                f"{PREFILL} passes block_skip_threshold as an unconditional keyword. "
                "Against stock AITER (aiter#5868 unmerged) this raises TypeError for "
                "every unified-attn deployment. Forward it behind a threshold>0 guard."
            )

        # There is a guard comparing the threshold against zero.
        guards = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.If)
            and any(
                isinstance(c, ast.Name) and c.id == "block_skip_threshold"
                for c in ast.walk(n.test)
            )
            and any(isinstance(c, ast.Compare) for c in ast.walk(n.test))
        ]
        assert guards, (
            f"{PREFILL} forwards block_skip_threshold but nothing guards it against "
            "0.0. The disabled path must omit the keyword, not pass 0.0."
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
            assert not _forwards_blasst(
                call
            ), f"{DECODE} now forwards block_skip_threshold. BLASST is prefill-only."

    def test_exactly_one_call_site_is_wired(self, unified_attention_calls):
        """A new call site that forwards the threshold is a deliberate decision,
        not something to acquire by copy-paste."""
        wired = sorted(
            name
            for name, calls in unified_attention_calls.items()
            if any(_forwards_blasst(c) for c in calls)
        )
        assert wired == [
            PREFILL
        ], f"expected only {PREFILL} to forward block_skip_threshold, found {wired}"
