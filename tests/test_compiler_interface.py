# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from atom.utils import compilation_counter
from atom.utils.compiler_inferface import _save_standalone_compiled_graph


def test_non_saveable_torch_213_artifact_is_skipped():
    class NonSaveableArtifact:
        def is_saveable(self):
            return False

        def save(self, **kwargs):
            pytest.fail("save must not run when is_saveable() is false")

    assert (
        _save_standalone_compiled_graph(
            NonSaveableArtifact(), "/tmp/not-used", "subgraph"
        )
        is None
    )


def test_legacy_no_aot_runtime_error_is_skipped():
    class LegacyArtifact:
        def save(self, **kwargs):
            raise RuntimeError(
                "CompiledArtifact.save failed to save due to no "
                "aot_autograd artifacts"
            )

    assert (
        _save_standalone_compiled_graph(LegacyArtifact(), "/tmp/not-used", "subgraph")
        is None
    )


def test_unexpected_save_runtime_error_is_not_swallowed():
    class BrokenArtifact:
        def save(self, **kwargs):
            raise RuntimeError("permission denied")

    with pytest.raises(RuntimeError, match="permission denied"):
        _save_standalone_compiled_graph(BrokenArtifact(), "/tmp/not-used", "subgraph")


def test_saveable_artifact_returns_cache_handle(monkeypatch):
    saved = {}

    class SaveableArtifact:
        def is_saveable(self):
            return True

        def save(self, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr(compilation_counter, "num_compiled_artifacts_saved", 0)

    handle = _save_standalone_compiled_graph(
        SaveableArtifact(), "/tmp/artifact", "subgraph"
    )

    assert handle == ("subgraph", "/tmp/artifact")
    assert saved == {"path": "/tmp/artifact", "format": "unpacked"}
    assert compilation_counter.num_compiled_artifacts_saved == 1


class _CountingArg:
    """An IR stand-in that counts how often it is printed."""

    def __init__(self):
        self.printed = 0

    def __str__(self):
        self.printed += 1
        return "IR"


def _operator_issue():
    exc = pytest.importorskip("torch._inductor.exc")
    from atom.utils import is_torch_equal_or_newer

    if is_torch_equal_or_newer("2.13.0.dev"):
        pytest.skip("torch >= 2.13 defers this log line itself")
    return exc


def test_filtered_fallback_log_does_not_print_the_ir():
    import logging

    exc = _operator_issue()
    arg = _CountingArg()
    log = logging.getLogger("atom.test.operator_str")
    log.setLevel(logging.WARNING)
    # The pre-2.13 GraphLowering.call_function call shape.
    log.info(
        "Creating implicit fallback for:\n%s",
        exc.MissingOperatorWithoutDecomp.operator_str("op", [arg], {}),
    )
    assert arg.printed == 0


def test_operator_str_text_is_unchanged_when_printed():
    exc = _operator_issue()
    deferred = exc.OperatorIssue.operator_str("op", [1, "x"], {"k": 2})
    assert (
        str(deferred) == "  target: op\n  args[0]: 1\n  args[1]: x\n  kwargs: {'k': 2}"
    )


def test_lowering_exception_message_keeps_the_operator_text():
    exc = _operator_issue()
    err = exc.LoweringException(ValueError("bad"), "op", [_CountingArg()], {})
    assert str(err) == "ValueError: bad\n  target: op\n  args[0]: IR"
