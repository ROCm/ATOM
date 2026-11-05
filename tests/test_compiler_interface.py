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
