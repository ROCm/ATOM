# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from atom.utils import compilation_counter, compiler_inferface
from atom.utils.compiler_inferface import (
    InductorStandaloneAdaptor,
    _save_standalone_compiled_graph,
)


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


def test_standalone_cpu_artifact_roundtrip_with_tuple_input(tmp_path, monkeypatch):
    """Reload retains tuple-input flattening and input-mutation argument order.

    The mutation is intentional: wrong flat argument indices after reloading
    used to pass a scalar to increment_version in the last piecewise graph.
    This compiles a real CPU kernel, saves it, and uses ATOM's real loader.
    """
    standalone = pytest.importorskip("torch._inductor.standalone_compile")
    # The adaptation must key on the artifact kind: torch < 2.10 has the same
    # flat reload under another class name.
    monkeypatch.setattr(
        compiler_inferface,
        "is_torch_equal_or_newer",
        lambda v, _newer=compiler_inferface.is_torch_equal_or_newer: v != "2.10"
        and _newer(v),
    )

    class Nested(torch.nn.Module):
        def forward(self, pair, n, residual):
            torch.ops.aten.add_.Tensor(
                residual,
                torch.ops.aten.add.Tensor(
                    torch.ops.aten.mul.Tensor(pair[0], pair[1]), n
                ),
            )
            return residual

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    graph = torch.fx.symbolic_trace(Nested())

    def inputs():
        x = torch.arange(8, dtype=torch.float32, device="cpu")
        scale = torch.full((8,), 2.0, device="cpu")
        residual = torch.ones(8, device="cpu")
        return (x, scale), 3, residual

    example_inputs = inputs()
    with torch._functorch.config.patch(enable_autograd_cache=True):
        fresh = standalone.standalone_compile(
            graph,
            example_inputs,
            dynamic_shapes="from_example_inputs",
            options={"config_patches": {"compile_threads": 1, "fx_graph_cache": True}},
        )
    if isinstance(fresh, getattr(standalone, "AOTCompiledArtifact", ())):
        pytest.skip("Torch selected the AOT standalone artifact format")
    path = str(tmp_path / "artifact")
    handle = _save_standalone_compiled_graph(fresh, path, "roundtrip")
    assert handle is not None, "The CPU probe must produce a serializable artifact"
    loaded = InductorStandaloneAdaptor().load(handle, graph, list(example_inputs), 0)
    expected = torch.arange(8, dtype=torch.float32) * 2 + 4
    fresh_args, loaded_args = inputs(), inputs()
    torch.testing.assert_close(fresh(*fresh_args), expected, rtol=0, atol=0)
    torch.testing.assert_close(loaded(*loaded_args), expected, rtol=0, atol=0)
    torch.testing.assert_close(loaded_args[-1], fresh_args[-1], rtol=0, atol=0)
