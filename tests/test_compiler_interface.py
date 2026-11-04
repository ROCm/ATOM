# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import operator

import pytest
import torch

from atom.utils import compilation_counter
from atom.utils.backends import split_graph
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


def test_split_graph_keeps_a_split_ops_getitems_in_its_piece():
    """A tuple from a split op is unpacked inside the split op's piece, so no
    compiled piece takes the tuple as one input (a reloaded standalone artifact
    cannot unpack it)."""

    def pair_op(x):
        return x + 1, x * 2

    pair_op.spliting_op = True
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    pair = graph.call_function(pair_op, (graph.call_function(torch.relu, (x,)),))
    first = graph.call_function(operator.getitem, (pair, 0))
    second = graph.call_function(operator.getitem, (pair, 1))
    graph.output(graph.call_function(torch.add, (first, second)))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    split_gm, items = split_graph(gm, [])
    compiled_inputs = []
    for item in items:
        if not item.is_splitting_graph:
            item.graph.register_forward_pre_hook(
                lambda _, args: compiled_inputs.extend(args)
            )
    x = torch.randn(4)
    torch.testing.assert_close(split_gm(x), gm(x), rtol=0, atol=0)
    assert [item.is_splitting_graph for item in items] == [False, True, False]
    assert compiled_inputs and all(
        isinstance(arg, torch.Tensor) for arg in compiled_inputs
    )
