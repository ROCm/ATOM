# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reaching the cudagraphs no walk can find.

A PIECEWISE capture -- `--level 3`, the default -- leaves its graphs in the
per-piece `CUDAGraphWrapper`s, which the compile backend installs off both
`_modules` and any path from the runner. Sleep has to drop them anyway, or it
frees the KV pool those graphs replay the base of. So they register, and the
release asks the registry.

The registry itself imports nothing beyond `weakref`, which is what keeps these
tests live on a CPU runner; the two tests that need the real holders skip there
rather than taking the whole file with them.
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

from atom.utils import graph_holders
from atom.utils.graph_holders import register_graph_holder, release_registered_graphs


@pytest.fixture(autouse=True)
def _own_registry(monkeypatch):
    """Give each test the registry to itself.

    It is process-wide by design -- the holders are reached by nothing else, so
    there is no runner to ask for its own -- which would otherwise make a count
    here depend on what else in the run is still alive.
    """
    monkeypatch.setattr(graph_holders, "_holders", weakref.WeakSet())


class _Holder:
    def __init__(self, graphs):
        self.graphs = graphs
        register_graph_holder(self)

    def release_graphs(self) -> int:
        dropped, self.graphs = self.graphs, 0
        return dropped


def test_every_holder_is_released_and_the_graphs_are_counted():
    holders = [_Holder(2), _Holder(3)]

    assert release_registered_graphs() == 5
    assert [h.graphs for h in holders] == [0, 0]


def test_a_holder_that_has_gone_away_is_not_asked():
    """Weak on purpose: a holder lives and dies with the compiled module it
    hangs off, and a registry is the last thing that should keep one alive."""
    _Holder(2)  # unreferenced, so collectable as soon as this returns
    gc.collect()

    assert release_registered_graphs() == 0


def test_registering_twice_asks_once():
    holder = _Holder(2)
    register_graph_holder(holder)

    assert release_registered_graphs() == 2


def test_a_holder_that_cannot_be_kept_by_identity_is_refused_at_registration():
    """`@dataclass` fills in `__eq__`, which sets `__hash__` to None. Refusing
    here beats a registry that silently merges two holders."""

    class _Unhashable:
        __hash__ = None  # type: ignore[assignment]

        def release_graphs(self) -> int:
            return 0

    with pytest.raises(TypeError):
        register_graph_holder(_Unhashable())


def _cuda_graph_module():
    """`atom/utils/cuda_graph.py` reaches aiter, which CI has no build of.

    Per-test rather than at module level, so the registry's own behaviour above
    is still checked where the release path runs without a GPU.

    Not `importorskip`: the non-GPU runner has `aiter` as a NAMESPACE package, so
    the failure is `cannot import name 'logger' from 'aiter'` -- an `ImportError`
    that is not a `ModuleNotFoundError`, which pytest 9.1 (what CI runs) treats
    as the caller's mistake and reports as a failure. `skip_if_dependency_missing`
    asks the right question: an `ImportError` naming a third-party module is an
    environment, and one naming ours is a bug that has to be seen.
    """
    try:
        from atom.utils import cuda_graph
    except ImportError as exc:
        skip_if_dependency_missing(exc, "atom/utils/cuda_graph.py imports aiter")
    return cuda_graph


def test_the_per_piece_wrapper_registers_itself_and_drops_its_entries():
    """The entry, not just the graph: `output` is a live reference into the
    graph's private pool, which is the memory the release wants back."""
    cg = _cuda_graph_module()
    wrapper = cg.CUDAGraphWrapper(
        runnable=lambda *args, **kwargs: None,
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace()),
        runtime_mode=cg.CUDAGraphMode.PIECEWISE,
    )
    descriptor = cg.BatchDescriptor(num_tokens=8)
    wrapper.concrete_cudagraph_entries[descriptor] = cg.CUDAGraphEntry(
        batch_descriptor=descriptor, cudagraph=object(), output=object()
    )
    cg._graph_pools[8] = ("a", "pool handle")

    assert release_registered_graphs() == 1
    # Reached at all only because it registered; nothing else knows it exists.
    assert wrapper.concrete_cudagraph_entries == {}
    # Recapture must make a new pool rather than reuse the handle of one whose
    # last graph has just gone away.
    assert 8 not in cg._graph_pools


def test_the_attention_core_runner_registers_itself_too():
    """AF_PIECEWISE's captured attention core writes the KV it attends, so its
    recordings hold the base of the pool the same way a decode graph does."""
    cg = _cuda_graph_module()
    runner = cg.CudagraphCaptureRunner()
    runner._graphs["a key"] = {"graph": object(), "in": {}, "out": None}
    runner._pool = "a pool handle"

    assert release_registered_graphs() == 1
    assert not runner.has_graph("a key")
    assert runner._pool is None
