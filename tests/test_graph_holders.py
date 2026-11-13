# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reaching the cudagraphs no walk can find.

A module's token graphs sit in the module's own `__dict__`, wherever in the
model it is, off any path from the runner. Sleep has to drop them anyway, or
it frees the KV pool those graphs replay the base of. So they register, and
the release asks the registry.
"""

import gc
import weakref

import pytest

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
