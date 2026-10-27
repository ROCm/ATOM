# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Who is holding a captured cudagraph, for whoever frees what it captured.

A graph replays the addresses it recorded, so releasing the weights or the KV
pool invalidates every graph that captured them -- see
``atom/rollout/memory_manager.py``, which does the releasing. Finding those
graphs is the problem this module exists for.

``runner.req_graph`` is the whole-forward store. A module's token graphs
(``atom/utils/region_graph.py``) are held by the module itself, in its
``__dict__``, wherever in the model it sits -- a draft model's, a compiled
model's -- so no walk from the runner finds them all. So every `TokenGraph`
registers itself here, a `ReqGraph` (``runner.req_graph``, each draft pass's)
included, and the releaser asks this module last: it walks the stores it can
reach first (``runner.req_graph``, ``Drafter.draft_graphs``), where the reader
can see what is being cleared, and this registry drops whatever no walk found.

Nothing here imports the compile stack, torch or aiter: the release path is
exercised on CPU, and a registry is names and lifetimes only.
"""

import weakref
from typing import Any

# Weak: a holder lives and dies with the compiled module it hangs off, and a
# registry is the last thing that should be what keeps one alive.
_holders: "weakref.WeakSet[Any]" = weakref.WeakSet()


def register_graph_holder(holder: Any) -> None:
    """Declare *holder* as owning cudagraphs, for `release_registered_graphs`.

    It must implement ``release_graphs() -> int``, dropping every graph it holds
    and returning how many were dropped, and be hashable by identity -- an
    `eq=True` dataclass is not, and is refused here rather than later.
    Registering twice is harmless.
    """
    _holders.add(holder)


def release_registered_graphs() -> int:
    """Drop every registered holder's graphs; return how many were dropped.

    Process-wide, which is as narrow as a registry gets: the holders are
    reached by nothing else, so there is no runner to ask for its own. The
    other graph store in this process is already process-global for the same
    reason (`region_graph.graph_pool`), and a rollout worker holds one runner.
    """
    return sum(holder.release_graphs() for holder in list(_holders))
