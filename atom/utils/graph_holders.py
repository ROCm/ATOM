# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Who is holding a captured cudagraph, for whoever frees what it captured.

A graph replays the addresses it recorded, so releasing the weights or the KV
pool invalidates every graph that captured them -- see
``atom/rollout/memory_manager.py``, which does the releasing. Finding those
graphs is the problem this module exists for.

``runner.graphs`` is only the manual whole-forward store. Under PIECEWISE the
graphs are one per compiled dense piece, each held by a ``CUDAGraphWrapper``
that the compile backend installs with ``module.__dict__[target] = ...``
(``atom/utils/backends.py``) on a submodule of a split ``fx.GraphModule`` that
only Dynamo's cache holds. Both halves of that defeat a walk: there is no path
from the runner to the graph module, and the ``__dict__`` assignment keeps the
wrapper out of ``_modules`` even for someone who has one. So a holder no walk
can reach registers itself here instead, and the releaser asks this module.

Registration is the last resort, not the convention: a store the releaser can
reach -- ``runner.graphs``, ``UBatchWrapper.tbo_graphs``,
``Drafter.draft_graphs`` -- is walked there, where the reader can see what is
being cleared.

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
    other graph stores in this process are already process-global for the same
    reason (`atom/utils/cuda_graph.py`'s graph pools, `deepseek_v4.py`'s
    attention-core capture runner), and a rollout worker holds one runner.
    """
    return sum(holder.release_graphs() for holder in list(_holders))
