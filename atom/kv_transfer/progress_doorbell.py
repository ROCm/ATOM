# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Worker -> engine doorbell for KV transfer progress.

A worker answers the engine's completion query (`async_proc_aggregation`) only
when the engine asks, and the engine asks after each forward. A tier load that
lands while the engine is scheduling is therefore seen one forward late. The
doorbell lets the thread that finished the load say so at once: it pushes one
small marker on the worker's KV output channel, the engine's KV output thread
turns it into a wake of the engine's KV progress event, and the engine then
asks for the completions as usual. The doorbell carries no completion data;
the aggregation protocol (one outstanding query, drain-once replies, TP quorum
in `KVOutputAggregator`) is unchanged.

Rings coalesce: after one ring, the next one is sent only once the worker has
answered a completion query (`rearm`, called before the query drains the
completion sets), so a burst of completions costs one message, and none is
lost: a completion recorded after `rearm` either is in that query's answer or
rings again.

Installed per worker process by `AsyncIOProc` only when the engine uses it
(`ATOM_OFFLOAD_LOAD_EVENT_WAIT`); uninstalled, `ring` is a no-op.
"""

from __future__ import annotations

import threading
from typing import Callable

# The marker sent on the KV output channel. A plain string so it pickles
# anywhere and can never be mistaken for a `KVConnectorOutput`.
KV_PROGRESS_DOORBELL = "ATOM_KV_PROGRESS_DOORBELL"

_lock = threading.Lock()
_send: Callable[[], None] | None = None
_coalesce = True
_rung = False


def install(send: Callable[[], None] | None, *, coalesce: bool = True) -> None:
    """Set (or with None clear) how this process sends the doorbell.

    `coalesce=False` where nothing calls `rearm` (the engine process, whose
    doorbell is a local event set, cheap to repeat)."""
    global _send, _coalesce, _rung
    with _lock:
        _send = send
        _coalesce = coalesce
        _rung = False


def ring() -> None:
    """Announce progress (a load finished or failed); coalesced, never raises."""
    global _rung
    with _lock:
        if _send is None or _rung:
            return
        _rung = _coalesce
        send = _send
    try:
        send()
    except Exception:  # noqa: BLE001 - a lost doorbell only costs latency
        with _lock:
            _rung = False


def rearm() -> None:
    """Allow the next ring. Called before a completion query drains, so every
    completion is either in that answer or rings again."""
    global _rung
    with _lock:
        _rung = False


def installed() -> bool:
    return _send is not None
