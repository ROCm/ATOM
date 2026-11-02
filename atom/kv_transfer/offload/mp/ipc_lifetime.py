# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""When an LMCache MP future, which holds an imported IPC event, is destroyed.

On ROCm 7.2.x destroying an interprocess event synchronises every stream of the
device and blocks event creation on every other thread meanwhile. A finished
future carries the server's imported event, so it is destroyed on the step
thread while that thread's stream is idle, and only once nothing else holds it.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from typing import Any

import torch

# Finished futures held while the stream is busy, past which a busy stream
# releases anyway.
OWNED_IPC_CAP = 8192
# The fewest a busy stream past the cap releases in one call, so a stream that
# is never idle (async scheduling) pays the backlog a batch at a time.
BUSY_RELEASE_BATCH = 64


def _stream_idle() -> bool:
    if not torch.cuda.is_available():
        return True
    return bool(torch.cuda.current_stream().query())


def _sole_owner_refcount() -> int:
    """`sys.getrefcount` of an object only a local variable holds."""
    entry = object()
    return sys.getrefcount(entry)


_SOLE_OWNER_REFCOUNT = _sole_owner_refcount()


def take_unshared(owned: deque[Any], limit: int | None = None) -> list[Any]:
    """Remove and return up to `limit` entries nothing else holds any more.

    Entries are checked until `limit` are taken: one held for good elsewhere
    (a logged exception's traceback, say) goes to the back rather than keep
    the ones behind it alive, and the unchecked rest stay queued in order.
    """
    unshared = []
    for _ in range(len(owned)):
        if limit is not None and len(unshared) >= limit:
            break
        entry = owned.popleft()
        if sys.getrefcount(entry) <= _SOLE_OWNER_REFCOUNT:
            unshared.append(entry)
        else:
            owned.append(entry)
    return unshared


class OwnedFutures:
    """Finished futures, handed over by any thread, destroyed on the step thread.

    LMCache keeps each future until its next submission, and the drain
    thread's locals hold some between rounds; letting go first would leave the
    destroy to whichever of them lets go last, on the drain thread.
    """

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self._owned: deque[Any] = deque()

    def __len__(self) -> int:
        return len(self._owned)

    def add_locked(self, future: Any) -> None:
        """Take ownership; the caller holds the connector's `_lock`."""
        if future is not None:
            self._owned.append(future)

    def release(self) -> None:
        """Destroy what nothing else holds, on the step thread.

        While the stream is idle -- under synchronous scheduling, the start of
        a step -- every such future, since a destroy then waits on nothing.
        While it is busy, only past `OWNED_IPC_CAP`, and then down to the cap
        (at least `BUSY_RELEASE_BATCH`), so the futures held here and nowhere
        else never exceed the cap by more than one step's worth.
        """
        with self._lock:
            owned = len(self._owned)
        if not owned:
            return
        idle = _stream_idle()
        if owned < OWNED_IPC_CAP and not idle:
            return
        limit = None if idle else max(BUSY_RELEASE_BATCH, owned - OWNED_IPC_CAP)
        # Added under `_lock` by the drain thread; destroyed outside it.
        with self._lock:
            unshared = take_unshared(self._owned, limit)
        del unshared


__all__ = ["BUSY_RELEASE_BATCH", "OWNED_IPC_CAP", "OwnedFutures", "take_unshared"]
