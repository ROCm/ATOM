# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wire types for the generic collective RPC.

Deliberately free of heavy imports. ``async_proc`` cannot be imported without a
real AITER build, and ``engine_utility`` needs these same types to build a
request, so keeping them here is what lets the dispatch layer stay importable on
a machine with no GPU -- which is also how the non-GPU test runner sees them.
"""

import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# The utility-command name the dispatch layer registers. Shared so the manager
# that sends it and the handler that receives it cannot drift apart.
COLLECTIVE_RPC_CMD = "collective_rpc"


@dataclass(frozen=True)
class RpcPayload:
    """One generic collective-RPC call: kwargs plus an explicit barrier request.

    The plain worker wire format is ``(func_name, *args)``, which cannot carry
    kwargs. Rather than change that format and every existing ``call_func``
    caller, a generic call sends exactly one positional argument -- this -- and
    the worker recognises it by type.

    ``request_id`` lets the manager match replies to the call that produced
    them. The older KV channel matches by count instead, which is why it can
    only ever have one aggregation outstanding.
    """

    request_id: str
    args: tuple = ()
    kwargs: dict | None = None
    barrier: bool = False

    def call_kwargs(self) -> dict:
        return self.kwargs or {}


@dataclass(frozen=True)
class RpcResult:
    """A generic-RPC reply, so a ``None`` return still reaches the caller.

    ``busy_loop`` only forwards non-``None`` worker returns, and
    ``call_func(wait_out=True)`` blocks on an untimed queue get, so a method
    that legitimately returns ``None`` deadlocks the caller. Wrapping every
    outcome -- including failures -- means the generic path always answers.
    """

    request_id: str
    tp_rank: int
    value: object = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class RpcResponseRouter:
    """Deliver DP-engine replies to whichever caller is waiting for them.

    ``broadcast_utility_command_sync`` reads a fixed number of replies off one
    shared queue, so it matches by count: two overlapping callers take each
    other's replies, and a late reply from an abandoned call is handed to the
    next caller as its own. Upstream already recorded that biting the old
    pull-based metrics.

    Registering a request id gives that call its own queue, so concurrent calls
    cannot collide and a reply nobody is waiting for is dropped rather than
    mis-delivered.
    """

    def __init__(self) -> None:
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    @contextmanager
    def register(self, request_id: str) -> Iterator[queue.Queue]:
        own: queue.Queue = queue.Queue()
        with self._lock:
            if request_id in self._queues:
                raise RuntimeError(f"request id already in flight: {request_id}")
            self._queues[request_id] = own
        try:
            yield own
        finally:
            # Unregister even on the timeout path, so a late reply is dropped by
            # ``route`` rather than kept for a caller that has given up.
            with self._lock:
                self._queues.pop(request_id, None)

    def route(self, request_id: str, item: object) -> bool:
        """Hand *item* to the waiter for *request_id*; False if nobody waits."""
        with self._lock:
            own = self._queues.get(request_id)
        if own is None:
            return False
        own.put_nowait(item)
        return True

    def in_flight(self) -> int:
        with self._lock:
            return len(self._queues)
