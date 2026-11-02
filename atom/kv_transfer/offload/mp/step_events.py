# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The interprocess events an LMCache MP transfer waits on before it moves bytes."""

from __future__ import annotations

from typing import Any

import torch


class StepEventRing:
    """Interprocess step events, re-recorded only once nothing waits on them.

    The server waits on a step's event before it moves the step's bytes, and
    re-recording an event it still waits on raises `HIP error: invalid
    argument`. So an event is recorded again only once every transfer
    submitted with it has reached a terminal state. An event is recorded only
    for a transfer that is submitted with it (`PendingStepEvent`) and held at
    once: with TP rank collapse a rank whose saves all settle without the
    server recorded its event step after step and exported it only at its
    first load, and the server's wait on it then failed in every such run.

    Events are not destroyed while the connector runs: on ROCm 7.2.x every
    destroy synchronises every stream of the device. `clear` destroys them at
    close. Callers hold the connector's `_lock` around every method.
    """

    def __init__(self) -> None:
        self._events: list[Any] = []
        self._holds: list[int] = []
        # Indices with no transfer holding them, ready to be recorded again.
        self._free: set[int] = set()

    def __len__(self) -> int:
        return len(self._events)

    def record_locked(self) -> tuple[int, Any]:
        """Record a free event on the current stream; the caller holds it."""
        if self._free:
            index = self._free.pop()
        else:
            index = len(self._events)
            self._events.append(torch.cuda.Event(interprocess=True))
            self._holds.append(0)
        event = self._events[index]
        event.record(torch.cuda.current_stream())
        return index, event

    def hold(self, index: int) -> None:
        """One more transfer submitted with this event.

        Taken off the free set too: a step's first transfer can turn terminal
        before its second is held, and the event must not be recorded again
        while the second still waits on it.
        """
        self._holds[index] += 1
        self._free.discard(index)

    def release(self, index: int) -> None:
        """One transfer submitted with this event became terminal."""
        self._holds[index] -= 1
        if not self._holds[index]:
            self._free.add(index)

    def clear(self) -> None:
        """Destroy every event; at close, once no transfer can wait on one."""
        self._events.clear()
        self._holds.clear()
        self._free.clear()


class PendingStepEvent:
    """This step's event, recorded when its first transfer reaches the server.

    A step whose transfers all settle without the server -- a rank that does
    not write under TP rank collapse, a save with nothing to store -- records
    nothing. Taken on the step thread, at the same point of its stream as an
    event recorded at the top of `start_load_kv`: no kernel is launched in
    between.
    """

    __slots__ = ("_ring", "_taken")

    def __init__(self, ring: StepEventRing) -> None:
        self._ring = ring
        self._taken: tuple[int, Any] | None = None

    def hold_locked(self) -> tuple[int, Any]:
        """The event and its ring index, held for one more transfer.

        Recorded on first use. The caller holds the connector's `_lock` and
        releases the index once that transfer is terminal.
        """
        if self._taken is None:
            self._taken = self._ring.record_locked()
        index, event = self._taken
        self._ring.hold(index)
        return index, event


__all__ = ["PendingStepEvent", "StepEventRing"]
