# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Defer HIP IPC event destruction until workers reach a forward rendezvous."""

from __future__ import annotations

import queue
import threading
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _ReadyEvent:
    event: Any
    owners: list[weakref.ReferenceType] = field(default_factory=list)

    def reusable(self) -> bool:
        for reference in self.owners:
            owner = reference()
            if owner is not None:
                try:
                    if not owner.query():
                        return False
                except Exception:  # noqa: BLE001 - unknown DMA status keeps the lease
                    return False
        return True


class IPCEventReclaimer:
    """Keep event destructors out of asynchronous transfer bookkeeping.

    HIP IPC event destruction can synchronize GPU streams. A worker doing
    this before the next DP metadata collective can deadlock with a peer
    waiting in that collective. Futures remain responsible for DMA lifetime;
    their finalizers hand event ownership to this queue instead of destroying
    events inline. Drain only after all peers have submitted the previous
    forward, at the next forward's metadata rendezvous.
    """

    def __init__(self, *, background: bool = False) -> None:
        self._retired: deque[Any] = deque()
        self._ready_events: list[_ReadyEvent] = []
        # With `background`, `release` hands retired events to a reaper thread
        # that destroys them. Destroying an imported HIP IPC event can block
        # even after `query()` reported it complete, and the work it waits for
        # can depend on this rank's next collective (the MP server's copy waits
        # on our producer event, our stream on the peer). Blocking the reaper
        # instead of the runner lets that collective launch, which breaks the
        # cycle.
        self._reaper = _Reaper() if background else None

    def acquire_ready_event(self, factory: Any) -> Any:
        # LMCache also retains exported events in a ring. Owning and reusing
        # them here prevents a later ring eviction from destroying an event
        # outside the rendezvous. Pool size follows peak in-flight batches;
        # events are re-recorded only after every old transfer has completed.
        for entry in self._ready_events:
            if entry.reusable():
                entry.owners.clear()
                return entry.event
        entry = _ReadyEvent(factory())
        self._ready_events.append(entry)
        return entry.event

    def track_ready_event(self, event: Any, owner: Any) -> None:
        # LMCache returns None when it rejects a transfer before submission.
        # No DMA owns the event in that case; other submitted owners still do.
        if owner is None:
            return
        for entry in self._ready_events:
            if entry.event is event:
                # The connector's pending tables own futures until terminal
                # completion. A weak reference avoids retaining their imported
                # completion events beyond normal transfer bookkeeping.
                entry.owners.append(weakref.ref(owner))
                return

    def defer(self, event: Any) -> None:
        self._retired.append(event)

    def retain_until_released(self, owner: Any, event: Any) -> None:
        if event is None:
            return
        finalizer = weakref.finalize(owner, self.defer, event)
        # Runtime teardown owns shutdown; do not run these before live futures
        # have drained merely because Python is executing atexit handlers.
        finalizer.atexit = False

    def release(self) -> None:
        # Swap before dropping references: callbacks that retire more events
        # while this batch is reclaimed leave them for the next safe point.
        retired, self._retired = self._retired, deque()
        if self._reaper is not None:
            self._reaper.put(retired)
            return
        # Destroy only events whose work has completed. Destroying an
        # incomplete HIP IPC event synchronizes on it, and the work it waits
        # for can depend on a DP peer: the MP server's copy for this rank may
        # sit behind one for a peer whose GPU is waiting in the collective
        # this rank has not launched yet. Unfinished events stay for the next
        # forward's rendezvous.
        keep = deque(event for event in retired if not _completed(event))
        retired.clear()
        self._retired.extendleft(reversed(keep))


def _completed(event: Any) -> bool:
    query = getattr(event, "query", None)
    if not callable(query):
        return True
    try:
        return bool(query())
    except Exception:  # noqa: BLE001 - unknown status: keep the event
        return False


class _Reaper:
    """Destroys retired events on a daemon thread, off the forward path."""

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name="atom-ipc-event-reaper", daemon=True
        )
        self._thread.start()

    def put(self, events: deque) -> None:
        if events:
            self._queue.put(events)

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until everything handed over so far is destroyed (tests)."""
        done = threading.Event()
        self._queue.put(done)
        return done.wait(timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if isinstance(item, threading.Event):
                item.set()
                continue
            # Drop the last references here: destruction may block.
            item.clear()
            del item
