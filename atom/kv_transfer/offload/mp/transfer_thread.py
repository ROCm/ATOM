# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Submit LMCache MP transfers off a TP worker's forward thread.

A submission builds the transfer key (client chunk hashes when LMCache sends
them), exports the producer event and serializes the request: milliseconds of
CPU the forward thread otherwise spends between launching kernels. Here the
forward thread records the fence and queues the call; one thread submits in
queue order.

Two things never happen on this thread:
- GPU work. The forward thread records every fence, at the stream position the
  transfer must follow; recording here would take this thread's stream.
- Dropping the last reference to an interprocess event. On ROCm that destroy
  waits for the process's GPU work and takes a device-wide lock, which would
  stall the forward thread's own event calls. Arguments of finished calls are
  handed back through `take_spent` and dropped on the forward thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger("atom")

# ROCm 7.2.4 fails a stream wait on an imported interprocess event once its
# exporter has recorded it more than 32 times; a fence gets a fresh event after
# this many records.
_FENCE_MAX_RECORDS = 31
# Poll period while a queued submission waits for its fence to complete.
_READY_POLL_S = 0.0005


class _ProducerFences:
    """Interprocess producer events reused across steps (forward thread only).

    Each fence is recorded where a transfer must start: before the forward for
    loads, after it for same-step saves. A server that waits on a slot recorded
    again since sees a later position, so it only waits longer. Slots rotate so
    a re-record is far behind any submission that still waits to go out. An
    event replaced after `_FENCE_MAX_RECORDS` is kept until `release_retired`,
    which the worker calls once the step's forward has finished.
    """

    def __init__(self, slots: int = 64) -> None:
        self._events: list[Any] = [None] * slots
        self._records = [0] * slots
        self._next = 0
        self._retired: list[Any] = []

    def record(self) -> Any:
        slot = self._next % len(self._events)
        self._next += 1
        event = self._events[slot]
        if event is None or self._records[slot] >= _FENCE_MAX_RECORDS:
            if event is not None:
                self._retired.append(event)
            event = torch.cuda.Event(interprocess=True)
            self._events[slot] = event
            self._records[slot] = 0
        event.record(torch.cuda.current_stream())
        self._records[slot] += 1
        return event

    def release_retired(self) -> None:
        """Drop replaced events; pending transfers keep their own references."""
        self._retired.clear()


class _TransferThread:
    """Runs queued transfer submissions in order on one thread."""

    def __init__(self, name: str) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._idle = threading.Condition()
        self._outstanding = 0
        self._spent: list[tuple[Any, ...]] = []
        self._spent_lock = threading.Lock()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        ready: Callable[[], bool] | None = None,
    ) -> None:
        """Queue ``fn(*args)``; with ``ready``, run it once ``ready()`` is true.

        Later submissions wait behind it: the queue keeps submission order.
        """
        with self._idle:
            self._outstanding += 1
        self._queue.put((fn, args, ready))

    def take_spent(self) -> list[tuple[Any, ...]]:
        """Arguments of finished calls, for the forward thread to drop."""
        with self._spent_lock:
            spent, self._spent = self._spent, []
        return spent

    def raise_error(self) -> None:
        """Re-raise, on the caller's thread, the first submission that raised.

        Inline, that exception stopped the engine on the forward thread; it
        still does, one step later.
        """
        error, self._error = self._error, None
        if error is not None:
            raise error

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until every queued submission has run. False on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._outstanding == 0, timeout)

    def close(self, timeout: float | None = None) -> None:
        if not self._thread.is_alive():
            return
        self.drain(timeout)
        self._queue.put(None)
        self._thread.join(timeout)

    @staticmethod
    def _wait_ready(ready: Callable[[], bool]) -> None:
        while True:
            try:
                if ready():
                    return
            except Exception:
                # Cannot tell: submit anyway, the server's stream wait still
                # orders the transfer behind the fence.
                logger.warning("LMCache MP transfer fence query failed", exc_info=True)
                return
            time.sleep(_READY_POLL_S)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, args, ready = item
            try:
                if ready is not None:
                    self._wait_ready(ready)
                fn(*args)
            except BaseException as exc:  # re-raised by raise_error
                logger.exception("LMCache MP transfer submission failed")
                if self._error is None:
                    self._error = exc
            finally:
                with self._spent_lock:
                    self._spent.append(args)
                del fn, args, ready, item
                with self._idle:
                    self._outstanding -= 1
                    if not self._outstanding:
                        self._idle.notify_all()


__all__ = ["_FENCE_MAX_RECORDS", "_ProducerFences", "_TransferThread"]
