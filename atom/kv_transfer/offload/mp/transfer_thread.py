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

Given a ``poll`` callable (the vLLM plugin's worker), the thread also polls
pending transfers: whenever its queue runs dry, when the forward thread asks
(`look`, `fence`), and every `_IDLE_POLL_S` otherwise. An asked-for round runs
after the submission in progress, whatever is still queued, so a backlog of
submissions delays neither a finished load's report nor a fence. A conc16 step
probe of the plugin priced the forward thread's share at 18.7 ms per submit and
13.8 ms per terminal poll (the IPC handle import) under its own launch
pressure. A future a poll finished holds the server's imported IPC event, so it
is handed back (`hand_back`) for the forward thread to drop once nothing else
holds it (`take_returned`): LMCache keeps each future until its next
submission, on this thread. Native ATOM polls on its forward thread.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from collections import deque
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
# With ``poll``: the round period when nothing wakes the thread, a backstop for
# a rank that has stopped stepping. Demand-driven otherwise: a 2 ms cadence
# with loads outstanding cost more end to end (ITL p50 23.1 -> 28.8 ms) than
# the forward-thread work it moved off.
_IDLE_POLL_S = 0.05
# Queued to wake a polling thread for a round; runs no call.
_WAKE = object()


def _sole_owner_refcount() -> int:
    """`sys.getrefcount` of an object only a local variable holds."""
    entry = object()
    return sys.getrefcount(entry)


_SOLE_OWNER_REFCOUNT = _sole_owner_refcount()


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
    """Runs queued transfer submissions in order on one thread.

    With ``poll``, also a round of polls whenever the queue runs dry or the
    forward thread asks for one (`look`, `fence`).
    """

    def __init__(self, name: str, poll: Callable[[], None] | None = None) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._idle = threading.Condition()
        self._outstanding = 0
        self._spent: list[tuple[Any, ...]] = []
        self._spent_lock = threading.Lock()
        self._error: BaseException | None = None
        self._poll = poll
        self._rounds_started = 0
        self._rounds_done = 0
        self._round_asked = False
        self._returned: deque[Any] = deque()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def polls(self) -> bool:
        return self._poll is not None

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
        with self._idle:
            error, self._error = self._error, None
        if error is not None:
            raise error

    def _record_error(self, exc: BaseException) -> None:
        """Keep the first failure for `raise_error`; under the lock, so the
        forward thread's swap cannot drop one recorded meanwhile."""
        with self._idle:
            if self._error is None:
                self._error = exc

    def hand_back(self, future: Any) -> None:
        """A future a poll finished, for the forward thread to drop."""
        if future is not None:
            with self._spent_lock:
                self._returned.append(future)

    @property
    def returned(self) -> int:
        """Handed-back futures not yet taken."""
        return len(self._returned)

    def take_returned(self, limit: int | None = None) -> list[Any]:
        """Up to ``limit`` handed-back futures nothing else holds any more.

        One LMCache still holds waits for its next submission rather than
        leave the drop to that submission, on this thread.
        """
        unshared = []
        with self._spent_lock:
            for _ in range(len(self._returned)):
                if limit is not None and len(unshared) >= limit:
                    break
                future = self._returned.popleft()
                if sys.getrefcount(future) <= _SOLE_OWNER_REFCOUNT:
                    unshared.append(future)
                else:
                    self._returned.append(future)
        return unshared

    def look(self, wait_s: float) -> None:
        """Ask for a poll round, and wait up to ``wait_s`` for it when it is a
        pure poll: nothing queued to submit and no round running. A round that
        has to submit first outlasts any wait short enough for a step; what it
        settles is reported on a later step."""
        with self._idle:
            pure = self._outstanding == 0 and self._rounds_done == self._rounds_started
        target = self._ask_round()
        if wait_s > 0 and pure:
            self._wait_for_round(target, wait_s)

    def fence(
        self,
        outstanding: Callable[[], list[Any]],
        deadline_s: float,
        what: str,
    ) -> None:
        """Block until ``outstanding()`` is empty, or fail-stop.

        ``outstanding`` names the transfers that still read or write the
        blocks being fenced, queued submissions included. Raises when this
        thread has failed or stopped with transfers outstanding, or after
        ``deadline_s``: returning would let the caller reuse blocks a transfer
        still reads.
        """
        if self._poll is None:
            raise NotImplementedError(
                f"LMCache MP fence for {what}: only a polling transfer thread "
                "fences (the vLLM plugin's)"
            )
        deadline = time.monotonic() + deadline_s
        while True:
            self.raise_error()
            left = outstanding()
            if not left:
                return
            if not self._thread.is_alive():
                raise RuntimeError(
                    f"LMCache MP fence for {what}: no transfer thread to submit or "
                    f"poll {left}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"LMCache MP fence for {what} timed out after {deadline_s:.0f}s, "
                    f"the transfer deadline, with {len(left)} transfer(s) in flight"
                )
            self._wait_round(min(remaining, _IDLE_POLL_S))

    def _ask_round(self) -> int:
        """Ask for a round after the submission in progress; its number."""
        with self._idle:
            self._round_asked = True
            target = self._rounds_started + 1
        self._queue.put(_WAKE)
        return target

    def _wait_for_round(self, target: int, timeout: float) -> bool:
        with self._idle:
            return (
                self._idle.wait_for(
                    lambda: self._rounds_done >= target or not self._thread.is_alive(),
                    timeout=timeout,
                )
                and self._rounds_done >= target
            )

    def _wait_round(self, timeout: float) -> bool:
        """Ask for a round and wait for it: one that starts after this call."""
        return self._wait_for_round(self._ask_round(), timeout)

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
            try:
                item = self._queue.get(
                    timeout=_IDLE_POLL_S if self._poll is not None else None
                )
            except queue.Empty:
                item = _WAKE
            if item is None:
                return
            if item is not _WAKE:
                fn, args, ready = item
                try:
                    if ready is not None:
                        self._wait_ready(ready)
                    fn(*args)
                except BaseException as exc:  # re-raised by raise_error
                    logger.exception("LMCache MP transfer submission failed")
                    self._record_error(exc)
                finally:
                    with self._spent_lock:
                        self._spent.append(args)
                    del fn, args, ready, item
                    with self._idle:
                        self._outstanding -= 1
                        if not self._outstanding:
                            self._idle.notify_all()
            if self._poll is None:
                continue
            with self._idle:
                asked = self._round_asked
            if asked or self._queue.empty():
                self._poll_round()

    def _poll_round(self) -> None:
        with self._idle:
            self._round_asked = False
            self._rounds_started += 1
            number = self._rounds_started
        try:
            self._poll()
        except BaseException as exc:  # re-raised by raise_error
            # A transfer deadline is fail-stop, and only the forward thread can
            # stop the engine.
            logger.exception("LMCache MP transfer poll failed")
            self._record_error(exc)
        finally:
            with self._idle:
                self._rounds_done = number
                self._idle.notify_all()


__all__ = ["_FENCE_MAX_RECORDS", "_ProducerFences", "_TransferThread"]
