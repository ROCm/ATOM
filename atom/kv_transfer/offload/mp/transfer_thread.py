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
(`look`, `fence`), and every `_IDLE_POLL_S` while transfers are in flight. An
asked-for round runs after the submission in progress, whatever is still
queued, so a backlog of submissions delays neither a finished load's report
nor a fence. Submits and terminal polls (the IPC handle import) both cost the
forward thread milliseconds under its own launch pressure. A failed round stops
the polling; the forward thread raises the failure on every step. A future a
poll finished holds the server's imported IPC event, so it is handed back
(`hand_back`) for the forward thread to drop once nothing else holds it
(`take_returned`): LMCache keeps each future until its next submission, on this
thread. Native ATOM polls on its forward thread.
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
# With ``poll`` and transfers in flight: the round period when nothing wakes the
# thread, a backstop for a rank that has stopped stepping. Rounds are otherwise
# asked for by the forward thread, never run on a short timer.
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
    forward thread asks for one (`request_round`). ``poll`` gets this thread
    and returns whether transfers are still in flight; only then does the
    thread poll on a timer too.
    """

    def __init__(
        self, name: str, poll: Callable[[_TransferThread], bool] | None = None
    ) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._idle = threading.Condition()
        self._outstanding = 0
        self._spent: list[tuple[Any, ...]] = []
        self._spent_lock = threading.Lock()
        self._error: BaseException | None = None
        self._poll = poll
        # Round numbers: the last started, the last ended, and the one the
        # forward thread has asked for, which runs after the submission in
        # progress.
        self._rounds_started = 0
        self._rounds_done = 0
        self._rounds_requested = 0
        self._poll_failed = False
        self._in_flight = False
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
        still does, one step later. A failed poll round is raised on every
        call: no round runs after it, so nothing else would report the
        transfers left in flight.
        """
        with self._idle:
            error = self._error
            if not self._poll_failed:
                self._error = None
        if error is not None:
            raise error

    def _record_error(self, exc: BaseException, *, poll: bool = False) -> None:
        """Keep the first failure for `raise_error`; under the lock, so the
        forward thread's swap cannot drop one recorded meanwhile, nor see a
        failed poll before it is marked as one."""
        with self._idle:
            if self._error is None:
                self._error = exc
            if poll:
                self._poll_failed = True

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

    def request_round(self) -> int:
        """Ask for a round after the submission in progress; its number."""
        with self._idle:
            target = self._rounds_started + 1
            self._rounds_requested = max(self._rounds_requested, target)
        self._queue.put(_WAKE)
        return target

    def wait_round(self, target: int, timeout: float) -> bool:
        """Wait for round ``target`` to end. False on timeout, or when no
        round will: the thread stopped, or its polling failed."""
        with self._idle:
            self._idle.wait_for(
                lambda: self._rounds_done >= target
                or self._poll_failed
                or not self._thread.is_alive(),
                timeout=timeout,
            )
            return self._rounds_done >= target

    def look(self, wait_s: float) -> None:
        """Ask for a round, and wait up to ``wait_s`` for it when it is a pure
        poll: nothing queued to submit and no round running. A round that has
        to submit first outlasts any wait short enough for a step; what it
        settles is reported on a later step."""
        with self._idle:
            pure = self._outstanding == 0 and self._rounds_done == self._rounds_started
        target = self.request_round()
        if wait_s > 0 and pure:
            self.wait_round(target, wait_s)

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
        deadline = time.monotonic() + deadline_s
        while True:
            self.raise_error()
            left = outstanding()
            if not left:
                return
            if not self._thread.is_alive():
                raise RuntimeError(
                    f"LMCache MP fence for {what}: no transfer thread polls {left}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"LMCache MP fence for {what} timed out after {deadline_s:.0f}s, "
                    f"the transfer deadline, with {len(left)} transfer(s) in flight"
                )
            self.wait_round(self.request_round(), min(remaining, _IDLE_POLL_S))

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until every queued submission has run. False on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._outstanding == 0, timeout)

    def close(self, timeout: float | None = None) -> bool:
        """Submit what is queued, stop, and join; whether the thread stopped."""
        if not self._thread.is_alive():
            return True
        self.drain(timeout)
        self._queue.put(None)
        self._thread.join(timeout)
        return not self._thread.is_alive()

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
                # On a timer only while transfers are in flight: with none, a
                # submission or a request wakes the thread.
                item = self._queue.get(
                    timeout=_IDLE_POLL_S if self._in_flight else None
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
            if self._poll is None or self._poll_failed:
                continue
            with self._idle:
                asked = self._rounds_requested > self._rounds_started
            if asked or self._queue.empty():
                self._poll_round()

    def _poll_round(self) -> None:
        with self._idle:
            self._rounds_started += 1
            number = self._rounds_started
        try:
            self._in_flight = bool(self._poll(self))
        except BaseException as exc:  # re-raised by raise_error
            # A transfer deadline is fail-stop, and only the forward thread can
            # stop the engine. Another round would only fail and log again.
            logger.exception("LMCache MP transfer poll failed; polling stops")
            self._record_error(exc, poll=True)
        finally:
            with self._idle:
                self._rounds_done = number
                self._idle.notify_all()


__all__ = ["_FENCE_MAX_RECORDS", "_ProducerFences", "_TransferThread"]
