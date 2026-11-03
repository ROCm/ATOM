# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where LMCache MP transfers are submitted and polled: a thread or the step.

`TransferDrain`, for the vLLM plugin, runs them on a thread of its own.
`InlineTransfers`, for native ATOM's PAGE-only worker, runs the same calls on
the step thread: a submit before the forward and a poll at the report, as
native ATOM always has.

Submitting a transfer and polling it to completion are both HIP IPC calls
(`export_event` / `from_ipc_handle`). On an idle GPU they cost microseconds,
but under the rank's own kernel-launch pressure milliseconds each -- and under
TP rank collapse only the writer pays, so the whole TP group waits at the next
collective. A conc16 step probe of the plugin priced it on the step thread: 18.7 ms per
`submit_retrieve_request` and 13.8 ms per terminal load poll (the IPC handle
import; a non-terminal poll costs 0.2 us). Both release the GIL, so on this
thread a hard per-step stall becomes a brief launch-latency tax.

Demand-driven, not timer-driven: a 2 ms poll cadence with loads outstanding
measured ~1400 polls/s/rank and cost more end to end (ITL p50 23.1 -> 28.8 ms)
than the step-thread work it moved off. The step wakes it instead, from
`start_load_kv`, so the IPC work lands while the step thread is inside the
model forward with the GIL released.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("atom")

# The wait for a wake when nothing asks: a backstop for teardown and for a
# rank that has stopped stepping, not the working cadence.
_IDLE_WAIT_S = 0.05


class TransferDrain:
    """One thread that submits queued transfers and polls pending ones.

    Every round submits the whole queue in order -- one queue for both legs,
    so a request's load still reaches the server before its save -- and then
    calls ``poll``. ``submit`` and ``poll`` are the connector's; they take the
    connector's ``lock`` themselves around its bookkeeping, and this class
    shares that lock for its round signal.
    """

    def __init__(
        self,
        lock: threading.Lock,
        submit: Callable[[Any], None],
        poll: Callable[[], None],
    ) -> None:
        self._submit = submit
        self._poll = poll
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._round_cv = threading.Condition(lock)
        self._rounds_started = 0
        self._rounds_done = 0
        # Deadline expiry is fail-stop, so it has to reach the step that
        # calls `get_finished` rather than just ending this thread.
        self.error: BaseException | None = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def put(self, item: Any) -> None:
        """Queue one submit, then make sure a thread will run it.

        Queued before the wake, so the round the wake starts submits it.
        """
        if self._stop.is_set():
            raise RuntimeError("LMCache MP transfer drain is closed")
        self._queue.put(item)
        self.ensure_running()

    def ensure_running(self) -> None:
        """Start the thread if needed, and wake it.

        Also for work that was not queued, such as a terminal entry only a
        round reports. Refused once `close` has been called.
        """
        if self._stop.is_set():
            raise RuntimeError("LMCache MP transfer drain is closed")
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="atom-lmcache-mp-transfer", daemon=True
            )
            self._thread.start()
        self._wake.set()

    def report_round(self, wait_s: float) -> None:
        """Before a report, with transfers in flight: wake the thread.

        Waits up to ``wait_s`` for the round only when it is a pure poll --
        nothing queued to submit, no round already running -- since a round
        that has to submit first outlasts any wait short enough for a step.
        What the round settles after the wait is reported on a later step.
        """
        with self._round_cv:
            target = self._rounds_started + 1
            pure_poll = (
                self._queue.empty() and self._rounds_done == self._rounds_started
            )
            self._wake.set()
            if wait_s <= 0 or not pure_poll or not self.alive:
                return
            self._round_cv.wait_for(
                lambda: self._rounds_done >= target or not self.alive,
                timeout=wait_s,
            )

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise self.error

    def wait_round(self, timeout: float) -> bool:
        """Wake the thread and wait for a round that starts after this call.

        That round has submitted everything queued before the call and polled
        every pending transfer once. Returns whether it finished within
        ``timeout``; False also when the thread is not running.
        """
        if not self.alive:
            return False
        with self._round_cv:
            target = self._rounds_started + 1
            self._wake.set()
            return (
                self._round_cv.wait_for(
                    lambda: self._rounds_done >= target or not self.alive,
                    timeout=timeout,
                )
                and self._rounds_done >= target
            )

    def fence(
        self,
        outstanding: Callable[[], list[Any]],
        deadline_s: float,
        what: str,
    ) -> None:
        """Block until ``outstanding()`` is empty, or fail-stop.

        ``outstanding`` names the transfers that still read or write the
        blocks being fenced, and is called with the connector's lock held.
        The drain polls those transfers and drops terminal ones; this thread
        only waits on its rounds. Raises when the drain has failed or stopped
        with transfers outstanding, or after ``deadline_s`` -- returning would
        let the caller reuse blocks a transfer still reads.
        """
        deadline = time.monotonic() + deadline_s
        while True:
            with self._round_cv:
                left = outstanding()
            if not left:
                return
            if self.error is not None:
                raise RuntimeError(
                    f"LMCache MP fence for {what}: the transfer drain failed with "
                    f"{len(left)} transfer(s) in flight"
                ) from self.error
            if not self.alive:
                raise RuntimeError(
                    f"LMCache MP fence for {what}: no drain thread to submit or "
                    f"poll {left}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"LMCache MP fence for {what} timed out after {deadline_s:.0f}s, "
                    f"the transfer deadline, with {len(left)} transfer(s) in flight"
                )
            self.wait_round(min(remaining, _IDLE_WAIT_S))

    def close(self, timeout: float) -> bool:
        """Stop after one last round over what is queued; whether it stopped.

        The handle is kept while the thread is still running (a submit can be
        blocked longer than ``timeout``), and no new thread starts once this is
        called: a second one would share the connector's client, which is not
        thread-safe, with the one still running.
        """
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _loop(self) -> None:
        while True:
            # Read before the round, so the round that sees it is a whole one:
            # what `close` finds queued is still submitted and polled once.
            stopping = self._stop.is_set()
            try:
                if not stopping:
                    self._wake.wait(timeout=_IDLE_WAIT_S)
                # Cleared before the work, so a wake raised while this round is
                # running is kept for the next round rather than swallowed.
                self._wake.clear()
                with self._round_cv:
                    self._rounds_started += 1
                    number = self._rounds_started
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    self._submit(item)
                self._poll()
                with self._round_cv:
                    self._rounds_done = number
                    self._round_cv.notify_all()
                if stopping:
                    return
            except BaseException as exc:
                # Deadline expiry and any unexpected failure are fail-stop, and
                # this thread is not the one that can stop the engine.
                if self.error is None:
                    self.error = exc
                logger.exception("LMCache MP transfer drain failed")
                with self._round_cv:
                    self._round_cv.notify_all()
                return


class InlineTransfers:
    """What the worker asks of `TransferDrain`, run on the calling thread.

    A submit happens in `put` and a round -- one poll of every pending
    transfer -- in `report_round`. Anything either raises, a transfer deadline
    included, raises to the caller, so nothing is left for `raise_if_failed`.
    """

    def __init__(
        self,
        lock: threading.Lock,
        submit: Callable[[Any], None],
        poll: Callable[[], None],
    ) -> None:
        del lock  # `submit` and `poll` take it themselves
        self._submit = submit
        self._poll = poll
        self._closed = False

    def put(self, item: Any) -> None:
        if self._closed:
            raise RuntimeError("LMCache MP transfers are closed")
        self._submit(item)

    def ensure_running(self) -> None:
        pass  # the report's own round reports what was not queued

    def report_round(self, wait_s: float) -> None:
        del wait_s  # the round runs here, to completion
        self._poll()

    def raise_if_failed(self) -> None:
        pass

    def fence(self, outstanding: Any, deadline_s: float, what: str) -> None:
        raise NotImplementedError(
            f"LMCache MP fence for {what}: only the vLLM plugin hands a "
            "preempted request's blocks on before its transfers end, and it "
            "runs them on a drain thread (`transfer_thread=True`)"
        )

    def close(self, timeout: float) -> bool:
        del timeout
        self._closed = True
        return True


__all__ = ["InlineTransfers", "TransferDrain"]
