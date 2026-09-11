# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Save executor whose cancelled tasks release their queue storage immediately."""

from __future__ import annotations

import os
import threading
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


def save_admission_enabled() -> bool:
    value = os.environ.get("OFFLOAD_SAVE_ADMISSION", "1").strip().lower()
    return bool(value) and value not in {"0", "false", "no", "off"}


class SaveQueueFull(RuntimeError):
    """The running-plus-queued save capacity has already been reserved."""


class SeenSaveGenerations:
    """Remember dispatch/cancel generations without retaining request metadata.

    A Dense scheduler issues globally unique, increasing generations. Adjacent
    generations compact to one interval, including rejected and cancelled jobs.
    Unlike a high watermark, intervals also handle cancellation before dispatch
    without suppressing an unseen earlier generation.
    """

    def __init__(self) -> None:
        self._intervals: list[tuple[int, int]] = []

    def remember(self, generation: int) -> bool:
        """Return True exactly once for each generation. Caller owns locking."""
        index = bisect_right(self._intervals, (generation, float("inf"))) - 1
        if index >= 0 and self._intervals[index][1] >= generation:
            return False
        index = bisect_left(self._intervals, (generation, generation))
        start = end = generation
        if index and self._intervals[index - 1][1] + 1 == generation:
            index -= 1
            start = self._intervals.pop(index)[0]
        if index < len(self._intervals) and self._intervals[index][0] == end + 1:
            end = self._intervals.pop(index)[1]
        self._intervals.insert(index, (start, end))
        return True


class _SaveFuture(Future):
    def __init__(self, executor: CancellableSaveExecutor) -> None:
        super().__init__()
        self._executor = executor

    def cancel(self) -> bool:
        return self._executor._cancel(self)


@dataclass
class _SaveTask:
    future: _SaveFuture
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class CancellableSaveExecutor:
    """Nonblocking admission with a physical running-plus-queued bound.

    Future.cancel() removes a queued task, including its function arguments,
    before invoking callbacks. Worker slots are released before done callbacks,
    so a retirement callback cannot free scheduler credit ahead of this slot.
    capacity=None retains the same cancellation protocol for a control run with
    admission disabled. The load executor is intentionally separate.
    """

    def __init__(
        self, *, max_workers: int, capacity: int | None, thread_name_prefix: str
    ) -> None:
        if max_workers <= 0 or (capacity is not None and capacity <= 0):
            raise ValueError("save workers and capacity must be positive")
        self._capacity = capacity
        self._condition = threading.Condition()
        self._queue: deque[_SaveTask] = deque()
        self._queued: dict[_SaveFuture, _SaveTask] = {}
        self._running = 0
        self._shutdown = False
        self._threads: list[threading.Thread] = []
        try:
            # Start before accepting any work: thread creation failure cannot
            # leave an executable task behind an unsuccessful submit().
            for index in range(max_workers):
                thread = threading.Thread(
                    target=self._run,
                    name=f"{thread_name_prefix}_{index}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        except BaseException:
            self.shutdown(wait=True, cancel_futures=True)
            raise

    def counts(self) -> tuple[int, int]:
        with self._condition:
            return len(self._queue), self._running

    def submit(self, fn: Callable[..., Any], /, *args, **kwargs) -> Future:
        with self._condition:
            if self._shutdown:
                raise RuntimeError("cannot schedule saves after shutdown")
            if (
                self._capacity is not None
                and len(self._queue) + self._running >= self._capacity
            ):
                raise SaveQueueFull("save queue capacity exhausted")
            future = _SaveFuture(self)
            task = _SaveTask(future, fn, args, kwargs)
            self._queued[future] = task
            try:
                self._queue.append(task)
            except BaseException:
                self._queued.pop(future, None)
                raise
            self._condition.notify()
            return future

    def _cancel(self, future: _SaveFuture) -> bool:
        with self._condition:
            task = self._queued.pop(future, None)
            if task is not None:
                self._queue.remove(task)
                self._condition.notify_all()
            # The worker marks a popped task RUNNING under this same lock.
            # A cancelled queued task cannot be observed by any worker now.
        return Future.cancel(future)

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._shutdown)
                if not self._queue:
                    return
                task = self._queue.popleft()
                self._queued.pop(task.future)
                if not task.future.set_running_or_notify_cancel():
                    continue
                self._running += 1
            result = None
            error = None
            try:
                result = task.fn(*task.args, **task.kwargs)
            except BaseException as exc:  # noqa: BLE001 - publish through the Future
                error = exc
            finally:
                with self._condition:
                    self._running -= 1
                    self._condition.notify_all()
            if error is None:
                task.future.set_result(result)
            else:
                task.future.set_exception(error)
            # Do not retain the previous request while this thread is idle.
            del task, result, error

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._condition:
            self._shutdown = True
            pending = list(self._queued) if cancel_futures else []
            self._condition.notify_all()
        for future in pending:
            future.cancel()
        if wait:
            for thread in self._threads:
                thread.join()
