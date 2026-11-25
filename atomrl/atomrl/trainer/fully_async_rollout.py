# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Streaming rollout for fully asynchronous training (verl's fully_async_policy).

The rollout pool generates prompt groups (one prompt x num_generations) one at a
time and hands each finished group to the trainer through a queue, so a long
group holds up only its own slot instead of the whole batch. The trainer takes a
step's worth of groups, updates, and every ``sync_every`` updates pushes new
weights to the rollout.

Generation may run ahead of the trainer's weights by a budgeted amount: within
one sync period the rollout aims at ``(1 + staleness_threshold)`` periods' worth
of useful groups, counting the ones still queued from before the sync. Groups
dropped by dynamic sampling do not count, or a run of dropped groups could leave
the trainer waiting on a rollout that believes it is ahead; groups in flight
count at the recent keep rate, so a period ends over budget when more of them
are kept than that rate predicts.

Around a weight sync the rollout either stops submitting and waits for the
groups in flight (``partial_rollout=False``), or freezes them in the engine and
lets them continue on the new weights (``partial_rollout=True``).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol


class GroupBackend(Protocol):
    """How the rollout starts, collects and scores prompt groups."""

    def submit(self, index: int) -> Any:
        """Start generating prompt ``index``; return a handle for it."""

    def wait(self, handles: list[Any], timeout: float) -> list[Any]:
        """Return the handles whose generation has finished."""

    def finish(self, handle: Any) -> Any | None:
        """The group's batch rows, or ``None`` if dynamic sampling drops it."""

    def pause(self) -> None:
        """Freeze in-flight requests so the engine can load new weights."""

    def resume(self) -> None:
        """Let frozen requests continue."""


@dataclass
class RolloutGroup:
    rows: Any
    # Rollout weights when generation started and when it finished; they
    # differ only for a group that a partial-rollout sync caught in flight.
    version: int
    finished_version: int
    submitted_at: float
    finished_at: float


class StreamingRollout:
    """Keeps the rollout pool busy with prompt groups on a background thread."""

    def __init__(
        self,
        backend: GroupBackend,
        merge: Callable[[list[Any]], Any],
        *,
        groups_per_step: int,
        sync_every: int,
        staleness_threshold: float,
        max_inflight: int,
        partial_rollout: bool,
        max_consecutive_drops: int = 0,
        start_version: int = 0,
        start_index: int = 0,
        poll_s: float = 0.05,
    ) -> None:
        if groups_per_step < 1 or sync_every < 1 or max_inflight < 1:
            raise ValueError(
                "groups_per_step, sync_every and max_inflight must be positive"
            )
        if staleness_threshold < 0:
            raise ValueError("staleness_threshold must be >= 0")
        self._backend = backend
        self._merge = merge
        self._groups_per_step = groups_per_step
        self._sync_every = sync_every
        self._budget = math.ceil(
            (1.0 + staleness_threshold) * groups_per_step * sync_every
        )
        self._max_inflight = max_inflight
        self._partial = partial_rollout
        self._max_drops = max_consecutive_drops
        self._poll_s = poll_s

        self._cv = threading.Condition()
        self._inflight: dict[Any, tuple[int, float]] = {}
        self._ready: deque[RolloutGroup] = deque()
        self._version = start_version
        self._next_index = start_index
        self._carried = 0
        self._kept_since_sync = 0
        self._updates_since_sync = 0
        self._finished = 0
        self._kept = 0
        self._consecutive_drops = 0
        self._submitting = True
        self._stop = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="atomrl-rollout",
            daemon=True,
        )

    # -- rollout thread ------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    @property
    def next_index(self) -> int:
        with self._cv:
            return self._next_index

    @property
    def version(self) -> int:
        """Policy version the rollout weights hold."""
        with self._cv:
            return self._version

    def _keep_rate(self) -> float:
        return (self._kept + 1) / (self._finished + 1)

    def _budget_used(self) -> float:
        expected_in_flight = len(self._inflight) * self._keep_rate()
        return self._carried + self._kept_since_sync + expected_in_flight

    def _can_submit(self) -> bool:
        return (
            self._submitting
            and not self._stop
            and len(self._inflight) < self._max_inflight
            and self._budget_used() < self._budget
        )

    def _run(self) -> None:
        try:
            while True:
                with self._cv:
                    if self._stop:
                        return
                    while self._can_submit():
                        handle = self._backend.submit(self._next_index)
                        self._next_index += 1
                        self._inflight[handle] = (self._version, time.time())
                    handles = list(self._inflight)
                    if not handles:
                        self._cv.wait(self._poll_s)
                        continue
                for handle in self._backend.wait(handles, self._poll_s):
                    rows = self._backend.finish(handle)
                    self._record(handle, rows)
        # Rollout worker transports every termination path to the trainer.
        except BaseException as exc:  # noqa: BLE001 -- boundary
            with self._cv:
                self._error = exc
                self._cv.notify_all()

    def _record(self, handle: Any, rows: Any | None) -> None:
        with self._cv:
            version, submitted_at = self._inflight.pop(handle)
            self._finished += 1
            if rows is None:
                self._consecutive_drops += 1
                if self._max_drops and self._consecutive_drops >= self._max_drops:
                    raise RuntimeError(
                        f"dynamic sampling dropped {self._consecutive_drops} prompt "
                        "groups in a row; check data difficulty / max_num_gen_batches."
                    )
            else:
                self._consecutive_drops = 0
                self._kept += 1
                self._kept_since_sync += 1
                self._ready.append(
                    RolloutGroup(
                        rows, version, self._version, submitted_at, time.time()
                    )
                )
            self._cv.notify_all()

    # -- trainer side --------------------------------------------------------

    def take(
        self, step: int, rollout_version: int
    ) -> tuple[Any, float, dict[str, float]]:
        """Block until a step's worth of groups is ready; return the merged batch.

        Returns ``(batch, gen_s, metrics)``; ``gen_s`` spans the first group's
        submission to the last one's completion. ``step`` is the trainer's
        policy version, so ``step - group.version`` is how many updates behind
        the weights that generated a group are.
        """
        del rollout_version  # the rollout tracks it from after_sync()
        started = time.perf_counter()
        with self._cv:
            while len(self._ready) < self._groups_per_step and self._error is None:
                self._cv.wait(1.0)
            if self._error is not None:
                raise RuntimeError("streaming rollout failed") from self._error
            groups = [self._ready.popleft() for _ in range(self._groups_per_step)]
            queued, in_flight = len(self._ready), len(self._inflight)
            keep_rate = self._kept / max(1, self._finished)
            self._updates_since_sync += 1
        wait_s = time.perf_counter() - started
        staleness = [step - g.version for g in groups]
        metrics = {
            "async/staleness_mean": sum(staleness) / len(staleness),
            "async/staleness_max": float(max(staleness)),
            "async/partial_groups": float(
                sum(g.finished_version != g.version for g in groups)
            ),
            "async/trainer_wait_s": wait_s,
            "async/queued_groups": float(queued),
            "async/inflight_groups": float(in_flight),
            "async/keep_rate": keep_rate,
        }
        gen_s = max(g.finished_at for g in groups) - min(g.submitted_at for g in groups)
        return self._merge([g.rows for g in groups]), gen_s, metrics

    def sync_due(self) -> bool:
        with self._cv:
            return self._updates_since_sync >= self._sync_every

    def before_sync(self) -> float:
        """Make the engine safe to load weights; return the seconds this blocked."""
        started = time.perf_counter()
        with self._cv:
            self._submitting = False
            if not self._partial:
                while self._inflight and self._error is None:
                    self._cv.wait(1.0)
        if self._partial:
            self._backend.pause()
        return time.perf_counter() - started

    def after_sync(self, rollout_version: int) -> None:
        """The rollout now holds ``rollout_version``; start the next period."""
        with self._cv:
            self._version = rollout_version
            self._carried = len(self._ready)
            self._kept_since_sync = 0
            self._updates_since_sync = 0
        if self._partial:
            self._backend.resume()
        with self._cv:
            self._submitting = True
            self._cv.notify_all()

    @contextmanager
    def submissions_paused(self) -> Iterator[None]:
        """Hold new groups back, e.g. while validation shares the engine."""
        with self._cv:
            was = self._submitting
            self._submitting = False
        try:
            yield
        finally:
            with self._cv:
                self._submitting = was
                self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=10.0)


class BackgroundValidation:
    """Runs one validation at a time on a thread while the trainer keeps training.

    ``run(step)`` returns the validation metrics; ``version()`` is the policy
    version the rollout holds, read when validation starts and when it ends.
    """

    def __init__(
        self, run: Callable[[int], dict[str, float]], version: Callable[[], int]
    ) -> None:
        self._run = run
        self._version = version
        self._thread: threading.Thread | None = None
        self._step = 0
        self._versions = (0, 0)
        self._elapsed_s = 0.0
        self._result: dict[str, float] = {}
        self._error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return self._thread is not None

    def start(self, step: int) -> None:
        if self._thread is not None:
            raise RuntimeError(
                "collect() the running validation before starting another"
            )
        self._step = step
        self._result, self._error = {}, None
        self._thread = threading.Thread(
            target=self._target, name="atomrl-validation", daemon=True
        )
        self._thread.start()

    def _target(self) -> None:
        started_version = self._version()
        started = time.perf_counter()
        try:
            self._result = dict(self._run(self._step))
        # Validation thread transports every termination path to collect().
        except BaseException as exc:  # noqa: BLE001 -- boundary
            self._error = exc
        self._elapsed_s = time.perf_counter() - started
        self._versions = (started_version, self._version())

    def collect(self, *, wait: bool) -> tuple[dict[str, float], float]:
        """Metrics of the finished validation and the seconds this blocked.

        Returns ``({}, 0.0)`` when none is running, or when one is and ``wait``
        is false.
        """
        if self._thread is None or (not wait and self._thread.is_alive()):
            return {}, 0.0
        started = time.perf_counter()
        self._thread.join()
        blocked_s = time.perf_counter() - started
        self._thread = None
        if self._error is not None:
            raise RuntimeError(
                f"validation at step {self._step + 1} failed"
            ) from self._error
        if not self._result:
            return {}, blocked_s
        started_version, finished_version = self._versions
        return {
            **self._result,
            "val/global_step": float(self._step + 1),
            "val/started_version": float(started_version),
            "val/finished_version": float(finished_version),
            "val/elapsed_s": self._elapsed_s,
        }, blocked_s
