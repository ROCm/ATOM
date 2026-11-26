# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""One-step off-policy rollout: generate step s+1 while step s trains.

With the rollout on its own GPUs (``controller.ray.rollout.num_workers > 0``)
the trainer does not touch the rollout engine while it computes log-probs and
updates the actor, so the next batch can be generated in that window. The
batch trained at step s then comes from the weights synced after step s-1, one
update behind the trainer; the rollout-correction weights account for that gap.
Weights are synced only after the in-flight batch has finished, so the engine
never loads new weights in the middle of a request.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from typing import Any


class OneStepOffPolicyRollout:
    """Keeps at most one rollout batch in flight on a background thread.

    ``collect(step)`` builds one step's batch; it must not use anything the
    training side mutates while it runs.
    """

    def __init__(self, collect: Callable[[int], Any], total_steps: int) -> None:
        self._collect = collect
        self._total_steps = total_steps
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="atomrl-rollout",
        )
        self._pending: Future | None = None
        self._pending_version = 0

    def submit(self, step: int, rollout_version: int) -> None:
        """Start ``step``'s batch on the weights of policy version ``rollout_version``."""
        if self._pending is not None:
            raise RuntimeError("a rollout batch is already in flight")
        self._pending_version = rollout_version
        self._pending = self._executor.submit(self._timed_collect, step)

    def _timed_collect(self, step: int) -> tuple[Any, float]:
        started = time.perf_counter()
        batch = self._collect(step)
        return batch, time.perf_counter() - started

    def take(
        self, step: int, rollout_version: int
    ) -> tuple[Any, float, dict[str, float]]:
        """Return ``step``'s batch and start generating the next one.

        ``rollout_version`` is the policy version now loaded in the rollout; the
        next batch is generated on it. Returns ``(batch, gen_s, metrics)``:
        ``gen_s`` is how long the batch took to generate; ``async/staleness``
        is how many updates the trainer (at version ``step``) has applied
        since the weights that generated it.
        """
        if self._pending is None:
            raise RuntimeError(f"no rollout batch in flight for step {step}")
        started = time.perf_counter()
        pending, self._pending = self._pending, None
        batch, gen_s = pending.result()
        metrics = {
            "async/staleness": float(step - self._pending_version),
            "async/trainer_wait_s": time.perf_counter() - started,
        }
        if step + 1 < self._total_steps:
            self.submit(step + 1, rollout_version)
        return batch, gen_s, metrics

    def drain(self) -> float:
        """Block until the in-flight batch finishes and return the seconds waited.

        A failure is not raised here; the next ``take`` raises it.
        """
        if self._pending is None:
            return 0.0
        started = time.perf_counter()
        wait([self._pending])
        return time.perf_counter() - started

    def sync_due(self) -> bool:
        return True

    def before_sync(self) -> float:
        """The engine must be idle while it loads the new weights."""
        return self.drain()

    def after_sync(self, rollout_version: int) -> None:
        """Nothing to do: the next batch is submitted by ``take``."""

    @contextmanager
    def submissions_paused(self) -> Iterator[None]:
        """Nothing to hold back: no batch is in flight after a sync."""
        yield

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._pending = None
