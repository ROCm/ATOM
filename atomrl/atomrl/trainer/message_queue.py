# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Thread-safe sample buffer between Rollouter and Trainer.

In the fully-async architecture, the Rollouter produces ``DataProto``
samples one at a time and pushes them into the queue.  The Trainer
pulls ``require_batches * mini_batch_size`` samples and assembles them
into a training batch.

Freshness control: each sample carries a ``param_version`` tag indicating
which parameter snapshot it was generated from.  Stale samples (version
older than ``current_version - 1``) are counted and optionally dropped
when the staleness threshold is exceeded.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

from atomrl.core.protocol import DataProto

logger = logging.getLogger(__name__)


@dataclass
class SampleItem:
    """Single rollout sample with generation metadata."""

    data: DataProto
    param_version: int = 0
    created_at: float = field(default_factory=time.time)


class AsyncMessageQueue:
    """Bounded, thread-safe queue for streaming samples between processes.

    Attributes:
        maxsize: Maximum number of ``SampleItem``s buffered.
        staleness_threshold: Maximum fraction of stale samples allowed
            in a training batch.  0 means fully synchronous.
    """

    def __init__(
        self,
        maxsize: int = 64,
        staleness_threshold: float = 0.0,
    ) -> None:
        self._queue: queue.Queue[SampleItem] = queue.Queue(maxsize=maxsize)
        self._staleness_threshold = staleness_threshold
        self._current_param_version = 0
        self._lock = threading.Lock()

        self._total_produced = 0
        self._total_consumed = 0
        self._stale_consumed = 0

    @property
    def current_param_version(self) -> int:
        return self._current_param_version

    @current_param_version.setter
    def current_param_version(self, v: int) -> None:
        with self._lock:
            self._current_param_version = v

    def put(self, item: SampleItem, timeout: float | None = None) -> None:
        """Add a sample to the queue (blocks if full)."""
        self._queue.put(item, timeout=timeout)
        with self._lock:
            self._total_produced += 1

    def get_batch(
        self,
        batch_size: int,
        timeout: float = 300.0,
    ) -> list[SampleItem]:
        """Collect ``batch_size`` samples, respecting staleness threshold.

        Blocks until enough fresh + allowed-stale samples are available
        or ``timeout`` is reached.

        Returns:
            List of ``SampleItem``s of length ``batch_size``.

        Raises:
            TimeoutError: If batch cannot be assembled within ``timeout``.
        """
        collected: list[SampleItem] = []
        stale_count = 0
        max_stale = int(batch_size * self._staleness_threshold)
        deadline = time.monotonic() + timeout

        while len(collected) < batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"AsyncMessageQueue.get_batch timed out: "
                    f"got {len(collected)}/{batch_size} in {timeout}s"
                )
            try:
                item = self._queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue

            is_stale = item.param_version < self._current_param_version
            if is_stale and stale_count >= max_stale:
                logger.debug(
                    "Dropping stale sample (version=%d, current=%d)",
                    item.param_version,
                    self._current_param_version,
                )
                continue

            collected.append(item)
            if is_stale:
                stale_count += 1

            with self._lock:
                self._total_consumed += 1
                if is_stale:
                    self._stale_consumed += 1

        return collected

    def qsize(self) -> int:
        return self._queue.qsize()

    def metrics(self) -> dict[str, float]:
        with self._lock:
            return {
                "queue/size": self._queue.qsize(),
                "queue/total_produced": self._total_produced,
                "queue/total_consumed": self._total_consumed,
                "queue/stale_consumed": self._stale_consumed,
                "queue/stale_ratio": (
                    self._stale_consumed / max(1, self._total_consumed)
                ),
            }
