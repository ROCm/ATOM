# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bookkeeping of `Scheduler._offload_prefetch`.

The scheduler allocates blocks for a waiting request and submits its tier
(LMCache) load before the request's turn to prefill. This keeps which
requests hold such blocks and which ones not to try again.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


class OffloadPrefetchState:
    """Prefetched requests, per-request skip memory, a few counters."""

    # Forget skip memory of requests that left `waiting` this often (in ticks).
    PRUNE_TICKS = 256

    def __init__(self) -> None:
        # seq id -> seq. A request stays until it is scheduled for prefill,
        # aborted or otherwise gone.
        self.entries: dict[Any, Any] = {}
        # Requests never to try again (no load, unaligned, taken back, ...).
        self.settled: set[Any] = set()
        # Requests to retry only at this tick (a KV-pressure skip).
        self.retry_at: dict[Any, int] = {}
        # Skips per reason, each request counted once per reason.
        self.skips: Counter = Counter()
        self._counted: set[tuple[Any, str]] = set()
        self.submitted = 0
        self.takebacks = 0
        self.tick = 0

    def skip(self, seq_id: Any, reason: str) -> None:
        key = (seq_id, reason)
        if key not in self._counted:
            self._counted.add(key)
            self.skips[reason] += 1

    def note_submitted(self, seq: Any) -> None:
        self.entries[seq.id] = seq
        self.submitted += 1

    def note_prefill(self, seq_id: Any) -> None:
        """`seq_id` was put on a prefill batch: its prefetch wait is over."""
        self.entries.pop(seq_id, None)

    def note_gone(self, seq_id: Any, **_: Any) -> None:
        self.entries.pop(seq_id, None)

    def prune(self, waiting_ids: set) -> None:
        self.settled &= waiting_ids
        self.retry_at = {k: v for k, v in self.retry_at.items() if k in waiting_ids}
        self._counted = {k for k in self._counted if k[0] in waiting_ids}
