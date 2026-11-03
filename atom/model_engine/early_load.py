# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bookkeeping of the scheduler's early tier loads.

`Scheduler._early_load_tier_prefixes` allocates blocks for a waiting request and
submits its LMCache load before the request's turn to prefill. This module keeps
what it needs to bound that (the requests holding such blocks) and to report on
it (one INFO line per interval).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

logger = logging.getLogger("atom")

# Skips that are about KV pressure; every other skip is a condition the request
# does not meet.
LIMIT_REASONS = ("watermark",)


class EarlyLoadState:
    """Requests holding early-loaded blocks, per-request skip memory, counters."""

    LOG_INTERVAL_S = 60.0
    # Forget skip memory of requests that left `waiting` this often (in ticks).
    PRUNE_TICKS = 256

    def __init__(self) -> None:
        # seq id -> (seq, monotonic time its load was submitted). A request
        # stays until it is scheduled for prefill, aborted or otherwise gone.
        self.entries: dict[Any, tuple[Any, float]] = {}
        # Requests never to try again (no load, unaligned, taken back, ...).
        self.settled: set[Any] = set()
        # Requests to retry only at this tick (a pressure skip).
        self.retry_at: dict[Any, int] = {}
        # (seq id, reason) pairs already counted: a request counts once per
        # reason however many ticks it is skipped for it.
        self._counted: set[tuple[Any, str]] = set()
        self.skips: Counter = Counter()
        self.submitted = 0
        self.submitted_total = 0
        self.takebacks = 0
        self.aborted = 0
        self.prefilled = 0
        self.wait_sum_s = 0.0
        self.wait_n = 0
        self.tick = 0
        self._last_log = time.monotonic()

    # -- counters -------------------------------------------------------------

    def skip(self, seq_id: Any, reason: str) -> None:
        key = (seq_id, reason)
        if key not in self._counted:
            self._counted.add(key)
            self.skips[reason] += 1

    def note_submitted(self, seq: Any) -> None:
        self.entries[seq.id] = (seq, time.monotonic())
        self.submitted += 1
        self.submitted_total += 1

    def note_prefill(self, seq_id: Any) -> None:
        """`seq_id` was put on a prefill batch: its early-load wait is over."""
        entry = self.entries.pop(seq_id, None)
        if entry is not None:
            self.wait_sum_s += time.monotonic() - entry[1]
            self.wait_n += 1
            self.prefilled += 1

    def note_gone(self, seq_id: Any, *, aborted: bool = False) -> None:
        if self.entries.pop(seq_id, None) is not None and aborted:
            self.aborted += 1

    def held_blocks(self) -> int:
        return sum(len(seq.block_table) for seq, _ in self.entries.values())

    def prune(self, waiting_ids: set) -> None:
        self.settled &= waiting_ids
        self.retry_at = {k: v for k, v in self.retry_at.items() if k in waiting_ids}
        self._counted = {k for k in self._counted if k[0] in waiting_ids}

    # -- reporting ------------------------------------------------------------

    def maybe_log(self, total_blocks: int) -> None:
        now = time.monotonic()
        if now - self._last_log < self.LOG_INTERVAL_S:
            return
        self._last_log = now
        if not (self.submitted or self.skips or self.entries or self.takebacks):
            return
        limit = {r: self.skips[r] for r in LIMIT_REASONS}
        cond = {r: n for r, n in sorted(self.skips.items()) if r not in LIMIT_REASONS}
        held = self.held_blocks()
        frac = held / total_blocks if total_blocks > 0 else 0.0
        wait_ms = 1e3 * self.wait_sum_s / self.wait_n if self.wait_n else 0.0
        logger.info(
            "[EARLY-LOAD] last %.0fs: submitted=%d (total %d) "
            "skipped_by_limit=%s skipped_by_condition=%s taken_back=%d aborted=%d "
            "| held_blocks=%d (%.1f%% of %d) held_requests=%d "
            "| to_prefill n=%d avg_wait=%.0fms",
            self.LOG_INTERVAL_S,
            self.submitted,
            self.submitted_total,
            limit,
            cond,
            self.takebacks,
            self.aborted,
            held,
            100.0 * frac,
            total_blocks,
            len(self.entries),
            self.wait_n,
            wait_ms,
        )
        self.skips.clear()
        self.submitted = 0
        self.takebacks = 0
        self.aborted = 0
        self.wait_sum_s = 0.0
        self.wait_n = 0
        self.prefilled = 0
