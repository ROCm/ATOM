# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Worker-side admission & lifecycle bookkeeping for DSV4 checkpoint saves.

Two independent resources bound how many terminal checkpoints can be saved
concurrently without ever stalling the running request (see
``dsv4-lmcache-bundle-plan.md`` Phase 2):

* **CSA state-pool slots** — each in-flight save holds one ``v4_state_pool`` slot
  (the @B snapshot produced by ``gather_slot``) until D2H completes.
* **in-flight save credits** — a cap on concurrent background saves (one temp /
  stream budget in V1).

:class:`DSV4CheckpointAdmission` hands out a ``(pool_idx, credit)`` pair or
refuses (``None``) when either resource is exhausted. Refusal means *skip this
checkpoint* — the current request continues unblocked.

:class:`DSV4SwaIoPins` reference-counts physical SWA pages that an in-flight save
is still reading. The SWA allocator must treat ``io_ref > 0`` pages as
non-reusable even after the sequence drops its own reference
(``free_after_prefill_chunk``), so an async gather never races page reuse.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Iterable


class DSV4CheckpointAdmission:
    """Bounded, non-blocking admission for concurrent checkpoint saves."""

    def __init__(self, *, state_pool_size: int, max_inflight_saves: int) -> None:
        self._pool_size = int(state_pool_size)
        self._max_inflight = int(max_inflight_saves)
        if self._pool_size < 0 or self._max_inflight < 0:
            raise ValueError("DSV4CheckpointAdmission: sizes must be >= 0")
        self._lock = threading.Lock()
        self._free_slots: list[int] = list(range(self._pool_size))
        self._inflight: int = 0

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    @property
    def free_slots(self) -> int:
        with self._lock:
            return len(self._free_slots)

    def try_admit(self) -> int | None:
        """Reserve one state-pool slot + one in-flight credit, or return None.

        None => resources exhausted => caller SKIPS this checkpoint (never waits).
        """
        with self._lock:
            if self._inflight >= self._max_inflight or not self._free_slots:
                return None
            self._inflight += 1
            return self._free_slots.pop()

    def complete(self, pool_idx: int) -> None:
        """Release the slot + credit once the background save finishes/aborts."""
        with self._lock:
            if not (0 <= int(pool_idx) < self._pool_size):
                raise ValueError(
                    f"DSV4CheckpointAdmission: bad pool_idx {pool_idx}"
                )
            if pool_idx in self._free_slots:
                raise ValueError(
                    f"DSV4CheckpointAdmission: pool_idx {pool_idx} double-released"
                )
            self._free_slots.append(int(pool_idx))
            self._inflight = max(0, self._inflight - 1)


class DSV4SwaIoPins:
    """Reference counts on physical SWA pages held by in-flight saves."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refs: dict[int, int] = defaultdict(int)

    def pin(self, pages: Iterable[int]) -> None:
        with self._lock:
            for p in pages:
                p = int(p)
                if p < 0:
                    continue  # window-freed / absent slot
                self._refs[p] += 1

    def unpin(self, pages: Iterable[int]) -> None:
        with self._lock:
            for p in pages:
                p = int(p)
                if p < 0:
                    continue
                cur = self._refs.get(p, 0)
                if cur <= 0:
                    raise ValueError(f"DSV4SwaIoPins: page {p} unpinned below zero")
                if cur == 1:
                    del self._refs[p]
                else:
                    self._refs[p] = cur - 1

    def is_pinned(self, page: int) -> bool:
        """The allocator consults this before reusing a physical SWA page."""
        with self._lock:
            return self._refs.get(int(page), 0) > 0

    def pinned_pages(self) -> set[int]:
        with self._lock:
            return set(self._refs)
