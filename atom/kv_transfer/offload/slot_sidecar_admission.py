# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded, thread-safe admission for SLOT sidecar staging rows."""

from __future__ import annotations

import heapq
from numbers import Integral
import threading


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


class SlotSidecarAdmission:
    """Allocate a bounded set of staging IDs without blocking.

    The smallest free ID is always returned first.  An ID remains acquired until
    the caller explicitly releases it; this class does not observe GPU work or
    synchronize streams.  Callers must synchronize the associated transfer
    before release.  Exhausting the pool returns ``None``.
    """

    def __init__(self, num_slots: int) -> None:
        capacity = _integer("num_slots", num_slots)
        if capacity <= 0:
            raise ValueError(f"num_slots must be > 0, got {capacity}")

        self._capacity = capacity
        self._free_ids = list(range(capacity))
        self._acquired = [False] * capacity
        self._quarantined = [False] * capacity
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def num_free(self) -> int:
        with self._lock:
            return len(self._free_ids)

    def try_acquire(self) -> int | None:
        """Return the smallest available ID, or ``None`` when full."""
        with self._lock:
            if not self._free_ids:
                return None
            slot_id = heapq.heappop(self._free_ids)
            self._acquired[slot_id] = True
            return slot_id

    def release(self, slot_id: int) -> None:
        """Return an ID after the caller has synchronized its associated work."""
        normalized_id = _integer("slot id", slot_id)
        if not 0 <= normalized_id < self._capacity:
            raise ValueError(
                f"slot id {normalized_id} outside pool [0, {self._capacity})"
            )

        with self._lock:
            if self._quarantined[normalized_id]:
                raise ValueError(f"slot id {normalized_id} is quarantined")
            if not self._acquired[normalized_id]:
                raise ValueError(f"slot id {normalized_id} is not acquired")
            self._acquired[normalized_id] = False
            heapq.heappush(self._free_ids, normalized_id)

    def quarantine(self, slot_id: int) -> None:
        """Permanently remove an acquired ID whose GPU work is not confirmed."""
        normalized_id = _integer("slot id", slot_id)
        if not 0 <= normalized_id < self._capacity:
            raise ValueError(
                f"slot id {normalized_id} outside pool [0, {self._capacity})"
            )

        with self._lock:
            if self._quarantined[normalized_id]:
                return
            if not self._acquired[normalized_id]:
                raise ValueError(f"slot id {normalized_id} is not acquired")
            self._acquired[normalized_id] = False
            self._quarantined[normalized_id] = True
