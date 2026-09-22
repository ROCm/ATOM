# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MP-only PAGE save ordering and pin-budget accounting."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from atom.kv_transfer.disaggregation.types import SaveOperationId

AdmissionResult = Literal["admitted", "busy"]


@dataclass(frozen=True)
class _Reservation:
    block_ids: frozenset[int]
    estimated_blocks: int


class MPSaveAdmission:
    """Bound MP save pins without changing standalone offload schedulers.

    The policy is opt-in through ``lmcache.mp.max_pinned_save_blocks``.  When
    enabled, finished requests are considered first because admitting them can
    return their remaining PAGEs to the allocator.  No candidate owns or pins
    a PAGE until the normal MP save path successfully reserves an operation.
    """

    CONFIG_KEY = "lmcache.mp.max_pinned_save_blocks"

    def __init__(
        self,
        max_pinned_blocks: int | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_pinned_blocks is not None and max_pinned_blocks <= 0:
            raise ValueError(f"{self.CONFIG_KEY} must be a positive integer")
        self.max_pinned_blocks = max_pinned_blocks
        self._clock = clock
        self._candidate_since: dict[str, tuple[object, float]] = {}
        self._reservations: dict[SaveOperationId, _Reservation] = {}
        self.rejected = 0
        self.oversized = 0

    @classmethod
    def from_extra_config(cls, extra: dict[str, Any]) -> MPSaveAdmission:
        configured = extra.get(cls.CONFIG_KEY)
        if configured is None:
            return cls(None)
        if isinstance(configured, bool):
            raise TypeError(f"{cls.CONFIG_KEY} must be a positive integer")
        try:
            value = int(configured)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{cls.CONFIG_KEY} must be a positive integer") from exc
        return cls(value)

    @property
    def enabled(self) -> bool:
        return self.max_pinned_blocks is not None

    @property
    def reserved_blocks(self) -> int:
        exact: set[int] = set()
        estimated = 0
        for reservation in self._reservations.values():
            exact.update(reservation.block_ids)
            estimated += reservation.estimated_blocks
        return len(exact) + estimated

    def forget_candidate(self, sid: str, seq: object) -> None:
        record = self._candidate_since.get(sid)
        if record is not None and record[0] is seq:
            self._candidate_since.pop(sid, None)

    def ordered_sids(
        self,
        tracker: dict[str, list],
        frontier: Callable[[object], int],
    ) -> list[str]:
        """Return MP candidates in release-oriented priority order."""

        if not self.enabled:
            return list(tracker)

        now = self._clock()
        live = {(sid, id(entry[0])) for sid, entry in tracker.items()}
        for sid, (seq, _since) in list(self._candidate_since.items()):
            if (sid, id(seq)) not in live:
                self._candidate_since.pop(sid, None)

        ranked: list[tuple[tuple[int, int, int, float, int], str]] = []
        inactive: list[str] = []
        for position, (sid, entry) in enumerate(tracker.items()):
            seq, saved = entry
            aligned = int(frontier(seq))
            if aligned <= int(saved):
                inactive.append(sid)
                continue
            record = self._candidate_since.get(sid)
            if record is None or record[0] is not seq:
                record = (seq, now)
                self._candidate_since[sid] = record
            finished_blocks = getattr(seq, "_offload_finished_block_ids", None)
            finished = finished_blocks is not None
            held_blocks = len(finished_blocks or ())
            dirty_tokens = aligned - int(saved)
            ranked.append(
                (
                    (
                        -int(finished),
                        -held_blocks,
                        -dirty_tokens,
                        record[1],
                        position,
                    ),
                    sid,
                )
            )
        ranked.sort(key=lambda item: item[0])
        return [sid for _key, sid in ranked] + inactive

    def reserve(
        self,
        operation: SaveOperationId,
        block_ids: list[int],
        *,
        saved: int,
        aligned: int,
        block_size: int,
    ) -> AdmissionResult:
        if not self.enabled:
            return "admitted"
        if operation in self._reservations:
            return "admitted"

        start = max(0, int(saved) // int(block_size))
        end = max(start, -(-int(aligned) // int(block_size)))
        source = block_ids[start : min(end, len(block_ids))]
        reservation = _Reservation(
            block_ids=frozenset(
                block_id
                for raw_block_id in source
                if (block_id := int(raw_block_id)) >= 0
            ),
            estimated_blocks=max(0, end - start - len(source)),
        )
        candidate_blocks = len(reservation.block_ids) + reservation.estimated_blocks
        assert self.max_pinned_blocks is not None
        if candidate_blocks > self.max_pinned_blocks:
            self.oversized += 1
            # A hard rejection would keep a finished request deferred forever.
            # Admit one oversized operation only when it is the sole owner of
            # the MP save budget, preserving progress while still serializing
            # it against every other save.
            if self._reservations:
                self.rejected += 1
                return "busy"
            self._reservations[operation] = reservation
            return "admitted"

        projected = dict(self._reservations)
        projected[operation] = reservation
        exact: set[int] = set()
        estimated = 0
        for item in projected.values():
            exact.update(item.block_ids)
            estimated += item.estimated_blocks
        if len(exact) + estimated > self.max_pinned_blocks:
            self.rejected += 1
            return "busy"

        self._reservations[operation] = reservation
        return "admitted"

    def source_safe(
        self,
        operation: SaveOperationId,
        remaining_block_ids: set[int],
    ) -> None:
        reservation = self._reservations.get(operation)
        if reservation is None:
            return
        remaining = reservation.block_ids & remaining_block_ids
        if remaining or reservation.estimated_blocks:
            self._reservations[operation] = _Reservation(
                frozenset(remaining), reservation.estimated_blocks
            )
        else:
            self._reservations.pop(operation, None)

    def release(self, operation: SaveOperationId | None) -> None:
        if operation is not None:
            self._reservations.pop(operation, None)


__all__ = ["AdmissionResult", "MPSaveAdmission"]
