# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared, CPU-only policy helpers for value-aware offload save admission."""

from __future__ import annotations

import array
import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from math import ceil
from typing import Any

PrefixDemandKey = tuple[int, bytes]
logger = logging.getLogger("atom")


def _nonnegative_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0 <= value < float("inf"):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_float(name: str, default: float) -> float:
    value = _nonnegative_float(name, default)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_nonnegative_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class SaveAdmissionConfig:
    min_observed_count: int
    aging_weight: float
    release_weight: float
    demand_block_tokens: int
    demand_max_entries: int
    demand_ttl_seconds: float
    max_pinned_ratio: float
    max_pinned_blocks: int | None


@dataclass(frozen=True)
class SaveBlockReservation:
    """One scheduler-local PAGE budget reservation.

    ``block_ids`` are physical IDs from this scheduler's own BlockManager.
    ``estimated_blocks`` accounts for source blocks whose current physical IDs
    are unavailable (notably a finished request after early deallocation).
    The two fields are additive; exact IDs are unioned across reservations and
    leases before the conservative estimate is added.
    """

    sid: str
    seq: Any
    generation: int
    block_ids: frozenset[int]
    estimated_blocks: int
    priority_score: float
    committed: bool


@dataclass(frozen=True)
class SaveCandidate:
    """Layout-neutral input to the save-admission policy."""

    sid: str
    seq: object
    generation: int
    saved: int
    aligned: int
    observed_count: int
    reusable_tokens: int
    enqueued_at: float
    finished: bool
    held_blocks: int

    @property
    def dirty_tokens(self) -> int:
        return max(0, self.aligned - self.saved)


def load_save_admission_config() -> SaveAdmissionConfig:
    """Read the shared priority save-admission configuration."""

    max_pinned_ratio = _nonnegative_float("OFFLOAD_SAVE_MAX_PINNED_RATIO", 0.20)
    if max_pinned_ratio > 0.30:
        raise ValueError("OFFLOAD_SAVE_MAX_PINNED_RATIO must be at most 0.30")
    max_pinned_blocks = _optional_nonnegative_int("OFFLOAD_SAVE_MAX_PINNED_BLOCKS")
    return SaveAdmissionConfig(
        min_observed_count=_nonnegative_int("OFFLOAD_SAVE_MIN_OBSERVED_COUNT", 2),
        aging_weight=_nonnegative_float("OFFLOAD_SAVE_AGING_WEIGHT", 0.01),
        release_weight=_nonnegative_float("OFFLOAD_SAVE_RELEASE_WEIGHT", 1.0),
        demand_block_tokens=_nonnegative_int("OFFLOAD_SAVE_DEMAND_BLOCK_TOKENS", 8192),
        demand_max_entries=_nonnegative_int("OFFLOAD_SAVE_DEMAND_MAX_ENTRIES", 65536),
        demand_ttl_seconds=_positive_float("OFFLOAD_SAVE_DEMAND_TTL_SECONDS", 600),
        max_pinned_ratio=max_pinned_ratio,
        max_pinned_blocks=max_pinned_blocks,
    )


@dataclass
class _PrefixDemandEntry:
    count: int
    last_seen: float


class PrefixDemandTracker:
    """Bounded rank-local demand counts for coarse cumulative prefixes.

    These counts deliberately do not share DP prefix-route hint state. Route
    hints are published only after prefill proves an HBM owner, whereas save
    admission needs request demand at entry, before a prefix is stored.
    """

    def __init__(
        self,
        *,
        block_tokens: int = 8192,
        max_entries: int = 65536,
        ttl_seconds: float = 600.0,
        clock=time.monotonic,
    ) -> None:
        if block_tokens <= 0 or max_entries <= 0:
            raise ValueError("prefix demand block size and capacity must be positive")
        if not 0 < ttl_seconds < float("inf"):
            raise ValueError("prefix demand TTL must be finite and positive")
        self.block_tokens = int(block_tokens)
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._entries: OrderedDict[PrefixDemandKey, _PrefixDemandEntry] = OrderedDict()

    def fingerprints(
        self, token_ids: array.array, prompt_tokens: int
    ) -> tuple[PrefixDemandKey, ...]:
        """Hash cumulative coarse prefixes without retaining token buffers."""

        prompt_tokens = int(prompt_tokens)
        if not 0 <= prompt_tokens <= len(token_ids):
            raise ValueError("prompt length exceeds the supplied token buffer")
        if not isinstance(token_ids, array.array):
            token_ids = array.array("i", token_ids)
        if token_ids.typecode != "i" or token_ids.itemsize != 4:
            raise ValueError("prefix demand expects an int32 token buffer")

        raw = memoryview(token_ids).cast("B")[: prompt_tokens * 4]
        block_bytes = self.block_tokens * 4
        digest = hashlib.blake2b(digest_size=16)
        keys = []
        for end in range(block_bytes, len(raw) + 1, block_bytes):
            digest.update(raw[end - block_bytes : end])
            keys.append((end // 4, digest.digest()))
        if not keys and raw:
            digest.update(raw)
            keys.append((prompt_tokens, digest.digest()))
        return tuple(keys)

    def observe(
        self, token_ids: array.array, prompt_tokens: int, now: float | None = None
    ) -> tuple[PrefixDemandKey, ...]:
        """Record one request and return reusable keys for later scoring."""

        observed_at = self._clock() if now is None else float(now)
        keys = self.fingerprints(token_ids, prompt_tokens)
        for key in keys:
            entry = self._entries.get(key)
            if entry is None or observed_at - entry.last_seen >= self.ttl_seconds:
                entry = _PrefixDemandEntry(count=1, last_seen=observed_at)
                self._entries[key] = entry
            else:
                entry.count += 1
                entry.last_seen = observed_at
            self._entries.move_to_end(key)
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return keys

    def heat(
        self,
        keys: tuple[PrefixDemandKey, ...],
        *,
        max_tokens: int,
        now: float | None = None,
    ) -> tuple[int, int]:
        """Return ``(observed_count, reusable_tokens)`` for the hottest key."""

        checked_at = self._clock() if now is None else float(now)
        reusable_limit = max(0, int(max_tokens))
        if reusable_limit == 0:
            return 0, 0
        best = (0, 0)
        for key in keys:
            entry = self._entries.get(key)
            if entry is None:
                continue
            if checked_at - entry.last_seen >= self.ttl_seconds:
                del self._entries[key]
                continue
            self._entries.move_to_end(key)
            best = max(best, (entry.count, min(int(key[0]), reusable_limit)))
        return best

    def clear(self) -> None:
        self._entries.clear()


class SaveAdmissionMixin:
    """Shared candidate, reservation, and eviction state machine.

    Layout implementations retain only the operations that genuinely depend on
    their source geometry: candidate construction, block accounting, metadata
    construction, and terminal cleanup. Keeping the transition machinery here
    prevents dense PAGE and DSV4 PAGE+SLOT admission from drifting apart.
    """

    def _init_save_admission_state(self, *, max_pending_saves: int) -> None:
        config = load_save_admission_config()
        self._max_pending_saves = int(max_pending_saves)
        self._save_min_observed_count = config.min_observed_count
        self._save_aging_weight = config.aging_weight
        self._save_release_weight = config.release_weight
        self._save_max_pinned_ratio = config.max_pinned_ratio
        self._save_max_pinned_blocks = config.max_pinned_blocks
        self._save_pin_total_blocks = 0
        self._save_pin_budget_blocks: int | None = None
        self._save_block_reservations: dict[str | object, SaveBlockReservation] = {}
        self._prefix_demand = PrefixDemandTracker(
            block_tokens=config.demand_block_tokens,
            max_entries=config.demand_max_entries,
            ttl_seconds=config.demand_ttl_seconds,
        )
        self._save_demand_keys: dict[
            str, tuple[object, tuple[PrefixDemandKey, ...]]
        ] = {}
        self._save_candidate_since: dict[str, tuple[object, float]] = {}
        self._save_candidate_generation: dict[str, tuple[object, int]] = {}
        self._save_candidate_nonce = 0
        self._save_committed: dict[str, object] = {}
        self._finished_save_requests: dict[str, object] = {}
        self._finished_save_failed: set[str] = set()
        self._save_inflight_since: dict[object, float] = {}
        self.total_save_admitted = 0
        self.total_save_budget_rejected = 0
        self.total_save_budget_rejected_blocks = 0
        self.total_save_budget_evicted = 0
        self.total_save_budget_evicted_blocks = 0
        self.total_save_oversized = 0
        self._save_drop_totals = {
            "capacity": 0,
            "low_value": 0,
            "terminal_failure": 0,
            "stale": 0,
        }
        self._save_drop_token_totals = dict.fromkeys(self._save_drop_totals, 0)
        self._block_manager = None

    def bind_block_manager(self, block_manager: Any) -> None:
        if self._block_manager is not None and self._block_manager is not block_manager:
            raise RuntimeError("offload scheduler is already bound to a block manager")
        self._block_manager = block_manager
        total_blocks = int(block_manager.total_allocatable_kv_blocks)
        if total_blocks <= 0:
            raise ValueError("BlockManager must expose a positive KV block capacity")
        ratio_budget = int(total_blocks * self._save_max_pinned_ratio)
        absolute = self._save_max_pinned_blocks
        self._save_pin_total_blocks = total_blocks
        self._save_pin_budget_blocks = (
            ratio_budget if absolute is None else min(ratio_budget, absolute)
        )

    def _reserved_save_resources_allow(self, committed_after: int) -> bool:
        """Layout-specific admission hook evaluated before any mutation."""

        del committed_after
        return True

    def _save_candidate_is_inflight(self, sid: str) -> bool:
        return sid in self._save_inflight

    def _save_candidate_is_busy_for_stats(self, sid: str, seq: object) -> bool:
        return self._save_candidate_is_inflight(sid) or (
            self._save_committed.get(sid) is seq
        )

    def _deferred_free_save_count(self) -> int:
        return len(
            {
                reservation.sid
                for reservation in self._save_block_reservations.values()
                if self._finished_save_requests.get(reservation.sid) is reservation.seq
            }
        )

    @staticmethod
    def _reservation_footprint(reservation: SaveBlockReservation) -> int:
        return len(reservation.block_ids) + int(reservation.estimated_blocks)

    def _reservation_fits(
        self,
        reservation: SaveBlockReservation,
        *,
        victim_keys: tuple[str, ...] = (),
    ) -> bool:
        projected = dict(self._save_block_reservations)
        for key in victim_keys:
            projected.pop(key, None)
        projected[reservation.sid] = reservation
        committed_after = sum(item.committed for item in projected.values())
        if not self._save_count_allows(committed_after):
            return False
        if not self._reserved_save_resources_allow(committed_after):
            return False
        budget = self._save_pin_budget_blocks
        return budget is None or self._save_block_usage(projected)[2] <= budget

    def _try_reserve_save_candidate(
        self,
        candidate: SaveCandidate,
        *,
        allow_eviction: bool,
        now: float | None = None,
    ) -> bool:
        """Atomically reserve resources, replacing only undispatched work."""

        existing = self._save_block_reservations.get(candidate.sid)
        if (
            existing is not None
            and existing.seq is candidate.seq
            and existing.generation == candidate.generation
            and existing.committed
        ):
            return True

        score = self._candidate_priority(
            candidate,
            time.monotonic() if now is None else now,
        )
        reservation = self._candidate_block_reservation(candidate, priority_score=score)
        victims: list[SaveBlockReservation] = []
        if not self._reservation_fits(reservation):
            if allow_eviction:
                eligible = sorted(
                    (
                        item
                        for key, item in self._save_block_reservations.items()
                        if isinstance(key, str)
                        and item.committed
                        and item.priority_score < reservation.priority_score
                        and self._save_committed.get(key) is item.seq
                        and not self._save_candidate_is_inflight(key)
                        and (entry := self._save_tracker.get(key)) is not None
                        and entry[0] is item.seq
                    ),
                    key=lambda item: (
                        item.priority_score,
                        item.sid,
                        item.generation,
                    ),
                )
                for victim in eligible:
                    victims.append(victim)
                    if self._reservation_fits(
                        reservation,
                        victim_keys=tuple(item.sid for item in victims),
                    ):
                        break
                else:
                    victims = []
            if not victims and not self._reservation_fits(reservation):
                footprint = self._reservation_footprint(reservation)
                self.total_save_budget_rejected += 1
                self.total_save_budget_rejected_blocks += footprint
                budget = self._save_pin_budget_blocks
                if budget is not None and footprint > budget:
                    self.total_save_oversized += 1
                return False

        # Apply replacements only after the complete victim set has been shown
        # to fit, so a failed admission cannot destroy useful committed work.
        for victim in victims:
            self.total_save_budget_evicted += 1
            self.total_save_budget_evicted_blocks += self._reservation_footprint(victim)
            self.drop_unadmitted_save(victim.seq, reason="capacity")
        self._save_committed[candidate.sid] = candidate.seq
        self._save_block_reservations[candidate.sid] = reservation
        self.total_save_admitted += 1
        return True

    def _release_save_reservation(
        self, key: str | object, *, owner: object | None = None
    ) -> SaveBlockReservation | None:
        reservation = self._save_block_reservations.get(key)
        if reservation is None or (owner is not None and reservation.seq is not owner):
            return None
        return self._save_block_reservations.pop(key)

    def _candidate_generation_for(self, sid: str, seq: object) -> int:
        record = self._save_candidate_generation.get(sid)
        if record is not None and record[0] is seq:
            return record[1]
        generation = self._save_candidate_nonce
        self._save_candidate_nonce += 1
        self._save_candidate_generation[sid] = (seq, generation)
        return generation

    def _candidate_since_for(self, sid: str, seq: object, now: float) -> float:
        record = self._save_candidate_since.get(sid)
        if record is not None and record[0] is seq:
            return record[1]
        self._save_candidate_since[sid] = (seq, now)
        return now

    def _candidate_demand(
        self,
        sid: str,
        seq: object,
        *,
        aligned: int,
        now: float,
    ) -> tuple[int, int]:
        record = self._save_demand_keys.get(sid)
        if record is None or record[0] is not seq:
            return 0, 0
        return self._prefix_demand.heat(record[1], max_tokens=aligned, now=now)

    def _candidate_priority(self, candidate: SaveCandidate, now: float) -> float:
        chunk = max(1, int(self.chunk_size or 256))
        cost_units = max(1, ceil(candidate.dirty_tokens / chunk))
        reusable = max(candidate.reusable_tokens, chunk)
        benefit = candidate.observed_count * reusable
        release_bonus = (
            self._save_release_weight
            * candidate.held_blocks
            * max(1, int(getattr(self, "virtual_block_size", self.block_size)))
        )
        age = max(0.0, now - candidate.enqueued_at)
        return (benefit + release_bonus) / cost_units + self._save_aging_weight * age

    def _candidate_sort_key(
        self, candidate: SaveCandidate, now: float
    ) -> tuple[float, int, float, str, int]:
        return (
            -self._candidate_priority(candidate, now),
            candidate.dirty_tokens,
            candidate.enqueued_at,
            candidate.sid,
            candidate.generation,
        )

    def _meets_save_value_threshold(self, candidate: SaveCandidate) -> bool:
        return candidate.observed_count >= self._save_min_observed_count

    def _commit_finished_save(self, candidate: SaveCandidate) -> bool:
        if not candidate.finished or not self._meets_save_value_threshold(candidate):
            return False
        entry = self._save_tracker.get(candidate.sid)
        if entry is None or entry[0] is not candidate.seq:
            return False
        return self._try_reserve_save_candidate(candidate, allow_eviction=True)

    def _forget_save_candidate(self, sid: str, seq: object) -> None:
        for mapping in (
            self._save_demand_keys,
            self._save_candidate_since,
            self._save_candidate_generation,
        ):
            record = mapping.get(sid)
            if record is not None and record[0] is seq:
                mapping.pop(sid, None)

    def _drop_layout_save_candidate(self, sid: str, seq: object) -> None:
        """Remove layout-specific state for an operation that never dispatched."""

        del sid, seq

    def drop_unadmitted_save(self, seq: object, *, reason: str) -> bool:
        """Atomically remove a candidate that no worker can be reading."""

        sid = str(seq.id)
        if self._save_candidate_is_inflight(sid):
            return False
        entry = self._save_tracker.get(sid)
        if entry is None or entry[0] is not seq:
            return False
        candidate = self._build_save_candidate(sid)
        dropped_tokens = 0 if candidate is None else candidate.dirty_tokens
        self._save_tracker.pop(sid, None)
        if self._save_committed.get(sid) is seq:
            self._save_committed.pop(sid, None)
        self._release_save_reservation(sid, owner=seq)
        if self._finished_save_requests.get(sid) is seq:
            self._finished_save_requests.pop(sid, None)
        self._finished_save_failed.discard(sid)
        self._drop_layout_save_candidate(sid, seq)
        self._forget_save_candidate(sid, seq)
        if reason not in self._save_drop_totals:
            reason = "stale"
        self._save_drop_totals[reason] += 1
        self._save_drop_token_totals[reason] += dropped_tokens
        logger.info(
            "LMCache offload: dropped unadmitted save req=%s reason=%s tokens=%d",
            sid,
            reason,
            dropped_tokens,
        )
        return True

    def _settle_finished_save(self, sid: str, *, failed: bool = False) -> None:
        seq = self._finished_save_requests.get(sid)
        if (
            seq is None
            or self._save_candidate_is_inflight(sid)
            or sid in self._save_committed
        ):
            return
        if failed:
            self._finished_save_failed.add(sid)
        if sid in self._finished_save_failed:
            self.drop_unadmitted_save(seq, reason="terminal_failure")
            return
        candidate = self._build_save_candidate(sid)
        if candidate is None:
            entry = self._save_tracker.get(sid)
            if entry is not None and entry[0] is seq:
                self._save_tracker.pop(sid, None)
            self._finished_save_requests.pop(sid, None)
            self._finished_save_failed.discard(sid)
            self._drop_layout_save_candidate(sid, seq)
            self._forget_save_candidate(sid, seq)
            return
        if self._commit_finished_save(candidate):
            return
        reason = (
            "low_value"
            if not self._meets_save_value_threshold(candidate)
            else "capacity"
        )
        self.drop_unadmitted_save(seq, reason=reason)

    def _admit_and_emit_save_candidates(
        self, metadata: Any, loading_sids: set[str]
    ) -> None:
        """Run one priority-admission pass and emit committed candidates."""

        if not self._do_save:
            return
        now = time.monotonic()
        for sid, seq in list(self._save_committed.items()):
            entry = self._save_tracker.get(sid)
            if entry is None or entry[0] is not seq:
                self._save_committed.pop(sid, None)
                self._release_save_reservation(sid, owner=seq)
                continue
            candidate = self._build_save_candidate(sid, now=now)
            if candidate is None:
                self._save_committed.pop(sid, None)
                self._release_save_reservation(sid, owner=seq)
                self._settle_finished_save(sid)

        candidates = [
            candidate
            for sid in self._save_tracker
            if (candidate := self._build_save_candidate(sid, now=now)) is not None
            and self._meets_save_value_threshold(candidate)
        ]
        candidates.sort(key=lambda candidate: self._candidate_sort_key(candidate, now))

        for candidate in candidates:
            sid = candidate.sid
            if sid in self._save_committed or self._save_candidate_is_inflight(sid):
                continue
            if sid in self._reqs_need_recv or sid in loading_sids:
                continue
            if not self._try_reserve_save_candidate(
                candidate,
                allow_eviction=True,
                now=now,
            ):
                self.drop_unadmitted_save(candidate.seq, reason="capacity")

        committed = []
        for sid, seq in list(self._save_committed.items()):
            candidate = self._build_save_candidate(sid, now=now)
            if candidate is None or candidate.seq is not seq:
                self._save_committed.pop(sid, None)
                self._release_save_reservation(sid, owner=seq)
                continue
            committed.append(candidate)
        committed.sort(key=lambda candidate: self._candidate_sort_key(candidate, now))
        for candidate in committed:
            sid = candidate.sid
            if sid in self._reqs_need_recv or sid in loading_sids:
                continue
            if self._save_candidate_is_inflight(sid):
                continue
            self._emit_save_candidate(metadata, candidate)

    def get_statistics(self) -> dict[str, int | float]:
        """Add layout-neutral save-admission gauges to transfer counters."""

        statistics = super().get_statistics()
        now = time.monotonic()
        candidates = []
        for sid, entry in self._save_tracker.items():
            seq = entry[0]
            if self._save_candidate_is_busy_for_stats(sid, seq):
                continue
            candidate = self._build_save_candidate(sid, now=now)
            if candidate is not None:
                candidates.append(candidate)

        candidate_scores = [
            self._candidate_priority(candidate, now) for candidate in candidates
        ]
        reserved_blocks, pinned_blocks, budget_used = self._save_block_usage()
        budget = self._save_pin_budget_blocks
        total_blocks = self._save_pin_total_blocks
        block_size = int(getattr(self, "virtual_block_size", self.block_size))
        statistics.update(
            save_candidates=len(candidates),
            save_candidates_finished=sum(
                candidate.finished for candidate in candidates
            ),
            save_committed=len(self._save_committed),
            save_admitted=self.total_save_admitted,
            save_candidate_wait_seconds=max(
                (max(0.0, now - candidate.enqueued_at) for candidate in candidates),
                default=0.0,
            ),
            save_priority_score=max(candidate_scores, default=0.0),
            save_inflight_wait_seconds=max(
                (
                    max(0.0, now - started)
                    for started in self._save_inflight_since.values()
                ),
                default=0.0,
            ),
            save_pinned_blocks=pinned_blocks,
            save_pinned_tokens=pinned_blocks * block_size,
            save_pin_budget_blocks=0 if budget is None else budget,
            save_reserved_blocks=reserved_blocks,
            save_pinned_ratio=(budget_used / total_blocks if total_blocks > 0 else 0.0),
            save_budget_available_blocks=(
                0 if budget is None else max(0, budget - budget_used)
            ),
            save_budget_rejected=self.total_save_budget_rejected,
            save_budget_rejected_blocks=self.total_save_budget_rejected_blocks,
            save_budget_evicted=self.total_save_budget_evicted,
            save_budget_evicted_blocks=self.total_save_budget_evicted_blocks,
            save_oversized=self.total_save_oversized,
            deferred_free_requests=self._deferred_free_save_count(),
        )
        for reason, total in self._save_drop_totals.items():
            statistics[f"save_dropped_{reason}"] = total
            statistics[f"save_dropped_tokens_{reason}"] = self._save_drop_token_totals[
                reason
            ]
        return statistics


__all__ = [
    "PrefixDemandKey",
    "PrefixDemandTracker",
    "SaveAdmissionConfig",
    "SaveAdmissionMixin",
    "SaveBlockReservation",
    "SaveCandidate",
    "load_save_admission_config",
]
