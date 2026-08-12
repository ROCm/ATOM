# SPDX-License-Identifier: MIT
"""Scheduler-side bookkeeping for the state-cache offload tier.

Two things live here and neither touches a device: the set of hashes believed
to be in LMCache, and the bounded queue of groups waiting to be read out of
HBM. The bytes and the transfers are the worker's business
(`kv_transfer/offload/state_tier.py`).

Why a plain in-memory set is right, and persisting it would be harmful:
`LocalDiskBackend.__init__` starts from an empty dict and never scans its
directory, so after a restart LMCache does not recognize its own files. An
index recovered from disk would be a pure false-positive generator. The index
and the bytes share one server lifetime — `LMCacheEngineBuilder.get_or_create`
runs inside `register_kv_caches` at model load, so the LMCache engine cannot
restart without the worker restarting.
"""

import logging
import os
from collections import deque

logger = logging.getLogger("atom")

# How many spills may be dropped back to back, with no slot coming back in
# between, before the ring is called starved rather than merely busy. A slot
# returns only once the worker's D2H has landed, so a burst of evictions can
# legitimately drop spills for a step or two; this is set an order of magnitude
# above any such burst so that crossing it means the slots are never coming
# back, not that the step was heavy.
_STARVATION_DROP_THRESHOLD = 256


class StateOffloadIndex:
    """What has been spilled, and what is queued to be.

    `hashes` answers the membership half of `StateGroupPool._resumable_from`.
    It is deliberately optimistic: LMCache's own LRU can drop bytes at any
    time, so a hash here means "was spilled once", never "is still there".
    The false positive costs one lookup and a park/unpark and is handled by
    the `failed_loading` path, which calls `forget`.
    """

    def __init__(self, staging_depth: int, kv_offload_enabled: bool) -> None:
        self.staging_depth = max(0, int(staging_depth))
        # Orphaned checkpoints (`unindex`) are worth spilling only when the KV
        # prefix can also come back: `resumable_hit` scans `block_hashes`, which
        # `can_allocate` builds from HBM `kv.lookup` hits only. With KV offload
        # off, a hash whose KV left HBM never reappears and the bytes are wasted.
        self.kv_offload_enabled = bool(kv_offload_enabled)
        self.hashes: set[int] = set()
        self._free_slots: deque[int] = deque(range(self.staging_depth))
        self._pending: deque[tuple[int, int]] = deque()
        self.spills_requested = 0
        self.spills_dropped = 0
        self.loads_attempted = 0
        self.loads_failed = 0
        # Drops since the last `release_staging`. A full ring is normal
        # backpressure only while slots are still coming back; if none has come
        # back in this many attempts the consumer has stopped draining and the
        # ring is starved for good. Reset by `release_staging`.
        self._consecutive_drops = 0
        self._warned_starved = False

    @property
    def enabled(self) -> bool:
        return self.staging_depth > 0

    def request_spill(self, h: int, group: int) -> int:
        """Reserve a staging slot for `group`, or -1 if the ring is full.

        Called from `pop()` on the scheduler's critical path, so this does no
        work beyond a deque pop. The caller copies `group` into the returned
        slot on the compute stream and `pop()` hands the original out
        immediately: spilling by copy rather than by pin, because `pop()` is
        called precisely when there is no free group to withhold.
        """
        # A negative hash is a caller bug, not a dropped spill: `_spill` already
        # refuses a group with no checkpoint, so nothing reaches here with h<0
        # in correct operation. Keep the guard, but leave `spills_dropped` for
        # real drops so the counter stays a clean backpressure signal.
        if h < 0:
            return -1
        if not self._free_slots:
            self.spills_dropped += 1
            self._note_drop()
            return -1
        slot = self._free_slots.popleft()
        self._pending.append((h, slot))
        self.spills_requested += 1
        return slot

    def _note_drop(self) -> None:
        """Warn once when drops stop looking like a burst and start looking
        like a consumer that has gone away."""
        self._consecutive_drops += 1
        if self._warned_starved or self._consecutive_drops < _STARVATION_DROP_THRESHOLD:
            return
        self._warned_starved = True
        logger.warning(
            "State offload staging ring is starved: %d spills dropped in a row "
            "with no slot released (staging_depth=%d). The consumer is very "
            "likely not calling StateGroupPool.take_spill_copies() every step, "
            "or not calling StateOffloadIndex.release_staging(slot) once the "
            "bytes are safe. Every further spill will be dropped until it does.",
            self._consecutive_drops,
            self.staging_depth,
        )

    def take_pending(self) -> list[tuple[int, int]]:
        """Drain the queue as `(hash, staging_slot)` pairs."""
        out = list(self._pending)
        self._pending.clear()
        return out

    def confirm_spill(self, h: int) -> None:
        """Index `h` once its bytes reached LMCache."""
        self.hashes.add(h)

    def release_staging(self, slot: int) -> None:
        if 0 <= slot < self.staging_depth and slot not in self._free_slots:
            self._free_slots.append(slot)
            # The ring is moving again, so any drops so far were a burst.
            self._consecutive_drops = 0

    def forget(self, h: int) -> None:
        """Drop a hash whose load failed, so the next request does not retry."""
        self.hashes.discard(h)

    def stats(self) -> dict[str, int]:
        return {
            "spills_requested": self.spills_requested,
            "spills_dropped": self.spills_dropped,
            "loads_attempted": self.loads_attempted,
            "loads_failed": self.loads_failed,
            "indexed": len(self.hashes),
        }


def state_offload_staging_groups() -> int:
    """K: staging *groups* to reserve, or 0 when the tier is off.

    Groups, not entries: a sizing site multiplies by its own
    `entries_per_req` to get rows, because `state_entry_views` is indexed by
    group. Returning entries here would make every caller divide back out.

    One function rather than two `os.environ` reads because the arena sizes
    itself from this and the pool wires itself from this; if they disagreed,
    the arena would be short exactly the rows the spill path addresses.
    """
    if os.environ.get("OFFLOAD_STATE", "0") != "1":
        return 0
    try:
        return max(0, int(os.environ.get("OFFLOAD_STATE_STAGING_GROUPS", "1")))
    except ValueError:
        return 1
