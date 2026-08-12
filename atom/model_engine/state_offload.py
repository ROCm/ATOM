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

from collections import deque


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
        if h < 0 or not self._free_slots:
            self.spills_dropped += 1
            return -1
        slot = self._free_slots.popleft()
        self._pending.append((h, slot))
        self.spills_requested += 1
        return slot

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
