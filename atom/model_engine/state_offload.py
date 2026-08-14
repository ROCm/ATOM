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

    `hashes` answers the membership half of `StateGroupPool._resumable_from`,
    and only once loads are wired: that predicate gates this set behind
    `state_pool.STATE_OFFLOAD_LOADS_WIRED`, because a hash no load path can act
    on would end the right-to-left scan on a boundary nothing can deliver. This
    set is populated either way — the spill leg is live — so what the flag
    changes is who may vote on a hit, not what is recorded here.

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


# The connector backends whose worker half builds a `StateOffloadTier`. Only
# the offload backend does; `multi` qualifies when it lists one.
_STATE_TIER_BACKENDS = frozenset({"lmcache_offload"})


def kv_connector_hosts_state_tier(kv_transfer_config) -> bool:
    """Whether the configured KV connector can actually run the state tier.

    The tier is not standalone. Its bytes ride the KV connector's worker half
    (`LMCacheOffloadConnector._maybe_build_state_tier`), its completions ride
    that connector's `get_finished`, and the engine only polls for them when
    `kv_transfer_enabled`. Against any other backend the staging ring would be
    installed, hand out slots, and never get one back -- so the answer here
    gates whether the ring is installed at all, and a false negative (the tier
    stays off) is much cheaper than a false positive (every slot leaks).

    Deliberately a name check rather than a capability probe: this runs in the
    engine process at `BlockManager.__init__`, long before any worker
    connector object exists to ask.
    """
    cfg = kv_transfer_config or {}
    if not isinstance(cfg, dict):
        return False
    name = cfg.get("kv_connector", "moriio")
    if name == "multi":
        subs = cfg.get("connectors") or ()
        return any(
            isinstance(s, dict) and s.get("kv_connector") in _STATE_TIER_BACKENDS
            for s in subs
        )
    return name in _STATE_TIER_BACKENDS


def state_offload_staging_groups() -> int:
    """K: staging *groups* to reserve, or 0 when the tier is off.

    Groups, not entries: a sizing site multiplies by its own
    `entries_per_req` to get rows, because `state_entry_views` is indexed by
    group. Returning entries here would make every caller divide back out.

    One function rather than two `os.environ` reads because the arena sizes
    itself from this and the pool wires itself from this; if they disagreed,
    the arena would be short exactly the rows the spill path addresses.

    Truthiness follows the offload module's own convention
    (`staged_transfer.py:_env_flag`) rather than an
    equality against "1": someone who exports `OFFLOAD_STATE=true` means to
    turn the tier on, and a silent 0 gives them a server that looks healthy
    and never spills.

    Both bad depths warn rather than raise, because this runs during model
    load: a non-integer falls back to 1, and a negative floors to 0. The
    negative case is the louder of the two even though it looks milder --
    0 is what `OFFLOAD_STATE=0` returns, so the tier is off outright while the
    flag says on.

    Stripped, and empty reads as off -- for a default-off feature the
    dangerous direction is the other one: `OFFLOAD_STATE=` (how a shell script
    clears a flag inline) and `OFFLOAD_STATE="off "` both used to turn the
    tier ON, spilling on a server whose operator had just written it off.
    """
    raw_flag = os.environ.get("OFFLOAD_STATE", "0").strip().lower()
    if not raw_flag or raw_flag in ("0", "false", "no", "off"):
        return 0
    raw = os.environ.get("OFFLOAD_STATE_STAGING_GROUPS")
    if raw is None:
        return 1
    try:
        depth = int(raw)
    except ValueError:
        # Warn rather than raise: this runs during model load, and a typo must
        # not be fatal. It must not be silent either -- a mistyped depth buys
        # one group instead of twenty, and the only other symptom is a starved
        # ring 256 dropped spills later.
        logger.warning(
            "state offload: invalid OFFLOAD_STATE_STAGING_GROUPS=%r; using 1",
            raw,
        )
        return 1
    if depth < 0:
        # Same reasoning as the typo above, and a worse outcome: a negative
        # depth turns the whole tier off even though OFFLOAD_STATE says on, so
        # the operator gets the healthy-looking server that never spills. Loud
        # for the same reason `banana` is.
        logger.warning(
            "state offload: negative OFFLOAD_STATE_STAGING_GROUPS=%r disables "
            "the tier entirely; using 0",
            raw,
        )
        return 0
    return depth
