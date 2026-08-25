# SPDX-License-Identifier: MIT
"""Scheduler-side bookkeeping for the state-cache offload tier.

Two things live here and neither touches a device: the set of hashes believed
to be in LMCache, and the bounded queue of groups waiting to be read out of
HBM. The bytes and the transfers are the worker's business
(`kv_transfer/offload/state_tier.py`).

The set is in memory and never persisted: `LocalDiskBackend.__init__` starts
from an empty dict and never scans its directory, so an index recovered from
disk would be a pure false-positive generator. Index and bytes share one
server lifetime.
"""

import logging
import os
import time
from collections import deque

logger = logging.getLogger("atom")

# How often `reclaim_stale_slots` actually scans, regardless of how often it is
# called. The scan is O(staging_depth) and self-throttled so wiring it into a
# per-step drain loop costs nothing when nothing is stuck.
_RECLAIM_SCAN_INTERVAL_S = 5.0

# Consecutive dropped spills, with no slot coming back, before the ring is
# called starved rather than busy. A slot returns only once the worker's D2H
# lands, so a burst of evictions can legitimately drop for a step or two; an
# order of magnitude above that, so crossing it means the slots are gone.
_STARVATION_DROP_THRESHOLD = 256


class StateOffloadIndex:
    """What has been spilled, what is queued to be, and what is coming back.

    `hashes` answers the membership half of `StateGroupPool._resumable_from`.
    It is deliberately optimistic: LMCache's own LRU can drop bytes at any
    time, so a hash here means "was spilled once", never "is still there". The
    false positive costs one lookup and a park/unpark, and is handled by the
    `failed_loading` path (`fail_load` -> `forget`).
    """

    def __init__(self, staging_depth: int) -> None:
        self.staging_depth = max(0, int(staging_depth))
        # Orphaned checkpoints (`unindex`) are worth spilling only when the KV
        # prefix can also come back: `resumable_hit` scans `block_hashes`, which
        # `can_allocate` builds from HBM `kv.lookup` hits only. With KV offload
        # off, a hash whose KV left HBM never reappears and the bytes are wasted.
        #
        # In production today this is always True: `BlockManager.__init__`
        # refuses to construct the index at all when the connector does not
        # host the tier, so the only path here has already established it. The
        # flag is kept, and the `unindex` gate with it, because the two
        # conditions are independent in principle -- a future connector could
        # host the tier's transport without offloading KV -- and because the
        # failure it prevents (spending LMCache capacity on hashes no load can
        # ever reach) is silent. Its False arm is covered by tests only.
        self.hashes: set[int] = set()
        self._free_slots: deque[int] = deque(range(self.staging_depth))
        self._pending: deque[tuple[int, int]] = deque()
        # req_id -> hash, for loads offered and not yet settled. Keyed by
        # request because that is what comes back, on the same
        # `finished_loading`/`failed_loading` channel a KV load uses. (Spill
        # reports are keyed by hash and slot -- their owner is long gone.)
        self.pending_loads: dict = {}
        # Read once, here, rather than per call: `_resumable_from` consults it
        # inside a per-block scan.
        self.spills_requested = 0
        self.spills_dropped = 0
        # Evictions that deliberately did not spill because the group was
        # going to a load. Apart from `spills_dropped`: a deeper ring does not
        # fix this one, it is the price of the load.
        self.spills_forgone = 0
        self.loads_attempted = 0
        self.loads_completed = 0
        self.loads_failed = 0
        # Drops since the last `release_staging`. A full ring is normal
        # backpressure only while slots come back; none in this many attempts
        # means the consumer stopped draining.
        self._consecutive_drops = 0
        self._warned_starved = False
        # slot -> time.monotonic() when it was reserved by `request_spill`.
        # A slot's only way home is `release_staging`, driven by the worker's
        # spill report. If that report never comes (the worker abandoned the
        # transfer, or its completion was lost the same way a KV save's can be),
        # the slot is pinned forever and the ring silently starves. This stamp
        # lets `reclaim_stale_slots` return a slot the report has clearly
        # abandoned. Reclaimed slots (`_slots_reclaimed`) are counted so the
        # symptom is visible rather than a silent throughput cliff.
        self._slot_reserved_at: dict[int, float] = {}
        self._next_reclaim_at: float = 0.0
        self._slots_reclaimed = 0

    @property
    def enabled(self) -> bool:
        return self.staging_depth > 0

    def request_spill(self, h: int, group: int) -> int:
        """Reserve a staging slot for `group`, or -1 if the ring is full.

        On `pop()`'s critical path, so no work beyond a deque pop. The caller
        copies `group` into the slot on the compute stream and `pop()` hands the
        original out immediately -- spilling by copy rather than by pin, because
        `pop()` runs precisely when there is no free group to withhold.
        """
        # A negative hash is a caller bug, not a dropped spill -- `_spill`
        # already refuses a group with no checkpoint. Guarded, but not counted,
        # so `spills_dropped` stays a clean backpressure signal.
        if h < 0:
            return -1
        if not self._free_slots:
            self.spills_dropped += 1
            self._note_drop()
            return -1
        slot = self._free_slots.popleft()
        self._pending.append((h, slot))
        self._slot_reserved_at[slot] = time.monotonic()
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
            self._slot_reserved_at.pop(slot, None)
            # The ring is moving again, so any drops so far were a burst.
            self._consecutive_drops = 0

    def reclaim_stale_slots(self, timeout_s: float, now: float | None = None) -> int:
        """Return slots whose spill report never came back to the free ring.

        `release_staging` is the only way a slot comes home, and it fires only
        on the worker's spill report. A lost report (the transfer was abandoned,
        or its completion vanished the way a stalled KV save's does) would pin
        that slot forever; enough of them and `request_spill` drops every spill
        (`spills_dropped` climbs, `_note_drop` warns) with no way to recover
        short of a restart. This is the ring-side twin of the engine's
        `_reconcile_stalled_deferred_saves`.

        Safety mirrors that reconciliation: `timeout_s` must be larger than the
        upstream (LMCache pin-monitor) abandon window, so the worker's staging
        buffer for this slot is no longer being read before the slot -- and thus
        the buffer -- is handed to a new spill. Caller passes the same abandon
        timeout used engine-side. Self-throttled to one real scan per
        `_RECLAIM_SCAN_INTERVAL_S`, so it is safe to call every step.

        A reclaimed slot's `_pending` entry (if it was never drained) is dropped
        too: its hash was never confirmed, so nothing indexed it and no load can
        ask for it. Returns the number of slots reclaimed this call.
        """
        if timeout_s <= 0 or not self._slot_reserved_at:
            return 0
        now = time.monotonic() if now is None else now
        if now < self._next_reclaim_at:
            return 0
        self._next_reclaim_at = now + _RECLAIM_SCAN_INTERVAL_S
        stale = [
            slot
            for slot, at in self._slot_reserved_at.items()
            if now - at >= timeout_s
        ]
        if not stale:
            return 0
        stale_set = set(stale)
        # Drop any still-queued pending entries for the stale slots; whatever is
        # left is a live spill still waiting to be drained this step.
        kept = [(h, slot) for (h, slot) in self._pending if slot not in stale_set]
        self._pending.clear()
        self._pending.extend(kept)
        for slot in stale:
            self._slot_reserved_at.pop(slot, None)
            if 0 <= slot < self.staging_depth and slot not in self._free_slots:
                self._free_slots.append(slot)
        self._slots_reclaimed += len(stale)
        # A reclaim means the ring was stuck, not bursting; reset the burst
        # counter so a fresh starvation warning can arm if it stalls again.
        self._consecutive_drops = 0
        self._warned_starved = False
        logger.warning(
            "State offload staging ring reclaimed %d slot(s) whose spill report "
            "never returned after %.0fs (staging_depth=%d, total reclaimed=%d). "
            "The worker very likely abandoned or lost the transfer's completion; "
            "the slot is returned to the ring so spills can resume.",
            len(stale),
            timeout_s,
            self.staging_depth,
            self._slots_reclaimed,
        )
        return len(stale)

    def forget(self, h: int) -> None:
        """Drop a hash whose load failed, so the next request does not retry."""
        self.hashes.discard(h)

    # -------------------------------- loads -------------------------------- #
    def request_load(self, req_id, h: int) -> bool:
        """Offer to fetch `h` back for `req_id`. False if this tier cannot.

        The guard between believing and delivering: a load is resolved only by
        a worker report, so offering one for a hash never stored would park the
        request against bytes no `get` can produce.
        """
        if h not in self.hashes:
            return False
        if req_id in self.pending_loads:
            # One request, one outstanding load: reports are keyed by request
            # id, so the first completion would unpark while the second is
            # still writing. No in-tree path reaches this today, and the
            # refusal costs only a disown.
            logger.warning(
                "state offload: request %s already has a load in flight; "
                "refusing a second one and letting the boundary be disowned.",
                req_id,
            )
            return False
        self.pending_loads[req_id] = int(h)
        self.loads_attempted += 1
        return True

    def complete_load(self, req_id) -> None:
        """The bytes landed in the request's group.

        The hash stays indexed: a load reads LMCache, it does not consume it,
        and the next request over the same prefix must still find it.
        """
        if self.pending_loads.pop(req_id, None) is not None:
            self.loads_completed += 1

    def fail_load(self, req_id) -> None:
        """No bytes came back. Retract the claim as well as counting it.

        A miss is the only evidence that LMCache's LRU dropped what this index
        advertises, so it has to be what un-advertises it. Leaving the hash makes
        every later request over that prefix park, miss and recompute.
        """
        h = self.pending_loads.pop(req_id, None)
        if h is None:
            return
        self.loads_failed += 1
        self.forget(h)

    def abandon_load(self, req_id) -> None:
        """The request went away before its bytes did. Neither outcome.

        Not `fail_load`: an abort says nothing about the bytes, and forgetting
        a loadable hash sends the next request back to a full recompute.
        """
        self.pending_loads.pop(req_id, None)

    def stats(self) -> dict[str, int]:
        """Counters for the periodic `state checkpoints:` line.

        Reached via `StateGroupPool.checkpoint_fates`, which prefixes every
        key with `state_offload_`. Read the three load counters together:
        `failed / attempted` is this index's false-positive rate, and
        `attempted - completed - failed` is what is in flight or was abandoned
        by an aborted request.
        """
        return {
            "spills_requested": self.spills_requested,
            "spills_dropped": self.spills_dropped,
            "spills_forgone": self.spills_forgone,
            "loads_attempted": self.loads_attempted,
            "loads_completed": self.loads_completed,
            "loads_failed": self.loads_failed,
            "indexed": len(self.hashes),
        }


# The connector backends whose worker half builds a `StateOffloadTier`. Only
# the offload backend does; `multi` qualifies when it lists one.
_STATE_TIER_BACKENDS = frozenset({"lmcache_offload"})


def kv_connector_hosts_state_tier(kv_transfer_config) -> bool:
    """Whether the configured KV connector can actually run the state tier.

    The tier is not standalone: its bytes ride the KV connector's worker half
    and its completions ride that connector's `get_finished`. Against any other
    backend the staging ring would hand out slots and never get one back, so
    this gates whether the ring is installed at all -- and a false negative
    (tier off) is much cheaper than a false positive (every slot leaks).

    A name check rather than a capability probe: this runs in the engine
    process at `BlockManager.__init__`, before any worker connector exists.
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
    """K: staging *groups* to reserve for the tier.

    Groups, not entries: a sizing site multiplies by its `entries_per_req`,
    because `state_entry_views` is indexed by group. One function rather than
    two `os.environ` reads, because the arena and the pool both size themselves
    from it and would otherwise disagree.

    This only sizes the ring; whether the tier runs follows the connector. Bad
    depths warn rather than raise (model load): non-integer falls back to 1,
    negative floors to 0.
    """
    raw = os.environ.get("OFFLOAD_STATE_STAGING_GROUPS")
    if raw is None:
        return 1
    try:
        depth = int(raw)
    except ValueError:
        # Warn rather than raise -- this runs during model load. Not silent
        # either: a mistyped depth buys one group instead of twenty, with a
        # starved ring 256 dropped spills later as the only other symptom.
        logger.warning(
            "state offload: invalid OFFLOAD_STATE_STAGING_GROUPS=%r; using 1",
            raw,
        )
        return 1
    if depth < 0:
        # Same reasoning as the typo above, and a worse outcome: a negative
        # depth turns the whole tier off, so the operator gets a
        # healthy-looking server that never spills.
        logger.warning(
            "state offload: negative OFFLOAD_STATE_STAGING_GROUPS=%r disables "
            "the tier entirely; using 0",
            raw,
        )
        return 0
    return depth
