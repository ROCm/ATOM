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
    """What has been spilled, what is queued to be, and what is coming back.

    `hashes` answers the membership half of `StateGroupPool._resumable_from`,
    which gates this set behind `state_pool.STATE_OFFLOAD_LOADS_WIRED` — now
    True, because the load that acts on the vote exists. The gate remains
    because a hash no load path can act on would end the right-to-left scan on
    a boundary nothing can deliver, so the two may only ever move together.
    The set is populated regardless of the flag: what the flag changes is who
    may vote on a hit, not what is recorded here.

    It is deliberately optimistic: LMCache's own LRU can drop bytes at any
    time, so a hash here means "was spilled once", never "is still there".
    The false positive costs one lookup and a park/unpark and is handled by
    the `failed_loading` path, which calls `fail_load` -> `forget`.
    """

    def __init__(self, staging_depth: int, kv_offload_enabled: bool) -> None:
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
        self.kv_offload_enabled = bool(kv_offload_enabled)
        self.hashes: set[int] = set()
        self._free_slots: deque[int] = deque(range(self.staging_depth))
        self._pending: deque[tuple[int, int]] = deque()
        # req_id -> hash, for loads the engine has offered and not yet settled.
        # Keyed by request because that is what comes back: the worker reports
        # a load through `finished_loading`/`failed_loading`, the same
        # request-keyed channel a KV load uses. (The spill's reports are keyed
        # by hash and slot instead -- by the time a spill lands, the request
        # that owned the checkpoint is long gone.)
        self.pending_loads: dict = {}
        # Read once, here, rather than per call: `_resumable_from` consults it
        # inside a per-block scan.
        self.min_load_tokens = state_offload_min_load_tokens()
        self.spills_requested = 0
        self.spills_dropped = 0
        self.loads_attempted = 0
        self.loads_completed = 0
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

    # -------------------------------- loads -------------------------------- #
    def request_load(self, req_id, h: int) -> bool:
        """Offer to fetch `h` back for `req_id`. False if this tier cannot.

        The refusal is the guard between believing and delivering. A load is
        resolved only by a report from the worker, so offering one for a hash
        the tier never stored would park the request against bytes no `get`
        can produce, and nothing would ever wake it.
        """
        if h not in self.hashes:
            return False
        if req_id in self.pending_loads:
            # One request, one outstanding load. Reports are keyed by request
            # id and nothing distinguishes two of them, so the first completion
            # would unpark the request while the second is still writing its
            # group. No in-tree path reaches this today -- a parked request is
            # in `waiting` and only running requests are preempted -- and the
            # refusal costs a disown, which is always a safe answer.
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

        A miss is the only evidence anyone gets that LMCache's LRU dropped what
        this index still advertises, so it has to be what un-advertises it.
        Leaving the hash in place makes every later request over that prefix
        park, miss, and recompute -- the cost of the false positive paid once
        per request instead of once.
        """
        h = self.pending_loads.pop(req_id, None)
        if h is None:
            return
        self.loads_failed += 1
        self.forget(h)

    def abandon_load(self, req_id) -> None:
        """The request went away before its bytes did. Neither outcome.

        Deliberately not `fail_load`: an abort says nothing about the bytes, and
        forgetting a hash that is still perfectly loadable would send the next
        request over that prefix back to a full recompute.
        """
        self.pending_loads.pop(req_id, None)

    def stats(self) -> dict[str, int]:
        """Counters for the periodic `state checkpoints:` line.

        Reached via `StateGroupPool.checkpoint_fates`, which prefixes every key
        with `state_offload_`.

        The three load counters are read together or not at all.
        `failed / attempted` is this index's false-positive rate: every attempt
        was made against a hash the index advertised, so a failure means
        LMCache's LRU had already dropped the bytes. `attempted - completed -
        failed` is what is still in flight or was abandoned by an aborted
        request.
        """
        return {
            "spills_requested": self.spills_requested,
            "spills_dropped": self.spills_dropped,
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


def should_load_state(hit_tokens: int, floor_tokens: int) -> bool:
    """Whether a state hit of `hit_tokens` tokens is worth an H2D.

    Lives here, engine-side, and not with the worker's transfer driver: the
    decision is taken inside `StateGroupPool._resumable_from`, in the engine
    process, before anything has been asked to move. The worker only ever
    executes a load somebody else already decided on.

    The zero case is not redundant with the floor. A floor of 0 means "no
    minimum", not "load a boundary of nothing": a 0-token boundary is a cold
    start, where there is no state to restore and the group is used as-is.
    """
    hit_tokens = int(hit_tokens)
    return hit_tokens > 0 and hit_tokens >= int(floor_tokens)


def state_offload_min_load_tokens() -> int:
    """Boundary below which a state load is not worth its H2D. 0 = load any.

    Deliberately **not** modelled on KV's `OFFLOAD_MIN_LOAD_TOKENS` (8192), and
    the arithmetic is different enough to be worth stating. A KV load moves
    bytes proportional to the hit, so a short one spends a whole round trip
    moving very little; a floor is the natural shape. A state load moves one
    flat entry -- 53.6 MiB measured, a few ms of H2D -- whatever the boundary
    is, while the prefill it saves grows *with* the boundary. There is no
    length below which the transfer is the expensive half, so the default
    declines nothing.

    What the knob is for: an entry much larger than that, or an index with a
    bad false-positive rate (`loads_failed / loads_attempted` on the periodic
    stats line), where a floor bounds what each miss costs.

    Consulted inside `StateGroupPool._resumable_from`, which is why a bad value
    floors to 0 rather than to something large: a mistyped floor that silently
    became huge would turn every load off and read as a broken tier.
    """
    raw = os.environ.get("OFFLOAD_STATE_MIN_LOAD_TOKENS")
    if raw is None:
        return 0
    try:
        floor = int(raw)
    except ValueError:
        logger.warning(
            "state offload: invalid OFFLOAD_STATE_MIN_LOAD_TOKENS=%r; using 0",
            raw,
        )
        return 0
    if floor < 0:
        logger.warning(
            "state offload: negative OFFLOAD_STATE_MIN_LOAD_TOKENS=%r; using 0",
            raw,
        )
        return 0
    return floor


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
