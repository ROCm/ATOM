# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Engine-side bookkeeping for the recurrent-state offload tier.

One object lives here and it touches no device: the set of hashes believed to be
in LMCache, plus the loads offered against them. The bytes and the transfers are
the worker's business (``kv_transfer/offload/hybrid/kimi_k3``).

The index is in memory and never persisted: LMCache's ``LocalDiskBackend``
starts from an empty dict and never scans its directory, so an index recovered
from disk would be a pure false-positive generator. Index and bytes share one
server lifetime. The consequence -- after a restart a K3 prefix cannot be
resumed even though its KV is still in LMCache -- is a known limitation, not an
oversight; see ``offload/README.md``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

logger = logging.getLogger("atom")


@dataclass
class _OutstandingLoad:
    """One dispatched state load that has not reached a terminal state."""

    prefix_hash: int
    slot: int
    at: float
    # Set when the request is torn down while its load is still in flight. The
    # slot then belongs to this index rather than to the sequence, because the
    # worker may still be scattering into it.
    orphaned: bool = False


class StateOffloadIndex:
    """What is believed to be in LMCache, and what is being fetched back.

    This object is the *sole* engine-side owner of a state load's lifecycle, and
    that is the point of it. It can therefore state its own invariant::

        dispatched == settled + outstanding

    Spreading those three facts over three owners is what made every load-path
    defect in the previous implementation unstatable, hence untestable. Here
    ``check_invariant`` is a method and ``TestStateOffloadInvariant`` asserts it.

    ``hashes`` answers the membership half of the tier's vote. It is
    deliberately optimistic: LMCache's own LRU can drop bytes at any time, so a
    hash here means "was stored once", never "is still there". A false positive
    costs one lookup and one recompute, never wrong output.
    """

    def __init__(
        self,
        *,
        can_store: bool,
        can_load: bool,
        chunk_tokens: int,
        release_slot: Callable[[int], None],
    ) -> None:
        # The two legs are separately granted. A `kv_producer` role saves and
        # never loads, a `kv_consumer` loads and never saves, and offering a
        # load the worker will not serve parks the request that took it.
        self.can_store = bool(can_store)
        self.can_load = bool(can_load)
        # The LMCache chunk grid, reported by the connector at attach time
        # rather than re-parsed from config here. Two independent derivations of
        # this number is how the engine comes to floor a boundary against one
        # grid while the worker validates it against another -- which is silent
        # wrong output, not a mismatch anyone would see. It also travels on the
        # load spec so the worker does not re-derive it either.
        self.chunk_tokens = int(chunk_tokens)
        # The only capability this index needs from the allocator: hand a slot
        # back when its load settles and the request that owned it is gone.
        self._release_slot = release_slot

        # The optimistic membership index. It grows on `note_stored` and shrinks
        # on `forget`, so a long-lived server that stored many distinct prefixes
        # would grow it without bound. `hashes` stays a plain set so membership
        # is O(1); `_hash_lru` records insertion order beside it purely so the
        # coldest hash can be dropped on overflow. The two move in lockstep --
        # every add and every forget touches both -- which `check_invariant`
        # also asserts, because a divergence turns the cap into either a leak or
        # a false-positive generator.
        self.hashes: set[int] = set()
        self._hash_lru: OrderedDict[int, None] = OrderedDict()
        # ~1M hashes -> a few tens of MB, well above any working set a tier of
        # realistic capacity holds, so it bounds a leak without evicting live
        # entries in normal operation.
        self._hash_cap = 1 << 20
        self.hashes_evicted = 0

        # req_id -> _OutstandingLoad. Keyed by request because that is what
        # comes back: under the fused design a state load reports on the same
        # `finished_loading` / `failed_loading` channel as its KV leg, in one
        # completion, so there is exactly one report per dispatch.
        self._outstanding: dict[object, _OutstandingLoad] = {}
        self.dispatched = 0
        self.settled = 0

        self.loads_completed = 0
        self.loads_failed = 0
        self.loads_abandoned = 0
        self.loads_refused_inflight = 0
        self.orphan_slots_reclaimed = 0
        # Store leg. `refused` is a wiring fault, not backpressure: any non-zero
        # value means the connector never took the store at all. Kept apart from
        # `failed`, which means the worker tried and could not.
        self.stores_attempted = 0
        self.stores_completed = 0
        self.stores_failed = 0
        self.stores_refused = 0

    # ------------------------------ invariant ------------------------------ #
    @property
    def outstanding(self) -> int:
        """Loads dispatched and not yet settled."""
        return len(self._outstanding)

    def check_invariant(self) -> None:
        """Assert this object's whole contract. Cheap; call it from tests and
        from the periodic stats path.

        Two properties, both of which were violated by the previous design in
        ways no test could express:

        1. every dispatched load reaches exactly one terminal state;
        2. `hashes` and `_hash_lru` never diverge.
        """
        if self.dispatched != self.settled + self.outstanding:
            raise AssertionError(
                "state offload index: dispatched != settled + outstanding "
                f"({self.dispatched} != {self.settled} + {self.outstanding})"
            )
        if len(self.hashes) != len(self._hash_lru):
            raise AssertionError(
                "state offload index: hashes and _hash_lru diverged "
                f"({len(self.hashes)} != {len(self._hash_lru)})"
            )

    # ------------------------------- stores -------------------------------- #
    def note_stored(self, h: int) -> None:
        """A store landed in LMCache and the hash is now worth voting for.

        Called from the report the worker sends back, never at submission: a
        hash advertised before its bytes exist parks the next request over that
        prefix against a `get` that must miss.
        """
        h = int(h)
        self.hashes.add(h)
        # Reinsert at the young end whether or not it was already present, so a
        # re-stored prefix counts as freshly used, then trim the cold end.
        self._hash_lru.pop(h, None)
        self._hash_lru[h] = None
        while len(self._hash_lru) > self._hash_cap:
            old, _ = self._hash_lru.popitem(last=False)
            self.hashes.discard(old)
            self.hashes_evicted += 1

    def forget(self, h: int) -> None:
        """Drop a hash whose load missed, so the next request does not retry."""
        h = int(h)
        self.hashes.discard(h)
        self._hash_lru.pop(h, None)

    def could_serve(self, h: int) -> bool:
        """Whether a load for `h` could be offered at all.

        The single tier-hit predicate. Every voter routes through it -- the
        checkpoint coordinator's reachability scan, the allocator's admission
        probe, and `dispatch` itself -- so the three cannot drift.

        Mirroring only `h in hashes` somewhere, without `can_load`, is not a
        cosmetic difference: a `kv_producer` (can_load False, but `hashes`
        populated from its own stores) then votes a tier hit it will refuse, the
        right-to-left resumable scan stops at that rung and skips a still-
        resident HBM rung, and the boundary is disowned -- forfeiting an HBM
        checkpoint to a full recompute on every such request.
        """
        return self.can_load and int(h) in self.hashes

    # -------------------------------- loads -------------------------------- #
    def dispatch(self, req_id, h: int, slot: int) -> bool:
        """Offer to fetch `h` back into `slot` for `req_id`. False if refused.

        Refusing is always safe: the caller disowns the boundary and recomputes.
        """
        if not self.could_serve(h):
            # A store-only role, or a hash never stored. Offering either parks
            # the request against a report that cannot come.
            return False
        if req_id in self._outstanding:
            # One request, one outstanding load. Reports are keyed by request,
            # so a second dispatch would let the first completion settle the
            # second load's slot -- and if the first was orphaned, its parked
            # slot would be overwritten and lost to the reclaimer, which
            # iterates this dict. Refusing costs one disowned boundary; the
            # alternative is a permanent slot leak with no backstop.
            self.loads_refused_inflight += 1
            logger.warning(
                "state offload: request %s already has a load in flight; "
                "refusing a second one and disowning the boundary.",
                req_id,
            )
            return False
        self._outstanding[req_id] = _OutstandingLoad(
            prefix_hash=int(h), slot=int(slot), at=monotonic()
        )
        self.dispatched += 1
        return True

    def orphan(self, req_id) -> bool:
        """The request is being torn down while its load is still in flight.

        The slot passes to this index: the worker may still be scattering into
        it, so it cannot go back on the free list until the report lands. Returns
        False when there was nothing in flight, which is the ordinary case and
        means the caller keeps the slot.
        """
        pending = self._outstanding.get(req_id)
        if pending is None:
            return False
        pending.orphaned = True
        return True

    def settle(self, req_id, ok: bool, *, missing: bool = False) -> None:
        """The one terminal transition. Exactly one per dispatch.

        `ok` is the *fused* verdict for the whole load, so a False may mean the
        KV leg failed while the state bytes were present and untouched.
        Retracting the hash on that would permanently deny state that is still
        there, so only `missing=True` -- which the worker sets when the state
        `get` itself missed -- forgets it.
        """
        pending = self._outstanding.pop(req_id, None)
        if pending is None:
            return
        self.settled += 1
        if ok:
            # The hash stays indexed: a load reads LMCache, it does not consume
            # it, and the next request over the same prefix must still find it.
            self.loads_completed += 1
        else:
            self.loads_failed += 1
            if missing:
                # A miss is the only evidence LMCache's LRU dropped what this
                # index advertises, so it has to be what un-advertises it.
                self.forget(pending.prefix_hash)
        if pending.orphaned:
            self._release_slot(pending.slot)

    def abandon(self, req_id) -> None:
        """The load was never handed to a worker, so nothing will report it.

        Terminal, like `settle`, but neither outcome: an abandon says nothing
        about the bytes, and forgetting a loadable hash sends the next request
        over that prefix back to a full recompute.
        """
        pending = self._outstanding.pop(req_id, None)
        if pending is None:
            return
        self.settled += 1
        self.loads_abandoned += 1
        if pending.orphaned:
            self._release_slot(pending.slot)

    def reclaim(self, timeout_s: float) -> int:
        """Free orphaned slots whose report never came, after `timeout_s`.

        Fusing the two legs removed the desync between them; it did not remove
        the possibility that a worker dies or a completion is dropped. Without
        this, one leak per lost report drains the state pool until the
        admission gate refuses every hybrid request -- no error, no warning.

        Only *orphaned* entries are reclaimed. A live request's slot must never
        be yanked out from under a worker that may still be writing it; a live
        request whose report is lost shows up as `outstanding` climbing, which
        is what the invariant is for.

        `timeout_s` must not be tighter than the store-pin abandon window, or
        this reclaimer becomes the very hazard it exists to prevent.
        """
        if timeout_s <= 0 or not self._outstanding:
            return 0
        deadline = monotonic() - timeout_s
        stale = [
            req_id
            for req_id, pending in self._outstanding.items()
            if pending.orphaned and pending.at <= deadline
        ]
        for req_id in stale:
            pending = self._outstanding.pop(req_id)
            self.settled += 1
            self.orphan_slots_reclaimed += 1
            self._release_slot(pending.slot)
        if stale:
            logger.warning(
                "state offload: reclaimed %d orphaned load slot(s) whose "
                "report never arrived after %.1fs",
                len(stale),
                timeout_s,
            )
        return len(stale)

    def has_outstanding(self) -> bool:
        """Whether the engine still owes work on the state load leg.

        Asks what the engine owes, not what the connector's queue holds -- the
        queue empties at dispatch, so a liveness predicate reading it reports
        idle while reports are still due.
        """
        return bool(self._outstanding)

    # -------------------------------- stats -------------------------------- #
    def stats(self) -> dict[str, int]:
        """Counters for the periodic `state checkpoints:` line.

        Every counter here is reachable from `checkpoint_funnel`. That is a
        requirement, not a nicety: in the previous implementation every
        observability hook added for this tier was itself unreachable, which is
        why six separate defects were diagnosed only by reading the source.

        Read `loads_failed / dispatched` as this index's false-positive rate,
        and `stores_completed` against `checkpoints_kept` as how much of what
        HBM keeps the CPU tier never received.
        """
        return {
            "stores_attempted": self.stores_attempted,
            "stores_completed": self.stores_completed,
            "stores_failed": self.stores_failed,
            "stores_refused": self.stores_refused,
            "loads_dispatched": self.dispatched,
            "loads_settled": self.settled,
            "loads_outstanding": self.outstanding,
            "loads_completed": self.loads_completed,
            "loads_failed": self.loads_failed,
            "loads_abandoned": self.loads_abandoned,
            "loads_refused_inflight": self.loads_refused_inflight,
            "orphan_slots_reclaimed": self.orphan_slots_reclaimed,
            "indexed": len(self.hashes),
            # Non-zero means the index hit `_hash_cap` and is dropping the
            # coldest hashes. Harmless -- a dropped hash costs one reuse -- but a
            # climbing value means the cap is below the live working set.
            "hashes_evicted": self.hashes_evicted,
        }
