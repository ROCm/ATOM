# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The store leg of the recurrent-state offload tier: one queue, one record.

A K3 checkpoint is a PAGE image living in ordinary KV blocks, owned by
``PageUnitCheckpointStore``. Copying that image out to LMCache cannot ride the
request that produced it -- a checkpoint outlives its request, which is the
whole point of the tier -- so the store leg needs a queue of its own and a
completion channel of its own. This is both.

A store has two ends and they are days apart in machine time:

1. ``source_released`` -- the device-side gather has drained, so the PAGE units
   belong to the KV pool again and must be unpinned *immediately*: they are
   pool bytes that live requests are competing for;
2. ``stored`` -- whether the CPU put succeeded, known only afterwards, and the
   only thing that licenses ``StateOffloadIndex.note_stored``.

Both are fields of one ``StateStoreRecord`` held by one owner, so this object
can state its own invariant (``check_invariant``) the way
``StateOffloadIndex`` states the load leg's. Splitting the two ends across two
owners is what made the previous implementation's store-path defects unstatable
and therefore untestable.

Pure Python: no torch, no device, no connector. The bytes are the worker's
business.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from dataclasses import dataclass
from time import monotonic

logger = logging.getLogger("atom")

# How many settled-by-reclaim operations to remember, so a late report for one
# can be recognised rather than counted as a completion. Reports arrive within
# seconds of their dispatch or not at all, so this only has to outlive the
# wire; it is a bound on a leak, not a working set.
_FORFEITED_MEMORY = 1024


@dataclass(frozen=True)
class StateStoreSpec:
    """One store to hand the connector: which image, and which units hold it.

    `op_id` is a plain int because it becomes `ConnectorCompletion.
    operation_id`, which the transfer types require to be hashable.
    """

    op_id: int
    prefix_hash: int
    unit_ids: tuple[int, ...]


@dataclass
class StateStoreRecord:
    """One dispatched store, from `take` to its terminal transition.

    `source_released` is not a terminal state. The units are back in the pool
    but the store is still dispatched-and-unreported, so the record stays here:
    it is what keeps a second attempt at the same hash off the wire, and what
    the reclaimer counts down.
    """

    op_id: int
    prefix_hash: int
    checkpoint_id: int
    unit_ids: tuple[int, ...]
    at: float
    source_released: bool = False


class StateStoreQueue:
    """Nominations, in-flight stores and their two completions.

    The invariant, asserted by `check_invariant`::

        nominated == settled + inflight + queued

    plus: no record holds a pin after it has settled. Every counter below is
    reachable from `stats`, which `BlockManager.checkpoint_funnel` surfaces
    beside `StateOffloadIndex.stats`. That is a requirement rather than a
    nicety: in the previous implementation every observability hook added for
    this tier was itself unreachable, which is why six separate defects were
    silent.
    """

    def __init__(self, *, store, index, max_inflight: int) -> None:
        # The PAGE-unit owner, for `pin_checkpoint` / `unpin_checkpoint` only.
        # This queue never allocates, evicts or indexes an image.
        self._store = store
        # The tier's engine-side index: `note_stored` and the store counters.
        self._index = index
        # Each in-flight store pins a whole image -- ~127 units on K3 -- out of
        # the KV pool for as long as the D2H copy and the CPU put take, which
        # spans several scheduler passes. Bounding the count is what keeps the
        # tier from bidding against live KV.
        self._max_inflight = max(1, int(max_inflight))

        # Nominated prefix hashes, oldest first. Hashes rather than records:
        # a nomination takes no pin, so the pool may spend its checkpoint while
        # it waits, and everything a dispatch needs is re-read at `take`.
        self._queued: deque[int] = deque()
        self._queued_set: set[int] = set()
        # op_id -> StateStoreRecord for stores dispatched and not yet settled.
        self._records: dict[int, StateStoreRecord] = {}
        self._next_op_id = 0
        # Pins currently held, counted here so `check_invariant` can compare it
        # against the records that claim to hold them.
        self._pins_held = 0
        # Operations the reclaimer settled. A late report for one of these is
        # recognised instead of being taken for a completion.
        self._forfeited: OrderedDict[int, None] = OrderedDict()

        # A nomination that cannot be dispatched still waits unpinned, so the
        # backlog costs list entries and nothing else -- but a tier draining
        # slower than checkpoints reach READY would grow it without bound.
        self._backlog_cap = 8192

        self.nominated = 0
        self.settled = 0
        self.nominations_collapsed = 0
        self.nominations_dropped = 0
        self.nominations_requeued = 0
        self.nominations_satisfied = 0
        self.nominations_stale = 0
        self.dispatched = 0
        self.sources_released = 0
        self.stores_reclaimed = 0
        self.stores_untrusted = 0

    # ------------------------------ invariant ------------------------------ #
    @property
    def inflight(self) -> int:
        """Stores dispatched and not yet settled."""
        return len(self._records)

    @property
    def queued(self) -> int:
        """Nominations waiting for an in-flight slot."""
        return len(self._queued)

    def check_invariant(self) -> None:
        """Assert this object's whole contract. Cheap; call it from tests and
        from the periodic stats path.

        Three properties:

        1. every accepted nomination is queued, in flight, or settled exactly
           once -- a nomination that is none of the three has been silently
           lost, which is the defect this object was written to make loud;
        2. the pins held match the records that claim to hold them, so a
           settled record cannot leave a pin behind (an image out of the pool
           for the process lifetime) and no pin is released twice (an underflow
           in `PageUnitCheckpointStore`, or worse, a live image handed back);
        3. `_queued` and `_queued_set` never diverge, since a divergence turns
           the collapse rule into either a duplicate store or a lost image.
        """
        if self.nominated != self.settled + self.inflight + self.queued:
            raise AssertionError(
                "state store queue: nominated != settled + inflight + queued "
                f"({self.nominated} != {self.settled} + {self.inflight} + "
                f"{self.queued})"
            )
        expected_pins = sum(
            1 for record in self._records.values() if not record.source_released
        )
        if self._pins_held != expected_pins:
            raise AssertionError(
                "state store queue: pins held != unreleased sources "
                f"({self._pins_held} != {expected_pins})"
            )
        if len(self._queued) != len(self._queued_set):
            raise AssertionError(
                "state store queue: _queued and _queued_set diverged "
                f"({len(self._queued)} != {len(self._queued_set)})"
            )

    # ---------------------------- nominations ------------------------------ #
    def nominate(self, prefix_hash: int, unit_ids) -> None:
        """Offer a freshly READY image to the tier. Takes no pin.

        Called from `PageUnitCheckpointStore.complete_inflight`'s READY
        transition. Nomination, not reservation: a queued candidate stays
        evictable, so the pool never waits on the CPU tier and a READY unpinned
        checkpoint still counts as space available to live KV. `unit_ids` is
        therefore not kept -- it can be stale by the time the store is
        dispatched, and `take` re-reads the units it actually pins.
        """
        del unit_ids
        if not getattr(self._index, "can_store", True):
            # A load-only role. A nomination nobody drains would sit in the
            # backlog until it aged out, and every stat would read as a tier
            # falling behind rather than as one that was never granted.
            return
        prefix_hash = int(prefix_hash)
        if prefix_hash in self._queued_set:
            # One boundary reached twice. Collapsing keeps `_queued` unique, so
            # the second entry cannot dispatch a duplicate put of bytes the
            # first one already sent. Not counted as a nomination: it never
            # entered the queue, so the invariant never owed a terminal state
            # for it.
            self.nominations_collapsed += 1
            return
        self._queued.append(prefix_hash)
        self._queued_set.add(prefix_hash)
        self.nominated += 1
        while len(self._queued) > self._backlog_cap:
            if self.nominations_dropped == 0:
                logger.warning(
                    "state offload: store nomination backlog exceeded %d; "
                    "dropping the oldest nominations. The CPU tier is draining "
                    "checkpoints slower than they reach READY.",
                    self._backlog_cap,
                )
            self._drop_queued(self._queued.popleft())
            self.nominations_dropped += 1

    def _drop_queued(self, prefix_hash: int) -> None:
        """Terminal for a nomination that left `_queued` without dispatching."""
        self._queued_set.discard(prefix_hash)
        self.settled += 1

    # ------------------------------- dispatch ------------------------------ #
    def take(self, limit: int) -> list[StateStoreSpec]:
        """Up to `limit` stores to hand the connector now, pinning each.

        The pin is taken HERE, not at nomination: a pin lives in this process
        while the D2H runs in the worker, so it spans several scheduler passes,
        and pinning at READY would make every checkpoint un-evictable across
        that window -- breaking the admission rule that a READY unpinned
        checkpoint counts as space available to live KV.

        Bounded by both `limit` and `max_inflight`, because each spec pins a
        whole image out of the pool.
        """
        out: list[StateStoreSpec] = []
        # Nominations that cannot dispatch yet but are still valid: held aside
        # and re-queued after the walk, never re-appended inside it, so a hash
        # that stays in flight cannot spin this call.
        deferred: list[int] = []
        room = min(int(limit), self._max_inflight - self.inflight)
        while self._queued and len(out) < room:
            prefix_hash = self._queued.popleft()
            self._queued_set.discard(prefix_hash)
            if self._hash_inflight(prefix_hash):
                # NOT terminal, and this is the defect that made the previous
                # implementation lose images: a bare drop here (the old
                # `popleft` + `continue`) retired a live nomination whose only
                # problem was an earlier attempt at the same hash still being on
                # the wire. Nomination happens once, at the READY transition, so
                # that image was never offered again -- and when the in-flight
                # attempt later failed, it was gone for good. Re-queue instead.
                deferred.append(prefix_hash)
                self.nominations_requeued += 1
                continue
            if prefix_hash in self._index.hashes:
                # The tier already believes it holds these bytes -- ordinarily
                # the earlier attempt this nomination waited behind, which has
                # since reported success. Storing them again would cost a whole
                # image's pins for a put that changes nothing. The index is
                # optimistic, so a hash LMCache later drops is dropped here too
                # by `forget` on the missing load, and is storable again at its
                # next READY.
                #
                # Bare membership, not `could_serve`: this is not a tier-hit
                # vote but a duplicate-put check, and a store-only role
                # (`can_load` False) that routed it through `could_serve` would
                # never see its own stores and would re-store every image.
                self.nominations_satisfied += 1
                self.settled += 1
                continue
            pinned = self._store.pin_checkpoint(prefix_hash)
            if pinned is None:
                # Spent while it waited, which nomination deliberately allows.
                # Terminal: there is no image left to store.
                self.nominations_stale += 1
                self.settled += 1
                continue
            checkpoint_id, unit_ids = pinned
            self._pins_held += 1
            self._next_op_id += 1
            op_id = self._next_op_id
            self._records[op_id] = StateStoreRecord(
                op_id=op_id,
                prefix_hash=prefix_hash,
                checkpoint_id=checkpoint_id,
                unit_ids=tuple(unit_ids),
                at=monotonic(),
            )
            self.dispatched += 1
            self._index.stores_attempted += 1
            out.append(
                StateStoreSpec(
                    op_id=op_id,
                    prefix_hash=prefix_hash,
                    unit_ids=tuple(unit_ids),
                )
            )
        # Put the deferred nominations back at the front, oldest first: the
        # drain is `popleft`, so appending them behind newer entries would
        # starve exactly the ones that have already waited longest.
        for prefix_hash in reversed(deferred):
            self._queued.appendleft(prefix_hash)
            self._queued_set.add(prefix_hash)
        return out

    def _hash_inflight(self, prefix_hash: int) -> bool:
        """Whether some attempt at `prefix_hash` is already dispatched.

        Two live stores of one image would copy the same bytes twice and the
        second report would unpin a record the first already released.
        """
        return any(
            record.prefix_hash == prefix_hash for record in self._records.values()
        )

    # ------------------------------ completions ---------------------------- #
    def settle_source(self, op_id) -> None:
        """Phase one: the gather drained, so hand the PAGE units back now.

        The earlier of a store's two ends, and the urgent one: holding an image
        out of the pool across the subsequent CPU put would cost reuse for a
        step that cannot touch the units. It does NOT retire the record -- the
        store is still dispatched-and-unreported, and `_hash_inflight` and
        `has_pending` must keep seeing it until the report lands.

        Idempotent, and it must be: a second call unpinning a second time would
        underflow the record, or -- once the same checkpoint id has been reused
        -- release a live image out from under whatever is reading it.
        """
        record = self._records.get(op_id)
        if record is None or record.source_released:
            return
        record.source_released = True
        self._pins_held -= 1
        self.sources_released += 1
        self._store.unpin_checkpoint(record.checkpoint_id)

    def settle_stored(self, op_id, ok: bool) -> None:
        """Phase two: the CPU put reported, either way. Retire the operation.

        The only thing that licenses `note_stored`: a hash advertised before its
        bytes exist parks the next request over that prefix against a `get` that
        must miss.
        """
        record = self._records.get(op_id)
        if record is None:
            if op_id in self._forfeited:
                # A report for a store the reclaimer already settled. It cannot
                # be counted as a completion, and the image cannot be indexed;
                # see `reclaim`.
                self.stores_untrusted += 1
            return
        # The report may beat the source release, or the store may have failed
        # before the gather ran at all. Either way the units come back here, and
        # they come back before the record is retired so no settled record ever
        # holds a pin.
        self.settle_source(op_id)
        del self._records[op_id]
        self.settled += 1
        if ok:
            self._index.note_stored(record.prefix_hash)
            self._index.stores_completed += 1
        else:
            self._index.stores_failed += 1

    def reclaim(self, timeout_s: float) -> int:
        """Settle stores whose report never came, after `timeout_s`.

        A record lives in this process while the copy runs in the worker, so a
        crashed worker or a dropped completion would hold a whole image out of
        the KV pool for the process lifetime -- one leak per lost report, no
        error and no warning, until the pool refuses every hybrid request.

        **It does not prove the reader stopped.** A K3 state store gathers ATOM
        PAGE units directly and is not covered by LMCache's GPU-source pin
        monitor, so nothing here can tell a lost report from a worker still
        inside the gather. What follows is not that the units may never be
        taken back, but that a store settled this way can never be indexed: if
        the reader had not stopped, the pool may since have handed those units
        to another request whose writes the gather picked up, and the CPU image
        would be a mix of two prefixes under the first one's hash. Resuming
        onto that is silent wrong output.

        So the operation is remembered in `_forfeited` and a late report counts
        as `stores_untrusted` rather than as a completion. That forfeits the
        occasional image whose source was already released and whose bytes were
        therefore fine; the alternative is a record that stays alive past its
        terminal transition, which is the split ownership this object exists to
        remove. One recompute is the cheaper half of that trade.
        """
        if timeout_s <= 0 or not self._records:
            return 0
        deadline = monotonic() - timeout_s
        stale = [
            op_id for op_id, record in self._records.items() if record.at <= deadline
        ]
        for op_id in stale:
            record = self._records[op_id]
            # Release the pin before retiring the record, so `check_invariant`
            # never sees a settled record still holding one.
            self.settle_source(op_id)
            del self._records[op_id]
            self.settled += 1
            self.stores_reclaimed += 1
            self._forfeited[op_id] = None
            while len(self._forfeited) > _FORFEITED_MEMORY:
                self._forfeited.popitem(last=False)
            logger.warning(
                "state offload: store %s for prefix %s never reported after "
                "%.1fs; taking its units back and forfeiting the image, which "
                "will not be indexed even if a report arrives later.",
                op_id,
                record.prefix_hash,
                timeout_s,
            )
        return len(stale)

    def has_pending(self) -> bool:
        """Whether the engine still owes work on the state store leg.

        True across `settle_source`, because the report is the true end of
        "dispatched but unreported" and the poll that carries it must keep
        running. Reclaim shares the same dict, so a lost report cannot latch
        this signal on.
        """
        return bool(self._queued or self._records)

    # -------------------------------- stats -------------------------------- #
    def stats(self) -> dict[str, int]:
        """Counters for the periodic `state checkpoints:` line.

        Read `nominations_stale` against `nominated` as how much of what reaches
        READY the pool spends before the tier can copy it -- a climbing ratio
        means `max_inflight` is throttling the tier below the rate checkpoints
        are produced. `nominations_dropped` and `stores_untrusted` should both
        stay at zero; either one moving is a fault, not backpressure.
        """
        return {
            "store_nominated": self.nominated,
            "store_settled": self.settled,
            "store_queued": self.queued,
            "store_inflight": self.inflight,
            "store_dispatched": self.dispatched,
            "store_sources_released": self.sources_released,
            "nominations_collapsed": self.nominations_collapsed,
            "nominations_dropped": self.nominations_dropped,
            "nominations_requeued": self.nominations_requeued,
            "nominations_satisfied": self.nominations_satisfied,
            "nominations_stale": self.nominations_stale,
            "stores_reclaimed": self.stores_reclaimed,
            "stores_untrusted": self.stores_untrusted,
        }
