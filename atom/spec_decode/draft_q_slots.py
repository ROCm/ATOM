"""Which rows of the persistent draft-q store belong to which request.

Lossless speculative sampling verifies each draft token against the q it was
drawn from, a step after the propose that drew it. The scheduler reorders the
batch in between and may run prefill-only steps that propose for other
requests, so q cannot live at a batch slot: each request owns a block of
``rows_per_slot`` rows in the store for as long as it keeps drafting.

Pure host bookkeeping, deliberately free of torch and aiter, so the policy is
testable without a device.
"""

import logging

logger = logging.getLogger("atom")


class DraftQSlots:
    """req_id -> persistent q slot, least-recently-drafted first.

    Release is by expiry. A worker is handed batches, never departures, so a
    finished, aborted or preempted request keeps its slot until the free list
    runs out and it is the stalest entry. Expiry is safe whoever the victim is:
    the victim drops out of the map, its next verify gets the -1 sentinel, and
    the accept kernel rejects its first draft and draws that token from the
    target alone. The output distribution is unchanged; the cost is one step of
    speculation for one request.

    The victim is usually a dead request, but nothing guarantees it. A request
    waiting through a long prefill-first burst is not refreshed while more than
    ``num_slots - live`` new ids claim slots, which a stream of prefill-only
    (``max_tokens=1``) requests can do.

    Two counters tell the cases apart. ``evictions`` counts every expiry, most
    of them a finished request's slot being reused, so it grows with the
    requests served whether or not anything was lost. ``live_expiries`` counts
    the ones that cost a step: an expired request that came back and found the
    sentinel. ``verifies`` counts the lookups that found their rows, so
    ``live_expiries / (verifies + live_expiries)`` is the fraction of lookups
    that lost their q. Only lookups a verify reads count: `DraftQStaging`
    asks for none on a step whose verify reads no q (prefill, align-only, all
    greedy), so a preempted request re-prefilled in chunks adds nothing until
    its first verify, which reads the slot its final chunk claimed. Expired ids
    are remembered for the last ``_EXPIRED_MEMORY * num_slots`` expiries only;
    a request that waits out more than that is missed, so under extreme churn
    the count is a lower bound.

    ``log``: whether this instance reports the counters. Every TP rank keeps
    the same map, so only the first rank of each DP group says so.
    """

    # How many expired ids `live_expiries` remembers, per slot.
    _EXPIRED_MEMORY = 16

    def __init__(self, num_slots: int, rows_per_slot: int, *, log: bool = True):
        if num_slots <= 0 or rows_per_slot <= 0:
            raise ValueError(
                f"need positive slots and rows, got {num_slots} x {rows_per_slot}"
            )
        self.num_slots = num_slots
        self.rows_per_slot = rows_per_slot
        self._log = log
        # A plain dict is insertion-ordered; `claim_bases` re-inserts, so the
        # first key is always the least recently drafted request.
        self._slot_of: dict = {}
        self._free = list(range(num_slots - 1, -1, -1))
        self.evictions = 0
        self.live_expiries = 0
        self.verifies = 0
        # Insertion-ordered set of expired ids, oldest first.
        self._expired: dict = {}
        self._next_eviction_log = 1
        self._next_lookup_log = 1024

    def __len__(self) -> int:
        return len(self._slot_of)

    def __contains__(self, req_id) -> bool:
        return req_id in self._slot_of

    def verify_bases(self, req_ids, out) -> None:
        """Write each request's first q row into ``out``, or -1 if it has none.

        Only for a step whose verify reads them. A found row counts in
        ``verifies``, a miss only if the id expired, so a request that never
        claimed (a greedy one) counts nowhere. Must run before `claim_bases`
        for the same batch, which may hand the slot of a request outside the
        batch to a new one. Both run on the host
        before the forward; the verify's read is ordered before the propose
        that rewrites a slot because both run on the one compute stream.
        """
        rows = self.rows_per_slot
        found = list(map(self._slot_of.get, req_ids))
        out[: len(req_ids)] = [-1 if slot is None else slot * rows for slot in found]
        missing = found.count(None)
        self.verifies += len(found) - missing
        if missing and self._expired:
            pop = self._expired.pop
            self.live_expiries += sum(
                1
                for req_id, slot in zip(req_ids, found)
                if slot is None and pop(req_id, False)
            )
        lookups = self.verifies + self.live_expiries
        if lookups >= self._next_lookup_log:
            # Powers of two again: a handful of lines over a long run.
            self._next_lookup_log *= 2
            if self._log:
                logger.info(
                    "draft q slots: %d of %d lookups lost their q to an expiry "
                    "(%d expiries in all, %d slots)",
                    self.live_expiries,
                    lookups,
                    self.evictions,
                    self.num_slots,
                )

    def claim_bases(self, req_ids, out, skip=None) -> None:
        """Give every request a slot to draft into and write its first row.

        ``skip`` (optional, one bool per request) marks requests whose q is
        never read; they claim nothing and get -1.

        A request that already owns a slot keeps it and becomes the most
        recently drafted. The claiming members' entries are taken out of the
        map before anything is claimed, so a new request cannot expire one of
        them, and the slots handed out are distinct. A skipped member is not
        taken out and may be expired: harmless, since a greedy member owns no
        slot and a mid-chunk member's slot holds no q a verify reads (its
        final chunk claims again before its first verify). There is always
        enough to claim from, since the free list plus the entries left in the
        map hold ``num_slots`` minus the claiming members' own slots, and
        ``num_slots`` is at least the batch size.
        """
        rows = self.rows_per_slot
        slots = self._slot_of
        if skip is not None:
            keep = [i for i, s in enumerate(skip) if not s]
            out[: len(req_ids)] = [-1] * len(req_ids)
            req_ids = [req_ids[i] for i in keep]
        owned = [slots.pop(req_id, None) for req_id in req_ids]
        claimed = [self._claim() if slot is None else slot for slot in owned]
        # Re-inserted last, in batch order: the freshest entries.
        slots.update(zip(req_ids, claimed))
        if skip is None:
            out[: len(req_ids)] = [slot * rows for slot in claimed]
        else:
            for i, slot in zip(keep, claimed):
                out[i] = slot * rows

    def _claim(self) -> int:
        if self._free:
            return self._free.pop()
        self.evictions += 1
        if self.evictions >= self._next_eviction_log:
            # Powers of two: visible from the first one on, quiet under churn.
            self._next_eviction_log *= 2
            if self._log:
                logger.info(
                    "draft q slots: %d expiries so far (%d slots); an expired "
                    "request that is still running rejects its next first draft",
                    self.evictions,
                    self.num_slots,
                )
        victim = next(iter(self._slot_of))
        self._expired[victim] = True
        if len(self._expired) > self._EXPIRED_MEMORY * self.num_slots:
            del self._expired[next(iter(self._expired))]
        return self._slot_of.pop(victim)
