"""Which rows of the persistent draft-q store belong to which request.

Lossless speculative sampling verifies each draft token against the q it was
drawn from, a step after the propose that drew it. The scheduler reorders the
batch in between and may run prefill-only steps that propose for other
requests, so q cannot live at a batch slot: each request owns a block of
``rows_per_slot`` rows in the store for as long as it keeps drafting.

Pure host bookkeeping, deliberately free of torch, so the policy is testable
without a device. ``DSparkProposer`` owns one and turns its answers into the
row bases the draft head writes through and the accept kernel reads through.
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
    (``max_tokens=1``) requests can do. ``evictions`` counts expiries of
    entries that were drafted for, so a live-eviction rate shows up as a
    number rather than as an unexplained acceptance dip.
    """

    def __init__(self, num_slots: int, rows_per_slot: int):
        if num_slots <= 0 or rows_per_slot <= 0:
            raise ValueError(
                f"need positive slots and rows, got {num_slots} x {rows_per_slot}"
            )
        self.num_slots = num_slots
        self.rows_per_slot = rows_per_slot
        # A plain dict is insertion-ordered; `claim_bases` re-inserts, so the
        # first key is always the least recently drafted request.
        self._slot_of: dict = {}
        self._free = list(range(num_slots - 1, -1, -1))
        self.evictions = 0
        self._next_eviction_log = 1

    def __len__(self) -> int:
        return len(self._slot_of)

    def __contains__(self, req_id) -> bool:
        return req_id in self._slot_of

    def verify_bases(self, req_ids, out) -> None:
        """Write each request's first q row into ``out``, or -1 if it has none.

        Must run before `claim_bases` for the same batch: a claim can expire a
        request of this very batch, and its verify still has to read the slot
        its drafts came from. That read is ordered before the propose that
        rewrites the slot because both run on the one compute stream, verify
        first.
        """
        rows = self.rows_per_slot
        slots = self._slot_of
        for i, req_id in enumerate(req_ids):
            slot = slots.get(req_id)
            out[i] = -1 if slot is None else slot * rows

    def claim_bases(self, req_ids, out) -> None:
        """Give every request a slot to draft into and write its first row.

        A request that already owns a slot keeps it and becomes the most
        recently drafted. Slots handed out within one call are distinct: each
        claim makes its request the freshest entry, and with ``num_slots`` at
        least the batch size an older entry is always there to expire first.
        """
        rows = self.rows_per_slot
        slots = self._slot_of
        for i, req_id in enumerate(req_ids):
            slot = slots.pop(req_id, None)
            if slot is None:
                slot = self._claim()
            slots[req_id] = slot
            out[i] = slot * rows

    def _claim(self) -> int:
        if self._free:
            return self._free.pop()
        self.evictions += 1
        if self.evictions >= self._next_eviction_log:
            # Powers of two: visible from the first one on, quiet under churn.
            self._next_eviction_log *= 2
            logger.info(
                "draft q slots: %d expiries so far (%d slots); an expired "
                "request that is still running rejects its next first draft",
                self.evictions,
                self.num_slots,
            )
        return self._slot_of.pop(next(iter(self._slot_of)))
