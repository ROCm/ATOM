"""Which rows of the persistent draft-q store belong to which request.

Lossless speculative sampling verifies each draft token against the q it was
drawn from, a step after the propose that drew it. The scheduler reorders the
batch in between and may run prefill-only steps that propose for other
requests, so q cannot live at a batch slot: each request owns a block of
``rows_per_slot`` rows in the store for as long as it keeps drafting.

Pure host bookkeeping, deliberately free of torch and aiter, so the policy is
testable without a device. ``DSparkProposer`` owns a `DraftQStaging`, which
owns the `DraftQSlots` and writes the row bases the draft head writes through
and the accept kernel reads through.
"""

import logging

import numpy as np

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
    asks for none on a step that verifies nothing (prefill, align-only), so a
    preempted request re-prefilled in chunks adds nothing until its first
    verify, which reads the slot its final chunk claimed. Expired ids are
    remembered for the last ``_EXPIRED_MEMORY * num_slots`` expiries only; a
    request that waits out more than that is missed, so under extreme churn
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

        Only for a step whose verify reads them: every call is counted. Must
        run before `claim_bases` for the same batch, which may hand the slot
        of a request outside the batch to a new one. Both run on the host
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
        recently drafted. The batch's own entries are taken out of the map
        before anything is claimed, so a new request can only expire a request
        outside this batch: every member keeps its slot, and the slots handed
        out are distinct. That always leaves enough to claim from, since the
        free list plus the entries left in the map hold ``num_slots`` minus the
        members' own slots, and ``num_slots`` is at least the batch size.
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


class DraftQStaging:
    """The host half of a lossless draft step: what to stage, row by row.

    Writes, into numpy views of the proposer's pinned staging buffers, the
    inputs the captured draft head and the ratio verify read: per-row draft
    temperature, greedy mask and top-k / top-p, the Philox step, each row's q
    destination and each request's verify q row. `stage` returns how many
    leading rows of each buffer to publish; the proposer turns those into its
    publication counts. Free of torch and aiter, so the policy is testable on
    any host.
    """

    def __init__(
        self,
        slots: DraftQSlots,
        *,
        temperatures,
        greedy_mask,
        top_k,
        top_p,
        seed,
        dest_row_base,
        verify_row_base,
        greedy_threshold: float,
    ):
        self.slots = slots
        self._temps = temperatures
        self._mask = greedy_mask
        self._top_k = top_k
        self._top_p = top_p
        self._seed = seed
        self._dest = dest_row_base
        self._verify = verify_row_base
        # `SAMPLER_EPS`: a row at or below it is greedy, as `Sampler` decides.
        self._eps = greedy_threshold
        self._step = 0
        # Leading destination rows that may still name a slot.
        self._dest_hi = 0
        # The forwards that last staged and last consumed a staging.
        self._staged = None
        self._consumed = None

    def stage(
        self, batch, bs: int, *, drafts_verifiable: bool, verifies: bool, step: int
    ):
        """Stage one step and return ``{buffer: rows to publish or None}``.

        ``drafts_verifiable``: this step's propose emits drafts a later verify
        may read. False for a dummy forward and an align-only middle chunk:
        those claim no slot and write no q, so their ids (warmup's 0..n-1, the
        DP dummy's -1) never enter the slot map.

        ``verifies``: this forward's verify scores drafts and so may read q
        (`ModelRunner._verifies_drafts`). False for a prefill, align-only or
        dummy forward: no lookup is made, so none is counted.

        ``step``: the forward this staging is for (`ModelRunner.forward_step`),
        which `consume` checks.
        """
        # The Philox key. A plain SPMD counter, so every TP rank reaches the
        # same number without exchanging it.
        self._seed[0] = self._step
        self._step = (self._step + 1) & 0x7FFFFFFF
        self._staged = step
        req_ids = batch.req_ids[:bs]
        # Verify bases before claims: a claim may hand a slot to a new request,
        # and a request's verify must still read the slot its drafts came from.
        if verifies and not batch.is_dummy_run:
            self.slots.verify_bases(req_ids, self._verify)
        else:
            self._verify[:bs] = -1
        # Every row the block runs must name a destination, the pad rows up to
        # `running_bs` included, and a pad row left holding a larger previous
        # batch's base would write its q into a live request's slot. So clear
        # back down to wherever the last claim reached.
        hi = max(bs, self._dest_hi)
        if drafts_verifiable and bs:
            # Per request, rows whose q nobody reads claim no slot and write no
            # q (the draft still emits the argmax for them):
            # * a greedy request: its verify is exact match against the target
            #   argmax;
            # * a request still mid-prompt in a mixed prefill batch: its drafts
            #   are dropped, like a whole align-only batch's.
            skip = batch.temperatures[:bs] <= self._eps
            final_chunk = getattr(batch, "is_final_chunk", None)
            if final_chunk is not None:
                skip |= ~np.asarray(final_chunk[:bs], dtype=bool)
            self.slots.claim_bases(req_ids, self._dest, skip if skip.any() else None)
            self._dest[bs:hi] = -1
            self._dest_hi = bs
        else:
            self._dest[:hi] = -1
            self._dest_hi = 0
        counts = {
            "seed": 1,
            "temperatures": bs or None,
            "greedy_mask": bs or None,
            "top_k": bs or None,
            "top_p": bs or None,
            "verify_row_base": bs or None,
            "dest_row_base": hi or None,
        }
        if not bs:
            return counts
        # The clamp `prepare_sample` applies to the target's copy, so the two
        # divide by the same number row for row and no row divides by zero.
        np.maximum(batch.temperatures[:bs], self._eps, out=self._temps[:bs])
        # The predicate `Sampler` tests, so a row cannot be greedy to the
        # target and stochastic to the draft.
        self._mask[:bs] = batch.temperatures[:bs] <= self._eps
        # The request's own filters, so the draft samples about what the
        # target keeps. Only acceptance rides on it: the verify filters p
        # itself. The draft spells "no top-k" 0, a request -1.
        np.maximum(batch.top_ks[:bs], 0, out=self._top_k[:bs])
        self._top_p[:bs] = batch.top_ps[:bs]
        return counts

    def consume(self, step: int) -> None:
        """Claim forward ``step``'s staging for the propose about to run, or
        refuse.

        A propose whose forward skipped `stage` would draft through an earlier
        forward's destination rows, into slots other requests may own by now.
        That earlier forward need not have proposed (a middle chunk stages and
        returns first), so what is checked is that this very forward staged,
        and only once consumed.
        """
        assert self._staged == step and self._consumed != step, (
            f"propose in forward {step} without a fresh "
            f"stage_draft_sampling_host (last staged {self._staged}, last "
            f"consumed {self._consumed}): the q row bases are another forward's"
        )
        self._consumed = step

    def clear_destinations(self) -> None:
        """Point every host-side destination row at no q row."""
        self._dest[:] = -1
        self._dest_hi = 0
