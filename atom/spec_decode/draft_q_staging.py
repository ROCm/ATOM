# SPDX-License-Identifier: MIT
"""The host half of a lossless draft step.

`DSparkLossless` owns a `DraftQStaging`, which owns the `DraftQSlots` and
writes the row bases the draft head writes through and the accept kernel reads
through.
"""

from typing import NamedTuple

import numpy as np

from atom.spec_decode.draft_q_slots import DraftQSlots


class StagedRows(NamedTuple):
    """How many leading rows of each staged buffer to publish (None: none)."""

    seed: int
    q_row_base: int | None
    verify_q_row_base: int | None


class DraftQStaging:
    """The host half of a lossless draft step: what to stage, row by row.

    Writes, into numpy views of `DSparkLossless`'s pinned staging buffers, the
    inputs the captured draft head and the ratio verify read beyond the
    target's own (temperatures and filters, which the draft reads from the
    runner): the Philox step, each row's q destination and each request's
    verify q row. `stage` returns how many leading rows of each buffer to
    publish (`StagedRows`); `DSparkLossless.stage` turns those into the
    runner's publication counts. Free of torch and aiter, so the policy is
    testable on any host.
    """

    def __init__(
        self,
        slots: DraftQSlots,
        *,
        seed,
        q_row_base,
        verify_q_row_base,
        greedy_threshold: float,
    ):
        self.slots = slots
        self._seed = seed
        self._dest = q_row_base
        self._verify = verify_q_row_base
        # `SAMPLER_EPS`: a row at or below it is greedy, as `Sampler` decides.
        self._eps = greedy_threshold
        self._step = 0
        # Leading destination rows that may still name a slot.
        self._dest_hi = 0
        # Staged since the last propose consumed a staging.
        self._fresh = False
        # Some row of the staged step names a q destination.
        self._draws_q = False

    def stage(self, batch, bs: int, *, drafts_verifiable: bool, verifies: bool):
        """Stage one step and return the rows to publish (`StagedRows`).

        ``drafts_verifiable``: this step's propose emits drafts a later verify
        may read. False for a dummy forward and an align-only middle chunk:
        those claim no slot and write no q, so their ids (warmup's 0..n-1, the
        DP dummy's -1) never enter the slot map.

        ``verifies``: this forward's verify reads q: it scores drafts
        (`ModelRunner._verifies_drafts`) and the batch is not all greedy, the
        gate `postprocess` asks `DSparkLossless.verify_inputs` under. False for
        a prefill, align-only, dummy or all-greedy forward: no lookup is made
        and no verify row is published, so the device rows keep an earlier
        step's bases, which nothing reads.
        """
        # The Philox key. A plain SPMD counter, so every TP rank reaches the
        # same number without exchanging it.
        self._seed[0] = self._step
        self._step = (self._step + 1) & 0x7FFFFFFF
        self._fresh = True
        req_ids = batch.req_ids[:bs]
        # Verify bases before claims: a claim may hand a slot to a new request,
        # and a request's verify must still read the slot its drafts came from.
        verify_rows = None
        if verifies:
            self.slots.verify_bases(req_ids, self._verify)
            verify_rows = bs or None
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
            if batch.is_final_chunk is not None:
                skip |= ~np.asarray(batch.is_final_chunk[:bs], dtype=bool)
            self.slots.claim_bases(req_ids, self._dest, skip if skip.any() else None)
            self._dest[bs:hi] = -1
            self._dest_hi = bs
            self._draws_q = not skip.all()
        else:
            self._dest[:hi] = -1
            self._dest_hi = 0
            self._draws_q = False
        return StagedRows(seed=1, q_row_base=hi or None, verify_q_row_base=verify_rows)

    @property
    def draws_q(self) -> bool:
        """Whether some row of the staged step names a q destination. False
        for an all-greedy, dummy or align-only step, or one whose every row is
        mid-prompt: its draft writes no q, so the greedy block drafts the same
        ids."""
        return self._draws_q

    def consume(self) -> None:
        """Claim the staging for the propose about to run, or refuse one that
        an earlier propose already drafted through: a second propose without
        a `stage` would draft through stale destination rows, into slots other
        requests may own by now. Every forward path stages
        (`ModelRunner.prepare_sample`) before it proposes."""
        if not self._fresh:
            raise RuntimeError(
                "propose without a fresh DSparkLossless.stage: the q row bases "
                "are an earlier propose's"
            )
        self._fresh = False

    def clear_destinations(self) -> None:
        """Point every host-side destination row at no q row."""
        self._dest[:] = -1
        self._dest_hi = 0
        self._draws_q = False
