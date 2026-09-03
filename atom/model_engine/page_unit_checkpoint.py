# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""State checkpoints backed by arbitrary PAGE-sized physical units."""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from time import monotonic

from atom.kv_transfer.disaggregation.types import StateStoreOperationId
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.sequence import Sequence

logger = logging.getLogger("atom")

#: How many reclaimed store operations to remember. Only has to outlive a
#: report already on the wire, which is one step, so this is generous.
_RECLAIMED_MEMORY = 4096

COPYING = "COPYING"
READY = "READY"
EVICTING = "EVICTING"


@dataclass(frozen=True)
class PagedStateCheckpointSpec:
    """Runtime geometry for PAGE-backed state checkpoints."""

    page_unit_bytes: int
    slot_bytes: int
    layout_id: str
    # Bytes of a slot a checkpoint image actually holds, which is less than
    # all of them: a resumer reads only part of the slot it resumes into, and
    # a compressor whose next pool starts exactly at the boundary reads none
    # of its own. `slot_bytes` stays for three things that still want the
    # whole slot — the `image_bytes <= slot_bytes` sanity check below, the
    # geometry cross-check in `allocate_per_req_cache`, and the startup log
    # line that reports an image as a fraction of one.
    image_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("page_unit_bytes", self.page_unit_bytes),
            ("slot_bytes", self.slot_bytes),
            ("image_bytes", self.image_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_bytes > self.slot_bytes:
            raise ValueError(
                f"image_bytes {self.image_bytes} exceeds the {self.slot_bytes} "
                "a slot holds"
            )
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("paged state checkpoints need a non-empty layout id")

    @property
    def units_per_checkpoint(self) -> int:
        return (self.image_bytes + self.page_unit_bytes - 1) // self.page_unit_bytes

    def to_wire(self) -> dict[str, int | str]:
        return {
            "page_unit_bytes": self.page_unit_bytes,
            "slot_bytes": self.slot_bytes,
            "image_bytes": self.image_bytes,
            "layout_id": self.layout_id,
        }

    @classmethod
    def from_wire(cls, wire: object) -> PagedStateCheckpointSpec:
        if not isinstance(wire, Mapping):
            raise TypeError("paged state checkpoint spec must be a mapping")
        expected = {"page_unit_bytes", "slot_bytes", "image_bytes", "layout_id"}
        if set(wire) != expected:
            raise ValueError(
                "invalid paged state checkpoint spec fields: "
                f"expected={sorted(expected)}, got={sorted(wire)}"
            )
        return cls(
            page_unit_bytes=wire["page_unit_bytes"],  # type: ignore[arg-type]
            slot_bytes=wire["slot_bytes"],  # type: ignore[arg-type]
            image_bytes=wire["image_bytes"],  # type: ignore[arg-type]
            layout_id=wire["layout_id"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CheckpointStoreOp:
    """Scatter the checkpointed part of an Active Slot into PAGE units."""

    src_slot: int
    unit_ids: tuple[int, ...]
    total_bytes: int
    layout_id: str


@dataclass(frozen=True)
class CheckpointRestoreOp:
    """Gather one ordered PAGE-unit image back into an Active Slot."""

    dst_slot: int
    unit_ids: tuple[int, ...]
    total_bytes: int
    layout_id: str


@dataclass
class CheckpointRecord:
    prefix_hash: int
    unit_ids: tuple[int, ...]
    state: str = COPYING
    pin_count: int = 0


@dataclass
class _OffloadPin:
    """A store that has been dispatched to the worker but not yet indexed.

    Two-phase, because a store's life has two independent ends. The PAGE units
    it gathers are the KV pool's and are free the moment the D2H gather drains
    (`source_released`); whether the CPU `batched_put` then succeeded is decided
    strictly afterwards and cannot touch them. The pin therefore holds the units
    only until the source is released, but the *record itself* stays in
    `_offload_pins` until the index report lands -- so `has_offload_pins` and
    `_hash_in_flight`, which mean "dispatched but unreported", keep answering
    from dispatch all the way to the report, across the gap the source release
    opens in the middle.
    """

    checkpoint_id: int
    pinned_at: float
    source_released: bool = False


class PageUnitCheckpointStore:
    """Content index and ownership table for split state images."""

    def __init__(
        self,
        pool: BlockPool,
        spec: PagedStateCheckpointSpec,
        offload_sink: bool = False,
    ):
        self.pool = pool
        self.spec = spec
        # Whether anything downstream can carry a store. False leaves the
        # offload queue permanently empty and takes no pins, which is what a
        # deployment with no CPU tier must cost: a pin nobody releases would
        # hold 127 blocks per checkpoint out of the pool forever.
        self._offload_sink = offload_sink
        self.hash_to_checkpoint: dict[int, int] = {}
        self.records: dict[int, CheckpointRecord] = {}
        self._pending_by_hash: dict[int, int] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._inflight_stores: list[int] = []
        self._queued_restores: list[tuple[int, CheckpointRestoreOp]] = []
        self._inflight_restores: list[int] = []
        self._next_checkpoint_id = 0
        self.evictions = 0
        # Checkpoints that reached READY and have not been offered to the CPU
        # tier yet. Candidates only: NOT pinned, so one still counts as free
        # space and `_next_victim` may spend it. The pin is taken later, in
        # `take_offload_stores`, and only for the few actually in flight.
        self._offload_ready: deque[int] = deque()
        # Ceiling on that backlog. Under sustained tier backpressure the drain
        # loop in `take_offload_stores` exits at its first iteration -- pins
        # already at `max_inflight` -- and so never reaches the stale-drop that
        # otherwise clears spent nominations. An unbounded nomination stream
        # would then grow `_offload_ready` without limit with ids whose
        # checkpoints have long since been evicted. A nomination is unpinned and
        # the pool may spend its checkpoint anyway, so dropping the oldest
        # (coldest) on overflow costs a missed offload, never a lost image: the
        # checkpoint stays HBM-resident until it is evicted. Sized well above
        # any real in-flight-plus-waiting count so it only bites a runaway.
        self._offload_backlog_cap = 8192
        self.offload_nominations_dropped = 0
        # StateStoreOperationId -> _OffloadPin (checkpoint, when pinned, whether
        # the source units are already back).
        # A store's report carries no request -- by the time one lands its owner
        # is long gone -- so the hash is the only thing left to name it by. The
        # hash ALONE is not an identity though: the same prefix is stored again
        # after an eviction or a load miss, and `KVOutputAggregator` tombstones
        # every `(channel, operation_id)` it has taken quorum on, so the second
        # store under a bare hash was dropped as a duplicate -- its pin held
        # until stale reclamation and its bytes never re-indexed. The
        # generation separates the attempts, and keying the pins by the whole
        # operation is also what stops a late report from an earlier attempt
        # from settling a newer pin for the same hash.
        self._offload_pins: dict[StateStoreOperationId, _OffloadPin] = {}
        self._offload_generation = 0
        self.offload_pins_reclaimed = 0
        # Operations whose pin the stale reclaimer took back. Bounded, because
        # this only has to outlive a report already on the wire. A store whose
        # source was reclaimed underneath it may not be indexed -- see
        # `was_reclaimed`.
        self._offload_reclaimed: OrderedDict = OrderedDict()

    @property
    def units_per_checkpoint(self) -> int:
        return self.spec.units_per_checkpoint

    def lookup(self, prefix_hash: int) -> int:
        checkpoint_id = self.hash_to_checkpoint.get(prefix_hash, -1)
        record = self.records.get(checkpoint_id)
        if record is None or record.state != READY:
            return -1
        return checkpoint_id

    def contains(self, prefix_hash: int) -> bool:
        return self.lookup(prefix_hash) >= 0

    def contains_or_pending(self, prefix_hash: int) -> bool:
        return self.contains(prefix_hash) or prefix_hash in self._pending_by_hash

    def _new_identity(self) -> int:
        checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1
        return checkpoint_id

    def _is_evictable(self, checkpoint_id: int, protected: int = -1) -> bool:
        """Whether this checkpoint may be spent. Eligibility, not policy.

        The one statement of what is evictable. Everything that asks about
        free units goes through it, so a new state, a grace period or a
        second kind of pin cannot leave two answers behind. Do not order
        here -- which eligible checkpoint to spend first is `_next_victim`.
        """
        record = self.records[checkpoint_id]
        return (
            checkpoint_id != protected
            and record.state == READY
            and record.pin_count == 0
        )

    def _evictable(self, protected: int = -1) -> Iterator[int]:
        """Every checkpoint that may be spent. Yield order carries no promise."""
        return (cid for cid in self._lru if self._is_evictable(cid, protected))

    def _next_victim(self, protected: int = -1) -> int:
        """Which eligible checkpoint to spend when the free list is short.

        This is the eviction policy, and the only place it lives: least
        recently used, which `_lru` already orders. A different policy
        replaces this method and nothing else -- in particular it must not
        touch `_is_evictable`, which is the eligibility rule three callers
        share.
        """
        return next(self._evictable(protected), -1)

    def has_available_units(
        self, count: int, protected_hash: int | None = None
    ) -> bool:
        """Whether `count` units could be had, evicting if it came to that.

        Asked once per waiting sequence in `can_allocate` and once per running
        one in `can_append`, so it is per-sequence per-pass and the walk has to
        be paid for. Two things keep it cheap. The free list is checked first,
        which is the whole answer whenever the pool is not tight. And the walk
        below stops at the shortfall rather than totalling the cache: the
        question is whether the eligible set reaches `count`, not how large it
        is, and a warm pool holds `num_kvcache_blocks / units_per_checkpoint`
        checkpoints -- thousands, walked for an answer a couple of them settle.

        Which checkpoints those are does not change the answer, only how soon
        the loop reaches it, so a future `_next_victim` cannot move this gate.
        """
        if count <= self.pool.num_free:
            return True
        protected = self.lookup(protected_hash) if protected_hash is not None else -1
        shortfall = count - self.pool.num_free
        for checkpoint_id in self._evictable(protected):
            shortfall -= len(self.records[checkpoint_id].unit_ids)
            if shortfall <= 0:
                return True
        return False

    def ensure_free_units(self, count: int) -> bool:
        """Raise the free list to `count`, spending checkpoints for the shortfall.

        Free units are taken first -- a caller asking for what is already
        there evicts nothing -- and `pop` hands out never-used blocks before
        cached ones, so a store reaches for the cache only once the pool has
        nothing spare. Each eviction returns a whole image's units, so the
        loop overshoots by at most one checkpoint.

        Unreachable counts are refused before anything is spent. The loop
        alone gives up only once it has evicted everything it can, so a count
        the cache cannot reach would destroy the cache on the way to saying
        no. The test lives here rather than in the one caller that used to
        carry it, because every caller needs it and only the argument being
        1 keeps `_fresh_block` from needing it today.
        """
        if not self.has_available_units(count):
            return False
        while self.pool.num_free < count:
            victim = self._next_victim()
            if victim < 0:
                return False
            self._evict(victim)
        return True

    def begin_store(self, prefix_hash: int, src_slot: int) -> CheckpointStoreOp | None:
        if self.lookup(prefix_hash) >= 0 or prefix_hash in self._pending_by_hash:
            return None
        needed = self.units_per_checkpoint
        # A store takes what its own image needs and nothing more. It used to
        # take a floor for live KV on top, which meant one accepted store
        # spent tens of checkpoints to build a cushion -- and the cushion
        # bought nothing: the pool cannot starve live KV. A READY unpinned
        # checkpoint is already counted as available by `has_available_units`,
        # so holding one costs live KV nothing; the unevictable set (COPYING,
        # or pinned by a restore) is created after every allocation in a pass
        # and resolved before the next one allocates; and every `_fresh_block`
        # sits behind a pin-aware check in its own pass, so the reachable
        # outcome is a refused admission, never the raise.
        #
        # A store that will be dropped has to cost nothing, which is what
        # `ensure_free_units` refusing before it evicts buys. Its answer is
        # read rather than assumed: `_next_victim` is meant to be replaced,
        # and a policy that passes over an eligible checkpoint would leave the
        # loop short after spending some -- taking an identity and a record
        # for a store that cannot happen would then be the second cost.
        if not self.ensure_free_units(needed):
            return None

        checkpoint_id = self._new_identity()
        owner = ("state-checkpoint", checkpoint_id)
        unit_ids = self.pool.reserve_units(needed, owner)
        if unit_ids is None:
            return None
        record = CheckpointRecord(
            prefix_hash=prefix_hash,
            unit_ids=tuple(unit_ids),
        )
        self.records[checkpoint_id] = record
        self._pending_by_hash[prefix_hash] = checkpoint_id
        self._inflight_stores.append(checkpoint_id)
        return CheckpointStoreOp(
            src_slot=src_slot,
            unit_ids=record.unit_ids,
            total_bytes=self.spec.image_bytes,
            layout_id=self.spec.layout_id,
        )

    def restore_queued_for(self, dst_slot: int) -> bool:
        """Whether a queued restore will write `dst_slot` on the next batch."""
        return any(op.dst_slot == dst_slot for _cid, op in self._queued_restores)

    def begin_restore(
        self, prefix_hash: int, dst_slot: int
    ) -> CheckpointRestoreOp | None:
        checkpoint_id = self.lookup(prefix_hash)
        if checkpoint_id < 0:
            return None
        record = self.records[checkpoint_id]
        record.pin_count += 1
        self._lru.move_to_end(checkpoint_id)
        op = CheckpointRestoreOp(
            dst_slot=dst_slot,
            unit_ids=record.unit_ids,
            total_bytes=self.spec.image_bytes,
            layout_id=self.spec.layout_id,
        )
        self._queued_restores.append((checkpoint_id, op))
        return op

    def take_restore_ops(self) -> tuple[CheckpointRestoreOp, ...]:
        queued, self._queued_restores = self._queued_restores, []
        self._inflight_restores.extend(checkpoint_id for checkpoint_id, _ in queued)
        return tuple(op for _, op in queued)

    def cancel_queued_restore(self, dst_slot: int) -> None:
        kept: list[tuple[int, CheckpointRestoreOp]] = []
        for checkpoint_id, op in self._queued_restores:
            if op.dst_slot == dst_slot:
                self._release_restore_pin(checkpoint_id)
            else:
                kept.append((checkpoint_id, op))
        self._queued_restores = kept

    def complete_inflight(self) -> None:
        stores, self._inflight_stores = self._inflight_stores, []
        for checkpoint_id in stores:
            record = self.records.get(checkpoint_id)
            if record is None:
                continue
            if self._pending_by_hash.get(record.prefix_hash) == checkpoint_id:
                del self._pending_by_hash[record.prefix_hash]
            if record.state == EVICTING:
                self._release_record(checkpoint_id)
                continue
            if record.state != COPYING:
                continue
            # Publish only after the scatter has ridden a batch.
            if self.lookup(record.prefix_hash) >= 0:
                self._release_record(checkpoint_id)
                continue
            record.state = READY
            self.hash_to_checkpoint[record.prefix_hash] = checkpoint_id
            self._lru[checkpoint_id] = None
            self._queue_offload_store(checkpoint_id, record)

        restores, self._inflight_restores = self._inflight_restores, []
        for checkpoint_id in restores:
            self._release_restore_pin(checkpoint_id)

    # ------------------------------- offload ------------------------------- #
    def _queue_offload_store(self, checkpoint_id: int, record) -> None:
        """Nominate this checkpoint for the CPU tier. Takes no pin.

        READY and nowhere else: `begin_store` is too early (the scatter has not
        ridden a batch, so the units hold no image yet) and `_evict`/`unindex`
        are too late (they fire when the pool wants those units *now*). READY is
        when the bytes first exist and when the record is least wanted -- it has
        just entered the LRU at the cold end.

        Nomination, not reservation: a queued candidate is unpinned, so
        `_next_victim` may still spend it. The pool never waits on the CPU tier.
        Dedup is free -- `begin_store` refuses a hash it already holds.
        """
        if not self._offload_sink:
            return
        self._offload_ready.append(checkpoint_id)
        # Bound the backlog (see `_offload_backlog_cap`). popleft-on-overflow
        # rather than a `deque(maxlen=)`: `take_offload_stores` re-queues
        # deferred nominations with `extendleft`, and a maxlen deque would evict
        # from whichever end an append pushed against -- dropping the freshly
        # re-queued oldest entries we take pains to preserve there.
        if len(self._offload_ready) > self._offload_backlog_cap:
            if self.offload_nominations_dropped == 0:
                logger.warning(
                    "state offload nomination backlog exceeded %d; dropping "
                    "oldest nominations. The CPU tier is draining checkpoints "
                    "slower than they reach READY -- offload is falling behind.",
                    self._offload_backlog_cap,
                )
            while len(self._offload_ready) > self._offload_backlog_cap:
                self._offload_ready.popleft()
                self.offload_nominations_dropped += 1

    @property
    def store_backlog(self) -> deque:
        """Nominations waiting for an in-flight slot. A gauge for the log."""
        return self._offload_ready

    def take_offload_stores(
        self, max_inflight: int
    ) -> list[tuple[StateStoreOperationId, tuple[int, ...]]]:
        """`(operation, unit_ids)` to hand the tier now, pinning each.

        The pin is taken HERE, not at READY, and that is what bounds it: a pin
        lives in this process while the D2H runs in the worker, so it spans
        several scheduler passes. Pinning at READY would make every checkpoint
        un-evictable for that window and break the admission invariant that a
        READY unpinned checkpoint counts as available to live KV.

        Over the cap, candidates wait rather than being dropped. Drained, not
        read: a second submission would store the same image twice and the
        second report would unpin a record the first released.

        Each hand-out gets its own generation, so a prefix stored again after
        an eviction is a different operation from the one that stored it
        before. Two attempts at the same hash may not be in flight at once --
        `_hash_in_flight` refuses that -- but they may follow one another
        closely enough that the earlier one's report is still on the wire.
        """
        out: list[tuple[StateStoreOperationId, tuple[int, ...]]] = []
        # Nominations whose hash is transiently in flight: held aside and
        # re-queued after the pass, NOT dropped (see below).
        deferred: list = []
        while self._offload_ready and len(self._offload_pins) < max_inflight:
            checkpoint_id = self._offload_ready.popleft()
            record = self.records.get(checkpoint_id)
            # Spent while it waited -- which nomination deliberately allows.
            # Terminal: there is nothing left to store, so drop it.
            if record is None or record.state != READY:
                continue
            if self._hash_in_flight(record.prefix_hash):
                # NOT terminal. The record is still READY and is a live, valid
                # nomination; its only problem is that an earlier generation of
                # the same hash is still on the wire (an eviction + re-store of
                # one prefix while the first store is mid-flight). Dropping it
                # here -- as a bare `continue` did -- lost this checkpoint image
                # from the store queue for good once the in-flight copy settled.
                # Defer it to a later pass instead. Held aside rather than
                # re-appended inside the loop so a hash that stays in flight
                # cannot spin this call.
                deferred.append(checkpoint_id)
                continue
            self._offload_generation += 1
            op = StateStoreOperationId(
                int(record.prefix_hash), self._offload_generation
            )
            record.pin_count += 1
            self._offload_pins[op] = _OffloadPin(checkpoint_id, monotonic())
            out.append((op, record.unit_ids))
        # Re-queue the deferred nominations for the next drain, once the
        # colliding in-flight generation has had a chance to settle. The drain
        # is `popleft()` (oldest at the left), so `extend` would append these
        # older, already-waiting nominations *behind* newer ones and let them
        # starve until they age out at the `state != READY` check. Put them
        # back at the front, preserving their original oldest-first order.
        if deferred:
            self._offload_ready.extendleft(reversed(deferred))
        return out

    def _hash_in_flight(self, prefix_hash: int) -> bool:
        """Whether some generation of `prefix_hash` is already pinned.

        Kept from when the pins were keyed by hash: two live stores of one
        image would copy the same bytes twice and the second report would unpin
        a record the first already released. The generation exists to tell
        *sequential* attempts apart, not to license concurrent ones.
        """
        return any(op.prefix_hash == prefix_hash for op in self._offload_pins)

    def release_offload_store_source(self, op: StateStoreOperationId) -> None:
        """Phase one: the gather drained, so hand the PAGE units back now.

        This is the earlier of a store's two ends. It returns the units to the
        KV pool the instant the D2H copy is done reading them -- holding an
        image out of the pool across the subsequent CPU `batched_put` would cost
        reuse for a step that cannot touch the units. But it does NOT remove the
        operation from `_offload_pins`: the store is still dispatched-but-
        unreported, so `has_offload_pins` and `_hash_in_flight` must keep
        seeing it until the index report lands. It only marks the source gone,
        so the report (or a reclaim) does not release the same units twice.

        Idempotent, and matched on the whole operation: a release for a
        generation no longer here -- already reported, or reclaimed -- does
        nothing.
        """
        pin = self._offload_pins.get(op)
        if pin is None or pin.source_released:
            return
        pin.source_released = True
        self._release_offload_pin(pin.checkpoint_id)

    def settle_offload_store(self, op: StateStoreOperationId) -> None:
        """Phase two: the store reported, either way. Retire the operation.

        Removes the pin so `has_offload_pins`/`_hash_in_flight` stop seeing this
        operation -- the report is the true end of "dispatched but unreported".
        The units are normally already back (`release_offload_store_source` ran
        first); this is the backstop for a store that failed or reported before
        its source was released, so the units are freed here only when they were
        not freed there. Whether the hash becomes reachable is
        `StateOffloadIndex`'s business, not this one's.

        Matched on the whole operation: a report from an attempt whose pin was
        already reclaimed names a generation no longer here and settles
        nothing, rather than releasing a newer attempt's pin.
        """
        pin = self._offload_pins.pop(op, None)
        if pin is None:
            return
        if not pin.source_released:
            self._release_offload_pin(pin.checkpoint_id)

    def has_offload_pins(self) -> bool:
        """Whether any dispatched store is still awaiting its index report.

        A pin lives here from the moment `take_offload_stores` hands a store to
        the worker until `settle_offload_store` retires it on the index report,
        or the last-resort `reclaim_stale_offload_pins` times it out. The source
        release in between hands the PAGE units back but leaves the pin, so this
        stays true across it: it is therefore true exactly while a store is
        dispatched-but-unreported, which is the interval the engine must keep
        polling across -- the report that clears the pin, and the reclaim that
        is its only other exit, both run only from `_poll_kv_transfer_progress`,
        which the idle loop skips once liveness reads False. Because reclaim
        shares this same dict, the signal cannot latch the busy loop: a lost
        report clears here when the reclaim fires.
        """
        return bool(self._offload_pins)

    def reclaim_stale_offload_pins(self, timeout_s: float) -> int:
        """Release pins whose report never came, after `timeout_s`.

        The pin lives here in the engine process and the D2H runs in the
        worker, so a lost report -- a crashed worker, a dropped completion --
        would hold a whole image out of the pool forever. This is the unit-side
        twin of `Scheduler._reconcile_stalled_deferred_saves`.

        **This is a last resort and it does not prove the reader stopped.** It
        used to be justified by LMCache's own pin timeout, on the grounds that
        by then upstream has force-unpinned and stopped reading. That argument
        does not hold for this source: a K3 state store bypasses
        `CacheEngine.store()` entirely and gathers ATOM PAGE units through
        `page_unit_views -> StagedTransfer.pack -> storage_manager.batched_put`,
        and those units are not covered by LMCache's GPU-source pin monitor.
        Nothing here can tell a lost report from a worker still inside the
        gather.

        What follows from that is not that the units may never be reclaimed --
        that would leak an image per dropped completion -- but that a store
        whose source was reclaimed can no longer be trusted: if the reader had
        not stopped, the pool may have handed those units to another request
        whose writes the gather then picked up, and the CPU image is a mix of
        two prefixes under the first one's hash. Resuming onto it is silent
        wrong output. So the operation is remembered here and
        `BlockManager.settle_state_store` refuses to index it. The reclaim
        recovers the memory and forfeits the entry, which is the only pair of
        outcomes this can honestly offer.
        """
        if timeout_s <= 0 or not self._offload_pins:
            return 0
        cutoff = monotonic() - timeout_s
        stale = [
            op for op, pin in self._offload_pins.items() if pin.pinned_at <= cutoff
        ]
        for op in stale:
            pin = self._offload_pins.pop(op)
            self.offload_pins_reclaimed += 1
            if pin.source_released:
                # The gather already drained and the units are back; only the
                # index report was lost. Nothing to release again -- doing so
                # would underflow the record -- and nothing to forfeit: the
                # bytes are trustworthy because the reader provably stopped, so
                # a late report may still index the hash. This is a lost report,
                # not a taken-back source.
                logger.warning(
                    "state offload: store %s never reported after %.1fs; its "
                    "units were already returned at source release, so only "
                    "the index report is lost.",
                    op,
                    timeout_s,
                )
                continue
            # The source was never released, so the worker may still be inside
            # the gather. Taking the units back now means they cannot be
            # trusted: the pool may hand them to another request whose writes
            # this gather then picks up. Release and forfeit the entry.
            self._release_offload_pin(pin.checkpoint_id)
            self._offload_reclaimed[op] = None
            while len(self._offload_reclaimed) > _RECLAIMED_MEMORY:
                self._offload_reclaimed.popitem(last=False)
            logger.warning(
                "state offload: reclaimed the units of store %s after %.1fs "
                "with no report. The worker may still be reading them, so the "
                "image will not be indexed even if a report arrives later.",
                op,
                timeout_s,
            )
        return len(stale)

    def was_reclaimed(self, op) -> bool:
        """Whether this store's source was taken back before it reported."""
        return op in self._offload_reclaimed

    def _release_offload_pin(self, checkpoint_id: int) -> None:
        record = self.records.get(checkpoint_id)
        if record is None:
            return
        if record.pin_count <= 0:
            raise AssertionError("checkpoint offload pin underflow")
        record.pin_count -= 1
        if record.state == EVICTING and record.pin_count == 0:
            # `unindex` fired during the copy: the boundary's KV block was
            # spent, so this image is unreachable in HBM from now on. The copy
            # was still worth finishing -- the CPU copy outliving the HBM one
            # is the entire point of the tier -- and only now do the units go
            # back.
            self._release_record(checkpoint_id)

    def _release_restore_pin(self, checkpoint_id: int) -> None:
        record = self.records.get(checkpoint_id)
        if record is None:
            return
        if record.pin_count <= 0:
            raise AssertionError("checkpoint restore pin underflow")
        record.pin_count -= 1
        if record.state == EVICTING and record.pin_count == 0:
            self._release_record(checkpoint_id)

    def unindex(self, prefix_hash: int) -> bool:
        checkpoint_id = self.hash_to_checkpoint.pop(prefix_hash, -1)
        if checkpoint_id < 0:
            checkpoint_id = self._pending_by_hash.pop(prefix_hash, -1)
        if checkpoint_id < 0:
            return False
        record = self.records.get(checkpoint_id)
        if record is None:
            return False
        record.state = EVICTING
        self._lru.pop(checkpoint_id, None)
        # Keep units alive while a queued GPU writer can still access them.
        if checkpoint_id not in self._inflight_stores and record.pin_count == 0:
            self._release_record(checkpoint_id)
        return True

    def clear(self) -> None:
        self.hash_to_checkpoint.clear()
        self._pending_by_hash.clear()
        self._lru.clear()
        inflight_stores = set(self._inflight_stores)
        for checkpoint_id in list(self.records):
            record = self.records[checkpoint_id]
            record.state = EVICTING
            if checkpoint_id not in inflight_stores and record.pin_count == 0:
                self._release_record(checkpoint_id)

    def _evict(self, checkpoint_id: int) -> None:
        record = self.records[checkpoint_id]
        if record.state != READY or record.pin_count:
            raise AssertionError("only an unpinned READY checkpoint is evictable")
        if self.hash_to_checkpoint.get(record.prefix_hash) == checkpoint_id:
            del self.hash_to_checkpoint[record.prefix_hash]
        record.state = EVICTING
        self._lru.pop(checkpoint_id, None)
        self._release_record(checkpoint_id)
        self.evictions += 1

    def _release_record(self, checkpoint_id: int) -> None:
        record = self.records.pop(checkpoint_id)
        self._lru.pop(checkpoint_id, None)
        if self.hash_to_checkpoint.get(record.prefix_hash) == checkpoint_id:
            del self.hash_to_checkpoint[record.prefix_hash]
        if self._pending_by_hash.get(record.prefix_hash) == checkpoint_id:
            del self._pending_by_hash[record.prefix_hash]
        self.pool.release_units(record.unit_ids, ("state-checkpoint", checkpoint_id))


class PagedStateCheckpointCoordinator:
    """Schedules PAGE-backed checkpoints for per-request state."""

    successor_room = 0.0
    # A PAGE image is written by a copy the runner issues after the forward that
    # produced the state, out of the slot that forward left behind. There are no
    # interior positions to slice: unlike a chunk kernel's `h`, the compressor
    # ring is not materialized at boundaries inside a step. So the engine keeps
    # cutting prefill chunks onto rungs for this class, and the three methods
    # below are the no-ops `StateCache` documents for a class that is not
    # readable midstep — present because `BlockManager` calls them across every
    # member of `state_caches` without asking which kind it holds.
    readable_midstep = False

    def reserve_midstep(self, seq, positions: list[tuple[int, int]]) -> list[tuple]:
        del seq, positions
        return []

    def publish_midstep(self, reservations: list[tuple], seq=None) -> None:
        del reservations, seq

    def cancel_midstep(self, reservations: list[tuple]) -> None:
        del reservations

    def __init__(
        self,
        pool: BlockPool,
        spec: PagedStateCheckpointSpec,
        enabled: bool,
        offload=None,
    ) -> None:
        self.enabled = enabled
        # The CPU tier beneath this one, or None. Not a member of
        # `BlockManager.state_caches`: every one of those is a veto over where a
        # prefix may resume, and this does the opposite -- it makes MORE
        # boundaries reachable. It is consulted in exactly two places,
        # `resumable_hit` (the vote) and `take_offload_stores` (the sink).
        self.offload = offload
        self.store = PageUnitCheckpointStore(
            pool, spec, offload_sink=offload is not None
        )
        # Keyed by `(seq id, prefix hash)` rather than by seq: two boundaries of
        # one prompt are two checkpoints, and keying by seq alone let the later
        # one overwrite the earlier before either was stored. Re-reaching the
        # *same* hash still collapses, which is what the hash in the key is
        # for -- that is one boundary reached twice, not two boundaries.
        self._pending: dict[tuple[int, int], tuple[Sequence, int]] = {}
        self._store_ops: list[CheckpointStoreOp] = []
        self.checkpoints_kept = 0
        self.checkpoints_dropped = 0
        self.checkpoints_orphaned = 0

    def applies(self, seq: Sequence) -> bool:
        return self.enabled and seq.has_per_req_cache

    def resumable_hit(
        self,
        seq: Sequence,
        hit: int,
        block_hashes: list[int],
        assume_checkpointed: bool = False,
    ) -> int:
        if not self.applies(seq):
            return hit
        for i in range(hit - 1, -1, -1):
            if assume_checkpointed or self._reachable(block_hashes[i]):
                return i + 1
        return 0

    def _reachable(self, h: int) -> bool:
        """Whether `h`'s image can be *produced*, in HBM or from the CPU tier.

        The scan runs right to left and stops at the first boundary it accepts,
        so accepting one nothing can deliver costs the whole walk-back, hiding
        every shorter checkpoint still in HBM. No preference rule is needed:
        both tiers key on the same content hash and `_attach_state_slots` tries
        HBM first regardless.

        The tier's half is deliberately optimistic -- `hashes` means "was stored
        once", never "is still there". A false positive costs one park plus a
        recompute and retracts itself (`fail_load` -> `forget`); being certain
        would cost a synchronous cross-process lookup on the admission path.

        The tier's half goes through `could_serve`, not a bare `h in hashes`, so
        it also honours `can_load`: a store-only role (`kv_producer`) populates
        `hashes` from its own stores but cannot load them back, and voting a hit
        it would then refuse strands the scan at the tier rung -- skipping a
        still-resident HBM rung -- and forfeits the checkpoint to a recompute.
        """
        if self.store.contains(h):
            return True
        return self.offload is not None and self.offload.could_serve(h)

    def checkpoint(self, seq: Sequence, boundary_blocks: int, h: int) -> None:
        """File a boundary to be stored, keyed by hash rather than by seq.

        Every boundary a seq reaches survives, not just its last: an anchor and
        the prompt-end checkpoint that follows it a chunk later are separate
        entries. Keying by seq alone would have the second overwrite the first
        before either is stored, which costs a shortened prefill chunk on every
        prompt and buys nothing. What makes keeping both affordable is the
        image's price against a whole Active Slot under `fork`.

        The anchor is the placement that pays, and the ladder is not (see
        `BlockManager._record_checkpoint_end`), so
        `--state-checkpoint-interval-tokens -1` drops the grid and leaves the
        anchor and the demand rung as the only two placements.

        One entry per seq per drain, though, and that is not the same as one
        per seq. A pending boundary names a hash and the slot that will be read
        for it, and the slot is read at the drain -- so two boundaries surviving
        into one drain would both be stored from whatever the *last* forward
        left there, filing the earlier hash over the later state. A request
        resuming on it would continue from a point ahead of its own prefix, and
        nothing downstream could tell: `_validate_paged_state_op` checks layout,
        size and unit count, all of which still match.

        A drain normally follows every forward, so the two boundaries of one
        prompt are ordinarily stored from separate slots correctly. The
        exception is a pass that schedules nothing (`scheduler.py:1828` passes
        `state_maintenance_ops=None` on an empty batch), which carries
        `_pending` into the next drain. `_supersede` resolves that the only way
        the bytes allow: the newer boundary is the one the slot actually holds,
        so it wins and the older is dropped rather than mis-stored.
        """
        del boundary_blocks
        if self.applies(seq) and seq.state_slot >= 0:
            self._supersede(id(seq))
            self._pending[(id(seq), h)] = (seq, h)

    def _supersede(self, seq_id: int) -> None:
        """Drop this seq's earlier pending boundaries; the slot has moved on.

        Counted as dropped, not silently forgotten: this is reuse the placement
        asked for and did not get, and it is the only signal that empty passes
        are costing checkpoints.
        """
        stale = [k for k in self._pending if k[0] == seq_id]
        for key in stale:
            del self._pending[key]
        self.checkpoints_dropped += len(stale)

    def forget_pending(self, seq: Sequence) -> None:
        """Drop every boundary this seq had pending, not just its last.

        A seq can now hold several. All of them describe state in the slot
        that is about to go back on the free list, so all of them die with it.
        """
        seq_id = id(seq)
        for key in [k for k in self._pending if k[0] == seq_id]:
            del self._pending[key]
        self.store.cancel_queued_restore(seq.state_slot)

    def restore_queued_for(self, dst_slot: int) -> bool:
        """Whether a queued restore will write `dst_slot` on the next batch.

        The evidence that a PAGE checkpoint really is behind a boundary the
        request has claimed; see `BlockManager._state_leg_secured`.
        """
        return self.store.restore_queued_for(dst_slot)

    def begin_restore(self, h: int, dst_slot: int) -> bool:
        return self.store.begin_restore(h, dst_slot) is not None

    def take_checkpoint_ops(
        self,
    ) -> tuple[tuple[CheckpointStoreOp, ...], tuple[CheckpointRestoreOp, ...]]:
        pending, self._pending = self._pending, {}
        for seq, h in pending.values():
            # Safe to read now because `checkpoint` keeps at most one pending
            # boundary per seq: this slot holds the state as of that boundary
            # and no other. See `_supersede`.
            src_slot = seq.state_slot
            if src_slot < 0 or self.store.contains_or_pending(h):
                continue
            op = self.store.begin_store(h, src_slot)
            if op is None:
                self.checkpoints_dropped += 1
                continue
            self._store_ops.append(op)
            self.checkpoints_kept += 1
        stores, self._store_ops = self._store_ops, []
        return tuple(stores), self.store.take_restore_ops()

    def complete_previous_batch(self) -> None:
        self.store.complete_inflight()

    def has_available_units(
        self, count: int, protected_hash: int | None = None
    ) -> bool:
        return self.store.has_available_units(count, protected_hash)

    def ensure_free_units(self, count: int) -> bool:
        return self.store.ensure_free_units(count)

    def contains(self, h: int) -> bool:
        """Whether HBM holds a READY image for `h` right now.

        Distinct from `_reachable`, which also counts the CPU tier: this one
        answers "would a resume be free", and is what the `state_hbm` /
        `state_tier` split reports.
        """
        return self.store.contains(h)

    def attach_offload(self, index, *, sink: bool = True) -> None:
        """Wire the CPU tier in, after both objects exist.

        Not a constructor argument because the two are built in the wrong
        order: this class needs `checkpoint_spec`, which comes off
        `state_runtime`, while the index needs the connector config. One call
        sets both halves -- the vote (`_reachable`) and the sink (whether a
        READY checkpoint is nominated at all) -- so they cannot be turned on
        separately, which would either vote for hashes nothing stores or pin
        units nothing releases.
        """
        self.offload = index
        # `sink` is the store half specifically: a load-only role still votes
        # off `hashes` but must nominate nothing, since a nomination that is
        # handed over takes a pin nobody will release.
        self.store._offload_sink = index is not None and bool(sink)

    def take_offload_stores(
        self, max_inflight: int
    ) -> list[tuple[StateStoreOperationId, tuple[int, ...]]]:
        """`(operation, unit_ids)` to hand the tier now. See the store."""
        return self.store.take_offload_stores(max_inflight)

    def release_offload_store_source(self, op: StateStoreOperationId) -> None:
        """The gather drained; hand the units back but keep the pin. See store."""
        self.store.release_offload_store_source(op)

    def settle_offload_store(self, op: StateStoreOperationId) -> None:
        """One store reported, either way; retire the operation. See store."""
        self.store.settle_offload_store(op)

    def has_offload_pins(self) -> bool:
        """Whether any dispatched store is still pinning its units. See store."""
        return self.store.has_offload_pins()

    def reclaim_stale_offload_pins(self, timeout_s: float) -> int:
        """Release offload pins whose report never came. See the store."""
        return self.store.reclaim_stale_offload_pins(timeout_s)

    def was_reclaimed(self, op) -> bool:
        """Whether this store's source was taken back before it reported."""
        return self.store.was_reclaimed(op)

    def unindex(self, h: int) -> None:
        # `_pending` is keyed by `(seq, hash)`, so one hash can be pending for
        # several sequences at once -- two turns of a conversation reaching the
        # same boundary. All of them lose it together.
        stale = [key for key, (_, pending_h) in self._pending.items() if pending_h == h]
        for key in stale:
            del self._pending[key]
        removed = self.store.unindex(h)
        if stale or removed:
            self.checkpoints_orphaned += 1

    def clear_index(self) -> None:
        """Drop everything, for `/reset_prefix_cache`-style admin calls.

        None of the four fates moves, and that is deliberate rather than an
        oversight: each argues for a different fix — `dropped` for a bigger
        pool, `evicted` for a longer-lived one, `orphaned` for a bigger paged
        pool — and an operator emptying the cache on purpose argues for none of
        them. Charging a reset to any of them would send tuning after a number
        the operator created. The reset is visible in the drop in
        `checkpoints_kept`'s growth rate, and in the admin call itself.
        """
        self._pending.clear()
        self.store.clear()

    def checkpoint_fates(self) -> dict[str, int]:
        """What became of this class's checkpoints, plus the CPU tier's own.

        The tier's counters are folded in here rather than reported separately
        because the two only mean anything against each other: `stores_completed`
        is worth reading only as a fraction of `checkpoints_kept`, and a tier
        that stored nothing is indistinguishable from a tier that was never
        asked unless both numbers sit on one line. Folding them in makes the
        load and store legs visible again under PAGE.

        Keys are prefixed `state_offload_` because the aggregation upstream is
        by key across every state class.
        """
        fates = {
            "checkpoints_kept": self.checkpoints_kept,
            "checkpoints_dropped": self.checkpoints_dropped,
            "checkpoints_evicted": self.store.evictions,
            # Non-zero means store reports are being lost, which would
            # otherwise show only as a pool that has quietly shrunk.
            "offload_pins_reclaimed": self.store.offload_pins_reclaimed,
            "checkpoints_orphaned": self.checkpoints_orphaned,
        }
        # `getattr`, not a direct call: `attach_offload` accepts anything that
        # answers `hashes`, and the tests attach a double that does not carry
        # counters. A missing `stats` means "no numbers to fold", not an error.
        stats = getattr(self.offload, "stats", None) if self.offload else None
        if callable(stats):
            fates.update((f"state_offload_{k}", v) for k, v in stats().items())
            # A gauge, not a counter: how many nominations are waiting for a
            # slot under `OFFLOAD_MAX_PENDING_SAVES`. The queue is bounded by
            # `_offload_backlog_cap` and drained oldest-first, so a value that
            # climbs toward the cap means checkpoints are reaching READY faster
            # than the tier drains them and fresh nominations are queueing behind
            # stale ones; at the cap the oldest are dropped, counted next to it.
            fates["state_offload_store_backlog"] = len(self.store.store_backlog)
            fates["state_offload_nominations_dropped"] = (
                self.store.offload_nominations_dropped
            )
        return fates
