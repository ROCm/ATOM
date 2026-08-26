# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""State checkpoints backed by arbitrary PAGE-sized physical units."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.sequence import Sequence

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


class PageUnitCheckpointStore:
    """Content index and ownership table for split state images."""

    def __init__(
        self,
        pool: BlockPool,
        spec: PagedStateCheckpointSpec,
    ):
        self.pool = pool
        self.spec = spec
        self.hash_to_checkpoint: dict[int, int] = {}
        self.records: dict[int, CheckpointRecord] = {}
        self._pending_by_hash: dict[int, int] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._inflight_stores: list[int] = []
        self._queued_restores: list[tuple[int, CheckpointRestoreOp]] = []
        self._inflight_restores: list[int] = []
        self._next_checkpoint_id = 0
        self.evictions = 0

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

        restores, self._inflight_restores = self._inflight_restores, []
        for checkpoint_id in restores:
            self._release_restore_pin(checkpoint_id)

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


class MidstepSlotAllocator(Protocol):
    """The slice of the state pool a midstep PAGE backend needs.

    A midstep boundary does not exist in any slot until the kernel writes it, so
    the coordinator has to hand the runner somewhere to put it. That staging
    slot comes from the ordinary state pool — the same free list live requests
    draw on — and goes back once the scatter that reads it has ridden a batch.

    Named as a capability rather than typed to `StateSlotPool` so the pool stays
    a plain allocator here: this protocol is deliberately the three methods the
    coordinator calls and nothing else, so nobody can reach the pool's index
    through it and end up with two owners of the checkpoint index.
    """

    def has_free(self, count: int = 1) -> bool: ...

    def pop(self) -> int: ...

    def release(self, slot: int) -> None: ...


class PagedStateCheckpointCoordinator:
    """Schedules PAGE-backed checkpoints for per-request state."""

    # A copy binds no successor: the owner keeps writing its own slot, so the
    # forward after a checkpoint owes it nothing. Constant across backends,
    # unlike the two capabilities below.
    successor_room = 0.0

    def __init__(
        self,
        pool: BlockPool,
        spec: PagedStateCheckpointSpec,
        enabled: bool,
        readable_midstep: bool = False,
        slot_allocator: MidstepSlotAllocator | None = None,
    ) -> None:
        self.enabled = enabled
        self.store = PageUnitCheckpointStore(pool, spec)
        # `(src_slot, prefix_hash, staged)` per boundary, many per seq. A list
        # rather than one entry keyed by `id(seq)`: a prompt can reach a grid
        # rung, a demand and the prompt-end anchor in a single forward, and
        # keying by sequence kept only whichever arrived last.
        self._pending: list[tuple[int, int, bool]] = []
        self._store_ops: list[CheckpointStoreOp] = []
        # Staging slots handed out by `reserve_midstep`, indexed by the slot the
        # scatter reads. Released once that scatter has ridden a batch, not when
        # the op is queued — see `complete_previous_batch`.
        self._staged: dict[int, int] = {}
        self.checkpoints_kept = 0
        self.checkpoints_dropped = 0
        self.checkpoints_orphaned = 0
        # Whether this backend's kernel can hand us a state from *inside* a
        # forward rather than only from the slot that forward ended on.
        #
        # An instance answer, not a class one, because the same coordinator now
        # serves both kinds. DeepSeek-V4's compressor ring is not materialized
        # at interior boundaries, so it answers False and the engine keeps
        # cutting prefill chunks onto rungs for it; a chunk-kernel backend
        # materializes every boundary in `h` and answers True.
        self.readable_midstep = readable_midstep and slot_allocator is not None
        self._slots = slot_allocator
        # Whether more than the last boundary a seq reached survives.
        #
        # Follows `readable_midstep` because the two have the same cause here:
        # only a midstep backend reserves a distinct staging slot per position,
        # and only then can two boundaries of one prompt coexist as two records.
        # Without it `checkpoint` sees one boundary per seq per forward, and the
        # prompt-end anchor is not worth the prefill chunk it costs — see
        # `BlockManager._record_checkpoint_end`, which reads this.
        self.keeps_interior_boundaries = self.readable_midstep

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
            if assume_checkpointed or self.store.contains(block_hashes[i]):
                return i + 1
        return 0

    def checkpoint(self, seq: Sequence, boundary_blocks: int, h: int) -> None:
        """Keep the state this forward *ended* on, filed under `h`.

        The non-midstep half of the protocol: the source is the seq's own live
        slot, because that is where a forward that stopped here left its state.
        A midstep backend reaches `_pending` through `publish_midstep` instead,
        naming a staging slot; both land in the same queue and are drained
        together, which is why the entry carries the slot rather than the seq.

        **The latest boundary replaces any earlier one on the same live slot.**
        Two calls before a drain name two positions but only one set of bytes:
        the slot holds the state the forward actually ended on, which is the
        later boundary. Keeping both would file *that* state under the earlier
        boundary's hash too — a checkpoint a resumer can find and which returns
        the wrong prefix's state, with nothing downstream able to tell.

        Staged entries are exempt because they are the case this does not
        describe: each owns a distinct staging slot holding its own position's
        bytes, so several can coexist and the anchor and a rung both survive.
        """
        del boundary_blocks
        if self.applies(seq) and seq.state_slot >= 0:
            live = seq.state_slot
            self._pending = [
                entry
                for entry in self._pending
                if entry[2] or entry[0] != live  # keep staged, drop stale live
            ]
            self._pending.append((live, h, False))

    def reserve_midstep(self, seq, positions: list[tuple[int, int]]) -> list[tuple]:
        """Take a staging slot for each `(position, hash)` this step covers.

        Returns `(slot, position, hash)` per reservation actually made — the
        runner needs `slot` and `position`, `publish_midstep` needs the hash.
        Best-effort and order-preserving, like the fork pool's: reservations
        stop at the first one the free list cannot fill, so the earliest
        position — the one an earlier forward reaches first — survives a
        shortage.

        Nothing is indexed here, and no PAGE units are reserved yet. The bytes
        do not exist until the forward runs; `publish_midstep` is what queues
        the scatter that gives them a home.
        """
        if not self.applies(seq) or not self.readable_midstep:
            return []
        assert self._slots is not None, "readable_midstep without a slot allocator"
        out = []
        for pos, h in positions:
            if not self._slots.has_free():
                self.checkpoints_dropped += 1
                break
            out.append((self._slots.pop(), pos, h))
        return out

    def publish_midstep(self, reservations: list[tuple], seq=None) -> None:
        """Queue each staged boundary for scatter, now that its bytes exist.

        Not indexed here either: unlike the fork pool, where publishing *is*
        filing the slot under its hash, a PAGE checkpoint is only findable once
        the scatter has ridden a batch (`PageUnitCheckpointStore` moves the
        record COPYING -> READY in `complete_inflight`). So this hands the
        boundary to the same pending queue `checkpoint` uses and lets the
        ordinary drain do the rest.

        `seq` is accepted and unused: the fork pool needs it to rank anchors
        above guesses in its LRU, and here the store owns eviction order.
        """
        del seq
        for slot, _pos, h in reservations:
            self._pending.append((slot, h, True))

    def cancel_midstep(self, reservations: list[tuple]) -> None:
        """Hand back staging slots whose forward never ran (preempt, abort).

        Released straight away rather than through `_staged`: nothing was
        queued, so no scatter will read these.
        """
        if self._slots is None:
            return
        for slot, _pos, _h in reservations:
            self._slots.release(slot)

    def forget_pending(self, seq: Sequence) -> None:
        live = seq.state_slot
        self._pending = [
            entry for entry in self._pending if entry[0] != live or entry[2]
        ]
        self.store.cancel_queued_restore(live)

    def begin_restore(self, h: int, dst_slot: int) -> bool:
        return self.store.begin_restore(h, dst_slot) is not None

    def take_checkpoint_ops(
        self,
    ) -> tuple[tuple[CheckpointStoreOp, ...], tuple[CheckpointRestoreOp, ...]]:
        pending, self._pending = self._pending, []
        for src_slot, h, staged in pending:
            if src_slot < 0 or self.store.contains_or_pending(h):
                if staged:
                    self._release_staged(src_slot)
                continue
            op = self.store.begin_store(h, src_slot)
            if op is None:
                self.checkpoints_dropped += 1
                if staged:
                    self._release_staged(src_slot)
                continue
            self._store_ops.append(op)
            self.checkpoints_kept += 1
            if staged:
                # Held until the scatter reading it has ridden a batch. Handing
                # it back now would put it on the free list during the pass that
                # admits the requests which could pop it, and one kernel would
                # then read and write it at once.
                self._staged[src_slot] = self._staged.get(src_slot, 0) + 1
        stores, self._store_ops = self._store_ops, []
        return tuple(stores), self.store.take_restore_ops()

    def _release_staged(self, slot: int) -> None:
        if self._slots is not None:
            self._slots.release(slot)

    def complete_previous_batch(self) -> None:
        self.store.complete_inflight()
        # Every scatter queued by the previous drain has now ridden a batch, so
        # the slots those scatters read are free again.
        staged, self._staged = self._staged, {}
        for slot, count in staged.items():
            for _ in range(count):
                self._release_staged(slot)

    def has_available_units(
        self, count: int, protected_hash: int | None = None
    ) -> bool:
        return self.store.has_available_units(count, protected_hash)

    def ensure_free_units(self, count: int) -> bool:
        return self.store.ensure_free_units(count)

    def unindex(self, h: int) -> None:
        kept = []
        dropped = False
        for entry in self._pending:
            if entry[1] != h:
                kept.append(entry)
                continue
            dropped = True
            # The KV block this was filed under is gone, so the scatter would
            # write an image nothing can look up. The staging slot goes back.
            if entry[2]:
                self._release_staged(entry[0])
        self._pending = kept
        removed = self.store.unindex(h)
        if dropped or removed:
            self.checkpoints_orphaned += 1

    def clear_index(self) -> None:
        for entry in self._pending:
            if entry[2]:
                self._release_staged(entry[0])
        self._pending.clear()
        self.store.clear()

    def checkpoint_fates(self) -> dict[str, int]:
        return {
            "checkpoints_kept": self.checkpoints_kept,
            "checkpoints_dropped": self.checkpoints_dropped,
            "checkpoints_evicted": self.store.evictions,
            "checkpoints_orphaned": self.checkpoints_orphaned,
        }
