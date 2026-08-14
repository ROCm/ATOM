# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""State checkpoints backed by arbitrary PAGE-sized physical units.

The running request still owns one ordinary contiguous Active Slot.  Only the
immutable checkpoint image is split: its canonical byte stream is stored in an
ordered tuple of PAGE units, so allocation succeeds whenever the *total* free
unit count is large enough.  Hashing, LRU, pinning and eviction remain atomic at
the checkpoint-record level; individual fragments are never independently
visible or reclaimable.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

from atom.model_engine.block_pool import BlockPool

COPYING = "COPYING"
READY = "READY"
EVICTING = "EVICTING"


@dataclass(frozen=True)
class PagedStateCheckpointSpec:
    """Runtime geometry shared by state-checkpoint producers and consumers.

    The spec is deliberately not part of :class:`Config`: its values only
    exist after the runner has sized the physical cache pools.  The PAGE count
    is derived at each consumer so it can never drift from the two byte sizes
    while crossing a process boundary.
    """

    page_unit_bytes: int
    slot_bytes: int
    layout_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("page_unit_bytes", self.page_unit_bytes),
            ("slot_bytes", self.slot_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("paged state checkpoints need a non-empty layout id")

    @property
    def units_per_checkpoint(self) -> int:
        return (self.slot_bytes + self.page_unit_bytes - 1) // self.page_unit_bytes

    def to_wire(self) -> dict[str, int | str]:
        """Return the explicit pickle-safe payload used by runner RPC."""
        return {
            "page_unit_bytes": self.page_unit_bytes,
            "slot_bytes": self.slot_bytes,
            "layout_id": self.layout_id,
        }

    @classmethod
    def from_wire(cls, wire: object) -> PagedStateCheckpointSpec:
        """Rebuild and validate a spec received from another process."""
        if not isinstance(wire, Mapping):
            raise TypeError("paged state checkpoint spec must be a mapping")
        expected = {"page_unit_bytes", "slot_bytes", "layout_id"}
        if set(wire) != expected:
            raise ValueError(
                "invalid paged state checkpoint spec fields: "
                f"expected={sorted(expected)}, got={sorted(wire)}"
            )
        return cls(
            page_unit_bytes=wire["page_unit_bytes"],  # type: ignore[arg-type]
            slot_bytes=wire["slot_bytes"],  # type: ignore[arg-type]
            layout_id=wire["layout_id"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CheckpointStoreOp:
    """Scatter one contiguous Active Slot into ordered PAGE units."""

    checkpoint_id: int
    generation: int
    prefix_hash: int
    src_slot: int
    unit_ids: tuple[int, ...]
    total_bytes: int
    last_unit_valid_bytes: int
    layout_id: str


@dataclass(frozen=True)
class CheckpointRestoreOp:
    """Gather one ordered PAGE-unit image into an Active Slot."""

    checkpoint_id: int
    generation: int
    prefix_hash: int
    dst_slot: int
    unit_ids: tuple[int, ...]
    total_bytes: int
    last_unit_valid_bytes: int
    layout_id: str


@dataclass
class CheckpointRecord:
    checkpoint_id: int
    generation: int
    prefix_hash: int
    boundary_blocks: int
    layout_id: str
    total_bytes: int
    unit_ids: tuple[int, ...]
    last_unit_valid_bytes: int
    state: str = COPYING
    pin_count: int = 0

    @property
    def owner(self) -> tuple[str, int, int]:
        return ("state-checkpoint", self.checkpoint_id, self.generation)


class PageUnitCheckpointStore:
    """Content index and ownership table for split state images.

    A store becomes hash-visible only in :meth:`complete_inflight`, which the
    scheduler calls at the beginning of the pass after the copy op rode a
    model batch.  That is the control-plane commit point corresponding to the
    compute stream having issued the previous batch.
    """

    def __init__(
        self,
        pool: BlockPool,
        spec: PagedStateCheckpointSpec,
    ):
        self.pool = pool
        self.spec = spec
        self.unit_bytes = spec.page_unit_bytes
        self.slot_bytes = spec.slot_bytes
        self.layout_id = spec.layout_id
        self.last_unit_valid_bytes = (
            self.slot_bytes - (self.units_per_checkpoint - 1) * self.unit_bytes
        )

        self.hash_to_checkpoint: dict[int, int] = {}
        self.records: dict[int, CheckpointRecord] = {}
        self._pending_by_hash: dict[int, int] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._inflight_stores: list[int] = []
        self._inflight_restores: list[int] = []
        self._next_checkpoint_id = 0
        self._next_generation = 1
        self.evictions = 0

    @property
    def units_per_checkpoint(self) -> int:
        """Derive the complete-image allocation from this process's spec."""
        return self.spec.units_per_checkpoint

    # ------------------------------ lookup ---------------------------------
    def lookup(self, prefix_hash: int) -> int:
        checkpoint_id = self.hash_to_checkpoint.get(prefix_hash, -1)
        record = self.records.get(checkpoint_id)
        if (
            record is None
            or record.state != READY
            or record.layout_id != self.layout_id
        ):
            return -1
        return checkpoint_id

    def contains(self, prefix_hash: int) -> bool:
        return self.lookup(prefix_hash) >= 0

    def contains_or_pending(self, prefix_hash: int) -> bool:
        return self.contains(prefix_hash) or prefix_hash in self._pending_by_hash

    def record(self, checkpoint_id: int) -> CheckpointRecord:
        return self.records[checkpoint_id]

    # ---------------------------- allocation -------------------------------
    def _new_identity(self) -> tuple[int, int]:
        checkpoint_id = self._next_checkpoint_id
        generation = self._next_generation
        self._next_checkpoint_id += 1
        self._next_generation += 1
        return checkpoint_id, generation

    def has_available_units(
        self, count: int, protected_hash: int | None = None
    ) -> bool:
        """Whether PAGE allocation can obtain `count`, including atomic LRU victims."""
        if count <= self.pool.num_free:
            return True
        protected = self.lookup(protected_hash) if protected_hash is not None else -1
        reclaimable = sum(
            len(self.records[cid].unit_ids)
            for cid in self._lru
            if cid != protected
            if self.records[cid].state == READY and self.records[cid].pin_count == 0
        )
        return self.pool.num_free + reclaimable >= count

    def ensure_free_units(self, count: int) -> bool:
        """Evict whole unpinned checkpoints until `count` PAGE units are free."""
        while self.pool.num_free < count:
            victim = next(
                (
                    cid
                    for cid in self._lru
                    if self.records[cid].state == READY
                    and self.records[cid].pin_count == 0
                ),
                -1,
            )
            if victim < 0:
                return False
            self._evict(victim)
        return True

    def begin_store(
        self, prefix_hash: int, boundary_blocks: int, src_slot: int
    ) -> CheckpointStoreOp | None:
        """Reserve all fragments and create a non-visible COPYING record."""
        if self.lookup(prefix_hash) >= 0 or prefix_hash in self._pending_by_hash:
            return None
        needed = self.units_per_checkpoint
        if not self.ensure_free_units(needed):
            return None

        checkpoint_id, generation = self._new_identity()
        owner = ("state-checkpoint", checkpoint_id, generation)
        unit_ids = self.pool.reserve_units(needed, owner)
        if unit_ids is None:
            return None
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            generation=generation,
            prefix_hash=prefix_hash,
            boundary_blocks=boundary_blocks,
            layout_id=self.layout_id,
            total_bytes=self.slot_bytes,
            unit_ids=tuple(unit_ids),
            last_unit_valid_bytes=self.last_unit_valid_bytes,
        )
        self.records[checkpoint_id] = record
        self._pending_by_hash[prefix_hash] = checkpoint_id
        self._inflight_stores.append(checkpoint_id)
        return CheckpointStoreOp(
            checkpoint_id=checkpoint_id,
            generation=generation,
            prefix_hash=prefix_hash,
            src_slot=src_slot,
            unit_ids=record.unit_ids,
            total_bytes=record.total_bytes,
            last_unit_valid_bytes=record.last_unit_valid_bytes,
            layout_id=record.layout_id,
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
        self._inflight_restores.append(checkpoint_id)
        return CheckpointRestoreOp(
            checkpoint_id=checkpoint_id,
            generation=record.generation,
            prefix_hash=prefix_hash,
            dst_slot=dst_slot,
            unit_ids=record.unit_ids,
            total_bytes=record.total_bytes,
            last_unit_valid_bytes=record.last_unit_valid_bytes,
            layout_id=record.layout_id,
        )

    # ------------------------- commit / cancellation ------------------------
    def complete_inflight(self) -> None:
        """Commit stores and release restore readers from the previous batch."""
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
            # Publish only after the scatter has ridden a batch.  A canonical
            # READY record that appeared meanwhile wins; the duplicate image
            # is discarded as one atomic allocation.
            if self.lookup(record.prefix_hash) >= 0:
                self._release_record(checkpoint_id)
                continue
            record.state = READY
            self.hash_to_checkpoint[record.prefix_hash] = checkpoint_id
            self._lru[checkpoint_id] = None

        restores, self._inflight_restores = self._inflight_restores, []
        for checkpoint_id in restores:
            record = self.records.get(checkpoint_id)
            if record is None:
                continue
            if record.pin_count <= 0:
                raise AssertionError("checkpoint restore pin underflow")
            record.pin_count -= 1
            if record.state == EVICTING and record.pin_count == 0:
                self._release_record(checkpoint_id)

    def unindex(self, prefix_hash: int) -> int:
        """Make a state boundary unreachable and reclaim it when readers drain."""
        checkpoint_id = self.hash_to_checkpoint.pop(prefix_hash, -1)
        if checkpoint_id < 0:
            checkpoint_id = self._pending_by_hash.pop(prefix_hash, -1)
        if checkpoint_id < 0:
            return -1
        record = self.records.get(checkpoint_id)
        if record is None:
            return -1
        record.state = EVICTING
        self._lru.pop(checkpoint_id, None)
        # A COPYING record still has a queued GPU writer.  Its units cannot be
        # returned until that writer has been issued and ordered.
        if checkpoint_id not in self._inflight_stores and record.pin_count == 0:
            self._release_record(checkpoint_id)
        return checkpoint_id

    def clear(self) -> None:
        """Remove every hash entry and reclaim records once readers/writers drain."""
        self.hash_to_checkpoint.clear()
        self._pending_by_hash.clear()
        self._lru.clear()
        inflight_stores = set(self._inflight_stores)
        for checkpoint_id in list(self.records):
            record = self.records[checkpoint_id]
            record.state = EVICTING
            if checkpoint_id not in inflight_stores and record.pin_count == 0:
                self._release_record(checkpoint_id)

    def _evict(self, checkpoint_id: int) -> int:
        record = self.records[checkpoint_id]
        if record.state != READY or record.pin_count:
            raise AssertionError("only an unpinned READY checkpoint is evictable")
        if self.hash_to_checkpoint.get(record.prefix_hash) == checkpoint_id:
            del self.hash_to_checkpoint[record.prefix_hash]
        record.state = EVICTING
        self._lru.pop(checkpoint_id, None)
        count = len(record.unit_ids)
        self._release_record(checkpoint_id)
        self.evictions += 1
        return count

    def _release_record(self, checkpoint_id: int) -> None:
        record = self.records.pop(checkpoint_id)
        self._lru.pop(checkpoint_id, None)
        if self.hash_to_checkpoint.get(record.prefix_hash) == checkpoint_id:
            del self.hash_to_checkpoint[record.prefix_hash]
        if self._pending_by_hash.get(record.prefix_hash) == checkpoint_id:
            del self._pending_by_hash[record.prefix_hash]
        self.pool.release_units(record.unit_ids, record.owner)

    @property
    def num_ready(self) -> int:
        return len(self._lru)

    @property
    def num_units(self) -> int:
        return sum(len(record.unit_ids) for record in self.records.values())
