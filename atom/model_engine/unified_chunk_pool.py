# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified per-layer-type KV slot allocator for DeepSeek-V4 (plan §10).

Goal: within ONE layer type (all CSA layers, or all HCA layers), let SWA and that
type's compressor draw physical space from a SINGLE free-list, so space freed by
one is reusable by the other — replacing the static ``swa_pages`` split.

Model (plan §10.1):
  * Allocation unit = one **slot** = ``block_size`` (128) rows = exactly one SWA
    block. SWA needs 128 contiguous rows, so the slot is the coarsest common unit
    that avoids external fragmentation.
  * A slot is homogeneous: held either by SWA (1 SWA block) or by compress
    (``blocks_per_slot = block_size // k`` compress blocks; CSA k=32 → 4, HCA
    k=1 → 128). A whole slot returns to the shared free-list only when empty.
  * Addressing is base-0 (plan §10.2), so physical block ids are non-negative
    (flydsl-safe, no reverse / negative offset):
        SWA block      : phys = slot                         (row = phys * 128)
        compress block : phys = slot * blocks_per_slot + j   (row = phys * k)
    where ``j`` in ``[0, blocks_per_slot)`` is the sub-index within the slot.

Per layer type there is one instance (CSA pool, HCA pool). All layers of that
type share it: identical k and structure → identical slot decisions apply to
every layer's own tensor. CSA↔HCA cannot share (different tensors / different k)
— that is physical, not a forced split.

Self-guarding: ``num_slots == 0`` → disabled (non-V4 / no compress). SWA-only
(dense) pools set ``k == 0`` (no compress side).
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional


def unified_slot_counts(
    num_swa_blocks: int, num_compress_blocks: int, block_size: int = 128
) -> tuple[int, int]:
    """Per-type 128-row slot counts for the CSA and HCA unified tensors (plan §11).

    Correctness-first sizing: each type reserves the SWA region [0, num_swa_blocks)
    plus an own-compress region large enough to hold every compress block WITHOUT
    borrowing, so flag-on physical capacity >= flag-off (enabling the flag never
    OOMs on its own; borrowing SWA-freed slots is pure upside). This equals the
    flag-off tensor size to within one slot: n_csa*128 == swa_pages + ceil rounding
    of num_compress_blocks*32. The SINGLE source of truth shared by BlockManager
    (coordinator sizing) and the attention builder (unified_kv tensor sizing).
    """
    csa_bps = block_size // 32  # CSA k=32 -> 4 compress blocks per 128-row slot
    hca_bps = block_size // 1   # HCA k=1  -> 128 per slot
    csa_own = (num_compress_blocks + csa_bps - 1) // csa_bps
    hca_own = (num_compress_blocks + hca_bps - 1) // hca_bps
    return num_swa_blocks + csa_own, num_swa_blocks + hca_own


class UnifiedTypePool:
    def __init__(self, num_slots: int, k: int, block_size: int = 128):
        self.enabled: bool = num_slots > 0
        self.num_slots: int = int(num_slots)
        self.block_size: int = int(block_size)
        self.k: int = int(k)  # compress rows per block (0 = SWA-only / dense)
        # compress blocks packed into one slot (0 when k==0)
        self.blocks_per_slot: int = (block_size // k) if k > 0 else 0

        # Shared slot free-list (both SWA and compress draw from it).
        self._free_slots: deque[int] = deque(range(num_slots))
        self._free_slots_set: set[int] = set(range(num_slots))

        # Slots currently held by SWA (each = 1 SWA block).
        self._swa_slots: set[int] = set()

        # Compress bookkeeping: a slot handed to compress is sub-divided into
        # blocks_per_slot sub-indices. Track, per open compress slot, the set of
        # FREE sub-indices; a slot with all sub-indices free returns to the pool.
        # `_cmp_open` holds slots with >=1 free sub-index (alloc source).
        self._cmp_free_sub: dict[int, set[int]] = {}  # slot -> {free sub-idx}
        self._cmp_open: deque[int] = deque()

    # ------------------------------- slots -------------------------------- #
    def _pop_free_slot(self) -> int:
        while self._free_slots:
            s = self._free_slots.popleft()
            if s in self._free_slots_set:
                self._free_slots_set.discard(s)
                return s
        raise AssertionError("No free KV slots available")

    def _return_slot(self, s: int) -> None:
        self._free_slots.append(s)
        self._free_slots_set.add(s)

    # ---- external reservation (option-2: SWA holds the SAME slot across the
    # CSA and HCA pools; the coordinator reserves it in both so neither pool's
    # compress side hands it out while SWA holds it). ---------------------- #
    def reserve_slot(self, s: int) -> bool:
        """Mark slot `s` as taken by an external owner (SWA). Returns True if it
        was free and is now reserved; False if it wasn't free (can't reserve)."""
        if s not in self._free_slots_set:
            return False
        self._free_slots_set.discard(s)
        return True

    def release_slot(self, s: int) -> None:
        """Release an externally-reserved slot back to this pool's free-list."""
        if s not in self._free_slots_set:
            self._return_slot(s)

    def slot_is_free(self, s: int) -> bool:
        return s in self._free_slots_set

    # --------------------------- capacity checks -------------------------- #
    def has_free_swa(self, n_blocks: int) -> bool:
        """Whether `n_blocks` SWA blocks can be admitted. Disabled → True."""
        if not self.enabled:
            return True
        return len(self._free_slots_set) >= n_blocks  # 1 slot per SWA block

    def has_free_compress(self, n_blocks: int) -> bool:
        """Whether `n_blocks` compress blocks fit in free sub-slots + free slots.
        Disabled / SWA-only → True."""
        if not self.enabled or self.blocks_per_slot == 0:
            return True
        open_capacity = sum(len(v) for v in self._cmp_free_sub.values())
        slot_capacity = len(self._free_slots_set) * self.blocks_per_slot
        return open_capacity + slot_capacity >= n_blocks

    def num_free_slots(self) -> int:
        return len(self._free_slots_set)

    # ------------------------------- SWA ---------------------------------- #
    def alloc_swa(self) -> int:
        """Take a slot for one SWA block; return its physical block id (= slot)."""
        s = self._pop_free_slot()
        self._swa_slots.add(s)
        return s

    def free_swa(self, phys: int) -> None:
        """Return an SWA block's slot to the shared free-list."""
        self._swa_slots.discard(phys)
        self._return_slot(phys)

    def is_free_swa(self, phys: int) -> bool:
        """True if slot `phys` is currently on the free-list (for cached-hit claim)."""
        return phys in self._free_slots_set

    def claim_swa(self, phys: int) -> None:
        """Claim a specific free slot for an SWA cached-hit reuse."""
        self._free_slots_set.discard(phys)
        self._swa_slots.add(phys)

    # ----------------------------- compress ------------------------------- #
    def alloc_compress(self) -> int:
        """Allocate one compress block; return its physical block id
        (= slot * blocks_per_slot + sub). Opens a new slot when needed."""
        assert self.blocks_per_slot > 0, "compress alloc on an SWA-only pool"
        # Reuse an open compress slot with a free sub-index.
        while self._cmp_open:
            s = self._cmp_open[0]
            free_sub = self._cmp_free_sub.get(s)
            if free_sub:
                j = free_sub.pop()
                if not free_sub:  # slot now full → drop from open list
                    self._cmp_open.popleft()
                    del self._cmp_free_sub[s]
                return s * self.blocks_per_slot + j
            self._cmp_open.popleft()  # stale
        # No open slot: take a fresh one, use sub 0, keep the rest open.
        s = self._pop_free_slot()
        rest = set(range(1, self.blocks_per_slot))
        if rest:
            self._cmp_free_sub[s] = rest
            self._cmp_open.append(s)
        return s * self.blocks_per_slot  # sub 0

    def free_compress(self, phys: int) -> None:
        """Free one compress block; return its slot to the pool when fully empty."""
        s, j = divmod(phys, self.blocks_per_slot)
        free_sub = self._cmp_free_sub.get(s)
        if free_sub is None:
            # Slot was full (not in open list): reopen it with this sub free.
            free_sub = set()
            self._cmp_free_sub[s] = free_sub
            self._cmp_open.append(s)
        free_sub.add(j)
        if len(free_sub) == self.blocks_per_slot:
            # All sub-blocks free → return the whole slot to the shared pool.
            del self._cmp_free_sub[s]
            # (leave the stale entry in _cmp_open; _pop/_alloc skip via set checks)
            self._return_slot(s)


class UnifiedKvCoordinator:
    """Option-2 (plan §10, SWA-single-table variant): SWA keeps ONE slot space
    shared across layer types; compress is per-type (CSA / HCA). Compress freely
    reuses slots SWA has freed (the main requirement: "compress reuses SWA's freed
    space"); SWA does not grow past its base region (bounded by the dense/HCA
    tensors), so SWA-side logic and tables stay unchanged.

    Slot = 128 rows (= one SWA block). SWA slot `s` maps to row `s*128` in EVERY
    layer's tensor. Compress block phys (base-0): `phys = slot * blocks_per_slot
    + sub`, row `phys * k` — non-negative (flydsl-safe).

    Shared region [0, num_swa_base):
      * A slot is SWA-held, or compress-held (independently per type in that
        type's tensor), or free.
      * SWA takes a slot only if free in BOTH compress pools (so it's unused in
        every tensor). Compress takes a slot if not SWA-held and free in its own
        pool. Once compress grabs an SWA-freed slot, SWA cannot reclaim it until
        compress frees it (shared budget → admission handles pressure).
    Compress-only region [num_swa_base, n_*_total): each type's own compress space.
    """

    def __init__(
        self,
        num_swa_base: int,   # SWA slot count (= num_swa_blocks); SWA region size
        n_csa_slots: int,    # total 128-row slots in a CSA-layer tensor
        n_hca_slots: int,    # total 128-row slots in an HCA-layer tensor
        block_size: int = 128,
    ):
        self.enabled = num_swa_base > 0
        self.num_swa_base = int(num_swa_base)
        self.block_size = int(block_size)
        self.csa = UnifiedTypePool(n_csa_slots, k=32, block_size=block_size)
        self.hca = UnifiedTypePool(n_hca_slots, k=1, block_size=block_size)
        # Reserve the SWA region [0, num_swa_base) in both compress pools so
        # compress starts only in its own region; SWA-freed slots are released in.
        for s in range(min(self.num_swa_base, n_csa_slots)):
            self.csa.reserve_slot(s)
        for s in range(min(self.num_swa_base, n_hca_slots)):
            self.hca.reserve_slot(s)
        from collections import deque as _dq
        self._swa_free = _dq(range(self.num_swa_base))
        self._swa_free_set = set(range(self.num_swa_base))
        self._swa_held: set[int] = set()
        # SWA slots currently RELEASED into the compress pools (SWA freed them,
        # compress may have taken them). A slot NOT in `_released` is still
        # reserved for SWA in both pools (init state) → SWA takes it directly.
        self._released: set[int] = set()
        # SWA slots that were freed while still holding *lazily-cached* SWA content
        # (hash live in the SWA pool, ref 0). If compress borrows such a slot, that
        # SWA content is overwritten → the SWA prefix-cache entry MUST be dropped
        # first, or a later SWA hash-hit would double-claim a compress-held slot
        # and corrupt KV (plan §11.1). `_swa_evict_cb(slot)` performs that drop.
        # Cleared when SWA reclaims the slot (its own reuse) or compress evicts it.
        self._swa_cached: set[int] = set()
        self._swa_evict_cb: Optional[Callable[[int], None]] = None

    def set_swa_evict_cb(self, cb: Callable[[int], None]) -> None:
        """Register the SWA pool's hook to drop the cache entry for an SWA slot
        that compress is about to overwrite (plan §11.1). `cb(slot)` must be a
        no-op if the slot holds no live SWA cache entry."""
        self._swa_evict_cb = cb

    def _evict_swa_if_cached(self, slot: int) -> None:
        if slot in self._swa_cached:
            self._swa_cached.discard(slot)
            if self._swa_evict_cb is not None:
                self._swa_evict_cb(slot)

    # ------------------------------- SWA ---------------------------------- #
    def alloc_swa(self) -> int:
        """Take an SWA slot free in every tensor. Skips slots compress grabbed."""
        while self._swa_free:
            s = self._swa_free.popleft()
            if s not in self._swa_free_set:
                continue
            self._swa_free_set.discard(s)
            if s not in self._released:
                # Never released → still reserved for SWA in both pools.
                self._swa_held.add(s)
                self._swa_cached.discard(s)
                return s
            # Released earlier → must re-reserve; skip if compress grabbed it.
            csa_ok = s >= self.csa.num_slots or self.csa.reserve_slot(s)
            if not csa_ok:
                continue
            hca_ok = s >= self.hca.num_slots or self.hca.reserve_slot(s)
            if not hca_ok:
                if s < self.csa.num_slots:
                    self.csa.release_slot(s)  # roll back
                continue
            self._released.discard(s)
            self._swa_cached.discard(s)
            self._swa_held.add(s)
            return s
        raise AssertionError("No free SWA slots available")

    def free_swa(self, s: int) -> None:
        """Free an SWA slot; release it to both compress pools for reuse. The slot
        may still hold lazily-cached SWA content (ref 0, hash live) → tracked in
        `_swa_cached` so a borrowing compress alloc evicts that entry first."""
        self._swa_held.discard(s)
        self._swa_free.append(s)
        self._swa_free_set.add(s)
        self._released.add(s)
        self._swa_cached.add(s)
        if s < self.csa.num_slots:
            self.csa.release_slot(s)
        if s < self.hca.num_slots:
            self.hca.release_slot(s)

    def _is_swa_reclaimable(self, s: int) -> bool:
        """A slot is SWA-reclaimable iff it's in the SWA region, not SWA-held, and
        free in BOTH compress pools (SWA needs the whole 128-row slot free in every
        layer tensor). Computed from the compress pools = always accurate."""
        if s >= self.num_swa_base or s in self._swa_held:
            return False
        csa_free = s >= self.csa.num_slots or self.csa.slot_is_free(s)
        hca_free = s >= self.hca.num_slots or self.hca.slot_is_free(s)
        return csa_free and hca_free

    def is_free_swa(self, s: int) -> bool:
        """True if slot `s` is currently reclaimable by SWA (on the SWA free-list
        AND not grabbed by compress). `_swa_free_set` is kept accurate as compress
        borrows/returns slots, so membership is the source of truth. Mirrors
        UnifiedTypePool.is_free_swa so the coordinator can back the SWA pool's
        chunk_pool in flag-on mode."""
        return s in self._swa_free_set

    def claim_swa(self, s: int) -> None:
        """SWA cache-hit reclaim of a specific free slot (caller checked
        is_free_swa). Re-reserves it in the compress pools if it had been
        released, and clears its released/cached tags."""
        self._swa_free_set.discard(s)
        if s in self._released:
            if s < self.csa.num_slots:
                self.csa.reserve_slot(s)
            if s < self.hca.num_slots:
                self.hca.reserve_slot(s)
            self._released.discard(s)
        self._swa_cached.discard(s)
        self._swa_held.add(s)

    def has_free_swa(self, n: int) -> bool:
        if not self.enabled:
            return True
        return len(self._swa_free_set) >= n

    # ----------------------------- compress ------------------------------- #
    def _on_compress_alloc(self, slot: int) -> None:
        """A compress pool just grabbed `slot`. If it's a released SWA slot, it is
        no longer SWA-reclaimable (that pool's 128 rows are now compress) — drop it
        from the SWA free-set and evict any lazily-cached SWA content there."""
        if slot < self.num_swa_base and slot in self._released:
            self._swa_free_set.discard(slot)
            self._evict_swa_if_cached(slot)

    def _on_compress_free(self, slot: int) -> None:
        """A compress pool returned space in `slot`. If it's a released SWA slot
        that is now fully free in both compress pools again, it becomes
        SWA-reclaimable — re-add it to the SWA free-set (empty; its SWA cache was
        evicted when compress borrowed it)."""
        if (
            slot < self.num_swa_base
            and slot in self._released
            and slot not in self._swa_free_set
            and self._is_swa_reclaimable(slot)
        ):
            self._swa_free_set.add(slot)
            self._swa_free.append(slot)

    def alloc_csa(self) -> int:
        phys = self.csa.alloc_compress()
        self._on_compress_alloc(phys // self.csa.blocks_per_slot)
        return phys

    def free_csa(self, phys: int) -> None:
        slot = phys // self.csa.blocks_per_slot
        self.csa.free_compress(phys)
        self._on_compress_free(slot)

    def alloc_hca(self) -> int:
        phys = self.hca.alloc_compress()
        self._on_compress_alloc(phys // self.hca.blocks_per_slot)
        return phys

    def free_hca(self, phys: int) -> None:
        slot = phys // self.hca.blocks_per_slot
        self.hca.free_compress(phys)
        self._on_compress_free(slot)

    def has_free_csa(self, n: int) -> bool:
        return self.csa.has_free_compress(n)

    def has_free_hca(self, n: int) -> bool:
        return self.hca.has_free_compress(n)
