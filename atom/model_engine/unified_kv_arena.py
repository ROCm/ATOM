# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified-KV control plane: elastic SWA / compressed KV split (byte arena).

DeepSeek-V4 shipped the sliding-window (SWA) KV and the compressed (CSA/HCA) KV
as separate fixed-size pools. An under-used pool wastes its reserved capacity.
This module removes the fixed split: every pool ("owner") borrows equal-size
physical **byte** chunks from one arena per *physical group* and hands them back
on demand.

Owners per group
----------------
* ``swa``      — full-resolution window KV, present in EVERY group (SWA is global
                 across all layers). In the c4 group a SWA chunk also carries the
                 fused CSA boundary-state snapshot in its tail byte segment
                 (feat/csa-swa-fusion) — that snapshot is NOT a separate owner.
* ``compress`` — CSA/HCA compressed KV, present in the c4 / c128 groups.

Byte chunks (B2): pages are sized in BYTES so each owner can sub-divide a chunk
at its own ``page_bytes`` (SWA/compress pages are ``head_dim``-wide fp8/bf16). A
chunk is single-owner at a time, so heterogeneous owners never alias. Consumers
reinterpret a chunk's bytes as their own (dtype, width) view (see
``deepseek_v4_attn`` per-layer uint8 buffer); the fused CSA state is an fp32 view
over the c4 SWA chunk's tail bytes.

Why per-group, not one global arena
-----------------------------------
A logical block id is GLOBAL across layers, but compressed rows/block differ by
ratio (c4 vs c128), so one global chunk→block assignment can't be consistent.
Fix: per-group arenas; a logical id resolves to a per-group physical page; every
layer in a group is driven by the SAME global allocations, staying in lock-step.

Lending model (v1-simple, single source of truth): the arena tracks only USED vs
TRULY-FREE pages. ``free_*`` is called ONLY at true eviction; ref-0-cached blocks
keep their pages in the pool's own LRU. Cross-pool lending is POOL-DRIVEN.

Self-guarding: ``enabled`` is False when disabled (feature off / non-V4).
"""

from __future__ import annotations

from atom.model_engine.chunk_arena import ArenaEmpty, ChunkArena, ChunkBackedFreeList

# Owner names. SWA is in every group; compress in c4/c128. The DSV4 CSA boundary
# snapshot is NOT an owner — it is fused into the SWA chunk's tail byte segment
# (feat/csa-swa-fusion) and rides the SWA block's alloc/free/pin.
OWNER_SWA = "swa"
OWNER_COMPRESS = "compress"


class ArenaGroup:
    """One physical group: identically-laid-out layers sharing a byte arena.

    ``free[owner]`` is that owner's page allocator over the shared chunk arena;
    ``phys[owner]`` maps a logical id → its physical page index within THIS group.
    An owner absent from ``free`` does not exist in this group (e.g. compress in a
    dense group).
    """

    def __init__(self, name: str, arena: ChunkArena, owner_page_bytes: dict[str, int]):
        self.name = name
        self.arena = arena
        self.free: dict[str, ChunkBackedFreeList] = {
            o: ChunkBackedFreeList(arena, page_bytes=pb)
            for o, pb in owner_page_bytes.items()
            if pb > 0
        }
        self.phys: dict[str, dict[int, int]] = {o: {} for o in self.free}


class UnifiedKvArena:
    """Elastic SWA / compressed KV backing shared across per-ratio groups.

    The block pools keep their own logical id spaces and lifecycle; this class
    only decides which physical page backs a logical id in each group. Cross-pool
    lending is pool-driven, so the arena here is a pure per-group page allocator.
    """

    def __init__(
        self,
        *,
        block_size: int,
        group_specs: list[dict],
    ):
        """``group_specs``: one dict per group::

            {"name": str, "num_chunks": int, "bytes_per_chunk": int,
             "owners": {owner_name: page_bytes, ...}}

        ``bytes_per_chunk`` must be a common multiple of every owner's
        ``page_bytes`` in the group (ChunkBackedFreeList asserts divisibility).
        ``block_size`` is retained for the row-stride resolution helpers.
        """
        self.block_size = int(block_size)
        self.enabled = bool(group_specs) and self.block_size > 0
        self.groups: dict[str, ArenaGroup] = {}
        for spec in group_specs:
            name = spec["name"]
            num_chunks = int(spec["num_chunks"])
            bytes_per_chunk = int(spec["bytes_per_chunk"])
            owners = {k: int(v) for k, v in spec["owners"].items()}
            arena = ChunkArena(num_chunks=num_chunks, bytes_per_chunk=bytes_per_chunk)
            self.groups[name] = ArenaGroup(name, arena, owners)

    # --------------------------- generic owner ops ------------------------- #
    def _alloc_owners(self, owners: list[str], lid: int) -> None:
        """Back ``lid`` with a page in every group that HAS each owner in
        ``owners``. All-or-nothing: on ``ArenaEmpty`` roll back the partial grant
        and re-raise (caller evicts a cold sibling block, then retries)."""
        granted: list[tuple[ArenaGroup, str]] = []
        try:
            for g in self.groups.values():
                for o in owners:
                    fl = g.free.get(o)
                    if fl is None or lid in g.phys[o]:
                        continue
                    g.phys[o][lid] = fl.pop()
                    granted.append((g, o))
        except ArenaEmpty:
            for g, o in granted:
                g.free[o].free(g.phys[o].pop(lid))
            raise

    def _free_owners(self, owners: list[str], lid: int) -> None:
        for g in self.groups.values():
            for o in owners:
                fl = g.free.get(o)
                if fl is None:
                    continue
                pid = g.phys[o].pop(lid, None)
                if pid is not None:
                    fl.free(pid)

    def _is_backed(self, owner: str, lid: int) -> bool:
        for g in self.groups.values():
            if owner in g.free:
                return lid in g.phys[owner]
        return True

    def _max_for(self, owner: str) -> int:
        """Upper bound on logical ids of ``owner`` (if it owned every chunk in
        every group that has it), bound by the tightest such group."""
        caps = [
            g.arena.num_chunks * g.free[owner].pages_per_chunk
            for g in self.groups.values()
            if owner in g.free
        ]
        return min(caps) if caps else 0

    def _available_for(self, owner: str) -> int:
        avail = [
            g.free[owner].available() for g in self.groups.values() if owner in g.free
        ]
        return min(avail) if avail else (1 << 30)

    # ------------------------------ sizing -------------------------------- #
    def max_compressed_blocks(self) -> int:
        return self._max_for(OWNER_COMPRESS) if self.enabled else 0

    def max_swa_blocks(self) -> int:
        return self._max_for(OWNER_SWA) if self.enabled else 0

    def compress_pages_per_chunk(self) -> int:
        """New compressed blocks a single evicted SWA block enables. One SWA evict
        frees one chunk per group, but a compressed block needs a page in every
        compress group, so the yield is bounded by the group with the FEWEST
        compress pages/chunk (e.g. c4 vs c128). MIN keeps admission sound."""
        if not self.enabled:
            return 0
        ppcs = [
            g.free[OWNER_COMPRESS].pages_per_chunk
            for g in self.groups.values()
            if OWNER_COMPRESS in g.free
        ]
        return min(ppcs) if ppcs else 0

    # ---------------------------- capacity gates --------------------------- #
    def compressed_available(self) -> int:
        return self._available_for(OWNER_COMPRESS) if self.enabled else (1 << 30)

    def swa_available(self) -> int:
        return self._available_for(OWNER_SWA) if self.enabled else (1 << 30)

    def can_alloc_compressed(self, n: int) -> bool:
        return not self.enabled or self.compressed_available() >= n

    def can_alloc_swa(self, n: int) -> bool:
        return not self.enabled or self.swa_available() >= n

    # ------------------------------ compressed ----------------------------- #
    def alloc_compressed(self, block_id: int) -> None:
        if self.enabled:
            self._alloc_owners([OWNER_COMPRESS], block_id)

    def free_compressed(self, block_id: int) -> None:
        if self.enabled:
            self._free_owners([OWNER_COMPRESS], block_id)

    def is_compressed_backed(self, block_id: int) -> bool:
        return self._is_backed(OWNER_COMPRESS, block_id) if self.enabled else True

    # --------------------------------- SWA --------------------------------- #
    def alloc_swa(self, swa_id: int) -> None:
        if self.enabled:
            self._alloc_owners([OWNER_SWA], swa_id)

    def free_swa(self, swa_id: int) -> None:
        if self.enabled:
            self._free_owners([OWNER_SWA], swa_id)

    def is_swa_backed(self, swa_id: int) -> bool:
        return self._is_backed(OWNER_SWA, swa_id) if self.enabled else True

    # -------------------- batch-shipping translation ----------------------- #
    def group_names(self) -> list[str]:
        return list(self.groups.keys())

    def compress_group_of_ratio(self, ratio: int) -> str:
        """Map a layer's compress ratio to its group name (4->c4, 128->c128,
        else dense)."""
        if ratio == 4:
            return "c4"
        if ratio == 128:
            return "c128"
        return "dense"

    def _physical_table(
        self, group: str, owner: str, logical_table: list[int]
    ) -> list[int]:
        g = self.groups.get(group)
        if g is None or owner not in g.free:
            return [max(0, b) for b in logical_table]
        phys = g.phys[owner]
        return [phys.get(b, 0) if b >= 0 else 0 for b in logical_table]

    def physical_compress_table(
        self, group: str, logical_table: list[int]
    ) -> list[int]:
        return self._physical_table(group, OWNER_COMPRESS, logical_table)

    def physical_swa_table(self, group: str, logical_swa_table: list[int]) -> list[int]:
        return self._physical_table(group, OWNER_SWA, logical_swa_table)

    # ---------------------------- resolution ------------------------------- #
    def compress_page(self, group: str, block_id: int) -> int:
        return self.groups[group].phys[OWNER_COMPRESS][block_id]

    def swa_page(self, group: str, swa_id: int) -> int:
        return self.groups[group].phys[OWNER_SWA][swa_id]
