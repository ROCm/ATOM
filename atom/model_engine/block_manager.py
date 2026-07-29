# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Iterable

import numpy as np
import xxhash

from atom.config import Config
from atom.distributed.kv_events import (
    MEDIUM_GPU,
    MEDIUM_REMOTE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from atom.kv_cache.batch import KvBatchTables
from atom.kv_cache.dsv4.arena import Dsv4UnifiedArena
from atom.kv_cache.dsv4.batch_tables import build_dsv4_batch_tables
from atom.kv_cache.dsv4.swa_pool import Dsv4SwaPool
from atom.kv_cache.pools.chunk_arena import ArenaEmpty
from atom.kv_cache.pools.pooled_free_list import PooledFreeList
from atom.model_engine.kv_block import Block
from atom.model_engine.sequence import Sequence
from atom.utils import envs


def _make_block_stored(
    hashes: list[int],
    tokens: list[int],
    parent: int | None,
    block_size: int,
    medium: str = MEDIUM_GPU,
) -> BlockStored:
    """Construct a BlockStored event from a coalesced run of new blocks."""
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=tokens,
        block_size=block_size,
        medium=medium,
    )


def _make_block_removed(hashes: list[int]) -> BlockRemoved:
    return BlockRemoved(block_hashes=hashes, medium=MEDIUM_GPU)


def _make_all_cleared() -> AllBlocksCleared:
    return AllBlocksCleared()


class BlockManager:
    @staticmethod
    def _make_primary_free_list(
        capacity: int, *, initially_backed: bool
    ) -> PooledFreeList:
        return PooledFreeList(capacity, initially_backed=initially_backed)

    def __init__(self, config: Config):
        block_size = config.kv_cache_block_size
        num_blocks = config.num_kvcache_blocks
        assert num_blocks > 0
        self.block_size = block_size
        self.dcp_world_size = getattr(config, "decode_context_parallel_size", 1)
        # dcp_rank is always 0 here: BlockManager runs only on the scheduler
        # (rank 0). DCP rank is used only to compute local token counts for
        # memory reservation; the actual per-rank routing is done in the workers.
        self.dcp_rank = 0

        # Unified-KV chunk arena (ATOM_V4_UNIFIED_KV_ARENA): elastic SWA/
        # compressed split. When on, a compressed block_id is a LOGICAL id and
        # the arena maps it to per-group physical pages; the logical id space is
        # sized to the arena's max compressed capacity so it can grow into
        # borrowed SWA chunks. Off -> block_id is its fixed physical slot (today).
        self.arena = self._build_arena(config, block_size)
        n_logical = (
            self.arena.max_compressed_blocks()
            if (self.arena is not None and self.arena.enabled)
            else num_blocks
        )
        self._arena_on = self.arena is not None and self.arena.enabled
        self.blocks: list[Block] = [Block(i) for i in range(n_logical)]
        self.hash_to_block_id: dict[int, int] = dict()
        # Free logical ids are split into two pools ONLY when the arena is on:
        #   free_block_ids       — BACKED-free: ref-0 ids that still hold an arena
        #                          page (cached KV or just-freed content). Reuse
        #                          needs NO new page; these are the ids the
        #                          cross-pool evictor lends by draining a chunk.
        #   _unbacked_free_ids   — ref-0 ids with NO arena page (their page was
        #                          lent to the SWA pool). Reuse must borrow a page
        #                          back via _arena_alloc_compressed.
        # A used id is always backed. Initially every id is UNbacked (no page
        # assigned until first _allocate_block). Arena OFF: block_id IS its fixed
        # physical slot (always "backed"), so the unbacked pool stays empty and
        # free_block_ids holds every id — byte-identical to the pre-arena path.
        self._free_list = self._make_primary_free_list(
            n_logical, initially_backed=not self._arena_on
        )
        # Migration aliases for existing diagnostics/tests. Allocation mechanics
        # are owned by PooledFreeList; these names no longer implement queues.
        self.free_block_ids = self._free_list.backed_ids
        self.free_block_ids_set = self._free_list.backed_set
        self._unbacked_free_ids = self._free_list.unbacked_ids
        self._unbacked_free_set = self._free_list.unbacked_set
        self.used_block_ids = self._free_list.used_ids
        self.enable_prefix_caching = config.enable_prefix_caching

        kv_events = getattr(config, "kv_events_config", None)
        self._events_enabled: bool = bool(kv_events and kv_events.enable)
        self._event_log: list[KVCacheEvent] | None = (
            [] if self._events_enabled else None
        )
        # Per-request cache slot pool. Used by attention types with a
        # stateful per-request buffer (GDN recurrent state, V4 compressor
        # state). The backing tensor is pre-allocated by ModelRunner sized
        # to max_num_seqs and excluded from `num_kvcache_blocks` at sizing
        # time, so admission only needs a free slot index from this list.
        # Each slot group contains slots_per_req() contiguous tensor indices
        # (1 for stateless / + num_spec for spec-decoding-aware variants).
        num_per_req_cache_groups: int = getattr(config, "num_per_req_cache_groups", 0)
        self.free_per_req_cache_groups: list[int] = list(
            range(num_per_req_cache_groups)
        )

        # SWA component: content-addressed sliding-window pool, the sole prefix-
        # cache sidecar owner now that the CSA boundary snapshot is fused into the
        # SWA chunk (feat/csa-swa-fusion) — no separate page pool. It is a no-op
        # for non-SWA models (num_swa_blocks == 0), so the compressed path stays
        # byte-identical. Under the unified-KV arena, size the SWA logical id
        # space to the arena's max SWA capacity so SWA can grow into borrowed
        # compressed chunks; else the fixed num_swa pool. full_retain/retention/
        # checkpoint carry the SWA sparse-checkpoint policy (and the arena
        # elastic-borrow lives inside SlidingWindowPool).
        _spec = getattr(config, "speculative_config", None)
        _mtp_k = int(getattr(_spec, "num_speculative_tokens", 0) or 0) if _spec else 0
        _num_swa = getattr(config, "num_swa_blocks", 0)
        if self.arena is not None and self.arena.enabled:
            _num_swa = max(_num_swa, self.arena.max_swa_blocks())
        self.swa = Dsv4SwaPool(
            num_blocks=_num_swa,
            window=getattr(config, "swa_window_size", 0),
            block_size=block_size,
            max_num_batched_tokens=getattr(config, "max_num_batched_tokens", 0),
            mtp_k=_mtp_k,
            full_retain=envs.ATOM_SWA_FULL_RETAIN,
            retention_interval=envs.ATOM_SWA_RETENTION_INTERVAL,
            checkpoint_frac=envs.ATOM_SWA_CHECKPOINT_FRAC,
        )
        # CSA boundary snapshot is fused into the SWA chunk (feat/csa-swa-fusion):
        # it has no separate page pool — capture writes into the block's SWA chunk
        # and retention rides the SWA pin. This flag only gates whether the
        # capture/restore plans are built; exposed via the
        # requires_csa_boundary_state property (a seam for a future non-SWA
        # sidecar that would not ride the SWA pin).
        self._require_csa_boundary_state = bool(
            getattr(config, "enable_v4_csa_prefix_state_cache", False)
        )
        # Wire the arena into SWA so SWA + compress borrow chunks from the shared
        # arena and reclaim from a sibling under pressure (pool-driven lending via
        # _evict_cold_for_borrow). CSA rides the SWA chunk, so no separate wiring.
        if self.arena is not None and self.arena.enabled:
            self.swa.attach_arena(self.arena, self._evict_cold_for_borrow)

    def _evict_cold_for_borrow(self) -> bool:
        """Free one arena page by evicting the coldest ref-0 page of either owner
        (compressed / SWA), so a starved owner can borrow the chunk. Tries each in
        turn; returns True if one was evicted (progress). Pool-driven lending.
        (CSA has no owner of its own — it rides the SWA chunk.)"""
        return self._evict_cold_compressed() or self.swa.evict_cold_for_arena()

    @staticmethod
    def _build_arena(config: Config, block_size: int):
        """Construct the unified-KV arena from ModelRunner-provided group specs
        when ATOM_V4_UNIFIED_KV_ARENA is on; None otherwise (fixed two-pool)."""
        if not envs.ATOM_V4_UNIFIED_KV_ARENA:
            return None
        specs = getattr(config, "v4_arena_group_specs", None)
        if not specs:
            return None
        return Dsv4UnifiedArena(block_size=block_size, group_specs=list(specs))

    def _evict_cold_compressed(self) -> bool:
        """Truly evict the coldest ref-0 compressed block (drop hash + return its
        arena pages) so the SWA pool can borrow the freed chunk. Returns False
        when no ref-0 compressed block is available. Pool-driven lending."""
        if self.arena is None:
            return False
        # free_block_ids holds only BACKED-free ids (all evictable candidates),
        # so no unbacked-skip dance is needed. Pop the coldest ref-0 backed id,
        # drop its hash + return its arena page, and move the id to the UNBACKED
        # free pool so it stays reusable (re-borrows a page on reuse) instead of
        # leaking out of circulation.
        while True:
            block_id = self._free_list.pop_backed()
            if block_id is None:
                return False
            block = self.blocks[block_id]
            if block.ref_count != 0:
                # Should not happen (used ids are not backed-free); self-heal
                # rather than spin on a stale queue entry.
                continue
            if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
                if self._event_log is not None:
                    self._event_log.append(_make_block_removed([block.hash]))
            block.reset()
            self.arena.free_compressed(block_id)
            self._free_list.move_to_unbacked(block_id)
            return True

    def _arena_alloc_compressed(self, block_id: int) -> None:
        """Back a compressed block with arena pages, evicting a cold sibling
        (SWA/CSA/compressed) on starvation and retrying (pool-driven three-way
        lending)."""
        if self.arena is None or self.arena.is_compressed_backed(block_id):
            return
        while True:
            try:
                self.arena.alloc_compressed(block_id)
                return
            except ArenaEmpty:
                if not self._evict_cold_for_borrow():
                    raise

    def _has_free_compressed(self, n: int) -> bool:
        """Whether ``n`` compressed blocks can be admitted. Off: free logical
        slots. On: enough free logical ids AND enough physical placement:
        reusing a BACKED-free id costs no new page, so only the shortfall beyond
        the backed-free ids must be backed by arena free pages + pages reclaimable
        from the SWA pool (one evicted SWA block frees one chunk per group, worth
        the tightest group's pages/chunk). Allocation pops backed-free first
        (`_pop_free_block`), so this accounting is sound."""
        if not self._arena_on:
            return len(self.free_block_ids_set) >= n
        backed_free = len(self.free_block_ids_set)
        total_free = backed_free + len(self._unbacked_free_set)
        if total_free < n:
            return False
        backable = (
            self.arena.compressed_available()
            + self.swa.num_evictable() * self.arena.compress_pages_per_chunk()
        )
        # backed_free ids reuse their held page (0 new pages); the remaining
        # (n - backed_free) must draw a page from `backable`.
        return backed_free + backable >= n

    @property
    def swa_enabled(self) -> bool:
        """Compatibility capability for callers that only need SWA status."""
        return self.swa.enabled

    @property
    def requires_csa_boundary_state(self) -> bool:
        """Whether CSA boundary-state capture/restore plans should be built.

        Kept as a property (not a bare read) as a seam: a future non-SWA sidecar
        that does not ride the SWA pin would gate its own lifecycle here.
        """
        return self._require_csa_boundary_state

    # ---------------- Scheduler-facing manager contract ---------------- #

    def materialize_window(self, seq: Sequence, seq_len: int) -> None:
        self.swa.materialize_window(seq, seq_len)

    def ensure_window_for_tokens(
        self, seq: Sequence, num_cached_tokens: int, num_new_tokens: int
    ) -> None:
        self.swa.ensure_for_tokens(seq, num_cached_tokens, num_new_tokens)

    def finish_prefill_chunk(self, seq: Sequence) -> None:
        self.swa.free_after_prefill_chunk(seq)

    def build_batch_tables(self, seqs: Iterable[Sequence]) -> KvBatchTables:
        seqs = list(seqs)
        block_tables = [seq.block_table for seq in seqs if seq.block_table]
        swa_block_tables = [seq.swa_block_table for seq in seqs if seq.block_table]
        boundary_sources = [
            int(getattr(seq, "csa_boundary_state_block_id", -1)) for seq in seqs
        ]
        return build_dsv4_batch_tables(
            arena=self.arena,
            block_tables=block_tables,
            swa_block_tables=swa_block_tables,
            v4_csa_boundary_source_ids=boundary_sources,
        )

    @property
    def num_total_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_free_per_req_cache_groups(self) -> int:
        return len(self.free_per_req_cache_groups)

    def kv_usage(self) -> float:
        return len(self.used_block_ids) / len(self.blocks) if self.blocks else 0.0

    def get_block(self, block_id: int) -> Block:
        return self.blocks[block_id]

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _pop_free_block(self) -> int:
        """Pop the next available free block id. Prefer a BACKED-free id (reuse
        its held arena page, no borrow) before an UNBACKED-free one (must borrow a
        page back on _allocate_block). Backed-first keeps `_has_free_compressed`
        accounting sound. Arena off: `_unbacked_free_ids` is empty (unchanged)."""
        return self._free_list.pop()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # Evict stale hash entry before resetting. ATOM's eviction is lazy:
        # blocks sit in the free queue with their hash intact until the slot
        # is re-allocated, so this point — not `deallocate()` — is the true
        # eviction event.
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
            if self._event_log is not None:
                self._event_log.append(_make_block_removed([block.hash]))
        block.reset()
        self._free_list.mark_used(block_id)
        # Ensure arena pages back this block (no-op off / already backed; borrows
        # from SWA under pressure). A backed id keeps its pages across content
        # cycles; they return to the arena only via _evict_cold_compressed, which
        # moves the id to the UNBACKED free pool.
        self._arena_alloc_compressed(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        assert self._free_list.deallocate(block_id), f"block {block_id} not in use"

    def _dcp_num_blocks(self, seq_len: int) -> int:
        if self.dcp_world_size <= 1:
            return (seq_len + self.block_size - 1) // self.block_size
        from atom.model_ops.dcp_ops import get_dcp_local_seq_lens

        local_len = get_dcp_local_seq_lens(
            np.array([seq_len]), self.dcp_world_size, self.dcp_rank
        )[0]
        return int((local_len + self.block_size - 1) // self.block_size)

    def _effective_block_size(self):
        return self.block_size * self.dcp_world_size

    def can_allocate(self, seq: Sequence) -> int:
        """Return number of cache-hit blocks (>=0) if seq fits, else -1.

        The hit count is the contiguous run of cache hits starting at the
        prompt's first block. On the first miss we break: subsequent blocks
        cannot match either (hash is chained, so a divergent token breaks the
        chain for the rest of the prompt). The last block is never considered
        for reuse — prefill must forward at least one block to produce
        sampler logits, so it always comes from the free pool.

        Caller (scheduler) passes the returned hit count to `allocate()`,
        avoiding a second hash pass.
        """
        # State cache (mamba / V4 compressor ring) has its own pre-allocated
        # tensor; admission only needs a free slot index, not extra paged
        # blocks. See `allocate()` for the budget reasoning.
        if seq.has_per_req_cache and not self.free_per_req_cache_groups:
            return -1
        if not self.enable_prefix_caching:
            if not self._has_free_compressed(self._dcp_num_blocks(len(seq))):
                return -1
            # SWA admission: only the per-request windowed peak (filled
            # incrementally + window-freed), not the whole prompt. No-op / True
            # when SWA disabled.
            if not self.swa.has_free(self.swa.admission_blocks(seq)):
                return -1
            return 0
        # Step 1: compressed prefix (CSA/HCA/indexer share the block hash and
        # read the WHOLE history, so this stays a full front-to-back chained
        # match). Record each block's hash for the SWA scan below.
        h = -1
        compressed_hit = 0
        block_hashes: list[int] = []
        compressed_block_ids: list[int] = []
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            block_hashes.append(h)
            compressed_block_ids.append(block_id)
            compressed_hit += 1
        # Step 2: SWA only needs the trailing window before the boundary to be
        # present (SWA is local). Scan right-to-left within the compressed prefix
        # for the largest boundary whose window is SWA-cached (vLLM
        # SlidingWindowManager; simple-hybrid one pass). Reduces compressed_hit
        # → num_cached_blocks so we never reuse a block whose in-window SWA is
        # gone (#1417), while out-of-window front blocks (SWA-freed) don't block
        # the hit.
        # SWA trailing-window gate: shrink the compressed hit to the largest
        # boundary whose trailing window is SWA-present. Under CSA-into-SWA fusion
        # the CSA boundary rides that same SWA block, so this one gate covers CSA
        # too — a boundary whose SWA window is present has its fused CSA state
        # present. The assert guards the caller's compressed-prefix invariant
        # (every counted hit block must have a materialized physical id).
        assert (
            len(compressed_block_ids) >= compressed_hit
        ), "missing physical ids for compressed prefix hit"
        num_cached_blocks = self.swa.bounded_hit(seq, compressed_hit, block_hashes)
        # Instrumentation: record the pre-gate compressed hit so CacheStats can
        # separate reuse lost to the SWA tail gate (compressed_hit -
        # num_cached_blocks) from reuse lost to compressed eviction.
        seq.num_compressed_hit_blocks = compressed_hit
        # Free-pool demand: blocks we actually reuse minus those already used
        # (shared ref); blocks we drop from the hit become fresh → counted.
        num_new_blocks = seq.num_blocks
        for i in range(num_cached_blocks):
            if self.hash_to_block_id[block_hashes[i]] in self.used_block_ids:
                num_new_blocks -= 1
        if not self._has_free_compressed(num_new_blocks):
            return -1
        # SWA new-block demand is bounded by the windowed peak (filled
        # incrementally + window-freed), not the full new-block count. No-op /
        # True when SWA disabled.
        if not self.swa.has_free(min(num_new_blocks, self.swa.admission_blocks(seq))):
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0):
        """Allocate blocks for `seq`. `num_cached_blocks` is the hit count
        returned by `can_allocate` (0 if caller didn't call it).

        Hash registration is deferred to hash_blocks(), called from
        scheduler.postprocess() once the forward has computed each block's
        KV. This keeps the manager correct under future chunked-prefill
        scheduling: a block spanning multiple steps must not be published as
        a hash until fully filled.
        """
        assert not seq.block_table
        # SWA tail-gate: only the trailing window before the hit boundary is
        # SWA-reused; earlier blocks are out of window (never read by the resumed
        # forward) → mark -1 (matches self.swa.bounded_hit; keeps swa_block_table
        # aligned with block_table). swa_hit_start == boundary - swa_tail_blocks
        # on a full-window hit, and 0 on a short/partial hit (whole prefix in
        # one window → all present, all claimed).
        # SWA tail-gate: only the trailing window before the hit boundary is
        # SWA-reused; earlier (out-of-window) blocks get -1. A tail size of zero
        # when disabled makes swa_hit_start == num_cached_blocks → every SWA call
        # below is a no-op (swa_block_table stays empty for non-SWA models).
        swa_hit_start = max(0, num_cached_blocks - self.swa.tail_blocks)
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                # Cache hit on a free-pool block — claim without _allocate_block
                # (whose reset() would evict the hash entry and destroy the
                # cache for everyone).
                assert block.ref_count == 0
                block.ref_count = 1
                # A cache hit lands only on a BACKED block (its KV is still
                # resident); unbacked ids have no hash. discard from both sets so
                # the id leaves the free pool cleanly.
                self._free_list.mark_used(block_id)
            seq.block_table.append(block_id)
            if i < swa_hit_start:
                self.swa.alloc_placeholder(seq)  # out of window: never read → -1
            else:
                self.swa.claim_cached(seq, h, token_ids)  # trailing window: reuse
        # Fused CSA (feat/csa-swa-fusion): the restore source is the terminal
        # cached block's LOGICAL c4 swa id — its physical SWA chunk (content-
        # addressed, retention-pinned) holds the captured boundary in its fused
        # tail segment. bound_hit already guaranteed that block's SWA window is
        # present, so its swa id is live (>= 0). The scheduler translates this to
        # the c4 physical swa page for the restore kernel; no separate pool pin is
        # needed (the SWA reuse claim + retention pin already protect the chunk).
        seq.csa_boundary_state_block_id = -1
        if (
            self.requires_csa_boundary_state
            and num_cached_blocks
            and len(seq.swa_block_table) >= num_cached_blocks
        ):
            # Fused CSA lives in the SWA chunk, so it exists only when SWA does.
            swa_id = seq.swa_block_table[num_cached_blocks - 1]
            seq.csa_boundary_state_block_id = (
                int(swa_id) if swa_id is not None and swa_id >= 0 else -1
            )
        for _ in range(num_cached_blocks, self._dcp_num_blocks(len(seq))):
            block_id = self._pop_free_block()
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
            # Uncached blocks: -1 placeholder keeps swa_block_table the same
            # length as block_table; ensure_for_tokens fills the current chunk's
            # window slots before each forward, free_after_prefill_chunk releases
            # out-of-window ones.
            self.swa.alloc_placeholder(seq)
        seq.num_cached_tokens = num_cached_blocks * self.block_size

        # Per-request cache: claim one slot index from the pre-allocated
        # state tensor (e.g. GDN mamba_k_cache, V4 compressor state + SWA
        # ring). The state tensor's memory was already excluded from
        # `num_kvcache_blocks` in ModelRunner._compute_kv_budget(), so
        # admitting a seq adds no further paged-block cost. The slot cap
        # (`free_per_req_cache_groups` size = `max_num_seqs`) is the sole
        # admission bound for state cache.
        if seq.has_per_req_cache:
            seq.per_req_cache_group = self.free_per_req_cache_groups.pop()

    def hash_blocks(self, seq: Sequence, num_new_tokens: int) -> None:
        """Register hashes for blocks finalized by the most recent step.

        Called from scheduler.postprocess() after the forward completes, so a
        block's hash is only published once its KV is actually computed. The
        `[start, end)` range covers blocks fully filled by this step:
          start = first block whose first token was at num_cached_tokens
          end   = first block not yet fully filled (excludes the partial one)
        Caller passes `num_new_tokens` = tokens forwarded in this step. For
        single-shot prefill that's `seq.num_tokens - seq.num_cached_tokens`;
        chunked prefill will pass the per-chunk count.
        """
        if not self.enable_prefix_caching:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + num_new_tokens) // self.block_size
        if start >= end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        record = self._event_log is not None
        store_run_parent: int | None = h if h != -1 else None
        store_run_hashes: list[int] = []
        store_run_tokens: list[int] = []
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
            # Publish the parallel SWA block under the same content hash so
            # cross-request hits can reuse its sliding-window KV (no-op when SWA
            # disabled or the slot is a -1 window-freed sentinel).
            # Publishing the SWA block under this hash ALSO publishes its fused
            # CSA boundary (feat/csa-swa-fusion): the capture kernel wrote the
            # boundary into this block's SWA chunk during the forward, so a later
            # prefix hit that reuses the content-addressed SWA block restores the
            # boundary for free — no separate CSA publish needed.
            self.swa.publish_hash(seq, i, h, token_ids)
            if record:
                store_run_hashes.append(h)
                store_run_tokens.extend(token_ids)
        if record and store_run_hashes:
            self._event_log.append(
                _make_block_stored(
                    store_run_hashes,
                    store_run_tokens,
                    store_run_parent,
                    self.block_size,
                )
            )

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        self.swa.release(
            seq
        )  # release SWA blocks + clear swa_block_table (no-op if disabled)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        seq.csa_boundary_state_block_id = -1
        if seq.has_per_req_cache and seq.per_req_cache_group >= 0:
            self.free_per_req_cache_groups.append(seq.per_req_cache_group)
            seq.per_req_cache_group = -1

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        seq_len = len(seq)
        current_blocks = len(seq.block_table)
        ebs = self._effective_block_size()
        needed_blocks = (seq_len + num_new_tokens + ebs - 1) // ebs
        new_blocks_needed = max(0, needed_blocks - current_blocks)
        if not self._has_free_compressed(new_blocks_needed):
            return False
        if not self.swa.has_free(new_blocks_needed):  # True when SWA disabled
            return False
        return True

    def may_append(self, seq: Sequence, num_new_tokens: int = 1):
        # Note: in disaggregated (P/D) mode the scheduler skips this call on
        # the first decode step after remote prefill, because blocks were
        # already allocated during the KV transfer phase.
        block_table = seq.block_table
        seq_len = len(seq)
        # Check if we need to allocate a new block
        # When len(seq) % block_size == 1, we need a new block for the next token
        # When block_size == 1, every token needs a new block
        ebs = self._effective_block_size()
        if 0 < seq_len % ebs <= num_new_tokens or ebs == 1:
            needed_blocks = (seq_len + ebs - 1) // ebs
            while len(block_table) < needed_blocks:
                # Decode-generated blocks: token not finalized yet (depends on
                # sampling / speculative verification), so we cannot compute a
                # correct hash here.  Just allocate the block without hashing.
                block_id = self._pop_free_block()
                self._allocate_block(block_id)
                block_table.append(block_id)
                self.swa.append_new(seq)  # lockstep SWA block (no-op if disabled)
        # Reclaim SWA blocks that just fell out of the window (no-op if disabled).
        self.swa.free_out_of_window(seq, len(seq))

    # ---------------- KV event API ---------------- #

    def take_events(self) -> list[KVCacheEvent]:
        """Drain and return events accumulated since the last call."""
        if self._event_log is None or not self._event_log:
            return []
        self._event_log, events = [], self._event_log
        return events

    def clear_cache(self) -> None:
        """Drop every prefix-cache entry. Used by `/reset_prefix_cache`-style
        admin APIs. Does NOT touch blocks currently held by live sequences —
        they remain valid via their block_table refs, just unhashable for
        future requests."""
        self.hash_to_block_id.clear()
        for block in self.blocks:
            if block.ref_count == 0:
                block.hash = -1
                block.token_ids = []
        if self._event_log is not None:
            self._event_log.append(_make_all_cleared())

    @property
    def kv_events_enabled(self) -> bool:
        """True iff KV events are being recorded."""
        return self._event_log is not None

    def record_remote_store(
        self,
        block_hashes: list[int],
        token_ids: list[int],
        parent_block_hash: int | None = None,
    ) -> None:
        """Emit a BlockStored(medium=REMOTE) for blocks received from a remote
        KV transfer producer (Mooncake/MoriIO decode side). Called by the
        KVConnector worker once the transfer completes so external KV-cache
        consumers (LMCache, etc.) can track remote-resident blocks."""
        if self._event_log is None or not block_hashes:
            return
        self._event_log.append(
            _make_block_stored(
                block_hashes,
                token_ids,
                parent_block_hash,
                self.block_size,
                medium=MEDIUM_REMOTE,
            )
        )
