# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque

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
from atom.model_engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


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
    def __init__(self, config: Config):
        block_size = config.kv_cache_block_size
        num_blocks = config.num_kvcache_blocks
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.free_block_ids_set: set[int] = set(range(num_blocks))
        self.used_block_ids: set[int] = set()
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

        # M2 paged-SWA: optional second block pool for the sliding-window KV,
        # with an independent free-list/hash so out-of-window SWA blocks can be
        # freed while the compressed blocks persist. `swa_enabled=False` →
        # every SWA branch below short-circuits and the compressed path is
        # byte-identical to before.
        num_swa_blocks: int = getattr(config, "num_swa_blocks", 0)
        self.swa_enabled: bool = num_swa_blocks > 0
        self.swa_window: int = getattr(config, "swa_window_size", 0)
        self.swa_blocks: list[Block] = [Block(i) for i in range(num_swa_blocks)]
        self.swa_hash_to_block_id: dict[int, int] = dict()
        self.swa_free_block_ids: deque[int] = deque(range(num_swa_blocks))
        self.swa_free_block_ids_set: set[int] = set(range(num_swa_blocks))
        self.swa_used_block_ids: set[int] = set()

    # ---------------- SWA pool primitives (mirror the compressed pool) ------- #
    def _pop_free_swa_block(self) -> int:
        while self.swa_free_block_ids:
            block_id = self.swa_free_block_ids.popleft()
            if block_id in self.swa_free_block_ids_set:
                self.swa_free_block_ids_set.discard(block_id)
                return block_id
        raise AssertionError("No free SWA blocks available")

    def _allocate_swa_block(self, block_id: int) -> Block:
        block = self.swa_blocks[block_id]
        assert block.ref_count == 0
        if (
            block.hash != -1
            and self.swa_hash_to_block_id.get(block.hash) == block_id
        ):
            del self.swa_hash_to_block_id[block.hash]
        block.reset()
        self.swa_free_block_ids_set.discard(block_id)
        self.swa_used_block_ids.add(block_id)
        return block

    def _deallocate_swa_block(self, block_id: int):
        assert self.swa_blocks[block_id].ref_count == 0
        self.swa_used_block_ids.remove(block_id)
        self.swa_free_block_ids.append(block_id)
        self.swa_free_block_ids_set.add(block_id)

    def _free_swa_out_of_window(self, seq: Sequence):
        """M2 brick-2: release SWA blocks that have fallen fully behind the
        sliding window — they're never read again by this request, and freeing
        them bounds live SWA memory to ~window per request.

        Block ``i`` covers tokens ``[i*bs, (i+1)*bs)``; the latest query (pos
        ``seq_len-1``) attends down to ``seq_len-window``, so block ``i`` is
        fully out of window once ``(i+1)*bs <= seq_len - window``. Freed blocks
        keep their hash + KV until their pool slot is actually reused (lazy
        eviction, same as the compressed pool), so a cross-request hit can still
        reuse a freed-but-not-overwritten SWA block. The trailing window (plus
        up to one boundary block) is retained for prefix reuse.
        """
        if not self.swa_enabled or self.swa_window <= 0:
            return
        seq_len = len(seq)
        free_before = max(0, (seq_len - self.swa_window) // self.block_size)
        free_before = min(free_before, len(seq.swa_block_table))
        for i in range(free_before):
            swa_id = seq.swa_block_table[i]
            if swa_id < 0:
                continue  # already window-freed
            block = self.swa_blocks[swa_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_swa_block(swa_id)
            seq.swa_block_table[i] = -1  # sentinel: out of window

    def _allocate_swa_for_cached(self, h: int, token_ids: list[int], seq: Sequence):
        """Claim the cached SWA block for hash `h` (caller guarantees it exists,
        via the can_allocate intersection) and append to seq.swa_block_table.
        Mirrors the compressed cached-hit claim."""
        swa_id = self.swa_hash_to_block_id[h]
        block = self.swa_blocks[swa_id]
        if swa_id in self.swa_used_block_ids:
            block.ref_count += 1
        else:
            assert block.ref_count == 0
            block.ref_count = 1
            self.swa_free_block_ids_set.discard(swa_id)
            self.swa_used_block_ids.add(swa_id)
        seq.swa_block_table.append(swa_id)

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _pop_free_block(self) -> int:
        """Pop the next available free block id from the FIFO queue (lazy cleanup)."""
        while self.free_block_ids:
            block_id = self.free_block_ids.popleft()
            if block_id in self.free_block_ids_set:
                self.free_block_ids_set.discard(block_id)
                return block_id
        raise AssertionError("No free blocks available")

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
        self.free_block_ids_set.discard(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        self.free_block_ids_set.add(block_id)

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
            if len(self.free_block_ids_set) < seq.num_blocks:
                return -1
            # M2: SWA pool allocates in parallel (1:1 with compressed in M2
            # brick-1; brick-2 window-freeing reduces this). Require space.
            if self.swa_enabled and len(self.swa_free_block_ids_set) < seq.num_blocks:
                return -1
            return 0
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            # M2: intersect with the SWA pool — a compressed hit is only usable
            # if this block's SWA is ALSO still cached (the SWA pool evicts /
            # window-frees independently). Stops the hit at the first block
            # whose SWA is gone, so allocate() never reuses compressed-without-SWA.
            if self.swa_enabled:
                swa_id = self.swa_hash_to_block_id.get(h, -1)
                if swa_id == -1 or self.swa_blocks[swa_id].token_ids != token_ids:
                    break
            num_cached_blocks += 1
            # Hits on already-used blocks share refs; only free-pool hits
            # consume a free slot (claimed inline in allocate()).
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids_set) < num_new_blocks:
            return -1
        # M2 brick-1: SWA pool is parallel (a cached compressed block has its
        # SWA cached too; allocate() mirrors the reuse), so SWA new-block demand
        # equals the compressed one. Require the SWA pool can cover it.
        if self.swa_enabled and len(self.swa_free_block_ids_set) < num_new_blocks:
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
                self.free_block_ids_set.discard(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
            if self.swa_enabled:
                self._allocate_swa_for_cached(h, token_ids, seq)
        for _ in range(num_cached_blocks, seq.num_blocks):
            block_id = self._pop_free_block()
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
            if self.swa_enabled:
                swa_id = self._pop_free_swa_block()
                self._allocate_swa_block(swa_id)
                seq.swa_block_table.append(swa_id)
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
            # M2: publish the parallel SWA block under the same content hash so
            # cross-request hits can reuse its sliding-window KV.
            if self.swa_enabled and i < len(seq.swa_block_table):
                swa_block = self.swa_blocks[seq.swa_block_table[i]]
                swa_block.update(h, token_ids)
                self.swa_hash_to_block_id[h] = swa_block.block_id
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
        if self.swa_enabled:
            for swa_id in reversed(seq.swa_block_table):
                if swa_id < 0:
                    continue  # window-freed slot (brick-2)
                block = self.swa_blocks[swa_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_swa_block(swa_id)
            seq.swa_block_table.clear()
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        if seq.has_per_req_cache and seq.per_req_cache_group >= 0:
            self.free_per_req_cache_groups.append(seq.per_req_cache_group)
            seq.per_req_cache_group = -1

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        seq_len = len(seq)
        current_blocks = len(seq.block_table)
        needed_blocks = (
            seq_len + num_new_tokens + self.block_size - 1
        ) // self.block_size
        new_blocks_needed = max(0, needed_blocks - current_blocks)
        if len(self.free_block_ids_set) < new_blocks_needed:
            return False
        if self.swa_enabled and len(self.swa_free_block_ids_set) < new_blocks_needed:
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
        if 0 < seq_len % self.block_size <= num_new_tokens or self.block_size == 1:
            needed_blocks = (seq_len + self.block_size - 1) // self.block_size
            while len(block_table) < needed_blocks:
                # Decode-generated blocks: token not finalized yet (depends on
                # sampling / speculative verification), so we cannot compute a
                # correct hash here.  Just allocate the block without hashing.
                block_id = self._pop_free_block()
                self._allocate_block(block_id)
                block_table.append(block_id)
                if self.swa_enabled:
                    swa_id = self._pop_free_swa_block()
                    self._allocate_swa_block(swa_id)
                    seq.swa_block_table.append(swa_id)
        # M2 brick-2: reclaim SWA blocks that just fell out of the window.
        if self.swa_enabled:
            self._free_swa_out_of_window(seq)

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
