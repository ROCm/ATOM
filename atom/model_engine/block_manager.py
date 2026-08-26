# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
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
from atom.model_engine.kv_block import Block
from atom.model_engine.sequence import Sequence
from atom.model_engine.swa_pool import SlidingWindowPool
from atom.utils import envs

logger = logging.getLogger("atom")


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


# Physical index reserved as the KV write sink under asymmetric rapidserve.
# MLA's KV latent is TP-replicated, so after a wide prefill EVERY rank holds an
# identical copy — but only the rank co-located with a request's target GPU may
# write it. The other ranks point their block table / SWA block table /
# per-request cache slot at this index, so their redundant writes land in
# scratch instead of corrupting the live pool. Masking indices (not control
# flow) keeps the TP ranks in lockstep on their shared communicator.
#
# Reserved ONLY when asymmetric rapidserve is active; every other configuration
# keeps its full block budget.
DUMP_INDEX = 0


class BlockManager:
    def __init__(self, config: Config):
        block_size = config.kv_cache_block_size
        num_blocks = config.num_kvcache_blocks
        assert num_blocks > 0
        # See DUMP_INDEX. Index 0 stays permanently unallocated in all three
        # pools so masked prefill ranks always have a safe sink.
        reserve_dump = getattr(config, "disagg_prefill_tp_size", 0) > 1
        first_free = DUMP_INDEX + 1 if reserve_dump else 0
        self.block_size = block_size
        self.dcp_world_size = config.decode_context_parallel_size
        # dcp_rank is always 0 here: BlockManager runs only on the scheduler
        # (rank 0). DCP rank is used only to compute local token counts for
        # memory reservation; the actual per-rank routing is done in the workers.
        self.dcp_rank = 0
        # Prefix-cache hashing / reuse granularity: under DCP one block_table
        # entry maps to a virtual block of `block_size * dcp_world_size` global
        # tokens (see _hash_block_size). == block_size when DCP is off.
        self.hash_block_size = self.block_size * self.dcp_world_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(first_free, num_blocks))
        self.free_block_ids_set: set[int] = set(range(first_free, num_blocks))
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
            range(min(first_free, num_per_req_cache_groups), num_per_req_cache_groups)
        )

        # Sliding-window KV pool (DeepSeek-V4). A separate content-addressed pool
        # with its own free-list/hash so out-of-window SWA blocks free while the
        # compressed blocks persist. BlockManager drives it in lockstep with the
        # compressed pool via `self.swa`. Disabled (no-op) for non-SWA models, so
        # every delegation below is unconditional and the compressed path stays
        # byte-identical. See atom/model_engine/swa_pool.py.
        _spec = getattr(config, "speculative_config", None)
        _mtp_k = int(getattr(_spec, "num_speculative_tokens", 0) or 0) if _spec else 0
        self.swa = SlidingWindowPool(
            num_blocks=getattr(config, "num_swa_blocks", 0),
            window=getattr(config, "swa_window_size", 0),
            block_size=block_size,
            max_num_batched_tokens=getattr(config, "max_num_batched_tokens", 0),
            mtp_k=_mtp_k,
            full_retain=envs.ATOM_SWA_FULL_RETAIN,
            retention_interval=envs.ATOM_SWA_RETENTION_INTERVAL,
            checkpoint_frac=envs.ATOM_SWA_CHECKPOINT_FRAC,
            reserve_dump_block=reserve_dump,
        )

    @property
    def swa_enabled(self) -> bool:
        return self.swa.enabled

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

    # --- Prefix-cache block accounting granularity ---------------------------
    # Under DCP one entry of `block_table` maps to a VIRTUAL block spanning
    # `block_size * dcp_world_size` consecutive global tokens (each rank stores
    # its `block_size` interleaved tokens in that physical block). So prefix
    # cache hashing / reuse must be done per virtual block, not per physical
    # block — otherwise the logical block index runs past the (dcp-shrunk)
    # block_table. For dcp_world_size == 1 this reduces to the physical size.
    def _hash_block_size(self) -> int:
        return self.hash_block_size

    def _n_hash_blocks(self, seq: Sequence) -> int:
        hbs = self.hash_block_size
        return (len(seq) + hbs - 1) // hbs

    def _hash_block_tokens(self, seq: Sequence, i: int) -> list[int]:
        hbs = self.hash_block_size
        return seq.token_ids[i * hbs : (i + 1) * hbs]

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
            if len(self.free_block_ids_set) < self._dcp_num_blocks(len(seq)):
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
        for i in range(self._n_hash_blocks(seq) - 1):
            token_ids = self._hash_block_tokens(seq, i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            block_hashes.append(h)
            compressed_hit += 1
        # Step 2: SWA only needs the trailing window before the boundary to be
        # present (SWA is local). Scan right-to-left within the compressed prefix
        # for the largest boundary whose window is SWA-cached (vLLM
        # SlidingWindowManager; simple-hybrid one pass). Reduces compressed_hit
        # → num_cached_blocks so we never reuse a block whose in-window SWA is
        # gone (#1417), while out-of-window front blocks (SWA-freed) don't block
        # the hit.
        num_cached_blocks = self.swa.bounded_hit(seq, compressed_hit, block_hashes)
        # Instrumentation: record the pre-gate compressed hit so CacheStats can
        # separate reuse lost to the SWA tail gate (compressed_hit -
        # num_cached_blocks) from reuse lost to compressed eviction.
        seq.num_compressed_hit_blocks = compressed_hit
        # Free-pool demand: blocks we actually reuse minus those already used
        # (shared ref); blocks we drop from the hit become fresh → counted.
        num_new_blocks = self._n_hash_blocks(seq)
        for i in range(num_cached_blocks):
            if self.hash_to_block_id[block_hashes[i]] in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids_set) < num_new_blocks:
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
        # forward) → mark -1 (matches swa.bounded_hit; keeps swa_block_table
        # aligned with block_table). swa_hit_start == boundary - swa_tail_blocks
        # on a full-window hit, and 0 on a short/partial hit (whole prefix in
        # one window → all present, all claimed).
        # SWA tail-gate: only the trailing window before the hit boundary is
        # SWA-reused; earlier (out-of-window) blocks get -1. swa.tail_blocks == 0
        # when disabled → swa_hit_start == num_cached_blocks → every SWA call
        # below is a no-op (swa_block_table stays empty for non-SWA models).
        swa_hit_start = max(0, num_cached_blocks - self.swa.tail_blocks)
        h = -1
        for i in range(num_cached_blocks):
            token_ids = self._hash_block_tokens(seq, i)
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
            if i < swa_hit_start:
                self.swa.alloc_placeholder(seq)  # out of window: never read → -1
            else:
                self.swa.claim_cached(seq, h, token_ids)  # trailing window: reuse
        for _ in range(num_cached_blocks, self._dcp_num_blocks(len(seq))):
            block_id = self._pop_free_block()
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
            # Uncached blocks: -1 placeholder keeps swa_block_table the same
            # length as block_table; ensure_for_tokens fills the current chunk's
            # window slots before each forward, free_after_prefill_chunk releases
            # out-of-window ones.
            self.swa.alloc_placeholder(seq)
        seq.num_cached_tokens = num_cached_blocks * self._hash_block_size()

        # Per-request cache: claim one slot index from the pre-allocated
        # state tensor (e.g. GDN mamba_k_cache, V4 compressor state + SWA
        # ring). The state tensor's memory was already excluded from
        # `num_kvcache_blocks` in ModelRunner._compute_kv_budget(), so
        # admitting a seq adds no further paged-block cost. The slot cap
        # (`free_per_req_cache_groups` size = `max_num_seqs`) is the sole
        # admission bound for state cache.
        if seq.has_per_req_cache:
            seq.per_req_cache_group = self.free_per_req_cache_groups.pop()

    def hash_blocks(
        self, seq: Sequence, num_new_tokens: int, start_tokens: int | None = None
    ) -> None:
        """Register hashes for blocks finalized by the most recent step.

        Called from scheduler.postprocess() after the forward completes, so a
        block's hash is only published once its KV is actually computed. The
        `[start, end)` range covers blocks fully filled by this step:
          start = first block whose first token was at num_cached_tokens
          end   = first block not yet fully filled (excludes the partial one)
        Caller passes `num_new_tokens` = tokens forwarded in this step. For
        single-shot prefill that's `seq.num_tokens - seq.num_cached_tokens`;
        chunked prefill will pass the per-chunk count.

        `start_tokens` overrides the token offset the range starts at. Pipeline-
        parallel schedule-time advancement already bumped seq.num_cached_tokens
        past this chunk, so the head passes the chunk's pre-advance offset here.
        """
        if not self.enable_prefix_caching:
            return
        hbs = self._hash_block_size()
        base = seq.num_cached_tokens if start_tokens is None else start_tokens
        start = base // hbs
        end = (base + num_new_tokens) // hbs
        if start >= end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        record = self._event_log is not None
        store_run_parent: int | None = h if h != -1 else None
        store_run_hashes: list[int] = []
        store_run_tokens: list[int] = []
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = self._hash_block_tokens(seq, i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
            # Publish the parallel SWA block under the same content hash so
            # cross-request hits can reuse its sliding-window KV (no-op when SWA
            # disabled or the slot is a -1 window-freed sentinel).
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

    def publish_loaded_prefix(
        self,
        seq: Sequence,
        start_token: int,
        end_token: int,
    ) -> int:
        """Publish a successfully loaded offload prefix into the GPU cache index.

        LMCache restores KV directly into already allocated physical blocks, so
        those blocks do not pass through ``hash_blocks()``. Without explicitly
        publishing them here, the current request can consume the restored KV,
        but later requests cannot discover it through ``can_allocate()`` and
        repeatedly load the same prefix from CPU.

        Only complete, hash-block-aligned loaded blocks are published. Existing
        canonical mappings win: concurrent requests may load the same prefix
        into different physical blocks, and replacing the canonical mapping
        would make its eventual eviction remove the wrong cache entry.
        """
        if not self.enable_prefix_caching:
            return 0

        start_token = max(0, int(start_token))
        end_token = min(int(end_token), int(seq.num_prompt_tokens))
        if end_token <= start_token:
            return 0
        hbs = self._hash_block_size()
        if start_token % hbs != 0:
            logger.warning(
                "Cannot publish offload prefix with unaligned start: "
                "seq=%s start=%d hash_block_size=%d",
                seq.id,
                start_token,
                hbs,
            )
            return 0

        start_block = start_token // hbs
        end_block = end_token // hbs
        if end_block <= start_block:
            return 0
        if end_block > len(seq.block_table):
            logger.warning(
                "Cannot publish offload prefix beyond block table: "
                "seq=%s end_block=%d blocks=%d",
                seq.id,
                end_block,
                len(seq.block_table),
            )
            return 0

        if start_block == 0:
            parent_hash = -1
        else:
            parent = self.blocks[seq.block_table[start_block - 1]]
            if parent.hash == -1:
                logger.warning(
                    "Cannot publish offload prefix without indexed parent: "
                    "seq=%s start_block=%d",
                    seq.id,
                    start_block,
                )
                return 0
            parent_hash = parent.hash

        indexed_tokens = 0
        for i in range(start_block, end_block):
            token_ids = self._hash_block_tokens(seq, i)
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            block_hash = self.compute_hash(token_ids, parent_hash)
            canonical_id = self.hash_to_block_id.get(block_hash)

            if block.hash not in (-1, block_hash):
                logger.warning(
                    "Refusing to overwrite indexed block during offload "
                    "promotion: seq=%s block=%d",
                    seq.id,
                    block_id,
                )
                break

            if canonical_id is not None:
                canonical = self.blocks[canonical_id]
                if canonical.token_ids != token_ids:
                    logger.warning(
                        "Hash collision while publishing offload prefix: "
                        "seq=%s block=%d canonical=%d",
                        seq.id,
                        block_id,
                        canonical_id,
                    )
                    break
                # Keep the canonical hash_to_block_id mapping, but annotate this
                # request's duplicate physical block as well. hash_blocks()
                # needs the final loaded block's hash as the parent when it
                # publishes the newly computed suffix.
                block.update(block_hash, token_ids)
            else:
                block.update(block_hash, token_ids)
                self.hash_to_block_id[block_hash] = block_id
                if self._event_log is not None:
                    self._event_log.append(
                        _make_block_stored(
                            [block_hash],
                            token_ids,
                            parent_hash if parent_hash != -1 else None,
                            self.block_size,
                        )
                    )

            indexed_tokens += hbs
            parent_hash = block_hash

        return indexed_tokens

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
        if seq.has_per_req_cache and seq.per_req_cache_group >= 0:
            self.free_per_req_cache_groups.append(seq.per_req_cache_group)
            seq.per_req_cache_group = -1

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        seq_len = len(seq)
        current_blocks = len(seq.block_table)
        ebs = self._effective_block_size()
        needed_blocks = (seq_len + num_new_tokens + ebs - 1) // ebs
        new_blocks_needed = max(0, needed_blocks - current_blocks)
        if len(self.free_block_ids_set) < new_blocks_needed:
            return False
        return self.swa.has_free(new_blocks_needed)  # True when SWA disabled

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
