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
from atom.model_engine.kv_block import STATE_SLOT_CLASS, Block
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_pool import StateCheckpointPool
from atom.model_engine.swa_pool import SWA_POOL_CLASS, SlidingWindowPool
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
    def __init__(self, config: Config):
        block_size = config.kv_cache_block_size
        num_blocks = config.num_kvcache_blocks
        assert num_blocks > 0
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
        # Sizing published `entries` per cache class plus the per-request
        # multiplicity the declaring backend asked for (1 for a single
        # committed state, + num_spec where a rollback slot per speculated
        # token is kept). One group is what a single request occupies, i.e.
        # `entries // entries_per_req` contiguous tensor indices.
        pool_entries: dict = getattr(config, "pool_entries", None) or {}
        pool_per_req: dict = getattr(config, "pool_entries_per_req", None) or {}
        state_entries = int(pool_entries.get(STATE_SLOT_CLASS, 0))
        state_per_req = int(pool_per_req.get(STATE_SLOT_CLASS, 1)) or 1
        # Total capacity, kept so callers can tell "all slots busy" (transient)
        # from "no slots were ever created" (permanent).
        self.num_per_req_cache_groups = state_entries // state_per_req
        # FIFO, matching `free_block_ids`: a group handed back keeps its content
        # and may still be indexed as a state checkpoint, so hand-out order must
        # be least-recently-freed first or every checkpoint is evicted by the
        # next admission. `_pop_state_group` performs the lazy eviction.
        self.free_per_req_cache_groups: deque[int] = deque(
            range(self.num_per_req_cache_groups)
        )
        # Content index over the free groups above (see StateCheckpointPool):
        # a state checkpoint IS a free group whose content is still valid, so
        # this pool holds no capacity of its own and never blocks admission.
        self.state = StateCheckpointPool(
            self.num_per_req_cache_groups, enabled=self.enable_prefix_caching
        )
        # Minimum tokens the forward after a fork must cover for the new group
        # to come out self-contained (see AttentionBackend.min_fork_tokens).
        # 0 = the backend cannot fork, so nothing is ever published and every
        # hit on a stateful model shrinks to 0.
        self.state_min_fork_tokens = int(
            getattr(config, "state_min_fork_tokens", 0) or 0
        )
        # Tokens between checkpoints (--state-checkpoint-interval-tokens).
        self.state_checkpoint_interval_tokens = max(
            0, int(getattr(config, "state_checkpoint_interval_tokens", 0) or 0)
        )
        # A checkpoint is filed under the content hash of the last block it
        # covers, so a publish position that isn't a hash-block boundary can
        # never be looked up — the ladder would publish into a void.
        assert not (
            self.state.enabled
            and self.state_checkpoint_interval_tokens
            and self.state_checkpoint_interval_tokens % self.hash_block_size
        ), (
            f"--state-checkpoint-interval-tokens="
            f"{self.state_checkpoint_interval_tokens} must be a multiple of the "
            f"prefix-cache hash block size {self.hash_block_size}"
        )

        # Sliding-window KV pool (DeepSeek-V4). A separate content-addressed pool
        # with its own free-list/hash so out-of-window SWA blocks free while the
        # compressed blocks persist. BlockManager drives it in lockstep with the
        # compressed pool via `self.swa`. Disabled (no-op) for non-SWA models, so
        # every delegation below is unconditional and the compressed path stays
        # byte-identical. See atom/model_engine/swa_pool.py.
        _spec = getattr(config, "speculative_config", None)
        _mtp_k = int(getattr(_spec, "num_speculative_tokens", 0) or 0) if _spec else 0
        _swa_blocks = int(pool_entries.get(SWA_POOL_CLASS, 0))
        _window = int(
            getattr(getattr(config, "hf_config", None), "sliding_window", 0) or 0
        )
        # A backend only declares the SWA class for a windowed architecture, so
        # the window has to be in hf_config. Fail here rather than let the pool
        # come up enabled with window=0, which frees every block immediately.
        assert not _swa_blocks or _window > 0, (
            f"sub-pool {SWA_POOL_CLASS!r} was sized to {_swa_blocks} blocks but "
            "hf_config has no sliding_window"
        )
        self.swa = SlidingWindowPool(
            num_blocks=_swa_blocks,
            window=_window if _swa_blocks else 0,
            block_size=block_size,
            max_num_batched_tokens=getattr(config, "max_num_batched_tokens", 0),
            mtp_k=_mtp_k,
            full_retain=envs.ATOM_SWA_FULL_RETAIN,
            retention_interval=envs.ATOM_SWA_RETENTION_INTERVAL,
            checkpoint_frac=envs.ATOM_SWA_CHECKPOINT_FRAC,
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

    def _pop_state_group(self) -> int:
        """Hand out a per-request state group, evicting its checkpoint if any.

        The state twin of `_pop_free_block`: groups sit in the FIFO carrying
        whatever the last owner left in them, and re-allocation — not the free
        — is the eviction event.
        """
        group = self.free_per_req_cache_groups.popleft()
        self.state.invalidate(group)
        return group

    def _claim_state_group(self, group: int) -> None:
        """Claim one specific free group — a checkpoint the caller looked up.

        Linear in the free list, unlike `_pop_state_group`. That is deliberate:
        the queue stays the single source of truth for "how many groups are
        free", which admission and every caller of `len()` rely on. The scan is
        bounded by max_num_seqs and runs once per resuming request, against a
        `can_allocate` that already hashed every block of the prompt.
        """
        self.free_per_req_cache_groups.remove(group)

    def _release_state_group(self, group: int) -> None:
        self.free_per_req_cache_groups.append(group)

    def release_state_pins(self) -> None:
        """Return the previous step's fork sources to the free list.

        Called once per engine step before scheduling. A fork source is read by
        the forward that was already issued when it is handed out again, and the
        next owner's forward is issued after that one on the same stream, so
        stream ordering covers the overlap.
        """
        for group in self.state.take_pins():
            self._release_state_group(group)

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

    def _state_bounded_hit(
        self, seq: Sequence, hit: int, block_hashes: list[int]
    ) -> int:
        """Shrink `hit` to the rightmost boundary with a resumable state."""
        if not (seq.has_per_req_cache and self.state.enabled):
            return hit
        hbs = self._hash_block_size()
        min_fork = self.state_min_fork_tokens

        def forkable(boundary: int) -> bool:
            # The resumed forward reads the checkpoint and writes a fresh group;
            # it must cover enough tokens to leave that group self-contained.
            return seq.num_tokens - boundary * hbs >= min_fork

        return self.state.bounded_hit(hit, block_hashes, forkable)

    def _gated_hit(
        self, seq: Sequence, compressed_hit: int, block_hashes: list[int]
    ) -> int:
        """Rightmost boundary that satisfies the SWA and state gates together.

        Both gates answer "the rightmost boundary <= X that I accept", and
        neither is monotone in the other, so they cannot be applied in series:
        the largest SWA-complete boundary need not carry a state checkpoint, and
        walking back to one that does can land on a boundary whose trailing SWA
        window is gone. `allocate` then calls `swa.claim_cached` for a hash the
        SWA pool never promised — which is exactly the guarantee that method's
        docstring asks the caller for.

        So alternate: take the SWA boundary, ask the state index for the nearest
        checkpoint at or below it, re-ask SWA about that one, and repeat. Each
        round strictly decreases the candidate, so it terminates; with either
        gate disabled the first round already agrees and returns.
        """
        boundary = self.swa.bounded_hit(seq, compressed_hit, block_hashes)
        while boundary > 0:
            with_state = self._state_bounded_hit(seq, boundary, block_hashes)
            if with_state == 0:
                return 0
            if with_state == boundary:
                return boundary  # already SWA-approved this round
            swa_ok = self.swa.bounded_hit(seq, with_state, block_hashes)
            if swa_ok == with_state:
                return with_state
            boundary = swa_ok
        return 0

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
        # the hit —
        # plus step 3, the per-request state: neither the SSM recurrent state nor
        # the V4 compressor ring can be rebuilt from cached blocks — the cache
        # holds the compressor's output, the state is its rolling input window —
        # so a boundary is only resumable where somebody published the state.
        # `_gated_hit` settles the two gates jointly; neither can be applied to
        # the other's answer.
        num_cached_blocks = self._gated_hit(seq, compressed_hit, block_hashes)
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
        # state tensor (e.g. GDN mamba_k_cache, the V4 compressor ring). The
        # state class took its bytes before the paged class was sized in
        # ModelRunner.get_num_blocks(), so admitting a seq adds no further
        # paged-block cost. The slot cap
        # (`free_per_req_cache_groups` size = `max_num_seqs`) is the sole
        # admission bound for state cache.
        if seq.has_per_req_cache:
            self._attach_state_group(seq, h if num_cached_blocks > 0 else -1)

    def _attach_state_group(self, seq: Sequence, hit_hash: int) -> None:
        """Give `seq` a state group, resuming from a checkpoint when one exists.

        `hit_hash` is the content hash of the last reused block (-1 for a cold
        start). `can_allocate` already shrank the hit to a boundary that carries
        a checkpoint, so a lookup miss here just means the pool is off.

        Resuming forks: the checkpoint stays published and the request writes a
        fresh group, so a second request hitting the same prefix still finds it.
        When no second group is free the request adopts the checkpoint instead
        — still correct, the state is exactly the one it wanted, it just spends
        the checkpoint rather than sharing it.

        A checkpoint is read-only, so several requests in one step may fork off
        the same one. The first takes it off the free list and the pin covers
        every reader until `release_state_pins`; a later one in that same step
        finds it already pinned and only needs a group to write into. Adopting
        is then off the table — the pin means someone else's forward still has
        to read it.
        """
        src = self.state.lookup(hit_hash) if hit_hash != -1 else -1
        if src < 0:
            seq.per_req_cache_group = self._pop_state_group()
            seq.state_fork_src = -1
            return
        shared = self.state.is_pinned(src)
        if not shared:
            self._claim_state_group(src)
        if self.free_per_req_cache_groups:
            seq.per_req_cache_group = self._pop_state_group()
            seq.state_fork_src = src
            # Held off the free list until the forward that reads it is issued.
            self.state.pin(src)
            return
        # `can_allocate` admitted this seq against a non-empty free list and
        # nothing else has run since, so the list can only be empty here if this
        # seq itself just took the last group — which is `src`, unshared.
        assert not shared, "no group to fork into and the source is being read"
        self.state.invalidate(src)
        seq.per_req_cache_group = src
        seq.state_fork_src = -1

    def hash_blocks(
        self,
        seq: Sequence,
        num_new_tokens: int,
        start_tokens: int | None = None,
        next_forward_tokens: int | None = None,
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

        `next_forward_tokens` reaches `is_state_publish_pos`; see there. Left
        unset it reads the prompt's remainder, which is the prefill answer.
        """
        if not self.enable_prefix_caching:
            return
        hbs = self._hash_block_size()
        base = seq.num_cached_tokens if start_tokens is None else start_tokens
        start = base // hbs
        end = (base + num_new_tokens) // hbs
        if start >= end:
            return
        # Watermark for the decode-side continuation, maintained here so every
        # prefill path feeds it without knowing about it.
        seq.num_hashed_tokens = max(seq.num_hashed_tokens, end * hbs)
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
        if self.is_state_publish_pos(seq, base + num_new_tokens, next_forward_tokens):
            self._publish_state_checkpoint(seq, h)

    def hash_decode_blocks(
        self, seq: Sequence, committed_kv_len: int, next_forward_tokens: int = 0
    ) -> None:
        """Register hashes for generated blocks filled up to `committed_kv_len`.

        `may_append` allocates decode blocks without hashing them: at allocation
        time their tokens have not been sampled, and under speculative decoding
        part of what the forward writes is about to be rejected.

        `committed_kv_len` counts the tokens for which neither still applies —
        id final, KV computed — and is a hard line, not a hint. It stops short
        of any token no forward has read yet: that token's KV slot is written
        by the next forward, and a block published over an unwritten slot hands
        a later request KV that may never arrive at all (the seq can finish
        first). Prefill's `hash_blocks` draws the same line from its own side,
        at `num_cached_tokens + chunk`.

        Without this the prefix cache indexes prompt blocks only, and a
        follow-up turn — previous prompt plus previous answer — matches nothing
        beyond the original prompt.

        `next_forward_tokens` reaches `is_state_publish_pos`; see there. It
        defaults to "no next forward", i.e. hash but never checkpoint, so a
        caller opts into decode-point publishing rather than out of it.
        """
        if not self.enable_prefix_caching:
            return
        base = seq.num_hashed_tokens
        if committed_kv_len > base:
            self.hash_blocks(
                seq,
                committed_kv_len - base,
                start_tokens=base,
                next_forward_tokens=next_forward_tokens,
            )

    def cancel_state_fork(self, seq: Sequence) -> bool:
        """Undo a pending fork by adopting its source group.

        Called when the forward that was going to carry the fork turns out too
        short to fill a fresh group (`min_fork_tokens`). Both flavours collapse
        to the same move — take the source over and spend its checkpoint:
        a resume becomes the non-sharing hit, a publish becomes no publish at
        all.

        Returns False when the source cannot be taken over because another
        request in this same step forks off it too: adopting means writing into
        a group that request's forward still has to read. The caller keeps the
        fork instead and must not shorten the forward below `min_fork_tokens`.
        """
        src = seq.state_fork_src
        if src < 0:
            return True
        if self.state.pin_count(src) > 1:
            return False
        self._release_state_group(seq.per_req_cache_group)
        self.state.invalidate(src)
        if self.state.is_pinned(src):
            self.state.unpin(src)  # resume source: was held off the free list
        else:
            self._claim_state_group(src)  # publish source: was handed back
        seq.per_req_cache_group = src
        seq.state_fork_src = -1
        return True

    def state_publish_limit(self, seq: Sequence) -> int:
        """Rightmost prompt position this seq may checkpoint at, 0 for none.

        `is_state_publish_pos` solved for prefill: the last rung of the interval
        ladder that still leaves `min_fork_tokens` of prompt to forward. Kept as
        its own method because the scheduler needs the bound up front, to cut
        prefill chunks so they land on the ladder.

        0 means this seq never publishes off its prompt — including every prompt
        shorter than one interval, which is the point: a workload whose prompts
        all fit under the interval pays nothing for a feature it would never hit.
        """
        if not (seq.has_per_req_cache and self.state.enabled):
            return 0
        if self.state_min_fork_tokens <= 0:
            return 0
        interval = self.state_checkpoint_interval_tokens
        if interval <= 0:
            return 0
        forkable = seq.num_prompt_tokens - self.state_min_fork_tokens
        return max((forkable // interval) * interval, 0)

    def is_state_publish_pos(
        self, seq: Sequence, pos: int, next_forward_tokens: int | None = None
    ) -> bool:
        """Whether a forward ending at `pos` should checkpoint its state.

        A ladder of resume points, one every `state_checkpoint_interval_tokens`
        of context. Publishing is capacity-neutral (the group handed away is
        replaced by one from the free list), but each one costs the publisher an
        extra forward — the prompt gets cut at the rung — so the interval is what
        keeps that cost amortized instead of per-request.

        `next_forward_tokens` is how many tokens the forward right after this one
        carries, and gates the whole thing: publishing hands the group away, so
        that forward has to fill the replacement by itself (`min_fork_tokens`).
        Unset means the prompt's remainder, the prefill answer; decode passes
        one. Everything else follows from that one number — a backend needing a
        long fork (V4's ring, 131) simply never qualifies mid-generation, and a
        request stopping on this step passes 0 and never publishes a checkpoint
        nothing will ever fork from.

        The position must be exact. The group holds state as of the forward's
        last token, so a forward that overshoots a rung is ahead of the hash it
        would be filed under; the scheduler cuts prefill chunks to land here,
        and a path that doesn't simply publishes nothing.
        """
        if not (seq.has_per_req_cache and self.state.enabled):
            return False
        interval = self.state_checkpoint_interval_tokens
        if interval <= 0 or self.state_min_fork_tokens <= 0:
            return False
        if next_forward_tokens is None:
            next_forward_tokens = seq.num_prompt_tokens - pos
        if next_forward_tokens < self.state_min_fork_tokens:
            return False
        return pos > 0 and pos % interval == 0

    def _publish_state_checkpoint(self, seq: Sequence, h: int) -> None:
        """Hand the seq's state group to the checkpoint index under hash `h`.

        The request moves to a fresh group; the next forward reads the published
        one and writes the new one, which is why a published group is never
        written again. Best-effort: with no free group the seq simply keeps
        writing its own and no checkpoint is taken.
        """
        old = seq.per_req_cache_group
        if old < 0 or not self.free_per_req_cache_groups:
            return
        seq.per_req_cache_group = self._pop_state_group()
        seq.state_fork_src = old
        self._release_state_group(old)
        self.state.publish(h, old)

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
        # The block table is gone, so nothing of this seq is hashed any more.
        # Covers preemption too, which frees through here and re-prefills.
        seq.num_hashed_tokens = 0
        seq.block_table.clear()
        if seq.has_per_req_cache and seq.per_req_cache_group >= 0:
            # Only the group the seq was writing. A checkpoint it published is
            # already back on the free list under the state index, and a fork
            # source it borrowed is returned by `release_state_pins`.
            self._release_state_group(seq.per_req_cache_group)
            seq.per_req_cache_group = -1
            seq.state_fork_src = -1

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        seq_len = len(seq)
        current_blocks = len(seq.block_table)
        ebs = self._effective_block_size()
        needed_blocks = (seq_len + num_new_tokens + ebs - 1) // ebs
        new_blocks_needed = max(0, needed_blocks - current_blocks)
        if len(self.free_block_ids_set) < new_blocks_needed:
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
