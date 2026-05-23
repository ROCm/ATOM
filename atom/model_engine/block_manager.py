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
) -> BlockStored:
    """Construct a BlockStored event from a coalesced run of new blocks."""
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=tokens,
        block_size=block_size,
        medium=MEDIUM_GPU,
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
        # Per-request cache: per-request slot pool + equiv-block accounting.
        # Used by attention types that maintain stateful per-request buffers
        # outside the paged KV pool — currently GDN recurrent state, future
        # DeepseekV4 ring buffer + compressor state. See
        # AttentionMetadataBuilder.compute_per_req_cache_bytes() for details.
        # Each slot group contains slots_per_req() contiguous tensor indices
        # (1 for stateless / + num_spec for spec-decoding-aware variants).
        self.per_req_cache_equiv_blocks: int = getattr(
            config, "per_req_cache_equiv_blocks", 0
        )
        num_per_req_cache_groups: int = getattr(config, "num_per_req_cache_groups", 0)
        self.free_per_req_cache_groups: list[int] = list(
            range(num_per_req_cache_groups)
        )
        # seq_id → list of accounting block_ids (memory bookkeeping only)
        self.per_req_cache_accounting: dict[int, list[int]] = {}

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

    def can_allocate(self, seq: Sequence) -> bool:
        per_req_cache_cost = (
            self.per_req_cache_equiv_blocks if seq.has_per_req_cache else 0
        )
        per_req_cache_slot_ok = (not seq.has_per_req_cache) or len(
            self.free_per_req_cache_groups
        ) > 0
        if not self.enable_prefix_caching:
            return (
                len(self.free_block_ids_set) >= seq.num_blocks + per_req_cache_cost
                and per_req_cache_slot_ok
            )
        # Dry-run: count how many blocks would be cache hits
        h = -1
        cache_miss = False
        needed_free = 0
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            # If the entire prompt would be cached, force the last full block
            # to recompute so prefill has at least one token to forward and
            # produce logits for the next-token sampler.
            if (
                not cache_miss
                and i == seq.num_blocks - 1
                and len(token_ids) == self.block_size
            ):
                cache_miss = True
            if cache_miss:
                needed_free += 1
        return (
            len(self.free_block_ids_set) >= needed_free + per_req_cache_cost
            and per_req_cache_slot_ok
        )

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False

        # Coalesce newly-stored blocks within this allocate() call into a
        # single BlockStored event. Parent hash is the last full-block hash
        # already in the cache when the new-store run begins (so subscribers
        # can reconstruct the prefix chain), or None at the prompt root.
        store_run_hashes: list[int] = []
        store_run_tokens: list[int] = []
        store_run_parent: int | None = None
        last_known_hash: int | None = None

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = (
                self.hash_to_block_id.get(h, -1) if self.enable_prefix_caching else -1
            )
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            # If the entire prompt would be cached, force the last full block
            # to recompute so prefill has at least one token to forward and
            # produce logits for the next-token sampler. Must mirror the same
            # condition in can_allocate() so the block budget agrees.
            if (
                not cache_miss
                and i == seq.num_blocks - 1
                and len(token_ids) == self.block_size
            ):
                cache_miss = True
            if cache_miss:
                block_id = self._pop_free_block()
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self.blocks[block_id]
                    assert block.ref_count == 0
                    block.ref_count = 1
                    self.free_block_ids_set.discard(block_id)
                    self.used_block_ids.add(block_id)
            if h != -1:
                # cache_miss && full block + real hash = a freshly stored block.
                # cache_miss is sticky once a block is missing, so this is the
                # right signal for "newly stored at this index".
                if self._event_log is not None and cache_miss:
                    if not store_run_hashes:
                        store_run_parent = last_known_hash
                    store_run_hashes.append(h)
                    store_run_tokens.extend(token_ids)
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
                last_known_hash = h
            seq.block_table.append(block_id)

        if store_run_hashes and self._event_log is not None:
            self._event_log.append(
                _make_block_stored(
                    store_run_hashes,
                    store_run_tokens,
                    store_run_parent,
                    self.block_size,
                )
            )

        # Per-request cache: allocate equiv blocks (memory accounting) +
        # one slot index from the per-req cache pool. Slot indexes into
        # ModelRunner's per-req cache tensors (e.g. mamba_k_cache for GDN).
        if seq.has_per_req_cache:
            accounting_blocks = []
            for _ in range(self.per_req_cache_equiv_blocks):
                block_id = self._pop_free_block()
                self._allocate_block(block_id)
                accounting_blocks.append(block_id)
            self.per_req_cache_accounting[seq.id] = accounting_blocks
            seq.per_req_cache_group = self.free_per_req_cache_groups.pop()

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        if seq.has_per_req_cache and seq.per_req_cache_group >= 0:
            for block_id in self.per_req_cache_accounting.pop(seq.id, []):
                block = self.blocks[block_id]
                block.ref_count = 0  # accounting blocks bypass ref-counting
                self._deallocate_block(block_id)
            self.free_per_req_cache_groups.append(seq.per_req_cache_group)
            seq.per_req_cache_group = -1

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        seq_len = len(seq)
        current_blocks = len(seq.block_table)
        needed_blocks = (
            seq_len + num_new_tokens + self.block_size - 1
        ) // self.block_size
        new_blocks_needed = max(0, needed_blocks - current_blocks)
        return len(self.free_block_ids_set) >= new_blocks_needed

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
            BlockStored(
                block_hashes=block_hashes,
                parent_block_hash=parent_block_hash,
                token_ids=token_ids,
                block_size=self.block_size,
                medium=MEDIUM_REMOTE,
            )
        )
