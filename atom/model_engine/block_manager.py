# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque

import numpy as np
import xxhash
from atom.config import Config
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
        # Evict stale hash entry before resetting
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
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
        # State cache (mamba / V4 compressor ring) has its own pre-allocated
        # tensor; admission only needs a free slot index, not extra paged
        # blocks. See `allocate()` for the budget reasoning.
        per_req_cache_slot_ok = (not seq.has_per_req_cache) or len(
            self.free_per_req_cache_groups
        ) > 0
        if not self.enable_prefix_caching:
            return (
                len(self.free_block_ids_set) >= seq.num_blocks and per_req_cache_slot_ok
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
        return len(self.free_block_ids_set) >= needed_free and per_req_cache_slot_ok

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False

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
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

        # Per-request cache: claim one slot index from the pre-allocated
        # state tensor (e.g. GDN mamba_k_cache, V4 compressor state + SWA
        # ring). The state tensor's memory was already excluded from
        # `num_kvcache_blocks` in ModelRunner._compute_kv_budget() — see
        # `available_for_pool = available_for_kv - per_req_cache_tensor_bytes`
        # — so admitting a seq adds no further paged-block cost. The slot
        # cap (`free_per_req_cache_groups` size = `max_num_seqs`) is the
        # sole admission bound for state cache.
        if seq.has_per_req_cache:
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
