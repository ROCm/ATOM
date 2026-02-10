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
        self.used_block_ids: set[int] = set()
        self.enable_prefix_caching = config.enable_prefix_caching

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        # self.free_block_ids.appendleft(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # def allocate(self, seq: Sequence, num_additional_tokens):
    #     assert not seq.block_table
    #     h = -1
    #     cache_miss = False
    #     num_blocks = (
    #         seq.num_tokens + num_additional_tokens + seq.block_size - 1
    #     ) // seq.block_size
    #     for i in range(num_blocks):
    #         token_ids = seq.block(i)
    #         h = (
    #             self.compute_hash(token_ids, h)
    #             if len(token_ids) == self.block_size
    #             else -1
    #         )
    #         block_id = (
    #             self.hash_to_block_id.get(h, -1) if self.enable_prefix_caching else -1
    #         )
    #         if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    #             cache_miss = True
    #         if cache_miss:
    #             block_id = self.free_block_ids[0]
    #             block = self._allocate_block(block_id)
    #         else:
    #             seq.num_cached_tokens += self.block_size
    #             if block_id in self.used_block_ids:
    #                 block = self.blocks[block_id]
    #                 block.ref_count += 1
    #             else:
    #                 block = self._allocate_block(block_id)
    #         if h != -1:
    #             block.update(h, token_ids)
    #             self.hash_to_block_id[h] = block_id
    #         seq.block_table.append(block_id)
    def allocate(self, seq: Sequence, num_additional_tokens: int):
        assert not seq.block_table
        total_tokens = seq.num_tokens + num_additional_tokens
        num_blocks = (total_tokens + self.block_size - 1) // self.block_size
        for i in range(num_blocks):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence, num_new_tokens: int = 1):
        block_table = seq.block_table
        seq_len = len(seq)

        final_len = seq_len + num_new_tokens - 1
        needed_blocks = (final_len + self.block_size - 1) // self.block_size

        while len(block_table) < needed_blocks:
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # TODO: support prefix cache
