# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque
from collections.abc import Callable

from atom.model_engine.kv_block import Block


class BlockPool:
    """Paged blocks with ref counts and a content-addressed index.

    Two instances exist: the compressed KV blocks `BlockManager` owns, and the
    sliding-window blocks `SlidingWindowPool` owns. They are separate index
    spaces over separate tensors — `sub_pool_spec.py` will not even let them
    share a count — but the bookkeeping has to be *identical*, because both are
    addressed by the same chained content hash. A prefix hit is a joint claim
    on the two, so a divergence in when either drops a hash would let one pool
    promise a boundary the other cannot honour.

    Eviction is lazy: a freed block keeps its hash and contents until its slot
    is handed out for something else, so a later request can still claim a
    freed-but-not-overwritten block. `allocate` — not `free` — is therefore the
    eviction event, and `on_evict` fires there.
    """

    def __init__(self, num_blocks: int, on_evict: Callable[[int], None] | None = None):
        self.num_blocks: int = num_blocks
        self._on_evict = on_evict
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self._hash_to_block_id: dict[int, int] = {}
        # The queue may hold ids that were re-claimed straight off the free list
        # (`claim`), so membership in the set — not presence in the queue — is
        # what makes an id free. `pop` skips the stale entries.
        self._free_queue: deque[int] = deque(range(num_blocks))
        self._free: set[int] = set(range(num_blocks))
        self._used: set[int] = set()

    # ------------------------------- counts -------------------------------- #
    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)

    @property
    def num_indexed(self) -> int:
        """Blocks reachable by content hash, live or merely not-yet-overwritten."""
        return len(self._hash_to_block_id)

    def has_free(self, n: int) -> bool:
        return len(self._free) >= n

    def is_used(self, block_id: int) -> bool:
        return block_id in self._used

    def block(self, block_id: int) -> Block:
        return self.blocks[block_id]

    # ------------------------------- index --------------------------------- #
    def lookup(self, h: int) -> int:
        """Block id indexed under content hash `h`, or -1."""
        return self._hash_to_block_id.get(h, -1)

    def publish(self, block_id: int, h: int, token_ids: list[int]) -> None:
        """Index `block_id` under the content hash of the tokens it now holds."""
        block = self.blocks[block_id]
        block.update(h, token_ids)
        self._hash_to_block_id[h] = block_id

    def clear_index(self) -> None:
        """Drop every content-hash entry, keeping blocks live sequences hold.

        Those stay valid through their block_table refs; they are simply no
        longer reachable by hash, so no future request can claim them.
        """
        self._hash_to_block_id.clear()
        for block in self.blocks:
            if block.ref_count == 0:
                block.hash = -1
                block.token_ids = []

    # ---------------------------- allocation ------------------------------- #
    def pop(self) -> int:
        """Next free block id, skipping ids the queue holds only staleley."""
        while self._free_queue:
            block_id = self._free_queue.popleft()
            if block_id in self._free:
                self._free.discard(block_id)
                return block_id
        raise AssertionError("No free blocks available")

    def allocate(self, block_id: int) -> Block:
        """Take `block_id` for fresh content, evicting whatever it held."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self._hash_to_block_id.get(block.hash) == block_id:
            del self._hash_to_block_id[block.hash]
            if self._on_evict is not None:
                self._on_evict(block.hash)
        block.reset()
        self._free.discard(block_id)
        self._used.add(block_id)
        return block

    def claim(self, block_id: int) -> Block:
        """Take a share of `block_id` for the content it already holds.

        The cache-hit counterpart of `allocate`, and deliberately not built on
        it: `allocate`'s reset would drop the hash and destroy the entry for
        every other request that could still hit it.
        """
        block = self.blocks[block_id]
        if block_id in self._used:
            block.ref_count += 1
        else:
            assert block.ref_count == 0
            block.ref_count = 1
            self._free.discard(block_id)
            self._used.add(block_id)
        return block

    def free(self, block_id: int) -> None:
        self.blocks[block_id].ref_count -= 1
        if self.blocks[block_id].ref_count == 0:
            self._used.remove(block_id)
            self._free_queue.append(block_id)
            self._free.add(block_id)
