"""DSV4 window pool: retention and shared-arena backing."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

from atom.kv_cache.pools.chunk_arena import ArenaEmpty
from atom.kv_cache.pools.windowed_kv_pool import WindowedKvPool
from atom.model_engine.sequence import Sequence


class Dsv4SwaPool(WindowedKvPool):
    """Add CSA-into-SWA retention and arena sibling lending."""

    def __init__(
        self,
        num_blocks: int,
        window: int,
        block_size: int,
        max_num_batched_tokens: int,
        mtp_k: int,
        full_retain: bool = False,
        retention_interval: int = 0,
        checkpoint_frac: float = 0.5,
    ):
        super().__init__(
            num_blocks=num_blocks,
            window=window,
            block_size=block_size,
            max_num_batched_tokens=max_num_batched_tokens,
            mtp_k=mtp_k,
            full_retain=full_retain,
        )
        self.retention_blocks = (
            retention_interval // block_size
            if retention_interval > 0 and block_size > 0
            else 0
        )
        self.sparse_retain = self.full_retain and self.retention_blocks > 0
        self.checkpoint_capacity = (
            int(num_blocks * checkpoint_frac) if self.sparse_retain else 0
        )
        self.checkpoint_lru: OrderedDict[int, None] = OrderedDict()
        self.arena: Any | None = None
        self._evict_sibling: Callable[[], bool] | None = None

    def attach_arena(self, arena: Any, evict_sibling: Callable[[], bool]) -> None:
        self.arena = arena
        self._evict_sibling = evict_sibling
        if arena is not None and getattr(arena, "enabled", False):
            self._free_list.move_all_to_unbacked()

    def _ensure_backing(self, block_id: int) -> None:
        if self.arena is None or self.arena.is_swa_backed(block_id):
            return
        while True:
            try:
                self.arena.alloc_swa(block_id)
                return
            except ArenaEmpty:
                if self._evict_sibling is None or not self._evict_sibling():
                    raise

    # Historical private name kept while callers migrate.
    _arena_alloc_swa = _ensure_backing

    def evict_cold_for_arena(self) -> bool:
        if self.arena is None:
            return False
        while True:
            block_id = self._free_list.pop_backed()
            if block_id is None:
                return False
            block = self.blocks[block_id]
            if block.ref_count != 0:
                continue
            if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
            block.reset()
            self.arena.free_swa(block_id)
            self._free_list.move_to_unbacked(block_id)
            return True

    def num_evictable(self) -> int:
        return len(self.free_block_ids_set) if self.arena is not None else 0

    def has_free(self, count: int) -> bool:
        if not self.enabled:
            return True
        if self.arena is None:
            return len(self.free_block_ids_set) >= count
        backed_free = len(self.free_block_ids_set)
        total_free = backed_free + len(self._unbacked_free_set)
        if total_free < count:
            return False
        return backed_free + self.arena.swa_available() >= count

    def _is_checkpoint(self, seq: Sequence, index: int) -> bool:
        if not self.sparse_retain:
            return True
        needed = self.tail_blocks
        if index % self.retention_blocks >= self.retention_blocks - needed:
            return True
        prompt_blocks = seq.num_prompt_tokens // self.block_size
        return index >= prompt_blocks - needed

    def _pin_checkpoint(self, block_id: int) -> None:
        if block_id in self.checkpoint_lru:
            self.checkpoint_lru.move_to_end(block_id)
            return
        self.blocks[block_id].ref_count += 1
        self.checkpoint_lru[block_id] = None
        while len(self.checkpoint_lru) > self.checkpoint_capacity:
            old_id, _ = self.checkpoint_lru.popitem(last=False)
            block = self.blocks[old_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._dealloc(old_id)


# External compatibility name used by existing integrations.
SlidingWindowPool = Dsv4SwaPool

__all__ = ["Dsv4SwaPool", "SlidingWindowPool"]
