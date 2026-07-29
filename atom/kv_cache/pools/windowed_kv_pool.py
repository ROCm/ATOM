"""Generic content-addressed sliding-window KV pool lifecycle."""

from __future__ import annotations

from collections import OrderedDict

from atom.kv_cache.pools.pooled_free_list import PooledFreeList
from atom.model_engine.kv_block import Block
from atom.model_engine.sequence import Sequence


class WindowedKvPool:
    """Logical SWA blocks with window freeing and content-hash reuse.

    Physical backing and retention policy are hooks for layout-specific
    subclasses.  A zero-sized pool self-disables every operation.
    """

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
        del retention_interval, checkpoint_frac
        self.enabled = num_blocks > 0
        self.window = int(window)
        self.block_size = int(block_size)
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.full_retain = bool(full_retain)
        win_with_spec = self.window + int(mtp_k)
        self.tail_blocks = (
            max(
                1,
                (win_with_spec - 1 + self.block_size - 1) // self.block_size,
            )
            if self.window > 0
            else 0
        )
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self._free_list = PooledFreeList(num_blocks)
        self.free_block_ids = self._free_list.backed_ids
        self.free_block_ids_set = self._free_list.backed_set
        self._unbacked_free_ids = self._free_list.unbacked_ids
        self._unbacked_free_set = self._free_list.unbacked_set
        self.used_block_ids = self._free_list.used_ids

        # Retention hooks are inactive in the generic pool.
        self.sparse_retain = False
        self.checkpoint_lru: OrderedDict[int, None] = OrderedDict()

    # -------------------------- subclass hooks -------------------------- #

    def _ensure_backing(self, block_id: int) -> None:
        del block_id

    def evict_cold_for_arena(self) -> bool:
        return False

    def num_evictable(self) -> int:
        return 0

    def _is_checkpoint(self, seq: Sequence, index: int) -> bool:
        del seq, index
        return False

    def _pin_checkpoint(self, block_id: int) -> None:
        del block_id

    # ----------------------------- primitives --------------------------- #

    def _pop(self) -> int:
        return self._free_list.pop(error_message="No free windowed KV blocks available")

    def _alloc(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self._free_list.mark_used(block_id)
        self._ensure_backing(block_id)
        return block

    def _dealloc(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        assert self._free_list.deallocate(
            block_id
        ), f"windowed KV block {block_id} not in use"

    # -------------------------- admission / hit ------------------------- #

    def has_free(self, count: int) -> bool:
        if not self.enabled:
            return True
        return len(self.free_block_ids_set) + len(self._unbacked_free_set) >= count

    def admission_blocks(self, seq: Sequence) -> int:
        if not self.enabled:
            return 0
        if self.full_retain:
            chunk_peak = (
                self.max_num_batched_tokens + self.block_size - 1
            ) // self.block_size + 1
            return min(chunk_peak, seq.num_blocks)
        return min(self.tail_blocks + 1, seq.num_blocks)

    def bounded_hit(
        self, seq: Sequence, prefix_blocks: int, block_hashes: list[int]
    ) -> int:
        """Return the largest prefix with a complete trailing SWA window."""
        if not self.enabled:
            return prefix_blocks
        needed = self.tail_blocks
        contiguous = 0
        for index in range(prefix_blocks - 1, -1, -1):
            block_id = self.hash_to_block_id.get(block_hashes[index], -1)
            if block_id != -1 and self.blocks[block_id].token_ids == seq.block(index):
                contiguous += 1
                if contiguous >= needed:
                    return index + contiguous
            else:
                contiguous = 0
        return contiguous

    # ----------------------------- allocation --------------------------- #

    def claim_cached(
        self, seq: Sequence, block_hash: int, token_ids: list[int]
    ) -> None:
        del token_ids
        if not self.enabled:
            return
        block_id = self.hash_to_block_id[block_hash]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            assert block.ref_count == 0
            block.ref_count = 1
            self._free_list.mark_used(block_id)
        if block_id in self.checkpoint_lru:
            self.checkpoint_lru.move_to_end(block_id)
        seq.swa_block_table.append(block_id)

    def alloc_placeholder(self, seq: Sequence) -> None:
        if self.enabled:
            seq.swa_block_table.append(-1)

    def append_new(self, seq: Sequence) -> None:
        if not self.enabled:
            return
        block_id = self._pop()
        self._alloc(block_id)
        seq.swa_block_table.append(block_id)

    def ensure_for_tokens(
        self, seq: Sequence, num_cached_tokens: int, num_new_tokens: int
    ) -> None:
        if not self.enabled or num_new_tokens <= 0:
            return
        seq_len = num_cached_tokens + num_new_tokens
        start_block = num_cached_tokens // self.block_size
        end_block = (seq_len - 1) // self.block_size
        free_before = (
            0
            if self.full_retain
            else max(0, (seq_len - self.window) // self.block_size)
        )
        start_block = max(start_block, free_before)
        table = seq.swa_block_table
        for index in range(start_block, end_block + 1):
            if index >= len(table):
                raise AssertionError(
                    f"ensure_swa: logical block {index} >= "
                    f"swa_block_table len {len(table)} (seq {seq.id})"
                )
            if table[index] < 0:
                block_id = self._pop()
                self._alloc(block_id)
                table[index] = block_id

    # ------------------------------- freeing ---------------------------- #

    def free_out_of_window(self, seq: Sequence, seq_len: int | None = None) -> None:
        if not self.enabled or self.window <= 0:
            return
        if seq_len is None:
            seq_len = len(seq)
        free_before = max(0, (seq_len - self.window) // self.block_size)
        free_before = min(free_before, len(seq.swa_block_table))
        for index in range(free_before):
            block_id = seq.swa_block_table[index]
            if block_id < 0:
                continue
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._dealloc(block_id)
            seq.swa_block_table[index] = -1

    def free_after_prefill_chunk(self, seq: Sequence) -> None:
        if self.enabled:
            self.free_out_of_window(seq, seq.num_cached_tokens)

    def materialize_window(self, seq: Sequence, seq_len: int) -> None:
        if not self.enabled or self.window <= 0:
            return
        free_before = max(0, (seq_len - self.window) // self.block_size)
        for index in range(free_before, len(seq.swa_block_table)):
            if seq.swa_block_table[index] < 0:
                block_id = self._pop()
                self._alloc(block_id)
                seq.swa_block_table[index] = block_id

    # --------------------------- hash / release ------------------------- #

    def publish_hash(
        self,
        seq: Sequence,
        index: int,
        block_hash: int,
        token_ids: list[int],
    ) -> None:
        if not self.enabled or index >= len(seq.swa_block_table):
            return
        block_id = seq.swa_block_table[index]
        if block_id < 0:
            return
        block = self.blocks[block_id]
        block.update(block_hash, token_ids)
        self.hash_to_block_id[block_hash] = block.block_id
        if self.sparse_retain and self._is_checkpoint(seq, index):
            self._pin_checkpoint(block_id)

    def release(self, seq: Sequence) -> None:
        if not self.enabled:
            return
        for block_id in reversed(seq.swa_block_table):
            if block_id < 0:
                continue
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._dealloc(block_id)
        seq.swa_block_table.clear()
