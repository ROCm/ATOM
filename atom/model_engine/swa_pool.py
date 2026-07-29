# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import OrderedDict

from atom.kv_cache.pools.pooled_free_list import PooledFreeList
from atom.model_engine.kv_block import Block
from atom.model_engine.sequence import Sequence


class SlidingWindowPool:
    """Physical content-addressed SWA component owned by ``BlockManager``.

    Owns an independent free-list + content-hash map so out-of-window SWA blocks
    can be freed while the compressed blocks persist. Mirrors vLLM's
    `SlidingWindowManager`; ``BlockManager`` holds one instance as ``self.swa``
    and drives its SWA lifecycle in lockstep with the compressed pool. Under the
    CSA-into-SWA fusion the CSA boundary snapshot rides this pool's blocks (no
    separate page pool), so this is the sole prefix-cache sidecar owner.
    `seq.swa_block_table` lives on `Sequence` (shared with attention / PD); this
    component only reads/writes it.

    Self-guarding: when `num_blocks == 0` (non-V4 models) the pool is DISABLED —
    every method is an identity/no-op, so ``BlockManager`` can call it
    unconditionally (no `if swa_enabled` scattered at the call sites). `has_free`
    returns True and `bounded_hit` returns the input length, so admission and
    hit-length are byte-identical to a no-SWA build.

    Hashing note: the chained content hash is computed by BlockManager (shared by
    the compressed and SWA pools). `bounded_hit` / `claim_cached` / `publish_hash`
    receive `h`/`block_hashes` as inputs — this pool never recomputes them, so it
    stays aligned with the compressed prefix.
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
        self.enabled: bool = num_blocks > 0
        self.window: int = window
        self.block_size: int = block_size
        self.max_num_batched_tokens: int = max_num_batched_tokens
        # Full-retention mode (ATOM_SWA_FULL_RETAIN): write + materialize EVERY
        # SWA block of a prefill chunk (not only the trailing window), so the
        # content-addressed cache holds the full history for cross-request replay
        # hits. The live sliding-window free stays on (bounds active refs); the
        # larger pool keeps freed-but-cached blocks resident until reuse.
        self.full_retain: bool = bool(full_retain)
        # Sparse checkpoint-tail retention (ATOM_SWA_RETENTION_INTERVAL, tokens).
        # 0 = dense (retain every written tail, relies on pool size). >0 = keep a
        # SWA tail only once per `retention_interval`-token segment plus at each
        # request's prompt boundary, and PIN those checkpoint blocks (an extra
        # ref so free_out_of_window / eviction skip them) so live-window churn
        # cannot overwrite them. Mirrors vLLM's SlidingWindowManager sparse
        # reachable_block_mask, adapted to ATOM's separate (small, isolated) SWA
        # pool where masking alone is insufficient — the pin is required. LRU-
        # capped at `checkpoint_frac` of the pool so live churn keeps headroom.
        self.retention_blocks: int = (
            retention_interval // block_size
            if (retention_interval > 0 and block_size > 0)
            else 0
        )
        self.sparse_retain: bool = self.full_retain and self.retention_blocks > 0
        self.checkpoint_capacity: int = (
            int(num_blocks * checkpoint_frac) if self.sparse_retain else 0
        )
        # block_id -> None, ordered by pin/access recency (front = LRU).
        self.checkpoint_lru: OrderedDict[int, None] = OrderedDict()
        # Prefix-cache hit gate: a hit only needs the trailing window before the
        # boundary to be SWA-present (SWA is local). `tail_blocks` = contiguous
        # blocks covering win_with_spec = window + mtp_k (spec-decode tail tokens
        # reach back further).
        win_with_spec = window + mtp_k
        self.tail_blocks: int = (
            max(1, (win_with_spec - 1 + block_size - 1) // block_size)
            if window > 0
            else 0
        )
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        # free_block_ids = BACKED-free (hold an arena page); _unbacked_free_ids =
        # ref-0 ids whose page was lent to the compressed pool (reuse re-borrows).
        # Arena is attached later; without it every id is a fixed slot ("backed")
        # and _unbacked stays empty — byte-identical to the pre-arena path.
        self._free_list = PooledFreeList(num_blocks)
        self.free_block_ids = self._free_list.backed_ids
        self.free_block_ids_set = self._free_list.backed_set
        self._unbacked_free_ids = self._free_list.unbacked_ids
        self._unbacked_free_set = self._free_list.unbacked_set
        self.used_block_ids = self._free_list.used_ids
        # Unified-KV chunk arena (ATOM_V4_UNIFIED_KV_ARENA). When attached, an
        # SWA block's physical page is borrowed from the shared arena instead of
        # being its fixed slot id, so idle SWA capacity can be lent to the
        # compressed pool. ``_evict_sibling`` evicts a cold compressed block to
        # return a chunk when the arena is starved (pool-driven lending).
        self.arena = None
        self._evict_sibling = None

    def attach_arena(self, arena, evict_sibling) -> None:
        """Wire the shared arena + a callback that evicts one cold compressed
        block (returning a chunk) so SWA can borrow it under pressure."""
        self.arena = arena
        self._evict_sibling = evict_sibling
        # With the arena on, an SWA id has NO physical page until first _alloc, so
        # every currently-free id starts UNbacked.
        if arena is not None and getattr(arena, "enabled", False):
            self._free_list.move_all_to_unbacked()

    def _arena_alloc_swa(self, block_id: int) -> None:
        """Back an SWA block with an arena page, evicting a cold compressed block
        on starvation and retrying (pool-driven cross-pool lending)."""
        from atom.kv_cache.pools.chunk_arena import ArenaEmpty

        if self.arena is None or self.arena.is_swa_backed(block_id):
            return
        while True:
            try:
                self.arena.alloc_swa(block_id)
                return
            except ArenaEmpty:
                if self._evict_sibling is None or not self._evict_sibling():
                    raise

    def evict_cold_for_arena(self) -> bool:
        """Truly evict the coldest ref-0 SWA block (drop its hash + return its
        arena pages) so the compressed pool can borrow the freed chunk. Returns
        False when no ref-0 block is available."""
        if self.arena is None:
            return False
        # free_block_ids holds only BACKED-free ids → all evictable candidates.
        # Drop the coldest ref-0 block's hash, return its arena page (a whole
        # chunk, since an SWA page == a chunk), and move the id to the UNBACKED
        # free pool so it stays reusable instead of leaking.
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
        """Count of ref-0 BACKED SWA blocks the compressed pool could reclaim
        (each holds a full chunk that lending returns to the arena)."""
        return len(self.free_block_ids_set) if self.arena is not None else 0

    # ----------------------------- primitives ------------------------------ #
    def _pop(self) -> int:
        # Prefer BACKED-free (reuse held page) before UNBACKED-free (re-borrow a
        # page on _alloc). Arena off: _unbacked_free_ids is empty (unchanged).
        return self._free_list.pop(error_message="No free SWA blocks available")

    def _alloc(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self._free_list.mark_used(block_id)
        # Ensure this SWA block has an arena page (no-op when the arena is off or
        # the block is already backed). Borrows from compressed under pressure.
        self._arena_alloc_swa(block_id)
        return block

    def _dealloc(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        assert self._free_list.deallocate(block_id), f"SWA block {block_id} not in use"

    # ------------------------ sparse checkpoint pins ----------------------- #
    def _is_checkpoint(self, seq: Sequence, i: int) -> bool:
        """Whether logical block `i` is a retained checkpoint tail: it sits in the
        trailing `tail_blocks` of a `retention_blocks`-sized segment, OR in the
        trailing `tail_blocks` before the prompt boundary (a proven reuse point).
        Mirrors vLLM SlidingWindowManager.reachable_block_mask (segment tails +
        reachable-boundary tails)."""
        if not self.sparse_retain:
            return True
        need = self.tail_blocks
        rb = self.retention_blocks
        if i % rb >= rb - need:  # last `need` blocks of this segment
            return True
        prompt_blocks = seq.num_prompt_tokens // self.block_size
        if i >= prompt_blocks - need:  # trailing tail before the prompt boundary
            return True
        return False

    def _pin_checkpoint(self, block_id: int) -> None:
        """Pin a checkpoint SWA block: hold an extra ref so free_out_of_window
        never returns it to the free list, keeping its content-addressed tail
        resident for cross-request reuse. LRU-evict the oldest pin when over
        capacity. Idempotent per block_id (a re-pin just refreshes recency)."""
        if block_id in self.checkpoint_lru:
            self.checkpoint_lru.move_to_end(block_id)
            return
        self.blocks[block_id].ref_count += 1  # the pin ref
        self.checkpoint_lru[block_id] = None
        while len(self.checkpoint_lru) > self.checkpoint_capacity:
            old_id, _ = self.checkpoint_lru.popitem(last=False)  # LRU
            blk = self.blocks[old_id]
            blk.ref_count -= 1  # drop the pin ref
            if blk.ref_count == 0:  # no live seq holds it → reclaim
                self._dealloc(old_id)

    # --------------------------- admission / hit --------------------------- #
    def has_free(self, n: int) -> bool:
        """Whether the pool has at least `n` free blocks. Disabled → True (never
        blocks admission). Arena on: also require `n` arena pages backable from
        SWA's own capacity — conservative (does not count chunks reclaimable from
        the compressed pool, which alloc can still borrow), so it never
        over-admits."""
        if not self.enabled:
            return True
        if self.arena is None:
            return len(self.free_block_ids_set) >= n
        # Reusing a BACKED-free id costs no new page; only the shortfall must be
        # backed from SWA's own arena capacity (conservative — ignores chunks
        # reclaimable from the compressed pool, so it never over-admits).
        backed_free = len(self.free_block_ids_set)
        total_free = backed_free + len(self._unbacked_free_set)
        if total_free < n:
            return False
        return backed_free + self.arena.swa_available() >= n

    def admission_blocks(self, seq: Sequence) -> int:
        """Peak concurrent SWA blocks one request holds during (chunked) prefill.
        Window-only prefill (ensure_for_tokens materializes only the trailing
        `window` blocks, not the whole chunk) → peak footprint == the trailing
        window = `tail_blocks` (+1 for the slide boundary), same as a decoding
        seq — NOT the old `window-1 + max_num_batched_tokens` full-chunk span.
        Capped by the prompt's block count. Admission gate instead of full
        `seq.num_blocks` since SWA is filled incrementally + window-freed."""
        if not self.enabled:
            return 0
        if self.full_retain:
            # Full-retain materializes the whole current chunk (ensure_for_tokens
            # free_before=0), so the peak concurrent footprint per prefill step is
            # the chunk's block span (bounded by max_num_batched_tokens), NOT the
            # trailing window. Freed after each chunk, so this is the per-step peak
            # not the whole prompt. Capped by the prompt's block count.
            bs = self.block_size
            chunk_peak = (self.max_num_batched_tokens + bs - 1) // bs + 1
            return min(chunk_peak, seq.num_blocks)
        cap = self.tail_blocks + 1
        return min(cap, seq.num_blocks)

    def bounded_hit(self, seq: Sequence, P: int, block_hashes: list[int]) -> int:
        """Prefix-cache gate (vLLM SlidingWindowManager, simple-hybrid one pass).
        Given the compressed prefix length `P` and each block's content hash,
        return the largest boundary `L <= P` whose trailing window
        `[L - tail_blocks, L)` is fully SWA-present — scanning right-to-left and
        stopping at the first (rightmost) complete window. Blocks before that
        window are out of the sliding window (never read by the resumed forward),
        so their SWA absence does NOT shorten the hit; `claim_cached` marks them
        -1 via BlockManager.allocate.

        Bounding the scan by `P` (only blocks the compressed match also covered)
        guarantees the returned `L` satisfies BOTH compressed[0,L) present and
        SWA[L-window,L) present — the boundary can never land on a block whose
        in-window SWA is missing (#1417).

        Falls through to the length of a contiguous run ending at block 0 (0 if
        block 0 is absent): covers short prompts (P < tail_blocks, whole prefix
        within one window) and vLLM's partial-hit case; the boundary's window then
        spans [0, L) which is present, so it stays safe. Disabled → return P."""
        if not self.enabled:
            return P
        need = self.tail_blocks
        num_contig = 0
        for i in range(P - 1, -1, -1):
            swa_id = self.hash_to_block_id.get(block_hashes[i], -1)
            if swa_id != -1 and self.blocks[swa_id].token_ids == seq.block(i):
                num_contig += 1
                if num_contig >= need:
                    return i + num_contig  # rightmost complete window → boundary
            else:
                num_contig = 0
        return num_contig  # short prompt / partial front run (window spans [0,L))

    # ---------------------------- allocation ------------------------------- #
    def claim_cached(self, seq: Sequence, h: int, token_ids: list[int]):
        """Claim the cached SWA block for hash `h` (caller guarantees it exists,
        via bounded_hit) and append to seq.swa_block_table. Mirrors the
        compressed cached-hit claim. Disabled → no-op."""
        if not self.enabled:
            return
        swa_id = self.hash_to_block_id[h]
        block = self.blocks[swa_id]
        if swa_id in self.used_block_ids:
            block.ref_count += 1
        else:
            assert block.ref_count == 0
            block.ref_count = 1
            # Cache hits land only on BACKED blocks (KV resident); discard from
            # both free sets so the id leaves the free pool cleanly.
            self._free_list.mark_used(swa_id)
        # Cross-request reuse of a pinned checkpoint → refresh its LRU recency.
        if swa_id in self.checkpoint_lru:
            self.checkpoint_lru.move_to_end(swa_id)
        seq.swa_block_table.append(swa_id)

    def alloc_placeholder(self, seq: Sequence):
        """Append a -1 placeholder, keeping swa_block_table the same length as
        block_table (positional alignment). Used for uncached blocks (filled
        later by ensure_for_tokens) and for out-of-window front blocks on a hit.
        Disabled → no-op (swa_block_table stays empty)."""
        if not self.enabled:
            return
        seq.swa_block_table.append(-1)

    def append_new(self, seq: Sequence):
        """Allocate a fresh SWA block for a new decode block and append it (keeps
        lockstep with block_table). Disabled → no-op."""
        if not self.enabled:
            return
        swa_id = self._pop()
        self._alloc(swa_id)
        seq.swa_block_table.append(swa_id)

    def ensure_for_tokens(
        self, seq: Sequence, num_cached_tokens: int, num_new_tokens: int
    ):
        """Fill the SWA blocks for the logical blocks this step's tokens touch.
        allocate() left uncached SWA slots as -1 placeholders (table length ==
        block_table length); here we replace the -1 in the current chunk's logical
        range with real physical blocks, BEFORE the forward writes SWA. In-place
        fill (never append/shorten) keeps swa_block_table positionally aligned
        with block_table — required by the index kernels (absolute logical
        indexing), may_append (lockstep), and PD transfer. Disabled → no-op."""
        if not self.enabled or num_new_tokens <= 0:
            return
        bs = self.block_size
        seq_len = num_cached_tokens + num_new_tokens
        start_blk = num_cached_tokens // bs
        end_blk = (seq_len - 1) // bs
        # OPT (window-only alloc): only materialize the trailing-window blocks
        # (blocks the SWA window will actually read + be written by the
        # window-only swa_write). Earlier blocks stay -1 (never written/read),
        # matching free_out_of_window's sentinel. Cuts prefill SWA allocation
        # from O(chunk_len/bs) to O(window/bs) — pairs with the window-only
        # swa_write in deepseek_v4.py. free_before mirrors free_out_of_window.
        # Full-retain: materialize every block of this chunk (free_before=0) so
        # the full-chunk swa_write below has a valid physical dst for every token,
        # and every block gets published for cross-request reuse. Default:
        # window-only (only the trailing-window blocks the SWA read touches).
        free_before = 0 if self.full_retain else max(0, (seq_len - self.window) // bs)
        start_blk = max(start_blk, free_before)
        table = seq.swa_block_table
        for i in range(start_blk, end_blk + 1):
            if i >= len(table):
                # allocate() sizes the table to seq.num_blocks; a chunk should
                # never index past it. Guard against desync loudly.
                raise AssertionError(
                    f"ensure_swa: logical block {i} >= swa_block_table len "
                    f"{len(table)} (seq {seq.id}); table not full-length?"
                )
            if table[i] < 0:  # -1 placeholder → materialize a real SWA block
                swa_id = self._pop()
                self._alloc(swa_id)
                table[i] = swa_id

    # ----------------------------- freeing --------------------------------- #
    def free_out_of_window(self, seq: Sequence, seq_len: int | None = None):
        """Release SWA blocks that have fallen fully behind the sliding window —
        they're never read again by this request, and freeing them bounds live
        SWA memory to ~window per request.

        Block ``i`` covers tokens ``[i*bs, (i+1)*bs)``; the latest query (pos
        ``seq_len-1``) attends down to ``seq_len-window``, so block ``i`` is fully
        out of window once ``(i+1)*bs <= seq_len - window``. Freed blocks keep
        their hash + KV until their pool slot is actually reused (lazy eviction),
        so a cross-request hit can still reuse a freed-but-not-overwritten SWA
        block.

        ``seq_len`` is the number of tokens whose KV has been COMPUTED so far.
        Decode passes None → ``len(seq)`` (whole sequence). Chunked prefill MUST
        pass ``seq.num_cached_tokens`` (post-increment): using ``len(seq)`` (the
        full prompt length) mid-prefill would free SWA for tokens later chunks
        have not written yet. Freeing only sets ``-1``; it never shortens the
        table. Disabled → no-op."""
        if not self.enabled or self.window <= 0:
            return
        if seq_len is None:
            seq_len = len(seq)
        free_before = max(0, (seq_len - self.window) // self.block_size)
        free_before = min(free_before, len(seq.swa_block_table))
        for i in range(free_before):
            swa_id = seq.swa_block_table[i]
            if swa_id < 0:
                continue  # already window-freed
            block = self.blocks[swa_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._dealloc(swa_id)
            seq.swa_block_table[i] = -1  # sentinel: out of window

    def free_after_prefill_chunk(self, seq: Sequence):
        """Chunk-boundary window-freeing, called from scheduler.postprocess AFTER
        ``seq.num_cached_tokens += chunk``. Uses the computed-so-far length so
        out-of-window SWA blocks are reclaimed during prefill (not only at the
        first decode step), bounding peak SWA to ~window per request. Disabled →
        no-op."""
        if not self.enabled:
            return
        self.free_out_of_window(seq, seq.num_cached_tokens)

    def materialize_window(self, seq: Sequence, seq_len: int):
        """PD consumer path: the decode instance receives KV via RDMA and never
        runs a prefill forward, so `ensure_for_tokens` is never called and its
        first `may_append` is skipped. Materialize exactly the trailing-window SWA
        blocks — the same logical positions the producer keeps live after
        `free_out_of_window` (both use `free_before = (seq_len - window)//bs`) —
        so the producer's RDMA write has real dst slots at matching logical
        indices. Blocks before the window stay -1, mirroring the producer's freed
        prefix (the consumer never reads them). Disabled → no-op."""
        if not self.enabled or self.window <= 0:
            return
        bs = self.block_size
        free_before = max(0, (seq_len - self.window) // bs)
        for i in range(free_before, len(seq.swa_block_table)):
            if seq.swa_block_table[i] < 0:
                swa_id = self._pop()
                self._alloc(swa_id)
                seq.swa_block_table[i] = swa_id

    # ------------------------- hashing / release --------------------------- #
    def publish_hash(self, seq: Sequence, i: int, h: int, token_ids: list[int]):
        """Publish the SWA block at logical index `i` under content hash `h`, so
        cross-request hits can reuse its sliding-window KV. Skips -1 slots
        (window-freed / not-yet-materialized): a block finalized this step is
        in-window and was filled by ensure_for_tokens, so this normally holds a
        real phys; the >= 0 guard prevents a silent blocks[-1] alias if a block
        fell out of window in the same step. Disabled → no-op."""
        if not self.enabled or i >= len(seq.swa_block_table):
            return
        swa_id = seq.swa_block_table[i]
        if swa_id >= 0:
            block = self.blocks[swa_id]
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
            # Sparse retention: pin this tail iff it is a checkpoint (segment or
            # prompt-boundary tail), so live-window free/churn cannot overwrite it
            # before a branch reuses it. Non-checkpoints stay unpinned and are
            # reclaimed normally.
            if self.sparse_retain and self._is_checkpoint(seq, i):
                self._pin_checkpoint(swa_id)

    def release(self, seq: Sequence):
        """Release all of seq's SWA blocks (skipping -1 window-freed slots) and
        clear its swa_block_table. Disabled → no-op."""
        if not self.enabled:
            return
        for swa_id in reversed(seq.swa_block_table):
            if swa_id < 0:
                continue  # window-freed slot
            block = self.blocks[swa_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._dealloc(swa_id)
        seq.swa_block_table.clear()
