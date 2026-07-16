# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque

from atom.model_engine.kv_block import Block
from atom.model_engine.sequence import Sequence


class SlidingWindowPool:
    """Content-addressed sliding-window KV block pool (DeepSeek-V4 SWA).

    Owns an independent free-list + content-hash map so out-of-window SWA blocks
    can be freed while the compressed blocks persist. Mirrors vLLM's
    `SlidingWindowManager`; `BlockManager` holds one instance (`self.swa`) and
    delegates all SWA lifecycle here, driving it in lockstep with the compressed
    pool. `seq.swa_block_table` lives on `Sequence` (shared with attention / PD);
    this pool only reads/writes it.

    Self-guarding: when `num_blocks == 0` (non-V4 models) the pool is DISABLED —
    every method is an identity/no-op, so `BlockManager` can call it
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
    ):
        self.enabled: bool = num_blocks > 0
        self.window: int = window
        self.block_size: int = block_size
        self.max_num_batched_tokens: int = max_num_batched_tokens
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
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.free_block_ids_set: set[int] = set(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # ----------------------------- primitives ------------------------------ #
    def _pop(self) -> int:
        while self.free_block_ids:
            block_id = self.free_block_ids.popleft()
            if block_id in self.free_block_ids_set:
                self.free_block_ids_set.discard(block_id)
                return block_id
        raise AssertionError("No free SWA blocks available")

    def _alloc(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.free_block_ids_set.discard(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _dealloc(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        self.free_block_ids_set.add(block_id)

    # --------------------------- admission / hit --------------------------- #
    def has_free(self, n: int) -> bool:
        """Whether the pool has at least `n` free blocks. Disabled → True (never
        blocks admission)."""
        if not self.enabled:
            return True
        return len(self.free_block_ids_set) >= n

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
            self.free_block_ids_set.discard(swa_id)
            self.used_block_ids.add(swa_id)
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
        free_before = max(0, (seq_len - self.window) // bs)
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
