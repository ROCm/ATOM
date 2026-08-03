# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from math import inf

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.sequence import Sequence

# Name of the sub-pool sizing class this pool is fed from. Owned here, not in
# the sizing layer: a backend that wants a sliding-window pool imports this to
# declare its spec, and this pool reads the count back under the same key.
SWA_POOL_CLASS = "swa"


class SlidingWindowPool:
    """Content-addressed sliding-window KV block pool (DeepSeek-V4 SWA).

    A `StateCache` class (see `state_cache.py`), but only the gate half of the
    protocol: it scales with in-flight requests and it answers how far back the
    nearest resumable boundary is. It never *keeps* a checkpoint, because only
    the trailing window is ever materialized — every earlier block is a `-1`
    sentinel — so there is nothing at an older boundary to hold on to.
    `successor_room` is `inf`, which is how the ladder is told to skip it.

    Holds its own `BlockPool` — the same bookkeeping the compressed KV blocks
    get, over a separate index space — so out-of-window SWA blocks can be freed
    while the compressed blocks persist. What this class adds on top is the
    window policy: which blocks are worth materializing, and when they fall out
    of reach. Mirrors vLLM's `SlidingWindowManager`; `BlockManager` holds one
    instance (`self.swa`) and delegates all SWA lifecycle here, driving it in
    lockstep with the compressed pool. `seq.swa_block_table` lives on `Sequence`
    (shared with attention / PD); this pool only reads/writes it.

    Self-guarding: when `num_blocks == 0` (non-V4 models) the pool is DISABLED —
    every method is an identity/no-op, so `BlockManager` can call it
    unconditionally (no `if swa_enabled` scattered at the call sites). `has_free`
    returns True and `resumable_hit` returns the input length, so admission and
    hit-length are byte-identical to a no-SWA build.

    Hashing note: the chained content hash is computed by BlockManager (shared by
    the compressed and SWA pools). `resumable_hit` / `claim_cached` /
    `publish_hash` receive `h`/`block_hashes` as inputs — this pool never
    recomputes them, so it stays aligned with the compressed prefix.
    """

    def __init__(
        self,
        num_blocks: int,
        window: int,
        block_size: int,
        mtp_k: int,
    ):
        self.enabled: bool = num_blocks > 0
        self.window: int = window
        self.block_size: int = block_size
        # This class cannot keep a checkpoint at all; see the class docstring.
        self.successor_room: float = inf
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
        self.pool = BlockPool(num_blocks)

    def _fresh_block(self) -> int:
        """Take a block for content this pool is about to write."""
        block_id = self.pool.pop()
        self.pool.allocate(block_id)
        return block_id

    # ------------------------ state-cache protocol ------------------------- #
    def applies(self, seq: Sequence) -> bool:
        """Whether this class gates or checkpoints anything for `seq`.

        The window is a property of the architecture, so it covers every seq the
        pool is enabled for.
        """
        return self.enabled

    def checkpoint(self, seq: Sequence, boundary_blocks: int, h: int) -> None:
        """Unreachable: `successor_room` is `inf`, so no rung ever selects this
        class. Present so the `inf` contract fails loudly rather than silently
        keeping nothing if the ladder ever stops honouring it."""
        raise AssertionError("the sliding window keeps no checkpoints")

    # --------------------------- admission / hit --------------------------- #
    def has_free(self, n: int) -> bool:
        """Whether the pool has at least `n` free blocks. Disabled → True (never
        blocks admission)."""
        if not self.enabled:
            return True
        return self.pool.has_free(n)

    def admission_blocks(self, seq: Sequence) -> int:
        """Peak concurrent SWA blocks one request holds during (chunked) prefill.
        Window-only prefill (ensure_for_tokens materializes only the trailing
        `window` blocks, not the whole chunk) → peak footprint == the trailing
        window = `tail_blocks` (+1 for the slide boundary), same as a decoding
        seq. Capped by the prompt's block count. Admission gate instead of full
        `seq.num_blocks` since SWA is filled incrementally + window-freed."""
        if not self.enabled:
            return 0
        cap = self.tail_blocks + 1
        return min(cap, seq.num_blocks)

    def resumable_hit(
        self,
        seq: Sequence,
        P: int,
        block_hashes: list[int],
        assume_checkpointed: bool = False,
    ) -> int:
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
        spans [0, L) which is present, so it stays safe. Disabled → return P.

        `assume_checkpointed` is inert here: this class keeps no checkpoints, so
        a dense ladder is still an empty one and what it declines it declines
        either way. That is the whole point of asking every class — a boundary
        this pool cannot serve is not worth checkpointing the ring at."""
        if not self.enabled:
            return P
        need = self.tail_blocks
        num_contig = 0
        for i in range(P - 1, -1, -1):
            swa_id = self.pool.lookup(block_hashes[i])
            if swa_id != -1 and self.pool.block(swa_id).token_ids == seq.block(i):
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
        swa_id = self.pool.lookup(h)
        self.pool.claim(swa_id)
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
        seq.swa_block_table.append(self._fresh_block())

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
                table[i] = self._fresh_block()

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
            self.pool.free(swa_id)
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
                seq.swa_block_table[i] = self._fresh_block()

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
            self.pool.publish(swa_id, h, token_ids)

    def release(self, seq: Sequence):
        """Release all of seq's SWA blocks (skipping -1 window-freed slots) and
        clear its swa_block_table. Disabled → no-op."""
        if not self.enabled:
            return
        for swa_id in reversed(seq.swa_block_table):
            if swa_id < 0:
                continue  # window-freed slot
            self.pool.free(swa_id)
        seq.swa_block_table.clear()
