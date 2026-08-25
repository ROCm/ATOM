# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque

from atom.model_engine.kv_block import Block
from atom.model_engine.sequence import Sequence


class StateCachePool:
    """Content-addressed pool of recurrent-state checkpoints (GDN / KDA).

    Linear-attention hybrids carry a per-request recurrent state (`conv_state` +
    `ssm_state`) that is mutated in place in a working slot, not stored per
    token in the paged KV pool. A paged prefix-cache hit is therefore unsafe on
    its own: the resumed sequence would skip forwarding tokens whose conv/ssm
    state its working slot never computed. This pool makes such hits safe by
    snapshotting the committed state every `state_cache_block_size` (M) tokens
    and keying each snapshot under the SAME chained content hash the paged pool
    uses for the last block of that span. A prefix-cache hit is then bounded to
    the largest M-boundary whose snapshot is still resident — which is exactly
    "prefix caching works at max(M, page_size) granularity".

    A checkpoint is coarse on purpose: it is a whole-layer snapshot whose size
    does not grow with the tokens that produced it, so checkpointing every
    `block_size` tokens would cost orders of magnitude more HBM than the paged
    KV it parallels. `ModelRunner.get_num_blocks()` sizes this pool in whole
    spans (1 checkpoint + M/block_size paged blocks) so neither pool outlives
    the other.

    How a checkpoint gets written without touching the kernels: the scheduler
    asks `trim_chunk_to_boundary` to trim each prefill chunk back to the last M
    boundary it already spans. A recurrent forward leaves exactly one state
    behind — the state after its last token — so a trimmed chunk makes that
    state the checkpoint, and saving it is a device-to-device copy out of the
    working slot. Trimming DOWN (not cutting at the next boundary ahead) is
    what keeps chunks full width; see that method. One checkpoint per chunk is
    the cost of not reaching into the scan for its intermediate states, which
    would mean editing `@support_torch_compile` model files. See
    `GDNAttentionMetadataBuilder.save_state_checkpoints` / `restore_state`.

    Lifecycle (publish-and-release). A checkpoint slot is reserved BEFORE the
    forward (`reserve_write`, so the forward has a destination), published
    under its content hash AFTER the forward (`publish`, once the state is
    actually computed), and its reference dropped immediately. Published
    checkpoints keep their hash and contents until their slot is popped for
    reuse — the same lazy eviction the paged pool uses — so they stay hittable
    without any live request holding them. This bounds a request's checkpoint
    footprint to O(1) rather than O(len / M), which matters when one checkpoint
    is several MB. The one reference a request does hold is the checkpoint it
    RESTORES from, released once the first forward has copied it into the
    request's working slot.

    Self-guarding: when the pool is disabled (`num_blocks == 0` or
    `state_cache_block_size == 0`) every method is an identity/no-op and
    `bounded_hit` returns its input, so `BlockManager` delegates
    unconditionally and non-recurrent models are byte-identical to a build
    without this pool. Note that "disabled" alone does NOT make a prefix-cache
    hit safe for a recurrent model — `EngineCore` turns prefix caching off
    engine-wide for those; this pool is what will let that gate be lifted.

    Hashing note: the chained content hash is computed by `BlockManager` and
    shared with the paged (and SWA) pools. `bounded_hit` / `claim_restore` /
    `publish` receive `h` as an input and never recompute it, so a checkpoint
    can never drift from the paged prefix it summarizes.

    Decode is deliberately not checkpointed: decode-generated blocks are never
    hashed either (their tokens are not final until sampling/verification), so
    there is no content hash to key a checkpoint under. Checkpoints are written
    during (chunked) prefill only.
    """

    def __init__(
        self,
        num_blocks: int,
        state_cache_block_size: int,
        block_size: int,
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = True,
    ):
        # A checkpoint exists only to make a prefix-cache hit resumable, so
        # with prefix caching off the pool has no consumer: reserving and
        # freeing slots every step would be pure churn (and `publish` would
        # never run, since `BlockManager.hash_blocks` returns early). Treat
        # that as disabled.
        #
        # Chunked prefill is required too, because `trim_chunk_to_boundary` works by
        # cutting a prefill short at an M boundary. With chunking off, the
        # scheduler admits a prompt expecting to finish it in one step, so a
        # shortened chunk would leave a partial prefill that only Phase-1
        # resume can complete — a behavior change for a deployment that opted
        # out of exactly that. Checkpointing is an optimization, so it yields.
        self.enabled: bool = (
            num_blocks > 0
            and state_cache_block_size > 0
            and enable_prefix_caching
            and enable_chunked_prefill
        )
        self.state_cache_block_size: int = state_cache_block_size
        self.block_size: int = block_size
        # Paged blocks spanned by one checkpoint. `state_cache_block_size` is
        # validated to be a multiple of `block_size` in Config.__post_init__,
        # so this division is exact and a checkpoint boundary is always a paged
        # block boundary — the property that lets ONE chained hash key both.
        self.blocks_per_ckpt: int = (
            state_cache_block_size // block_size if self.enabled else 0
        )
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.free_block_ids_set: set[int] = set(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # ----------------------------- primitives ------------------------------ #
    def _pop(self) -> int:
        """Pop a free checkpoint slot, or -1 when none is free.

        Unlike the paged pools this does NOT raise on exhaustion: a checkpoint
        is a cache entry, never a correctness requirement, so a request that
        cannot reserve one simply does not checkpoint that span and pays a
        future cache miss. Raising here would fail a request over a missing
        optimization.
        """
        while self.free_block_ids:
            block_id = self.free_block_ids.popleft()
            if block_id in self.free_block_ids_set:
                self.free_block_ids_set.discard(block_id)
                return block_id
        return -1

    def _alloc(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # Lazy eviction: a published checkpoint sits in the free queue with its
        # hash intact until its slot is actually reused, so THIS is the true
        # eviction point. Drop the stale mapping before the contents change,
        # otherwise a later lookup would hand out a slot holding another
        # sequence's state.
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

    def _release(self, block_id: int):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._dealloc(block_id)

    # --------------------------- admission / hit --------------------------- #
    def bounded_hit(self, seq: Sequence, P: int, block_hashes: list[int]) -> int:
        """Truncate a paged prefix-cache hit to the last checkpointed boundary.

        `P` is the paged hit length in blocks and `block_hashes[i]` the chained
        content hash of block `i` (both from `BlockManager.can_allocate`).
        Returns the largest `L <= P` that is a multiple of `blocks_per_ckpt`
        and whose checkpoint — keyed by `block_hashes[L - 1]`, the hash of the
        span's LAST block — is still resident.

        Scanning down from `P` rather than up from 0 finds the longest usable
        prefix directly. Anything not on an M boundary is unusable no matter
        how much paged KV is cached: without a snapshot at that exact token
        count there is no recurrent state to resume from, and reusing the paged
        KV alone would silently drop the conv/ssm history for the skipped
        tokens. 0 (recompute everything) is always a valid answer.

        Disabled → return `P` unchanged.
        """
        if not self.enabled:
            return P
        step = self.blocks_per_ckpt
        for L in range((P // step) * step, 0, -step):
            h = block_hashes[L - 1]
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id != -1 and self.blocks[block_id].token_ids == seq.block(L - 1):
                return L
        return 0

    # ---------------------------- restore side ----------------------------- #
    def claim_restore(self, seq: Sequence, h: int):
        """Take a reference on the checkpoint the sequence will resume from and
        record it on `seq.state_restore_slot`.

        Caller guarantees the hash is resident (it came from `bounded_hit` in
        the same scheduler step, and nothing between the two can evict it: only
        `_alloc` evicts, and `_alloc` runs from `reserve_write`, later in the
        step). The reference keeps the slot alive until the first forward has
        copied its contents into the request's working slot; `release_restore`
        drops it afterwards. Disabled → no-op.
        """
        if not self.enabled:
            return
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            # Hit on a slot sitting in the free queue: claim it WITHOUT _alloc,
            # whose reset() would evict the very hash we just matched.
            assert block.ref_count == 0
            block.ref_count = 1
            self.free_block_ids_set.discard(block_id)
            self.used_block_ids.add(block_id)
        seq.state_restore_slot = block_id

    def release_restore(self, seq: Sequence):
        """Drop the restore reference once the forward has consumed it.

        Called after the sequence's first prefill chunk. Holding it for the
        request's whole life would pin up to one checkpoint per running
        request, which at several MB each is real memory for a slot that is
        read exactly once. Idempotent. Disabled → no-op.
        """
        if not self.enabled or seq.state_restore_slot < 0:
            return
        self._release(seq.state_restore_slot)
        seq.state_restore_slot = -1

    # ----------------------------- write side ------------------------------ #
    def trim_chunk_to_boundary(self, _num_cached_tokens: int, chunk: int) -> int:
        """Return the scheduler's chunk unchanged.

        Checkpoints are emitted in-kernel at every M boundary crossed by a
        forward, so checkpointing must not alter prefill chunking. The retained
        method keeps older callers source-compatible while making the new
        no-recompute/latest-state behavior explicit.
        """
        return chunk

    def reserve_write(self, seq: Sequence, num_cached_tokens: int, chunk: int):
        """Reserve slots for every M boundary crossed by this forward.

        ``state_ckpt_writes`` is keyed by the last paged block summarized by a
        checkpoint. The kernels receive the corresponding slot IDs as a block
        table and scatter each intermediate recurrent state directly into the
        pool. Exhaustion skips only the affected checkpoint; forwarding remains
        correct and the request's mutable latest state is still updated.
        """
        if not self.enabled or chunk <= 0:
            return
        end = num_cached_tokens + chunk
        boundary = (
            num_cached_tokens // self.state_cache_block_size + 1
        ) * self.state_cache_block_size
        while boundary <= end:
            block_id = self._pop()
            if block_id < 0:
                break
            self._alloc(block_id)
            seq.state_ckpt_writes[boundary // self.block_size - 1] = block_id
            boundary += self.state_cache_block_size

    def publish(self, seq: Sequence, i: int, h: int, token_ids: list[int]):
        """Publish the checkpoint written at paged block `i` under hash `h`.

        Called from `BlockManager.hash_blocks` for each block the finished
        forward finalized, so a checkpoint becomes visible only once its state
        has actually been computed — the same deferral the paged pool uses. The
        reference is dropped right after publishing (see the class docstring):
        the contents survive in the slot until it is popped for reuse, so the
        checkpoint stays hittable without pinning memory. No-op for blocks that
        are not a span's last block, or whose reservation was skipped.
        Disabled → no-op.
        """
        if not self.enabled:
            return
        block_id = seq.state_ckpt_writes.pop(i, -1)
        if block_id < 0:
            return
        block = self.blocks[block_id]
        # Store the span's LAST block tokens, not all M of them: `bounded_hit`
        # is only reached after BlockManager has already verified every block
        # of the paged prefix token-by-token, so this final block is the one
        # comparison that is not already covered.
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block_id
        self._release(block_id)

    def end_step(self, seq: Sequence):
        """Drop reservations the step did not publish.

        `reserve_write` reserves exactly the boundary the scheduled chunk ends
        on, so normally nothing is left. This releases the slot anyway if a
        chunk is cut short or a forward is abandoned, so a reservation can
        never leak a slot out of the pool for good. Disabled → no-op.
        """
        if not self.enabled or not seq.state_ckpt_writes:
            return
        for block_id in seq.state_ckpt_writes.values():
            self._release(block_id)
        seq.state_ckpt_writes.clear()

    # ------------------------------ teardown ------------------------------- #
    def release(self, seq: Sequence):
        """Release everything `seq` still holds. Disabled → no-op."""
        if not self.enabled:
            return
        self.end_step(seq)
        self.release_restore(seq)

    def clear_cache(self) -> None:
        """Drop every checkpoint entry, mirroring `BlockManager.clear_cache`.

        Slots held by live sequences keep working (their holders reference them
        by id); they just stop being reachable by hash.
        """
        if not self.enabled:
            return
        self.hash_to_block_id.clear()
        for block in self.blocks:
            if block.ref_count == 0:
                block.hash = -1
                block.token_ids = []
