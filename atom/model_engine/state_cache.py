# SPDX-License-Identifier: MIT
"""Content-addressed cache of recurrent (SSM/conv) state checkpoints.

Background
----------
GDN-style linear attention (Qwen3-Next / Qwen3.5 / Kimi KDA) keeps its
recurrent state in a per-request slot outside the paged KV pool. That state is
a fold over EVERY token, so it cannot be reconstructed from paged KV — which is
why a plain KV prefix hit corrupts these models (see
``BlockManager.can_allocate``): the recurrence would enter the uncached suffix
having never seen the prefix.

This pool supplies the missing piece. A checkpoint taken at a token position
``P`` lets a later request that shares the first ``P`` tokens *resume* the
recurrence there instead of replaying from zero, which in turn makes it safe to
take the paged-KV hit up to ``P``.

Design
------
* **Positions are grid-aligned.** ``P`` must be a multiple of ``granularity``
  (default 64 = ``lcm(kv_cache_block_size, FLA chunk size)``). Measured on
  ``fla`` 0.5.2: split-and-replay via ``initial_state`` is *bit-exact* at
  multiples of 64, and merely approximate off-grid. 64 also divides the KV
  block size, so every checkpoint position is a KV block boundary and
  "KV hit >= P" is expressible.

* **Keyed by content, not position.** Position 5000 in one conversation is
  unrelated to position 5000 in another. The key is the chained block hash at
  ``P`` (same chaining as the KV block pool), so a divergent token breaks the
  chain exactly as it does for KV.

* **``pin_count`` is not a refcount.** A checkpoint is a *read-once copy
  source*: it is copied into a request's runtime slot and then never touched
  again, so there is no request-lifetime holder. The pin only covers an
  in-flight DMA. It is a counter rather than a flag because several requests in
  one step may copy from the same source concurrently.

* **Eviction is by hit count, recency as tiebreak.** Every request writes
  checkpoints, most of them inside content nobody else will ever share. Those
  are always the *most recently* written, so pure LRU keeps the junk and evicts
  the shared branch points the cache exists for. Hit count separates the two
  immediately.

This module is pure CPU bookkeeping — it owns slot *indices*, never tensors.
The actual device copies are issued by the runner. Keeping the policy here (in
the single, rankless scheduler process) is what makes eviction decisions
identical across TP ranks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("atom")


@dataclass
class StateCacheEntry:
    """One checkpoint: the recurrent state of a prefix, in a pool slot."""

    key: int
    """Chained content hash of tokens ``[0, position)``."""

    position: int
    """Token offset; always a multiple of the pool's ``granularity``."""

    slot: int
    """Index into the shared state tensor's cascade region."""

    token_ids: tuple[int, ...] = ()
    """Trailing block's tokens, for the hash-collision guard."""

    pin_count: int = 0
    """Outstanding DMAs touching this slot (see module docstring)."""

    hit_count: int = 0
    """Times this checkpoint has been matched. Primary eviction key."""

    last_access: int = 0
    """Monotonic tick, tiebreak for eviction."""

    published: bool = False
    """False while the write DMA is still in flight; lookups skip these."""


@dataclass
class StateCacheStats:
    lookups: int = 0
    hits: int = 0
    writes: int = 0
    skipped_full: int = 0
    evictions: int = 0
    saved_tokens: int = 0
    """Tokens skipped thanks to a hit (the benefit)."""

    fork_checkpoints: int = 0
    """Checkpoints placed at an observed fork rather than at a prompt end."""

    forks_off_grid: int = 0
    """Divergences seen off the checkpoint grid, so uncreditable.

    A high ratio against ``fork_checkpoints`` means the grid is too coarse for
    this traffic's branch points.
    """

    def summary(self) -> str:
        """One-line human-readable form for periodic logging."""
        rate = (self.hits / self.lookups) if self.lookups else 0.0
        return (
            f"hit_rate={rate:.1%} ({self.hits}/{self.lookups}) "
            f"writes={self.writes} evict={self.evictions} "
            f"skipped_full={self.skipped_full} "
            f"saved_tok={self.saved_tokens} "
            f"fork_ckpts={self.fork_checkpoints} "
            f"forks_off_grid={self.forks_off_grid}"
        )

    def as_dict(self) -> dict:
        hit_rate = (self.hits / self.lookups) if self.lookups else 0.0
        return {
            "state_cache_lookups": self.lookups,
            "state_cache_hits": self.hits,
            "state_cache_hit_rate": round(hit_rate, 4),
            "state_cache_writes": self.writes,
            "state_cache_skipped_full": self.skipped_full,
            "state_cache_evictions": self.evictions,
            "state_cache_saved_tokens": self.saved_tokens,
            "state_cache_fork_checkpoints": self.fork_checkpoints,
            "state_cache_forks_off_grid": self.forks_off_grid,
        }


class StateCachePool:
    """Slot allocator + content index for recurrent-state checkpoints.

    Slots are indices into the cascade region of the shared per-request state
    tensor. The runtime region (one live slot per concurrent request) is owned
    by ``BlockManager.free_per_req_cache_groups`` and is not touched here.

    Deliberately four members:

    * ``_free``      — slot indices nobody holds.
    * ``_allocated`` — the slots that ARE held, keyed by their content hash.
      Together with ``_free`` this partitions the pool. Keyed by hash because
      that is how a checkpoint is found; everything else per-checkpoint (hit
      count, recency, pin count, position) lives on the entry, so eviction
      reads one dict and needs no side table.
    * ``_pending``   — entries reserved but not yet written, keyed by seq id.
      Not an index over slots: it is the reservation lifecycle, and without it
      a reservation could not be matched back to the sequence that owns it.
    * ``_tick``      — monotonic counter, the recency source.

    A slot -> entry reverse index was removed: its only readers were
    ``acquire_load`` / ``release_load``, which now carry the entry on the
    sequence (``seq.state_load_entry``) instead of re-deriving it. A fork
    table was removed too; see ``credit_demand``.
    """

    # Counter ceiling: keeps a once-hot checkpoint from becoming immortal.
    MAX_HIT_COUNT = 32

    def __init__(
        self,
        num_slots: int,
        granularity: int = 64,
        block_size: int = 16,
    ):
        assert granularity > 0, "granularity must be positive"
        self.num_slots = max(0, int(num_slots))
        self.granularity = int(granularity)
        # Checkpoint positions must be BOTH a grid multiple and a KV block
        # boundary, so the pool needs the block size to convert between them.
        # This pool never hashes. BlockManager.can_allocate owns block hashing
        # and hands the chain over on the sequence; a checkpoint key is just
        # that chain sampled on the grid. Sharing the KV pool's hashes is what
        # makes a state checkpoint findable from a KV match.
        self.block_size = int(block_size)
        self.blocks_per_ckpt = (
            self.granularity // self.block_size if self.block_size else 0
        )
        self._free: list[int] = list(range(self.num_slots))
        self._allocated: dict[int, StateCacheEntry] = {}
        self._tick = 0
        self.stats = StateCacheStats()
        # Checkpoints reserved but not yet published (the forward has not
        # reached their position). Keyed by seq id.
        self._pending: dict[int, list[StateCacheEntry]] = {}

    # ── queries ────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self.num_slots > 0

    def __len__(self) -> int:
        return len(self._allocated)

    @property
    def num_free(self) -> int:
        return len(self._free)

    def floor_to_grid(self, n: int) -> int:
        return (n // self.granularity) * self.granularity

    def log_summary(self) -> None:
        """Emit the running counters. Called on the scheduler's log interval.

        Without this the stats are write-only: every counter below is
        incremented on the hot path but nothing reads them, so a cache that
        silently stops hitting looks identical to one that is working.
        """
        logger.info(
            "SSM state cache: %s | entries=%d/%d",
            self.stats.summary(),
            len(self._allocated),
            self.num_slots,
        )

    def lookup(
        self, key: int, token_ids: tuple[int, ...] | None = None
    ) -> StateCacheEntry | None:
        """Return the published entry for ``key``, or None.

        ``token_ids`` (the trailing block's tokens) is compared when supplied,
        mirroring the KV pool's collision guard — resuming a recurrence from a
        foreign sequence would be silently wrong, not merely a miss.
        """
        self.stats.lookups += 1
        entry = self._allocated.get(key)
        if entry is None or not entry.published:
            return None
        if token_ids is not None and entry.token_ids != token_ids:
            return None
        self._tick += 1
        entry.last_access = self._tick
        entry.hit_count = min(entry.hit_count + 1, self.MAX_HIT_COUNT)
        self.stats.hits += 1
        return entry

    # ── fork observation ───────────────────────────────────────────────────

    def credit_demand(self, key: int, position: int = -1) -> None:
        """Note that a request wanted a checkpoint at ``key``, and got one.

        Called when a sequence's hash chain breaks: `key` is the chained block
        hash of the last MATCHING block, the point where two sequences stopped
        agreeing. If an entry already lives there, this IS demand for it —
        another request wanted a checkpoint at exactly that position. Credit
        it, or an entry that many requests keep re-deriving looks as cold to
        eviction as one nobody wants.

        There is deliberately no table of divergence points. One was tried:
        `_forks`, queried by a `_fork_pos` search when a sequence had no
        divergence of its own. It never fired — `can_allocate` records
        `fork_hit_blocks` on *every* sequence whose chain breaks, and a
        request arriving at a known fork with a different tail breaks there
        too, so it always took the direct handoff. Measured on agent-loop
        traffic (one system prompt, 30 divergent tails): the search ran once
        and hit zero times; the handoff served all 29.

        ``position`` must be grid-aligned when given — a checkpoint can only
        exist where the chunk kernel materializes state. Pass -1 to skip the
        check.
        """
        if not self.enabled or key == -1:
            return
        if position >= 0 and position % self.granularity != 0:
            # A caller skipped the floor. Count it rather than credit a
            # position that can never hold a checkpoint: a high count here
            # means the grid is too coarse for this traffic's branch points.
            self.stats.forks_off_grid += 1
            return
        entry = self._allocated.get(key)
        if entry is None:
            return
        self._tick += 1
        entry.last_access = self._tick
        entry.hit_count = min(entry.hit_count + 1, self.MAX_HIT_COUNT)

    # ── writes ─────────────────────────────────────────────────────────────

    def try_reserve(
        self, key: int, position: int, token_ids: tuple[int, ...] = ()
    ) -> StateCacheEntry | None:
        """Reserve a slot for a checkpoint at ``position``.

        Returns None when the pool is exhausted — checkpointing is an
        optimization, so a failure must degrade to "no checkpoint", never block
        admission or fail a request.
        """
        if not self.enabled:
            return None
        assert position % self.granularity == 0, (
            f"checkpoint position {position} is not a multiple of "
            f"granularity {self.granularity}"
        )
        existing = self._allocated.get(key)
        if existing is not None:
            # Already cached (or in flight). Nothing to reserve, but this IS
            # demand for that position: another request wanted a checkpoint
            # exactly there. Count it, or an entry that many requests keep
            # re-deriving looks as cold as one nobody wants and gets evicted.
            self._tick += 1
            existing.last_access = self._tick
            existing.hit_count = min(existing.hit_count + 1, self.MAX_HIT_COUNT)
            return None
        slot = self._alloc_slot()
        if slot is None:
            self.stats.skipped_full += 1
            return None
        self._tick += 1
        entry = StateCacheEntry(
            key=key,
            position=position,
            slot=slot,
            token_ids=token_ids,
            last_access=self._tick,
        )
        self._allocated[key] = entry
        return entry

    def publish(self, entry: StateCacheEntry) -> None:
        """Make an entry visible to ``lookup``.

        Called only once the write DMA has retired; publishing earlier would
        let another request match bytes still in flight.
        """
        if entry.key in self._allocated and self._allocated[entry.key] is entry:
            entry.published = True
            self.stats.writes += 1

    def cancel(self, entry: StateCacheEntry) -> None:
        """Drop a reserved-but-unpublished entry (e.g. the seq was preempted).

        Preemption truncates ``seq.token_ids``, so retiring a checkpoint staged
        against the pre-truncation tokens would publish a key matching no real
        prefix.
        """
        cur = self._allocated.get(entry.key)
        if cur is entry and not entry.published:
            self._release(entry)

    # ── pinning ────────────────────────────────────────────────────────────

    def pin(self, entry: StateCacheEntry) -> None:
        entry.pin_count += 1

    def unpin(self, entry: StateCacheEntry) -> None:
        assert entry.pin_count > 0, "unpin without a matching pin"
        entry.pin_count -= 1

    # ── eviction ───────────────────────────────────────────────────────────

    def _alloc_slot(self) -> int | None:
        if self._free:
            return self._free.pop()
        victim = self._pick_victim()
        if victim is None:
            return None
        self._release(victim)
        self.stats.evictions += 1
        return self._free.pop()

    def _pick_victim(self) -> StateCacheEntry | None:
        """Lowest hit count wins; oldest access breaks ties.

        Pinned entries are never evictable (a DMA is reading them). Entries
        still in flight are likewise skipped — their slot is being written.
        """
        best: StateCacheEntry | None = None
        for entry in self._allocated.values():
            if entry.pin_count > 0 or not entry.published:
                continue
            if best is None or (entry.hit_count, entry.last_access) < (
                best.hit_count,
                best.last_access,
            ):
                best = entry
        return best

    def _release(self, entry: StateCacheEntry) -> None:
        self._allocated.pop(entry.key, None)
        self._free.append(entry.slot)

    # ── per-sequence policy ────────────────────────────────────────────────
    #
    # These own the decisions "where does this sequence resume from" and
    # "where should it leave a checkpoint". They live here rather than in
    # BlockManager so the placement policy sits next to the structure it
    # indexes; BlockManager just calls them at the right points in a
    # sequence's lifecycle.

    def bounded_hit(self, seq, num_cached_blocks: int, block_hashes: list[int]) -> int:
        """Clamp a KV hit to the deepest usable checkpoint.

        Scans candidate boundaries deepest-first and stops at the first
        published checkpoint — only the deepest match matters, so this is
        right-to-left with early exit (same shape as vLLM's
        ``MambaManager.find_longest_cache_hit``, for the same reason).

        Read-only: records the candidate slot on ``seq.state_load_slot`` but
        does NOT pin it, because ``can_allocate`` doubles as a KV-pressure
        probe that may never admit the sequence. ``acquire_load`` pins it at
        admission. Returns 0 on a miss — the sequence prefills normally and
        leaves a checkpoint for the next one.
        """
        seq.state_load_slot = -1
        seq.state_load_entry = None
        if num_cached_blocks <= 0 or self.blocks_per_ckpt <= 0:
            return 0
        step = self.blocks_per_ckpt
        n = (num_cached_blocks // step) * step
        while n >= step:
            entry = self.lookup(block_hashes[n - 1], tuple(seq.block(n - 1)))
            if entry is not None:
                seq.state_load_slot = entry.slot
                # Park the entry itself, not just its slot: acquire_load and
                # release_load need to pin/unpin THIS entry, and a slot->entry
                # index would have to be maintained on every reserve, release
                # and clear just to undo the lookup we already did here.
                seq.state_load_entry = entry
                # The runtime slot will hold this seq's own recurrent state
                # (as of entry.position) before its first forward, so that
                # forward must SEED from it. Without this the kernel is told
                # has_initial_state=False and discards the checkpoint — the
                # recurrence then restarts from zero at the resume point,
                # having skipped the cached prefix entirely (observed: a
                # 1100-note prompt answered as "1000 notes", at full speed).
                seq.has_recurrent_state = True
                return n
            n -= step
        return 0

    def acquire_load(self, seq) -> None:
        """Pin the checkpoint this seq will resume from. Called at admission.

        Split from ``bounded_hit`` so a probe that never admits the sequence
        cannot leak a pin — a pinned entry is never evictable, so repeated
        probing would permanently wedge slots.
        """
        if seq.state_load_slot < 0:
            return
        entry = seq.state_load_entry
        # The entry can be evicted between the probe and admission, so a
        # stale one must not be pinned: its slot may already hold another
        # checkpoint. Identity against the live index is the check.
        if entry is None or self._allocated.get(entry.key) is not entry:
            seq.state_load_slot = -1
            seq.state_load_entry = None
            return
        self.pin(entry)
        self.stats.saved_tokens += entry.position

    def _grid_hashes(self, seq) -> dict[int, int]:
        """``{block_count: chained_hash}`` at every grid boundary in the prompt.

        Pure indexing — this pool never hashes. ``BlockManager.can_allocate``
        owns block hashing and hands over the full chain in
        ``seq.block_hashes``; a checkpoint position is just that chain sampled
        every ``blocks_per_ckpt`` entries. Using the KV pool's own hashes is
        what makes a state checkpoint findable from a KV match.
        """
        step = self.blocks_per_ckpt
        known = getattr(seq, "block_hashes", None) or []
        # Sample the whole prompt: the prompt-end anchor sits at the last
        # block, so stopping short would leave that position unkeyable. The
        # anchor is a WRITE position, so nothing here excludes the last block.
        n_grid = (min(seq.num_blocks, len(known)) // step) * step
        return {i + 1: known[i] for i in range(n_grid) if (i + 1) % step == 0}

    def _reserve_at(self, seq, pos: int, grid_hashes: dict[int, int]):
        """Reserve one checkpoint at ``pos``. Returns the entry or None."""
        if pos <= seq.num_cached_tokens or pos < self.granularity:
            return None
        nblocks = pos // self.block_size
        if nblocks > len(seq.block_table):
            return None
        h = grid_hashes.get(nblocks)
        if h is None:
            return None
        return self.try_reserve(h, pos, tuple(seq.block(nblocks - 1)))

    def plan_save(self, seq) -> None:
        """Reserve checkpoint slots for this sequence.

        **Both** candidate positions are reserved when they differ, because
        they serve different traffic and must not compete for one slot:

        1. **An observed fork** inside the prompt — a position some earlier
           request diverged at. Serves mid-prefix branching (fixed system
           prompt, varying tail), which diverges *before* the prompt end.
        2. **The prompt end**, grid-floored — serves whole-prompt reuse (a
           follow-up turn resending the prompt verbatim).

        Reserving only the fork (the original behaviour) was actively worse
        than having no fork detection: the request skipped its prompt-end
        anchor AND, if no forward happened to land exactly on the fork, the
        fork reservation was cancelled too — so it wrote no checkpoint at all.

        Reservation is soft: a full pool simply skips whichever it cannot fit.
        """
        seq.state_save_slots = []
        seq.state_save_positions = []
        seq.state_save_slot = -1
        seq.state_save_pos = -1
        if not seq.has_per_req_cache:
            return

        # Floor to the grid: a prefill step ends wherever the token budget put
        # it, which is almost never grid-aligned. Whether the state at `pos`
        # is obtained by splitting the step or by reading the kernel's
        # intermediates is the runner's choice.
        # One hash pass, shared by both reservations.
        grid_hashes = self._grid_hashes(seq)
        candidates = []
        # `can_allocate` already found where this seq's hash chain broke and
        # floored it to the grid, so the fork needs no search — every seq that
        # diverges carries its own divergence point. (A table of previously
        # seen forks was tried and removed: it never fired, because a request
        # arriving at a known fork with a different tail breaks there too and
        # so takes this same path.)
        fork_pos = getattr(seq, "fork_hit_blocks", 0) * self.block_size
        if fork_pos > seq.num_cached_tokens:
            candidates.append(fork_pos)
        end_pos = self.floor_to_grid(seq.num_prompt_tokens)
        if end_pos != fork_pos:
            candidates.append(end_pos)

        entries = []
        for pos in candidates:
            entry = self._reserve_at(seq, pos, grid_hashes)
            if entry is None:
                continue
            entries.append(entry)
            seq.state_save_slots.append(entry.slot)
            seq.state_save_positions.append(pos)
            if pos == fork_pos:
                self.stats.fork_checkpoints += 1
        if entries:
            self._pending[seq.id] = entries
            # Back-compat scalars: the FIRST (deepest-value) reservation.
            seq.state_save_slot = entries[0].slot
            seq.state_save_pos = entries[0].position

    def commit_save(self, seq, reached: int) -> None:
        """Publish the checkpoints whose position the forward has now covered.

        Per-position, not all-or-nothing: a request typically reserves a fork
        checkpoint partway through the prompt and an anchor at the end, and
        the earlier one becomes valid on an earlier step. Entries not yet
        covered stay pending.

        ``reached`` is how many of this sequence's tokens now have their state
        folded in (``seq.num_cached_tokens`` after the step).
        """
        entries = self._pending.get(seq.id)
        if not entries:
            return
        still_pending = []
        for entry in entries:
            if reached >= entry.position:
                self.publish(entry)
            else:
                still_pending.append(entry)
        if still_pending:
            self._pending[seq.id] = still_pending
            seq.state_save_slot = still_pending[0].slot
            seq.state_save_pos = still_pending[0].position
        else:
            self._pending.pop(seq.id, None)
            seq.state_save_slot = -1
            seq.state_save_pos = -1
        seq.state_save_slots = [e.slot for e in still_pending]
        seq.state_save_positions = [e.position for e in still_pending]

    def cancel_save(self, seq) -> None:
        """Drop every staged checkpoint for this seq (preemption / abort).

        ``preempt`` truncates ``seq.token_ids``, so publishing an entry keyed
        on the pre-truncation prefix would expose a checkpoint matching no
        real sequence.
        """
        for entry in self._pending.pop(seq.id, ()):
            self.cancel(entry)
        seq.state_save_slot = -1
        seq.state_save_pos = -1
        seq.state_save_slots = []
        seq.state_save_positions = []

    def release_load(self, seq) -> None:
        """Unpin the checkpoint a seq loaded from, once the copy has landed."""
        if seq.state_load_slot < 0:
            return
        entry = seq.state_load_entry
        if entry is not None and entry.pin_count > 0:
            self.unpin(entry)
        seq.state_load_slot = -1
        seq.state_load_entry = None

    def clear(self) -> None:
        """Drop every checkpoint. Mirrors ``BlockManager.clear_cache()``.

        Pinned entries are kept: their slot is being read by an in-flight DMA,
        and reusing it would corrupt the reader.
        """
        pinned = {k: e for k, e in self._allocated.items() if e.pin_count > 0}
        self._allocated = pinned
        held = {e.slot for e in pinned.values()}
        self._free = [s for s in range(self.num_slots) if s not in held]
