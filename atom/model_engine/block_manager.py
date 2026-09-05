# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import array
import logging
from dataclasses import dataclass
from math import inf, isinf

import numpy as np
import xxhash

from atom.config import Config, DCPConfig
from atom.distributed.kv_events import (
    MEDIUM_GPU,
    MEDIUM_REMOTE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointCoordinator
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_cache import StateCache, StateCheckpointCache
from atom.model_engine.state_pool import StateSlotPool
from atom.model_engine.state_runtime import (
    DEFAULT_STATE_RUNTIME,
    StateMaintenanceOps,
    StateRuntime,
    StateTransfer,
)
from atom.utils import envs

logger = logging.getLogger("atom")


def _make_block_stored(
    hashes: list[int],
    tokens: list[int],
    parent: int | None,
    block_size: int,
    medium: str = MEDIUM_GPU,
) -> BlockStored:
    """Construct a BlockStored event from a coalesced run of new blocks."""
    # A list, not the `array("i")` the publish paths carry: the event is
    # msgpack-encoded and msgspec has no encoding for an array. The publisher
    # counts encode failures rather than raising, so an array here takes the
    # event stream down without stopping anything.
    assert isinstance(
        tokens, list
    ), f"BlockStored.token_ids must be a list, got {type(tokens).__name__}"
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=tokens,
        block_size=block_size,
        medium=medium,
    )


def _make_block_removed(hashes: list[int]) -> BlockRemoved:
    return BlockRemoved(block_hashes=hashes, medium=MEDIUM_GPU)


def _make_all_cleared() -> AllBlocksCleared:
    return AllBlocksCleared()


@dataclass(frozen=True)
class JointBoundaryDecision:
    """The outcome of the joint-boundary computation, as a value.

    `_joint_kv_boundary` is a pure function of the prompt and the current pool
    state: it reads the seq's KV-leg inputs and the caches, computes where (if
    anywhere) both legs can jointly resume, and returns this object WITHOUT
    touching the seq or any funnel counter. The caller decides whether to
    commit it -- a fit probe discards it, a real admission applies it via
    `_commit_joint_boundary`. That is what removed the `record` flag that used
    to thread down into the boundary computation: computing and recording are
    now two separate steps in two different places.

    `skip_reason is None` marks a real boundary and the four token/hash fields
    carry it; otherwise `skip_reason` names why no joint boundary was found and
    the other fields are unset. `state_from_hbm` tells the commit which tier the
    STATE leg came from (HBM image vs. an image-sized H2D), decided here because
    it is a read of the same cache state the boundary was computed against.
    """

    boundary_tokens: int = 0
    boundary_hash: int = -1
    kv_tokens: int = 0
    claim_tokens: int = 0
    state_from_hbm: bool = False
    skip_reason: str | None = None


class BlockManager:
    def __init__(
        self,
        config: Config,
        *,
        state_runtime: StateRuntime = DEFAULT_STATE_RUNTIME,
    ):
        block_size = config.kv_cache_block_size
        num_blocks = config.num_kvcache_blocks
        assert num_blocks > 0
        self.block_size = block_size
        self.dcp_world_size = config.decode_context_parallel_size
        # DCP KV-cache interleave granularity S (1 = token-level round-robin).
        self.cp_kv_cache_interleave_size = getattr(
            config, "dcp_config", DCPConfig()
        ).interleave_size
        # dcp_rank is always 0 here: BlockManager runs only on the scheduler
        # (rank 0). DCP rank is used only to compute local token counts for
        # memory reservation; the actual per-rank routing is done in the workers.
        self.dcp_rank = 0
        # Prefix-cache hashing / reuse granularity: under DCP one block_table
        # entry maps to a virtual block of `block_size * dcp_world_size` global
        # tokens (see _hash_block_size). == block_size when DCP is off.
        self.hash_block_size = self.block_size * self.dcp_world_size
        self.enable_prefix_caching = config.enable_prefix_caching
        self.total_evicted_blocks: int = 0

        kv_events = getattr(config, "kv_events_config", None)
        self._events_enabled: bool = bool(kv_events and kv_events.enable)
        self._event_log: list[KVCacheEvent] | None = (
            [] if self._events_enabled else None
        )
        # The compressed KV blocks. Same class the sliding window uses for its
        # own index space — hash eviction has to happen at the same moment in
        # both or a prefix hit could be honoured by one pool and not the other.
        self.kv = BlockPool(num_blocks, on_evict=self._record_evicted)
        # Per-request cache slot pool. Used by attention types with a
        # stateful per-request buffer (GDN recurrent state, V4 compressor
        # state). The backing tensor is pre-allocated by ModelRunner and
        # excluded from `num_kvcache_blocks` at sizing time, so admission only
        # needs free slot indices from this list.
        #
        # Slots are counted raw, not divided into per-request groups. What one
        # request occupies is a property of the request — `1 + num_spec` while
        # it speculates, 1 otherwise — and a checkpoint occupies exactly one
        # whatever the model does, so there is no single width to divide by.
        # `state_slots_per_req` is what a live request asks for.
        pool_entries: dict = getattr(config, "pool_entries", None) or {}
        pool_per_req: dict = getattr(config, "pool_entries_per_req", None) or {}
        # Total capacity, kept so callers can tell "all slots busy" (transient)
        # from "no slots were ever created" (permanent).
        self.num_state_slots = int(pool_entries.get(STATE_SLOT_CLASS, 0))
        self.state_slots_per_req = int(pool_per_req.get(STATE_SLOT_CLASS, 1)) or 1
        # Tokens between rungs of the checkpoint ladder, shared by every
        # Pool.STATE class (--state-checkpoint-interval-tokens).
        #
        # Three regimes, and the sign carries the distinction:
        #   >0  ladder on, a rung every N tokens.
        #    0  state checkpointing off entirely. Nothing is kept, anywhere.
        #   -1  ladder off, checkpointing on: the demand rung and the prompt-end
        #       anchor still place checkpoints, the interval grid does not.
        #
        # -1 rather than reusing 0 for this, even though "no interval" reads
        # like zero: 0 is the documented off switch and it is also *reachable
        # by accident* — the snap below rounds an off-grid interval down and
        # can land on 0, so a `--block-size` typo currently fails safe. Giving 0
        # a second meaning would make that typo silently enable a caching policy
        # instead of disabling one.
        self.state_checkpoint_interval_tokens = max(
            -1, int(getattr(config, "state_checkpoint_interval_tokens", 0) or 0)
        )
        # Independent of the interval, because the demand rung is not part of
        # the grid: it is the one placement a *refused* hit makes for itself.
        # See `_record_checkpoint_demand` for what turning it off is testing.
        #
        # ATOM_STATE_CHECKPOINT_DEMAND wins over the config field when it is
        # exported, so the policy can be flipped for one run without editing a
        # launch script: =0 forces the rung off, =1 forces it on. Unset leaves
        # --state-checkpoint-demand in charge, so an unexported variable costs
        # nothing.
        self.state_checkpoint_demand = bool(
            getattr(config, "state_checkpoint_demand", True)
        )
        if envs.is_set("ATOM_STATE_CHECKPOINT_DEMAND"):
            self.state_checkpoint_demand = envs.ATOM_STATE_CHECKPOINT_DEMAND
        if not self.state_checkpoint_demand:
            logger.info(
                "[State Cache] demand rung disabled: the prompt-end anchor is "
                "the only checkpoint placement."
            )
        checkpoint_spec = state_runtime.checkpoint_spec
        # Read here rather than beside `state_offload` below, because the
        # coordinator has to know at construction whether anything can carry a
        # store: without a sink it must nominate nothing, since a pin nobody
        # releases holds a whole image out of the pool forever.
        from atom.model_engine.state_offload import (
            state_tier_capability,
            state_tier_chunk_tokens,
        )

        # A capability derived from the whole config, not the connector's name.
        # The name only says which class is constructed; whether that class
        # builds a `StateOffloadTier` also depends on the layout it resolves,
        # the pipeline depth, and its role. Installing an index against a
        # worker that will refuse the tier is what left stores emitted with
        # nowhere to go.
        self.state_tier_capability = state_tier_capability(config)
        kv_offload_enabled = self.state_tier_capability.hosts_state_tier
        # Only worth saying for a model that has state to offload. Every dense
        # engine would otherwise announce at startup that a tier it could never
        # use is off, which is true and useless.
        if (
            not kv_offload_enabled
            and checkpoint_spec is not None
            and self.state_tier_capability.reason
        ):
            logger.info(
                "[State Cache] CPU state tier off: %s.",
                self.state_tier_capability.reason,
            )
        self.paged_state_checkpoints: PagedStateCheckpointCoordinator | None = None
        if checkpoint_spec is not None:
            enabled = self.enable_prefix_caching and self.num_state_slots > 0
            self.paged_state_checkpoints = PagedStateCheckpointCoordinator(
                self.kv,
                checkpoint_spec,
                enabled=enabled,
            )
        # The rolling state class: per-request slots plus a content index over
        # the free ones. A checkpoint IS a free slot whose content is still
        # valid, so it holds no capacity of its own and never blocks admission.
        self.state = StateSlotPool(
            self.num_state_slots,
            transfer=(
                StateTransfer.none()
                if self.paged_state_checkpoints is not None
                else state_runtime.transfer
            ),
            hash_block_size=self.hash_block_size,
            enabled=self.enable_prefix_caching,
        )
        self._state_checkpoint_cache: StateCheckpointCache = (
            self.paged_state_checkpoints or self.state
        )
        # A checkpoint is filed under the content hash of the last block it
        # covers, so a rung that isn't a hash-block boundary can never be looked
        # up — the ladder would checkpoint into a void. The interval defaults to
        # 8192 while `hash_block_size` follows `--block-size` and
        # `--decode-context-parallel-size`, so ordinary flag combinations
        # (`--block-size 100`, dcp 3) land off the grid through no choice of the
        # user's. Snap down to the grid and say so, rather than refusing to
        # start — and rather than asserting, which `python -O` would drop and
        # leave the ladder cutting prefill chunks onto rungs nothing can reach.
        # `> 0` rather than truthy: -1 is not an interval to snap onto the grid,
        # it is the absence of one. Snapping it would give -4 and a warning
        # about a flag the user set deliberately.
        if (
            self.state.enabled
            and self.state_checkpoint_interval_tokens > 0
            and self.state_checkpoint_interval_tokens % self.hash_block_size
        ):
            snapped = (
                self.state_checkpoint_interval_tokens // self.hash_block_size
            ) * self.hash_block_size
            logger.warning(
                f"--state-checkpoint-interval-tokens="
                f"{self.state_checkpoint_interval_tokens} is not a multiple of "
                f"the prefix-cache hash block size {self.hash_block_size}; "
                f"snapping to {snapped or 'off (0)'}."
            )
            self.state_checkpoint_interval_tokens = snapped

        # Every Pool.STATE class. A tuple of one today — the sliding window
        # used to be the second member, back when it was a content-addressed
        # block pool that could gate a hit. It is now a per-request ring carried
        # by the state checkpoint, so it has nothing to say about hit length.
        # Kept plural because GDN's recurrent state is a second member the
        # moment it stops forking (see the state-cache protocol).
        self.state_caches: tuple[StateCache, ...] = (self._state_checkpoint_cache,)

        # Class names already warned about in `state_checkpoint_fates`. See
        # there for why the warning latches.
        self._warned_no_checkpoint_fates: set[str] = set()
        from atom.model_engine.state_offload import StateOffloadIndex

        # Joint boundaries, split by what the state leg cost: `hbm` forked a
        # resident checkpoint, `tier` paid an entry-sized H2D and a park. Almost
        # all `tier` means the state pool is too small for the concurrency.
        self.joint_boundaries = 0
        self.state_hbm_boundaries = 0
        # The number that says the tier is doing anything at all. Every other
        # counter here can be non-zero with the CPU tier switched off; this one
        # cannot, which makes it the only honest test of "did this feature run".
        # Suffixed `_boundaries` so an int counter does not share the bare name
        # `state_tier` with the state-tier object and module
        # (`kimi_k3.state_tier`, `_state_tier`). The exported funnel key and the
        # log label keep the short `state_tier` string.
        self.state_tier_boundaries = 0
        # Admissions whose gated boundary neither tier could produce by the time
        # `allocate` ran. Non-zero is expected under pressure (the CPU index is
        # optimistic, and an HBM checkpoint can be unindexed inside the same
        # pass); a large fraction of `joint_boundaries` means the gate is
        # accepting boundaries that do not survive to attach.
        self.state_gate_lost_boundary = 0
        # Why the rest got none, keyed by the gate that stopped them.
        self.joint_skips: dict[str, int] = {}
        # The LMCache chunk size in tokens, read where the config is rather than
        # off the connector object.
        self._joint_chunk_tokens = 0
        # Only when a joint KV load is possible at all. Reading it reaches into
        # LMCache, an optional dependency: probing on an engine that hosts no
        # state tier logged a full `ModuleNotFoundError` traceback at WARNING
        # on every start, for a number that engine has no use for. The
        # capability above is the same gate the tier itself uses.
        if kv_offload_enabled:
            self._joint_chunk_tokens = state_tier_chunk_tokens(config)
        self.state_offload: StateOffloadIndex | None = None
        # `kv_offload_enabled` is the whole switch: without a connector to
        # carry a transfer there is nothing beneath the pool, and a hash whose
        # KV left HBM could not be resumed from anyway.
        #
        # The index no longer reaches into `StateSlotPool`. A K3 checkpoint is
        # a PAGE image in the KV pool (#2045), so the spill source is a set of
        # units the coordinator owns, not a slot the pool is about to hand
        # away -- which is what the staging ring existed to rescue.
        if kv_offload_enabled:
            # `release_slot` is the index's only allocator dependency. It owns
            # a load's destination slot once the request is torn down under it
            # (`orphan`), because a worker may still be scattering into that
            # slot, and hands it back when the load finally settles.
            self.state_offload = StateOffloadIndex(
                can_store=self.state_tier_capability.can_store_state,
                can_load=self.state_tier_capability.can_load_state,
                release_slot=self.state.release,
            )
            # Attached rather than passed at construction: the coordinator is
            # built before the switch is read (it needs `checkpoint_spec`,
            # which comes off `state_runtime`). One object, two uses -- the
            # coordinator votes off `hashes` and drains stores into it.
            if self.paged_state_checkpoints is not None:
                self.paged_state_checkpoints.attach_offload(
                    self.state_offload,
                    sink=self.state_tier_capability.can_store_state,
                )
        # The demand funnel: recorded at admission, cut for when a prefill
        # chunk is shortened to land on it, kept when the state pool files it.
        # Counted at all three because a gap between any two is a different
        # bug, and they are indistinguishable in the hit rate alone.
        self.demands_recorded: int = 0
        self.chunks_cut_for_demand: int = 0
        self.demands_declined_no_room: int = 0
        # The anchor's own cut counter, kept separate from the demand's: the
        # anchor fires on nearly every prompt and would drown the convergence
        # signal `chunks_cut_for_demand` exists to expose.
        self.chunks_cut_for_end: int = 0

    @classmethod
    def compute_hash(cls, token_ids: array.array, prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        # dtype pinned even though every caller now passes an `array("i")`:
        # `np.array` infers int64 from a list and int32 from an array, so the
        # digest used to depend on the caller's Python type. int64 is what
        # lists gave, which leaves every hash recorded before that where it
        # was -- and keeps a caller who does pass a list from silently
        # computing a different one.
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    def complete_previous_state_batch(self) -> None:
        """Complete state reads and copies issued by the previous batch."""
        self.state.release_pins()
        if self.paged_state_checkpoints is not None:
            self.paged_state_checkpoints.complete_previous_batch()

    def take_state_maintenance_ops(self) -> StateMaintenanceOps:
        """Drain state maintenance for the batch being built."""
        relocations = self.state.take_relocations()
        stores = restores = ()
        if self.paged_state_checkpoints is not None:
            stores, restores = self.paged_state_checkpoints.take_checkpoint_ops()
        return StateMaintenanceOps(
            relocations=relocations,
            checkpoint_stores=stores,
            checkpoint_restores=restores,
        )

    def state_checkpoint_fates(self) -> dict[str, int]:
        """Summed fates across every state class, for the periodic stats line.

        Accumulates whatever each class's ``checkpoint_fates()`` returns, so a
        new counter appears here with no change to this method. A class without
        the method is skipped with a warning, latched per class: the caller is a
        periodic stats line and the omission is a static property of the build.
        """
        totals: dict[str, int] = {}
        for cache in self.state_caches:
            fates_fn = getattr(cache, "checkpoint_fates", None)
            if fates_fn is None:
                name = type(cache).__name__
                if name not in self._warned_no_checkpoint_fates:
                    self._warned_no_checkpoint_fates.add(name)
                    logger.warning(
                        "state_checkpoint_fates: %s does not implement "
                        "checkpoint_fates(); its counters are excluded from "
                        "totals",
                        name,
                    )
                continue
            for k, v in fates_fn().items():
                totals[k] = totals.get(k, 0) + v
        return totals

    def _record_evicted(self, h: int) -> None:
        """A hash the block pool just dropped: report it, and settle the state.

        The crossing belongs here rather than in either pool — the two are
        addressed by one chained content hash and a prefix hit claims both, so
        neither can be left holding a boundary the other can no longer honour.
        Without this the state pool keeps handing slots to checkpoints nothing
        can reach and spends live ones to make room for them.
        """
        self.total_evicted_blocks += 1
        if self._event_log is not None:
            self._event_log.append(_make_block_removed([h]))
        self._state_checkpoint_cache.unindex(h)

    def _fresh_block(self) -> int:
        """Take a block for content this step is about to compute.

        The raise is unreachable through `Scheduler` and the checkpoint cache
        cannot make it reachable: a READY unpinned checkpoint counts as
        available, and both callers sit behind a pin-aware check in the same
        pass. `allocate` protects the one checkpoint it is about to pin and
        sees the pins taken before it; `may_append` runs only in a pass that
        scheduled no prefill (`scheduler.py`, `if num_seqs_prefill > 0` returns
        first), so every pin was already released at the top of that pass.
        Under contention the reachable outcome is a refused admission, not
        this.

        That second half rests on prefill and decode never sharing a pass. If
        the mixed batch that `scheduler.py` has a TODO for lands, `may_append`
        starts running alongside this pass's pins and the argument has to be
        redone.
        """
        if not self._ensure_page_units(1):
            raise AssertionError("No PAGE unit available for a fresh KV block")
        block_id = self.kv.pop()
        self.kv.allocate(block_id)
        return block_id

    def disown_claimed_prefix(self, seq: Sequence, keep_blocks: int = 0) -> bool:
        """Make this seq's claimed canonical prefix private, in place.

        A disown sets ``num_cached_tokens = 0`` so the forward recomputes from
        token 0, but ``block_table`` still points at the hash-indexed blocks
        this seq took with ``kv.claim`` -- blocks another sequence may be
        decoding out of. Recomputing writes KV in place into those shared
        blocks and tears the other holder's values (non-reproducible
        corruption -- this is review finding #2). This hands each claimed
        prefix block in ``block_table[keep_blocks:]`` back (``kv.free``) and
        drops a fresh private block (``_fresh_block``) into the same slot, so
        the table length and every other position's mapping are unchanged. The
        fresh tail (``hash == -1``) is already private and is left untouched;
        so is a still-shared block below ``keep_blocks``.

        Returns ``False`` when the pool cannot back the private copies -- the
        caller must then abandon this admission (deallocate + requeue) rather
        than run a forward over shared blocks. Only blocks still shared at
        ``ref_count > 1`` cost a net PAGE unit here (``free`` releases nothing
        while another holder remains), and ``can_allocate`` never reserved them
        (it subtracts already-used hit blocks from ``num_new_blocks``); those
        are reserved up front. ``ref_count == 1`` blocks recycle their own unit
        and never fail.
        """
        if not self.enable_prefix_caching:
            return True
        table = seq.block_table
        n_shared = sum(
            1
            for i in range(keep_blocks, len(table))
            if self.kv.block(table[i]).hash != -1
            and self.kv.block(table[i]).ref_count > 1
        )
        if n_shared and not self._ensure_page_units(n_shared):
            return False
        for i in range(keep_blocks, len(table)):
            if self.kv.block(table[i]).hash == -1:
                continue  # already-private fresh tail (or a prior disown)
            self.kv.free(table[i])
            table[i] = self._fresh_block()
        return True

    def _checkpoint_has_room(
        self, live_blocks: int = 0, protected_hash: int | None = None
    ) -> bool:
        """Whether an image still fits once `live_blocks` have been taken.

        `live_blocks` is what the admission asking this is about to allocate.
        Counting it is the difference between "there is room for an image" and
        "there is room for this request and an image", and only the second is
        the question: the request's blocks are taken first.

        `protected_hash` is the checkpoint the same admission is about to pin,
        excluded from what eviction could reclaim — the same argument
        `can_allocate` passes to `_has_page_units` on the next line, so the
        two gates in one pass agree on what is spendable.

        `True` when no PAGE-backed checkpoints exist at all: a fork checkpoint
        costs the pool nothing, so there is nothing to gate.
        """
        if self.paged_state_checkpoints is None:
            return True
        return self.paged_state_checkpoints.has_available_units(
            live_blocks + self.paged_state_checkpoints.store.units_per_checkpoint,
            protected_hash=protected_hash,
        )

    def _has_page_units(
        self, count: int, protected_checkpoint_hash: int | None = None
    ) -> bool:
        if self.paged_state_checkpoints is None:
            return self.kv.has_free(count)
        return self.paged_state_checkpoints.has_available_units(
            count, protected_hash=protected_checkpoint_hash
        )

    def _ensure_page_units(self, count: int) -> bool:
        if self.paged_state_checkpoints is None:
            return self.kv.has_free(count)
        return self.paged_state_checkpoints.ensure_free_units(count)

    def num_pool_blocks(self, seq_len: int) -> int:
        """KV pool blocks a `seq_len`-token sequence occupies on this rank.

        Under DCP a rank stores only its interleaved shard, so this is a factor
        of `dcp_world_size` below the global `ceil(seq_len / block_size)`. The
        pool is sized in these same per-rank units, so this is the only count
        that may be compared against `kv.num_blocks` — whether to draw from the
        pool (`can_allocate`/`allocate`) or to reject a prompt as too large for
        it (`Scheduler._unschedulable_reason`).
        """
        if self.dcp_world_size <= 1:
            return (seq_len + self.block_size - 1) // self.block_size
        from atom.model_ops.dcp_ops import get_dcp_local_seq_lens

        local_len = get_dcp_local_seq_lens(
            np.array([seq_len]),
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )[0]
        return int((local_len + self.block_size - 1) // self.block_size)

    @property
    def max_pool_tokens(self) -> int:
        """Longest prompt, in global tokens, whose KV fits an entirely empty pool.

        Bisects `num_pool_blocks`, which is monotone in `seq_len`, rather than
        inverting it in closed form: under block-level interleaving
        (`cp_kv_cache_interleave_size > 1`) a rank's share is not a plain
        `seq_len / dcp_world_size`, and an inverse derived by hand would drift
        from the allocator as soon as that arithmetic moved. Runs once, at
        startup.

        Mirrors the ceiling `Scheduler._unschedulable_reason` enforces, so the
        frontend can predict that verdict and refuse an oversized prompt with an
        error while it is still answering the client, instead of leaving the
        scheduler to discover it once the client is already waiting. The API
        server needs it published because `num_kvcache_blocks` is measured in
        the engine subprocess and its own Config never learns the value.
        """
        capacity = self.kv.num_blocks
        # A prompt this long needs more than `capacity` blocks on some rank, so
        # it bounds the search from above: each rank holds at least
        # `1 / dcp_world_size` of it, i.e. over `capacity` blocks' worth.
        hi = (capacity + 1) * self.block_size * max(1, self.dcp_world_size)
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.num_pool_blocks(mid) <= capacity:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _effective_block_size(self):
        return self.block_size * self.dcp_world_size

    def _hash_block_size(self) -> int:
        return self.hash_block_size

    def _n_hash_blocks(self, seq: Sequence) -> int:
        hbs = self.hash_block_size
        return (len(seq) + hbs - 1) // hbs

    def _hash_block_tokens(self, seq: Sequence, i: int) -> array.array:
        hbs = self.hash_block_size
        return seq.token_ids[i * hbs : (i + 1) * hbs]

    def _gated_hit(
        self,
        seq: Sequence,
        compressed_hit: int,
        block_hashes: list[int],
        assume_checkpointed: bool = False,
    ) -> int:
        """Rightmost boundary every Pool.STATE class can resume from.

        Each class answers "the rightmost boundary <= X that I accept", and no
        class is monotone in another's answer, so they cannot be applied in
        series: the largest SWA-complete boundary need not carry a state
        checkpoint, and walking back to one that does can land on a boundary
        whose trailing SWA window is gone. `allocate` then calls
        `swa.claim_cached` for a hash the SWA pool never promised — which is
        exactly the guarantee that method's docstring asks the caller for.

        So run to a fixpoint: keep passing the candidate around the classes
        until a full round changes nothing. Every answer is <= its input, so
        each round either terminates or strictly decreases; 0 is absorbing.
        Classes that do not apply are the identity, so a build with one class
        settles on the first round.

        `assume_checkpointed` passes straight through to every class, giving the
        joint counterfactual: not "the answer minus one class's gate" but "the
        answer if every ladder were dense". A boundary the other classes decline
        anyway is one no checkpoint would rescue.
        """
        boundary = compressed_hit
        while boundary > 0:
            settled = True
            for cache in self.state_caches:
                accepted = cache.resumable_hit(
                    seq,
                    boundary,
                    block_hashes,
                    assume_checkpointed=assume_checkpointed,
                )
                if accepted != boundary:
                    boundary = accepted
                    settled = False
                    if boundary == 0:
                        return 0
            if settled:
                break
        return boundary

    def pool_occupancy(self) -> dict[str, int]:
        used = self.kv.num_used
        free = self.kv.num_free
        hashed = self.kv.num_indexed
        return {
            "used": used,
            "free": free,
            "total": self.kv.num_blocks,
            "hashed": hashed,
            "retained": max(0, hashed - used),
            "evicted_total": self.total_evicted_blocks,
        }

    def _joint_kv_boundary(
        self,
        seq: Sequence,
        hbm_boundary: int,
        block_hashes: list[int],
    ) -> JointBoundaryDecision:
        """The boundary above the HBM hit that BOTH legs can reach, as a value.

        A pure computation: it reads the seq's KV-leg inputs and the caches and
        returns a `JointBoundaryDecision`, but writes nothing on the seq and
        moves no funnel counter. `can_allocate` decides whether to commit the
        result -- a real admission calls `_commit_joint_boundary`, a fit probe
        discards it. Separating the two is what let the `record` flag that used
        to thread down through here disappear.

        `can_allocate` walks the HBM prefix cache and nothing else, so an
        evicted prefix stops at the first miss even when LMCache holds every
        block. This looks past that: the KV leg fetches `[hbm, B)` from LMCache
        while the state leg fetches B's checkpoint from the tier, and the two
        land on one boundary or neither runs.

        The grids do not line up -- a state rung is a hash-block boundary, the
        KV leg moves whole chunks. Rather than discard every unaligned rung, the
        KV leg is aimed at the chunk *covering* B while the request claims only
        B: overshooting costs one chunk into blocks the forward is about to
        rewrite, undershooting would be silent wrong output.
        """
        if self.state_offload is None:
            return self._no_joint("off")
        # PAGE is now the served class, not the excluded one. This used to
        # read `is not None` and refuse -- correct while a K3 checkpoint was an
        # Active Slot the tier spilled out of `StateSlotPool`. #2045 moved the
        # image into the KV pool, and with it the whole reason the joint path
        # exists: HBM's `state ⊆ KV` (`_record_evicted` unindexes a checkpoint
        # the instant its boundary block is spent) means a checkpoint can no
        # longer outlive its KV in HBM -- so when LMCache hands the KV back,
        # nothing hands the state back unless the two are fetched together.
        if not seq.has_per_req_cache or self.paged_state_checkpoints is None:
            return self._no_joint("no_paged_checkpoints")
        hbs = self._hash_block_size()
        # Resolved once from config at construction (`state_tier_chunk_tokens`),
        # not off the connector object: the scheduler holds whatever
        # `get_kvconnector` returned, and one without `chunk_size` would zero
        # this and disable the feature silently.
        chunk = self._joint_chunk_tokens
        lmc_tokens = int(seq.offload_joint.kv_prefix_tokens or 0)
        if chunk <= 0:
            return self._no_joint("no_chunk_size")
        # Floored to the grid everything below compares against, so the
        # covering-chunk check becomes true by construction rather than by
        # luck. (`get_num_new_matched_tokens` withholds one token on a
        # full-prompt hit, which takes the lookup off the grid.)
        lmc_tokens = (lmc_tokens // chunk) * chunk
        if lmc_tokens <= hbm_boundary * hbs:
            return self._no_joint("lmcache_within_hbm")
        # The KV leg moves whole chunks and the blocks below the HBM prefix are
        # shared, so an unaligned start cannot be rounded down.
        if (hbm_boundary * hbs) % chunk != 0:
            return self._no_joint("hbm_off_chunk_grid")
        cap = min(lmc_tokens // hbs, self._n_hash_blocks(seq) - 1)
        if cap <= hbm_boundary:
            return self._no_joint("no_room_above_hbm")
        chain = self._chain_to(seq, block_hashes, cap)
        # One scan: `_gated_hit` already returns the rightmost rung
        # `_resumable_from` accepts, so a decrement-and-rescan walk would spend
        # a fixpoint pass per rung to reach the same answer.
        candidate = self._gated_hit(seq, cap, chain)
        if candidate <= hbm_boundary:
            return self._no_joint("no_rung_above_hbm")
        tokens = candidate * hbs
        h = chain[candidate - 1]
        # Cannot fail given the two premises above; kept as their assertion.
        kv_tokens = -(-tokens // chunk) * chunk
        if kv_tokens > lmc_tokens:
            return self._no_joint("covering_chunk_beyond_lookup")
        # Everything up to the compressed hit is this prompt's KV and is still
        # in the pool -- what `_gated_hit` cut was resumability, not residency.
        # Below the joint boundary those blocks are read-only for this request
        # (the forward starts at the boundary), so claiming them is free, and
        # it is the difference between the KV leg fetching `[hit, B)` and
        # fetching `[0, B)` with the front half already sitting in HBM.
        #
        # `candidate > hbm_boundary` and `hbm_boundary` is the RIGHTMOST rung
        # `<= compressed_hit`, so the boundary is always past the compressed
        # hit and this min is always the compressed hit. Written as a min
        # anyway: the invariant belongs to `_gated_hit`, not here.
        claim_blocks = min(len(block_hashes), candidate)
        # Floored to the chunk grid so the KV leg starts aligned and writes
        # only into blocks below it that nobody else can be reading.
        claim_tokens = ((claim_blocks * hbs) // chunk) * chunk
        claim_tokens = max(claim_tokens, hbm_boundary * hbs)
        # `allocate` claims the boundary in whole hash blocks --
        # `claim_tokens // hbs` -- but the value above is floored to
        # the *chunk* grid, not the hash-block grid. When `hbs` does not divide
        # `chunk`, a chunk-aligned claim is not hash-block-aligned: the floor
        # drops the `[floor(claim_tokens/hbs)*hbs, claim_tokens)` tail into a
        # fresh, unfilled block that `num_cached_tokens` (raised to the boundary
        # once the KV leg lands) then counts as computed -- silent wrong output,
        # with no exception. It is invisible whenever `hbs | chunk`, which is the
        # shipping case (`block_size * dcp_world_size == LMCACHE_CHUNK_SIZE`), so
        # it only surfaces at DCP world sizes that take `hbs` off the chunk grid.
        # Refuse the joint and recompute the tail rather than resume over a gap.
        if claim_tokens % hbs != 0:
            return self._no_joint("claim_off_hash_grid")
        # Which tier the STATE leg came from. The KV leg's own tier is decided
        # separately by `_decide_load_after_alloc`, which is why this is not
        # named for the boundary as a whole. Read here, off the same cache the
        # boundary was computed against, and carried on the decision so the
        # commit does not have to look it up again.
        return JointBoundaryDecision(
            boundary_tokens=tokens,
            boundary_hash=h,
            kv_tokens=kv_tokens,
            claim_tokens=claim_tokens,
            state_from_hbm=self.paged_state_checkpoints.contains(h),
        )

    @staticmethod
    def _no_joint(reason: str) -> JointBoundaryDecision:
        """The empty decision, naming why no joint boundary was found.
        Every reason is counted at commit time, including "this build is not
        even trying", because a silent zero is indistinguishable from a feature
        that ran and found nothing.
        """
        return JointBoundaryDecision(skip_reason=reason)

    def _commit_joint_boundary(
        self, seq: Sequence, decision: JointBoundaryDecision
    ) -> None:
        """Apply a `JointBoundaryDecision` to the seq and the funnel counters.

        The recording half of what `_joint_kv_boundary` used to do inline under
        `record=True`: a real admission calls this, a fit probe does not, so the
        counters move once per admission and never for a seq that only probed.
        `reset_joint` first so a boundary that resolved to a skip this pass
        leaves no stale span from a prior one -- the same clean-slate the old
        code got from resetting at the top of every boundary walk.
        """
        seq.offload_joint.reset_joint()
        if decision.skip_reason is not None:
            self.joint_skips[decision.skip_reason] = (
                self.joint_skips.get(decision.skip_reason, 0) + 1
            )
            return
        seq.offload_joint.boundary_tokens = decision.boundary_tokens
        seq.offload_joint.boundary_hash = decision.boundary_hash
        seq.offload_joint.kv_tokens = decision.kv_tokens
        seq.offload_joint.claim_tokens = decision.claim_tokens
        self.joint_boundaries += 1
        if decision.state_from_hbm:
            self.state_hbm_boundaries += 1
        else:
            self.state_tier_boundaries += 1

    def _chain_to(
        self, seq: Sequence, block_hashes: list[int], blocks: int
    ) -> list[int]:
        """`block_hashes` continued to `blocks` entries, hashes only.
        A chained hash is a function of the prompt alone, so it is computable
        past where the HBM cache stops -- which is what makes an LMCache-only
        boundary addressable. Resumes from the last APPENDED hash: on a miss the
        caller's loop variable holds a hash that is not in the chain.
        """
        chain = list(block_hashes)
        h = chain[-1] if chain else -1
        for i in range(len(chain), blocks):
            h = self.compute_hash(self._hash_block_tokens(seq, i), h)
            chain.append(h)
        return chain

    def can_allocate(self, seq: Sequence, record: bool = True) -> int:
        """Return number of cache-hit blocks (>=0) if seq fits, else -1.

        `record=False` marks a fit probe -- asking only whether the seq *could*
        be admitted. The fit answer is identical, but the probe is not
        side-effect-free: the instrumentation and checkpoint-demand/-end writes
        below run before the `record` gate. That is safe -- the sole probe caller
        reads only the `>= 0` return, and the next real `can_allocate` overwrites
        those fields first. `record` gates only the joint-boundary commit (seq
        joint fields + funnel counters), so a probe cannot inflate the
        operator-visible funnel (see `_commit_joint_boundary`).

        The hit count is the contiguous run of cache hits starting at the
        prompt's first block. On the first miss we break: subsequent blocks
        cannot match either (hash is chained, so a divergent token breaks the
        chain for the rest of the prompt). The last block is never considered
        for reuse — prefill must forward at least one block to produce
        sampler logits, so it always comes from the free pool.

        Caller (scheduler) passes the returned hit count to `allocate()`,
        avoiding a second hash pass.
        """
        # Active Slots are preallocated; PAGE checkpoints share the KV pool.
        #
        # The full per-request width, because that is what `allocate` will take:
        # gating on one slot would admit a request the pool cannot give a
        # rollback set to.
        if seq.has_per_req_cache and not self.state.has_free(self.state_slots_per_req):
            return -1
        if not self.enable_prefix_caching:
            if not self._has_page_units(self.num_pool_blocks(len(seq))):
                return -1
            return 0
        # Step 1: compressed prefix (CSA/HCA/indexer share the block hash and
        # read the WHOLE history, so this stays a full front-to-back chained
        # match). Record each block's hash for the SWA scan below.
        h = -1
        compressed_hit = 0
        block_hashes: list[int] = []
        for i in range(self._n_hash_blocks(seq) - 1):
            token_ids = self._hash_block_tokens(seq, i)
            h = self.compute_hash(token_ids, h)
            block_id = self.kv.lookup(h)
            if block_id == -1 or self.kv.block(block_id).token_ids != token_ids:
                break
            block_hashes.append(h)
            compressed_hit += 1
        # Step 2: SWA only needs the trailing window before the boundary to be
        # present (SWA is local). Scan right-to-left within the compressed prefix
        # for the largest boundary whose window is SWA-cached (vLLM
        # SlidingWindowManager; simple-hybrid one pass). Reduces compressed_hit
        # → num_cached_blocks so we never reuse a block whose in-window SWA is
        # gone (#1417), while out-of-window front blocks (SWA-freed) don't block
        # the hit —
        # plus step 3, the per-request state: neither the SSM recurrent state nor
        # the V4 compressor ring can be rebuilt from cached blocks — the cache
        # holds the compressor's output, the state is its rolling input window —
        # so a boundary is only resumable where somebody checkpointed the state.
        # `_gated_hit` settles the two gates jointly; neither can be applied to
        # the other's answer.
        num_cached_blocks = self._gated_hit(seq, compressed_hit, block_hashes)
        # Instrumentation: the pre-gate hit, so EngineStats can separate reuse
        # the gates declined (compressed_hit - num_cached_blocks) from reuse
        # lost to compressed eviction (everything above compressed_hit).
        seq.num_compressed_hit_blocks = compressed_hit
        # Free-pool demand: blocks we actually reuse minus those already used
        # (shared ref); blocks we drop from the hit become fresh → counted.
        num_new_blocks = self._n_hash_blocks(seq)
        for i in range(num_cached_blocks):
            if self.kv.is_used(self.kv.lookup(block_hashes[i])):
                num_new_blocks -= 1
        protected_hash = (
            block_hashes[num_cached_blocks - 1] if num_cached_blocks else None
        )
        # After `num_new_blocks`, not before: the demand's room check has to
        # account for what this very admission is about to take, or it reads a
        # pool it then drains itself.
        self._record_checkpoint_demand(
            seq,
            hit=num_cached_blocks,
            compressed_hit=compressed_hit,
            block_hashes=block_hashes,
            live_blocks=num_new_blocks,
            protected_hash=protected_hash,
        )
        self._record_checkpoint_end(seq)
        if not self._has_page_units(num_new_blocks, protected_hash):
            return -1
        # A boundary LMCache and the tier can jointly reach, above this hit.
        # Committed to the seq rather than returned: what `allocate` claims from
        # HBM is still `num_cached_blocks`, and the joint boundary only decides
        # where the two loads are aimed. `record` is the probe/admission split:
        # a real admission computes the boundary and commits it (seq fields +
        # funnel counters); a fit probe skips it entirely. The `num_cached_blocks`
        # returned below does not depend on the decision, so gating the
        # computation -- not just the commit -- spares a probe the O(prompt) work
        # for a value it would only discard: `_joint_kv_boundary` walks a chained
        # xxhash up to the LMCache-only cap (`_chain_to`) plus a `_gated_hit`
        # rescan. A 128k prompt at the front of a KV-pressured queue paid that
        # full chain on every scheduling pass (`is_mixed_batch` peeks up to four
        # waiting seqs with `record=False`) for a decision nothing read. Placed
        # below the refusal, not above it, for the same reason `_extend_hash_chain`
        # is: a refused admission would discard it, and nothing between here and
        # the refusal reads the seq's joint fields -- this is that move's twin.
        if record:
            decision = self._joint_kv_boundary(seq, num_cached_blocks, block_hashes)
            self._commit_joint_boundary(seq, decision)
        # After the refusal, not before it. The chain is O(prompt) xxhash plus
        # two temporaries per block, and a refused admission discards it — a
        # 128k prompt queued behind a full pool paid ~2000 rounds per waiting
        # request per scheduling pass for a list nobody read. Nothing above
        # consumes it: `_record_checkpoint_demand` takes `block_hashes` as an
        # argument, and `_record_checkpoint_end` derives its position from
        # `num_prompt_tokens` alone. Its one reader is `midstep_positions`.
        #
        # Which makes it free today: the call returns at its first line while
        # no backend declares `readable_midstep` (see
        # `GDNStateMixin.state_transfer`). Kept in place, and in the right
        # place, because that is a policy decision and this is not.
        self._extend_hash_chain(seq, block_hashes)
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0) -> bool:
        """Allocate blocks for `seq`. `num_cached_blocks` is the hit count
        returned by `can_allocate` (0 if caller didn't call it).

        Hash registration is deferred to hash_blocks(), called from
        scheduler.postprocess() once the forward has computed each block's
        KV. This keeps the manager correct under future chunked-prefill
        scheduling: a block spanning multiple steps must not be published as
        a hash until fully filled.

        Returns ``True`` on success. Returns ``False`` only when a state-less
        joint boundary had to be disowned (finding #2) and the pool could not
        back the private prefix copies; the caller must then deallocate and
        requeue the seq rather than run a forward over shared blocks.
        """
        assert not seq.block_table
        # Two extents, and they are not the same number. `num_cached_blocks` is
        # what the request may call cached -- it needs a resumable state behind
        # it. `claim_blocks` is what it may point its block table at: every
        # block the prefix walk matched, whether or not a checkpoint sits there.
        # The gap between them is real reuse that the state gate declined, and
        # taking it costs nothing: those blocks are below the joint boundary the
        # forward will start from, so nobody writes them.
        #
        # Only a joint boundary widens it -- without one there is nothing above
        # `num_cached_blocks` this request will ever treat as computed, and
        # claiming further would pin blocks the forward is about to overwrite.
        hbs = self._hash_block_size()
        claim_blocks = max(
            num_cached_blocks,
            int(seq.offload_joint.claim_tokens or 0) // hbs,
        )
        h = -1
        hit_hash = -1
        for i in range(claim_blocks):
            token_ids = self._hash_block_tokens(seq, i)
            h = self.compute_hash(token_ids, h)
            block_id = self.kv.lookup(h)
            if block_id == -1:
                # Evicted between `can_allocate` and here. For the widened joint
                # tail this is expected -- LMCache or the state tier still holds
                # it and the forward refetches. For the cached range it should be
                # unreachable (nothing evicts inside one admission), but an
                # `assert i >= num_cached_blocks` vanishes under `python -O`, and
                # the old fall-through then recorded `num_cached_blocks * hbs` as
                # cached over a prefix it could not claim, left `hit_hash`
                # unassigned (cold-start state exit), and clamped only
                # `claim_tokens` while `boundary`/`kv` stayed above
                # the claimed region -- silent wrong output on both counts. Use
                # real control flow instead: never treat more than the `i` blocks
                # actually claimed as cached, and drop any joint boundary that
                # sits above them so the KV leg cannot resume over a gap.
                claimed_tokens = i * hbs
                num_cached_blocks = min(num_cached_blocks, i)
                # Partial reset by design: the claim is clamped, not zeroed,
                # so this cannot go through `reset_joint`.
                if int(seq.offload_joint.boundary_tokens or 0) > claimed_tokens:
                    seq.offload_joint.boundary_tokens = 0
                    seq.offload_joint.boundary_hash = -1
                    seq.offload_joint.kv_tokens = 0
                seq.offload_joint.claim_tokens = min(
                    int(seq.offload_joint.claim_tokens or 0),
                    claimed_tokens,
                )
                break
            self.kv.claim(block_id)
            seq.block_table.append(block_id)
            # Track the hash of the last claimed *cached* block as we go, not
            # only when the loop reaches num_cached_blocks-1. If the block_id==-1
            # branch above clamps num_cached_blocks to an earlier `i` (an
            # eviction inside the cached range), the `i + 1 == num_cached_blocks`
            # test never fired, leaving hit_hash == -1 while num_cached_tokens
            # stayed > 0 -- a cold-start state restore over a prefix that is in
            # fact cached. Assigning on every cached block leaves hit_hash on the
            # last one actually claimed, whichever way the loop exits.
            if i < num_cached_blocks:
                hit_hash = h
        # Pin the restore before fresh blocks can evict its checkpoint.
        state_holds = True
        if seq.has_per_req_cache and self.paged_state_checkpoints is not None:
            # The joint boundary, when there is one -- the same rule the fork
            # branch below already used, and the PAGE branch did not.
            #
            # `hit_hash` is the hash at the HBM hit, and a joint boundary sits
            # strictly above it by construction. Two ways that went wrong here:
            # with `num_cached_blocks == 0` the loop never assigned `hit_hash`
            # at all, so this took the `-1` cold-start exit and requested no
            # state whatsoever; with a non-zero hit it restored the checkpoint
            # covering `[0, hbm)`. Either way the KV leg then loaded to the
            # joint boundary and `_claim_after_load` raised `num_cached_tokens`
            # to it, so the forward resumed over a prefix the state does not
            # cover. Silent wrong output, no exception.
            joint_hash = seq.offload_joint.boundary_hash
            state_holds = self._attach_state_slots(
                seq, joint_hash if joint_hash != -1 else hit_hash
            )
        for _ in range(len(seq.block_table), self.num_pool_blocks(len(seq))):
            seq.block_table.append(self._fresh_block())
        seq.num_cached_tokens = num_cached_blocks * hbs

        # Per-request cache: claim this seq's slot indices from the
        # pre-allocated state tensor (e.g. GDN mamba_k_cache, the V4 compressor
        # ring). The state class took its bytes before the paged class was sized
        # in ModelRunner.get_num_blocks(), so admitting a seq adds no further
        # paged-block cost. The state pool's free list is the sole admission
        # bound for state cache.
        if seq.has_per_req_cache and self.paged_state_checkpoints is None:
            # A joint load aims the state leg at its own boundary, which is
            # above the HBM hit by construction. `num_cached_tokens` stays at
            # the HBM prefix until the KV leg lands, so the forward covers
            # `[hbm, num_prompt)` if anything goes wrong from here.
            joint_hash = seq.offload_joint.boundary_hash
            state_holds = self._attach_state_slots(
                seq, joint_hash if joint_hash != -1 else hit_hash
            )
        if seq.has_per_req_cache:
            seq._state_initialized_after_alloc = False
        # A claimed joint boundary must have a state leg behind it. The two
        # decisions are made in different places -- `can_allocate` picks the
        # boundary, `_attach_state_slots` secures the state -- and every way
        # they have disagreed so far ends the same way: the KV leg loads to the
        # boundary, `_claim_after_load` claims it, and the forward runs over a
        # prefix no state covers. This is the backstop that turns any such
        # disagreement into a recompute instead, and it is deliberately a
        # separate check from `state_holds` rather than folded into it: it
        # holds even if the code above stops asking the right question.
        if (
            state_holds
            and seq.has_per_req_cache
            and seq.offload_joint.boundary_hash != -1
            and not self._state_leg_secured(seq)
        ):
            self.state_gate_lost_boundary += 1
            logger.warning(
                "state offload: a joint boundary was admitted for request %s "
                "with no state restore or load behind it; disowning it. This "
                "is a bug in the joint gate, not a cache miss.",
                seq.id,
            )
            state_holds = False
        if not state_holds:
            # No state behind the boundary means it is not this request's
            # history. Disown it -- the forward recomputes from 0. Privatise the
            # claimed prefix first (finding #2): recomputing over the shared
            # canonical blocks would tear another sequence's decode. Keeping the
            # boundary is silent wrong output too, since `has_initial_state`
            # (`gdn_attn.py`) is `num_cached_tokens > 0`.
            if not self.disown_claimed_prefix(seq):
                # Pool cannot back the private copies; signal the caller to
                # deallocate and requeue for a clean recompute next pass.
                return False
            seq.num_cached_tokens = 0
            # Clears only the boundary here, as the pre-record code did -- a
            # partial reset, not a full `reset_joint`.
            seq.offload_joint.boundary_tokens = 0
            seq.offload_joint.boundary_hash = -1
        return True

    def _state_leg_secured(self, seq: Sequence) -> bool:
        """Whether something will really put state behind this seq's boundary.

        Exactly three ways that can be true, and they are the three exits
        `_attach_state_slots` takes when it returns True with a boundary
        claimed: a PAGE restore was queued, a fork source was adopted or read,
        or a CPU load was requested. A cold start is not one of them -- it
        returns True as well, which is correct with no boundary and wrong with
        one.
        """
        if seq.offload_joint.load_hash != -1:
            return True  # a CPU load is in flight for it
        if getattr(seq, "state_fork_src", -1) != -1:
            return True  # a fork checkpoint is its source
        if self.paged_state_checkpoints is None:
            return False
        return self.paged_state_checkpoints.restore_queued_for(seq.state_slot)

    def _attach_state_slots(self, seq: Sequence, hit_hash: int) -> bool:
        """Give `seq` its state slots, resuming from a checkpoint when one exists.

        Returns whether `hit_hash`'s state is really the one `seq` now holds.
        False means the caller must drop the boundary; see `allocate`.
        `hit_hash` is the content hash of the last reused block (-1 for a cold
        start). `can_allocate` already shrank the hit to a boundary that
        `_resumable_from` accepted, and that is not the same as "in HBM": the
        tier votes too. So a hash whose slot went to LMCache arrives as a miss
        and becomes a load (`_start_state_load`), and so does one whose bytes
        LMCache's own LRU has since dropped — which the tier cannot know until
        the fetch misses. The two are told apart by whether the tier still
        holds the hash, because only one of them may keep the boundary.

        A seq takes `state_slots_per_req` slots — one committed state plus a
        rollback slot per speculated token — however it starts. The checkpoint
        it resumes from is one slot wide, because speculation scratch is not
        state anybody resumes into, so a resume costs the pool exactly what a
        cold start does.

        Resuming shares: the checkpoint stays indexed and the request gets slots
        of its own, so a second request hitting the same prefix still finds it.
        How the state reaches the committed slot is the backend's
        `StateTransfer` — a fork reads the checkpoint for one forward, a copy is
        handed the bytes — and the two differ by one line here. When the pool
        cannot give a full set the request adopts the checkpoint as its
        committed slot instead: still correct, the state is exactly the one it
        wanted, it just spends the checkpoint rather than sharing it.

        A checkpoint is read-only, so several requests in one step may resume
        off the same one. The first takes it off the free list and the pin
        covers every reader until `release_state_pins`; a later one in that same
        step finds it already pinned and only needs slots to write into.
        Adopting is then off the table — the pin means someone else's forward
        still has to read it, or copy out of it.

        PAGE checkpoints gather into a fresh committed slot rather than being
        adopted: the bytes live in the KV pool, not in a state slot, so there is
        nothing here to hand over. Only fork checkpoints can be adopted.
        """
        width = self.state_slots_per_req
        if self.paged_state_checkpoints is not None:
            seq.state_slots = self.state.pop_many(width)
            seq.state_fork_src = -1
            if hit_hash == -1:
                return True  # cold start: nothing claimed, nothing to restore
            if self.paged_state_checkpoints.begin_restore(hit_hash, seq.state_slot):
                # Queued for the next batch, which gathers the image out of its
                # PAGE units into the committed slot. What the caller holds is
                # what it asked for.
                return True
            # HBM does not have it. That used to be an invariant violation and
            # this used to raise -- `can_allocate` had shrunk the hit to a
            # boundary the HBM index carried, so a miss here meant the gate and
            # the store disagreed.
            #
            # It is a normal path now, and it has to be: the gate consults the
            # CPU tier too, so a hash it accepted may live only there. Leaving
            # the raise in would take the engine down on the first request that
            # actually used the tier.
            if self._start_state_load(seq, hit_hash):
                return True
            # Neither tier can produce it. Reachable rather than defensive:
            # the tier's index is optimistic (`hashes` means "was stored once"),
            # and an HBM checkpoint can be unindexed between `can_allocate` and
            # here by another seq's `_fresh_block` in the same pass. Disown the
            # boundary -- blocks stay claimed and the forward recomputes.
            self.state_gate_lost_boundary += 1
            return False

        src = self.state.lookup(hit_hash) if hit_hash != -1 else -1
        if src < 0:
            wants_load = (
                hit_hash != -1
                and self.state_offload is not None
                and self.state_offload.could_serve(hit_hash)
            )
            seq.state_slots = self.state.pop_many(width)
            seq.state_fork_src = -1
            if wants_load and self._start_state_load(seq, hit_hash):
                return True
            # A fresh slot holds the previous occupant's bytes. That is fine
            # for a cold start (nothing claims otherwise) and wrong for a hit,
            # so the hit only survives if there was none to begin with.
            return hit_hash == -1
        # Being resumed from is the evidence a guessed position was right, so a
        # checkpoint that pays off stops being spent first — see
        # `StateSlotPool.mark_speculative`. Here rather than in `claim`, which
        # `_set_hash` also calls to re-file a slot that nobody read.
        self.state.promote(src)
        shared = self.state.is_pinned(src)
        if not shared:
            self.state.claim(src)
        if self.state.has_free(width):
            seq.state_slots = self.state.pop_many(width)
            seq.state_fork_src = src
            # Held off the free list until the forward that reads it is issued.
            self.state.pin(src)
            return True
        # `can_allocate` admitted this seq against a free list holding at least
        # `width`, and nothing else has run since, so the list can only be short
        # here if this seq's own `claim` above took the slot that made up the
        # count — which is `src`, unshared. Adopting it hands that slot straight
        # back to this seq, so the width is met.
        assert not shared, "no slot to resume into and the source is being read"
        self.state.invalidate(src)
        seq.state_slots = [src] + self.state.pop_many(width - 1)
        seq.state_fork_src = -1
        return True

    def _start_state_load(self, seq: Sequence, hit_hash: int) -> bool:
        """Ask the tier to fetch `hit_hash` into the slot `seq` just took.

        Only reached when the HBM index missed -- the case the tier exists for.
        The bytes land in the committed slot, where the resuming forward reads
        them, and the connector picks the load up off `seq.offload_joint` when
        it builds this step's metadata.

        `state_fork_src` stays -1. The loaded slot *is* the incoming state, and
        naming a source would send the forward to a different slot than the one
        being filled.

        False means the tier cannot serve it. `request_load` refuses an unknown
        hash because a load is resolved only by a report, so offering one for
        bytes no `get` can produce would park the request forever. The caller
        then disowns the boundary -- the pre-tier answer.
        """
        if self.state_offload is None:
            return False
        if not self.state_offload.request_load(seq.id, hit_hash, seq.state_slot):
            return False
        seq.offload_joint.load_hash = hit_hash
        return True

    def take_state_stores(self, max_inflight: int) -> list[tuple]:
        """`(operation, unit_ids)` for checkpoints to hand the CPU tier.

        Pins each as it hands it over; `settle_state_store` releases. Empty
        without a coordinator (the fork backends), which is correct rather than
        a gap: a fork checkpoint is a state slot, and #2045 is what moved K3's
        into the KV pool where a set of units can be read without being
        rescued first.
        """
        if self.paged_state_checkpoints is None or self.state_offload is None:
            return []
        if not self.state_offload.can_store:
            # A load-only role. Handing over a store pins units against a
            # report the worker's save half will never produce.
            return []
        out = self.paged_state_checkpoints.take_offload_stores(max_inflight)
        self.state_offload.stores_attempted += len(out)
        return out

    def settle_state_store(self, op, ok: bool, attempted: bool = True) -> None:
        """One store reported. Release the units, and index the hash if it landed.

        Success and failure release the pin identically -- it existed to keep
        the bytes still during the copy, and the copy is over either way.
        Only the indexing differs, because only a hash whose bytes are really
        there may be voted for.

        `attempted` is `False` for a store a connector *refused* to carry: the
        pin still has to be released here, but the copy never reached a worker,
        so it is not a `stores_failed` -- the caller counts the refusal once as
        `stores_refused`, and bumping `stores_failed` too would double-count the
        same event under two names. For the same reason it backs the optimistic
        `stores_attempted` bump `take_state_stores` made at hand-out back out, so
        a refusal does not linger in `attempted - completed - failed`.

        `op` is a `StateStoreOperationId`: the pin is released for that exact
        generation, so a late report from a superseded attempt settles nothing.
        The index is keyed by hash, because what a resume asks is whether the
        prefix is in LMCache -- not which attempt put it there.
        """
        reclaimed = False
        if self.paged_state_checkpoints is not None:
            reclaimed = self.paged_state_checkpoints.was_reclaimed(op)
            # Idempotent: normally the source release already returned these
            # units, and this is the backstop for a store that failed before
            # it ever read them.
            self.paged_state_checkpoints.settle_offload_store(op)
        if self.state_offload is None:
            return
        if ok and reclaimed:
            # The stale reclaimer took this store's units back before it
            # reported, and nothing can say whether the worker had stopped
            # reading them. If it had not, the pool may have handed them to
            # another request whose writes the gather picked up, making the CPU
            # image a mix of two prefixes filed under the first one's hash --
            # and a resume onto that is silent wrong output. The bytes may well
            # be fine; there is no way to know, so they are forfeited.
            self.state_offload.stores_untrusted += 1
            logger.warning(
                "state offload: %s reported stored, but its units were "
                "reclaimed while it ran; refusing to index it.",
                op,
            )
            return
        if ok:
            self.state_offload.stores_completed += 1
            self.state_offload.note_stored(int(op.prefix_hash))
        elif attempted:
            self.state_offload.stores_failed += 1
        else:
            # A refusal. `take_state_stores` bumped `stores_attempted` when it
            # handed this op over, but it never reached a worker, so back that
            # out: the event is counted once as `stores_refused`, and leaving it
            # in `stores_attempted` would read as permanently in-flight in the
            # `attempted - completed - failed` funnel.
            self.state_offload.stores_attempted -= 1

    def release_state_store_source(self, op) -> None:
        """The GPU has finished reading this store's PAGE units; hand them back.

        Separate from `settle_state_store` because the two answer different
        questions: the units are the KV pool's and are free as soon as the D2H
        drains, while whether the CPU put succeeded is decided afterwards and
        cannot touch them. Holding an image out of the pool across that would
        cost reuse for nothing.

        Phase one of the two-phase pin: it releases the units but leaves the
        operation pinned as dispatched-but-unreported, so `has_offload_pins`
        (and the liveness gate that ORs it in) keeps the engine polling until
        the index report actually lands. Retiring the pin here -- as it once
        did -- let liveness read idle while the store's report still sat
        undrained in the worker, so `note_stored` never ran and the checkpoint
        was never votable.
        """
        if self.paged_state_checkpoints is not None:
            self.paged_state_checkpoints.release_offload_store_source(op)

    def has_pending_state_store_pins(self) -> bool:
        """Whether a dispatched state store is still awaiting its report.

        The engine's liveness gate (`EngineCore.has_pending_kv_work`) ORs this
        in so a step whose only outstanding work is a state store already handed
        to the worker does not read as idle. Without it the store's PAGE units
        stay pinned out of the KV pool until they age out, because the report
        that would release them -- and the reclaim that is the report's only
        backstop -- are both drained by `_poll_kv_transfer_progress`, which the
        idle loop stops calling the moment liveness goes False. This is the
        unit-store twin of the `deferred_free_blocks` condition the gate already
        checks for the KV save leg.
        """
        if self.paged_state_checkpoints is None:
            return False
        return self.paged_state_checkpoints.has_offload_pins()

    def reclaim_stale_state_store_pins(self, timeout_s: float) -> int:
        """Release store pins whose report never came. See the store."""
        if self.paged_state_checkpoints is None:
            return 0
        return self.paged_state_checkpoints.reclaim_stale_offload_pins(timeout_s)

    def _chain_parent_hash(self, seq: Sequence, start: int) -> int | None:
        """Return the chained hash of block ``start - 1``, or ``None`` on a gap.

        All source paths (register_prefill_hashes, postprocess, offload wake)
        are expected to hash blocks before this is called. A gap means a
        source-level bug; callers skip the range rather than mint false hashes.
        """
        if start <= 0:
            return -1
        h = self.kv.block(seq.block_table[start - 1]).hash
        if h != -1:
            return h
        logger.error(
            "Unhashed parent block %d for seq %s — skipping hash "
            "registration for blocks %d onward",
            start - 1,
            seq.id,
            start,
        )
        return None

    def hash_blocks(
        self,
        seq: Sequence,
        num_new_tokens: int,
        start_tokens: int | None = None,
        next_forward_tokens: int | None = None,
        aimed: bool = True,
    ) -> None:
        """Register hashes for blocks finalized by the most recent step.

        Called from scheduler.postprocess() after the forward completes, so a
        block's hash is only published once its KV is actually computed. The
        `[start, end)` range covers blocks fully filled by this step:
          start = first block whose first token was at num_cached_tokens
          end   = first block not yet fully filled (excludes the partial one)
        Caller passes `num_new_tokens` = tokens forwarded in this step. For
        single-shot prefill that's `seq.num_tokens - seq.num_cached_tokens`;
        chunked prefill will pass the per-chunk count.

        `start_tokens` overrides the token offset the range starts at. Pipeline-
        parallel schedule-time advancement already bumped seq.num_cached_tokens
        past this chunk, so the head passes the chunk's pre-advance offset here.

        `next_forward_tokens` reaches `checkpointers_at`; see there. Left
        unset it reads the prompt's remainder, which is the prefill answer.
        """
        if not self.enable_prefix_caching:
            return
        hbs = self._hash_block_size()
        base = seq.num_cached_tokens if start_tokens is None else start_tokens
        start = base // hbs
        end = (base + num_new_tokens) // hbs
        # A finished or preempted seq has had its block table released; the
        # deferred publish paths can still reach it with a stale token count.
        end = min(end, len(seq.block_table))
        if start >= end:
            return
        h = self._chain_parent_hash(seq, start)
        if h is None:
            return
        # Watermark for the decode-side continuation, maintained here so every
        # prefill path feeds it without knowing about it.
        seq.num_hashed_tokens = max(seq.num_hashed_tokens, end * hbs)
        record = self._event_log is not None
        store_run_parent: int | None = h if h != -1 else None
        store_run_hashes: list[int] = []
        store_run_tokens: list[int] = []
        for i in range(start, end):
            token_ids = self._hash_block_tokens(seq, i)
            h = self.compute_hash(token_ids, h)
            self.kv.publish(seq.block_table[i], h, token_ids)
            if record:
                store_run_hashes.append(h)
                store_run_tokens.extend(token_ids)
        if record and store_run_hashes:
            self._event_log.append(
                _make_block_stored(
                    store_run_hashes,
                    store_run_tokens,
                    store_run_parent,
                    self.hash_block_size,
                )
            )
        pos = base + num_new_tokens
        kept = self.checkpointers_at(seq, pos, next_forward_tokens, aimed)
        for cache in kept:
            cache.checkpoint(seq, end, h)
        if kept:
            seq.last_checkpoint_pos = pos
        # The midstep half of the same moment. Here rather than at each of the
        # three prefill call sites because this is what they have in common:
        # the forward for `[base, base + num_new_tokens)` has completed, which
        # is precisely when the reserved bytes exist. A no-op with nothing
        # reserved, so decode and unreadable backends pay a branch.
        self.commit_midstep(seq)

    def hash_decode_blocks(
        self, seq: Sequence, committed_kv_len: int, next_forward_tokens: int = 0
    ) -> None:
        """Register hashes for generated blocks filled up to `committed_kv_len`.

        `may_append` allocates decode blocks without hashing them: at allocation
        time their tokens have not been sampled, and under speculative decoding
        part of what the forward writes is about to be rejected.

        `committed_kv_len` counts the tokens for which neither still applies —
        id final, KV computed — and is a hard line, not a hint. It stops short
        of any token no forward has read yet: that token's KV slot is written
        by the next forward, and a block published over an unwritten slot hands
        a later request KV that may never arrive at all (the seq can finish
        first). Prefill's `hash_blocks` draws the same line from its own side,
        at `num_cached_tokens + chunk`.

        Without this the prefix cache indexes prompt blocks only, and a
        follow-up turn — previous prompt plus previous answer — matches nothing
        beyond the original prompt.

        `next_forward_tokens` reaches `checkpointers_at`; see there. It
        defaults to "no next forward", i.e. hash but never checkpoint, so a
        caller opts into decode-point checkpointing rather than out of it.
        """
        if not self.enable_prefix_caching:
            return
        base = seq.num_hashed_tokens
        if committed_kv_len > base:
            self.hash_blocks(
                seq,
                committed_kv_len - base,
                start_tokens=base,
                next_forward_tokens=next_forward_tokens,
                # Generation cannot choose where a step ends, least of all a
                # speculative one, so it is held to spacing rather than to the
                # grid.
                aimed=False,
            )

    def cancel_state_fork(self, seq: Sequence) -> bool:
        """Undo a pending fork by adopting its source slot.

        Called when the forward that was going to carry the fork turns out too
        short to fill a fresh slot (`min_fork_tokens`). Both flavours collapse
        to the same move — take the source over as this seq's committed slot and
        spend its checkpoint: a resume becomes the non-sharing hit, a checkpoint
        becomes no checkpoint at all. Only the committed slot changes hands; the
        speculation scratch stays where it is, since the source never carried
        any.

        Returns False when the source cannot be taken over because another
        request in this same step forks off it too: adopting means writing into
        a slot that request's forward still has to read. The caller keeps the
        fork instead and must not shorten the forward below `min_fork_tokens`.
        """
        src = seq.state_fork_src
        if src < 0:
            return True
        if self.state.pin_count(src) > 1:
            return False
        self.state.release(seq.state_slot)
        self.state.invalidate(src)
        # Both flavours of source are pinned — held off the free list for the
        # forward that has to read them — so taking one over is just dropping
        # this request's claim on it. It used to matter which flavour it was:
        # `checkpoint` handed its source straight back, so adopting it meant
        # claiming it off the free list, and `pin_count` then undercounted the
        # readers this refuses to overwrite.
        self.state.unpin(src)
        seq.state_slot = src
        seq.state_fork_src = -1
        return True

    def checkpoint_limit(self, seq: Sequence) -> int:
        """Rightmost prompt position any state class may checkpoint at, 0 none.

        `checkpointers_at` solved for prefill: the last rung of the ladder that
        still leaves the widest-reaching class its `successor_room` of prompt to
        forward. Kept as its own method because the scheduler needs the bound up
        front, to cut prefill chunks so they land on the ladder.

        0 means the grid places no rung on this prompt — every prompt shorter
        than one interval, among others. It does not mean the seq keeps
        nothing: a demand rung sits outside the grid and `checkpoint_cut`
        takes it either way.
        """
        interval = self.state_checkpoint_interval_tokens
        if interval <= 0:
            return 0
        # The smallest room reaches furthest right, and `inf` — no class can
        # checkpoint this seq at all — falls out as 0 without a special case.
        room = min(
            (c.successor_room for c in self.state_caches if c.applies(seq)),
            default=inf,
        )
        if isinf(room):
            return 0
        return max(int((seq.num_prompt_tokens - room) // interval) * interval, 0)

    def _record_checkpoint_demand(
        self,
        seq: Sequence,
        hit: int,
        compressed_hit: int,
        block_hashes: list[int],
        live_blocks: int,
        protected_hash: int | None,
    ) -> None:
        """Ask the hit counterfactually, and turn the gap into a rung.

        Whenever the gates cut a hit short, the same question is worth asking a
        second time with every ladder dense: how far would it have reached? What
        that recovers is reuse being declined only because nobody checkpointed
        there. What it does not recover is gone whatever anybody stores. The two
        land in `num_wanted_hit_blocks` (which `EngineStats` splits the declined
        reuse by) and `checkpoint_demand_pos` (which the ladder acts on).

        The demand is a rung of this seq's own, off the interval grid, and the
        seq that found the gap is the one best placed to fill it: it collects
        none of that reuse and has to compute the prefix regardless.

        Decided here, with both numbers in hand, rather than by the readers:
        `hit` survives only as `seq.num_cached_tokens`, which the scheduler
        advances as chunks land — under pipeline parallelism it is already past
        this chunk by the time `hash_blocks` runs, so a reader comparing against
        it would drop the demand on exactly the forward that was cut for it.

        A demand is not measured against the interval. The grid guesses
        where reuse will resume; a demand is reuse that was asked for and
        refused, and the granularity of the guess is no reason to discard it —
        gating one by the other left every prompt shorter than an interval
        declining all the reuse it had. The position comes from the same
        forkable test as the hit, so it always satisfies
        `num_prompt_tokens - pos >= successor_room`: somebody can really keep it.

        The shape that pays for this is a template header whose checkpoint is
        invalidated before anyone reaches it — there each request cuts a chunk
        and none collects. What bounds that is convergence rather than a
        threshold: found once, filled once, gone. `chunks_cut_for_demand`
        against `demands_recorded` is where it would show if it did not.

        `--no-state-checkpoint-demand` drops the rung and leaves the prompt-end
        anchor alone. Worth measuring: the rung is most of the write traffic
        and little of the read-back (`StateSlotPool.mark_speculative` has the
        numbers), and each write evicts something. Whether *removing* those
        writes beats merely demoting them is the open question — the CPU replay
        scores the two identically, which is the signature of a harness that
        models eviction order but not the cost of the write.
        """
        wanted = (
            self._gated_hit(seq, compressed_hit, block_hashes, assume_checkpointed=True)
            if hit < compressed_hit
            else hit
        )
        seq.num_wanted_hit_blocks = wanted
        # Zero switches checkpointing off entirely — `checkpointers_at` keeps
        # nothing then, so a cut for a demand would buy nothing either. -1 is
        # the other thing: the interval grid is off but checkpointing is on, and
        # the demand rung is one of the two placements that then carries it.
        interval_on = self.state_checkpoint_interval_tokens != 0
        demand = (
            wanted * self.hash_block_size
            if interval_on and self.state_checkpoint_demand and wanted > hit
            else 0
        )
        # A demand is an instruction to cut a prefill chunk onto a rung, and
        # that cut costs the request a forward. Buying one for a store
        # `begin_store` is about to refuse is the only part of this funnel
        # that is pure loss — the attribution above stays either way, because
        # the reuse really was declined for want of a checkpoint.
        #
        # Asked afresh on every attempt, because that is the question: a
        # demand recorded while the pool had room is not still affordable once
        # it does not, and letting the earlier answer stand is exactly the cut
        # this gate exists to withhold. What must not repeat is the *counting*,
        # which is why the seq carries its own marker rather than the gate
        # reading the position it is about to overwrite.
        #
        # Asked with this admission's own blocks included, because they are
        # taken first: a pool with room for an image but not for the request
        # *and* the image would answer yes here and refuse at `begin_store`,
        # with the cut already bought and the funnel showing nothing.
        #
        # It is still a sample. The store happens many forwards later, at the
        # rung this cut creates, against a pool that has moved since — no
        # question asked here can be the one `begin_store` asks. What this
        # gate removes is the loss that was knowable at admission;
        # `checkpoints_dropped` is what counts the rest, and the two are meant
        # to be read together.
        if demand and not self._checkpoint_has_room(live_blocks, protected_hash):
            self.demands_declined_no_room += not seq.checkpoint_demand_declined
            seq.checkpoint_demand_declined = True
            demand = 0
        seq.checkpoint_demand_pos = demand
        # Counted when the demand first appears rather than once per attempt —
        # otherwise one deferred request inflates the denominator the
        # convergence check above is read against. A separate marker from the
        # decline above: a decline zeroes the position, so the position alone
        # would let a recorded demand be counted twice the next time the pool
        # has room.
        if demand:
            self.demands_recorded += not seq.checkpoint_demand_counted
            seq.checkpoint_demand_counted = True

    def _record_checkpoint_end(self, seq: Sequence) -> None:
        """Reserve this prompt's own end as a resume point.

        The ladder guesses where reuse will resume and the demand learns it one
        refusal late. Neither covers the case that dominates agentic traffic: a
        conversation whose next turn resends this whole prompt and continues
        past it. That resumes at *this* prompt's end, and the only request in a
        position to leave a checkpoint there is this one.

        Measured on the SemiAnalysis cc-traces (4,808 resumes with a nonzero KV
        hit): 93.5% land on a previous prompt end, 0.0% on the 8192 ladder.

        Off the grid by construction, like a demand: `checkpoint_limit` bounds
        the *ladder*, and a prompt shorter than one interval places no rung at
        all while still being a perfectly good thing to resume from. The
        `successor_room` test cannot be skipped though — it is what makes the
        position keepable, and `checkpointers_at` re-checks it against the
        forward that actually lands. Applying it here as well keeps the cut and
        the keep from disagreeing, which is the standing contract between them.
        """
        seq.checkpoint_end_pos = 0
        if not self.state.enabled:
            return
        # 0 is the off switch for checkpointing as a whole, not just the grid,
        # and `checkpointers_at` enforces it by refusing every position. An
        # anchor recorded here anyway would still be cut for — a shortened
        # prefill chunk on every prompt, storing nothing, with no error to
        # show for it. Pinned by `test_interval_zero_anchors_nothing`.
        if self.state_checkpoint_interval_tokens == 0:
            return
        room = min(
            (c.successor_room for c in self.state_caches if c.applies(seq)),
            default=inf,
        )
        if isinf(room):
            return
        # The rightmost grid position that still leaves `room` behind it.
        #
        # Not simply `num_prompt_tokens` floored: a checkpoint at P binds the
        # forward after it to carry `room` tokens, and the floored end leaves
        # at most `hash_block_size - 1`. So for any class whose room reaches a
        # block or more — MIN_FORK 8 against block 4 in the tests, V4's 131
        # against 256 — the exact end is never keepable and an anchor placed
        # there would be cut for and then refused by `checkpointers_at`.
        #
        # Stepping back to the last keepable boundary costs at most one block
        # of the next turn's reuse and is what makes the anchor exist at all.
        # Measured on the cc-traces, the step-back anchor is worth +0.3 to +0.7
        # points over insisting on the exact end, and is never worse.
        end = (
            (seq.num_prompt_tokens - int(room)) // self.hash_block_size
        ) * self.hash_block_size
        # And never past the last block anybody can match on. `can_allocate`
        # stops one block short of the prompt (prefill must forward a block to
        # produce logits), so a checkpoint filed under the final block's hash is
        # one no scan ever looks up. For the continuation this anchor is aimed
        # at — the next turn resends this prompt and keeps going — that block is
        # interior and would match, which is why the ceiling is not obvious. But
        # a *re-sent identical* prompt caps at `n - 1`, and there the uncapped
        # anchor is not merely useless: it is stored, so it evicts the ladder
        # rung that would have served the resume, and the hit goes to 0 rather
        # than merely getting no better. Capping costs the continuation at most
        # one block of reuse — the same trade `room` above already takes.
        end = min(end, (self._n_hash_blocks(seq) - 1) * self.hash_block_size)
        if end > 0:
            seq.checkpoint_end_pos = end

    def _extend_hash_chain(self, seq: Sequence, block_hashes: list[int]) -> None:
        """Cache the chained hash of *every* block of the prompt on `seq`.

        `block_hashes` stops at the first cache miss — it is the *hit*, and the
        gates read it as one. A midstep reservation names a position the forward
        has not reached yet, which is past that miss for every reservation worth
        making, so the chain it is named by has to run further.

        A separate list rather than an extension of `block_hashes` in place.
        Nothing downstream would misread a longer list today — `_gated_hit` and
        the free-pool loop both index it under `compressed_hit` — but "the
        blocks that hit" and "the blocks of the prompt" are two facts, and one
        list holding both means the next reader of either has to know which. It
        is also cheap: this runs after the gates, so the copy is of a list the
        callers are already done with.

        Only for a backend that reserves midstep. Everywhere else the hash a
        checkpoint is filed under is computed by `hash_blocks` walking the
        blocks the forward just finished, and this would be a hash pass over the
        whole prompt on every admission, spent on nothing.
        """
        if not self.state.readable_midstep or not self.state.applies(seq):
            seq.block_hashes = []
            return
        # The continuation itself -- resume from the last APPENDED hash (on a
        # miss the loop's `h` holds the hash of the block that failed to match,
        # which was never appended and is not part of this chain) -- is exactly
        # `_chain_to`, run to the full prompt width. Delegate rather than
        # duplicate the loop; the only thing local to this path is the midstep
        # gate above and writing the result to `seq`.
        seq.block_hashes = self._chain_to(seq, block_hashes, self._n_hash_blocks(seq))

    def midstep_positions(self, seq: Sequence, start: int, end: int) -> list[tuple]:
        """`(position, hash)` for every checkpoint a forward over `(start, end]`
        can take without being cut short.

        The midstep twin of `checkpoint_cut`. Same candidate positions — the
        grid rung, the demand, the prompt-end anchor — but where `checkpoint_cut`
        must pick *one* and shorten the chunk onto it, this returns all of them:
        a backend that reads its chunk kernel's interior states takes a snapshot
        at each without the forward ending there.

        `checkpointers_at` is not consulted. That function answers "the forward
        ended here, keep or not", and its `successor_room` test is about a fork's
        successor forward — which a midstep snapshot does not have, because
        nothing is handed over and the owner keeps writing where it is. Applying
        it here would refuse exactly the block-aligned prompt end the anchor
        exists to reserve.

        Every candidate is a multiple of `hash_block_size` by construction: the
        interval is snapped onto that grid at construction and the rung is a
        multiple of the interval, the demand is `blocks * hash_block_size`, and
        the anchor is floored to it. The assertion below states that rather than
        trusting it, because a position off the grid indexes the wrong block's
        hash — a checkpoint filed under a name no resumer will ever compute, and
        nothing downstream that could notice.

        No `interval == 0` early-out, unlike the three placements this reads.
        Each already carries the off switch — `checkpoint_limit` returns 0 for
        any non-positive interval, `_record_checkpoint_demand` gates on
        `interval_on`, `_record_checkpoint_end` returns before setting the
        anchor — so under 0 all three candidates are 0 and the loop has nothing
        to walk. A fourth check would read as the one enforcing it, and be the
        one nobody dared remove. `test_interval_zero_reserves_nothing` pins the
        behavior end to end rather than the check.
        """
        if not self.state.readable_midstep or not self.state.applies(seq):
            return []
        interval = self.state_checkpoint_interval_tokens
        hbs = self.hash_block_size
        hashes = seq.block_hashes
        rung = 0
        if limit := self.checkpoint_limit(seq):
            rung = min(end, limit)
            rung -= rung % interval
        out = []
        for pos in sorted(
            {p for p in (rung, seq.checkpoint_demand_pos, seq.checkpoint_end_pos) if p}
        ):
            if not start < pos <= end:
                continue
            assert not pos % hbs, f"checkpoint position {pos} is off the {hbs} grid"
            nblocks = pos // hbs
            # A position past the chain is one `_extend_hash_chain` could not
            # name — nothing to file it under, so it cannot be found again.
            if nblocks > len(hashes):
                continue
            out.append((pos, hashes[nblocks - 1]))
        return out

    def plan_midstep(self, seq: Sequence, start: int, end: int) -> None:
        """Reserve a destination slot for every checkpoint this forward covers.

        Called once the chunk `(start, end]` is settled and before the batch is
        built, which is the only window where both are true: the positions are
        known, and the free list can still be committed against without racing
        an admission later in the same scheduling pass.

        Overwrites rather than accumulates. A reservation is good for exactly
        one forward, so a second plan for the same seq means the first forward
        never ran — hand its slots back rather than leak them.
        """
        if not self.state.readable_midstep:
            return
        if seq.midstep_reservations:
            self.cancel_midstep(seq)
        seq.midstep_reservations = self.state.reserve_midstep(
            seq, self.midstep_positions(seq, start, end)
        )

    def commit_midstep(self, seq: Sequence) -> None:
        """File this forward's reservations, now that its bytes exist.

        Drains the list: whatever is not committed here is not a checkpoint,
        and leaving it in place would let the next `plan_midstep` publish a
        position this seq has already moved past.
        """
        if not seq.midstep_reservations:
            return
        self.state.publish_midstep(seq.midstep_reservations, seq)
        # The rightmost published position, for the decode-side spacing rule in
        # `checkpointers_at`. Generation is never midstep-readable — its
        # forwards end where acceptance puts them — so it still measures from
        # here, and would otherwise re-checkpoint immediately after a prompt
        # that just filed one.
        seq.last_checkpoint_pos = max(
            seq.last_checkpoint_pos, max(p for _g, p, _h in seq.midstep_reservations)
        )
        seq.midstep_reservations = []

    def cancel_midstep(self, seq: Sequence) -> None:
        """Hand back reservations whose forward is not going to run."""
        if not seq.midstep_reservations:
            return
        self.state.cancel_midstep(seq.midstep_reservations)
        seq.midstep_reservations = []

    def checkpoint_cut(self, seq: Sequence, start: int, end: int) -> int:
        """Earliest ladder position in `(start, end]`, or 0 if there is none.

        What a prefill chunk is cut at so its forward lands exactly on a rung.
        The counterpart of `checkpointers_at`, which decides what a forward
        ending there keeps: the two have to agree position for position, so the
        grid arithmetic lives here rather than at the scheduler's call site.

        Earliest, not latest, because a prompt can now want two positions and
        the chunk loop reaches the second one only by being handed the first.
        Within the grid a later rung still dominates an earlier one — one class,
        one resume point, take the rightmost — so the grid collapses to a single
        candidate before the choice is made. The prompt-end anchor is the
        exception that forces the loop: it does not dominate the rung and the
        rung does not dominate it. See `_record_checkpoint_end`.

        0 once every class that applies reads its checkpoints midstep. The cut
        exists only to put a forward's *end* on a rung, and a readable class
        does not need the end — `midstep_positions` hands it the same positions
        to snapshot inside a full-length chunk. Every, not any: one unreadable
        class still needs the forward to land there, and the readable ones lose
        nothing by being handed a position they would have taken anyway.

        This gate and the one in `checkpointers_at` are a pair and change
        together. Suppressing the cut alone leaves `checkpointers_at` refusing
        every off-grid position it is then handed — zero checkpoints kept, no
        error, no warning, and a hit rate that quietly collapses.
        """
        applicable = [c for c in self.state_caches if c.applies(seq)]
        if all(c.readable_midstep for c in applicable):
            return 0
        rung = 0
        if limit := self.checkpoint_limit(seq):
            # `limit` is itself a multiple of the interval, so a chunk cut at
            # it needs no special case.
            rung = min(end, limit)
            rung -= rung % self.state_checkpoint_interval_tokens
        # A demand is capped by neither the grid nor `limit`. `limit` is the
        # last position on the *grid* that leaves the widest class its room; a
        # demand carries that room by construction, so it may sit to the right
        # of the last rung — or, on a prompt too short for the grid to place a
        # rung at all, be the only position either side has.
        demand = seq.checkpoint_demand_pos
        # The prompt-end anchor sits outside the grid for the same reason and
        # on the same terms — see `_record_checkpoint_end`, which applies the
        # `successor_room` test that makes it keepable.
        anchor = seq.checkpoint_end_pos
        # The earliest candidate strictly inside the chunk. Taking the latest
        # instead is what a single-position ladder could afford and this one
        # cannot: with the anchor at 36 and a rung at 32, `max` cuts at 36 and
        # the forward never ends at 32, so the rung is not merely deferred —
        # it is never kept at all. A class the anchor is out of reach for then
        # loses the rung it used to resume from, permanently, and the demand
        # that fires in its place loses the same comparison on every following
        # request: reuse falls to zero and stays there while the demand counter
        # climbs. Pinned by
        # `test_reuse_another_class_declines_is_not_charged_to_the_ladder`.
        candidates = [p for p in (rung, demand, anchor) if p and start < p <= end]
        if not candidates:
            return 0
        target = min(candidates)
        # Beating the grid means an off-grid position chose this cut and the
        # grid would not have. `target < end` is the other half: at `end` the
        # chunk is not shortened and the cut cost nothing, and counting those
        # made the funnel report cuts that never happened — which is the one
        # number meant to expose a shape that pays per request and never
        # converges.
        #
        # Attributed to whichever off-grid position actually chose `target`:
        # the anchor fires on nearly every prompt, so folding its cuts into
        # `chunks_cut_for_demand` would swamp the convergence signal that
        # counter exists to expose. The demand is checked first because when
        # the two coincide it is the demand that evidenced the position.
        if target != rung and target < end:
            if target == demand:
                self.chunks_cut_for_demand += 1
            else:
                self.chunks_cut_for_end += 1
        return target

    def checkpoint_funnel(self) -> dict[str, int]:
        """Every stage a wanted checkpoint passes through, in order.

        Assembled here because the stages live in two objects — the ladder
        decides what to ask for, the pool decides what survives — and a reader
        needs them side by side to tell which stage lost it.
        Pool-level fates are collected via ``state_checkpoint_fates()`` so that
        a second state class is automatically included — calling
        ``self.state.checkpoint_fates()`` directly would miss it.
        """
        return {
            "demands_recorded": self.demands_recorded,
            "demands_declined_no_room": self.demands_declined_no_room,
            "chunks_cut_for_demand": self.chunks_cut_for_demand,
            "chunks_cut_for_end": self.chunks_cut_for_end,
            "joint_boundaries": self.joint_boundaries,
            "state_hbm": self.state_hbm_boundaries,
            "state_tier": self.state_tier_boundaries,
            "state_gate_lost_boundary": self.state_gate_lost_boundary,
            # Off the index, which owns an orphaned load's slot from the moment
            # its request is torn down until the report lands.
            "orphan_load_slots_reclaimed": (
                self.state_offload.orphan_load_slots_reclaimed
                if self.state_offload is not None
                else 0
            ),
        } | self.state_checkpoint_fates()

    def pool_pressure(self) -> dict[str, int]:
        """Both pools' eviction counts and occupancy, side by side.

        The pair is the point. A hit rate says reuse was lost but not which
        pool lost it, and the two are sized against each other out of one
        budget (`plan_pools`) — so the actionable reading is always a
        comparison: paged evicting while state sits mostly vacant means the
        split is wrong, both evicting means the budget is.

        The fates come from `state_checkpoint_fates()`, the aggregator, not
        from `self.state` and not from a single `_state_checkpoint_cache`.
        Under PAGE `self.state` is built with `StateTransfer.none()` and never
        sees a `checkpoint()`, so reading it here printed four zeros for the
        life of the server; and reading `_state_checkpoint_cache` alone omits
        any second state class the aggregator folds in. Either way the number
        disagreed with the `checkpoint_funnel` line -- which already goes
        through the aggregator -- and two outputs with the same metric names
        disagreeing is how a tuning session concludes checkpointing never
        fires. Occupancy stays on `self.state`: that is the slot pool either
        way, and the coordinator has none.
        """
        # `state_checkpoint_fates()`, matching `checkpoint_funnel`: it sums
        # every state class, so a build with a second class is not silently
        # undercounted here while the funnel line reports the full total.
        # `occupancy()` stays on the pool, which is the thing that has slots.
        return (
            self.kv.eviction_stats()
            | self.state.occupancy()
            | self.state_checkpoint_fates()
        )

    def checkpointers_at(
        self,
        seq: Sequence,
        pos: int,
        next_forward_tokens: int | None = None,
        aimed: bool = True,
    ) -> list[StateCache]:
        """State classes that should keep a checkpoint at `pos`, in class order.

        A ladder of resume points, one every `state_checkpoint_interval_tokens`
        of context, shared by every class. Keeping one is capacity-neutral for a
        rolling class (the slot handed away is replaced from the free list) and
        capacity-bounded for an immutable one (an LRU-capped pin), but either
        way it costs the *keeper* an extra forward — its prompt gets cut at the
        rung — so the interval is what keeps that cost amortized instead of
        per-request.

        `next_forward_tokens` is how many tokens the forward right after this
        one carries, and is what each class's `successor_room` is compared
        against. Unset means the prompt's remainder, the prefill answer; decode
        passes one. Everything else follows from that one number — a class
        needing a long hand-over (V4's ring, 131) simply never qualifies
        mid-generation, one that hands nothing over (a retaining SWA pool, 0)
        always does, one that cannot keep a checkpoint at all (`inf`) never
        does, and a request stopping on this step passes 0 and keeps nothing
        that nothing will ever resume from.

        The position must be exact. A checkpoint holds state as of the forward's
        last token, so a forward that overshoots a rung is ahead of the hash it
        would be filed under; the scheduler cuts prefill chunks to land here,
        and a path that doesn't simply keeps nothing.

        On top of the grid sit two rungs of this seq's own.
        `checkpoint_demand_pos` is a boundary this seq was denied for want of a
        checkpoint (`_record_checkpoint_demand`); `checkpoint_end_pos` is this
        prompt's own end, which is the position agentic traffic actually resumes
        at (see `_record_checkpoint_end`). Both
        are admitted on the same terms: off the grid, and still subject to the
        `successor_room` test below — which `_record_checkpoint_end` pre-checks,
        so the cut and the keep cannot disagree. `checkpoint_cut` reads the same
        two fields, so the cut and the keep cannot drift apart.

        `aimed` says whether the caller could place the forward's end. Prefill
        can — `checkpoint_cut` shortens the chunk — so it is held to the exact
        grid and the two agree position for position. A speculative decode step
        cannot: it commits `1 + accepted`, so it steps over most rungs, and
        holding it to the grid made a decode checkpoint a one-in-`toks/fwd`
        chance. Bounding the drafts would land it there, at the price of
        throwing away speculation and under-reporting the acceptance rate,
        which counts drafts offered as `mtp_k` regardless.

        Nothing needs that price. The grid exists to space checkpoints out, and
        a step that lands on any hash-block boundary far enough past the last
        one serves the purpose exactly as well — the position only has to be
        *findable*, and a resumer finds it by hash, never by arithmetic. So an
        unaimed caller is held to the spacing rather than the grid.

        What that buys scales with how many boundaries a rung spans. A step
        lands on any given boundary with probability `1 / toks_per_forward`, so
        the chance of keeping a checkpoint per rung is
        `1 - (1 - 1/toks_per_forward) ** (interval / hash_block_size)`: at
        DeepSeek-V4's 256-token block and 4.3 tokens a forward, 23% when the
        interval is one block and effectively certain at the 8192 default. The
        two rules coincide exactly when the interval *is* the block, which is
        also the finest grid V4 admits — so a test at that setting measures
        nothing, and `demand_config` exists to avoid it.

        The demand rung is absent from that branch by construction, not by
        omission: a demand is at most the prompt's own hit ceiling, and every
        unaimed position is at or past the end of the prompt, so generation
        cannot reach one. `checkpoint_end_pos` is absent for a near-identical
        reason: it is the prompt's end stepped back to the grid, and an unaimed
        position at or past the prompt end is past it too.
        """
        interval = self.state_checkpoint_interval_tokens
        if interval == 0 or pos <= 0:
            return []
        # -1: no grid, so `pos % interval` has nothing to say and asking it
        # would be a ZeroDivisionError's less honest cousin — -1 divides
        # everything, admitting every position as a rung. The demand and the
        # anchor are then the only two placements, which is the whole point.
        on_grid = interval > 0 and not pos % interval
        if aimed:
            if (
                not on_grid
                and pos != seq.checkpoint_demand_pos
                and pos != seq.checkpoint_end_pos
            ):
                return []
        elif interval < 0:
            # Decode-side spacing has no grid to fall back on either. The two
            # aimed placements are both prompt positions, so nothing here is
            # reachable and generation keeps no checkpoints under -1.
            return []
        elif pos % self.hash_block_size or pos - seq.last_checkpoint_pos < interval:
            return []
        if next_forward_tokens is None:
            next_forward_tokens = seq.num_prompt_tokens - pos
        return [
            c
            for c in self.state_caches
            # A readable class checkpoints its prompt through
            # `plan_midstep`/`commit_midstep` instead, so keeping it here as
            # well would keep the same boundary twice. Both halves of that are
            # damage: two slots spent on one hash, the loser orphaned free but
            # unindexed — and, under `fork`, the seq hands its live slot away
            # and takes a fresh one, binding the next forward to refill a
            # replacement it had no reason to need.
            #
            # Only on the aimed path. Midstep positions are all prompt
            # positions, so generation is where `aimed=False` puts every class
            # back on this one. The twin of the `checkpoint_cut` gate; see
            # there for why the two cannot move separately.
            if not (aimed and c.readable_midstep)
            and c.applies(seq)
            and next_forward_tokens >= c.successor_room
        ]

    def publish_loaded_prefix(
        self,
        seq: Sequence,
        start_token: int,
        end_token: int,
    ) -> int:
        """Publish a successfully loaded offload prefix into the GPU cache index.

        LMCache restores KV directly into already allocated physical blocks, so
        those blocks do not pass through ``hash_blocks()``. Without explicitly
        publishing them here, the current request can consume the restored KV,
        but later requests cannot discover it through ``can_allocate()`` and
        repeatedly load the same prefix from CPU.

        Only complete, hash-block-aligned loaded blocks are published. Existing
        canonical mappings win: concurrent requests may load the same prefix
        into different physical blocks, and replacing the canonical mapping
        would make its eventual eviction remove the wrong cache entry.
        """
        if not self.enable_prefix_caching:
            return 0

        start_token = max(0, int(start_token))
        end_token = min(int(end_token), int(seq.num_prompt_tokens))
        if end_token <= start_token:
            return 0
        hbs = self._hash_block_size()
        if start_token % hbs != 0:
            logger.warning(
                "Cannot publish offload prefix with unaligned start: "
                "seq=%s start=%d hash_block_size=%d",
                seq.id,
                start_token,
                hbs,
            )
            return 0

        start_block = start_token // hbs
        end_block = end_token // hbs
        if end_block <= start_block:
            return 0
        if end_block > len(seq.block_table):
            logger.warning(
                "Cannot publish offload prefix beyond block table: "
                "seq=%s end_block=%d blocks=%d",
                seq.id,
                end_block,
                len(seq.block_table),
            )
            return 0

        parent_hash = self._chain_parent_hash(seq, start_block)
        if parent_hash is None:
            return 0

        indexed_tokens = 0
        for i in range(start_block, end_block):
            token_ids = self._hash_block_tokens(seq, i)
            block_id = seq.block_table[i]
            block = self.kv.block(block_id)
            block_hash = self.compute_hash(token_ids, parent_hash)
            canonical_id = self.kv.lookup(block_hash)

            if block.hash not in (-1, block_hash):
                logger.warning(
                    "Refusing to overwrite indexed block during offload "
                    "promotion: seq=%s block=%d",
                    seq.id,
                    block_id,
                )
                break

            if canonical_id != -1:
                canonical = self.kv.block(canonical_id)
                if canonical.token_ids != token_ids:
                    logger.warning(
                        "Hash collision while publishing offload prefix: "
                        "seq=%s block=%d canonical=%d",
                        seq.id,
                        block_id,
                        canonical_id,
                    )
                    break
                # Keep the canonical index entry, but annotate this request's
                # duplicate physical block as well: `hash_blocks()` needs the
                # final loaded block's hash as the parent when it publishes the
                # newly computed suffix. Annotating without indexing is what
                # separates `Block.update` from `BlockPool.publish` here, and it
                # is safe because `_unindex` only drops an entry that still
                # points at the block being reused.
                block.update(block_hash, token_ids)
            else:
                self.kv.publish(block_id, block_hash, token_ids)
                if self._event_log is not None:
                    self._event_log.append(
                        _make_block_stored(
                            [block_hash],
                            # The only BlockStored site fed straight from
                            # `_hash_block_tokens`; the other two accumulate
                            # into a list already. See `_make_block_stored`.
                            list(token_ids),
                            parent_hash if parent_hash != -1 else None,
                            self.hash_block_size,
                        )
                    )

            indexed_tokens += hbs
            parent_hash = block_hash

        return indexed_tokens

    def register_received_prefix(self, seq: Sequence) -> int:
        """Hash received prompt blocks into the prefix cache so subsequent
        turns can match them locally and transfer only the delta.

        Under DCP, one block-table entry represents ``dcp_world_size`` physical
        blocks and therefore ``hash_block_size`` global tokens. Hashing at the
        physical ``block_size`` would attach several incompatible token ranges
        to the same virtual block.

        Blocks before ``num_cached_tokens`` are already indexed locally; only
        the received suffix needs registration. The trailing partial hash block
        remains unpublished, matching ``hash_blocks``.

        Returns the number of complete received suffix blocks processed for
        this sequence. This includes blocks annotated from an existing canonical
        hash, so it is neither the total number of hashed prompt blocks nor
        necessarily the number of newly inserted hash-index entries.
        """
        if not self.enable_prefix_caching:
            return 0

        hbs = self._hash_block_size()
        num_full = seq.num_prompt_tokens // hbs
        num_full = min(num_full, len(seq.block_table))
        start = min(seq.num_cached_tokens // hbs, num_full)
        h = self._chain_parent_hash(seq, start)
        if h is None:
            return 0

        for i in range(start, num_full):
            token_ids = self._hash_block_tokens(seq, i)
            h = self.compute_hash(token_ids, h)
            block_id = seq.block_table[i]
            block = self.kv.block(block_id)
            indexed_block_id = self.kv.lookup(h)
            if indexed_block_id == -1:
                self.kv.publish(block_id, h, token_ids)
            else:
                indexed_block = self.kv.block(indexed_block_id)
                if indexed_block.token_ids != token_ids:
                    raise RuntimeError(
                        "Hash collision while registering received prefix: "
                        f"seq={seq.id} block={block_id} indexed={indexed_block_id}"
                    )
                block.update(h, token_ids)

        seq.num_hashed_tokens = max(seq.num_hashed_tokens, num_full * hbs)
        # The decode consumer has no local prefill postprocess to publish these
        # prompt blocks. Mark that one-shot work complete so its first decode
        # output does not publish the same physical blocks again.
        seq.prefix_hashes_published = True
        return num_full - start

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self.kv.free(block_id)
        seq.num_cached_tokens = 0
        # The block table is gone, so nothing of this seq is hashed any more.
        # Covers preemption too, which frees through here and re-prefills.
        seq.num_hashed_tokens = 0
        # Likewise the demand: it describes one admission against one cache
        # state, and a re-admitted seq gets a fresh answer from `can_allocate`
        # — including a fresh place in both funnel counters.
        seq.checkpoint_demand_pos = 0
        seq.checkpoint_demand_counted = False
        seq.checkpoint_demand_declined = False
        # The anchor is derived from `num_prompt_tokens` alone, so it would
        # still be correct on re-admission — but `_record_checkpoint_end` runs
        # unconditionally in `can_allocate`, and leaving a stale value here
        # would let a seq that never reaches that path keep a position nothing
        # re-validated against the current `state_caches`.
        seq.checkpoint_end_pos = 0
        seq.last_checkpoint_pos = 0
        # An uncommitted checkpoint describes state in a slot that is about to
        # go back on the free list, so the intent dies with it.
        if self.paged_state_checkpoints is not None:
            self.paged_state_checkpoints.forget_pending(seq)
        # A midstep reservation is the same intent one step earlier: a slot
        # held for a forward that is now not going to run. Handed back vacant —
        # publishing it would file a hash over bytes nobody wrote. Before the
        # block table is cleared, because a preempted seq re-prefills through
        # `can_allocate` and would otherwise re-plan on top of a live list.
        self.cancel_midstep(seq)
        del seq.block_table[:]  # main's `array("i")` has no `.clear()`
        if seq.has_per_req_cache and seq.state_slots:
            # Every slot the seq held — the committed one and its speculation
            # scratch, which is this request's alone and has no meaning to the
            # next one. A checkpoint it took is already back on the free list
            # under the state index; the source it was going to fork off is
            # dropped here rather than left to `release_state_pins`, because the
            # forward that owed the read is not going to happen and the slot
            # should not sit out a pass for a reader that no longer exists.
            # Unless a state load is in flight into the committed slot -- a
            # worker is writing it, so handing it back now would give the next
            # request a buffer someone else is filling, with `has_initial_state`
            # already true over it. `orphan` moves that slot to the index, which
            # releases it when the load's one report lands (or, if that report
            # is lost to a dead worker, when its reclaimer fires). The rollback
            # scratch is nobody's destination and goes back regardless.
            if self.state_offload is not None and self.state_offload.orphan(seq.id):
                self.state.release_many(seq.state_slots[1:])
            else:
                self.state.release_many(seq.state_slots)
            # No next forward will read a pending fork source after deallocation.
            self.state.drop_reader(seq.state_fork_src)
            seq.state_slots = []
            seq.state_fork_src = -1

        seq.offload_joint.load_hash = -1

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        seq_len = len(seq)
        current_blocks = len(seq.block_table)
        ebs = self._effective_block_size()
        needed_blocks = (seq_len + num_new_tokens + ebs - 1) // ebs
        new_blocks_needed = max(0, needed_blocks - current_blocks)
        return self._has_page_units(new_blocks_needed)

    def may_append(self, seq: Sequence, num_new_tokens: int = 1):
        # Note: in disaggregated (P/D) mode the scheduler skips this call on
        # the first decode step after remote prefill, because blocks were
        # already allocated during the KV transfer phase.
        block_table = seq.block_table
        seq_len = len(seq)
        # Check if we need to allocate a new block
        # When len(seq) % block_size == 1, we need a new block for the next token
        # When block_size == 1, every token needs a new block
        ebs = self._effective_block_size()
        if 0 < seq_len % ebs <= num_new_tokens or ebs == 1:
            needed_blocks = (seq_len + ebs - 1) // ebs
            while len(block_table) < needed_blocks:
                # Decode-generated blocks: token not finalized yet (depends on
                # sampling / speculative verification), so we cannot compute a
                # correct hash here.  Just allocate the block without hashing.
                block_table.append(self._fresh_block())

    # ---------------- KV event API ---------------- #

    def take_events(self) -> list[KVCacheEvent]:
        """Drain and return events accumulated since the last call."""
        if self._event_log is None or not self._event_log:
            return []
        self._event_log, events = [], self._event_log
        return events

    def clear_cache(self) -> None:
        """Drop every prefix-cache entry. Used by `/reset_prefix_cache`-style
        admin APIs. Does NOT touch blocks currently held by live sequences —
        they remain valid via their block_table refs, just unhashable for
        future requests."""
        self.kv.clear_index()
        self._state_checkpoint_cache.clear_index()
        if self._event_log is not None:
            self._event_log.append(_make_all_cleared())

    @property
    def kv_events_enabled(self) -> bool:
        """True iff KV events are being recorded."""
        return self._event_log is not None

    def record_remote_store(
        self,
        block_hashes: list[int],
        token_ids: list[int],
        parent_block_hash: int | None = None,
    ) -> None:
        """Emit a BlockStored(medium=REMOTE) for blocks received from a remote
        KV transfer producer (Mooncake/MoriIO decode side). Called by the
        KVConnector worker once the transfer completes so external KV-cache
        consumers (LMCache, etc.) can track remote-resident blocks."""
        if self._event_log is None or not block_hashes:
            return
        self._event_log.append(
            _make_block_stored(
                block_hashes,
                token_ids,
                parent_block_hash,
                self.hash_block_size,
                medium=MEDIUM_REMOTE,
            )
        )
