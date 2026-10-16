# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's per-request CSA2 state, as a tier beside the PAGE bytes.

V4.1 buys its cache in two currencies. PAGE bytes scale with history and are
what the byte codec moves. STATE bytes do not: every in-flight request owns one
fixed-size entry holding a window ring per layer, the compressor's incomplete
group per owner, and the Engram cursor. A restored PAGE prefix whose STATE was
not restored with it is not a degraded answer -- `PagedAttentionCache` refuses
it outright ("replay from a recoverable boundary"), because the cursor it finds
is not the position the scheduler claims. There is no partial reconstruction:
the window ring holds raw per-token rows the compressed pages do not contain.

So the two travel together or not at all, and this module is the half that
decides *which boundary* they travel at. The bytes themselves are moved by
`KdaStateTier` and `StateByteCodec`, which are model-agnostic -- the codec
passes its slot argument through untouched -- so nothing here re-implements a
transfer.

Two things differ from the Kimi-K3 leg this is modelled on, and both change the
policy rather than the mechanism:

**The source is live, not copy-on-written.** K3's boundary state sits in a
block vLLM copied aside and the planner pins; it is stable for as long as the
pin holds. V4.1's state sits in the request's own slot and is overwritten by
every forward. It therefore has to be *snapshotted* before the forward that
would overwrite it, into staging this module owns, and the store reads the
snapshot. `advance_cursor` states the same pairing from the runtime side: the
image a checkpoint takes is "the ring and the cursor of the step before this
one, which is the pair that agrees".

**A boundary is offered once or never.** K3 can ask its block pool whether the
state at an older boundary is still intact, and retry later if not. Here the
only boundary whose state exists is the request's current frontier -- a step
later the ring has moved. A frontier that jumps over a boundary has lost it,
which is what `boundary_passed` counts; it is not a hole to retry, and the fix
when it is non-zero is to align the scheduler's token budget to the interval.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from atom.model_engine.state_offload import StateOffloadIndex

from .kda_state import KdaLoad, KdaStore, boundary_prefix_hash

logger = logging.getLogger("atom")


#: Serve vLLM's own prefix-cache hits by capping them to one block below a
#: claimed CSA2 boundary and letting the tier supply that block. Off, and the
#: long comment in `cap_local_hit` says why it is not a flag waiting to be
#: turned on: the position it invents is one no request may legally sit at.
#: Kept because the measurements it enables are the case for doing this
#: properly -- local hits 0 -> 74% of queries, tier traffic down to one block
#: per request -- not because it is nearly ready.
_LOCAL_PREFIX_HITS_ENABLED = os.environ.get("ATOM_V41_LOCAL_PREFIX_HITS") in (
    "1",
    "true",
    "True",
)


#: Whether an image may back a LOCAL prefix hit. Separate from whether
#: images are wired, so the two halves can be measured apart: storing and
#: restoring images is testable without opening the gate that lets vLLM
#: serve a hit which -- if no image happens to sit at that exact position --
#: is a dead engine rather than a slow one.
_IMAGE_LOCAL_HITS_ENABLED = os.environ.get("ATOM_V41_IMAGE_LOCAL_HITS") in (
    "1",
    "true",
    "True",
)


# How many intervals `cap_hit` walks back before giving up. The same bound K3
# uses: a hit far past every stored boundary is a cold tier, and descending
# further costs a lookup per step for a prefix that will be recomputed anyway.
_MAX_CAP_DESCENT = 64

# Default tokens between STATE checkpoints. The image is fixed-size while PAGE
# bytes scale with history, so this is the whole cost knob: the tax is
# `state_bytes / (interval * page_bytes_per_token)`. For the shipped BF16
# geometry (5,276,672 B against 2,890 B/token) 8192 puts it near 22%, and the
# break-even against simply recomputing is around 1,826 tokens.
DEFAULT_STATE_INTERVAL = 8192


@dataclass(frozen=True)
class V41Store:
    """One boundary the scheduler decided is worth checkpointing.

    It names the request and the boundary rather than a buffer, because the
    bytes are not reachable from the process that decides this. The worker
    resolves the slot, verifies the cursor and snapshots; see
    `V41StateViews.snapshot`.
    """

    op_id: int
    prefix_hash: int
    req_id: str
    boundary: int


@dataclass
class _PendingStore:
    """One in-flight store, awaiting a report from every rank."""

    prefix_hash: int
    reports: int = 0
    failures: int = 0


class V41StateViews:
    """The byte views the codec reads and writes, plus the staging ring.

    `page_unit_views` (store source) and `state_entry_views` (load
    destination) address *different* arenas on purpose: a store reads a
    snapshot this object owns, while a load writes the live slot. They are the
    same number of bytes in the same order, which is all the codec requires of
    them -- it intersects two ordered byte streams and never interprets either.
    """

    def __init__(self, cache, *, stage_depth: int = 8) -> None:
        self._cache = cache
        self.entry_bytes = int(cache.geometry.state_bytes)
        self.num_slots = int(cache.num_slots)
        depth = max(1, int(stage_depth))
        # One contiguous arena rather than `depth` allocations: it is handed
        # out a slab at a time and never resized, so a single reservation is
        # also a single statement of how much HBM this leg costs.
        # Allocated after vLLM sized the KV pool against
        # `gpu_memory_utilization`, so these bytes are outside that budget.
        # Nothing subtracts them, which at a tight utilisation turns into an
        # OOM during connector registration -- and the thing that has to shrink
        # is the KV pool or this depth, not the model. Say the figure at the
        # point of allocation so the traceback has it.
        # The device the state lives on, named for the duck type the recurrent
        # leg's own views satisfy (`KdaPageViews.device`): under the `mp`
        # backend `_state_storage_for_mp` reads it to place this tier's
        # staging, because there is no in-process engine to ask. Taken from
        # the cache rather than from the staging ring below, so it is right
        # even if the ring fails to allocate.
        self.device = cache.state_bytes.device
        want = depth * self.entry_bytes
        try:
            self._staging = torch.empty(
                (depth, self.entry_bytes),
                dtype=torch.uint8,
                device=cache.state_bytes.device,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise torch.cuda.OutOfMemoryError(
                f"ATOM LMCache offload: V4.1 state staging needs {want} B "
                f"({depth} x {self.entry_bytes}) on top of the KV pool vLLM "
                "already sized to fill gpu_memory_utilization. Lower "
                "`atom.offload.v41.state_stage_depth`, or leave the pool room "
                "with a smaller --gpu-memory-utilization."
            ) from exc
        self._free: list[int] = list(range(depth))
        self._lock = threading.Lock()
        self.layout_id = str(cache.geometry.layout_id)

    @property
    def stage_depth(self) -> int:
        return int(self._staging.shape[0])

    def acquire_stage(self) -> int | None:
        """A free staging slab, or None when every one is still in flight."""
        with self._lock:
            return self._free.pop() if self._free else None

    def release_stage(self, stage: int) -> None:
        with self._lock:
            self._free.append(int(stage))

    def snapshot(self, slot: int, stage: int) -> None:
        """Copy one live slot into staging, on the caller's current stream.

        Device-to-device and ordered by the stream, so it lands after the
        previous forward wrote the slot and before the next one overwrites it.
        The caller records its own event afterwards; the store thread waits on
        that rather than this copy blocking anyone.
        """
        self._staging[int(stage)].copy_(self._cache.state_bytes[int(slot)])

    def committed_positions(self, slots) -> list[int]:
        """What each slot's cursor says its state is at, in one read.

        `PagedAttentionCache` will refuse any request whose cursor is not
        exactly the frontier the scheduler claims, so this is the value that
        decides whether a snapshot is worth storing at all.

        Gathered for the whole batch at once because the read is a blocking
        device-to-host copy on the compute stream, inside `start_load_kv` --
        one sync per candidate store put that stall on the critical path of
        every step that stores anything, which is the path this leg is
        supposed to stay off.
        """
        index = [int(s) for s in slots]
        if not index:
            return []
        rows = torch.as_tensor(index, device=self._cache.cursor.device)
        return torch.index_select(self._cache.cursor, 0, rows)[:, 0].tolist()

    # -- codec contract --------------------------------------------------
    def page_unit_views(self, unit_ids) -> list[torch.Tensor]:
        """Store source: the staging slab named by *unit_ids*."""
        stage = self._one(unit_ids)
        return [self._staging[stage]]

    def state_entry_views(self, slot) -> list[torch.Tensor]:
        """Load destination: the live state slot named by *slot*."""
        index = self._one(slot)
        if not 0 <= index < self.num_slots:
            raise IndexError(
                f"V4.1 state offload: slot {index} is outside the pool's "
                f"{self.num_slots} slots"
            )
        return [self._cache.state_bytes[index]]

    @staticmethod
    def _one(ids) -> int:
        """V4.1's state is one entry, so its id is one integer.

        Spelled as a sequence because the codec addresses K3's by one block id
        per mamba group and passes whatever it was given straight through.
        """
        if isinstance(ids, (int,)):
            return int(ids)
        values = [int(i) for i in ids]
        if len(values) != 1:
            raise ValueError(
                "V4.1 state offload: an entry is one region, got " f"{len(values)} ids"
            )
        return values[0]


class V41StateWorkerLeg:
    """Worker half: resolves a decided boundary to bytes, or says why not.

    Three things can stop a store the scheduler asked for, and all three are
    only knowable here: the request may hold no state slot, its cursor may not
    read the boundary, or the staging ring may be full. Each is counted and
    reported back rather than logged and forgotten -- a tier that stores
    nothing has to be able to say which of the three it was.
    """

    def __init__(self, views: V41StateViews, tier, slot_allocator) -> None:
        self._views = views
        self._tier = tier
        self._warned_no_slot = False
        self._slots = slot_allocator
        self._stage_of_op: dict[int, int] = {}

    def snapshot_and_submit(self, stores) -> dict[str, list[int]]:
        """Copy each decided boundary into staging and queue its store.

        The copy runs on the caller's current stream, which is the compute
        stream inside ``start_load_kv`` -- before this step's forward is
        enqueued and after the previous one wrote the slot. That is the pairing
        the runtime itself states: the image a checkpoint takes is the ring and
        the cursor of the step before this one, which is the pair that agrees.
        """
        refused: dict[str, list[int]] = {}

        def refuse(reason: str, op_id: int) -> None:
            refused.setdefault(reason, []).append(int(op_id))

        submitted: list[tuple[int, int]] = []
        # One cursor read for the batch, before the per-store walk below.
        candidates = [
            (store, self._slots.slot_for(store.req_id)) for store in stores or ()
        ]
        positions = self._views.committed_positions(
            [slot for _, slot in candidates if slot is not None]
        )
        cursors = iter(positions)
        for store, slot in candidates:
            if slot is None:
                # Say which key missed and what the allocator does hold. The
                # two halves of this leg name a request independently -- the
                # scheduler half from `frontiers`, the worker half from the
                # batch the forward ran -- so "no slot" is as often a key that
                # does not match as a request that has none, and the counter
                # alone cannot tell those apart.
                if not self._warned_no_slot:
                    self._warned_no_slot = True
                    known = list(getattr(self._slots, "_key_to_slot", {}))[:4]
                    logger.warning(
                        "V4.1 state offload: no STATE slot for %r (%s); the "
                        "allocator holds %d key(s), e.g. %r. If the keys look "
                        "like the same request spelled differently, the two "
                        "halves are naming it differently and no boundary will "
                        "ever be stored. allocator=%#x",
                        store.req_id,
                        type(store.req_id).__name__,
                        len(getattr(self._slots, "_key_to_slot", {})),
                        known,
                        id(self._slots),
                    )
                refuse("worker_no_slot", store.op_id)
                continue
            committed = next(cursors)
            if committed != int(store.boundary):
                # The one failure nothing downstream can detect: an image keyed
                # to a position it is not at would be accepted by `cap_hit` and
                # then refused by the forward it was restored into.
                refuse("cursor_mismatch", store.op_id)
                logger.warning(
                    "V4.1 state offload: %s is scheduled at %d but its slot's "
                    "cursor reads %d; dropping the store rather than keying an "
                    "image to a position it is not at",
                    store.req_id,
                    store.boundary,
                    committed,
                )
                continue
            stage = self._views.acquire_stage()
            if stage is None:
                refuse("stage_full", store.op_id)
                continue
            self._views.snapshot(int(slot), stage)
            self._stage_of_op[int(store.op_id)] = stage
            submitted.append((store, stage))
        if submitted:
            # One event for the whole batch of copies: they are all on this
            # stream and in order, so the last one completing means every slab
            # is readable. A fresh event each time -- an event reused across
            # steps is recorded and waited on by two different steps at once.
            event = torch.cuda.Event()
            event.record()
            for store, stage in submitted:
                self._tier.submit_store(
                    KdaStore(int(store.op_id), int(store.prefix_hash), (stage,)),
                    event,
                )
        return refused

    def release_reported(self, op_ids) -> None:
        """Give back the staging slabs of ops every rank has reported on."""
        for op_id in op_ids or ():
            stage = self._stage_of_op.pop(int(op_id), None)
            if stage is not None:
                self._views.release_stage(stage)


#: The planner most recently built, so the worker-side refusal dump can
#: read its counters. Scheduler and worker share a process here; when they
#: do not, this is None and the dump says so rather than guessing.
_LAST_PLANNER = None


def last_planner():
    return _LAST_PLANNER


class V41BoundaryPlanner:
    """Scheduler half: picks the boundaries, and caps hits to them.

    Everything this decides is about keeping one invariant: a request that vLLM
    is told has `n` computed tokens must have a STATE image whose cursor reads
    exactly `n`. The two ways to break it are restoring a prefix past the last
    stored boundary (`cap_hit` prevents) and storing a snapshot that was not
    taken at the boundary it is keyed by (`cursor_mismatch` catches).
    """

    def __init__(
        self,
        *,
        hash_block_size: int,
        boundary_grid: int | None = None,
        chunk_size: int,
        state_interval: int,
        max_num_batched_tokens: int,
        world_size: int,
        can_store: bool = True,
        can_load: bool = True,
        state_checkpoints_can_serve: bool = False,
    ) -> None:
        self.hash_block_size = int(hash_block_size)
        # Two different things that used to be one.
        #
        # `hash_block_size` is the unit vLLM's `request.block_hashes` are in;
        # indexing that list with anything else reads the wrong entry.
        # `boundary_grid` is the grid a STATE boundary may sit on, which for
        # V4.1 is its PAGE block: state is only restorable at a whole PAGE
        # edge. They coincide while there is one kv cache group and diverge
        # the moment there are two, because the hash unit is engine-wide and
        # becomes the smallest group's -- with DSpark, 32 against the proxy's
        # 256. Measured: `cap_hit` then accepted a boundary at 96 tokens,
        # which is a legal 32-grid point, is not a PAGE edge, and floors to
        # zero chunks at the caller. The hit it capped was 1280.
        self.boundary_grid = int(boundary_grid or hash_block_size)
        if self.boundary_grid % self.hash_block_size:
            raise ValueError(
                f"DeepSeek-V4.1 state boundary grid {self.boundary_grid} is "
                f"not a multiple of the hash block size "
                f"{self.hash_block_size}; a boundary on that grid could not "
                "be hashed"
            )
        self.chunk_size = int(chunk_size)
        self.state_interval = int(state_interval)
        self._world_size = max(1, int(world_size))
        # Decided by the caller, which holds the VllmConfig. Asking vLLM for
        # its *current* config from here returns None -- the scheduler calls
        # `cap_local_hit` outside any `set_current_vllm_config` scope -- so a
        # check made here was false on every request and this gate never
        # opened (measured: local_cap_image=0 with ATOM_V41_IMAGE_LOCAL_HITS=1).
        self._state_checkpoints_can_serve = bool(state_checkpoints_can_serve)
        self._index = StateOffloadIndex(can_store=can_store, can_load=can_load)
        self._next_op_id = 0
        self._pending_stores: dict[int, _PendingStore] = {}
        self._loads: list[KdaLoad] = []
        self._lookup_ctx: tuple[str, Any] | None = None
        # req_id -> the last frontier this request was swept at, so one step's
        # frontier is offered once even if the connector sweeps twice.
        self._swept: dict[str, int] = {}
        # The last frontier this request had a boundary offered at. The grid
        # is a MINIMUM SPACING from here, not a fixed ladder of multiples of
        # `state_interval`: a chunk that ends a few tokens past a rung used to
        # miss it and lose that boundary for good, where spacing lets the next
        # block-aligned frontier serve instead. The store count per request is
        # unchanged by that -- still about one per interval of progress -- so
        # it buys coverage rather than bandwidth.
        self._last_offered: dict[str, int] = {}
        # req_id -> boundary claimed for a purely local prefix hit, read
        # once by `update_state_after_alloc` and dropped there.
        global _LAST_PLANNER
        _LAST_PLANNER = self
        self._local_caps: dict[str, int] = {}
        # Requests whose local cap the tier failed to back. They decline
        # on the retry rather than being deferred again and again.
        self._local_cap_withdrawn: set[str] = set()
        self._counters: dict[str, int] = {
            "sweep_offered": 0,
            "sweep_stores": 0,
            "sweep_no_hash": 0,
            "sweep_known": 0,
            # V4.1-specific: the frontier moved past a boundary without
            # stopping on it, so that boundary's state no longer exists.
            # Non-zero means prefill chunks are not landing on the interval --
            # not a transient. Counted only from a request's second sighting
            # on, because the first says where it *started*, not what it
            # skipped.
            "boundary_passed": 0,
            # Requests whose first frontier was already past zero, i.e. served
            # a restored prefix. Their lower boundaries are in the tier, not
            # lost, and were once counted as passed -- which made a working
            # restore read like a budget problem.
            "restored_start": 0,
            # The prompt-end anchor: the one boundary that makes a prompt
            # shorter than `state_interval` reusable at all.
            "anchor_stores": 0,
            # Previously silent: a request skipped because it was preempted or
            # had finished, and one the scheduler no longer knows. Both are
            # real coverage losses and neither was counted, so the sweep's
            # own numbers could not be made to add up.
            "sweep_skipped": 0,
            "sweep_unknown_request": 0,
            "stage_full": 0,
            # The guard on the hard contract. Must stay zero; anything else
            # means a snapshot was taken at a position the scheduler did not
            # claim, and the store was dropped rather than keyed wrongly.
            "cursor_mismatch": 0,
            "cap_kept": 0,
            "cap_declined": 0,
            "cap_unarmed": 0,
            # The same two, for hits vLLM served from its own pool without
            # asking the connector. Kept apart because they answer different
            # questions: the connector pair measures the tier, this pair
            # measures what prefix caching costs on a model whose state the
            # block pool does not carry.
            "local_cap_declined": 0,
            #: Local hits passed through whole because a checkpoint image
            #: can restore the state at the hit. Zero while images are off.
            "local_cap_image": 0,
            "local_cap_kept": 0,
            "local_cap_withdrawn": 0,
            "admission_refused": 0,
        }
        # One warning per planner, not per hit: an unarmed cap fires once
        # per request on whatever path is missing its arming.
        self._warned_unarmed = False
        self._last_stats_log = 0.0
        if self.state_interval % self.chunk_size:
            raise ValueError(
                f"V4.1 state offload: state interval {self.state_interval} is "
                f"not a multiple of the LMCache chunk size {self.chunk_size}; "
                "a boundary would then fall inside a chunk and the PAGE prefix "
                "could not end where the state does"
            )
        if self.state_interval % self.boundary_grid:
            raise ValueError(
                f"V4.1 state offload: state interval {self.state_interval} is "
                f"not a multiple of the boundary grid {self.boundary_grid}; "
                "a boundary would have no block hash to be keyed by"
            )
        budget = int(max_num_batched_tokens)
        if budget and budget % self.state_interval:
            # Not fatal, because decode still lands on every boundary and a
            # short prompt never reaches one. But during chunked prefill the
            # frontier advances a whole budget at a time, so every boundary
            # inside a chunk is passed rather than seen -- which shows up as a
            # tier that stores almost nothing, with `boundary_passed` saying
            # why.
            logger.warning(
                "V4.1 state offload: max_num_batched_tokens %d is not a "
                "multiple of the state interval %d; prefill frontiers will "
                "step over boundaries (watch boundary_passed). Set one to a "
                "multiple of the other to checkpoint during prefill.",
                budget,
                self.state_interval,
            )

    # -- lookup ----------------------------------------------------------
    def begin_lookup(self, request) -> None:
        self._lookup_ctx = (str(request.request_id), request)

    def end_lookup(self) -> None:
        self._lookup_ctx = None

    def cap_hit(self, seq, hit: int) -> int:
        """Shorten *hit* to the last boundary this index still claims.

        Returning a shorter hit costs a recompute. Returning a longer one
        hands the forward a cursor that disagrees with the scheduler, which
        `_report_stale_state` turns into a dead engine -- so when nothing is
        claimed the answer is 0, never *hit*.
        """
        hit = int(hit)
        if hit <= 0:
            return hit
        if self._lookup_ctx is None:
            # Fail closed, which is what the paragraph above promises and what
            # the code did not do. An unarmed hook used to return the hit
            # untouched, so an external hit that never reached a boundary test
            # was served whole: PAGE restored to a position whose STATE was
            # never written, and `_report_stale_state` then killing the engine
            # ("needs state at 6912, found 3"). Declining costs a recompute of
            # a prefix the tier may well have held; serving it costs the
            # server.
            self._counters["cap_unarmed"] += 1
            if not self._warned_unarmed:
                self._warned_unarmed = True
                logger.warning(
                    "V4.1 state offload: the hit cap fired outside a lookup "
                    "(seq=%s, hit=%d) and declined it. Every external hit has "
                    "to pass a boundary test, so a path that caps without "
                    "arming is a path that would restore PAGE bytes with no "
                    "STATE behind them.",
                    getattr(seq, "id", None),
                    hit,
                )
            return 0
        req_id, request = self._lookup_ctx
        if str(seq.id) != req_id:
            logger.warning(
                "V4.1 state offload: hit cap armed for %s but called for %s; "
                "declining the external hit",
                req_id,
                seq.id,
            )
            return 0
        block_hashes = getattr(request, "block_hashes", None) or ()
        # Descend the BLOCK grid, not the interval ladder. The store side
        # offers any block-aligned frontier spaced an interval apart, plus the
        # prompt-end anchor, so the boundaries that exist are not multiples of
        # the interval -- an interval-stepped search walks straight past them
        # and, for a prompt shorter than one interval, starts at zero.
        #
        # The descent is still bounded by `_MAX_CAP_DESCENT`, so the search
        # reaches back `_MAX_CAP_DESCENT` blocks rather than that many
        # intervals. That is the right span for this grid: the nearest stored
        # boundary is at most one interval below the hit, and the step is
        # chosen so the search covers an interval whenever the two are
        # configured together (`state_interval % hash_block_size == 0` is
        # checked at construction).
        # Two phases, because the grid has two scales. Near the hit the
        # boundaries are block-aligned and an interval apart but NOT at
        # multiples of the interval -- the store side takes any block-aligned
        # frontier far enough from the last, plus the prompt-end anchor -- so
        # an interval-stepped search walks straight past them, and for a prompt
        # shorter than one interval it starts at zero. Far from the hit, a
        # block-stepped search would need `interval / block` lookups per
        # interval of reach to cover the same span.
        #
        # So: walk the block grid for one interval, then the interval ladder
        # for the rest of the budget. Near-field precision at the old reach,
        # and at most `2 * _MAX_CAP_DESCENT` index lookups.
        boundary = self._claimed_boundary_at_or_below(block_hashes, hit)
        if os.environ.get("ATOM_OFFLOAD_LOOKUP_PROBE") in ("1", "true", "True"):
            # The boundary the index claims, against the hit being capped to
            # it. A kept cap that lands below one chunk is floored to zero by
            # the caller and reads, from outside, exactly like a decline.
            logger.info(
                "V4.1 cap_hit: req=%s hit=%d boundary=%d hashes=%d -> %d",
                req_id,
                hit,
                boundary,
                len(block_hashes),
                0 if boundary <= 0 else min(hit, boundary),
            )
        if boundary <= 0:
            self._counters["cap_declined"] += 1
            return 0
        self._counters["cap_kept"] += 1
        return min(hit, boundary)

    def _claimed_boundary_at_or_below(self, block_hashes, hit: int) -> int:
        """The highest boundary <= *hit* the index still claims, or 0.

        Shared by both cap paths so they cannot disagree about which
        boundaries exist: a local hit capped by a different rule than the
        external one is a local hit whose state nobody stored.
        """
        block = self.boundary_grid
        boundary = (hit // block) * block
        near = min(_MAX_CAP_DESCENT, max(1, self.state_interval // block))
        for phase_step, phase_len in (
            (block, near),
            (self.state_interval, _MAX_CAP_DESCENT),
        ):
            for _ in range(phase_len):
                if boundary <= 0:
                    return 0
                h = self.boundary_hash(block_hashes, boundary)
                if h is not None and self._index.could_serve(h):
                    return boundary
                boundary -= phase_step
        return 0

    def counters(self) -> dict:
        return dict(self._counters)

    def cap_local_hit(self, request, hit: int) -> int:
        """Shorten vLLM's *own* prefix-cache hit so the tier can back it.

        The connector-side `cap_hit` only sees hits vLLM asked it about. With
        prefix caching on, vLLM's block pool answers first and keeps whatever
        it can serve locally -- a path the connector is never consulted on. A
        request admitted that way arrives with its PAGE prefix restored and a
        state slot that was never written, which `_report_stale_state` turns
        into a dead engine. That is a hard `raise`, not a degradation, so a
        local hit may only be served if something is *guaranteed* to restore
        the state behind it.

        Nothing can be guaranteed from here: vLLM forbids parking a request
        that has no external tokens (`assert num_external_computed_tokens > 0`
        under `load_kv_async`), so a purely local hit runs its forward in the
        same step the load is issued, and a load that misses then has no step
        left in which to invalidate anything.

        So the hit is capped one block *below* the claimed boundary rather
        than to it. The tier is then asked for that last block, the request
        parks on it like any external hit, and the whole proven path applies:
        PAGE and STATE arrive together, and a miss invalidates the blocks and
        recomputes instead of killing the engine. The win is that the other
        `boundary - block` tokens come from HBM rather than over PCIe -- the
        earlier code declined the local hit outright and paid for the entire
        prefix from the tier.

        If the tier cannot supply that last block the cap is withdrawn by
        `get_num_new_matched_tokens`, which defers the request one step; the
        mark it leaves makes this decline on the retry, which is the old
        behaviour and is safe.
        """
        hit = int(hit)
        self._counters["local_cap_calls"] = self._counters.get("local_cap_calls", 0) + 1
        req_id = str(getattr(request, "request_id", "") or "")
        self._local_caps.pop(req_id, None)
        if hit <= 0:
            return hit
        if self._state_checkpoints_can_serve and _IMAGE_LOCAL_HITS_ENABLED:
            # Left alone. vLLM has already reduced this hit to a boundary
            # the state group can serve: its coordinator takes the minimum
            # across groups, and the image group only offers boundaries it
            # has a cached checkpoint block for (measured: the group
            # answered 1280 for a 1345-token prompt whose checkpoint is at
            # 1280).
            #
            # Capping it here manufactured the very mismatch this branch
            # was written to avoid. `get_mamba_prefill_checkpoint_position`
            # is a rule about a raw prefix length; applied to an
            # already-snapped hit it returns one block lower, which is a
            # position nothing wrote -- `needs state at 1024, found 3`.
            self._counters["local_cap_image"] += 1
            return hit
        if not _LOCAL_PREFIX_HITS_ENABLED:
            # Off, and the reason is structural rather than a race that is
            # waiting to be found. Traced, for one request:
            #
            #   cap_local  hit=9472 claimed=8192 capped=7936
            #   alloc      cached=7936 ext=256 total=8192  -> load at 8192
            #   alloc      cached=7936 ext=0   total=7936
            #
            # The second line is vLLM re-admitting the request after its
            # dense load failed, with the external half gone and the frontier
            # rewound to the local prefix. That prefix is 7936 -- the point
            # this cap invents, one block below a boundary -- and by
            # construction nobody ever stored state there. The engine then
            # dies on `needs state at 7936`, and it died there in every run,
            # under eager and under FULL_AND_PIECEWISE alike, with only the
            # `found` value differing.
            #
            # It cannot be rescued by refusing afterwards: the request is no
            # longer parked, so a load-failure report arrives after its
            # forward has already run. Nor by storing a boundary at B - one
            # block, because with 8192-token prefill chunks that position is
            # never a frontier anyone can snapshot. The capped point has to be
            # one the request may legally sit at, and the only such points are
            # stored boundaries and zero -- which is what declining gives.
            #
            # Declining costs the HBM-speed path and keeps the tier serving
            # the whole prefix, with its state. That is correct and is what
            # every default run measures.
            self._counters["local_cap_declined"] += 1
            return 0
        if req_id in self._local_cap_withdrawn:
            # Asked once, the tier had nothing. Declining sends the whole
            # prefix through the connector, which is what used to happen
            # always.
            self._counters["local_cap_declined"] += 1
            return 0
        block_hashes = getattr(request, "block_hashes", None) or ()
        boundary = self._claimed_boundary_at_or_below(block_hashes, hit)
        if boundary <= self.boundary_grid:
            # Nothing claimed, or the only boundary is the first block and
            # capping below it would leave nothing local to keep.
            self._counters["local_cap_declined"] += 1
            return 0
        self._local_caps[req_id] = boundary
        self._counters["local_cap_kept"] += 1
        return min(hit, boundary - self.boundary_grid)

    def withdraw_local_cap(self, request) -> bool:
        """Give up on a local cap the tier did not back. True if there was one.

        The caller defers the request a step. The mark is what keeps that from
        repeating: on the retry the cap declines and the request takes the
        connector path for its whole prefix.
        """
        req_id = str(getattr(request, "request_id", "") or "")
        if self._local_caps.pop(req_id, None) is None:
            return False
        self._local_cap_withdrawn.add(req_id)
        self._counters["local_cap_withdrawn"] += 1
        return True

    def confirm_local_cap(self, request) -> None:
        """The tier answered; the cap stands and needs no retry mark."""
        self._local_caps.pop(str(getattr(request, "request_id", "")), None)

    def _prompt_anchor(self, request) -> int:
        """The last block-aligned point at or before the end of the prompt.

        The single most valuable boundary a request has: a later request with
        the same prompt reuses everything up to it. Without it a prompt shorter
        than `state_interval` has no boundary at all -- `floor(N / interval)`
        is zero -- so the tier stores nothing for it and `cap_hit` can only
        return zero, which reads as a cold tier rather than as a prompt the
        grid cannot express.

        Block-aligned rather than the prompt end itself, because a boundary
        needs a block hash to be keyed by and vLLM hashes whole blocks. The
        frontier reaches this point only if a prefill chunk ends there, which
        is what the scheduler patch arranges.
        """
        total = int(getattr(request, "num_prompt_tokens", 0) or 0)
        return (total // self.boundary_grid) * self.boundary_grid

    def boundary_hash(self, block_hashes, boundary_tokens: int) -> int | None:
        """The prefix-hash key for the state committed at *boundary_tokens*."""
        if boundary_tokens <= 0 or boundary_tokens % self.hash_block_size:
            return None
        index = boundary_tokens // self.hash_block_size - 1
        if index < 0 or index >= len(block_hashes):
            return None
        return boundary_prefix_hash(block_hashes[index])

    # -- store -----------------------------------------------------------
    def collect_frontier_stores(
        self, frontiers, requests_by_id, skip_req_ids=()
    ) -> list[V41Store]:
        """Snapshot every scheduled request that is sitting on a boundary.

        Decides only. The bytes live in the worker process, so the snapshot,
        the cursor check and the staging slab are all its half of this; what
        travels is the request id and the boundary, which is everything needed
        to find the state and to know what it must read.

        *frontiers* is `{req_id: num_computed_tokens}` -- the tokens committed
        *before* this step, which is exactly the state the slot holds right
        now. A request is offered at most once per frontier: the state at an
        older boundary is already gone, so unlike K3's sweep there is no hole
        to come back to.
        """
        accepted: list[V41Store] = []
        if not self._index.can_store:
            return accepted
        for req_id, frontier in (frontiers or {}).items():
            req_id = str(req_id)
            if req_id in skip_req_ids:
                self._counters["sweep_skipped"] += 1
                continue
            frontier = int(frontier)
            previous = self._swept.get(req_id)
            if previous is not None and previous >= frontier:
                continue
            # Count the boundaries this step stepped over. They are lost, not
            # deferred -- the ring has already moved past them. Counted from
            # zero on a request's first sighting too: a prefill whose first
            # scheduled chunk is larger than the interval passed every
            # boundary inside it, and not counting those would report a clean
            # sweep for the configuration that stores the least.
            if previous is None:
                # First sighting. A request that was restored starts at the
                # boundary its hit was capped to, and every boundary below it
                # is in the tier already -- counting those as lost would
                # report the restore as a miss and send the next person
                # tuning a budget that is not the problem. A request that was
                # not restored starts at 0 and has crossed nothing.
                self._counters["restored_start"] += 1 if frontier else 0
            else:
                crossed = (
                    frontier // self.state_interval - previous // self.state_interval
                )
                if frontier % self.state_interval:
                    self._counters["boundary_passed"] += max(0, crossed)
                else:
                    # The one it landed on is not passed -- it is about to be
                    # offered.
                    self._counters["boundary_passed"] += max(0, crossed - 1)
            self._swept[req_id] = frontier
            request = requests_by_id.get(req_id)
            if frontier <= 0 or frontier % self.boundary_grid:
                # Not a block boundary, so `boundary_hash` has no key for it
                # whatever else is true. Nothing to count: these are most of
                # the frontiers a decode produces.
                continue
            if request is None:
                self._counters["sweep_unknown_request"] += 1
                continue
            is_anchor = (
                req_id not in self._last_offered
                and frontier == self._prompt_anchor(request)
            )
            spaced = frontier - self._last_offered.get(req_id, 0) >= self.state_interval
            if not (spaced or is_anchor):
                continue
            self._counters["sweep_offered"] += 1
            if is_anchor and not spaced:
                self._counters["anchor_stores"] += 1
            prefix_hash = self.boundary_hash(
                getattr(request, "block_hashes", None) or (), frontier
            )
            if prefix_hash is None:
                self._counters["sweep_no_hash"] += 1
                continue
            if prefix_hash in self._index.hashes or self._offer_in_flight(prefix_hash):
                self._counters["sweep_known"] += 1
                continue
            op_id = self._next_op_id
            self._next_op_id += 1
            self._pending_stores[op_id] = _PendingStore(prefix_hash)
            self._last_offered[req_id] = frontier
            self._counters["sweep_stores"] += 1
            accepted.append(V41Store(op_id, prefix_hash, req_id, frontier))
        return accepted

    def absorb_worker_counters(self, counts) -> None:
        """Fold the worker half's refusal tallies into the leg's own stats.

        The worker owns the cursor guard and the staging ring, so it is where
        `cursor_mismatch` and `stage_full` actually happen -- but `log_stats`
        here is the single line an operator reads. Without this they are
        initialised to zero and never written, which makes the alarm the
        docstring calls "must stay zero" structurally incapable of firing: a
        run in which every store was dropped for a cursor mismatch reports
        `cursor_mismatch=0`.
        """
        for reason, count in (counts or {}).items():
            self._counters[reason] = self._counters.get(reason, 0) + int(count)

    def absorb_reports(self, stored, failed) -> None:
        """Fold the worker's per-rank store reports into the index.

        A boundary is claimed only when every rank stored it: an image that
        exists on three of four ranks is one `cap_hit` would accept and one
        rank would then fail to load, leaving that rank's prefix restored over
        the state of whatever ran there before.
        """
        touched: set[int] = set()
        for source, is_failure in ((stored or {}, False), (failed or {}, True)):
            for op_id, count in source.items():
                pending = self._pending_stores.get(int(op_id))
                if pending is None:
                    continue
                pending.reports += int(count)
                if is_failure:
                    pending.failures += int(count)
                touched.add(int(op_id))
        for op_id in touched:
            pending = self._pending_stores[op_id]
            # Quorum over stored|failed, not stored alone: a rank that could
            # not write never sends a second report, so waiting for one would
            # hold its staging slab for the life of the process.
            if pending.reports < self._world_size:
                continue
            del self._pending_stores[op_id]
            if pending.failures == 0:
                self._index.note_stored(pending.prefix_hash)

    # -- load ------------------------------------------------------------
    def resolve_load(
        self, request, num_total_computed: int, error_block_ids=()
    ) -> bool:
        """Queue the state load for a request whose PAGE prefix was restored.

        False when there is nothing to restore, which the caller must treat as
        "this request has no state leg" and not as a failure: a request served
        entirely from vLLM's own pool never reaches here.
        """
        boundary = int(num_total_computed)
        # The BLOCK grid, not the interval ladder. `cap_hit` accepts any
        # block-aligned boundary the index claims -- the store side offers
        # frontiers spaced an interval apart plus the prompt-end anchor, so
        # the boundaries that exist are not multiples of the interval. Testing
        # for one here declined exactly the hits the cap had just accepted,
        # leaving PAGE restored with no state load queued behind it. The
        # authority on whether a boundary exists is the index, which
        # `request_load` below consults; this test only rejects a frontier
        # that cannot be hashed at all.
        if boundary <= 0 or boundary % self.hash_block_size:
            return False
        prefix_hash = self.boundary_hash(
            getattr(request, "block_hashes", None) or (), boundary
        )
        if prefix_hash is None:
            return False
        req_id = str(request.request_id)
        if not self._index.request_load(req_id, prefix_hash):
            return False
        self._loads.append(
            KdaLoad(
                req_id,
                prefix_hash,
                (),  # filled by the worker, which knows the slot
                tuple(error_block_ids),
            )
        )
        return True

    def refuse_admission(self, request, error_block_ids=()) -> None:
        """Fail a request whose prefix was admitted with no load behind it.

        `resolve_load` returning False used to be discarded by its caller, and
        that is the whole of the race: the admission goes ahead, PAGE is in
        place at the boundary, and nothing is queued to put STATE there. The
        forward then finds whatever the slot held -- measured as `needs state
        at 7936, found 9657` (the request's own later state, from before it
        was preempted) and `found 5` (another tenant's).

        Refusing cannot undo the allocation; what it can do is make the
        request take the path a missing load already takes. An unresolvable
        destination is failed by the tier without touching the device, which
        reports through `_join_kda`, invalidates these blocks and recomputes.
        So the two halves live or die together, which is the invariant the
        boundary design rests on and had never been enforced.
        """
        self._counters["admission_refused"] += 1
        self._loads.append(
            KdaLoad(
                str(request.request_id),
                0,
                (-1,),  # slot zero is valid; only a negative slot refuses
                tuple(error_block_ids),
            )
        )

    def take_loads(self) -> list[KdaLoad]:
        loads, self._loads = self._loads, []
        return loads

    def on_load_result(self, req_id: str, ok: bool) -> None:
        if ok:
            self._index.complete_load(str(req_id))
        else:
            # The index advertised bytes LMCache no longer holds. Dropping the
            # claim is what stops the next request paying for the same miss.
            self._index.fail_load(str(req_id))

    def _offer_in_flight(self, prefix_hash: int) -> bool:
        return any(p.prefix_hash == prefix_hash for p in self._pending_stores.values())

    def forget_request(self, req_id: str) -> None:
        """Drop a finished or preempted request's sweep cursor.

        A preempted request resumes at `num_computed == 0` under the same id
        and its slot is reset, so the cursor from its previous life would
        suppress every boundary of its new one.
        """
        self._swept.pop(str(req_id), None)
        self._last_offered.pop(str(req_id), None)
        self._local_caps.pop(str(req_id), None)
        self._local_cap_withdrawn.discard(str(req_id))

    # -- observability ---------------------------------------------------
    def stats(self) -> dict[str, int]:
        merged = dict(self._counters)
        index_stats = getattr(self._index, "stats", None)
        if callable(index_stats):
            merged.update(index_stats())
        return merged

    #: Seconds between stat lines. 60 by default, and settable because a
    #: workload shorter than one interval produces its last line BEFORE the
    #: load and none during it -- which reads exactly like a load in which
    #: nothing happened. `ATOM_V41_STATS_INTERVAL=5` is for measuring.
    _STATS_INTERVAL = float(os.environ.get("ATOM_V41_STATS_INTERVAL", "") or 60.0)

    def log_stats(
        self, *, interval_s: float | None = None, force: bool = False
    ) -> None:
        """Say what this leg did, including when it did nothing.

        Unconditional, for the reason K3's counters exist: a run that produced
        exactly zero joint hits otherwise produces exactly zero lines saying
        so, and a by-construction null reads as a quiet success.
        """
        now = time.monotonic()
        if interval_s is None:
            interval_s = self._STATS_INTERVAL
        if not force and now - self._last_stats_log < interval_s:
            return
        self._last_stats_log = now
        logger.info(
            "ATOM LMCache offload: V4.1 state leg %s",
            ", ".join(f"{k}={v}" for k, v in sorted(self.stats().items())),
        )
