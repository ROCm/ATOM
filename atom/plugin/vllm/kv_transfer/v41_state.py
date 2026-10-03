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
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from atom.model_engine.state_offload import StateOffloadIndex

from .kda_state import KdaLoad, KdaStore, boundary_prefix_hash

logger = logging.getLogger("atom")

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
        self._staging = torch.empty(
            (depth, self.entry_bytes), dtype=torch.uint8, device=cache.state_bytes.device
        )
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
                "V4.1 state offload: an entry is one region, got "
                f"{len(values)} ids"
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
            submitted.append((int(store.op_id), stage))
        if submitted:
            # One event for the whole batch of copies: they are all on this
            # stream and in order, so the last one completing means every slab
            # is readable. A fresh event each time -- an event reused across
            # steps is recorded and waited on by two different steps at once.
            event = torch.cuda.Event()
            event.record()
            for op_id, stage in submitted:
                store = next(s for s in stores if int(s.op_id) == op_id)
                self._tier.submit_store(
                    KdaStore(op_id, int(store.prefix_hash), (stage,)), event
                )
        return refused

    def release_reported(self, op_ids) -> None:
        """Give back the staging slabs of ops every rank has reported on."""
        for op_id in op_ids or ():
            stage = self._stage_of_op.pop(int(op_id), None)
            if stage is not None:
                self._views.release_stage(stage)


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
        chunk_size: int,
        state_interval: int,
        max_num_batched_tokens: int,
        world_size: int,
        can_store: bool = True,
        can_load: bool = True,
    ) -> None:
        self.hash_block_size = int(hash_block_size)
        self.chunk_size = int(chunk_size)
        self.state_interval = int(state_interval)
        self._world_size = max(1, int(world_size))
        self._index = StateOffloadIndex(can_store=can_store, can_load=can_load)
        self._next_op_id = 0
        self._pending_stores: dict[int, _PendingStore] = {}
        self._stores: list[KdaStore] = []
        self._loads: list[KdaLoad] = []
        self._lookup_ctx: tuple[str, Any] | None = None
        # req_id -> the last frontier this request was swept at, so one step's
        # frontier is offered once even if the connector sweeps twice.
        self._swept: dict[str, int] = {}
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
            "stage_full": 0,
            # The guard on the hard contract. Must stay zero; anything else
            # means a snapshot was taken at a position the scheduler did not
            # claim, and the store was dropped rather than keyed wrongly.
            "cursor_mismatch": 0,
            "cap_kept": 0,
            "cap_declined": 0,
            # The same two, for hits vLLM served from its own pool without
            # asking the connector. Kept apart because they answer different
            # questions: the connector pair measures the tier, this pair
            # measures what prefix caching costs on a model whose state the
            # block pool does not carry.
            "local_cap_kept": 0,
            "local_cap_declined": 0,
        }
        self._last_stats_log = 0.0
        if self.state_interval % self.chunk_size:
            raise ValueError(
                f"V4.1 state offload: state interval {self.state_interval} is "
                f"not a multiple of the LMCache chunk size {self.chunk_size}; "
                "a boundary would then fall inside a chunk and the PAGE prefix "
                "could not end where the state does"
            )
        if self.state_interval % self.hash_block_size:
            raise ValueError(
                f"V4.1 state offload: state interval {self.state_interval} is "
                f"not a multiple of the hash block size {self.hash_block_size}; "
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
        if hit <= 0 or self._lookup_ctx is None:
            return hit
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
        boundary = (hit // self.state_interval) * self.state_interval
        for _ in range(_MAX_CAP_DESCENT):
            if boundary <= 0:
                self._counters["cap_declined"] += 1
                return 0
            h = self.boundary_hash(block_hashes, boundary)
            if h is not None and self._index.could_serve(h):
                self._counters["cap_kept"] += 1
                return min(hit, boundary)
            boundary -= self.state_interval
        self._counters["cap_declined"] += 1
        return 0

    def cap_local_hit(self, request, hit: int) -> int:
        """Shorten vLLM's *own* prefix-cache hit to a claimed boundary.

        The connector-side `cap_hit` only sees hits vLLM asked it about. With
        prefix caching on, vLLM's block pool answers first and keeps whatever
        it can serve locally -- a path the connector is never consulted on. A
        request admitted that way arrives with its PAGE prefix restored and a
        state slot that was never written, which `_report_stale_state` turns
        into a dead engine.

        So the local hit is capped by the same rule and the same index. Unlike
        the connector path there is no armed lookup to read, so the request is
        the only context and is taken directly.
        """
        hit = int(hit)
        if hit <= 0:
            return hit
        # Refused outright, not shortened. Shortening a local hit to a
        # boundary the index holds looks right and is not: nothing then
        # *restores* that state. `resolve_load` runs only for tokens the
        # connector supplied, so a locally served prefix arrives with its
        # pages in HBM and a slot that was never written -- measured as
        # `needs state at 8192, found 0`, the cap having picked 8192 and
        # nobody having filled it.
        #
        # For this model PAGE reuse and STATE restore are one operation, and
        # only the connector performs both. Declining here does not lose the
        # reuse: the connector is asked about the whole prompt next and serves
        # the same prefix from the tier, with its state. What it costs is the
        # HBM-speed path, which for V4.1 was never admissible on its own.
        self._counters["local_cap_declined"] += 1
        return 0

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
                    frontier // self.state_interval
                    - previous // self.state_interval
                )
                if frontier % self.state_interval:
                    self._counters["boundary_passed"] += max(0, crossed)
                else:
                    # The one it landed on is not passed -- it is about to be
                    # offered.
                    self._counters["boundary_passed"] += max(0, crossed - 1)
            self._swept[req_id] = frontier
            if frontier <= 0 or frontier % self.state_interval:
                continue
            self._counters["sweep_offered"] += 1
            request = requests_by_id.get(req_id)
            if request is None:
                continue
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
        if boundary <= 0 or boundary % self.state_interval:
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

    def take_stores(self) -> list[KdaStore]:
        stores, self._stores = self._stores, []
        return stores

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

    # -- observability ---------------------------------------------------
    def stats(self) -> dict[str, int]:
        merged = dict(self._counters)
        index_stats = getattr(self._index, "stats", None)
        if callable(index_stats):
            merged.update(index_stats())
        return merged

    def log_stats(self, *, interval_s: float = 60.0, force: bool = False) -> None:
        """Say what this leg did, including when it did nothing.

        Unconditional, for the reason K3's counters exist: a run that produced
        exactly zero joint hits otherwise produces exactly zero lines saying
        so, and a by-construction null reads as a quiet success.
        """
        now = time.monotonic()
        if not force and now - self._last_stats_log < interval_s:
            return
        self._last_stats_log = now
        logger.info(
            "ATOM LMCache offload: V4.1 state leg %s",
            ", ".join(f"{k}={v}" for k, v in sorted(self.stats().items())),
        )
