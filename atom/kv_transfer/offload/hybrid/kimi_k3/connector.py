# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 offload: dense paged KV plus the KDA per-request state tier.

K3 keeps a recurrent (KDA) state alongside its paged KV, and a prefix is only
resumable where both are available. This variant is the dense connector plus a
CPU tier for that state: same paged-KV path, one extra leg.
"""

from __future__ import annotations

import logging
import time

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadOperationId,
    StateStoreOperationId,
)
from atom.kv_transfer.offload._offload_common import (
    _SAVE_ABANDON_MARGIN_S,
    StateOffloadFace,
    max_pending_saves,
    offload_save_abandon_timeout_s,
)
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.hybrid.kimi_k3.tier_builder import build_state_tier
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    StateLoadSpec,
)
from atom.utils import envs

logger = logging.getLogger("atom")

#: Completion channels this variant owns. The generic aggregator transports
#: them opaquely and takes a failure-dominant TP quorum, which is what makes a
#: partial store resolve instead of pinning a key forever.
STATE_INDEX_CHANNEL = "k3_state_index"

#: The other half of a store's completion: the GPU has stopped reading the
#: checkpoint's PAGE units. Separate from the index channel because the units
#: are the KV pool's and are free as soon as the D2H drains, while whether the
#: CPU put succeeded is decided afterwards and cannot touch them.
STATE_SOURCE_CHANNEL = "k3_state_source"

#: Tag distinguishing a hash verdict from a store report on
#: `STATE_INDEX_CHANNEL`. Both are statements about one hash's membership in the
#: engine's index, so they share the channel rather than adding a third; the tag
#: is what keeps `connector_completion` from settling a store pin against a
#: load's verdict. The key is `("state_load", hash, load_key)` -- a plain
#: hashable tuple, which is what `ConnectorCompletion.operation_id` and the TP
#: aggregator's key require. `load_key` is the load's exact generation
#: (`_load_completion_id`) because the aggregator tombstones every
#: connector-completion key it takes quorum on; see `take_hash_verdicts`.
STATE_LOAD_VERDICT_TAG = "state_load"

#: The connector's standalone stall clock, used only when reclamation is
#: disabled (`LMCACHE_EC_PIN_TIMEOUT_SEC <= 0`) so there is no abandon window to
#: stay under. A save outstanding longer than this is a backend that stopped,
#: not one that is busy: a 4096-token store costs ~65ms.
_SAVE_STALL_DEFAULT_S = 120.0


def save_stall_seconds() -> float:
    """Seconds a save may sit before this connector calls the path stalled.

    Kept strictly under the scheduler's abandon window by construction. The two
    are complements on one deferred save: this connector releases the blocks of
    a save the backend never took, and `Scheduler._reconcile_stalled_deferred_
    saves` reclaims a save already handed out once its report is not coming --
    the scheduler docstring asserts this connector fires "on a shorter clock".
    A hardcoded 120 s broke that the moment `LMCACHE_EC_PIN_TIMEOUT_SEC` put the
    abandon window (`offload_save_abandon_timeout_s()` = pin + margin) below 120
    -- e.g. pin=60 gives abandon 90 < 120, inverting the order with nothing to
    detect it. Derive both from the one source instead: fire at LMCache's pin
    timeout itself (abandon minus the margin -- the point upstream has
    force-unpinned the source), but never above the 120 s default, so the
    ordering holds for every pin value and the default behaviour is unchanged.
    When reclamation is disabled (abandon <= 0) there is no window to stay
    under, so use the default.
    """
    abandon = offload_save_abandon_timeout_s()
    if abandon <= 0:
        return _SAVE_STALL_DEFAULT_S
    return min(_SAVE_STALL_DEFAULT_S, abandon - _SAVE_ABANDON_MARGIN_S)


class KimiK3OffloadConnector(DenseOffloadConnector):
    """Worker side: dense KV, plus spill/load of the per-request state."""

    # This connector owns a state tier and moves the per-request recurrent state
    # through it, so the dense codec must SKIP the state tensor rather than
    # reject it. The base rejects it (silent-wrong-output guard); we opt in here.
    _permit_per_request_state = True
    _supports_early_block_release = False

    def __init__(self, config) -> None:
        super().__init__(config)
        self._state_tier = None
        # Stores this worker could not attempt because the tier never built.
        # Drained in `get_finished` into the same completion channels a real tier
        # failure uses (index-failed + source-release), so the engine unpins
        # their PAGE units now. A *worker-process* field: writing a scheduler-only
        # attribute from here (as the earlier code did) raised AttributeError on
        # the no-tier store path and took the whole step's KV loads/saves down
        # with it (super().start_load_kv never ran).
        self._store_failed_no_tier: set[StateStoreOperationId] = set()

    def close(self) -> None:
        """Drain then join the state tier before the base executors.

        The tier's store/load threads copy PAGE units out of the KV pool. Draining
        first lets an in-flight transfer finish against a pool that is still
        mapped; `shutdown` then joins the tier's executors. Only after that
        does the base close its save/load pools. Guarded because the tier is
        `None` until `register_kv_caches` builds it (and stays `None` under PP or
        on a non-owning layout).
        """
        tier = self._state_tier
        if tier is not None:
            tier.drain()
            tier.shutdown()
        super().close()

    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        super().register_kv_caches(kv_caches, transfer_tensors, num_blocks)
        self._build_state_tier(transfer_tensors)

    # -- tier construction -------------------------------------------------
    def _build_state_tier(self, transfer_tensors) -> None:
        self._state_tier = build_state_tier(
            self._config, getattr(self, "_engine", None), transfer_tensors
        )

    # -- per-step ----------------------------------------------------------
    def start_load_kv(self, metadata) -> None:
        """Dispatch this step's state stores, then let the base dispatch the
        loads.

        Only the store half is here: a request's state LOAD rides its
        `LMCacheReqMeta`, and the load task super() dispatches for it submits
        and joins the state leg itself (`_do_load_req`).

        `_start_state_stores` isolates each submit, but a raise escaping it
        would skip super() and drop every KV load and save this step, so it runs
        in a try/finally rather than on trust.
        """
        if not isinstance(metadata, LMCacheOffloadMetadata):
            super().start_load_kv(metadata)
            return
        try:
            self._start_state_stores(metadata)
        finally:
            super().start_load_kv(metadata)

    # -- copy daemon thread ------------------------------------------------
    def _do_load_req(self, req: LMCacheReqMeta) -> None:
        """Both legs of one request's load, with one completion.

        The state leg is submitted to the tier's own lane first and joined after
        the KV leg, so a joint load costs the longer leg rather than the sum.
        The join is on every path, a raise included: reporting before the state
        H2D has landed would let the request resume, or recompute, over a slot
        the tier is still writing. If the KV leg failed, the request recomputes
        from 0 and never reads the state it restored, so running the legs
        concurrently cannot resume a history that has no KV behind it.
        """
        spec = req.state_load_spec
        state = None
        ok = False
        try:
            if spec is not None:
                state = self._submit_state_leg(req)
            ok = self._load_kv_bytes(req)
        except Exception:
            logger.warning(
                "kimi_k3 offload: load failed for req=%s", req.req_id, exc_info=True
            )
            ok = False
        if spec is not None:
            if state is None:
                ok = False
            else:
                try:
                    ok = bool(state.result()) and ok
                except Exception:
                    logger.warning(
                        "kimi_k3 offload: state leg failed for req=%s",
                        req.req_id,
                        exc_info=True,
                    )
                    ok = False
        self._finish_load(req, ok)

    def _submit_state_leg(self, req: LMCacheReqMeta):
        """Hand the state leg to the tier, or refuse it; None when refused.

        A refused leg still reports a verdict on its key: the TP quorum acts
        only on a key every rank reported, and a rank that stays silent leaves
        it one report short forever. Neutral, not a miss -- this rank never
        asked, so it holds no evidence LMCache dropped the bytes -- and only on
        this path, never ahead of a leg that is about to run (see
        `StateOffloadTier.note_load_unrun`).
        """
        spec = req.state_load_spec
        tier = self._state_tier
        key = self._load_completion_id(req)
        if tier is None:
            # No verdict ledger on this rank. It also never indexed a hash: its
            # stores fail through `_store_failed_no_tier` and the store quorum
            # is failure-dominant, so no load reaches here unless ranks disagree
            # about building a tier, which the engine gate decides globally.
            logger.warning(
                "kimi_k3 offload: no state tier on this rank; failing the load "
                "for req=%s so it recomputes.",
                req.req_id,
            )
            return None
        try:
            if not self._state_leg_valid(req):
                tier.note_load_unrun(spec.boundary_hash, key)
                return None
            return tier.submit_load(spec.boundary_hash, int(spec.destination_slot), key)
        except Exception:
            tier.note_load_unrun(spec.boundary_hash, key)
            raise

    def _state_leg_valid(self, req: LMCacheReqMeta) -> bool:
        """Refuse a leg the engine and worker disagree about; never clamp one.

        Coverage: the state image is the history of `[0, boundary_tokens)`, so
        a joint load's KV leg must complete at least that far. A state-only
        request carries no KV leg; its prefix is resident by the engine's own
        condition (`boundary <= num_cached_tokens`). And `seq.state_slot` is -1
        once `state_slots` is empty, which `codec.get` would negative-index.
        """
        spec = req.state_load_spec
        ls = req.load_spec
        if ls is not None and int(spec.boundary_tokens) > int(ls.lmcache_cached_tokens):
            logger.error(
                "kimi_k3 offload: req=%s state boundary %d is past the %d tokens "
                "the KV leg covers; refusing the state leg so the request "
                "recomputes.",
                req.req_id,
                spec.boundary_tokens,
                ls.lmcache_cached_tokens,
            )
            return False
        if int(spec.destination_slot) < 0:
            logger.error(
                "kimi_k3 offload: req=%s has no state slot (%d); refusing the "
                "state leg so the request recomputes.",
                req.req_id,
                spec.destination_slot,
            )
            return False
        return True

    def _start_state_stores(self, metadata) -> None:
        """Hand this step's ready checkpoints to the tier's executor.

        No producer fence and no staging copy: the source is the checkpoint's
        PAGE units, reserved out of the KV pool and pinned by the engine, so the
        packer gathers straight from where they sit.

        A store with no tier is reported failed rather than dropped -- the
        engine holds those units pinned against a report, and silence would
        leave them to the reconciler's full timeout.
        """
        stores = getattr(metadata, "state_stores", None)
        if not stores:
            return
        if self._state_tier is None:
            logger.warning(
                "kimi_k3 offload: %d state store(s) with no tier; failing them "
                "so the engine releases their units now rather than on timeout.",
                len(stores),
            )
            for op, _units in stores:
                self._store_failed_no_tier.add(op)
            return
        for op, unit_ids in stores:
            # Isolate each submit (finding #4): route a submit failure into the
            # same no-tier failure channel (get_finished drains it) so the engine
            # releases the store's pinned PAGE units now rather than on the
            # reconciler's full timeout, without escaping past super().
            try:
                self._state_tier.submit_store(op, tuple(int(u) for u in unit_ids))
            except Exception:
                logger.exception(
                    "kimi_k3 offload: submit_store failed for %s; failing it.",
                    op,
                )
                self._store_failed_no_tier.add(op)

    # -- completions -------------------------------------------------------
    def get_finished(self):
        out = super().get_finished()
        # No-tier store failures must reach the engine even when the tier never
        # built -- the engine pinned their PAGE units and is holding them
        # against a report. Emit the tier's own failure pairing: index-failed so
        # the aggregator takes quorum instead of waiting for a second report,
        # plus a source-release so the units are freed now rather than on the
        # reconciler's full timeout. Drained BEFORE the tier-None early return
        # below, because that is exactly the case that populated this set.
        if self._store_failed_no_tier:
            for op in self._store_failed_no_tier:
                out.connector_completions.add(
                    ConnectorCompletion(STATE_INDEX_CHANNEL, op, False)
                )
                out.connector_completions.add(
                    ConnectorCompletion(STATE_SOURCE_CHANNEL, op, True)
                )
            self._store_failed_no_tier = set()
        if self._state_tier is None:
            return out
        indexed, index_failed = self._state_tier.take_store_reports()
        # Store reports have no request identity (the owner is long gone), so
        # they ride the connector-owned channel with its failure-dominant quorum.
        # Keyed by operation, not bare hash: `KVOutputAggregator` tombstones each
        # `(channel, operation_id)` it takes quorum on, so a bare hash made the
        # second store of a re-evicted prefix a dropped duplicate -- its pin
        # waited for stale reclamation and the CPU index never learned it was back.
        for op in indexed:
            out.connector_completions.add(
                ConnectorCompletion(STATE_INDEX_CHANNEL, op, True)
            )
        for op in index_failed:
            out.connector_completions.add(
                ConnectorCompletion(STATE_INDEX_CHANNEL, op, False)
            )
        for op in self._state_tier.take_source_releases():
            out.connector_completions.add(
                ConnectorCompletion(STATE_SOURCE_CHANNEL, op, True)
            )
        # The ONLY event that may retract a hash from the engine's index. The
        # fused load report cannot: a verdict of "failed" may mean the KV leg
        # while the state bytes are present and untouched, and retracting on
        # that would permanently deny state that is still there. Successes ride
        # the same channel because the TP quorum acts only on a key every rank
        # reported.
        # The key carries the load generation as well as the hash. The
        # aggregator tombstones every connector-completion key it takes quorum
        # on, and drops a report whose key is already tombstoned, so a bare hash
        # is reportable exactly once per process, and a bare request id once per
        # request. `_load_completion_id` is the same on every rank, so the quorum
        # is unchanged.
        for (h, load_key), ok in self._state_tier.take_hash_verdicts().items():
            out.connector_completions.add(
                ConnectorCompletion(
                    STATE_INDEX_CHANNEL,
                    (STATE_LOAD_VERDICT_TAG, int(h), load_key),
                    bool(ok),
                )
            )
        return out


class KimiK3OffloadScheduler(DenseOffloadScheduler, StateOffloadFace):
    """Scheduler side: dense KV, plus the state tier's store queue, the state
    leg attached to each load, and the save-stall guard that keeps a stopped
    backend from stopping the engine.

    Inherits `StateOffloadFace` -- the only offload scheduler that hosts the KDA
    state tier -- so routing can select it with `isinstance` rather than probing
    for a method the delegating shell defines on every layout (see
    `StateOffloadFace` and the shell's `has_state_tier`)."""

    _supports_early_block_release = False

    def __init__(self, config) -> None:
        super().__init__(config)
        self._pending_state_stores: list[tuple] = []
        # sid -> seq for sequences whose state leg still has to be attached to
        # this step's metadata. Recorded at `update_state_after_alloc`, where
        # the pending load is visible, and consumed by `build_connector_meta`.
        self._state_load_seqs: dict[str, object] = {}
        # sid -> seq for STATE-ONLY loads: KV resident up to the boundary, state
        # in the tier. Recorded when the park is decided and emitted by the
        # next `build_connector_meta` as a request of their own, with no KV leg.
        self._state_only_loads: dict[str, object] = {}
        # Hashes whose state `get` missed on some rank. Drained by the engine,
        # which is the only owner of the index that advertises them.
        self._state_load_missed: set[int] = set()
        # A finished request whose save is queued keeps its blocks pinned
        # (`should_defer_free`), so the queue depth is also how much of the pool
        # a slow backend can hold. Same knob and default the DSV4 layout bounds
        # its worker queue with, read from the other end.
        self._max_pending_saves = max_pending_saves(
            getattr(config, "kv_transfer_config", None) or {},
            envs.OFFLOAD_COPY_WORKERS,
        )
        self._save_inflight_since: dict[object, float] = {}
        self._save_stalled = False
        self._warned_save_stalled = False
        # Channel reports drained by the engine each step. No-tier store
        # failures arrive here too, via the worker's STATE_INDEX_CHANNEL /
        # STATE_SOURCE_CHANNEL completions -- the worker cannot write a
        # scheduler field directly (different process), so there is no
        # engine-side "failed locally" set to merge.
        self._state_indexed: set = set()
        self._state_index_failed: set = set()
        self._state_source_released: set = set()

    # -- state store queue -------------------------------------------------
    def enqueue_state_stores(self, stores) -> bool:
        if not stores:
            return False
        self._pending_state_stores.extend(stores)
        return True

    def update_state_after_alloc(self, seq) -> None:
        """Base KV bookkeeping, plus arming this seq's state leg."""
        super().update_state_after_alloc(seq)
        if not self._do_load:
            return
        joint = getattr(seq, "offload_joint", None)
        if joint is None or int(getattr(joint, "load_hash", -1)) == -1:
            return
        if getattr(seq, "offload_load_cancelled", False):
            # Under `kv_connector: multi` this sub lost the KV arbitration and
            # the composite withdrew its load. The state arm reads
            # `seq.offload_joint`, which the engine owns and the cancel does not
            # reach, so without this it would put a second writer into the
            # winner's request. Whether the state leg then travels at all is the
            # scheduler's call (`carries_state_load`).
            return
        self._state_load_seqs[str(seq.id)] = seq

    def _is_state_only(self, seq) -> bool:
        """KV resident up to the state boundary; only the state is in the tier."""
        joint = getattr(seq, "offload_joint", None)
        return (
            getattr(seq, "has_per_req_cache", False)
            and joint is not None
            and int(joint.load_hash) != -1
            and int(joint.boundary_tokens or 0) <= int(seq.num_cached_tokens)
        )

    def should_park_for_load_after_alloc(self, seq) -> bool:
        """A state-only load parks on a request of its own, with no KV leg.

        Expressing it as a no-op KV spec instead made the shared KV machinery
        special-case it -- the build-time pin check, the transport end, the
        load-error blocks -- and shipped the whole resident prompt to every rank
        for a leg defined to move zero KV bytes.
        """
        if not (self._do_load and self._is_state_only(seq)):
            return super().should_park_for_load_after_alloc(seq)
        sid = str(seq.id)
        if sid not in self._state_load_seqs:
            # Never armed: a composite withdrew this sub's load.
            return False
        # A KV spec the lookup left behind would move nothing the boundary
        # needs; its lookup id stays queued, so the dispatch still unpins it.
        self._clear_pending_load(sid)
        self._state_only_loads[sid] = seq
        seq.offload_loaded_tokens = int(seq.num_cached_tokens)
        return True

    def _clear_pending_load(self, sid: str) -> None:
        super()._clear_pending_load(sid)
        self._state_only_loads.pop(sid, None)

    def _state_load_spec(self, seq) -> StateLoadSpec:
        joint = seq.offload_joint
        return StateLoadSpec(
            boundary_tokens=int(joint.boundary_tokens or 0),
            boundary_hash=int(joint.load_hash),
            destination_slot=int(seq.state_slot),
        )

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        self._refresh_save_stall()
        meta = super().build_connector_meta()
        # Drained, not copied: a second submission would store the same image
        # twice, and the second report would unpin a record the first already
        # released.
        meta.state_stores = self._pending_state_stores
        self._pending_state_stores = []
        # Joint loads: attach the state leg to the KV request dense built. A
        # post-pass rather than a fork of dense's builder loop, which is how a
        # layout stops receiving dense's fixes.
        for req in meta.requests:
            seq = self._state_load_seqs.pop(str(req.req_id), None)
            if seq is None or req.load_spec is None:
                # No arm, or a save-only request, which `start_load_kv` gives no
                # load task: a spec attached there would never travel.
                continue
            if int(seq.offload_joint.load_hash) == -1:
                continue
            req.state_load_spec = self._state_load_spec(seq)
        # Anything left never reached the metadata as a KV load. A parked joint
        # load dropped there is reported by the base (`take_undispatched_loads`).
        self._state_load_seqs.clear()
        # State-only loads: a request of their own, identified exactly like a
        # KV load so the one completion settles it.
        for sid, seq in self._state_only_loads.items():
            if int(seq.offload_joint.load_hash) == -1:
                continue
            operation = LoadOperationId(seq.id, self._load_nonce)
            self._load_nonce += 1
            seq._load_operation = operation
            self._active_load_operations[sid] = (seq, operation)
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=[],
                    block_ids=[],
                    load_operation=operation,
                    state_load_spec=self._state_load_spec(seq),
                )
            )
        self._state_only_loads.clear()
        return meta

    def _may_emit_save(self) -> bool:
        """Nothing new goes out while the backend is stalled, and never more
        than `OFFLOAD_MAX_PENDING_SAVES` requests pinned at once."""
        return (
            not self._save_stalled
            and len(self._save_inflight) < self._max_pending_saves
        )

    def has_pending_work(self) -> bool:
        """Base KV liveness, plus this variant's state queues.

        `DenseOffloadScheduler.has_pending_work` ORs only the KV load/save
        trackers, so a step whose only outstanding work is a last-of-burst state
        checkpoint reads as idle -- and the engine can stop stepping before the
        tier is ever handed that work. The queue is drained into metadata every
        `build_connector_meta`, so OR-ing it keeps the predicate monotone: it
        goes False the step after the work is dispatched and never latches the
        busy loop.
        """
        return (
            super().has_pending_work()
            or bool(self._pending_state_stores)
            or bool(self._state_only_loads)
        )

    # -- save stall --------------------------------------------------------
    def _refresh_save_stall(self) -> None:
        """Decide whether the save path has stopped draining.

        Ages are tracked off `_save_inflight` rather than stamped at emission,
        so this needs no hook inside the base class's save loop.
        """
        now = time.monotonic()
        # A long prefill can emit many consecutive saves for the same request.
        # Age the exact operation, not the request across completed generations.
        inflight = set(self._save_inflight.values())
        for operation in inflight - set(self._save_inflight_since):
            self._save_inflight_since[operation] = now
        for operation in set(self._save_inflight_since) - inflight:
            del self._save_inflight_since[operation]
        if not self._save_inflight_since:
            if self._save_stalled:
                logger.info("kimi_k3 offload: save path draining again")
            self._save_stalled = False
            self._warned_save_stalled = False
            return
        oldest = min(self._save_inflight_since.values())
        self._save_stalled = (now - oldest) > save_stall_seconds()
        if self._save_stalled and not self._warned_save_stalled:
            self._warned_save_stalled = True
            logger.warning(
                "kimi_k3 offload: no save completed in %.0fs (%d in flight); "
                "releasing the blocks of requests whose save was never sent.",
                now - oldest,
                len(self._save_inflight),
            )

    def abandon_save(self, req_id) -> None:
        """Drop a reclaimed save, then recompute the stall latch.

        K3 ages saves off `_save_inflight` (`_refresh_save_stall`). Once the
        base drops the inflight entry, re-run the refresh so the age index sheds
        the abandoned sid (its `_save_inflight_since` sweep) and the stall latch
        clears -- otherwise `_save_stalled` stays stuck True on a save that is
        already gone.
        """
        super().abandon_save(req_id)
        self._refresh_save_stall()

    def _save_is_stall_escaped(self, seq) -> bool:
        """Whether a stalled, never-dispatched save lets these blocks go free.

        A save already handed out is reading these blocks, so freeing them would
        let the next request write into them mid-transfer and index the result
        under this prefix's hash. One never handed out has no reader, and holding
        it is what turns a stopped backend into a stopped engine.

        The handed-out save is left to `Scheduler._reconcile_stalled_deferred_saves`,
        on a longer clock tied to LMCache's force-unpin window. Neither subsumes
        the other: this asks whether the backend ever took the save, that whether
        it ever answered.
        """
        sid = str(seq.id)
        return (
            self._save_stalled
            and sid not in self._save_inflight
            and self._has_pending_save(seq)
        )

    def should_defer_free(self, seq) -> bool:
        """Pure query: base behaviour, plus the stall escape.

        An active load is checked *first*: the escape must not override the base
        holding blocks under a live load, or releasing mid-load is
        free-while-writing corruption (a stalled-save request can still have one).

        Predicate only -- `_is_preemptable`/`_maybe_release_deferred` probe it, so
        it must not mutate. The `_save_tracker` cleanup the escape needs on the
        preempt free is done by `release_stalled_save`; the finished path pops
        the tracker in `request_finished` (its `not should_defer_free` guard
        reads False here).
        """
        if self._has_active_load(seq):
            return True
        if self._save_is_stall_escaped(seq):
            return False
        return super().should_defer_free(seq)

    def release_stalled_save(self, seq) -> None:
        """Drop the tracker for a stall-escaped save whose blocks are being freed.

        The mutator half of the escape in `should_defer_free`. The scheduler
        calls this at `preempt`'s `block_manager.deallocate`, which runs no
        `request_finished`, so nothing else removes this sid. Left in, the save
        loop would later (once the stall clears) emit a save reading a now-freed,
        possibly-reused `block_table` -- silent cross-prefix corruption, as the
        loop does not re-check liveness. Dropping the save is intended (this KV
        goes un-offloaded rather than wedging the engine); guarded by the same
        escape predicate so a non-stalled save is never dropped.
        """
        if self._save_is_stall_escaped(seq):
            self._save_tracker.pop(str(seq.id), None)

    # -- joint boundary ----------------------------------------------------
    def _decide_load_after_alloc(self, seq, ls):
        """Clamp a hybrid's KV leg to the boundary the state leg is aimed at.

        A hybrid's per-request state is the compressed history of exactly
        `[0, hbm)`. Raising the KV-loaded length past that would have the
        forward skip `[hbm, lmc)` while the linear layers never see it: wrong
        output, no exception. So a hybrid loads only when `can_allocate` picked
        one boundary for both legs (`_joint_kv_boundary`), and this clamps the
        KV leg down to it.
        """
        if not getattr(seq, "has_per_req_cache", False):
            return super()._decide_load_after_alloc(seq, ls)

        hbm = int(seq.num_cached_tokens)
        lmc = int(ls.lmcache_cached_tokens)
        chunk = self.chunk_size or 256
        joint = int(seq.offload_joint.boundary_tokens or 0)
        if joint <= hbm:
            return False, "per_req_cache_state_boundary", hbm, lmc, lmc - hbm, chunk
        # Where the transfer starts -- NOT where the request may call itself
        # cached. `allocate` claimed every matched block, not just resumable
        # ones, so the KV below this is already resident; asking LMCache to
        # resend it would land a second copy in HBM (`publish_loaded_prefix`
        # keeps the canonical mapping, fresh blocks stay private). Floored to the
        # chunk grid by `_joint_kv_boundary`, so aligned whenever `hbm` was.
        start = max(hbm, int(seq.offload_joint.claim_tokens or 0))
        # The KV leg moves whole chunks and the blocks below `start` are shared,
        # so an unaligned start cannot be rounded down.
        if start % chunk != 0:
            return False, "joint_unaligned_hbm_prefill", start, lmc, lmc - start, chunk
        # Transfer the chunk covering the boundary, claim only the boundary.
        kv_target = int(seq.offload_joint.kv_tokens or 0) or joint
        if joint > lmc or kv_target > lmc:
            return False, "joint_boundary_above_lookup", start, lmc, lmc - start, chunk
        if kv_target <= start:
            # The whole boundary is already resident. Unreachable while
            # `_gated_hit` returns the rightmost rung -- a boundary at or below
            # the compressed hit would have been the plain hit -- but a state
            # leg with no KV leg is a shape this must not emit silently.
            return False, "joint_kv_already_resident", start, lmc, 0, chunk
        # Both ends of the transfer, together: the base class writes the start
        # back on its own path and the worker reads `[hbm_cached_tokens,
        # lmcache_cached_tokens)`, so leaving the start at the value the lookup
        # recorded would fetch from token 0 every time.
        ls.hbm_cached_tokens = start
        ls.lmcache_cached_tokens = kv_target
        # Deliberately past the min-load floor: the boundary was chosen for both
        # legs, and refusing on size would leave the state leg claiming a prefix
        # whose KV never came.
        return True, "joint_state_and_kv", start, kv_target, kv_target - start, chunk

    def _claim_after_load(self, seq, hbm: int, lmc: int) -> int:
        """How far the request may call itself cached once the load lands.

        For a joint load that is the *state* boundary, which sits at or below
        the transfer's end: the KV leg is aimed at the chunk covering it, and
        claiming the rounded-up figure would have the forward skip tokens the
        recurrent state does not cover.
        """
        joint = int(seq.offload_joint.boundary_tokens or 0)
        return max(hbm, min(joint, lmc)) if joint else max(hbm, lmc)

    # -- connector-owned channels -----------------------------------------
    def connector_completion(self, completion) -> bool | None:
        # Return-value contract (`_offload_common.process_completions`):
        # `False` = not this connector's channel, `True` = handled AND a
        # terminal save for the scheduler's deferred-free, `None` = handled
        # non-terminal milestone. Every channel below is a state-tier milestone
        # whose operation id is a hash or a store id, never a request id, so
        # every one of them returns `None`. Returning `True` fed those ids into
        # `KVConnectorOutput.finished_saving`, where the scheduler looked each
        # up as a request id and silently found nothing -- harmless by luck
        # (request ids are strings), but a lie about what had finished saving.
        if completion.channel == STATE_SOURCE_CHANNEL:
            self._state_source_released.add(completion.operation_id)
            return None
        if completion.channel == STATE_INDEX_CHANNEL:
            op = completion.operation_id
            if isinstance(op, tuple) and op and op[0] == STATE_LOAD_VERDICT_TAG:
                # A load's own verdict on a hash, not a store's. Only a miss is
                # evidence LMCache dropped the bytes, so only a miss is recorded
                # -- the fused load report says nothing about which leg failed.
                if not completion.succeeded:
                    self._state_load_missed.add(int(op[1]))
                return None
            target = (
                self._state_indexed
                if completion.succeeded
                else self._state_index_failed
            )
            target.add(op)
            return None
        # Channels this connector does not own go to the base, which DOES
        # define `connector_completion` (`ChunkedOffloadSchedulerBase`, since
        # #1953) and owns the dense PAGE channels. Returning a bare `False`
        # here -- on the claim that no base defined the method -- swallowed
        # them: this class inherits its save path from `DenseOffloadConnector`,
        # which emits `dense.page.source_safe` and `dense.page.store` under
        # early block release, and the scheduler half needs both to mark a
        # staging group source-safe and to retire a store. Dropped, they were
        # logged as "unhandled channel" and the deferred blocks were never
        # released. The base still answers `False` for a channel nobody owns,
        # so the caller's contract is unchanged.
        return super().connector_completion(completion)

    def take_state_source_releases(self) -> set:
        """Drain the stores whose PAGE units the GPU has finished reading.

        A method of its own rather than a third element of
        `take_state_reports`: that tuple's arity is a contract between this
        class and the delegating shell's fallback, and widening it once
        already cost every TP worker in the pool.
        """
        released = self._state_source_released
        self._state_source_released = set()
        return released

    def take_state_reports(self) -> tuple[set[int], set[int]]:
        """Drain this step's tier store reports for the engine-side index.

        Both real tier failures and no-tier worker failures land in
        `_state_index_failed` via STATE_INDEX_CHANNEL, so there is a single
        source of truth to drain.
        """
        indexed = self._state_indexed
        failed = self._state_index_failed
        self._state_indexed = set()
        self._state_index_failed = set()
        return indexed, failed

    def take_missed_state_hashes(self) -> set[int]:
        """Drain the hashes whose state `get` missed on some rank.

        The engine forgets these, and nothing else may: a fused load verdict of
        "failed" can mean the KV leg while the state bytes are present, and
        retracting on that would permanently deny state that is still there.
        """
        missed = self._state_load_missed
        self._state_load_missed = set()
        return missed
