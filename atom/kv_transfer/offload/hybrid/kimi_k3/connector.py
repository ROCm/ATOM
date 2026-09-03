# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 offload: dense paged KV plus the KDA per-request state tier.

K3 keeps a recurrent (KDA) state alongside its paged KV, and a prefix is only
resumable where both are available. This variant is the dense connector plus a
CPU tier for that state: same paged-KV path, one extra leg.
"""

from __future__ import annotations

import logging
import os
import time

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    StateStoreOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import (
    _SAVE_ABANDON_MARGIN_S,
    StateOffloadFace,
    max_pending_saves,
    offload_save_abandon_timeout_s,
    pp_aware_rank_and_world,
)
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.hybrid.kimi_k3.staging import StagedTransfer
from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import StateByteCodec
from atom.kv_transfer.offload.hybrid.kimi_k3.state_tier import StateOffloadTier
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    StateLoadSpec,
)

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
#: load's verdict. `("state_load", hash)` is a plain hashable tuple, which is
#: what `ConnectorCompletion.operation_id` and the TP aggregator's key require.
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
        mapped; `shutdown` then joins the tier's own executors. Only after that
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
        # The config refusals come first, and deliberately above the aiter
        # import: whether this configuration can host a tier at all is decided
        # by config alone, so it must not depend on a GPU library being
        # importable -- otherwise the refusal cannot be reached, or tested, on a
        # CPU runner.
        # PP breaks the tier: the CacheEngineKey has no PP component, so two
        # stages at the same TP rank would overwrite each other. Refused rather
        # than half-supported. Paged KV is unaffected.
        pp_size = int(getattr(self._config, "pipeline_parallel_size", 1) or 1)
        if pp_size > 1:
            # Refused loudly rather than warned about. `CacheEngineKey` carries
            # no PP component, so two stages at one TP rank would overwrite each
            # other's state images. The engine agrees independently
            # (`state_tier_capability`), so both legs of a K3 request are then
            # declined and the offload does nothing at all -- a server started
            # with `--kv-transfer-config` and `pp_size > 1` would serve at
            # baseline speed while its operator believed offload was on, and one
            # warning line among thousands is not how that gets noticed.
            raise ValueError(
                "kimi_k3 offload: the recurrent-state tier does not support "
                f"pipeline parallelism (pipeline_parallel_size={pp_size}). The "
                "LMCache key carries no PP component, so two stages at one TP "
                "rank would overwrite each other's state images, and with the "
                "tier off a K3 request's KV leg is declined too -- the offload "
                "would be inert. Run with pipeline_parallel_size=1, or drop "
                "--kv-transfer-config."
            )

        backend = getattr(transfer_tensors, "state_backend", None)
        if backend is None:
            logger.warning(
                "kimi_k3 offload: no attention backend published; state tier off."
            )
            return
        # The geometry the bytes are written under, folded into every key so a
        # changed build cannot read another's images. Read from the runtime, not
        # recomputed, so HBM and CPU sides share one owner of the string.
        spec = getattr(getattr(backend, "model_runner", None), "state_runtime", None)
        spec = getattr(spec, "checkpoint_spec", None)
        layout_id = getattr(spec, "layout_id", None)
        if not layout_id:
            logger.warning(
                "kimi_k3 offload: no checkpoint layout id published; state tier "
                "off. Without it a build that changed the state geometry could "
                "read another's images back as valid."
            )
            return
        try:
            views = backend.state_entry_views(0)
            entry_bytes = sum(int(v.numel()) * v.element_size() for v in views)
        except (NotImplementedError, AttributeError):
            # No per-request state on this backend. IndexError is deliberately
            # not caught: a zero-entry pool with the tier on is a sizing bug.
            logger.warning(
                "kimi_k3 offload: %s owns no per-request state views; tier off.",
                type(backend).__name__,
            )
            return

        # The store reads `page_unit_views`, a *different* method than the load's
        # `state_entry_views` (validated above). A backend with one but not the
        # other would build the tier, pass every load, then AttributeError on the
        # first store -- which the tier's blind `except` masks as an endlessly
        # "failed" store. Probe it here so the mismatch fails fast and visibly.
        if not callable(getattr(backend, "page_unit_views", None)):
            logger.warning(
                "kimi_k3 offload: %s has state_entry_views but no callable "
                "page_unit_views; the store path needs it, so a tier would fail "
                "every store silently. State tier off.",
                type(backend).__name__,
            )
            return

        # The store reads PAGE units and the load writes an Active Slot, so the
        # blob must be the same length both ways (equal for K3: a checkpoint
        # covers the whole slot). A model where they differ would truncate the
        # store or over-read the load -- checked here, where both are in scope.
        image_bytes = int(getattr(spec, "image_bytes", 0) or 0)
        if image_bytes and image_bytes != entry_bytes:
            logger.warning(
                "kimi_k3 offload: a checkpoint image is %d B but an Active Slot "
                "is %d B; the store reads units and the load writes a slot, so "
                "they must match. State tier off.",
                image_bytes,
                entry_bytes,
            )
            return

        # Imported here, at its one use site: everything above is a refusal
        # decidable without a GPU library.
        from aiter.dist.parallel_state import get_tp_group

        tp = get_tp_group()
        rank, world = pp_aware_rank_and_world(self._config, tp)
        cfg = offcfg.build_lmcache_config(
            getattr(self._config, "kv_transfer_config", None)
        )
        meta = offcfg.build_lmcache_metadata(self._config, cfg, world, rank)

        # One flat entry, packed on the tier's own `lmc-state` thread. Sized to
        # the entry rather than shared with the KV staging buffer, which is
        # sized in LMCache chunks and is routinely an order of magnitude smaller.
        gpu_connector = self._engine.gpu_connector
        staged = StagedTransfer(
            gpu_connector.device,
            staging_buffer_bytes=entry_bytes,
            release_after_transfer=gpu_connector.release_gpu_staging_after_transfer,
        )
        codec = StateByteCodec(
            backend,
            staged,
            entry_bytes,
            model_name=meta.model_name,
            world_size=world,
            worker_id=rank,
            layout_id=layout_id,
        )
        # ONE pool, shared with paged KV. A request writes its KV chunks and its
        # one state object in the same prefill window, so both enter LMCache's
        # LRU together and cool at the same rate -- exactly right, since a joint
        # boundary needs both legs to survive together and a boundary whose KV is
        # gone is worthless. `LMCACHE_MAX_LOCAL_CPU_SIZE` is the one size to tune.
        codec.bind_storage_manager(self._engine.storage_manager)
        # No index here: StateOffloadIndex lives in the engine process; both
        # directions report and the engine applies.
        self._state_tier = StateOffloadTier(codec)
        logger.info(
            "kimi_k3 offload: state tier up, entry=%.2f MiB rank=%d, "
            "sharing the paged-KV CPU pool, layout=%s",
            entry_bytes / (1 << 20),
            rank,
            layout_id,
        )

    # -- per-step ----------------------------------------------------------
    def start_load_kv(self, metadata) -> None:
        """Dispatch this step's state stores, then let the base dispatch the
        loads.

        Only the store half is here now: a request's state LOAD rides its
        `LMCacheReqMeta` and runs inside the KV load task super() submits, so it
        needs no dispatch of its own.

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
        """Both legs of one request's load, in one task, with one completion.

        The state leg runs only if the KV leg landed: state at the boundary is
        the compressed history of exactly the prefix the KV leg was asked to
        complete, so restoring it over a prefix whose KV never arrived would
        have the forward resume on a history it does not hold -- silent wrong
        output rather than an error.

        `_finish_load` is reached on every path including a raise out of either
        leg, so one dispatch produces exactly one report whatever happens.
        """
        try:
            ok = self._load_kv_bytes(req)
            if ok and req.state_load_spec is not None:
                ok = self._load_state_bytes(req)
        except Exception:
            logger.warning(
                "kimi_k3 offload: load failed for req=%s", req.req_id, exc_info=True
            )
            ok = False
        self._finish_load(req, ok)

    def _load_state_bytes(self, req: LMCacheReqMeta) -> bool:
        """Restore this request's recurrent state into the slot it was given."""
        spec = req.state_load_spec
        tier = self._state_tier
        if tier is None:
            # Nothing on this rank can serve the leg, so the request must
            # recompute. Passing the KV leg through as a success instead would
            # have the engine count a state restore that never happened.
            logger.warning(
                "kimi_k3 offload: no state tier on this rank; failing the load "
                "for req=%s so it recomputes.",
                req.req_id,
            )
            return False
        return tier.load_state(spec.boundary_hash, spec.destination_slot)

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
        for h, ok in self._state_tier.take_hash_verdicts().items():
            out.connector_completions.add(
                ConnectorCompletion(
                    STATE_INDEX_CHANNEL, (STATE_LOAD_VERDICT_TAG, int(h)), bool(ok)
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

    def __init__(self, config) -> None:
        super().__init__(config)
        self._pending_state_stores: list[tuple] = []
        # sid -> seq for sequences whose state leg still has to be attached to
        # this step's metadata. Recorded at `update_state_after_alloc`, where
        # the pending load is visible, and consumed by `build_connector_meta`.
        self._state_load_seqs: dict[str, object] = {}
        # Hashes whose state `get` missed on some rank. Drained by the engine,
        # which is the only owner of the index that advertises them.
        self._state_load_missed: set[int] = set()
        # A finished request whose save is queued keeps its blocks pinned
        # (`should_defer_free`), so the queue depth is also how much of the pool
        # a slow backend can hold. Same knob and default the DSV4 layout bounds
        # its worker queue with, read from the other end.
        self._max_pending_saves = max_pending_saves(
            getattr(config, "kv_transfer_config", None) or {},
            int(os.environ.get("OFFLOAD_COPY_WORKERS", "1") or 1),
        )
        self._save_inflight_since: dict[str, float] = {}
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
        """Base KV bookkeeping, plus arming this seq's state leg.

        A STATE-ONLY load -- the KV is resident but the recurrent state is not --
        gets a no-op `LoadSpec` here so it travels the ORDINARY load path: same
        metadata, same worker task, same single completion. Giving it a path of
        its own is what previously required a second channel and a park to
        reconcile the two reports.
        """
        super().update_state_after_alloc(seq)
        if not self._do_load:
            return
        joint = getattr(seq, "offload_joint", None)
        if joint is None or int(getattr(joint, "load_hash", -1)) == -1:
            return
        sid = str(seq.id)
        self._state_load_seqs[sid] = seq
        if self._load_specs.get(sid) is not None:
            return
        hbm = int(getattr(seq, "num_cached_tokens", 0))
        self._load_specs[sid] = LoadSpec(
            hbm_cached_tokens=hbm, lmcache_cached_tokens=hbm, can_load=True
        )
        self._reqs_need_recv[sid] = seq

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        self._refresh_save_stall()
        meta = super().build_connector_meta()
        # Drained, not copied: a second submission would store the same image
        # twice, and the second report would unpin a record the first already
        # released.
        meta.state_stores = self._pending_state_stores
        self._pending_state_stores = []
        # A post-pass over what the base built, not a fork of its builder loop:
        # forking that loop is how a layout stops receiving dense's fixes.
        for req in meta.requests:
            seq = self._state_load_seqs.pop(str(req.req_id), None)
            if seq is None:
                continue
            joint = seq.offload_joint
            h = int(joint.load_hash)
            if h == -1:
                continue
            req.state_load_spec = StateLoadSpec(
                boundary_tokens=int(joint.boundary_tokens or 0),
                boundary_hash=h,
                destination_slot=int(seq.state_slot),
                chunk_tokens=int(self.chunk_size or 0),
            )
        # Anything left never reached the metadata (its KV leg was refused after
        # the arm), so drop it rather than attach it to a later step's request.
        self._state_load_seqs.clear()
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
        return super().has_pending_work() or bool(self._pending_state_stores)

    # -- save stall --------------------------------------------------------
    def _refresh_save_stall(self) -> None:
        """Decide whether the save path has stopped draining.

        Ages are tracked off `_save_inflight` rather than stamped at emission,
        so this needs no hook inside the base class's save loop.
        """
        now = time.monotonic()
        inflight = set(self._save_inflight)
        for sid in inflight - set(self._save_inflight_since):
            self._save_inflight_since[sid] = now
        for sid in set(self._save_inflight_since) - inflight:
            del self._save_inflight_since[sid]
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
        if joint <= hbm and int(seq.offload_joint.load_hash) != -1:
            # State-only load: the KV covering the boundary is already resident,
            # so the KV leg has nothing to move -- but it must still be emitted,
            # because the state leg rides this request's metadata and travels in
            # its task. `_load_kv_bytes` treats `lmc <= hbm` as the no-op
            # success it is. Refusing here instead would clear the load spec,
            # and the request would park on a transfer with no carrier.
            ls.hbm_cached_tokens = hbm
            ls.lmcache_cached_tokens = hbm
            return True, "state_only_load", hbm, hbm, 0, chunk
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
    def connector_completion(self, completion) -> bool:
        if completion.channel == STATE_SOURCE_CHANNEL:
            self._state_source_released.add(completion.operation_id)
            return True
        if completion.channel == STATE_INDEX_CHANNEL:
            op = completion.operation_id
            if isinstance(op, tuple) and op and op[0] == STATE_LOAD_VERDICT_TAG:
                # A load's own verdict on a hash, not a store's. Only a miss is
                # evidence LMCache dropped the bytes, so only a miss is recorded
                # -- the fused load report says nothing about which leg failed.
                if not completion.succeeded:
                    self._state_load_missed.add(int(op[1]))
                return True
            target = (
                self._state_indexed
                if completion.succeeded
                else self._state_index_failed
            )
            target.add(op)
            return True
        # Channels this connector does not own. `DenseOffloadConnector` and the
        # rest of the MRO define no `connector_completion`, so `super().` would
        # raise AttributeError; `False` is the caller's contract for "unhandled"
        # (see `_offload_common._apply_connector_completions`) and matches the
        # DSV4 sibling connector.
        return False

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
