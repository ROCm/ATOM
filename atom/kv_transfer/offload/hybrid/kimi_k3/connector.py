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
    max_pending_saves,
    pp_aware_rank_and_world,
)
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.hybrid.kimi_k3.staging import StagedTransfer
from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import StateByteCodec
from atom.kv_transfer.offload.hybrid.kimi_k3.state_tier import (
    StateOffloadTier,
    _JointPark,
)
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata

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

#: A save outstanding longer than this is a backend that stopped, not one that
#: is busy: a 4096-token store costs ~65ms.
SAVE_STALL_SECONDS = 120.0


class KimiK3OffloadConnector(DenseOffloadConnector):
    """Worker side: dense KV, plus spill/load of the per-request state."""

    # This connector owns a state tier and moves the per-request recurrent state
    # through it, so the dense codec must SKIP the state tensor rather than
    # reject it. The base rejects it (silent-wrong-output guard); we opt in here.
    _permit_per_request_state = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self._state_tier = None
        # Inert until a request has both legs, which only a joint boundary
        # produces; costs one dict lookup per report otherwise.
        self._joint_park = _JointPark()
        # Stores this worker could not even attempt because the tier never
        # built. Drained in `get_finished` into the SAME completion channels a
        # real tier failure uses (index-failed + source-release), so the engine
        # unpins their PAGE units now. This is a *worker-process* field; the
        # scheduler's report path lives in a different process and cannot see a
        # set written here -- the earlier code wrote a scheduler-only attribute
        # from the worker, which does not exist here and raised AttributeError
        # on the no-tier store path, taking the whole step's KV loads/saves down
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
        tier = getattr(self, "_state_tier", None)
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
        from aiter.dist.parallel_state import get_tp_group

        # PP breaks the tier: the CacheEngineKey has no PP component, so two
        # stages at the same TP rank would overwrite each other. Refused rather
        # than half-supported. Paged KV is unaffected.
        pp_size = int(getattr(self._config, "pipeline_parallel_size", 1) or 1)
        if pp_size > 1:
            logger.warning(
                "kimi_k3 offload: the state tier is unsupported under pipeline "
                "parallelism (pipeline_parallel_size=%d); paged KV is unaffected.",
                pp_size,
            )
            return

        backend = getattr(transfer_tensors, "state_backend", None)
        if backend is None:
            logger.warning(
                "kimi_k3 offload: no attention backend published; state tier off."
            )
            return
        # The geometry the bytes are written under, folded into every key so a
        # build that changed any of it cannot read another's images. Read from
        # the runtime rather than recomputed, so there is one owner of the
        # string and the HBM and CPU sides cannot disagree about it.
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

        # The load direction reads `state_entry_views` (validated just above);
        # the STORE direction reads `page_unit_views`, a *different* backend
        # method (StateByteCodec.put -> backend.page_unit_views). A backend that
        # implements one but not the other would build the tier and pass every
        # load, then AttributeError on the first store -- and the tier's blind
        # `except Exception` masks it as an endlessly "failed" store. Probe the
        # store method at construction so the mismatch fails fast and visibly.
        if not callable(getattr(backend, "page_unit_views", None)):
            logger.warning(
                "kimi_k3 offload: %s has state_entry_views but no callable "
                "page_unit_views; the store path needs it, so a tier would fail "
                "every store silently. State tier off.",
                type(backend).__name__,
            )
            return

        # The store reads the checkpoint's PAGE units and the load writes an
        # Active Slot, so the blob has to be the same length both ways. They are
        # equal for K3 (a checkpoint covers the whole slot), and a model where
        # they differ would silently truncate the store or over-read the load --
        # so it is asserted here rather than assumed, at the one point both
        # numbers are in scope.
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
        # ONE pool, shared with paged KV, and one `cache_policy` with it.
        # A request writes its KV chunks and its one state object inside the
        # same prefill window, so both enter LMCache's LRU together and cool at
        # the same rate -- state is retired alongside its own KV, which is what
        # we want, since a boundary whose KV is gone is worthless. Two pools
        # would buy independent eviction policies that must drift, while a joint
        # boundary needs both legs to survive together.
        # `LMCACHE_MAX_LOCAL_CPU_SIZE` is the one size to tune.
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
        if isinstance(metadata, LMCacheOffloadMetadata):
            self._arm_joint_loads(metadata)
            self._start_state_loads(metadata)
            self._start_state_stores(metadata)
        super().start_load_kv(metadata)

    def _arm_joint_loads(self, metadata) -> None:
        """Hold a request owning both legs until both report.

        Both legs surface on the KV completion channel, so an id in both would
        collapse into one wake and resume the suffix prefill while the other
        transfer is still writing.

        **The two legs do not report the same identity.** The KV worker reports
        `_load_completion_id(req)`, which is `req.load_operation` whenever the
        scheduler issued one -- and `build_connector_meta` issues one for every
        load, so in practice it is always the typed `LoadOperationId`. The state
        tier is keyed by request and reports the bare id. Arming under the bare
        id therefore parked nothing the KV leg could ever settle: `waits_for`
        was false for the `LoadOperationId`, the KV completion passed straight
        through, and the engine could resume the suffix prefill while the state
        H2D was still writing the Active Slot -- silent wrong output, and a park
        entry leaked per joint load. The park is filed under the KV identity,
        which is also the one that has to reach the engine.

        **No tier is armed exactly like a tier.** An earlier guard skipped the
        arm when `_state_tier is None`, reasoning that an unsettleable state leg
        left the park owing the KV leg forever. But `_start_state_loads` fails
        these loads (`_fail_state_loads` -> `settle_state(ok=False)`), and with
        the arm in place that failure lands on a real park entry -- the pair is
        marked failed and owes only the KV leg. `get_finished` then drains the
        park on the no-tier path too (before its `_state_tier is None` return),
        so the KV completion settles the pair into `failed_loading` and the
        request recomputes. Skipping the arm instead let the KV leg pass through
        as `finished_loading`, and `Scheduler._settle_state_load(ok=True)`
        counted a state restore that never happened -- silent wrong output. The
        arm is what makes the failure reportable; the drain is what keeps it
        from leaking (the KV completion always arrives and releases the entry).
        """
        loads = getattr(metadata, "state_loads", None) or ()
        state_ids = {req_id for req_id, _h, _slot in loads}
        if not state_ids or not self._do_load:
            return
        for req in metadata.requests:
            if req.req_id not in state_ids:
                continue
            if req.load_spec is not None:
                # Both legs: file the park under the KV identity, because that is
                # the one that has to reach the engine on finished/failed_loading.
                self._joint_park.arm(
                    req.req_id,
                    needs_kv=True,
                    needs_state=True,
                    kv_id=self._load_completion_id(req),
                )
            else:
                # State-only load (no KV leg -- a plain HBM-index miss where the
                # KV is resident but the recurrent state is not). Arm it too, on
                # the state leg alone. Its SUCCESS already reached the engine via
                # the _settle_joint passthrough (the tier reports the bare id and
                # the engine waits on it), but its FAILURE did not: _fail_state_
                # loads (including the no-tier path) settled a park that was never
                # armed, so nothing was emitted and the request sat in
                # WAITING_FOR_REMOTE_KVS for the life of the process --
                # reconcile_orphan_load_slots only reclaims slots of already-
                # deallocated seqs, it never wakes a live parked request. Arming
                # on the state leg alone makes both outcomes land on a real entry:
                # take_ready surfaces success into finished_loading and failure
                # into failed_loading (recompute), under the same bare id the
                # passthrough already used, so the engine matches it unchanged.
                self._joint_park.arm(
                    req.req_id,
                    needs_kv=False,
                    needs_state=True,
                )

    def _start_state_loads(self, metadata) -> None:
        """Hand this step's state loads to the tier's executor.

        No producer fence, unlike the save path: a load writes the entry, the
        owning request is parked so no forward touches it, and unpack
        synchronizes the producing stream before it returns.
        """
        loads = getattr(metadata, "state_loads", None)
        if not loads:
            return
        if self._state_tier is None:
            # The engine's index can outlive a tier that refused to build. Fail
            # them so the requests recompute rather than park forever.
            logger.warning(
                "kimi_k3 offload: %d state load(s) with no tier; failing them.",
                len(loads),
            )
            self._fail_state_loads(loads)
            return
        for req_id, h, group in loads:
            self._state_tier.submit_load(req_id, int(h), int(group))

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
            self._state_tier.submit_store(op, tuple(int(u) for u in unit_ids))

    def _fail_state_loads(self, loads) -> None:
        for req_id, _h, _group in loads:
            self._joint_park.settle_state(req_id, False)

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
            # Both joint and state-only loads were armed by `_arm_joint_loads`
            # even with no tier, and `_fail_state_loads` has already failed their
            # state leg. Drain the park here -- before the return -- with empty
            # tier reports (the tier that would produce them never built).
            #   * A joint load (needs_kv) now owes only the KV leg: its landed KV
            #     completion settles the pair into `failed_loading` and the
            #     request recomputes, instead of flowing through as
            #     `finished_loading` and being miscounted a state restore by
            #     `Scheduler._settle_state_load(ok=True)`. The KV completion also
            #     releases the entry, so the arm cannot leak.
            #   * A state-only load (no KV leg) was released the moment
            #     `_fail_state_loads` settled its only leg, so `take_ready`
            #     (inside `_settle_joint`) surfaces it into `failed_loading` now
            #     and the request recomputes -- rather than sitting in
            #     WAITING_FOR_REMOTE_KVS until an abandon window that never wakes
            #     a live parked request.
            out.finished_loading, out.failed_loading = self._settle_joint(
                out.finished_loading, out.failed_loading, set(), set()
            )
            return out
        indexed, index_failed = self._state_tier.take_store_reports()
        state_done, state_failed = self._state_tier.get_finished()
        out.finished_loading, out.failed_loading = self._settle_joint(
            out.finished_loading, out.failed_loading, state_done, state_failed
        )
        # Store reports have no request identity -- by the time one lands its
        # owner is long gone -- so they ride the connector-owned channel; the
        # aggregator's quorum is failure-dominant, which is what resolves a
        # partial store instead of pinning the key.
        #
        # The operation, not the bare hash. `KVOutputAggregator` tombstones
        # every `(channel, operation_id)` it has taken quorum on, so a hash
        # alone made the second store of a re-evicted prefix a duplicate: it
        # was dropped before quorum, its pin waited for stale reclamation, and
        # the CPU index never learned the bytes were back.
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
        return out

    def _settle_joint(self, kv_done, kv_failed, state_done, state_failed):
        """Merge the two report channels, holding armed pairs back.

        `waits_for` is asked first because `_settle` ignores ids it never
        armed, which is indistinguishable from a leg that landed.
        """
        park = self._joint_park
        passthrough_done: set = set()
        passthrough_failed: set = set()
        for settle, reports, ok in (
            (park.settle_kv, kv_done, True),
            (park.settle_kv, kv_failed, False),
            (park.settle_state, state_done, True),
            (park.settle_state, state_failed, False),
        ):
            for req_id in reports:
                if park.waits_for(req_id):
                    settle(req_id, ok)
                elif ok:
                    passthrough_done.add(req_id)
                else:
                    passthrough_failed.add(req_id)
        ready, ready_failed = park.take_ready()
        return passthrough_done | ready, passthrough_failed | ready_failed


class KimiK3OffloadScheduler(DenseOffloadScheduler):
    """Scheduler side: dense KV, plus the state tier's load queue and the
    save-stall guard that keeps a stopped backend from stopping the engine."""

    def __init__(self, config) -> None:
        super().__init__(config)
        # (req_id, state_hash, target_group) drained into each step's metadata.
        # A state load shares no shape with a KV transfer -- no token ids, no
        # block ids, no chunking -- only the park/report lifecycle.
        self._pending_state_loads: list[tuple] = []
        self._pending_state_stores: list[tuple] = []
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

    # -- state load queue --------------------------------------------------
    def enqueue_state_loads(self, loads) -> bool:
        if not loads:
            return False
        self._pending_state_loads.extend(loads)
        return True

    def enqueue_state_stores(self, stores) -> bool:
        if not stores:
            return False
        self._pending_state_stores.extend(stores)
        return True

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        self._refresh_save_stall()
        meta = super().build_connector_meta()
        # Drained, not copied: a second submission would write the same entry
        # into a group the first transfer is already filling.
        meta.state_loads = self._pending_state_loads
        self._pending_state_loads = []
        # Drained for the same reason: a second submission would store the same
        # image twice, and the second report would unpin a record the first
        # already released.
        meta.state_stores = self._pending_state_stores
        self._pending_state_stores = []
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
        trackers, so a step whose only outstanding work is a queued state load
        or a last-of-burst state checkpoint reads as idle -- and the engine can
        stop stepping before the tier is ever handed that work. Both queues are
        drained into metadata every `build_connector_meta`, so OR-ing them keeps
        the predicate monotone: it goes False the step after the work is
        dispatched and never latches the busy loop.
        """
        return (
            super().has_pending_work()
            or bool(self._pending_state_loads)
            or bool(self._pending_state_stores)
        )

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
        self._save_stalled = (now - oldest) > SAVE_STALL_SECONDS
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

        An active load is checked *first*: the base holds blocks while a load is
        still reading/writing them, and the escape must not override that -- a
        stalled-save request can still have a live load into the same table, and
        releasing mid-load is free-while-writing corruption.

        This is a predicate only. `_is_preemptable`/`_maybe_release_deferred`
        probe it, so it must not mutate. The `_save_tracker` cleanup the escape
        needs on the preempt free (`block_manager.deallocate`, no
        `request_finished`) is done by `release_stalled_save`, which the
        scheduler calls at that free; the finished path pops the tracker in
        `request_finished` (its `not should_defer_free` guard reads False here).
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
        `request_finished`, so nothing else removes this sid from
        `_save_tracker`. Left in, the base save loop would later (once the stall
        clears and `_may_emit_save` re-opens) emit a save reading a now-freed,
        possibly-reused `seq.block_table`: silent cross-prefix corruption. The
        loop does not re-check liveness, so the entry has to go the moment its
        blocks are surrendered. Dropping the save is the intended outcome (this
        request's KV goes un-offloaded rather than wedging the engine); guarded
        by the same escape predicate so a non-stalled save is never dropped.
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
        joint = int(getattr(seq, "state_joint_boundary_tokens", 0) or 0)
        if joint <= hbm:
            return False, "per_req_cache_state_boundary", hbm, lmc, lmc - hbm, chunk
        # Where the transfer starts, which is NOT where the request may call
        # itself cached. `allocate` claimed every block the prefix walk matched,
        # not just the resumable ones, so the KV below this is already in the
        # pool and asking LMCache to send it back would move bytes the GPU
        # holds -- and land a second copy of them in HBM, since
        # `publish_loaded_prefix` keeps the existing canonical mapping and the
        # freshly written blocks stay private to this request.
        #
        # `state_joint_claim_tokens` is floored to the chunk grid by
        # `_joint_kv_boundary`, so this start is aligned whenever `hbm` was.
        start = max(hbm, int(getattr(seq, "state_joint_claim_tokens", 0) or 0))
        # The KV leg moves whole chunks and the blocks below `start` are shared,
        # so an unaligned start cannot be rounded down.
        if start % chunk != 0:
            return False, "joint_unaligned_hbm_prefill", start, lmc, lmc - start, chunk
        # Transfer the chunk covering the boundary, claim only the boundary.
        kv_target = int(getattr(seq, "state_joint_kv_tokens", 0) or 0) or joint
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
        joint = int(getattr(seq, "state_joint_boundary_tokens", 0) or 0)
        return max(hbm, min(joint, lmc)) if joint else max(hbm, lmc)

    # -- connector-owned channels -----------------------------------------
    def connector_completion(self, completion) -> bool:
        if completion.channel == STATE_SOURCE_CHANNEL:
            self._state_source_released.add(completion.operation_id)
            return True
        if completion.channel == STATE_INDEX_CHANNEL:
            target = (
                self._state_indexed
                if completion.succeeded
                else self._state_index_failed
            )
            target.add(completion.operation_id)
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
