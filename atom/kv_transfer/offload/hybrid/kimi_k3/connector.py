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

#: The per-request state-load verdict, one completion per released joint/state
#: load (finding #6). `succeeded=True` means the state H2D landed and the index
#: must keep the hash even when the pair failed (a dropped KV chunk):
#: `abandon_load`, not `fail_load`. `succeeded=False` means the state leg itself
#: missed and the hash must be forgotten. Emitted for successful loads too so
#: the aggregator reaches quorum on the same step the KV load report does,
#: rather than leaving a partially-reported key pending forever.
STATE_LOAD_DISPOSITION_CHANNEL = "k3_state_load_disposition"

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
        # Inert until a request has both legs, which only a joint boundary
        # produces; costs one dict lookup per report otherwise.
        self._joint_park = _JointPark()
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
        if not isinstance(metadata, LMCacheOffloadMetadata):
            super().start_load_kv(metadata)
            return
        # The arm + three state helpers run before super() so the KV leg is
        # submitted against a park that already exists. None of them may skip
        # super(): _JointPark's correctness -- and the refutation of the
        # "park leaks on abort" candidate -- both rest on "the KV leg always
        # reports", which holds only while super().start_load_kv() cannot be
        # skipped. `ThreadPoolExecutor.submit` raises RuntimeError if the
        # executor was shut down by a racing close() or the OS refuses a thread,
        # and ModelRunner.process_kvconnector_output has no handler -- so each
        # submit is isolated inside its helper (a failure fails just that item,
        # into the same failed-load / failed-store channels a no-tier step uses,
        # so the armed park resolves to a recompute instead of hanging), and
        # super() runs in a finally so a KV load or save is never dropped.
        # `_arm_joint_loads` is inside the try too: it builds park state and can
        # raise (a MemoryError filing the dicts, say), and the same "KV leg must
        # still report" invariant that puts super() in a finally requires the
        # KV submit to run even when arming half-completed -- an arriving KV
        # completion then settles or evicts whatever the arm did file.
        try:
            self._arm_joint_loads(metadata)
            self._start_state_loads(metadata)
            self._start_state_stores(metadata)
        finally:
            super().start_load_kv(metadata)

    def _arm_joint_loads(self, metadata) -> None:
        """Hold a request owning both legs until both report.

        Both legs surface on the KV completion channel, so an unheld id would
        collapse into one wake and resume the suffix prefill while the other
        transfer is still writing.

        **The two legs report different identities.** The KV worker reports the
        typed `LoadOperationId` (always issued for a load); the state tier the
        bare req id. So the park is filed under the KV identity (the one that has
        to reach the engine), with the bare id aliased onto it. Arming under the
        bare id instead parked nothing the KV leg could settle -- it passed
        through and the engine resumed prefill while the state H2D was still
        writing: silent wrong output, one leaked park per joint load.

        **No tier is armed exactly like a tier.** An earlier guard skipped the
        arm when `_state_tier is None`; but `_start_state_loads` fails these
        loads, and only an armed park turns that failure into a real
        `failed_loading` (recompute). Skipping it let the KV leg pass through as
        `finished_loading`, and `Scheduler._settle_state_load(ok=True)` counted a
        state restore that never happened -- silent wrong output. `get_finished`
        drains the park on the no-tier path too, so the arriving KV completion
        settles the pair and the entry cannot leak.
        """
        loads = getattr(metadata, "state_loads", None) or ()
        state_ids = {req_id for req_id, _h, _slot in loads}
        if not state_ids or not self._do_load:
            return
        # Drive the loop off the state loads, not `metadata.requests`. The two
        # sets have independent sources: `state_loads` arrives via
        # `BlockManager.take_state_loads()` -> `Scheduler._publish_state_loads`,
        # while `metadata.requests` is populated by the base KV scheduler. A
        # state-only load -- KV resident, recurrent state not; the
        # `_park_for_remote_load` path reached exactly when `needs_remote_load`
        # came back False -- has no `LMCacheReqMeta` this step, so it is absent
        # from `metadata.requests`. Iterating that set skipped precisely the
        # state-only requests the `else` branch below exists for: unarmed, their
        # failure emitted nothing and they hung in WAITING_FOR_REMOTE_KVS
        # forever. Look the KV leg up per state id instead of the other way
        # round.
        by_id = {req.req_id: req for req in metadata.requests}
        for req_id in state_ids:
            req = by_id.get(req_id)
            if req is not None and req.load_spec is not None:
                # Both legs: file the park under the KV identity, because that is
                # the one that has to reach the engine on finished/failed_loading.
                self._joint_park.arm(
                    req_id,
                    needs_kv=True,
                    needs_state=True,
                    kv_id=self._load_completion_id(req),
                )
            else:
                # State-only load (KV resident, or no KV meta this step). Arm it
                # on the state leg alone. Its SUCCESS already reached the engine
                # via the `_settle_joint` passthrough, but its FAILURE did not: an
                # unarmed park emitted nothing, so the request sat in
                # WAITING_FOR_REMOTE_KVS forever (the orphan reclaimer never wakes
                # a live parked request). Arming makes `take_ready` surface either
                # outcome under the same bare id the passthrough already used.
                self._joint_park.arm(
                    req_id,
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
        if not self._do_load:
            # Symmetric with `_arm_joint_loads`' gate. Without it, a step that
            # does not load (e.g. `kv_role: kv_consumer`) leaves the park unarmed
            # -- `_arm_joint_loads` returned early on the same predicate -- yet
            # still submits/fails these loads here. For a state-only load with no
            # tier, `_fail_state_loads` -> `settle_state(req_id, False)` finds no
            # armed park (`need is None`) and emits nothing, so the request sits
            # in WAITING_FOR_REMOTE_KVS with nothing to wake it (the orphan
            # reclaimer frees the slot but never wakes a live parked request).
            # Gate both halves on the one predicate so neither runs without the
            # other.
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
            # Isolate each submit (finding #4): a shut-down executor or an OS
            # thread refusal fails only this load -- settle its state leg False
            # so the armed park resolves to a recompute -- and cannot take down
            # the remaining loads or, via an escape past start_load_kv, the KV
            # legs super() still owes.
            try:
                self._state_tier.submit_load(req_id, int(h), int(group))
            except Exception:
                logger.exception(
                    "kimi_k3 offload: submit_load failed for %s; failing it.",
                    req_id,
                )
                self._joint_park.settle_state(req_id, False)

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

    def _fail_state_loads(self, loads) -> None:
        for req_id, _h, _group in loads:
            self._joint_park.settle_state(req_id, False)

    # -- completions -------------------------------------------------------
    def get_finished(self):
        out = super().get_finished()
        # Sweep parks whose report never came before draining this step's, on
        # every path including no-tier (finding #3). Cheap: `_armed_at` holds
        # only in-flight joint loads and is usually empty. Ordering-independent
        # -- a park stale enough to evict is one whose window already elapsed, so
        # a report arriving this same step for it is exactly the late-report case
        # `reclaim_stale_parks` is built to pass through harmlessly.
        self._reclaim_stale_parks()
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
            # `_arm_joint_loads` armed these even with no tier and
            # `_fail_state_loads` failed their state leg; drain the park here,
            # before the return, with empty tier reports. A joint load then owes
            # only the KV leg -- its arriving KV completion settles the pair into
            # `failed_loading` (recompute) and releases the entry, instead of
            # passing through as a miscounted state restore. A state-only load
            # was released when its one leg failed, so `take_ready` surfaces it
            # into `failed_loading` now rather than leaving it stuck in
            # WAITING_FOR_REMOTE_KVS.
            out.finished_loading, out.failed_loading = self._settle_joint(
                out.finished_loading, out.failed_loading, set(), set()
            )
            self._emit_load_dispositions(out)
            return out
        indexed, index_failed = self._state_tier.take_store_reports()
        state_done, state_failed = self._state_tier.get_finished()
        out.finished_loading, out.failed_loading = self._settle_joint(
            out.finished_loading, out.failed_loading, state_done, state_failed
        )
        self._emit_load_dispositions(out)
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

    def _emit_load_dispositions(self, out) -> None:
        """Turn this step's park verdicts into state-index completions (finding #6).

        One `STATE_LOAD_DISPOSITION_CHANNEL` completion per released joint/state
        load, keyed by the same identity the KV leg reported on
        `finished/failed_loading` so the engine can correlate them. `succeeded`
        is the state leg's own outcome: the engine keeps the hash (`abandon_load`)
        for a failed pair whose state bytes are intact, and forgets it
        (`fail_load`) only when the state leg itself missed. Emitting for the
        successful loads too is deliberate -- the failure-dominant TP quorum
        needs every worker that reported the load to report a verdict, or the
        key never drains.
        """
        for key, state_intact in self._joint_park.take_dispositions().items():
            out.connector_completions.add(
                ConnectorCompletion(
                    STATE_LOAD_DISPOSITION_CHANNEL, key, bool(state_intact)
                )
            )

    def _reclaim_stale_parks(self) -> None:
        """Evict joint parks whose report never came (finding #3).

        A joint park is released only by both legs reporting. Three cases leave
        one leg forever unreported: the request is aborted mid-load (the KV leg
        is cancelled scheduler-side and never reports, while the state tier's leg
        still lands -- half a report cannot release the pair), a worker thread is
        killed mid-transfer, or a completion is dropped between worker and
        engine. Without an exit the key -- and the `_alias`/`_alias_of` under it
        -- sits in `_need` for the life of the process, and worse, `_settle_
        joint` swallows every later KV completion that reuses the stale `kv_id`,
        wedging that request in WAITING_FOR_REMOTE_KVS with its blocks held.

        The park is worker-side, so it is swept here -- `get_finished` is its one
        per-step worker driver. The abort signal itself is scheduler-side
        (`request_finished`/`cancel_pending_load` on the scheduler connector,
        which cannot reach this object), so an expiry sweep, not a synchronous
        abort call, is what closes the gap; the same-request re-admission shape
        is already handled by `arm`'s pre-file `_purge`. The window is LMCache's
        own save-abandon timeout -- the same clock the engine's pin and orphan-
        slot reconcilers use -- and shares their caveat: a report that truly
        arrives after it finds the key gone and passes through `_settle_joint`
        harmlessly for the long-departed request. `<= 0` disables reclamation
        (the operator turned the pin timeout off), matching those reconcilers.
        """
        window = offload_save_abandon_timeout_s()
        if window <= 0:
            return
        reclaimed = self._joint_park.reclaim_stale_parks(window)
        if reclaimed:
            logger.warning(
                "kimi_k3 offload: reclaimed %d joint load park(s) whose report "
                "never came (aborted request, killed worker, or dropped "
                "completion); those requests recompute.",
                reclaimed,
            )


class KimiK3OffloadScheduler(DenseOffloadScheduler, StateOffloadFace):
    """Scheduler side: dense KV, plus the state tier's load queue and the
    save-stall guard that keeps a stopped backend from stopping the engine.

    Inherits `StateOffloadFace` -- the only offload scheduler that hosts the KDA
    state tier -- so routing can select it with `isinstance` rather than probing
    for a method the delegating shell defines on every layout (see
    `StateOffloadFace` and the shell's `has_state_tier`)."""

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
        # Requests whose state H2D landed even though the joint load failed
        # (finding #6). Drained by the engine before it settles `failed_loading`,
        # to abandon rather than forget their still-present state hash.
        self._state_load_survived: set = set()

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
        if completion.channel == STATE_LOAD_DISPOSITION_CHANNEL:
            # Finding #6: a state load whose bytes survived a failed pair. Record
            # only the survivors -- their `failed_loading` report must abandon
            # the index (keep the hash) instead of failing it (forget). A
            # genuine state miss drains `succeeded=False` and is left out, so the
            # engine's default `fail_load` still forgets it. Normalise the KV
            # identity to the bare request id here, matching how
            # `process_completions` normalises `finished/failed_loading`, so the
            # engine's set membership test lines up.
            if completion.succeeded:
                op = completion.operation_id
                self._state_load_survived.add(
                    op.req_id if hasattr(op, "req_id") else op
                )
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

    def take_state_load_survived(self) -> set:
        """Drain the requests whose state bytes outlived a failed joint load.

        Finding #6. The engine consults this before settling `failed_loading`:
        a member here abandons its state-index entry (keeps the loadable hash)
        instead of failing it (forgetting a hash whose bytes are present).
        """
        survived = self._state_load_survived
        self._state_load_survived = set()
        return survived
