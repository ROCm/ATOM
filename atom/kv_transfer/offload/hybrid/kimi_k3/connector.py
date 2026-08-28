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

from atom.kv_transfer.disaggregation.types import ConnectorCompletion
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

#: A save outstanding longer than this is a backend that stopped, not one that
#: is busy: a 4096-token store costs ~65ms.
SAVE_STALL_SECONDS = 120.0


class KimiK3OffloadConnector(DenseOffloadConnector):
    """Worker side: dense KV, plus spill/load of the per-request state."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._state_tier = None
        # Inert until a request has both legs, which only a joint boundary
        # produces; costs one dict lookup per report otherwise.
        self._joint_park = _JointPark()

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
        # ONE pool, shared with paged KV. A separate `atom-state-{rank}` engine
        # used to be built here, on the argument that the KV write stream would
        # evict the checkpoints. It does not hold up: state is 3-13% of a
        # boundary's bytes, not a rounding error, and -- the part that decides
        # it -- a request writes its KV chunks and its one state object inside
        # the same prefill window, so they enter LMCache's LRU together and
        # cool at the same rate. State is retired ALONGSIDE its own KV, which
        # is what we want: a boundary whose KV is gone is worthless anyway.
        #
        # Two pools cost what one does not: independent eviction policies that
        # must drift, and a joint boundary needs BOTH legs to survive together.
        # #2045 made the same argument one tier up and won with it (state moved
        # out of its reserved slots into the KV pool, 81.85% -> 93.60%).
        #
        # What one pool costs is a single `cache_policy` for both legs. That is
        # real, and currently free: LRU is right for both. The FIFO override
        # this used to set rested on "a state entry is written once and read
        # once", which #2045's own numbers refute -- ~4,808 resumes over ~1,508
        # checkpoints is ~3 reads each, and bursty re-access is LRU's home
        # ground.
        #
        # `OFFLOAD_STATE_CPU_SIZE` is gone with it. #2045 deleted the HBM-side
        # reservation for exactly this reason (`extra_entries=0`, 1.7 GiB
        # returned) -- keeping the same knob one tier down repeats the thing it
        # disproved. `LMCACHE_MAX_LOCAL_CPU_SIZE` is the one size to tune.
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
        """
        loads = getattr(metadata, "state_loads", None) or ()
        state_ids = {req_id for req_id, _h, _slot in loads}
        if not state_ids or not self._do_load:
            return
        for req in metadata.requests:
            if req.load_spec is not None and req.req_id in state_ids:
                self._joint_park.arm(req.req_id, needs_kv=True, needs_state=True)

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

        No producer fence and no staging copy. The source is the checkpoint's
        PAGE units, which #2045 reserves out of the KV pool and the engine pins
        for the duration -- so nothing on the compute stream is writing them
        and the packer gathers straight from where they sit.

        A store with no tier is reported failed rather than dropped: the engine
        is holding those units pinned against a report, and silence would leave
        them to the reconciler's full timeout.
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
            for h, _units in stores:
                self._state_store_failed_locally.add(int(h))
            return
        for h, unit_ids in stores:
            self._state_tier.submit_store(int(h), tuple(int(u) for u in unit_ids))

    def _fail_state_loads(self, loads) -> None:
        for req_id, _h, _group in loads:
            self._joint_park.settle_state(req_id, False)

    # -- completions -------------------------------------------------------
    def get_finished(self):
        out = super().get_finished()
        if self._state_tier is None:
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
        for h in indexed:
            out.connector_completions.add(
                ConnectorCompletion(STATE_INDEX_CHANNEL, int(h), True)
            )
        for h in index_failed:
            out.connector_completions.add(
                ConnectorCompletion(STATE_INDEX_CHANNEL, int(h), False)
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
        # Channel reports drained by the engine each step.
        self._state_indexed: set[int] = set()
        self._state_index_failed: set[int] = set()
        # Stores this worker could not even attempt (no tier). Reported as
        # failures so the engine releases their pinned units immediately.
        self._state_store_failed_locally: set[int] = set()

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

    def should_defer_free(self, seq) -> bool:
        """Base behaviour, plus an escape when the backend has stopped.

        A save already handed out is reading these blocks, so freeing them
        would let the next request write into them mid-transfer and index the
        result under this prefix's hash. One never handed out has no reader,
        and holding it is what turns a stopped backend into a stopped engine.

        That leaves the handed-out save with no way out, which is what
        `Scheduler._reconcile_stalled_deferred_saves` covers, on a longer clock
        tied to LMCache's own force-unpin window. Neither mechanism subsumes the
        other: this one asks whether the backend ever took the save, that one
        whether it ever answered.
        """
        sid = str(seq.id)
        if (
            self._save_stalled
            and sid not in self._save_inflight
            and self._has_pending_save(seq)
        ):
            self._save_tracker.pop(sid, None)
            return False
        return super().should_defer_free(seq)

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
        if completion.channel == STATE_INDEX_CHANNEL:
            target = (
                self._state_indexed
                if completion.succeeded
                else self._state_index_failed
            )
            target.add(int(completion.operation_id))
            return True
        return super().connector_completion(completion)

    def take_state_reports(self) -> tuple[set[int], set[int]]:
        """Drain this step's tier store reports for the engine-side index."""
        indexed = self._state_indexed
        failed = self._state_index_failed | self._state_store_failed_locally
        self._state_indexed = set()
        self._state_index_failed = set()
        self._state_store_failed_locally = set()
        return indexed, failed
