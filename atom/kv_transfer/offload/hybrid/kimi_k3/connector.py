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
    _JointPark,
    StateOffloadTier,
)
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata

logger = logging.getLogger("atom")

#: Completion channels this variant owns. The generic aggregator transports
#: them opaquely and takes a failure-dominant TP quorum, which is what makes a
#: partial store resolve instead of pinning a key forever.
STATE_INDEX_CHANNEL = "k3_state_index"
STATE_STAGING_CHANNEL = "k3_state_staging"

#: A save outstanding longer than this is a backend that stopped, not one that
#: is busy: a 4096-token store costs ~65ms.
SAVE_STALL_SECONDS = 120.0


class KimiK3OffloadConnector(DenseOffloadConnector):
    """Worker side: dense KV, plus spill/load of the per-request state."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._state_tier = None
        self._state_engine = None
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

        from atom.model_engine.state_offload import state_offload_staging_groups

        if state_offload_staging_groups() <= 0:
            return

        # PP breaks the tier twice over: the CacheEngineKey has no PP component,
        # so two stages at the same TP rank would overwrite each other; and only
        # the head stage drains staging releases, so the ring starves. Refused
        # rather than half-supported. Paged KV is unaffected.
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
        )
        codec.bind_storage_manager(self._state_storage_manager(cfg, meta, rank))
        # No index here: StateOffloadIndex lives in the engine process; both
        # directions report and the engine applies.
        self._state_tier = StateOffloadTier(codec)
        logger.info(
            "kimi_k3 offload: state tier up, entry=%.2f MiB rank=%d",
            entry_bytes / (1 << 20),
            rank,
        )

    def _state_storage_manager(self, cfg, meta, rank: int):
        """The tier's own CPU pool, not the paged-KV one.

        Sharing loses both ways: the KV write stream is several times the
        state volume so it evicts the checkpoints, and a stopped KV backend
        takes the tier down with it.
        """
        raw = os.environ.get("OFFLOAD_STATE_CPU_SIZE", "16")
        try:
            gib = float(raw)
        except ValueError:
            logger.warning("kimi_k3 offload: invalid OFFLOAD_STATE_CPU_SIZE=%r", raw)
            gib = 16.0
        if gib <= 0:
            logger.info("kimi_k3 offload: state tier shares the paged-KV CPU pool.")
            return self._engine.storage_manager
        try:
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            from lmcache.v1.memory_management import MemoryFormat

            cfg.max_local_cpu_size = gib
            # FIFO, while the KV pool keeps LMCache's LRU. A state entry is
            # written once and read once (turn N+1 resumes off turn N's anchor
            # and nobody looks again), so what decides its worth is age against
            # the think time. LRU would promote an entry already consumed over
            # one still waiting. Tied to the ladder being off; with rungs shared
            # across conversations, entries stop being one-shot and LRU wins.
            cfg.cache_policy = "FIFO"
            engine = LMCacheEngineBuilder.get_or_create(
                f"atom-state-{rank}",
                cfg,
                meta,
                self._engine.gpu_connector,
                lambda t, s: None,
                lambda o, s: o,
            )
            engine.fmt = MemoryFormat.KV_2LTD
            engine.post_init()
            self._state_engine = engine
            logger.info(
                "kimi_k3 offload: state pool %.0f GiB/rank (paged KV keeps %s)",
                gib,
                os.environ.get("LMCACHE_MAX_LOCAL_CPU_SIZE", "<default>"),
            )
            return engine.storage_manager
        except Exception:
            logger.warning(
                "kimi_k3 offload: no separate state pool; sharing the paged-KV "
                "pool, where the KV write stream will evict checkpoints.",
                exc_info=True,
            )
            return self._engine.storage_manager

    # -- per-step ----------------------------------------------------------
    def start_load_kv(self, metadata) -> None:
        if isinstance(metadata, LMCacheOffloadMetadata):
            self._arm_joint_loads(metadata)
            self._start_state_loads(metadata)
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

    def _fail_state_loads(self, loads) -> None:
        for req_id, _h, _group in loads:
            self._joint_park.settle_state(req_id, False)

    # -- completions -------------------------------------------------------
    def get_finished(self):
        out = super().get_finished()
        if self._state_tier is None:
            return out
        indexed, released, index_failed = self._state_tier.take_spill_reports()
        state_done, state_failed = self._state_tier.get_finished()
        out.finished_loading, out.failed_loading = self._settle_joint(
            out.finished_loading, out.failed_loading, state_done, state_failed
        )
        # Spill and staging reports have no request identity, so they ride the
        # connector-owned channels; the aggregator's quorum is failure-dominant,
        # which is what resolves a partial store instead of pinning the key.
        for h in indexed:
            out.connector_completions.add(
                ConnectorCompletion(STATE_INDEX_CHANNEL, int(h), True)
            )
        for h in index_failed:
            out.connector_completions.add(
                ConnectorCompletion(STATE_INDEX_CHANNEL, int(h), False)
            )
        for group in released:
            out.connector_completions.add(
                ConnectorCompletion(STATE_STAGING_CHANNEL, int(group), True)
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
        self._state_staging_released: set[int] = set()

    # -- state load queue --------------------------------------------------
    def enqueue_state_loads(self, loads) -> bool:
        if not loads:
            return False
        self._pending_state_loads.extend(loads)
        return True

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        self._refresh_save_stall()
        meta = super().build_connector_meta()
        # Drained, not copied: a second submission would write the same entry
        # into a group the first transfer is already filling.
        meta.state_loads = self._pending_state_loads
        self._pending_state_loads = []
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
        """
        sid = str(seq.id)
        if self._save_stalled and sid not in self._save_inflight:
            if self._has_pending_save(seq):
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
        # The KV leg moves whole chunks and the blocks below `hbm` are shared,
        # so an unaligned start cannot be rounded down.
        if hbm % chunk != 0:
            return False, "joint_unaligned_hbm_prefill", hbm, lmc, lmc - hbm, chunk
        # Transfer the chunk covering the boundary, claim only the boundary.
        kv_target = int(getattr(seq, "state_joint_kv_tokens", 0) or 0) or joint
        if joint > lmc or kv_target > lmc:
            return False, "joint_boundary_above_lookup", hbm, lmc, lmc - hbm, chunk
        ls.lmcache_cached_tokens = kv_target
        # Deliberately past the min-load floor: the boundary was chosen for both
        # legs, and refusing on size would leave the state leg claiming a prefix
        # whose KV never came.
        return True, "joint_state_and_kv", hbm, kv_target, kv_target - hbm, chunk

    @staticmethod
    def _claim_after_load(seq, hbm: int, lmc: int) -> int:
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
        if completion.channel == STATE_STAGING_CHANNEL:
            self._state_staging_released.add(int(completion.operation_id))
            return True
        return super().connector_completion(completion)

    def take_state_reports(self) -> tuple[set[int], set[int], set[int]]:
        """Drain this step's tier reports for the engine-side index."""
        indexed = self._state_indexed
        failed = self._state_index_failed
        released = self._state_staging_released
        self._state_indexed = set()
        self._state_index_failed = set()
        self._state_staging_released = set()
        return indexed, released, failed
