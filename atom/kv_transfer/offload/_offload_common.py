# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Shared machinery for ATOM's dense and hybrid offload families.

The public ``lmcache_offload`` shell selects either ordinary dense raw-block KV
or DSV4 PAGE+SLOT storage. Both families share the worker-side executor, role,
completion, and LMCache-engine plumbing here; family modules retain only their
payload mapping and PAGE/SLOT policy.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadCompletionId,
    SaveCompletionId,
)
from atom.kv_transfer.offload import config as offcfg

logger = logging.getLogger("atom")
_VALID_KV_ROLES = {"offload", "kv_both", "kv_producer", "kv_consumer"}


def validated_kv_role(kvc: dict) -> str:
    role = kvc.get("kv_role", "offload")
    if role not in _VALID_KV_ROLES:
        raise ValueError(
            f"invalid kv_role {role!r}; expected one of {sorted(_VALID_KV_ROLES)}"
        )
    return role


def build_offload_engine(
    config,
    *,
    engine_id: str,
    block_size: int,
    bytes_per_block: int,
    gpu_connector_factory,
    world: int,
    rank: int,
    cfg=None,
):
    """Build + post_init a per-rank LMCache engine for opaque uint8 offload.

    ``gpu_connector_factory(cfg, meta)`` builds the LMCache
    ``GPUConnectorInterface`` once the validated chunk size and uint8 metadata
    exist. Returns ``(engine, cfg, meta)``. The metadata forces uint8 shapes;
    ``fmt`` is a tensor-accepting ``MemoryFormat`` purely to satisfy the
    LocalCPU allocator.
    """
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.v1.memory_management import MemoryFormat

    from atom.kv_transfer.offload.metadata import ATOMRawBytesLMCacheMetadata

    if cfg is None:
        cfg = offcfg.build_lmcache_config(getattr(config, "kv_transfer_config", None))
    base_meta = offcfg.build_lmcache_metadata(config, cfg, world, rank)
    meta = ATOMRawBytesLMCacheMetadata(
        base_meta, atom_block_size=int(block_size), bytes_per_block=int(bytes_per_block)
    )
    gpu_connector = gpu_connector_factory(cfg, meta)
    engine = LMCacheEngineBuilder.get_or_create(
        engine_id, cfg, meta, gpu_connector, lambda t, s: None, lambda o, s: o
    )
    engine.fmt = MemoryFormat.KV_2LTD
    engine.post_init()
    return engine, cfg, meta


class OffloadWorkerMixin:
    """Executor plumbing + completion reporting shared by offload workers.

    Subclasses call :meth:`_init_worker_common` from ``__init__`` and use the
    ``_save_executor`` / ``_load_executor`` + the ``_done_save`` / ``_done_load``
    / ``_failed_load`` tallies. Override :meth:`_on_load_fail` for connectors that
    hold a lookup pin to release on failure.
    """

    is_producer = False

    def _init_worker_common(
        self,
        config,
        *,
        save_workers: int | None = None,
        thread_name_prefix: str = "offload",
    ) -> None:
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = validated_kv_role(kvc)
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        # Separate executors so a load (on the TTFT critical path) never queues
        # behind fire-and-forget saves. OFFLOAD_COPY_WORKERS tunes the save pool.
        n_save = (
            int(os.environ.get("OFFLOAD_COPY_WORKERS", "1"))
            if save_workers is None
            else int(save_workers)
        )
        if n_save <= 0:
            raise ValueError("offload save worker count must be positive")
        self._save_executor = ThreadPoolExecutor(
            max_workers=n_save, thread_name_prefix=f"{thread_name_prefix}-save"
        )
        self._load_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{thread_name_prefix}-load"
        )
        self._lock = threading.Lock()
        self._done_save: set[SaveCompletionId] = set()
        self._done_load: set[LoadCompletionId] = set()
        self._failed_load: set[LoadCompletionId] = set()

    @staticmethod
    def _load_completion_id(req) -> LoadCompletionId:
        return getattr(req, "load_operation", None) or req.req_id

    @staticmethod
    def _save_completion_id(req) -> SaveCompletionId:
        return getattr(req, "save_operation", None) or req.req_id

    def _on_load_fail(self, req_id) -> None:
        """Release the LMCache lookup pin held by a failed load."""

        self._lookup_unpin(req_id)

    def _lookup_unpin(self, req_id) -> None:
        """Best-effort release of one worker-side LMCache lookup pin."""

        engine = getattr(self, "_engine", None)
        if engine is None:
            return
        try:
            engine.lookup_unpin(str(req_id))
        except Exception:  # optional third-party cleanup boundary
            logger.debug(
                "LMCache offload: lookup unpin failed for req=%s",
                req_id,
                exc_info=True,
            )

    @staticmethod
    def _profile_enabled() -> bool:
        return os.environ.get("OFFLOAD_PROFILE", "0").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _last_gpu_connector_transfer_stats(self) -> dict[str, int | float]:
        gpu_connector = getattr(getattr(self, "_engine", None), "gpu_connector", None)
        if gpu_connector is None or not hasattr(gpu_connector, "last_transfer_stats"):
            return {}
        try:
            return dict(gpu_connector.last_transfer_stats())
        except Exception:  # optional instrumentation hook
            logger.debug(
                "LMCache offload: transfer stats collection failed",
                exc_info=True,
            )
            return {}

    def _reset_gpu_connector_transfer_stats(self) -> None:
        gpu_connector = getattr(getattr(self, "_engine", None), "gpu_connector", None)
        if gpu_connector is None or not hasattr(gpu_connector, "reset_transfer_stats"):
            return
        try:
            gpu_connector.reset_transfer_stats()
        except Exception:  # optional instrumentation hook
            logger.debug(
                "LMCache offload: transfer stats reset failed",
                exc_info=True,
            )

    def _guard(self, kind: str, fn, req) -> None:
        """Run a copy job off the RPC thread, tallying success/failure."""
        try:
            fn(req)
        except Exception:
            logger.exception(
                "offload %s failed for %s",
                getattr(fn, "__name__", kind),
                getattr(req, "req_id", req),
            )
            rid = getattr(req, "req_id", req)
            if kind == "load":
                self._on_load_fail(rid)
                with self._lock:
                    self._failed_load.add(self._load_completion_id(req))
            else:
                # A failed save just loses this offload opportunity; still report
                # finished_saving so the scheduler releases any deferred free.
                with self._lock:
                    self._done_save.add(self._save_completion_id(req))

    def get_finished(self) -> KVConnectorOutput:
        with self._lock:
            dl, fl, ds = self._drain_common_completions_locked()
        return KVConnectorOutput(
            finished_sending=set(),
            finished_loading=dl,
            failed_loading=fl,
            finished_saving=ds,
        )

    def _drain_common_completions_locked(
        self,
    ) -> tuple[set[LoadCompletionId], set[LoadCompletionId], set[SaveCompletionId]]:
        """Drain base completion sets while the caller holds ``self._lock``."""

        done_load = set(self._done_load)
        failed_load = set(self._failed_load)
        done_save = set(self._done_save)
        self._done_load.clear()
        self._failed_load.clear()
        self._done_save.clear()
        return done_load, failed_load, done_save

    def get_finished_recv_blocks(self) -> list[int]:
        return []


class OffloadSchedulerMixin:
    """Layout-independent scheduler policy shared by dense and DSV4 offload.

    Subclasses own lookup construction, metadata serialization, and any
    state-checkpoint policy. This mixin contains only token-frontier and load
    handoff mechanics whose invariants are identical for both layouts.
    """

    def _chunk_floor(self, tokens: int) -> int:
        chunk = int(self.chunk_size or 256)
        return (max(0, int(tokens)) // chunk) * chunk

    def _lmcache_hit_save_floor(self, load_spec) -> int:
        if load_spec is None:
            return 0
        return self._chunk_floor(load_spec.lmcache_cached_tokens)

    def _set_save_frontier(self, sid: str, seq, saved: int) -> None:
        saved = self._chunk_floor(saved)
        if sid not in self._save_tracker:
            self._save_tracker[sid] = [seq, saved]
        else:
            self._save_tracker[sid][0] = seq
            self._save_tracker[sid][1] = saved

    def _maybe_start_unaligned_handoff(
        self,
        seq,
        load_spec,
        hbm: int,
        lmc: int,
        chunk: int,
    ) -> bool:
        boundary = ((hbm + chunk - 1) // chunk) * chunk
        remaining_after_boundary = lmc - boundary
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        if boundary <= hbm or remaining_after_boundary < min_load:
            return False

        sid = str(seq.id)
        load_spec.hbm_cached_tokens = boundary
        load_spec.can_load = True
        self._reqs_need_recv.pop(sid, None)
        self._handoff_loads.add(sid)
        seq.offload_loaded_tokens = hbm
        seq.offload_handoff_boundary_tokens = boundary
        logger.debug(
            "[OFFLOAD-LOAD-HANDOFF] seq=%s hbm_cached=%d boundary=%d "
            "lmc_cached=%d need_after_boundary=%d min_load=%d chunk=%d",
            seq.id,
            hbm,
            boundary,
            lmc,
            remaining_after_boundary,
            min_load,
            chunk,
        )
        return True

    def should_park_partial_prefill_for_load(self, seq) -> bool:
        if not self._do_load:
            return False
        sid = str(seq.id)
        if sid not in self._handoff_loads:
            return False
        load_spec = self._load_specs.get(sid)
        if load_spec is None:
            self._handoff_loads.discard(sid)
            return False
        boundary = int(getattr(seq, "offload_handoff_boundary_tokens", 0) or 0)
        hbm = int(getattr(seq, "num_cached_tokens", 0))
        if boundary > 0 and hbm < boundary:
            return False

        should_load, reason, hbm, lmc, need, chunk = self._decide_load_after_alloc(
            seq, load_spec
        )
        if not should_load:
            self._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
            self._clear_pending_load(sid)
            return False

        load_spec.can_load = True
        self._reqs_need_recv[sid] = seq
        self._handoff_loads.discard(sid)
        seq.offload_loaded_tokens = max(hbm, lmc)
        logger.debug(
            "[OFFLOAD-LOAD-HANDOFF-READY] seq=%s hbm_cached=%d "
            "lmc_cached=%d offload_loaded=%d need=%d",
            seq.id,
            hbm,
            lmc,
            seq.offload_loaded_tokens,
            need,
        )
        return True

    def _mark_load_skip(
        self,
        seq,
        reason: str,
        hbm: int,
        lmc: int,
        need: int,
        chunk: int,
    ) -> None:
        seq.offload_loaded_tokens = hbm
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        logger.debug(
            "[OFFLOAD-LOAD-SKIP] seq=%s hbm_cached=%d lmc_cached=%d "
            "need=%d min_load=%d chunk=%d reason=%s",
            seq.id,
            hbm,
            lmc,
            need,
            min_load,
            chunk,
            reason,
        )

    def should_park_for_load_after_alloc(self, seq) -> bool:
        if not self._do_load:
            return False
        sid = str(seq.id)
        load_spec = self._load_specs.get(sid)
        if load_spec is None:
            return False
        should_load, reason, hbm, lmc, need, chunk = self._decide_load_after_alloc(
            seq, load_spec
        )
        if not should_load:
            if (
                reason == "unaligned_hbm_prefill"
                and self._maybe_start_unaligned_handoff(seq, load_spec, hbm, lmc, chunk)
            ):
                return False
            self._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
            self._clear_pending_load(sid)
            return False
        seq.offload_loaded_tokens = max(hbm, lmc)
        return True

    def connector_meta_dispatched(self, meta) -> None:
        dispatched = set(getattr(meta, "lookup_requests_in_step", ()) or ())
        if dispatched:
            self._lookup_in_step = [
                sid for sid in self._lookup_in_step if sid not in dispatched
            ]

    def _save_frontier(self, seq) -> int:
        computed = min(
            int(getattr(seq, "num_cached_tokens", 0)),
            int(getattr(seq, "num_prompt_tokens", 0)),
        )
        return self._chunk_floor(computed)

    def _has_pending_save(self, seq) -> bool:
        sid = str(seq.id)
        entry = self._save_tracker.get(sid)
        if entry is None:
            return False
        return self._save_frontier(seq) > int(entry[1])
