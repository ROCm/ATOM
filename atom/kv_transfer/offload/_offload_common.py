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

    def _on_load_fail(self, req_id) -> None:  # override to release a lookup pin
        pass

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
