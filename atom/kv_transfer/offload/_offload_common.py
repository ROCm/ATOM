# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Shared machinery for ATOM offload connectors.

Both offload backends — ``lmcache_offload`` (token-chunked MHA/MLA) and
``dsv4_offload`` (single opaque offload unit per 128-aligned checkpoint) — share
the same worker-side plumbing: separate save/load copy executors kept off the RPC
thread, a lock-guarded completion tally, and the ``KVConnectorOutput`` reporting
contract. They also build their LMCache storage engine the same way (opaque
uint8 metadata + ``KV_2LTD`` allocator-compat hack).

The *payload* model differs (chunked token-major vs one opaque unit) and stays in
each connector; this module holds only what is genuinely common, so the two
backends no longer duplicate the executor/engine boilerplate.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading

from atom.kv_transfer.disaggregation.types import KVConnectorOutput, ReqId
from atom.kv_transfer.offload import config as offcfg

logger = logging.getLogger("atom")


def build_offload_engine(
    config,
    *,
    engine_id: str,
    block_size: int,
    bytes_per_block: int,
    gpu_connector_factory,
    world: int,
    rank: int,
):
    """Build + post_init a per-rank LMCache engine for opaque uint8 offload.

    ``gpu_connector_factory(cfg, meta)`` builds the LMCache ``GPUConnectorInterface``
    once ``cfg`` (needed for chunk size) and the uint8 ``meta`` exist — the
    chunked path needs ``cfg.chunk_size`` for its connector, and DSV4 passes an
    inert one. Returns ``(engine, cfg, meta)``. The metadata forces uint8 shapes;
    ``fmt`` is a tensor-accepting ``MemoryFormat`` purely to satisfy the LocalCPU
    allocator.
    """
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.v1.memory_management import MemoryFormat

    from atom.kv_transfer.offload.metadata import ATOMRawBytesLMCacheMetadata

    cfg = offcfg.build_lmcache_config()
    offcfg.apply_extra_overrides(cfg, getattr(config, "kv_transfer_config", None))
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

    def _init_worker_common(self, config) -> None:
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = kvc.get("kv_role", "offload")
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        # Separate executors so a load (on the TTFT critical path) never queues
        # behind fire-and-forget saves. OFFLOAD_COPY_WORKERS tunes the save pool.
        n_save = int(os.environ.get("OFFLOAD_COPY_WORKERS", "1"))
        self._save_executor = ThreadPoolExecutor(
            max_workers=n_save, thread_name_prefix="offload-save"
        )
        self._load_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="offload-load"
        )
        self._lock = threading.Lock()
        self._done_save: set[ReqId] = set()
        self._done_load: set[ReqId] = set()
        self._failed_load: set[ReqId] = set()

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
                    self._failed_load.add(rid)
            else:
                # A failed save just loses this offload opportunity; still report
                # finished_saving so the scheduler releases any deferred free.
                with self._lock:
                    self._done_save.add(rid)

    def get_finished(self) -> KVConnectorOutput:
        with self._lock:
            dl = set(self._done_load)
            fl = set(self._failed_load)
            ds = set(self._done_save)
            self._done_load.clear()
            self._failed_load.clear()
            self._done_save.clear()
        return KVConnectorOutput(
            finished_sending=set(),
            finished_recving=dl,
            failed_recving=fl,
            finished_saving=ds,
        )

    def get_finished_recv_blocks(self) -> list[int]:
        return []
