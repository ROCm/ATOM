# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public ``lmcache_offload`` connector selection and delegation shell.

The public connector name stays stable while the implementation is selected on
both the scheduler and worker from configuration alone:

* ``dense`` stores ordinary token-indexed KV chunks;
* ``hybrid`` stores DSV4 compressed PAGE chunks plus complete SLOT sidecars.

Keeping selection config-only is important because the scheduler process does
not have access to the worker's transfer tensors.
"""

from __future__ import annotations

import logging

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.offload.config import select_offload_layout

logger = logging.getLogger("atom")


def select_variant(config) -> str:
    """Return the offload family selected for ``config``."""

    return select_offload_layout(config)


def _build_worker(config):
    variant = select_variant(config)
    logger.info("lmcache_offload: worker family=%s", variant)
    if variant == "hybrid":
        from atom.kv_transfer.offload.hybrid.connector import HybridOffloadConnector

        return HybridOffloadConnector(config)

    from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector

    return DenseOffloadConnector(config)


def _build_scheduler(config):
    variant = select_variant(config)
    logger.info("lmcache_offload: scheduler family=%s", variant)
    if variant == "hybrid":
        from atom.kv_transfer.offload.hybrid.connector import HybridOffloadScheduler

        return HybridOffloadScheduler(config)

    from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler

    return DenseOffloadScheduler(config)


class LMCacheOffloadConnector(KVConnectorBase):
    """Worker-side shell delegating to the selected implementation."""

    is_producer = False

    def __init__(self, config) -> None:
        self._impl = _build_worker(config)

    def register_kv_caches(
        self, kv_caches, transfer_tensors=None, num_blocks=None
    ) -> None:
        self._impl.register_kv_caches(kv_caches, transfer_tensors, num_blocks)

    def start_load_kv(self, metadata) -> None:
        self._impl.start_load_kv(metadata)

    def get_finished(self):
        return self._impl.get_finished()

    def get_finished_recv_blocks(self):
        return self._impl.get_finished_recv_blocks()


class LMCacheOffloadConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler-side shell delegating to the selected implementation."""

    is_producer = False
    is_offload = True

    def __init__(self, config) -> None:
        self._impl = _build_scheduler(config)
        self.recap_prefill_after_finalize = bool(
            getattr(self._impl, "recap_prefill_after_finalize", False)
        )

    def get_num_new_matched_tokens(self, seq):
        return self._impl.get_num_new_matched_tokens(seq)

    def update_state_after_alloc(self, seq) -> None:
        self._impl.update_state_after_alloc(seq)

    def build_connector_meta(self):
        return self._impl.build_connector_meta()

    def request_finished(self, seq) -> None:
        self._impl.request_finished(seq)

    def should_park_for_load_after_alloc(self, seq) -> bool:
        return self._impl.should_park_for_load_after_alloc(seq)

    def should_defer_free(self, seq) -> bool:
        return self._impl.should_defer_free(seq)

    def save_finished(self, req_id) -> None:
        self._impl.save_finished(req_id)

    def load_failed(self, req_id):
        return self._impl.load_failed(req_id)

    def adjust_prefill_chunk_after_alloc(self, seq, chunk):
        callback = getattr(self._impl, "adjust_prefill_chunk_after_alloc", None)
        return callback(seq, chunk) if callback is not None else chunk

    def should_park_partial_prefill_for_load(self, seq) -> bool:
        callback = getattr(self._impl, "should_park_partial_prefill_for_load", None)
        return callback(seq) if callback is not None else False

    def should_pause_partial_prefill_for_save(self, seq) -> bool:
        callback = getattr(self._impl, "should_pause_partial_prefill_for_save", None)
        return callback(seq) if callback is not None else False

    def cancel_pending_load(self, seq) -> None:
        callback = getattr(self._impl, "cancel_pending_load", None)
        if callback is not None:
            callback(seq)

    def connector_meta_dispatched(self, meta) -> None:
        callback = getattr(self._impl, "connector_meta_dispatched", None)
        if callback is not None:
            callback(meta)

    def sidecar_save_finished(self, req_id) -> None:
        callback = getattr(self._impl, "sidecar_save_finished", None)
        if callback is not None:
            callback(req_id)

    def sidecar_save_failed(self, req_id) -> None:
        callback = getattr(self._impl, "sidecar_save_failed", None)
        if callback is not None:
            callback(req_id)

    def load_finished(self, req_id):
        callback = getattr(self._impl, "load_finished", None)
        return callback(req_id) if callback is not None else True


__all__ = [
    "LMCacheOffloadConnector",
    "LMCacheOffloadConnectorScheduler",
    "select_variant",
]
