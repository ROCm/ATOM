# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public model selector for LMCache multiprocess offload connectors."""

from __future__ import annotations

import logging

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.offload.mp.registry import resolve_plugin

logger = logging.getLogger("atom")


def select_model(config) -> str:
    """Return the name of the auto-discovered model plugin."""

    return resolve_plugin(config).name


def _build_worker(config):
    plugin = resolve_plugin(config)
    logger.info("lmcache_mp: worker model=%s", plugin.name)
    return plugin.build_worker(config)


def _build_scheduler(config):
    plugin = resolve_plugin(config)
    logger.info("lmcache_mp: scheduler model=%s", plugin.name)
    return plugin.build_scheduler(config)


class LMCacheMPConnector(KVConnectorBase):
    """Worker-side shell delegating to the selected model implementation."""

    is_producer = False

    def __init__(self, config) -> None:
        self._impl = _build_worker(config)

    def register_kv_caches(
        self,
        kv_caches,
        transfer_tensors=None,
        num_blocks=None,
    ) -> None:
        self._impl.register_kv_caches(kv_caches, transfer_tensors, num_blocks)

    def start_load_kv(self, metadata) -> None:
        self._impl.start_load_kv(metadata)

    def get_finished(self):
        return self._impl.get_finished()

    def get_finished_recv_blocks(self):
        return self._impl.get_finished_recv_blocks()


class LMCacheMPConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler-side shell delegating to the selected model implementation."""

    is_producer = False
    is_offload = True

    def __init__(self, config) -> None:
        self._impl = _build_scheduler(config)

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

    def has_pending_work(self) -> bool:
        return self._impl.has_pending_work()

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

    def cancel_pending_load(self, seq) -> None:
        callback = getattr(self._impl, "cancel_pending_load", None)
        if callback is not None:
            callback(seq)

    def load_finished(self, req_id):
        callback = getattr(self._impl, "load_finished", None)
        return callback(req_id) if callback is not None else True

    def process_completions(self, output):
        return self._impl.process_completions(output)

    def get_statistics(self) -> dict[str, int]:
        return self._impl.get_statistics()


__all__ = [
    "LMCacheMPConnector",
    "LMCacheMPConnectorScheduler",
    "select_model",
]
