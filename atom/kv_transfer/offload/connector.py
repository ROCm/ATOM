# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Public ``lmcache_offload`` connector: family selection + delegating shell.

A single ``kv_connector: "lmcache_offload"`` name serves both offload layout
families. This module is the one the factory registers; it (a) picks the family
from ``config`` — on BOTH the scheduler and worker side, so the two processes
agree without needing ``transfer_tensors`` — and (b) exposes one worker class
(:class:`LMCacheOffloadConnector`) and one scheduler class
(:class:`LMCacheOffloadConnectorScheduler`), each a thin shell that delegates
every hook to the chosen family impl:

* **dense** (:mod:`atom.kv_transfer.offload.dense.connector`) — token-dense KV
  (DSV2/V3 MLA, MHA), stored as reusable LMCache token chunks.
* **hybrid** (:mod:`atom.kv_transfer.offload.hybrid.connector`) — models carrying
  non-token-indexed per-request state (DSV4 compressor state, Qwen3-Next GDN
  recurrent state), stored as opaque terminal bundles. New target = new profile.

Selection order:

1. Explicit ``kv_transfer_config["offload_layout"]`` override (``"hybrid"`` |
   ``"dense"``; legacy ``terminal_unit`` / ``chunked`` / ``chunked_mla`` accepted).
2. Otherwise inferred from ``hf_config``: ``compress_ratios`` (DeepSeek-V4) or
   ``linear_num_key_heads`` (Qwen3-Next GDN) => ``hybrid``; otherwise ``dense``.
"""

from __future__ import annotations

import logging

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)

logger = logging.getLogger("atom")

# Accepted offload_layout overrides. Old names map to the current family names.
_OVERRIDE_ALIASES = {
    "hybrid": "hybrid",
    "dense": "dense",
    "terminal_unit": "hybrid",  # legacy
    "chunked": "dense",  # legacy
    "chunked_mla": "dense",  # legacy
}


def select_variant(config) -> str:
    """Return ``"hybrid"`` or ``"dense"`` for *config* (config-only, both sides)."""
    kvc = getattr(config, "kv_transfer_config", {}) or {}
    override = kvc.get("offload_layout")
    if override is not None:
        mapped = _OVERRIDE_ALIASES.get(override)
        if mapped is not None:
            return mapped
        logger.warning(
            "lmcache_offload: unknown offload_layout=%r; falling back to auto",
            override,
        )

    hf = getattr(config, "hf_config", None)
    if hf is not None:
        if getattr(hf, "compress_ratios", None):  # DeepSeek-V4
            return "hybrid"
        if getattr(hf, "linear_num_key_heads", None):  # Qwen3-Next GDN hybrid
            return "hybrid"
    return "dense"


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


# =====================================================================
# Worker shell
# =====================================================================
class LMCacheOffloadConnector(KVConnectorBase):
    """Worker-side shell: delegates to the family impl picked from ``config``."""

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


# =====================================================================
# Scheduler shell
# =====================================================================
class LMCacheOffloadConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler-side shell: delegates to the family impl picked from ``config``."""

    is_producer = False
    is_offload = True

    def __init__(self, config) -> None:
        self._impl = _build_scheduler(config)

    # -- required hooks ---------------------------------------------------
    def get_num_new_matched_tokens(self, seq):
        return self._impl.get_num_new_matched_tokens(seq)

    def update_state_after_alloc(self, seq) -> None:
        self._impl.update_state_after_alloc(seq)

    def build_connector_meta(self):
        return self._impl.build_connector_meta()

    def request_finished(self, seq) -> None:
        self._impl.request_finished(seq)

    # -- optional hooks present on both families --------------------------
    def should_park_for_load_after_alloc(self, seq) -> bool:
        return self._impl.should_park_for_load_after_alloc(seq)

    def should_defer_free(self, seq) -> bool:
        return self._impl.should_defer_free(seq)

    def save_finished(self, req_id) -> None:
        self._impl.save_finished(req_id)

    def load_failed(self, req_id) -> None:
        self._impl.load_failed(req_id)

    # -- dense-only hooks: default no-op when the impl omits them ---------
    def adjust_prefill_chunk_after_alloc(self, seq, chunk):
        fn = getattr(self._impl, "adjust_prefill_chunk_after_alloc", None)
        return fn(seq, chunk) if fn is not None else chunk

    def should_park_partial_prefill_for_load(self, seq) -> bool:
        fn = getattr(self._impl, "should_park_partial_prefill_for_load", None)
        return fn(seq) if fn is not None else False
