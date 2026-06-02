# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Per-request transfer descriptors for the LMCache CPU/NVMe offload connector.

Ported (type-substituted) from vLLM's ``lmcache_integration/vllm_v1_adapter.py``
(``LoadSpec`` / ``SaveSpec`` / ``RequestTracker`` / ``ReqMeta``) onto ATOM's
``Sequence`` model. These travel from the scheduler-side connector to the
worker-side connector inside :class:`LMCacheOffloadMetadata`, which subclasses
ATOM's :class:`ConnectorMetadata` so the engine forwards it opaquely through
``process_kvconnector_output`` → ``start_load_kv``.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.kv_transfer.disaggregation.types import ConnectorMetadata, ReqId


@dataclass
class LoadSpec:
    """How many tokens to load for a request, split HBM-cached vs LMCache-cached."""

    # Tokens already resident in ATOM's HBM prefix cache (num_cached_tokens).
    hbm_cached_tokens: int
    # Total tokens LMCache can supply (>= hbm_cached_tokens). The load fills the
    # gap [hbm_cached_tokens, lmcache_cached_tokens).
    lmcache_cached_tokens: int
    # Set True by update_state_after_alloc once blocks are reserved for the load.
    can_load: bool = False


@dataclass
class SaveSpec:
    """How many leading tokens of a request are already saved to LMCache."""

    # Tokens at the prefix already persisted (skip these on the next store).
    skip_leading_tokens: int
    # Set False to suppress the store for this step (e.g. nothing new to save).
    can_save: bool = True


@dataclass
class LMCacheReqMeta:
    """Everything the worker needs to load/save one request's KV this step."""

    req_id: ReqId
    # Token ids covering the prefix being moved (used to derive chunk-256 keys via
    # LMCache's ChunkedTokenDatabase). For load: prompt[:lmcache_cached_tokens];
    # for save: computed token ids.
    token_ids: list[int]
    # The sequence's GPU block table (logical block ids). A chunk spanning token
    # range [start, end) maps to blocks block_ids[start // bs : ceil(end / bs)].
    block_ids: list[int]
    load_spec: LoadSpec | None = None
    save_spec: SaveSpec | None = None
    # True on the request's final prefill chunk (store the unaligned tail too).
    is_last_prefill: bool = True


class LMCacheOffloadMetadata(ConnectorMetadata):
    """Connector metadata snapshot for one engine step.

    Subclasses ATOM's :class:`ConnectorMetadata` (so it satisfies the
    ``build_connector_meta() -> ConnectorMetadata`` contract and is forwarded
    opaquely by the engine) while carrying the richer per-request offload
    descriptors the worker consumes in ``start_load_kv``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[LMCacheReqMeta] = []
        # req_ids whose scheduler-side lookup pin should be released this step.
        self.lookup_requests_in_step: list[str] = []

    def add_request(self, meta: LMCacheReqMeta) -> None:
        self.requests.append(meta)
