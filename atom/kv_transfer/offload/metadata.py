# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Metadata helpers for the LMCache CPU/NVMe offload connector.

``ATOMRawBytesLMCacheMetadata`` adapts LMCache's engine metadata so MemoryObjs
are allocated as opaque uint8 buffers. The remaining classes are per-request
transfer descriptors that travel from the scheduler-side connector to the
worker-side connector inside :class:`LMCacheOffloadMetadata`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from atom.kv_transfer.disaggregation.types import ConnectorMetadata, ReqId


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


class ATOMRawBytesLMCacheMetadata:
    """Proxy around ``LMCacheMetadata`` with ATOM raw-byte allocation shapes."""

    def __init__(
        self,
        base_metadata: Any,
        *,
        atom_block_size: int,
        bytes_per_block: int,
    ) -> None:
        self._atom_base_metadata = base_metadata
        self.__dict__.update(vars(base_metadata))
        self.atom_block_size = int(atom_block_size)
        self.atom_bytes_per_block = int(bytes_per_block)
        chunk_size = int(getattr(base_metadata, "chunk_size"))
        if self.atom_block_size <= 0:
            raise ValueError("ATOM raw-byte metadata: atom_block_size must be > 0")
        if self.atom_bytes_per_block <= 0:
            raise ValueError("ATOM raw-byte metadata: bytes_per_block must be > 0")
        if chunk_size % self.atom_block_size != 0:
            raise ValueError(
                "LMCache chunk size must be divisible by ATOM KV block size: "
                f"chunk_size={chunk_size}, block_size={self.atom_block_size}"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._atom_base_metadata, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ATOMRawBytesLMCacheMetadata):
            return (
                self._atom_base_metadata == other._atom_base_metadata
                and self.atom_block_size == other.atom_block_size
                and self.atom_bytes_per_block == other.atom_bytes_per_block
            )
        return False

    def is_first_rank(self) -> bool:
        return self._atom_base_metadata.is_first_rank()

    def get_dtypes(self) -> list[torch.dtype]:
        return [torch.uint8]

    def get_shapes(self, num_tokens: int | None = None) -> list[torch.Size]:
        if num_tokens is None:
            num_tokens = int(self.chunk_size)
        nblocks = _cdiv(int(num_tokens), self.atom_block_size)
        return [torch.Size((nblocks * self.atom_bytes_per_block,))]

    def get_num_groups(self) -> int:
        return 1


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
