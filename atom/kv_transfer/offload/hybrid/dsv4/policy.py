# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 PAGE+SLOT geometry, cadence, and commit policy.

This module deliberately contains only CPU-side policy.  GPU layout movement
lives in :mod:`.codec`, while LMCache and scheduler orchestration live in
:mod:`.connector`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableSet
from dataclasses import dataclass
import hashlib
import json
from math import lcm
import os


@dataclass(frozen=True)
class DSV4OffloadProfile:
    """Resolved token grids and cache dimensions for DeepSeek-V4 offload."""

    name: str
    block_size: int
    dcp_size: int
    hash_block_size: int
    chunk_size: int
    resume_alignment: int
    checkpoint_interval: int
    sidecar_interval: int
    kv_head_dim: int
    index_head_dim: int


# Compatibility name for callers that imported the former single-use generic
# profile.  New DSV4 code should use ``DSV4OffloadProfile`` directly.
HybridProfile = DSV4OffloadProfile


def build_dsv4_profile(config, *, chunk_size: int) -> DSV4OffloadProfile:
    """Resolve DSV4 geometry from config without consulting worker tensors."""

    block_size = int(config.kv_cache_block_size)
    dcp_size = int(getattr(config, "decode_context_parallel_size", 1) or 1)
    chunk_size = int(chunk_size)
    if block_size <= 0 or dcp_size <= 0 or chunk_size <= 0:
        raise ValueError("DSV4 block, DCP, and LMCache chunk sizes must be positive")

    hash_block_size = block_size * dcp_size
    if chunk_size % hash_block_size:
        raise ValueError(
            "DSV4 LMCache chunk size must be divisible by the virtual DCP "
            f"block size: chunk={chunk_size}, virtual_block={hash_block_size}"
        )
    resume_alignment = lcm(chunk_size, hash_block_size)

    checkpoint_interval = max(
        0,
        int(getattr(config, "state_checkpoint_interval_tokens", 0) or 0),
    )
    checkpoint_interval -= checkpoint_interval % hash_block_size
    sidecar_interval = (
        lcm(checkpoint_interval, resume_alignment) if checkpoint_interval else 0
    )

    hf_config = getattr(config, "hf_config", None)
    return DSV4OffloadProfile(
        name="deepseek-v4-page-slot",
        block_size=block_size,
        dcp_size=dcp_size,
        hash_block_size=hash_block_size,
        chunk_size=chunk_size,
        resume_alignment=resume_alignment,
        checkpoint_interval=checkpoint_interval,
        sidecar_interval=sidecar_interval,
        kv_head_dim=int(getattr(hf_config, "kv_head_dim", 512) or 512),
        index_head_dim=int(getattr(hf_config, "index_head_dim", 128) or 128),
    )


def sidecar_boundary_tokens(
    *,
    num_prompt_tokens: int,
    resume_alignment: int,
    sidecar_interval: int,
) -> tuple[int, ...]:
    """Return only regular interval-aligned PAGE+SLOT checkpoints.

    PAGE still saves every LMCache chunk.  A terminal prompt that is not on the
    configured SLOT interval must not create an extra state checkpoint.
    """

    num_prompt_tokens = max(0, int(num_prompt_tokens))
    resume_alignment = int(resume_alignment)
    sidecar_interval = max(0, int(sidecar_interval))
    if resume_alignment <= 0 or sidecar_interval <= 0:
        return ()
    terminal = (num_prompt_tokens // resume_alignment) * resume_alignment
    if terminal <= 0:
        return ()
    return tuple(
        boundary
        for boundary in range(sidecar_interval, terminal + 1, sidecar_interval)
        if boundary > 0 and boundary % resume_alignment == 0
    )


def select_pending_sidecar_boundary(
    records: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    start: int,
    end: int,
    committed_hashes,
    inflight: tuple[object, int, int] | None,
    failed: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Select the earliest unpublished boundary crossed by a prefill chunk."""

    if inflight is not None:
        return None

    for boundary, boundary_hash in records:
        identity = (boundary, boundary_hash)
        if not int(start) < boundary <= int(end):
            continue
        if boundary_hash in committed_hashes or identity in failed:
            continue
        return identity
    return None


class _BoundedLRUSet(MutableSet):
    """Set-like bounded index whose duplicate adds refresh recency."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("bounded LRU set capacity must be a positive integer")
        self.capacity = capacity
        self._entries: OrderedDict[object, None] = OrderedDict()

    def __contains__(self, value: object) -> bool:
        return value in self._entries

    def __iter__(self) -> Iterator:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, value) -> None:
        self._entries.pop(value, None)
        self._entries[value] = None
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def discard(self, value) -> None:
        self._entries.pop(value, None)

    def clear(self) -> None:
        self._entries.clear()

    def __eq__(self, other) -> bool:
        if isinstance(other, (set, _BoundedLRUSet)):
            return set(self) == set(other)
        return NotImplemented


def _committed_sidecar_capacity(kvc) -> int:
    extra = (kvc or {}).get("kv_connector_extra_config", kvc or {}) or {}
    configured = extra.get("committed_sidecar_index_capacity")
    if configured is None:
        raw = os.environ.get("OFFLOAD_COMMITTED_SIDECAR_CAPACITY", "65536")
        try:
            capacity = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "committed sidecar index capacity must be a positive integer"
            ) from exc
    else:
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise ValueError(
                "committed sidecar index capacity must be a positive integer"
            )
        capacity = configured
    if capacity <= 0:
        raise ValueError("committed sidecar index capacity must be a positive integer")
    return capacity


def _chained_prefix_hashes(
    token_ids: list[int],
    hash_block_size: int,
) -> dict[int, int]:
    """Return each full-block prefix hash using BlockManager's exact chain."""

    if hash_block_size <= 0:
        raise ValueError("hash_block_size must be positive")

    from atom.model_engine.block_manager import BlockManager

    hashes: dict[int, int] = {}
    parent = -1
    for boundary in range(hash_block_size, len(token_ids) + 1, hash_block_size):
        parent = BlockManager.compute_hash(
            token_ids[boundary - hash_block_size : boundary],
            parent,
        )
        hashes[boundary] = parent
    return hashes


def _compute_slot_fingerprint(
    *,
    model_tag: str,
    page_namespace: str,
    kv_dtype: str,
    compress_ratios,
    block_size: int,
    kv_head_dim: int,
    index_head_dim: int,
    num_slots: int,
    slot_regions,
    tp_size: int,
    tp_rank: int,
) -> bytes:
    """Hash stable model and SLOT geometry into a rank-local 16-byte identity."""

    document = {
        "schema": "atom-slot-sidecar-v1",
        "model_tag": str(model_tag),
        "page_namespace": str(page_namespace),
        "kv_dtype": str(kv_dtype),
        "compress_ratios": [int(ratio) for ratio in compress_ratios],
        "block_size": int(block_size),
        "kv_head_dim": int(kv_head_dim),
        "index_head_dim": int(index_head_dim),
        "num_slots": int(num_slots),
        "slot_regions": [
            {
                "unit_bytes": int(region.unit_bytes),
                "total_bytes": int(region.total_bytes),
                "reverse_indexed": bool(region.reverse_indexed),
            }
            for region in slot_regions
        ],
        "tp_size": int(tp_size),
        "tp_rank": int(tp_rank),
    }
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.blake2b(
        canonical,
        digest_size=16,
        person=b"ATOM-SLOT-CFG-v1",
    ).digest()


__all__ = [
    "DSV4OffloadProfile",
    "HybridProfile",
    "build_dsv4_profile",
    "select_pending_sidecar_boundary",
    "sidecar_boundary_tokens",
]
