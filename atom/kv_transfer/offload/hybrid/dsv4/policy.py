# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 PAGE+SLOT geometry, cadence, and commit policy.

This module deliberately contains only CPU-side policy.  GPU layout movement
lives in :mod:`.codec`, while LMCache and scheduler orchestration live in
:mod:`.connector`.
"""

from __future__ import annotations

import array
import hashlib
import heapq
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Iterator, MutableSet
from dataclasses import dataclass
from math import lcm
from numbers import Integral


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


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        # Admission exposes ValueError for all invalid capacity/id values.
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    return int(value)


class DSV4StagingAdmission:
    """Allocate connector-owned DSV4 checkpoint staging rows.

    A row remains acquired until its GPU work has been fenced.  Rows whose
    completion cannot be proven are quarantined instead of being reused.
    """

    def __init__(self, num_slots: int) -> None:
        capacity = _integer("num_slots", num_slots)
        if capacity <= 0:
            raise ValueError(f"num_slots must be > 0, got {capacity}")

        self._capacity = capacity
        self._free_ids = list(range(capacity))
        self._acquired = [False] * capacity
        self._quarantined = [False] * capacity
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def num_free(self) -> int:
        with self._lock:
            return len(self._free_ids)

    def try_acquire(self) -> int | None:
        """Return the smallest available row ID, or ``None`` when full."""
        with self._lock:
            if not self._free_ids:
                return None
            slot_id = heapq.heappop(self._free_ids)
            self._acquired[slot_id] = True
            return slot_id

    def release(self, slot_id: int) -> None:
        """Return a row after the caller has synchronized its GPU work."""
        normalized_id = _integer("slot id", slot_id)
        if not 0 <= normalized_id < self._capacity:
            raise ValueError(
                f"slot id {normalized_id} outside pool [0, {self._capacity})"
            )

        with self._lock:
            if self._quarantined[normalized_id]:
                raise ValueError(f"slot id {normalized_id} is quarantined")
            if not self._acquired[normalized_id]:
                raise ValueError(f"slot id {normalized_id} is not acquired")
            self._acquired[normalized_id] = False
            heapq.heappush(self._free_ids, normalized_id)

    def quarantine(self, slot_id: int) -> None:
        """Permanently remove a row whose GPU completion is uncertain."""
        normalized_id = _integer("slot id", slot_id)
        if not 0 <= normalized_id < self._capacity:
            raise ValueError(
                f"slot id {normalized_id} outside pool [0, {self._capacity})"
            )

        with self._lock:
            if self._quarantined[normalized_id]:
                return
            if not self._acquired[normalized_id]:
                raise ValueError(f"slot id {normalized_id} is not acquired")
            self._acquired[normalized_id] = False
            self._quarantined[normalized_id] = True


def build_dsv4_profile(config, *, chunk_size: int) -> DSV4OffloadProfile:
    """Resolve DSV4 geometry from config without consulting worker tensors."""

    block_size = _integer("DSV4 block size", config.kv_cache_block_size)
    raw_dcp_size = getattr(config, "decode_context_parallel_size", 1)
    dcp_size = _integer(
        "DSV4 DCP size",
        1 if raw_dcp_size is None else raw_dcp_size,
    )
    chunk_size = _integer("LMCache chunk size", chunk_size)
    if block_size <= 0 or dcp_size <= 0 or chunk_size <= 0:
        raise ValueError("DSV4 block, DCP, and LMCache chunk sizes must be positive")

    hash_block_size = block_size * dcp_size
    if chunk_size % hash_block_size:
        raise ValueError(
            "DSV4 LMCache chunk size must be divisible by the virtual DCP "
            f"block size: chunk={chunk_size}, virtual_block={hash_block_size}"
        )
    # Divisibility above makes the least common multiple exactly chunk_size.
    resume_alignment = chunk_size

    raw_checkpoint_interval = getattr(
        config,
        "state_checkpoint_interval_tokens",
        0,
    )
    # Three policies, and the sign carries the distinction -- see
    # `BlockManager.__init__`, which clamps the same field to `max(-1, ...)`:
    #
    #   >0  a rung every N tokens, and the sidecar aligns to it
    #    0  checkpointing off entirely
    #   -1  the grid is off but checkpointing is ON: the prompt-end anchor and
    #       the demand rung still place checkpoints, off any grid
    #
    # Clamping to `max(0, ...)` here folded -1 into 0, which for this consumer
    # means no sidecar checkpoints at all -- so a run launched with
    # `--state-checkpoint-interval-tokens -1` kept placing checkpoints in the
    # engine while offload resume silently degraded to zero reuse. There is no
    # grid to align to under -1, so the sidecar takes `resume_alignment` on its
    # own: aligned boundaries, no interval multiple imposed on top.
    checkpoint_interval = _integer(
        "DSV4 checkpoint interval",
        0 if raw_checkpoint_interval is None else raw_checkpoint_interval,
    )
    if checkpoint_interval < 0:
        checkpoint_interval = -1
        sidecar_interval = resume_alignment
    else:
        checkpoint_interval -= checkpoint_interval % hash_block_size
        sidecar_interval = (
            lcm(checkpoint_interval, resume_alignment) if checkpoint_interval else 0
        )

    hf_config = getattr(config, "hf_config", None)
    raw_kv_head_dim = getattr(hf_config, "kv_head_dim", 512)
    raw_index_head_dim = getattr(hf_config, "index_head_dim", 128)
    kv_head_dim = _integer(
        "DSV4 KV head dimension",
        512 if raw_kv_head_dim is None else raw_kv_head_dim,
    )
    index_head_dim = _integer(
        "DSV4 index head dimension",
        128 if raw_index_head_dim is None else raw_index_head_dim,
    )
    if kv_head_dim <= 0 or index_head_dim <= 0:
        raise ValueError("DSV4 KV and index head dimensions must be positive")
    return DSV4OffloadProfile(
        name="deepseek-v4-page-slot",
        block_size=block_size,
        dcp_size=dcp_size,
        hash_block_size=hash_block_size,
        chunk_size=chunk_size,
        resume_alignment=resume_alignment,
        checkpoint_interval=checkpoint_interval,
        sidecar_interval=sidecar_interval,
        kv_head_dim=kv_head_dim,
        index_head_dim=index_head_dim,
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
    token_ids: array.array,
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
    """Hash stable model and semantic SLOT layout into a rank-local identity."""

    normalized_block_size = _integer("SLOT block size", block_size)
    normalized_kv_head_dim = _integer("SLOT KV head dimension", kv_head_dim)
    normalized_index_head_dim = _integer(
        "SLOT index head dimension",
        index_head_dim,
    )
    normalized_num_slots = _integer("SLOT count", num_slots)
    normalized_tp_size = _integer("SLOT TP size", tp_size)
    normalized_tp_rank = _integer("SLOT TP rank", tp_rank)
    if (
        min(
            normalized_block_size,
            normalized_kv_head_dim,
            normalized_index_head_dim,
            normalized_num_slots,
            normalized_tp_size,
        )
        <= 0
    ):
        raise ValueError("SLOT geometry dimensions must be positive")
    if not 0 <= normalized_tp_rank < normalized_tp_size:
        raise ValueError("SLOT TP rank must be within the TP world")

    normalized_regions = []
    for region_index, region in enumerate(slot_regions):
        role = getattr(region, "semantic_role", None)
        if role is None:
            raise ValueError(f"SLOT region {region_index} semantic_role is required")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("SLOT region semantic_role must be a non-empty string")
        reverse_indexed = getattr(region, "reverse_indexed", None)
        if type(reverse_indexed) is not bool:
            raise ValueError("SLOT region reverse_indexed must be a boolean")
        normalized_regions.append(
            {
                "role": role,
                "unit_bytes": _integer(
                    f"SLOT region {region_index} unit bytes",
                    region.unit_bytes,
                ),
                "total_bytes": _integer(
                    f"SLOT region {region_index} total bytes",
                    region.total_bytes,
                ),
                "reverse_indexed": reverse_indexed,
            }
        )

    document = {
        "schema": "atom-slot-sidecar-v2",
        "layout_schema": "dsv4-semantic-region-order-v1",
        "model_tag": str(model_tag),
        "page_namespace": str(page_namespace),
        "kv_dtype": str(kv_dtype),
        "compress_ratios": [
            _integer(f"SLOT compression ratio {index}", ratio)
            for index, ratio in enumerate(compress_ratios)
        ],
        "block_size": normalized_block_size,
        "kv_head_dim": normalized_kv_head_dim,
        "index_head_dim": normalized_index_head_dim,
        "num_slots": normalized_num_slots,
        "slot_regions": normalized_regions,
        "tp_size": normalized_tp_size,
        "tp_rank": normalized_tp_rank,
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
        person=b"ATOM-SLOT-CFG-v2",
    ).digest()


__all__ = [
    "DSV4OffloadProfile",
    "DSV4StagingAdmission",
    "build_dsv4_profile",
    "select_pending_sidecar_boundary",
    "sidecar_boundary_tokens",
]
