# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Scheduler-owned credits for saves, including sources of live requests."""

import os
from dataclasses import dataclass
from time import monotonic

from atom.kv_transfer.disaggregation.types import SaveOperationId
from atom.kv_transfer.offload._offload_common import max_pending_saves
from atom.kv_transfer.offload.dense.save_executor import save_admission_enabled


def build_save_budget(config, virtual_block_size: int) -> "SaveAdmissionBudget":
    kvc = getattr(config, "kv_transfer_config", {}) or {}
    extra = kvc.get("kv_connector_extra_config", kvc) or {}

    def limit(name, env, default):
        raw = extra.get(name, os.environ.get(env, default))
        if raw is None:
            return None
        if isinstance(raw, bool) or str(raw) != str(int(raw)) or int(raw) <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(raw)

    pool_blocks = int(getattr(config, "num_kvcache_blocks", 0) or 0)
    block_bytes = int(getattr(config, "kv_cache_block_bytes", 0) or 0)
    source_cap = limit(
        "max_reserved_source_blocks",
        "OFFLOAD_MAX_RESERVED_SOURCE_BLOCKS",
        max(1, pool_blocks // 10) if pool_blocks else None,
    )
    byte_cap = limit(
        "max_pending_save_bytes",
        "OFFLOAD_MAX_PENDING_SAVE_BYTES",
        source_cap * block_bytes if source_cap and block_bytes else None,
    )
    token_cap = limit(
        "max_pending_save_tokens",
        "OFFLOAD_MAX_PENDING_SAVE_TOKENS",
        (
            (source_cap * virtual_block_size if source_cap else 131072)
            if not block_bytes
            else None
        ),
    )
    enabled = save_admission_enabled()
    return SaveAdmissionBudget(
        max_operations=(
            max_pending_saves(kvc, int(os.environ.get("OFFLOAD_COPY_WORKERS", "1")))
            if enabled
            else None
        ),
        max_source_blocks=source_cap if enabled else None,
        max_pending_bytes=byte_cap if enabled else None,
        max_pending_tokens=token_cap if enabled else None,
        bytes_per_block=block_bytes,
    )


@dataclass
class SaveReservation:
    unsafe_blocks: set[int]
    tokens: int
    byte_count: int
    created_at: float


class SaveAdmissionBudget:
    """Single scheduler-thread accounting; failed stores retain their credits.

    Source credits retire incrementally, but operation/payload credits remain
    until all workers have both terminated and proved no future source reads.
    Shared physical blocks are conservatively charged once per operation.
    """

    def __init__(
        self,
        *,
        max_operations: int | None,
        max_source_blocks: int | None,
        max_pending_bytes: int | None,
        max_pending_tokens: int | None,
        bytes_per_block: int,
    ) -> None:
        for cap in (
            max_operations,
            max_source_blocks,
            max_pending_bytes,
            max_pending_tokens,
        ):
            if cap is not None and cap <= 0:
                raise ValueError("save admission limits must be positive")
        if bytes_per_block < 0 or (max_pending_bytes and not bytes_per_block):
            raise ValueError("a byte limit requires measured PAGE block bytes")
        self.max_operations = max_operations
        self.max_source_blocks = max_source_blocks
        self.max_pending_bytes = max_pending_bytes
        self.max_pending_tokens = max_pending_tokens
        self.bytes_per_block = bytes_per_block
        self.operations: dict[SaveOperationId, SaveReservation] = {}
        self.source_blocks = 0
        self.pending_bytes = 0
        self.pending_tokens = 0

    def reserve(self, operation, block_ids, tokens: int) -> str | None:
        if operation in self.operations:
            raise ValueError("save operation already reserved")
        blocks = set(block_ids)
        byte_count = len(blocks) * self.bytes_per_block
        for value, cap, reason in (
            (len(self.operations) + 1, self.max_operations, "pending_ops"),
            (self.source_blocks + len(blocks), self.max_source_blocks, "source_blocks"),
            (self.pending_bytes + byte_count, self.max_pending_bytes, "pending_bytes"),
            (self.pending_tokens + tokens, self.max_pending_tokens, "pending_tokens"),
        ):
            if cap is not None and value > cap:
                return reason
        self.operations[operation] = SaveReservation(
            blocks, tokens, byte_count, monotonic()
        )
        self.source_blocks += len(blocks)
        self.pending_bytes += byte_count
        self.pending_tokens += tokens
        return None

    def source_safe(self, operation, block_ids) -> set[int]:
        reservation = self.operations.get(operation)
        if reservation is None:
            return set()
        safe = reservation.unsafe_blocks.intersection(block_ids)
        reservation.unsafe_blocks.difference_update(safe)
        self.source_blocks -= len(safe)
        return safe

    def retire(self, operation) -> None:
        reservation = self.operations.pop(operation, None)
        if reservation is not None:
            self.source_blocks -= len(reservation.unsafe_blocks)
            self.pending_bytes -= reservation.byte_count
            self.pending_tokens -= reservation.tokens

    def oldest_age_ms(self) -> int:
        if not self.operations:
            return 0
        oldest = min(item.created_at for item in self.operations.values())
        return max(0, int((monotonic() - oldest) * 1000))
