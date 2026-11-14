# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The registered memory every Store transfer of one worker goes through.

The store-agnostic :class:`~atom.kv_transfer.offload.kv_store.pool.TransferSlotPool`,
named for the Mooncake Store. The allocation sits on the worker's GPU: the NIC
reads and writes the HBM directly (GPUDirect RDMA). Mooncake gives up on a
transfer without cancelling its RDMA work, held by a NIC whose completion
queue stalls, which is what quarantine is for. Mooncake splits a large
registration into pieces of at most ``MC_MAX_MR_SIZE``; the pool's page-aligned
slots on a huge-page-aligned base keep a split from cutting a slot.
"""

from __future__ import annotations

from atom.kv_transfer.offload.kv_store import pool as _store_pool
from atom.kv_transfer.offload.kv_store.pool import (
    REGIONS,
    Slot,
    SlotPoolExhausted,
    pool_allocation_bytes,
)


class TransferSlotPool(_store_pool.TransferSlotPool):
    """The transfer slots of one Mooncake Store worker."""

    LABEL = "Mooncake Store offload"
    SETTING_PREFIX = "mooncake_store."


__all__ = [
    "REGIONS",
    "Slot",
    "SlotPoolExhausted",
    "TransferSlotPool",
    "pool_allocation_bytes",
]
