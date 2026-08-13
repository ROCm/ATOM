# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 PAGE+SLOT LMCache offload vertical package."""

from atom.kv_transfer.offload.hybrid.dsv4.codec import (
    DSV4CheckpointCodec,
    DSV4CheckpointCorruptionError,
    DSV4CheckpointError,
    DSV4CheckpointHeader,
    DSV4CheckpointKey,
    DSV4CheckpointStore,
    DSV4CopyPlan,
    DSV4CopySpan,
    DSV4PageSlotCodec,
    DSV4PayloadKind,
    DSV4PayloadSection,
)
from atom.kv_transfer.offload.hybrid.dsv4.policy import (
    DSV4OffloadProfile,
    build_dsv4_profile,
)


def __getattr__(name: str):
    if name in {"DSV4OffloadConnector", "DSV4OffloadScheduler"}:
        from atom.kv_transfer.offload.hybrid.dsv4.connector import (
            HybridOffloadConnector,
            HybridOffloadScheduler,
        )

        return (
            HybridOffloadConnector
            if name == "DSV4OffloadConnector"
            else HybridOffloadScheduler
        )
    raise AttributeError(name)


__all__ = [
    "DSV4CheckpointCodec",
    "DSV4CheckpointCorruptionError",
    "DSV4CheckpointError",
    "DSV4CheckpointHeader",
    "DSV4CheckpointKey",
    "DSV4CheckpointStore",
    "DSV4CopyPlan",
    "DSV4CopySpan",
    "DSV4OffloadConnector",
    "DSV4OffloadProfile",
    "DSV4OffloadScheduler",
    "DSV4PageSlotCodec",
    "DSV4PayloadKind",
    "DSV4PayloadSection",
    "build_dsv4_profile",
]
