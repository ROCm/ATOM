# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports for the DSV4 checkpoint store."""

from atom.kv_transfer.offload.hybrid.dsv4.codec import (
    DSV4CheckpointCorruptionError,
    DSV4CheckpointStore,
)

SlotSidecarCorruptionError = DSV4CheckpointCorruptionError
SlotSidecarStore = DSV4CheckpointStore

__all__ = [
    "DSV4CheckpointCorruptionError",
    "DSV4CheckpointStore",
    "SlotSidecarCorruptionError",
    "SlotSidecarStore",
]
