# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility alias for the former single-use hybrid profile."""

from atom.kv_transfer.offload.hybrid.dsv4.policy import (
    DSV4OffloadProfile,
    HybridProfile,
)

__all__ = ["DSV4OffloadProfile", "HybridProfile"]
