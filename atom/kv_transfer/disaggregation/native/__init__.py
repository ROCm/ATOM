# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""HIP VMM KV-transfer primitives: scale-up (XGMI, POSIX fd) and scale-out
(IFOE fabric handle) share one code path, selected by topology."""

from atom.kv_transfer.disaggregation.native.vmm import (
    VmmBuffer,
    supported,
    supported_fabric,
)

__all__ = ["VmmBuffer", "supported", "supported_fabric"]
