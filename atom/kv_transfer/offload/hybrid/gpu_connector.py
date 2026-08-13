# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hybrid-family facade for the shared raw-block LMCache GPU adapter."""

from atom.kv_transfer.offload._block_gpu_connector import (
    BlockByteCodec,
    BlockGPUConnector,
)


class HybridGPUConnector(BlockGPUConnector):
    """PAGE-region name for the common block-byte GPU adapter."""


__all__ = ["BlockByteCodec", "HybridGPUConnector"]
