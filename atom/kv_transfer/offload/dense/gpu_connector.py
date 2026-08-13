# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense-family facade for the shared raw-block LMCache GPU adapter."""

from atom.kv_transfer.offload._block_gpu_connector import (
    BlockByteCodec,
    BlockGPUConnector,
)


class DenseGPUConnector(BlockGPUConnector):
    """Dense MHA/MLA name for the common block-byte GPU adapter."""


# Kept as a source-compatible protocol name for callers developed before the
# common adapter was split from the dense family.
ATOMBlockByteCodec = BlockByteCodec

__all__ = ["ATOMBlockByteCodec", "DenseGPUConnector"]
