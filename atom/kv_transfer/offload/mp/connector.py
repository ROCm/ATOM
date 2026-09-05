# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public LMCache multiprocess connector.

The implementation is layout-neutral: attention backends describe their PAGE
storage through ``KVTransferTensors`` and this connector transports those views
without selecting a model-specific implementation.
"""

from atom.kv_transfer.offload.mp.backend import (
    LMCacheMPConnector,
    LMCacheMPConnectorScheduler,
)

__all__ = ["LMCacheMPConnector", "LMCacheMPConnectorScheduler"]
