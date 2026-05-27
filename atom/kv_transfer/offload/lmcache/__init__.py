# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache-backed offload connector.

Loaded lazily via the KVConnectorFactory registry entry in
``atom.kv_transfer.offload.__init__``. Importing this submodule directly
also works (and forces lmcache + c_ops to be imported at process start).
"""

from atom.kv_transfer.offload.lmcache.lmcache_connector import (
    LMCacheOffloadConnector,
    LMCacheOffloadConnectorScheduler,
)

__all__ = [
    "LMCacheOffloadConnector",
    "LMCacheOffloadConnectorScheduler",
]
