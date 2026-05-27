# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Local KV-cache offload (CPU DRAM / NVMe) for ATOM standalone serving.

Importing this package registers all built-in offload backends with the
shared :class:`atom.kv_transfer.disaggregation.factory.KVConnectorFactory`
so they can be selected via ``--kv-transfer-config '{"kv_connector": ...}'``
on the command line. See the package README for the data flow and the
``LMCacheOffloadConnector`` recipe for end-to-end setup.
"""

from atom.kv_transfer.offload.base import (
    OffloadConnectorBase,
    OffloadConnectorSchedulerBase,
)
from atom.kv_transfer.offload.types import (
    OffloadConnectorMetadata,
    OffloadReqMeta,
)

__all__ = [
    "OffloadConnectorBase",
    "OffloadConnectorSchedulerBase",
    "OffloadConnectorMetadata",
    "OffloadReqMeta",
]

# Backend registrations happen here so a bare `import atom.kv_transfer.offload`
# arms every built-in. The concrete lmcache module isn't imported eagerly
# (avoids pulling lmcache at engine start if offload is not in use); the
# factory uses the module string and lazy-imports on instantiation.
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

KVConnectorFactory.register(
    "lmcache_offload",
    worker_module="atom.kv_transfer.offload.lmcache.lmcache_connector",
    worker_class="LMCacheOffloadConnector",
    scheduler_module="atom.kv_transfer.offload.lmcache.lmcache_connector",
    scheduler_class="LMCacheOffloadConnectorScheduler",
)
