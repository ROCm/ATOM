# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM LMCache KV-offload connectors.

Registers the legacy in-process ``lmcache_offload`` backend and the standalone
server ``lmcache_mp`` backend with the shared KV connector factory.
Enable via ``--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'``
plus LMCache env (``LMCACHE_LOCAL_CPU=True``, ``LMCACHE_MAX_LOCAL_CPU_SIZE``,
``LMCACHE_CHUNK_SIZE=256``, optional ``LMCACHE_LOCAL_DISK`` for the NVMe L3 tier).

For standalone MP mode, start ``lmcache server`` and select
``{"kv_connector":"lmcache_mp","kv_role":"offload"}``; the active attention
backend publishes its PAGE layout directly to the model-neutral connector.
"""

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

KVConnectorFactory.register(
    "lmcache_offload",
    worker_module="atom.kv_transfer.offload.connector",
    worker_class="LMCacheOffloadConnector",
    scheduler_module="atom.kv_transfer.offload.connector",
    scheduler_class="LMCacheOffloadConnectorScheduler",
    aliases=("LMCacheOffloadConnector", "LMCacheConnectorV1"),
    requires_pd_staging=False,
)

KVConnectorFactory.register(
    "lmcache_mp",
    worker_module="atom.kv_transfer.offload.mp.connector",
    worker_class="LMCacheMPConnector",
    scheduler_module="atom.kv_transfer.offload.mp.connector",
    scheduler_class="LMCacheMPConnectorScheduler",
    aliases=("LMCacheMPConnector",),
    requires_pd_staging=False,
)
