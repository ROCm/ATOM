# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM standalone LMCache CPU/NVMe KV-offload connector.

Registers the ``lmcache_offload`` backend with the shared KV connector factory.
Enable via ``--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'``
plus LMCache env (``LMCACHE_LOCAL_CPU=True``, ``LMCACHE_MAX_LOCAL_CPU_SIZE``,
``LMCACHE_CHUNK_SIZE=256``, optional ``LMCACHE_LOCAL_DISK`` for the NVMe L3 tier).
"""

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

KVConnectorFactory.register(
    "lmcache_offload",
    worker_module="atom.kv_transfer.offload.connector",
    worker_class="LMCacheOffloadConnector",
    scheduler_module="atom.kv_transfer.offload.connector",
    scheduler_class="LMCacheOffloadConnectorScheduler",
)
