# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM standalone LMCache CPU/NVMe KV-offload connector.

Registers the ``lmcache_offload`` backend with the shared KV connector factory.
Enable via ``--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'``
plus LMCache env (``LMCACHE_LOCAL_CPU=True``, ``LMCACHE_MAX_LOCAL_CPU_SIZE``,
``LMCACHE_CHUNK_SIZE=256``, optional ``LMCACHE_LOCAL_DISK`` for the NVMe L3 tier).

One backend name, two layout families: ``lmcache_offload`` resolves to the
``hybrid`` connector (DSV4 terminal opaque-unit; extensible to Qwen3-Next /
Kimi K3 hybrid KV via profiles) or the ``dense`` connector (DSV2/V3 MLA / MHA)
from ``config`` — see :mod:`atom.kv_transfer.offload.connector`. DSV2/V3 and DSV4
serve with the same ``kv_connector`` value. There is no separate backend name
per model; pick the family with ``offload_layout`` if auto-inference is wrong.
"""

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

# Single public worker + scheduler shell. Each shell picks its layout-family
# impl (dense / hybrid) from config at construction and delegates every hook;
# see offload/shell.py + offload/dispatch.py::select_variant.
KVConnectorFactory.register(
    "lmcache_offload",
    worker_module="atom.kv_transfer.offload.connector",
    worker_class="LMCacheOffloadConnector",
    scheduler_module="atom.kv_transfer.offload.connector",
    scheduler_class="LMCacheOffloadConnectorScheduler",
)
