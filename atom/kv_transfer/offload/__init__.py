# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM KV-offload connectors.

Registers three backends with the shared KV connector factory:

* ``lmcache_offload`` -- the legacy in-process LMCache engine. Enable via
  ``ATOM_KV_OFFLOAD=lmcache`` (or the equivalent
  ``--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'``)
  plus LMCache env (``LMCACHE_LOCAL_CPU=True``, ``LMCACHE_MAX_LOCAL_CPU_SIZE``,
  ``LMCACHE_CHUNK_SIZE=256``, optional ``LMCACHE_LOCAL_DISK`` for the NVMe L3
  tier).
* ``lmcache_mp`` -- a standalone ``lmcache server``: start it and set
  ``ATOM_KV_OFFLOAD=lmcache_mp`` (or select
  ``{"kv_connector":"lmcache_mp","kv_role":"offload"}``); the active attention
  backend publishes its PAGE layout directly to the model-neutral connector.
* ``mooncake_store`` -- a Mooncake Store reached directly, no LMCache involved:
  start its master and owners and set ``ATOM_KV_OFFLOAD=mooncake_store`` with
  ``ATOM_KV_OFFLOAD_EXTRA_CONFIG`` naming them (``mooncake_store.master`` and
  ``mooncake_store.metadata``, or ``mooncake_store.pools``). Dense KV, and
  Kimi-K3's KV with its KDA state images.
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
    offload=True,
    # The dense codec builds from `KVCacheTensor`s and ignores the region map;
    # the hybrid, m3 and kimi_k3 layouts do read it, and which one applies comes
    # from the model, so `topology_reads_block_regions` refines this by layout.
    reads_block_regions=False,
)

KVConnectorFactory.register(
    "lmcache_mp",
    worker_module="atom.kv_transfer.offload.mp.connector",
    worker_class="LMCacheMPConnector",
    scheduler_module="atom.kv_transfer.offload.mp.connector",
    scheduler_class="LMCacheMPConnectorScheduler",
    aliases=("LMCacheMPConnector",),
    requires_pd_staging=False,
    offload=True,
    # Groups regions by per-block shape and copies whole blocks, so a layer may
    # publish extra planes (the FP4 indexer's e8m0 scale) as more regions.
    copies_whole_block_regions=True,
)

KVConnectorFactory.register(
    "mooncake_store",
    # Each picks the dense or the Kimi-K3 class by the model's layout.
    worker_module="atom.kv_transfer.offload.mooncake_store.connector",
    worker_class="MooncakeStoreConnector",
    scheduler_module="atom.kv_transfer.offload.mooncake_store.connector",
    scheduler_class="MooncakeStoreConnectorScheduler",
    requires_pd_staging=False,
    offload=True,
    # The dense codec builds from `KVCacheTensor`s, and K3's state images come
    # from the attention backend (`state_backend`): the region map is unread.
    reads_block_regions=False,
)
