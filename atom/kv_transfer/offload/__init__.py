# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM KV-offload connectors.

Defines four backends registered by the shared KV connector factory:

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
  ``mooncake_store.metadata``, or ``mooncake_store.pools``). Moves the dense
  KV layout, MiniMax-M3/M3.1 PAGE layout, and V4.1's PAGE regions (state is
  rebuilt by replay). Other stateful layouts and DCP greater than 1 stay
  refused.
* ``umbp_store`` -- put and get KV chunks to a Mori UMBP standalone server
  that runs outside ATOM: start ``umbp_standalone_server`` and select
  ``{"kv_connector":"umbp_store","kv_role":"offload",
  "kv_connector_extra_config":{"umbp_store.address":"unix:///..."}}``. A
  backend of its own, copied from ``mooncake_store``; dense and
  MiniMax-M3/M3.1 PAGE layouts, stateful layouts and DCP greater than 1
  refused.
"""


def offload_layout_reads_block_regions(config) -> bool:
    """Whether this model's offload codec reads the PAGE region map.

    The dense codec builds from ``KVCacheTensor``s and ignores the map. Every
    other layout packs PAGE bytes from it, and which layout applies is a
    property of the model. ``offload.config`` imports torch, and this package
    is imported while backends are registered, so the selector is loaded when
    the question is asked.
    """
    from atom.kv_transfer.offload.config import select_offload_layout

    return select_offload_layout(config) != "dense"


def register_backends(factory) -> None:
    """Register offload transports after the connector factory is initialized."""
    factory.register(
        "lmcache_offload",
        worker_module="atom.kv_transfer.offload.connector",
        worker_class="LMCacheOffloadConnector",
        scheduler_module="atom.kv_transfer.offload.connector",
        scheduler_class="LMCacheOffloadConnectorScheduler",
        aliases=("LMCacheOffloadConnector", "LMCacheConnectorV1"),
        requires_pd_staging=False,
        offload=True,
        # The dense codec ignores the region map; every other layout reads it.
        # Which one applies comes from the model, so the registration says so.
        reads_block_regions=offload_layout_reads_block_regions,
    )

    factory.register(
        "lmcache_mp",
        worker_module="atom.kv_transfer.offload.mp.connector",
        worker_class="LMCacheMPConnector",
        scheduler_module="atom.kv_transfer.offload.mp.connector",
        scheduler_class="LMCacheMPConnectorScheduler",
        aliases=("LMCacheMPConnector",),
        requires_pd_staging=False,
        offload=True,
        # One LMCache server for every DP rank.
        shared_tier=True,
        chunk_tokens="atom.kv_transfer.offload.config:lmcache_chunk_tokens",
        # Groups regions by per-block shape and copies whole blocks, so a layer may
        # publish extra planes (the FP4 indexer's e8m0 scale) as more regions.
        copies_whole_block_regions=True,
    )

    factory.register(
        "mooncake_store",
        worker_module="atom.kv_transfer.offload.mooncake_store.worker",
        worker_class="MooncakeStoreOffloadConnector",
        scheduler_module="atom.kv_transfer.offload.mooncake_store.scheduler",
        scheduler_class="MooncakeStoreOffloadScheduler",
        requires_pd_staging=False,
        offload=True,
        # One Mooncake Store for every DP rank.
        shared_tier=True,
        chunk_tokens="atom.kv_transfer.offload.mooncake_store.config:store_chunk_tokens",
        # Same predicate as lmcache_offload: dense ignores the region map, and
        # M3 and V4.1 PAGE layouts read it. Other layouts are refused at
        # startup; see `check_engine_compatibility`.
        reads_block_regions=offload_layout_reads_block_regions,
        # Dense and PAGE codecs preserve all planes, including FP4 index scales.
        copies_whole_block_regions=True,
    )

    factory.register(
        "umbp_store",
        worker_module="atom.kv_transfer.offload.umbp_store.worker",
        worker_class="UMBPStoreOffloadConnector",
        scheduler_module="atom.kv_transfer.offload.umbp_store.scheduler",
        scheduler_class="UMBPStoreOffloadScheduler",
        requires_pd_staging=False,
        offload=True,
        # One UMBP standalone server per node, shared by every DP rank there.
        shared_tier=True,
        chunk_tokens="atom.kv_transfer.offload.umbp_store.config:store_chunk_tokens",
        # Dense ignores the region map, the M3 PAGE layout reads it (its NSA
        # index cache included). Other layouts are refused at startup; see
        # `check_engine_compatibility`.
        reads_block_regions=offload_layout_reads_block_regions,
        # Dense and PAGE codecs preserve all planes.
        copies_whole_block_regions=True,
    )
