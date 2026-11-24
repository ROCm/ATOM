# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Connectors that provide the paged KV pool itself (an L1 the ranks share),
rather than moving KV around a pool this process manages."""


def register_backends(factory) -> None:
    """Register the pool-providing connectors once the factory exists."""
    factory.register(
        "umbp_pool",
        worker_module="atom.kv_transfer.pool.umbp.connector",
        worker_class="UmbpPoolConnector",
        scheduler_module="atom.kv_transfer.pool.umbp.connector",
        scheduler_class="UmbpPoolConnectorScheduler",
        requires_pd_staging=False,
        # A rank that lacks a prefix copies it from whichever rank holds it.
        shares_pool="atom.kv_transfer.pool.umbp.connector:pool_sharing",
        # A block moves between ranks as the bytes of its PAGE regions, whatever
        # they encode, so NVFP4 KV and index pages move as any other page.
        moves_nvfp4_pages=True,
        # ... and so does a DSpark draft's own pool, whose regions are merged
        # into the block's: its dtype does not enter either.
        moves_own_dtype_draft=True,
    )
