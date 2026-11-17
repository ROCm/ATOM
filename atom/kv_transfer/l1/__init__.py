# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Connectors that provide the paged KV pool itself (an L1 the ranks share),
rather than moving KV around a pool this process manages."""


def register_backends(factory) -> None:
    """Register the pool-providing connectors once the factory exists."""
    factory.register(
        "umbp_l1",
        worker_module="atom.kv_transfer.l1.umbp_l1.connector",
        worker_class="UmbpL1Connector",
        scheduler_module="atom.kv_transfer.l1.umbp_l1.connector",
        scheduler_class="UmbpL1ConnectorScheduler",
        requires_pd_staging=False,
        # A rank that lacks a prefix copies it from whichever rank holds it.
        shares_pool="atom.kv_transfer.l1.umbp_l1.connector:pool_sharing",
    )
