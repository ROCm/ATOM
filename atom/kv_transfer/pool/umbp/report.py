# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The shared device pool as the Atomesh router reads it (`/kv_event_sources`).

From the report of each rank the router finds its share group: the ranks of one
node whose pools are in one group of one UMBP server copy the prefix they miss
from each other's pools before they compute it, so a rank is scored by what its
group holds, not by its own blocks alone.
"""

from __future__ import annotations

from typing import Any

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.pool.umbp.settings import UmbpPoolSettings


def kv_pool_report(config: Any) -> dict[str, Any] | None:
    """The report of the `umbp_pool` connector `config` runs, on its own view
    of `config` (a `multi` entry's); None without one."""
    for name, leaf in KVConnectorFactory.leaf_connectors(config) or ():
        if name == "umbp_pool":
            return _umbp_pool_report(leaf)
    return None


def _umbp_pool_report(config: Any) -> dict[str, Any]:
    # `pool` imports the block pools it builds on: only a server that runs the
    # connector loads it.
    from atom.kv_transfer.pool.umbp.pool import share_group_for

    settings = UmbpPoolSettings.from_transfer_config(config.kv_transfer_config)
    return {
        "backend": "umbp_pool",
        "address": settings.address,
        # As each rank's pool derives it when the setting is empty: engines
        # started alike on one server report one group.
        "group": share_group_for(config, settings),
        "share_ratio": settings.share_ratio,
        "fetch": settings.fetch,
    }
