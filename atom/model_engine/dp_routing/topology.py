# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Topology requirements shared by prefix routing and its storage adapters."""

from __future__ import annotations

from typing import Any


def shared_prefix_routing_enabled(
    config: Any, *, routable_engines: int | None = None
) -> bool:
    """Whether this configuration routes requests between prefix-aware DP ranks.

    A manager supplies its actual engine count (including simulated DP).
    Workers use the equivalent count from their serialized parallel config.
    DP-attention configurations work before and after TP is folded into DP.
    RapidServe's two engines are stages of one replica, not routing targets.
    """
    if (
        getattr(config, "dp_load_balance", None) not in ("prefix_spread", "share_cache")
        or int(getattr(config, "pipeline_parallel_size", 1) or 1) > 1
        or getattr(config, "enable_rapidserve", False)
    ):
        return False
    if routable_engines is None:
        parallel = getattr(config, "parallel_config", None)
        routable_engines = int(
            getattr(
                parallel, "data_parallel_size", getattr(config, "data_parallel_size", 1)
            )
            or 1
        )
        if getattr(config, "enable_dp_attention", False):
            routable_engines *= int(getattr(config, "tensor_parallel_size", 1) or 1)
    return routable_engines > 1


def prefix_spread_enabled(config: Any, *, routable_engines: int | None = None) -> bool:
    """Whether prefix-spread placement has multiple actual routing targets."""
    return getattr(
        config, "dp_load_balance", None
    ) == "prefix_spread" and shared_prefix_routing_enabled(
        config, routable_engines=routable_engines
    )
