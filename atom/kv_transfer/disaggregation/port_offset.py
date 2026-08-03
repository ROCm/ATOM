# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic side-channel port layout for KV disaggregation."""

from __future__ import annotations


def side_channel_port_offset(
    dp_rank: int,
    tp_rank: int,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    dp_size: int = 1,
) -> int:
    """Return the unique port offset for a worker's (pp, dp, tp) position."""
    return pp_rank * (dp_size * tp_size) + dp_rank * tp_size + tp_rank


def consumer_region_indices(
    num_local_regions: int,
    num_local_layers: int,
    start_layer: int,
    num_hidden_layers: int,
    pp_size: int,
) -> list[int] | None:
    """Map a PP stage's local RDMA regions to consumer indices (group-major layout).

    Returns None if num_local_regions is not a multiple of num_local_layers.
    """
    if pp_size == 1 or num_local_layers == 0 or num_local_regions == 0:
        return list(range(num_local_regions))
    groups, remainder = divmod(num_local_regions, num_local_layers)
    if remainder != 0:
        return None
    return [
        g * num_hidden_layers + start_layer + layer
        for g in range(groups)
        for layer in range(num_local_layers)
    ]
