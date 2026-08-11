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
    num_consumer_regions: int,
    pp_size: int,
) -> list[int] | None:
    """Map a PP stage's local RDMA regions to consumer indices (group-major layout).

    The group stride is derived from the consumer's own region count rather than
    from ``num_hidden_layers``: a group holds one region per *bound* layer, and a
    node running speculative decode binds the draft's KV layer too, so its groups
    are one entry wider than the target's layer count. Reading the stride off the
    consumer keeps producer and consumer in agreement without either side having
    to know whether the other drafts.

    Returns None when the layout is not expressible as uniform group-major —
    regions not a whole number of groups, groups that do not divide the consumer's
    list evenly, or a mapping that would run past the consumer's last region.
    """
    if pp_size == 1 or num_local_layers == 0 or num_local_regions == 0:
        return list(range(num_local_regions))
    groups, remainder = divmod(num_local_regions, num_local_layers)
    if remainder != 0:
        return None
    stride, extra = divmod(num_consumer_regions, groups)
    if extra != 0:
        return None
    indices = [
        g * stride + start_layer + layer
        for g in range(groups)
        for layer in range(num_local_layers)
    ]
    if indices[-1] >= num_consumer_regions:
        return None
    return indices
