# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic side-channel port layout for KV disaggregation.

Each worker binds a ZMQ side-channel at ``base_handshake_port + offset``,
where ``offset`` is a unique index for the worker's ``(pp_rank, dp_rank,
tp_rank)`` position within the instance's ``(pp_size, dp_size, tp_size)``
topology.  The consumer recomputes a producer worker's offset from the same
formula, so no port needs to be exchanged.

The pipeline dimension is the outermost stride: under pipeline parallelism a
prefill instance is several stage processes that share ``dp_rank``/``tp_rank``
and would otherwise collide on one port.
"""

from __future__ import annotations


def side_channel_port_offset(
    dp_rank: int,
    tp_rank: int,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    dp_size: int = 1,
) -> int:
    """Return the unique port offset for a worker's position in its topology.

    ``pp_rank=0, pp_size=1`` reproduces the pre-PP ``dp_rank*tp_size + tp_rank``.
    """
    return pp_rank * (dp_size * tp_size) + dp_rank * tp_size + tp_rank


def consumer_region_indices(
    num_local_regions: int,
    num_local_layers: int,
    start_layer: int,
    num_hidden_layers: int,
    pp_size: int,
) -> list[int] | None:
    """Map each of a PP stage's local RDMA regions to a consumer region index.

    Backends register KV regions **group-major**: all layers' regions of one
    kind, then all layers' regions of the next kind — e.g. MLA emits
    ``[kv_0..kv_{N-1}, index_0..index_{N-1}]``.  The consumer registers this
    over all ``num_hidden_layers`` layers; a stage registers the same groups
    over only its ``[start_layer, start_layer+num_local_layers)`` layers.

    Local region ``i`` therefore sits in group ``g = i // num_local_layers`` at
    intra-group layer ``l = i % num_local_layers`` (global layer
    ``start_layer + l``), which maps to consumer index
    ``g * num_hidden_layers + start_layer + l``.

    Returns the identity map for the non-PP case (and empty input), and ``None``
    when ``num_local_regions`` is not an integer multiple of
    ``num_local_layers`` (a non-uniform, e.g. interleaved or per-subset, layout
    this group-major mapping cannot express).
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
