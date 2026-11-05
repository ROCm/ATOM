# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whether K2 merges the full-attention layers' DCP decode output
(``kernels.attention.dcp_merge``), decided when mono is installed.

Decided there, not when the runner is built on mono's first step: the layers'
steps mono does not serve must combine with K2's arithmetic
(``cp_lse_a2a(rank_order=True)``), and a CUDA graph captured before the
runner exists would keep the production combine. Plain Python, so the
installer does not pull FlyDSL in.
"""

import logging

from atom.models.kimi_k3.mono.config import HEAD_DIM, MLA_HEADS
from atom.mono.runtime.consensus import tp_agree

logger = logging.getLogger("atom")


def mla_impls(inner) -> list:
    """The full-attention layers' MLA attention impls."""
    return [
        layer.self_attn.attn.impl for layer in inner.layers if not layer.is_linear_attn
    ]


def dcp_merge_ok(impls, rank: int, tp: int) -> bool:
    """This rank's verdict: a DCP group that is the TP group in rank order,
    every rank decoding all 96 heads (128 wide after project-before-merge),
    and the a2a backend -- the production merge K2's arithmetic matches."""
    if not impls:
        return False
    why = None
    for impl in impls:
        group = impl.dcp_group
        if impl.dcp_world_size != tp or group is None:
            why = f"DCP {impl.dcp_world_size} is not TP {tp}"
        elif group.rank_in_group != rank:
            why = "the DCP group's rank order is not the TP group's"
        elif not impl.pbm_enabled or impl.dcp_comm_backend != "a2a":
            why = "the merge is not the project-before-merge a2a one"
        elif impl.num_heads * tp != MLA_HEADS or impl.v_head_dim != HEAD_DIM:
            why = f"{impl.num_heads} x {impl.v_head_dim} heads a rank"
        if why is not None:
            break
    if why is not None:
        logger.info("Kimi-K3 mono DCP merge off: %s", why)
        return False
    return True


def arm_dcp_merge(inner, rank: int, tp: int, group) -> bool:
    """Every rank's verdict agreed (collective over ``group``); when it holds,
    the full-attention layers' decode steps combine in rank order from now on.
    A rank whose check fails votes no rather than raising before the vote."""
    impls = []
    try:
        impls = mla_impls(inner)
        ok = dcp_merge_ok(impls, rank, tp)
    except Exception:
        logger.exception("Kimi-K3 mono DCP merge off: checking it failed")
        ok = False
    armed = tp_agree(ok, group)
    for impl in impls if armed else ():
        impl.dcp_rank_order_merge = True
    return armed
