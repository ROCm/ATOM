# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM tensor-parallel all_reduce layer.

A thin wrapper over the aiter TP all_reduce that centralises the TBO-aware
routing, so call sites (model AR points such as attention output-proj and MoE
combine) don't each hand-roll an ``if tbo_aware: tbo_all_reduce else: plain``
branch. Pass ``tbo_aware=True`` at a site that wants its all_reduce to overlap
the partner ubatch's compute under Two-Batch Overlap; leave it False (default)
for the plain reduce.

``tbo_aware`` is a per-site opt-in rather than an internal ``tbo_active()``
probe on purpose: TBO+DP overlaps via the DP gather/scatter path, not this
pure-TP reduce, so only sites explicitly on the pure-TP+TBO path should route
here. The ``tbo_all_reduce`` custom op still no-ops back to a plain reduce when
TBO is inactive or ``ATOM_TBO_TP_AR_MODE != overlap``.
"""

import torch


def tensor_model_parallel_all_reduce(
    x: torch.Tensor, tbo_aware: bool = False
) -> torch.Tensor:
    """TP all_reduce; routes through the TBO-aware custom op when tbo_aware.

    The aiter import is lazy (inside the function): this module is pulled in
    very early via atom.model_ops.__init__ -> ... -> linear, before aiter.dist
    is fully initialised, so a top-level ``import aiter.dist.communication_op``
    resolves wrong / circularly. Importing at call time (same pattern as
    module_dispatch_ops.tbo_all_reduce) sidesteps that.
    """
    if tbo_aware:
        return torch.ops.aiter.tbo_all_reduce(x)
    from aiter.dist.communication_op import tensor_model_parallel_all_reduce as ar

    return ar(x)
