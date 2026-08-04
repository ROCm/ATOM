# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM tensor-parallel all_reduce layer.

A thin wrapper over the aiter TP all_reduce that centralises the TBO-aware
routing, so call sites (model AR points such as attention output-proj and MoE
combine) don't each hand-roll an ``if tbo_aware: tbo_all_reduce else: plain``
branch — nor even thread a ``tbo_aware`` flag through. Whether an all_reduce
should overlap the partner ubatch's compute is decided here, once, from the
runtime config via ``_tbo_aware_tp_reduce``.

Only the pure TP+TBO case routes through the overlap op: TBO+DP overlaps via the
DP gather/scatter path (not this pure-TP reduce), and non-TBO wants the plain
reduce with no custom-op / Dynamo-barrier indirection. The ``tbo_all_reduce``
custom op itself still no-ops back to a plain reduce when TBO is inactive or
``ATOM_TBO_TP_AR_MODE != overlap``.
"""

import torch


def _tbo_aware_tp_reduce(tp_size: int) -> bool:
    """True iff a pure-TP all_reduce should route through the TBO-aware custom op.

    Only the pure TP+TBO case (tp>1, TBO on, no DP) benefits: the ubatch's AR
    then overlaps the partner ubatch's compute. Non-TBO keeps the plain
    all_reduce (no custom-op / Dynamo-barrier indirection), and TBO+DP is left
    untouched (that path overlaps via DP gather/scatter, not this pure-TP AR).
    """
    if tp_size <= 1:
        return False
    from atom.config import get_current_atom_config

    cfg = get_current_atom_config()
    return (
        getattr(cfg, "enable_tbo", False)
        and cfg.parallel_config.data_parallel_size <= 1
    )


def tensor_model_parallel_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """TP all_reduce; routes through the TBO-aware custom op on the pure-TP+TBO
    path (decided internally via ``_tbo_aware_tp_reduce``), else a plain reduce.

    The aiter import is lazy (inside the function): this module is pulled in
    very early via atom.model_ops.__init__ -> ... -> linear, before aiter.dist
    is fully initialised, so a top-level ``import aiter.dist.communication_op``
    resolves wrong / circularly. Importing at call time (same pattern as
    module_dispatch_ops.tbo_all_reduce) sidesteps that.
    """
    from aiter.dist.parallel_state import get_tp_group

    if _tbo_aware_tp_reduce(get_tp_group().world_size):
        return torch.ops.aiter.tbo_all_reduce(x)
    from aiter.dist.communication_op import tensor_model_parallel_all_reduce as ar

    return ar(x)
