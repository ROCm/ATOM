# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM tensor-parallel all_reduce layer.

A thin wrapper over the aiter TP all_reduce -- and the fused all_reduce +
RMSNorm (+ quant) the norms run -- that centralises the TBO-aware
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


def _register_tbo_ops() -> None:
    """Registers the TBO-aware ops (``torch.ops.aiter.tbo_*``). They live in
    ``module_dispatch_ops``, which a model need not import: the model registry
    imports a model's module only when it is built."""
    from atom.model_ops import module_dispatch_ops  # noqa: F401


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
        _register_tbo_ops()
        return torch.ops.aiter.tbo_all_reduce(x)
    from aiter.dist.communication_op import tensor_model_parallel_all_reduce as ar

    return ar(x)


def tensor_model_parallel_fused_allreduce_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """aiter's fused TP all_reduce + residual add + RMSNorm, ``(out,
    residual_out)``; ``gemma_norm`` normalizes with ``x * (1 + w)``. Routed like
    ``tensor_model_parallel_all_reduce``: on the pure-TP+TBO path through the
    TBO-aware op, which on a split step hands the all_reduce to the comm stream
    and runs the norm after it, and runs the fused kernel otherwise."""
    from aiter.dist.parallel_state import get_tp_group

    if _tbo_aware_tp_reduce(get_tp_group().world_size):
        _register_tbo_ops()
        return torch.ops.aiter.tbo_fused_allreduce_rmsnorm(
            x, residual, weight, eps, gemma_norm
        )
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm as fused,
    )

    return fused(x, residual, weight, eps, gemma_norm=gemma_norm)


def tensor_model_parallel_fused_allreduce_rmsnorm_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    quant_type: str = "per_token",
    group_size: int = 128,
    emit_bf16: bool = False,
    transpose_scale: bool = False,
    gemma_norm: bool = False,
):
    """aiter's fused TP all_reduce + residual add + RMSNorm + quant, ``(out,
    residual_out, scale)``, plus the bf16 normed activation if ``emit_bf16``.
    ``transpose_scale`` lays a per-group scale out column-major. Routed like
    ``tensor_model_parallel_fused_allreduce_rmsnorm`` for the FP8 quants a
    split step runs unfused, ``"per_token"`` and ``"per_group"``; any other
    ``quant_type`` keeps the fused kernel."""
    from aiter.dist.parallel_state import get_tp_group

    if quant_type in ("per_token", "per_group") and _tbo_aware_tp_reduce(
        get_tp_group().world_size
    ):
        _register_tbo_ops()
        out, residual_out, scale, bf16 = (
            torch.ops.aiter.tbo_fused_allreduce_rmsnorm_quant(
                x,
                residual,
                weight,
                eps,
                quant_type,
                group_size,
                emit_bf16,
                transpose_scale,
                gemma_norm,
            )
        )
        if emit_bf16:
            return out, residual_out, scale, bf16
        return out, residual_out, scale
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm_quant as fused,
    )

    return fused(
        x,
        residual,
        weight,
        eps,
        quant_type=quant_type,
        group_size=group_size,
        emit_bf16=emit_bf16,
        transpose_scale=transpose_scale,
        gemma_norm=gemma_norm,
    )


def tensor_model_parallel_all_reduce_add(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """All-reduce ``x + y`` across the TP group.

    The result is the sum over ranks of each rank's ``x + y``, not
    ``all_reduce(x) + y``. Requires aiter at or after ROCm/aiter#7499
    (``06b7fd32``). aiter takes the gfx1250 HIP Lamport one-shot kernel where
    the input fits and otherwise adds, then all-reduces. The TBO path stays
    here, so it gets that materialized sum.
    """
    from aiter.dist.parallel_state import get_tp_group

    if _tbo_aware_tp_reduce(get_tp_group().world_size):
        return tensor_model_parallel_all_reduce(x + y)
    from aiter.dist.communication_op import (
        tensor_model_parallel_all_reduce_add as aiter_all_reduce_add,
    )

    return aiter_all_reduce_add(x, y)
