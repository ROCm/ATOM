# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which kernel runs renormalized softmax top-k routing: the Triton one in
`triton_topk_softmax.py` or AITER's `topk_softmax`.

A MoE layer decides once, when it is built, with `triton_topk_softmax_rows`;
each launch then only compares its row count. No Triton or AITER import at
module level, so the GPU-less CI runs their tests too.
"""

from atom.utils import envs

# Per GPU architecture, launches of up to this many rows per compute unit take
# the Triton kernel, longer ones AITER's. Both kernels' times step with rows
# per CU: Triton runs one single-wave program per row, AITER packs 8-64 rows
# into each block. Measured on MI355X (gfx950, 256 CUs), Triton is faster up
# to 20 rows per CU (256 experts / top-8 with or without a shared expert,
# 32 / top-4), 22 (128 / top-4), 24 (128 / top-8, 512 / top-10) or 28 (512 /
# top-10 + a shared expert), and AITER past it; 20 keeps Triton only where it
# is faster for all. An architecture without an entry has not been measured
# and keeps every launch on AITER.
#
# The count is per CU the launch can use. On RapidServe's CU-masked streams
# (half of MI355X's CUs) the two are about even at 22-24 rows per CU and
# Triton is slower from 26 (30 for 512 / top-10 + a shared expert), by up to
# 36% at 40; at 20 it is 10-24% faster for all of the shapes above.
TRITON_TOPK_SOFTMAX_MAX_ROWS_PER_CU = {"gfx950": 20}


def _gfx() -> str:
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx()


def _cu_num() -> int:
    from aiter.jit.utils.chip_info import get_cu_num

    return get_cu_num()


def _masked_cu_fraction() -> float:
    from atom.model_engine.model_runner import RapidServeModelRunner

    # A prefill stream gets fraction f of the CUs, its decode stream the rest.
    return min(min(f, 1 - f) for f in RapidServeModelRunner._CU_POOL_FRACTIONS)


def triton_topk_softmax_max_rows(cu_masked: bool = False) -> int | None:
    """The most rows a launch can have and still take the Triton kernel on
    this GPU; None where the crossover has not been measured. `cu_masked`:
    launches may run on RapidServe's CU-masked streams, so the cutoff is
    counted on the fewest CUs such a stream gets."""
    per_cu = TRITON_TOPK_SOFTMAX_MAX_ROWS_PER_CU.get(_gfx())
    if per_cu is None:
        return None
    cus = _cu_num()
    if cu_masked:
        cus = round(cus * _masked_cu_fraction())
    return per_cu * cus


def triton_topk_softmax_supported(
    num_experts: int,
    topk: int,
    renormalize: bool,
    num_shared_experts: int,
    shared_experts_scoring_func: str,
) -> bool:
    """Whether `triton_topk_softmax` takes routings of this shape: the ones
    AITER's fused `topkGatingSoftmax` handles (power-of-two expert counts up
    to 512; past 512 AITER takes `moeSoftmax` + `moeTopK`, which round the
    weights differently), renormalized, the shared columns unscored or
    sigmoid-scored. `num_shared_experts` counts the scored shared columns,
    whose logits follow the `num_experts` routed ones."""
    return (
        renormalize
        and shared_experts_scoring_func in ("", "sigmoid")
        and 32 <= num_experts <= 512
        and num_experts & (num_experts - 1) == 0
        and 0 < topk <= 32
        and num_shared_experts in (0, 1, 2, 4, 8)
    )


def triton_topk_softmax_rows(
    num_experts: int,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int = 0,
    fused_shared_experts_scoring_func: str | None = None,
    cu_masked: bool = False,
) -> int | None:
    """For a layer's softmax top-k routing (`rocm_aiter_topk_softmax_impl`'s
    arguments, `num_experts` routed ones), the most rows a launch can have
    and still run `triton_topk_softmax` rather than AITER's `topk_softmax`;
    None if none can: `ATOM_TRITON_TOPK_SOFTMAX=0`, a shape the Triton
    kernel does not take, or a GPU without a measured cutoff. `cu_masked`:
    the layer may run on RapidServe's CU-masked streams
    (`--disagg-constrained`). Called once, when the layer is built, so
    changing the switch later does not move the layer.

    Only the row count is left to each launch, never whether the stream is
    capturing or CU-masked, so a pass of a given size runs the same kernel
    eager or replayed, on any stream. On MI355X the cutoff (5120 rows; 2560
    with `cu_masked`) is above every default decode CUDA graph (512 sequences
    x 4 tokens with MTP 3), and without `cu_masked` above the prefill token
    graphs (3072 tokens), so those replays all run Triton; a long eager
    prefill runs AITER, which the Triton kernel matches bit for bit."""
    # The kernels score the shared columns only given a scoring func.
    if fused_shared_experts_scoring_func is None:
        num_shared, scoring_func = 0, ""
    else:
        num_shared = num_fused_shared_experts
        scoring_func = fused_shared_experts_scoring_func
    if not (
        envs.ATOM_TRITON_TOPK_SOFTMAX
        and triton_topk_softmax_supported(
            num_experts, topk, renormalize, num_shared, scoring_func
        )
    ):
        return None
    return triton_topk_softmax_max_rows(cu_masked)
