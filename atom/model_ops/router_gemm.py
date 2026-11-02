# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which kernel runs an unquantized bf16 MoE router GEMM at decode M.

Qwen3-Next / Qwen3.5 merge the shared_expert_gate row into the router weight,
so the router is ``[num_experts + 1, hidden]`` -- 513 x 4096 on Qwen3.5-397B.
aiter's tuned bf16 table has no N=513 row, and the FlyDSL hgemm rejects an
odd N, so ``tgemm.mm`` hands the shape to hipBLASLt. At decode M
that is a split-K GEMM plus a separate reduce kernel: 6.2 + 4.5 us per layer at
M=4 in a same-host trace, against 5.0 us for the single FlyDSL kernel SGLang
runs on its 512-row router (SGLang computes the shared-expert gate elsewhere).

``MergedReplicatedRouterLinear`` zero-pads the weight's rows to a multiple of
``ROUTER_ROW_ALIGN`` once after loading and, for the M this table covers, runs
the padded GEMM on a FlyDSL hgemm kernel aiter already ships; any larger M
(prefill) goes through ``tgemm.mm`` on the padded weight. The table is keyed by
device and padded shape, so a model or device it was not measured on keeps the
unpadded ``tgemm.mm`` path untouched.

Pure Python (no AITER import at module scope) so the CPU CI can check the
table.
"""

import functools
import logging

logger = logging.getLogger("atom")

ROUTER_ROW_ALIGN = 16

# (gfx, cu_num, padded N, K) -> ((largest M, FlyDSL hgemm kernel name), ...)
# ascending in M. A batch of M rows runs the first kernel whose largest M is
# >= M; a batch above the last entry falls back to tgemm.mm.
#
# Kernel names are aiter's own tuned-CSV names (parsed by
# aiter.ops.flydsl.gemm_kernels.get_flydsl_hgemm_kernel_params); all five
# appear in aiter's shipped bf16 tuned tables for other models. Picked on
# MI355X from those tables plus a local grid around the winners, timing the
# padded Qwen3.5-397B router (bf16 x[M, 4096] @ w[528, 4096]^T) by CUDA-graph
# replay over 60 distinct weights, median of 100 replays. Per call, us (the
# tgemm.mm -> hipBLASLt path on the 513-row weight in brackets):
#   M=4 4.25 [8.22]  M=16 4.49 [8.28]  M=32 4.54 [8.20]  M=64 5.10 [7.23]
#   M=128 5.77 [9.27]  M=256 7.90 [10.35]
# Above M=256 hipBLASLt wins again (M=1024: FlyDSL 20.0 vs 15.5).
# Every kernel here is split-K: partial sums meet in bf16 atomics, so the
# bf16 result carries a few roundings (not one) and its low bits vary run to
# run, as with the split-K kernels aiter's own Qwen3.5 table uses for N=512.
_FLYDSL_ROUTER_KERNELS: dict[tuple[str, int, int, int], tuple[tuple[int, str], ...]] = {
    ("gfx950", 256, 528, 4096): (
        (
            16,
            "flydsl_hgemm_abf16_wbf16_bf16_t16x32x64x8_ks8_w1x2x1_bias0_ktail0_gm0_pft_gfx950",
        ),
        (
            32,
            "flydsl_hgemm_abf16_wbf16_bf16_t32x32x64x8_ks8_w2x2x1_bias0_ktail0_gm0_pft_gfx950",
        ),
        (
            64,
            "flydsl_hgemm_abf16_wbf16_bf16_t64x32x64x6_ks8_w4x1x1_bias0_ktail0_gm0_pft_gfx950",
        ),
        (
            128,
            "flydsl_hgemm_abf16_wbf16_bf16_t64x64x64x6_ks8_w4x2x1_bias0_ktail0_gm0_pft_gfx950",
        ),
        (
            256,
            "flydsl_hgemm_abf16_wbf16_bf16_t64x64x64x5_ks4_w4x2x1_bias0_ktail0_gm0_pft_gfx950",
        ),
    ),
}


@functools.cache
def router_flydsl_params(kernel_name: str) -> dict | None:
    """aiter's FlyDSL hgemm parameters for ``kernel_name``, or None when this
    aiter cannot run it: the name is not in its catalog, or the FlyDSL module
    itself is missing (``aiter.ops.flydsl.gemm_kernels`` imports ``flydsl`` at
    module scope, so an install without FlyDSL raises on import)."""
    try:
        from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params
    except (ImportError, AttributeError, RuntimeError, OSError) as error:
        logger.warning("AITER FlyDSL hgemm unavailable for the router: %s", error)
        return None
    return get_flydsl_hgemm_kernel_params(kernel_name)


def padded_router_rows(rows: int) -> int:
    """The row count the router weight is padded to (a multiple of 16)."""
    return -(-rows // ROUTER_ROW_ALIGN) * ROUTER_ROW_ALIGN


def router_kernel_table(
    gfx: str, cu_num: int, padded_rows: int, k: int
) -> tuple[tuple[int, str], ...] | None:
    """The M-bucketed kernels for this device and padded shape, if measured."""
    return _FLYDSL_ROUTER_KERNELS.get((gfx, cu_num, padded_rows, k))


def select_router_kernel(table: tuple[tuple[int, str], ...], m: int) -> str | None:
    """The kernel for a batch of ``m`` rows, or None to fall back to tgemm."""
    for largest_m, kernel_name in table:
        if m <= largest_m:
            return kernel_name
    return None


def use_triton_topk_softmax(
    gating_output,
    num_fused_shared_experts: int,
    fused_shared_experts_scoring_func: str,
    num_valid_rows,
    max_tokens: int,
) -> bool:
    """Whether ``atom/model_ops/triton_topk_softmax.py`` routes this batch
    instead of AITER ``topk_softmax``.

    Small-M Qwen3-Next / Qwen3.5 routing only: softmax top-k plus a
    sigmoid-gated fused shared expert, at most ``max_tokens`` rows. AITER's
    kernel takes ~9.5 us here at 512 experts; the Triton router gives the same
    outputs. A batch with ``num_valid_rows`` (EP: graph-padding rows must route
    to -1) stays on AITER, which masks them; the Triton router does not.
    """
    return (
        num_valid_rows is None
        and num_fused_shared_experts > 0
        and fused_shared_experts_scoring_func == "sigmoid"
        and gating_output.dim() == 2
        and gating_output.stride(1) == 1
        and gating_output.shape[0] <= max_tokens
    )
