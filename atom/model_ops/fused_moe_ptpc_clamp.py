# SPDX-License-Identifier: MIT
"""Routed experts with FP8 per-channel weights and a SwiGLU clamp.

aiter's per-token FP8 MoE (the one-stage asm ``fmoe_*_pertokenFp8``
kernels, which it always picks when the intermediate is not a multiple of 128)
fuses the activation and ignores ``swiglu_limit``. Models whose experts were
trained with a clamp (DeepSeek-V4/V4.1: gate <= limit, |up| <= limit) need it:
on V4.1 about 0.1% of the pre-activations exceed 10, some by 20x. AITER's
FlyDSL MoE applies it on its SiLU path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _route_rows_kernel(
    IDS, WTS, IDS_OUT, WTS_OUT, E, stride_i, stride_w,
    TOPK: tl.constexpr, OUT_K: tl.constexpr, SHARED_SLOT: tl.constexpr,
    NUM_VALID=None,
):  # fmt: skip
    # One contiguous [M, OUT_K] routing for the FlyDSL backend: the row's top-k,
    # plus the folded shared expert at weight 1 when SHARED_SLOT >= 0. Ids
    # outside [0, E) and every slot of a row past the scheduled count become -1
    # (weight 0), which the kernels drop.
    row = tl.program_id(0).to(tl.int64)
    KP: tl.constexpr = triton.next_power_of_2(OUT_K)
    k = tl.arange(0, KP)
    routed = k < TOPK
    ids = tl.load(IDS + row * stride_i + k, mask=routed, other=SHARED_SLOT)
    wts = tl.load(WTS + row * stride_w + k, mask=routed, other=1.0)
    ok = (ids >= 0) & (ids < E)
    if NUM_VALID is not None:
        ok = ok & (row < tl.load(NUM_VALID))
    tl.store(IDS_OUT + row * OUT_K + k, tl.where(ok, ids, -1), mask=k < OUT_K)
    tl.store(WTS_OUT + row * OUT_K + k, tl.where(ok, wts, 0.0), mask=k < OUT_K)


def _flydsl_config(m: int) -> str:
    """AITER FlyDSL MoE config per token count, measured at V4.1-Flash
    TP4 (E 385 with the folded shared expert, top-7, inter 576, cold weights):
    the direct batch-1 kernels at 2 tokens, 16-row split-K decode tiles to 16,
    the 128-wide ones to 256 (its decode limit), the prefill pipeline above."""
    if m == 2:
        return "16_16_16_False_True:default:0"
    if m <= 16:
        return "16_16_16_False:default:0"
    if m <= 256:
        return "16_128_64_False:default:0"
    return "64_128_128_True:default:0"


def fused_moe_ptpc_clamp(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    swiglu_limit: float,
    shared_slot: int | None = None,
    num_valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """x [M, D] bf16; w13 [E, 2I, D] / w2 [E, D, I] FP8, (16, 16)-preshuffled,
    with per-channel fp32 scales [E, 2I] / [E, D]; topk [M, topk]. Returns
    [M, D] in x.dtype.

    ``shared_slot``: index of a shared expert folded into the buffers; every
    row then also runs it at weight 1, as one more top-k slot.
    ``num_valid_rows``: the pass's scheduled row count (1-element GPU tensor);
    rows past it -- CUDA-graph padding, routed to -1 -- run no expert and come
    out 0. Ids outside [0, E) are skipped on every row.

    Needs an AITER whose FP8 Gate/Up clamps its SiLU path by swiglu_limit and
    offsets expert weights in 64 bits (E * 2I * D past 2^31 at V4.1 TP4)."""
    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.fused_moe_gfx942 import run_flydsl_moe_gfx942

    M = x.shape[0]
    E = w13.shape[0]
    topk = topk_ids.shape[1]
    out_k = topk + (shared_slot is not None)
    ids = torch.empty(M, out_k, dtype=torch.int32, device=x.device)
    wts = torch.empty(M, out_k, dtype=torch.float32, device=x.device)
    _route_rows_kernel[(M,)](
        topk_ids, topk_weights, ids, wts, E, topk_ids.stride(0), topk_weights.stride(0),
        TOPK=topk, OUT_K=out_k, SHARED_SLOT=-1 if shared_slot is None else shared_slot,
        NUM_VALID=num_valid_rows, num_warps=1,
    )  # fmt: skip
    return run_flydsl_moe_gfx942(
        x.contiguous(), w13, w2, wts, ids, ActivationType.Silu, QuantType.per_Token,
        w13_scale, w2_scale, None, None, 0, _flydsl_config(M), swiglu_limit=swiglu_limit,
    )  # fmt: skip
