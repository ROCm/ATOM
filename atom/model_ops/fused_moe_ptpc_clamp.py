# SPDX-License-Identifier: MIT
"""Routed experts with FP8 per-channel weights and a SwiGLU clamp.

aiter's per-token FP8 MoE (the one-stage asm ``fmoe_*_pertokenFp8``
kernels, which it always picks when the intermediate is not a multiple of 128)
fuses the activation and ignores ``swiglu_limit``. Models whose experts were
trained with a clamp (DeepSeek-V4/V4.1: gate <= limit, |up| <= limit) need it:
on V4.1 about 0.1% of the pre-activations exceed 10, some by 20x.

This is the same arithmetic split in three launches so the clamp has a place:

1. GEMM1: per-token FP8 activations x per-channel FP8 ``w13`` -> bf16
   ``[gate | up]`` rows, one per (token, top-k slot).
2. clamp, SiLU(gate) * up, rounded to bf16, then per-row FP8 for GEMM2.
3. GEMM2: those rows x per-channel FP8 ``w2``, routing-weighted, summed over
   the top-k slots.

Both GEMMs are grouped by expert over ``aiter.moe_sorting``'s blocks (a sorted
id carries the token in its low 24 bits and the top-k slot in its high 8), with
the scales applied in the epilogue, so the K loop is plain FP8 MFMA. Weights stay
row-major ``[E, N, K]`` (not the asm kernels' (16, 16) shuffle).
"""

import functools

import torch
import triton
import triton.language as tl
from aiter import dtypes
from aiter import fused_moe as _aiter_fused_moe
from aiter.fused_moe import moe_sorting

# FlyDSL's sort is ~1.5-2x faster than the default (opus) one at decode widths
# (M <= 16: 15 vs 22-30 us); from 32 tokens the default multiphase
# sort is faster. Same output contract (sorted ids, weights, block experts,
# valid count).
_flydsl_moe_sorting = getattr(_aiter_fused_moe, "_flydsl_moe_sorting", None)
_FLYDSL_SORT_MAX_M = 16


def _sort(topk_ids, topk_weights, num_experts, model_dim, dtype, block_m):
    if _flydsl_moe_sorting is not None and topk_ids.shape[0] <= _FLYDSL_SORT_MAX_M:
        return _flydsl_moe_sorting(
            topk_ids, topk_weights, num_experts, model_dim, dtype, block_m, None, None
        )
    return moe_sorting(topk_ids, topk_weights, num_experts, model_dim, dtype, block_m)


FP8 = dtypes.fp8
_FP8_MAX = float(torch.finfo(FP8).max)
_TL_FP8 = tl.float8e4b8 if FP8 == torch.float8_e4m3fnuz else tl.float8e4nv


@triton.jit
def _moe_gemm_ptpc_kernel(
    A,
    A_scale,
    B,
    B_scale,
    C,
    sorted_ids,
    expert_ids,
    num_valid,
    sorted_w,
    M_tok,
    N,
    K,
    TOPK: tl.constexpr,
    A_BY_TOKEN: tl.constexpr,
    MUL_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_B: tl.constexpr = ".cg",
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_m * BLOCK_M >= tl.load(num_valid):
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    sid = tl.load(sorted_ids + offs_m)
    tok = sid & 0xFFFFFF
    slot = sid >> 24
    valid = tok < M_tok
    flat = tok * TOPK + slot
    a_row = tok if A_BY_TOKEN else flat
    e = tl.load(expert_ids + pid_m).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptr = A + a_row[:, None].to(tl.int64) * K + offs_k[None, :]
    b_ptr = B + e * N * K + offs_n[None, :].to(tl.int64) * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_ok = (k + offs_k) < K
        a = tl.load(a_ptr, mask=valid[:, None] & k_ok[None, :], other=0.0)
        b = tl.load(
            b_ptr, mask=k_ok[:, None] & n_ok[None, :], other=0.0, cache_modifier=CACHE_B
        )
        acc = tl.dot(a, b, acc)
        a_ptr += BLOCK_K
        b_ptr += BLOCK_K
    sa = tl.load(A_scale + a_row, mask=valid, other=0.0)
    sb = tl.load(B_scale + e * N + offs_n, mask=n_ok, other=0.0)
    acc = acc * sa[:, None] * sb[None, :]
    if MUL_W:
        w = tl.load(sorted_w + offs_m, mask=valid, other=0.0)
        acc = acc * w[:, None]
    c_ptr = C + flat[:, None].to(tl.int64) * N + offs_n[None, :]
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=valid[:, None] & n_ok[None, :])


@triton.jit
def _clamp_silu_mul_quant_kernel(
    H, Q, S, I, LIMIT, FP8MAX: tl.constexpr, OUT_T: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < I
    g = tl.load(H + row * 2 * I + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(H + row * 2 * I + I + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.minimum(g, LIMIT)
    u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    y = (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-10) / FP8MAX
    tl.store(Q + row * I + offs, (y / s).to(OUT_T), mask=m)
    tl.store(S + row, s)


@triton.jit
def _quant_rows_kernel(
    X, Q, S, K, FP8MAX: tl.constexpr, OUT_T: tl.constexpr, BLOCK: tl.constexpr,
    IDS=None, WTS=None, IDS_OUT=None, WTS_OUT=None, TOPK: tl.constexpr = 0,
    SHARED_SLOT: tl.constexpr = -1, NUM_VALID=None,
):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < K
    x = tl.load(X + row * K + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-10) / FP8MAX
    tl.store(Q + row * K + offs, (x / s).to(OUT_T), mask=m)
    tl.store(S + row, s)
    if SHARED_SLOT >= 0:
        # The row's top-k plus the folded shared expert at weight 1 (the routed
        # weights already carry the routed scaling factor). Rows past the pass's
        # scheduled count route the shared slot to -1 too, as routing does
        # theirs: no expert runs on them.
        KP: tl.constexpr = triton.next_power_of_2(TOPK + 1)
        k = tl.arange(0, KP)
        routed = k < TOPK
        shared = SHARED_SLOT
        if NUM_VALID is not None:
            shared = tl.where(row < tl.load(NUM_VALID), SHARED_SLOT, -1)
        ids = tl.load(IDS + row * TOPK + k, mask=routed, other=0)
        ids = tl.where(routed, ids, shared)
        wts = tl.load(WTS + row * TOPK + k, mask=routed, other=1.0)
        tl.store(IDS_OUT + row * (TOPK + 1) + k, ids, mask=k <= TOPK)
        tl.store(WTS_OUT + row * (TOPK + 1) + k, wts, mask=k <= TOPK)


@triton.jit
def _topk_sum_kernel(
    Y, IDS, OUT, D, E, TOPK: tl.constexpr, BLOCK: tl.constexpr
):  # fmt: skip
    # out[m] = sum_k y[m * TOPK + k], in fp32, written in OUT's dtype: the
    # top-k combine and the down-cast in one pass over GEMM2's fp32 rows. A slot
    # routed outside [0, E) (-1 on rows past the scheduled count) is dropped by
    # the sort, so GEMM2 never wrote its row: it adds nothing, and a row with
    # no valid slot comes out 0.
    m = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        e = tl.load(IDS + m * TOPK + k)
        ok = (e >= 0) & (e < E)
        acc += tl.load(Y + (m * TOPK + k) * D + offs, mask=mask & ok, other=0.0)
    tl.store(OUT + m * D + offs, acc.to(OUT.dtype.element_ty), mask=mask)


def _configs(m: int):
    """((BLOCK_M, BLOCK_N, BLOCK_K, warps, stages[, B cache modifier]) for GEMM1,
    same for GEMM2), swept at V4.1-Flash TP4 (E 384, hidden
    5120, inter 576, top-6). BLOCK_M is also the sort's block size, so the two
    must agree. The decode widths are per CUDA-graph capture size, swept with
    measured routing over eight distinct layers a graph (no Infinity Cache
    reuse) and kept on a fixed-seed serving A/B: TPOT -4.4/-3.2/-3.3% at 8/16/32
    requests, even at 4."""
    if m <= 1:
        return (16, 64, 512, 4, 1), (16, 64, 128, 2, 2, "")
    if m <= 2:
        return (16, 64, 256, 4, 2), (16, 64, 128, 4, 2, "")
    if m <= 4:
        return (16, 32, 256, 2, 1), (16, 128, 128, 4, 2)
    if m <= 16:
        return (16, 64, 256, 4, 2), (16, 32, 128, 2, 1, "")
    if m <= 48:
        return (16, 32, 256, 2, 1), (16, 64, 128, 2, 1, "")
    if m <= 64:
        # The 64-row decode graph (shared expert folded, ~105 experts live):
        # 16-row blocks waste less of each expert's rows than 32, GEMMs 611 ->
        # 430 us a layer over the 32-row prefill tiles.
        return (16, 32, 256, 2, 1), (16, 128, 64, 2, 1, "")
    if m <= 512:
        return (32, 128, 128, 4, 2), (32, 128, 64, 4, 2)
    if m <= 2048:
        return (32, 128, 128, 4, 2), (32, 256, 64, 4, 2)
    return (128, 128, 128, 4, 2), (128, 128, 64, 4, 2)


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


@functools.cache
def flydsl_moe_available() -> bool:
    try:
        from aiter.ops.flydsl.fused_moe_gfx942 import (
            run_flydsl_moe_gfx942,  # noqa: F401 (availability probe)
        )
    except ImportError:
        return False
    return True


def flydsl_moe_ptpc_clamp(
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
    """`fused_moe_ptpc_clamp`'s contract on AITER's FlyDSL MoE: w13 / w2
    preshuffled (16, 16), per-channel fp32 scales [E, 2I] / [E, D]. Needs an
    AITER whose FP8 Gate/Up clamps its SiLU path by swiglu_limit and
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
    """x [M, D] bf16; w13 [E, 2I, D] / w2 [E, D, I] FP8 row-major with per-channel
    fp32 scales [E, 2I] / [E, D]; topk [M, topk]. Returns [M, D] in x.dtype.

    ``shared_slot``: index of a shared expert folded into the buffers; every
    row then also runs it at weight 1, as one more top-k slot.
    ``num_valid_rows``: the pass's scheduled row count (1-element GPU tensor);
    rows past it -- CUDA-graph padding, routed to -1 -- run no expert and come
    out 0. Ids outside [0, E) are skipped on every row."""
    M, D = x.shape
    E, N1, _ = w13.shape
    inter = N1 // 2
    topk = topk_ids.shape[1]
    cfg1, cfg2 = _configs(M)
    bm = cfg1[0]
    dev = x.device
    xq = torch.empty(M, D, dtype=FP8, device=dev)
    xs = torch.empty(M, dtype=torch.float32, device=dev)
    if shared_slot is None:
        _quant_rows_kernel[(M,)](
            x, xq, xs, D, _FP8_MAX, _TL_FP8, BLOCK=triton.next_power_of_2(D), num_warps=8
        )  # fmt: skip
    else:
        ids = torch.empty(M, topk + 1, dtype=topk_ids.dtype, device=dev)
        wts = torch.empty(M, topk + 1, dtype=topk_weights.dtype, device=dev)
        _quant_rows_kernel[(M,)](
            x, xq, xs, D, _FP8_MAX, _TL_FP8, BLOCK=triton.next_power_of_2(D),
            IDS=topk_ids.contiguous(), WTS=topk_weights.contiguous(), IDS_OUT=ids,
            WTS_OUT=wts, TOPK=topk, SHARED_SLOT=shared_slot, NUM_VALID=num_valid_rows,
            num_warps=8,
        )  # fmt: skip
        topk_ids, topk_weights, topk = ids, wts, topk + 1
    topk_ids = topk_ids.contiguous()
    sorted_ids, sorted_w, sorted_eids, nvalid, _ = _sort(
        topk_ids, topk_weights, E, D, x.dtype, bm
    )
    h = torch.empty(M * topk, N1, dtype=torch.bfloat16, device=dev)
    # Every valid sorted block holds at least one (token, slot) pair, so at most
    # M * topk of the sort's (worst-case sized) blocks are live. Launching only
    # those keeps a decode step from dispatching thousands of programs that just
    # read num_valid and exit.
    m_blocks = min(sorted_eids.numel(), M * topk)
    _, bn, bk, nw, ns, *cb = cfg1
    _moe_gemm_ptpc_kernel[(m_blocks * triton.cdiv(N1, bn),)](
        xq, xs, w13, w13_scale, h, sorted_ids, sorted_eids, nvalid, sorted_w,
        M, N1, D, TOPK=topk, A_BY_TOKEN=True, MUL_W=False, CACHE_B=cb[0] if cb else ".cg",
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns,
    )  # fmt: skip
    aq = torch.empty(M * topk, inter, dtype=FP8, device=dev)
    as_ = torch.empty(M * topk, dtype=torch.float32, device=dev)
    _clamp_silu_mul_quant_kernel[(M * topk,)](
        h, aq, as_, inter, float(swiglu_limit), _FP8_MAX, _TL_FP8,
        BLOCK=triton.next_power_of_2(inter), num_warps=4,
    )  # fmt: skip
    y = torch.empty(M * topk, D, dtype=torch.float32, device=dev)
    _, bn, bk, nw, ns, *cb = cfg2
    _moe_gemm_ptpc_kernel[(m_blocks * triton.cdiv(D, bn),)](
        aq, as_, w2, w2_scale, y, sorted_ids, sorted_eids, nvalid, sorted_w,
        M, D, inter, TOPK=topk, A_BY_TOKEN=False, MUL_W=True, CACHE_B=cb[0] if cb else ".cg",
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns,
    )  # fmt: skip
    out = torch.empty(M, D, dtype=x.dtype, device=dev)
    _topk_sum_kernel[(M, triton.cdiv(D, 1024))](
        y, topk_ids, out, D, E, TOPK=topk, BLOCK=1024, num_warps=4
    )  # fmt: skip
    return out
