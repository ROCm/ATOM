# SPDX-License-Identifier: MIT
"""The Triton delayed mHC seam with a column-parallel reduce.

AITER's `mhc_fused_post_pre_delayed_rmsnorm` is two launches: the main kernel
(post-mix, gate projection partials) and a reduce with one program per token,
which sums the partials, runs the gates and Sinkhorn, then RMS-normalizes the
token's H columns in a serial loop. At decode widths that loop is the reduce's
critical path on a few CUs. Here the reduce's grid is (T, H / BLOCK_C): every
program normalizes one column chunk (each re-summing the norm's partial, 160
floats), and the chunk-0 program alone computes the gates. Same arithmetic per
value; on MI308X the reduce drops from ~11.5 to ~8 us at 1-32 tokens.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
    _mhc_fused_post_pre_delayed_rmsnorm_main_kernel,
)
from aiter.ops.triton.utils.mhc_config_utils import (
    get_mhc_fused_post_pre_delayed_rmsnorm_config,
)
from triton.language.extra.hip import libdevice

_REDUCE_BLOCK_C = 512


@triton.jit
def _mhc_seam_reduce_cols_kernel(
    partial_ptr,  # (T, NUM_KSPLIT, 32) fp32: [0:24] projection, [24] sum R'^2, [25] sum x^2
    scale_ptr,  # (3,) fp32
    base_ptr,  # (24,) fp32
    next_pre_ptr,  # (T, 4) fp32
    post_out_ptr,  # (T, 4) fp32
    comb_out_ptr,  # (T, 4, 4) fp32
    layer_input_ptr,  # (T, H) bf16, normalized in place
    norm_w_ptr,  # (H,)
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult,
    norm_eps,
    H: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Grid (T, H // BLOCK_C)."""
    tl.static_assert(H % BLOCK_C == 0, "the norm has no column mask")
    KSPLIT_POW2: tl.constexpr = triton.next_power_of_2(NUM_KSPLIT)
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    ks = tl.arange(0, KSPLIT_POW2)
    ks_mask = ks < NUM_KSPLIT
    row = (t * NUM_KSPLIT + ks) * 32  # this token's partial rows are contiguous

    sq_x = tl.sum(tl.load(partial_ptr + row + 25, mask=ks_mask, other=0.0), 0)
    rstd_n = tl.math.rsqrt(sq_x * (1.0 / H) + norm_eps)
    rc = c * BLOCK_C + tl.arange(0, BLOCK_C)
    xp = layer_input_ptr + t * H + rc
    xv = tl.load(xp).to(tl.float32)
    w = tl.load(norm_w_ptr + rc).to(tl.float32)
    tl.store(xp, ((xv * rstd_n) * w).to(tl.bfloat16))

    if c == 0:
        r4 = tl.arange(0, 4)
        r16 = tl.arange(0, 16)
        pre_p = tl.sum(
            tl.load(
                partial_ptr + row[:, None] + r4[None, :],
                mask=ks_mask[:, None],
                other=0.0,
            ),
            0,
        )
        post_p = tl.sum(
            tl.load(
                partial_ptr + row[:, None] + 4 + r4[None, :],
                mask=ks_mask[:, None],
                other=0.0,
            ),
            0,
        )
        comb_p = tl.sum(
            tl.load(
                partial_ptr + row[:, None] + 8 + r16[None, :],
                mask=ks_mask[:, None],
                other=0.0,
            ),
            0,
        )
        sq_r = tl.sum(tl.load(partial_ptr + row + 24, mask=ks_mask, other=0.0), 0)
        rstd = tl.math.rsqrt(sq_r * (1.0 / (4 * H)) + rms_eps)
        s0 = tl.load(scale_ptr)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        pre = tl.sigmoid(pre_p * rstd * s0 + tl.load(base_ptr + r4)) + hc_pre_eps
        tl.store(next_pre_ptr + t * 4 + r4, pre)
        post = (
            tl.sigmoid(post_p * rstd * s1 + tl.load(base_ptr + 4 + r4)) * hc_post_mult
        )
        tl.store(post_out_ptr + t * 4 + r4, post)
        # Sinkhorn exactly as AITER's reduce: row softmax, + eps, / (colsum + eps),
        # then NUM_SINKHORN_ITERS - 1 rounds of / (rowsum + eps), / (colsum + eps).
        A = tl.reshape(comb_p * rstd * s2 + tl.load(base_ptr + 8 + r16), (4, 4))
        P = tl.exp(A - tl.max(A, axis=1)[:, None])
        P = libdevice.fast_dividef(P, tl.sum(P, axis=1)[:, None]) + hc_sinkhorn_eps
        P = libdevice.fast_dividef(P, tl.sum(P, axis=0)[None, :] + hc_sinkhorn_eps)
        for _ in range(NUM_SINKHORN_ITERS - 1):
            P = libdevice.fast_dividef(P, tl.sum(P, axis=1)[:, None] + hc_sinkhorn_eps)
            P = libdevice.fast_dividef(P, tl.sum(P, axis=0)[None, :] + hc_sinkhorn_eps)
        tl.store(comb_out_ptr + t * 16 + r16, tl.reshape(P, (16,)))


def mhc_seam_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor,
    sublayer_out: torch.Tensor | None = None,
    post_layer_mix: torch.Tensor | None = None,
    comb_res_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
):
    """`mhc_fused_post_pre_delayed_rmsnorm`'s contract and main kernel, with the
    column-parallel reduce. Returns (residual_out, post (T, 4, 1), comb (T, 4, 4),
    layer_input (T, H) bf16, next_pre (T, 4))."""
    T, n, H = residual.shape
    assert n == 4 and residual.dtype == torch.bfloat16 and residual.is_contiguous()
    assert fn.shape == (24, 4 * H) and fn.dtype == torch.float32 and fn.is_contiguous()
    assert norm_weight is not None and norm_weight.shape == (H,)
    assert H % _REDUCE_BLOCK_C == 0
    device = residual.device
    pre_2d = pre_mix.reshape(T, 4).contiguous()
    norm_weight = norm_weight.contiguous()
    has_post = sublayer_out is not None
    if has_post:
        sublayer_out = sublayer_out.contiguous()
        post_2d = post_layer_mix.reshape(T, 4).contiguous()
        comb_3d = comb_res_mix.reshape(T, 4, 4).contiguous()
        residual_out = torch.empty_like(residual)
    else:
        residual_out = residual
        sublayer_out = post_2d = comb_3d = residual  # unused

    post_out = torch.empty(T, 4, 1, dtype=torch.float32, device=device)
    comb_out = torch.empty(T, 4, 4, dtype=torch.float32, device=device)
    next_pre = torch.empty(T, 4, dtype=torch.float32, device=device)
    layer_input = torch.empty(T, H, dtype=torch.bfloat16, device=device)
    if T == 0:
        return residual_out, post_out, comb_out, layer_input, next_pre

    config = get_mhc_fused_post_pre_delayed_rmsnorm_config(T)
    partial = torch.empty(
        T, config["NUM_KSPLIT"], 32, dtype=torch.float32, device=device
    )
    _mhc_fused_post_pre_delayed_rmsnorm_main_kernel[
        (triton.cdiv(T, config["BLOCK_M"]), config["NUM_KSPLIT"])
    ](
        residual,
        sublayer_out,
        post_2d,
        comb_3d,
        pre_2d,
        fn,
        residual_out,
        layer_input,
        partial,
        T,
        H=H,
        BLOCK_M=config["BLOCK_M"],
        TILE_K=config["TILE_K"],
        NUM_KSPLIT=config["NUM_KSPLIT"],
        HAS_POST=has_post,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    _mhc_seam_reduce_cols_kernel[(T, H // _REDUCE_BLOCK_C)](
        partial,
        hc_scale,
        hc_base,
        next_pre,
        post_out,
        comb_out,
        layer_input,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        norm_eps,
        H=H,
        NUM_KSPLIT=config["NUM_KSPLIT"],
        NUM_SINKHORN_ITERS=int(sinkhorn_repeat),
        BLOCK_C=_REDUCE_BLOCK_C,
        num_warps=1,
    )
    return residual_out, post_out, comb_out, layer_input, next_pre
