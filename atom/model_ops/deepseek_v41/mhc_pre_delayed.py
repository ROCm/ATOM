# SPDX-License-Identifier: MIT
"""The mHC pre seam, driven as AITER stages instead of ~140 torch launches.

Ported from vLLM's ROCm DeepSeek-V4.1 (`vllm/_aiter_ops.py`,
`model_executor/kernels/mhc/triton.py`), which solves the same problem this
model has: V4's single `aiter.mhc_pre` cannot serve it, because the collapse
applies the pre-mix carried from the PREVIOUS sublayer while the one computed
here is carried to the next. AITER has no kernel for that shape, so its two
stages are driven directly --

    mhc_pre_gemm_sqrsum      project the residual, keeping split-k unreduced
    mhc_pre_big_fuse         post and comb gates, every Sinkhorn iteration

-- and the two pieces AITER does not return are recovered here: the pre gate
from the same GEMM output (`pre_mix_from_projection`), and the delayed collapse
(`collapse_streams`). `mhc_pre_big_fuse` also writes a collapse against its own
pre-mix, which this formulation cannot use; that redundant store is the price
of there being no native delayed kernel.

When the preceding post block is folded in, `mhc_fused_post_pre_gemm_sqrsum`
computes the new residual and projects it in one kernel. AITER's own heuristic
decides when that stops paying.
"""

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.mhc import (
    get_mhc_fused_post_pre_config,
    get_mhc_pre_splitk,
    mhc_fused_post_pre_gemm_sqrsum,
    mhc_pre_big_fuse,
    mhc_pre_gemm_sqrsum,
)


@triton.jit
def _collapse_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for mix_idx in tl.static_range(0, hc_mult):
        pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
            tl.float32
        )
        x = tl.load(
            x_ptr
            + token_idx * x_stride_t
            + mix_idx * x_stride_m
            + offsets * x_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * x

    tl.store(
        out_ptr + token_idx * out_stride_t + offsets * out_stride_h, acc, mask=mask
    )


def collapse_streams(residual, pre_mix):
    """BF16 residual streams weighted by FP32 pre-mix, as `collapse()` does."""
    num_tokens, hc_mult, hidden_size = residual.shape
    out = torch.empty(
        num_tokens, hidden_size, dtype=residual.dtype, device=residual.device
    )
    if num_tokens == 0:
        return out
    block_h = 1024
    _collapse_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pre_mix,
        residual,
        out,
        hidden_size,
        hc_mult,
        pre_mix.stride(0),
        pre_mix.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
        # Keep the multiply and the sum separate, as the torch body has them.
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _pre_mix_kernel(
    gemm_ptr,
    sqrsum_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    out_ptr,
    splitk,
    hc_mult: tl.constexpr,
    gemm_stride_k,
    gemm_stride_t,
    gemm_stride_j,
    sqrsum_stride_k,
    sqrsum_stride_t,
    out_stride_t,
    out_stride_j,
    inv_hc_hidden,
    rms_eps,
    hc_pre_eps,
    SPLITK_BLOCK: tl.constexpr,
    HC_BLOCK: tl.constexpr,
):
    """Recover the pre gate from the split-k projection AITER already ran."""
    token_idx = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, SPLITK_BLOCK)
    js = tl.arange(0, HC_BLOCK)
    kmask = ks < splitk
    jmask = js < hc_mult

    # Only the leading hc_mult columns of the projection feed the pre gate.
    gemm = tl.load(
        gemm_ptr
        + ks[:, None] * gemm_stride_k
        + token_idx * gemm_stride_t
        + js[None, :] * gemm_stride_j,
        mask=kmask[:, None] & jmask[None, :],
        other=0.0,
    ).to(tl.float32)
    mixes = tl.sum(gemm, 0)

    sqrsum = tl.load(
        sqrsum_ptr + ks * sqrsum_stride_k + token_idx * sqrsum_stride_t,
        mask=kmask,
        other=0.0,
    ).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(sqrsum, 0) * inv_hc_hidden + rms_eps)

    scale = tl.load(hc_scale_ptr).to(tl.float32)
    base = tl.load(hc_base_ptr + js, mask=jmask, other=0.0).to(tl.float32)

    pre = tl.sigmoid(mixes * rstd * scale + base) + hc_pre_eps
    tl.store(out_ptr + token_idx * out_stride_t + js * out_stride_j, pre, mask=jmask)


def pre_mix_from_projection(
    gemm_out, sqrsum, hc_scale, hc_base, hc_mult, hc_hidden_size, rms_eps, hc_pre_eps
):
    """`mhc_pre_big_fuse` consumes this projection but returns only post/comb.

    The pre gate is the same slice of the same numbers, so it is recovered from
    the unreduced split-k output rather than by projecting a second time.
    """
    splitk, num_tokens = gemm_out.shape[0], gemm_out.shape[1]
    out = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=gemm_out.device)
    if num_tokens == 0:
        return out
    _pre_mix_kernel[(num_tokens,)](
        gemm_out,
        sqrsum,
        hc_scale,
        hc_base,
        out,
        splitk,
        hc_mult,
        gemm_out.stride(0),
        gemm_out.stride(1),
        gemm_out.stride(2),
        sqrsum.stride(0),
        sqrsum.stride(1),
        out.stride(0),
        out.stride(1),
        1.0 / hc_hidden_size,
        rms_eps,
        hc_pre_eps,
        SPLITK_BLOCK=triton.next_power_of_2(splitk),
        HC_BLOCK=triton.next_power_of_2(hc_mult),
        num_warps=1,
    )
    return out


def prefers_unfused(num_tokens: int) -> bool:
    """AITER's own `fused_m_upper_bound` table: folding the post in stops
    paying once the residual no longer fits in cache."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return num_tokens >= {"gfx950": 1024, "gfx942": 128}.get(get_gfx_runtime(), 128)


def pre_delayed(
    residual,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    post_mult,
    sublayer_output=None,
    post_mix=None,
    combination=None,
):
    """One sublayer seam: optional post, then the delayed pre.

    Returns `(residual, layer_input, next_pre_mix, post_mix, combination)`.
    `residual` is the caller's own tensor when no post was requested.
    """
    outputs = v41_mhc_pre_delayed(
        residual,
        pre_mix,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        post_mult=post_mult,
        sublayer_output=sublayer_output,
        post_mix=post_mix,
        combination=combination,
    )
    # Preserve the no-post input alias outside the custom op. The guarded
    # implementation returns four new tensors, or five when post updates it.
    return (residual, *outputs) if sublayer_output is None else tuple(outputs)


def _pre_delayed_fake(
    residual, pre_mix, hc_fn, hc_scale, hc_base, *, sublayer_output=None, **kwargs
):
    *outer, hc, hidden = residual.shape
    outputs = [
        residual.new_empty((*outer, hidden)),
        residual.new_empty((*outer, hc), dtype=torch.float32),
        residual.new_empty((*outer, hc), dtype=torch.float32),
        residual.new_empty((*outer, hc, hc), dtype=torch.float32),
    ]
    return (
        outputs if sublayer_output is None else [torch.empty_like(residual), *outputs]
    )


@torch_compile_guard(mutates_args=[], gen_fake=_pre_delayed_fake)
def v41_mhc_pre_delayed(
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult: float,
    sublayer_output: torch.Tensor | None = None,
    post_mix: torch.Tensor | None = None,
    combination: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Choose AITER's shape/device tuning at execution time, outside Dynamo."""
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    outer = residual.shape[:-2]
    flat = residual.view(-1, hc_mult, hidden)
    rows = flat.shape[0]
    hc_hidden = hc_mult * hidden

    fold_post = sublayer_output is not None and not prefers_unfused(rows)
    if sublayer_output is not None and not fold_post:
        from atom.model_ops.deepseek_v41.mhc import expand_residual

        residual = expand_residual(sublayer_output, residual, post_mix, combination)
        flat = residual.view(-1, hc_mult, hidden)

    # AITER's wrappers allocate without an explicit device.
    with torch.device(flat.device):
        if fold_post:
            splitk, tile_m, tile_n, tile_k = get_mhc_fused_post_pre_config(rows, hidden)
        else:
            splitk, tile_k = get_mhc_pre_splitk(rows, hc_hidden)
        mixes3 = hc_mult * 2 + hc_mult * hc_mult
        # AITER pads the projection to a multiple of 32 columns.
        padded = torch.empty(
            splitk, rows, (mixes3 + 31) // 32 * 32, dtype=torch.float32
        )
        gemm_out = padded[:, :, :mixes3]
        sqrsum = torch.empty(splitk, rows, dtype=torch.float32)
        if fold_post:
            projected = torch.empty_like(flat)
            mhc_fused_post_pre_gemm_sqrsum(
                gemm_out,
                sqrsum,
                projected,
                sublayer_output.view(rows, hidden),
                flat,
                post_mix.view(rows, hc_mult, 1).squeeze(-1),
                combination.view(rows, hc_mult, hc_mult),
                hc_fn,
                tile_m,
                tile_n,
                tile_k,
                0,
            )
            residual = projected.view(*outer, hc_mult, hidden)
        else:
            projected = flat
            mhc_pre_gemm_sqrsum(gemm_out, sqrsum, projected, hc_fn, tile_k, 0)

        next_post = torch.empty(rows, hc_mult, 1, dtype=torch.float32)
        next_comb = torch.empty(rows, hc_mult, hc_mult, dtype=torch.float32)
        # Written against a pre-mix the delayed formulation cannot use.
        discarded = torch.empty(rows, hidden, dtype=residual.dtype)
        mhc_pre_big_fuse(
            next_post,
            next_comb,
            discarded,
            gemm_out,
            sqrsum,
            hc_scale,
            hc_base,
            projected,
            rms_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
        )

    next_pre = pre_mix_from_projection(
        gemm_out, sqrsum, hc_scale, hc_base, hc_mult, hc_hidden, rms_eps, hc_eps
    )
    layer_input = collapse_streams(projected, pre_mix.reshape(rows, hc_mult))
    outputs = [
        layer_input.view(*outer, hidden),
        next_pre.view(*outer, hc_mult),
        next_post.view(*outer, hc_mult),
        next_comb.view(*outer, hc_mult, hc_mult),
    ]
    return outputs if sublayer_output is None else [residual, *outputs]
