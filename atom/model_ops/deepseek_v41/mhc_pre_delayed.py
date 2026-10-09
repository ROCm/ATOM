# SPDX-License-Identifier: MIT
"""The delayed mHC seam, one AITER kernel pair per sublayer boundary.

V4's single `aiter.mhc_pre` cannot serve this model: the collapse applies the
pre-mix carried from the PREVIOUS sublayer while the one computed here is
carried to the next. `mhc_fused_post_pre_delayed_rmsnorm` is that seam whole --
the owed post, the gate projection, Sinkhorn, the collapse with the carried
pre-mix and the sublayer's input RMSNorm -- as a split-K main kernel and a
per-token reduce; vLLM's ROCm V4.1 runs the same kernel.

The norm is the sublayer's own: its output is BF16, and a layer whose first
GEMM takes FP8 quantizes it there, as the reference model does.
"""

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
    mhc_fused_post_pre_delayed_rmsnorm,
)


def pre_delayed(
    residual,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    *,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    post_mult,
    norm_eps,
    sublayer_output=None,
    post_mix=None,
    combination=None,
):
    """One sublayer seam: optional post, the delayed pre, the input RMSNorm.

    Returns `(residual, normed_input, next_pre_mix, post_mix, combination)`.
    `residual` is the caller's own tensor when no post was requested.
    """
    outputs = v41_mhc_pre_delayed(
        residual,
        pre_mix,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        post_mult=post_mult,
        norm_eps=norm_eps,
        sublayer_output=sublayer_output,
        post_mix=post_mix,
        combination=combination,
    )
    # Preserve the no-post input alias outside the custom op. The guarded
    # implementation returns four new tensors, or five when post updates it.
    return (residual, *outputs) if sublayer_output is None else tuple(outputs)


def _pre_delayed_fake(
    residual,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    *,
    sublayer_output=None,
    **kwargs,
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
    norm_weight: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult: float,
    norm_eps: float,
    sublayer_output: torch.Tensor | None = None,
    post_mix: torch.Tensor | None = None,
    combination: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The seam as one guarded op, so Dynamo treats the launches as opaque."""
    *outer, hc_mult, hidden = residual.shape
    rows = residual.numel() // (hc_mult * hidden)
    with_post = sublayer_output is not None
    new_residual, post, comb, normed, pre = mhc_fused_post_pre_delayed_rmsnorm(
        residual.view(rows, hc_mult, hidden),
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        post_mult,
        sinkhorn_iters,
        pre_mix.reshape(rows, hc_mult),
        sublayer_out=sublayer_output.reshape(rows, hidden) if with_post else None,
        post_layer_mix=post_mix.reshape(rows, hc_mult) if with_post else None,
        comb_res_mix=(
            combination.reshape(rows, hc_mult, hc_mult) if with_post else None
        ),
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    outputs = [
        normed.view(*outer, hidden),
        pre.view(*outer, hc_mult),
        post.view(*outer, hc_mult),
        comb.view(*outer, hc_mult, hc_mult),
    ]
    return outputs if not with_post else [new_residual.view_as(residual), *outputs]
