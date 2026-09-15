# SPDX-License-Identifier: MIT
"""Single-Pass mHC math; incoming pre-mix belongs to the previous sublayer."""

from dataclasses import dataclass

import torch
from aiter import mhc_post

from atom.model_ops.deepseek_v41.projections import hc_projection
from atom.model_ops.sparse_attn_v4 import hc_split_sinkhorn


@dataclass(frozen=True)
class SinglePassHCState:
    residual: torch.Tensor
    pre_mix: torch.Tensor

    @classmethod
    def from_embeddings(cls, hidden: torch.Tensor, hc_mult: int):
        residual = (
            hidden.unsqueeze(-2)
            .expand(*hidden.shape[:-1], hc_mult, hidden.shape[-1])
            .contiguous()
        )
        pre_mix = torch.zeros(
            (*hidden.shape[:-1], hc_mult), dtype=torch.float32, device=hidden.device
        )
        pre_mix[..., 0] = 1
        return cls(residual, pre_mix)

    def collapse(self):
        return (
            (self.residual.float() * self.pre_mix.unsqueeze(-1))
            .sum(-2)
            .to(self.residual.dtype)
        )


def predict_mixes(
    residual,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    norm_eps=1e-20,
    sinkhorn_eps=1e-6,
    sinkhorn_iters=20,
):
    """Predict next pre-mix and current post/comb, preserving FP32 hc_fn."""
    if (
        hc_fn.dtype != torch.float32
        or hc_scale.dtype != torch.float32
        or hc_base.dtype != torch.float32
    ):
        raise ValueError("mHC coefficients must retain their FP32 checkpoint dtype")
    flat = residual.flatten(-2).float()
    norm = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = hc_projection(flat, hc_fn) * norm
    return hc_split_sinkhorn(
        mixes, hc_scale, hc_base, residual.shape[-2], sinkhorn_iters, sinkhorn_eps
    )


def expand_residual(sublayer_output, residual, post_mix, combination):
    """combination[..., input_stream, output_stream], with FP32 accumulation.

    The torch body materializes a [tokens, hc, hc, dim] product before reducing
    it. `aiter.mhc_post` is the same expression as one ROCm op, and it is the
    one V4 and vLLM's ROCm V4.1 both call here. The shape test is the kernel's
    own coverage, which it traps on rather than reports; vLLM gates it the same
    way. It is also what keeps the small-hidden reference tests, which demand
    bit-exactness against the pinned model, on the body below.
    """
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    if hidden % 256 == 0 and hc_mult == 4:
        flat = residual.view(-1, hc_mult, hidden)
        rows = flat.shape[0]
        out = torch.empty_like(flat)
        mhc_post(
            out,
            sublayer_output.view(rows, hidden),
            flat,
            post_mix.view(rows, hc_mult, 1),
            combination.view(rows, hc_mult, hc_mult),
        )
        return out.view_as(residual)
    mixed = (combination.unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(-3)
    return (mixed + post_mix.unsqueeze(-1) * sublayer_output.float().unsqueeze(-2)).to(
        sublayer_output.dtype
    )


def apply_sublayer(
    state,
    sublayer,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    norm_eps=1e-20,
    sinkhorn_eps=1e-6,
    sinkhorn_iters=20,
):
    """Run one attention/FFN callable; its pre-mix is used by the next callable.

    The callable owns its input RMSNorm and operation. Engram updates residual
    before this boundary and must retain the incoming state's pre_mix.
    """
    pre, post, comb = predict_mixes(
        state.residual,
        hc_fn,
        hc_scale,
        hc_base,
        norm_eps=norm_eps,
        sinkhorn_eps=sinkhorn_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    output = sublayer(state.collapse())
    return SinglePassHCState(expand_residual(output, state.residual, post, comb), pre)
