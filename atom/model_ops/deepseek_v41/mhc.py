# SPDX-License-Identifier: MIT
"""Single-Pass mHC math; incoming pre-mix belongs to the previous sublayer."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from aiter import mhc_post, mhc_post_reduce

from atom.model_ops.deepseek_v41.projections import hc_projection
from atom.model_ops.sparse_attn_v4 import hc_split_sinkhorn


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


@dataclass(frozen=True)
class SinglePassHCState:
    """The residual between two sublayers, and the post step one of them owes.

    A sublayer's post expands its output back over the streams and the next
    sublayer's pre projects the result; AITER does both in one kernel, so the
    post is carried here and folded in there. `settle` pays it for the seams
    that cannot fold -- an Engram layer, and the end of the stack. `residual`
    is the value BEFORE that post, so nothing reads it while `pending` is set.
    """

    residual: torch.Tensor
    pre_mix: torch.Tensor
    pending: torch.Tensor | None = None
    post_mix: torch.Tensor | None = None
    combination: torch.Tensor | None = None

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

    def settle(self):
        """This state with any owed post applied, so `residual` is current."""
        if self.pending is None:
            return self
        return SinglePassHCState(
            expand_residual(
                self.pending, self.residual, self.post_mix, self.combination
            ),
            self.pre_mix,
        )

    def fields(self):
        """The five tensors, in constructor order."""
        return (
            self.residual,
            self.pre_mix,
            self.pending,
            self.post_mix,
            self.combination,
        )

    def collapse(self):
        """BF16 streams weighted by the FP32 pre-mix, settling an owed post.

        The Triton body is the same expression in one launch instead of four,
        and keeps its multiply and sum apart so the two agree to the bit. The
        torch one stays for callers off the GPU. An owed post and the collapse
        run as one aiter launch where it covers the shape.
        """
        if self.pending is not None and post_reduce_supported(
            self.pending, self.residual
        ):
            *outer, _, hidden = self.residual.shape
            return post_reduce_streams(
                self.pending,
                self.residual,
                self.post_mix,
                self.combination,
                self.pre_mix,
            ).view(*outer, hidden)
        residual = self.settle().residual
        if not residual.is_cuda:
            return (
                (residual.float() * self.pre_mix.unsqueeze(-1))
                .sum(-2)
                .to(residual.dtype)
            )
        *outer, hc_mult, hidden = residual.shape
        # `view`, not `reshape`: a residual that stopped being contiguous
        # should raise rather than be copied, because that copy is the launch
        # this saves, paid twice.
        return collapse_streams(
            residual.view(-1, hc_mult, hidden),
            self.pre_mix.view(-1, hc_mult),
        ).view(*outer, hidden)

    def stream_mean(self):
        """The mean over the streams, settling an owed post: what a DSpark
        draft reads at a target layer's input (V4.1's taps and the vLLM
        plugin's aux hidden states). An owed post and the mean run as one
        aiter launch where it covers the shape, without writing the settled
        residual; elsewhere `settle()` and torch's mean."""
        if self.pending is not None and post_reduce_supported(
            self.pending, self.residual
        ):
            *outer, _, hidden = self.residual.shape
            return post_reduce_streams(
                self.pending, self.residual, self.post_mix, self.combination
            ).view(*outer, hidden)
        return self.settle().residual.mean(dim=-2)


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
    one V4 and vLLM's ROCm V4.1 both call here. The test (`mhc_post_covers`) is
    the kernel's own coverage, which it traps on rather than reports; vLLM
    gates the shape the same way. It is also what keeps the small-hidden
    reference tests, which demand bit-exactness against the pinned model, on
    the body below.
    """
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    if mhc_post_covers(residual):
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


def mhc_post_covers(residual) -> bool:
    """The shapes `aiter.mhc_post` serves, which it traps on rather than
    reports: 4 streams, the 16-bit float dtypes its dispatch takes (and rounds
    every stream to), and a hidden size its tiles cover -- a multiple of 256
    holding at least two of the widest tile (1024, 512 or 256 columns) that
    divides it, because the kernel prefetches one tile ahead."""
    hidden = residual.shape[-1]
    tile = next((t for t in (1024, 512, 256) if hidden % t == 0), None)
    return (
        residual.shape[-2] == 4
        and tile is not None
        and hidden >= 2 * tile
        and residual.dtype in (torch.bfloat16, torch.float16)
    )


def post_reduce_supported(pending, residual) -> bool:
    """Where an owed post and the stream reduction after it run as one
    `aiter.mhc_post_reduce`: `mhc_post`'s coverage with hidden a multiple of
    512, both operands contiguous, on the GPU."""
    return (
        residual.is_cuda
        and mhc_post_covers(residual)
        and residual.shape[-1] % 512 == 0
        and residual.is_contiguous()
        and pending.is_contiguous()
    )


def post_reduce_streams(pending, residual, post_mix, combination, pre_mix=None):
    """`settle()` and then `collapse_streams` -- or, with ``pre_mix=None``,
    the mean over the streams -- as one `aiter.mhc_post_reduce`, which never
    writes the settled residual. aiter computes each stream with `mhc_post`'s
    own body and reduces it as those two do (FP32, streams in order, products
    and sums rounded apart), so the result is the two-kernel path's to the
    bit. Callers check `post_reduce_supported` first."""
    hidden = residual.shape[-1]
    flat = residual.view(-1, 4, hidden)
    rows = flat.shape[0]
    out = torch.empty(rows, hidden, dtype=residual.dtype, device=residual.device)
    mhc_post_reduce(
        out,
        pending.view(rows, hidden),
        flat,
        post_mix.reshape(rows, 4, 1).contiguous(),
        combination.reshape(rows, 4, 4).contiguous(),
        None if pre_mix is None else pre_mix.reshape(rows, 4).contiguous(),
    )
    return out


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
