# SPDX-License-Identifier: MIT
"""Single-Pass mHC math; incoming pre-mix belongs to the previous sublayer."""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from aiter import (
    mhc_post,
    mhc_post_reduce,
    mhc_post_reduce_supported,
    mhc_post_supported,
)

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

    def _reduce_owed(self, pre_mix=None, out=None):
        """An owed post and the stream reduction after it as one
        `aiter.mhc_post_reduce` (the ``pre_mix`` weighted sum, or with
        ``pre_mix=None`` the mean), shaped like one stream, written into
        ``out`` if given (see `post_reduce_streams`). None unless a post is
        owed and `post_reduce_supported` holds; the caller then settles and
        reduces in two steps."""
        if self.pending is None or not post_reduce_supported(
            self.pending, self.residual
        ):
            return None
        *outer, _, hidden = self.residual.shape
        return post_reduce_streams(
            self.pending,
            self.residual,
            self.post_mix,
            self.combination,
            pre_mix,
            out=out,
        ).view(*outer, hidden)

    def collapse(self):
        """BF16 streams weighted by the FP32 pre-mix, settling an owed post.

        The Triton body is the same expression in one launch instead of four,
        and keeps its multiply and sum apart so the two agree to the bit. The
        torch one stays for callers off the GPU. An owed post and the collapse
        run as one aiter launch where `post_reduce_supported` holds.
        """
        fused = self._reduce_owed(self.pre_mix)
        if fused is not None:
            return fused
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

    def stream_mean(self, out=None):
        """The mean over the streams, settling an owed post: what a DSpark
        draft reads at a target layer's input (V4.1's taps and the vLLM
        plugin's aux hidden states). An owed post and the mean run as one
        aiter launch where `post_reduce_supported` holds, without writing the
        settled residual; elsewhere `settle()` and torch's mean.

        ``out``: a contiguous ``[tokens, H]`` destination (a capture buffer's
        rows) that receives the mean on either path; the result is then a view
        of it. Any other ``out`` raises."""
        fused = self._reduce_owed(out=out)
        if fused is not None:
            return fused
        mean = self.settle().residual.mean(dim=-2)
        if out is None:
            return mean
        *outer, _, hidden = self.residual.shape
        _check_out(out, math.prod(outer), self.residual)
        out.copy_(mean.view(out.shape))
        return out.view(*outer, hidden)


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
    one V4 and vLLM's ROCm V4.1 both call here, where aiter says it runs
    (`mhc_post_supported`, asked through `_aiter_serves`). That is also what
    keeps the small-hidden reference tests, which demand bit-exactness against
    the pinned model, on the body below.
    """
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    if _aiter_serves(hidden, hc_mult, residual.dtype, False):
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


@torch.compiler.assume_constant_result
def _aiter_serves(hidden_size: int, hc_mult: int, dtype: torch.dtype, reduce: bool):
    """aiter's own answer to whether `mhc_post` (with ``reduce``,
    `mhc_post_reduce`) runs this shape, so ATOM does not restate its tile
    dispatch. A traced forward takes the answer as a constant: the query
    reads the live GPU arch, which Dynamo must not trace into."""
    query = mhc_post_reduce_supported if reduce else mhc_post_supported
    return query(hidden_size, hc_mult, dtype)


def post_reduce_supported(pending, residual) -> bool:
    """Where an owed post and the stream reduction after it run as one
    `aiter.mhc_post_reduce`: aiter takes the shape and dtype
    (`mhc_post_reduce_supported`), and the operands are on the GPU and
    contiguous."""
    return (
        residual.is_cuda
        and residual.is_contiguous()
        and pending.is_contiguous()
        and _aiter_serves(residual.shape[-1], residual.shape[-2], residual.dtype, True)
    )


def _check_out(out, rows, residual):
    """``out=`` is honoured or refused, never dropped for a new tensor: a
    destination that silently stopped being written would bring back the copy
    it exists to save."""
    hidden = residual.shape[-1]
    if not (
        out.shape == (rows, hidden)
        and out.dtype == residual.dtype
        and out.device == residual.device
        and out.is_contiguous()
    ):
        raise ValueError(
            f"out must be a contiguous [{rows}, {hidden}] {residual.dtype} tensor "
            f"on {residual.device}; got {list(out.shape)} {out.dtype} on "
            f"{out.device} (contiguous={out.is_contiguous()})"
        )


def post_reduce_streams(
    pending, residual, post_mix, combination, pre_mix=None, out=None
):
    """`settle()` and then `collapse_streams` -- or, with ``pre_mix=None``,
    the mean over the streams -- as one `aiter.mhc_post_reduce`, which never
    writes the settled residual. aiter computes each stream with `mhc_post`'s
    own body and reduces it as those two do (FP32, streams in order, products
    and sums rounded apart), so the result is the two-kernel path's to the
    bit. Callers check `post_reduce_supported` first.

    ``out``: the contiguous ``[tokens, H]`` destination to write (e.g. a
    capture buffer's rows), returned as the result; anything else raises."""
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    flat = residual.view(-1, hc_mult, hidden)
    rows = flat.shape[0]
    if out is None:
        out = torch.empty(rows, hidden, dtype=residual.dtype, device=residual.device)
    else:
        _check_out(out, rows, residual)
    mhc_post_reduce(
        out,
        pending.view(rows, hidden),
        flat,
        post_mix.reshape(rows, hc_mult, 1).contiguous(),
        combination.reshape(rows, hc_mult, hc_mult).contiguous(),
        None if pre_mix is None else pre_mix.reshape(rows, hc_mult).contiguous(),
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
