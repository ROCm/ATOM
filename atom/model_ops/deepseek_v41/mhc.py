# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Single-Pass mHC math; incoming pre-mix belongs to the previous sublayer."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from aiter import mhc_post

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


def initial_pre_mix(rows_shape, hc_mult: int, device) -> torch.Tensor:
    """The embedding's pre gates, ``rows_shape + (hc_mult,)`` fp32: 1 on the
    first stream, 0 on the others."""
    pre_mix = torch.zeros((*rows_shape, hc_mult), dtype=torch.float32, device=device)
    pre_mix[..., 0] = 1
    return pre_mix


# Hidden columns a program copies when `from_embeddings` builds the state.
_EMBED_BLOCK_H = 2048


@triton.jit
def _stream_state_kernel(
    anchor_ptr,  # [rows / width, H]: the embedding of every width-th row
    fill_ptr,  # [H]: the embedding of every other row
    residual_ptr,  # [rows, M, H]
    pre_mix_ptr,  # [rows, M] fp32
    hidden_size,
    width,
    M: tl.constexpr,  # hc_mult
    BLOCK_H: tl.constexpr,
):
    # The initial mHC state in one pass: each row's embedding copied to its M
    # streams, and a pre-mix of (1, 0, ..., 0) beside it -- the bytes the
    # expand().contiguous() + zeros + fill of `from_embeddings` write. Row r
    # carries anchor r // width when r % width == 0 and the fill row otherwise:
    # width 1 is every row its own embedding, width > 1 the DSpark draft
    # block (the anchor, then the noise token).
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < hidden_size
    src = tl.where(
        row % width == 0, anchor_ptr + (row // width) * hidden_size, fill_ptr
    )
    value = tl.load(src + cols, mask=mask)
    _store_streams(value, row, cols, mask, residual_ptr, pre_mix_ptr, hidden_size, M)


@triton.jit
def _peer_stream_state_kernel(
    ids_ptr,  # [rows / width]: the token of every width-th row
    shards_ptr,  # [ranks] int64: every rank's vocab shard (`PeerEmbedding`)
    fill_ptr,  # [H]: the embedding of every other row, local
    residual_ptr,  # [rows, M, H]
    pre_mix_ptr,  # [rows, M] fp32
    hidden_size,
    width,
    rows_per_rank,
    ranks,
    M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # `_stream_state_kernel` with the anchors read where they lie: an anchor
    # row from its owner rank's shard, zeros for an id no rank owns (the rows
    # the masked lookup + all-reduce of `VocabParallelEmbedding` gives).
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < hidden_size
    if row % width == 0:
        token = tl.load(ids_ptr + row // width).to(tl.int64)
        owner = token // rows_per_rank
        owned = (token >= 0) & (owner < ranks)
        owner = tl.where(owned, owner, 0)
        local = tl.where(owned, token - owner * rows_per_rank, 0)
        shard = tl.load(shards_ptr + owner).to(tl.pointer_type(tl.bfloat16))
        value = tl.load(
            shard + local * hidden_size + cols, mask=mask & owned, other=0.0
        )
    else:
        value = tl.load(fill_ptr + cols, mask=mask)
    _store_streams(value, row, cols, mask, residual_ptr, pre_mix_ptr, hidden_size, M)


@triton.jit
def _store_streams(
    value, row, cols, mask, residual_ptr, pre_mix_ptr, hidden_size, M: tl.constexpr
):
    # Row `row`'s M streams all `value`, and a pre-mix of (1, 0, ..., 0).
    dst = residual_ptr + row * M * hidden_size
    for stream in tl.static_range(M):
        tl.store(dst + stream * hidden_size + cols, value, mask=mask)
    if tl.program_id(1) == 0:
        streams = tl.arange(0, M)
        tl.store(
            pre_mix_ptr + row * M + streams,
            tl.where(streams == 0, 1.0, 0.0).to(tl.float32),
        )


def write_peer_stream_state(residual, pre_mix, peer, ids, fill, *, block_h, width):
    """`write_stream_state` of a draft block whose anchors ``ids`` are read
    from ``peer``'s vocab shards (`PeerEmbedding`) rather than embedded first:
    no lookup launch, no all-reduce. ``fill`` [H] is local."""
    rows, hc_mult, hidden_size = residual.shape
    if not rows:
        return
    _peer_stream_state_kernel[(rows, triton.cdiv(hidden_size, block_h))](
        ids,
        peer.view.addresses,
        fill,
        residual,
        pre_mix,
        hidden_size,
        width,
        peer.rows_per_rank,
        peer.ranks,
        M=hc_mult,
        BLOCK_H=block_h,
        num_warps=4,
    )


def write_stream_state(residual, pre_mix, anchors, *, block_h, width=1, fill=None):
    """Write the initial mHC state into ``residual [rows, M, H]`` and
    ``pre_mix [rows, M]`` (both contiguous) in one launch.

    Row r holds ``anchors[r // width]`` where ``r % width == 0`` and ``fill``
    elsewhere; ``width=1`` (no ``fill``) is `from_embeddings`, ``width > 1``
    the DSpark draft block (`build_block_state`). ``M`` must be a power of
    two (`tl.arange`). ``block_h`` is the caller's hidden tile.
    """
    rows, hc_mult, hidden_size = residual.shape
    if not rows:
        return
    _stream_state_kernel[(rows, triton.cdiv(hidden_size, block_h))](
        anchors,
        anchors if fill is None else fill,
        residual,
        pre_mix,
        hidden_size,
        width,
        M=hc_mult,
        BLOCK_H=block_h,
        num_warps=4,
    )


def embed_state(
    hidden: torch.Tensor, hc_mult: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(residual, pre_mix) of `from_embeddings`, bit for bit, in one launch.

    None where the torch body serves instead: CPU tensors, non-contiguous
    rows, an ``hc_mult`` that is not a power of two up to 16 (the kernel
    spans the streams with one `tl.arange`), and a traced forward
    (`torch.compiler.is_compiling()`), where the compiler fuses the torch
    body itself.
    """
    if torch.compiler.is_compiling() or not (
        hidden.is_cuda and hidden.is_contiguous() and hc_mult in (1, 2, 4, 8, 16)
    ):
        return None
    *outer, hidden_size = hidden.shape
    residual = torch.empty(
        *outer, hc_mult, hidden_size, dtype=hidden.dtype, device=hidden.device
    )
    pre_mix = torch.empty(*outer, hc_mult, dtype=torch.float32, device=hidden.device)
    rows = hidden.numel() // hidden_size if hidden_size else 0
    write_stream_state(
        residual.view(rows, hc_mult, hidden_size),
        pre_mix.view(rows, hc_mult),
        hidden,
        block_h=_EMBED_BLOCK_H,
    )
    return residual, pre_mix


@triton.jit
def _post_collapse_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    MEAN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # hc_mult = 4. Each output stream is aiter.mhc_post's expression as its
    # compiler contracts it, fma(x, post[j], r0 * comb[0][j]) and then an FMA
    # per further input stream, rounded to BF16; the streams
    # then reduce as collapse_streams (FP32 pre[j] * y[j] summed in order) or,
    # with MEAN, as torch's BF16 mean ((y0 + y1) + y2) + y3 times 1/4.
    token = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size
    x = tl.load(x_ptr + token * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    base = res_ptr + token * 4 * hidden_size + offsets
    r0 = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(base + hidden_size, mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(base + 2 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    r3 = tl.load(base + 3 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for j in tl.static_range(0, 4):
        v = tl.fma(
            x,
            tl.load(post_ptr + token * 4 + j),
            r0 * tl.load(comb_ptr + token * 16 + j),
        )
        v = tl.fma(r1, tl.load(comb_ptr + token * 16 + 4 + j), v)
        v = tl.fma(r2, tl.load(comb_ptr + token * 16 + 8 + j), v)
        v = tl.fma(r3, tl.load(comb_ptr + token * 16 + 12 + j), v)
        y = v.to(tl.bfloat16).to(tl.float32)
        if MEAN:
            acc = acc + y
        else:
            acc = acc + tl.load(pre_ptr + token * 4 + j) * y
    if MEAN:
        acc = acc * 0.25
    tl.store(
        out_ptr + token * hidden_size + offsets,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def post_collapse_supported(residual, pending):
    """Shapes where settle() uses aiter.mhc_post, which this fusion matches."""
    return (
        residual.is_cuda
        and residual.shape[-2] == 4
        and residual.shape[-1] % 256 == 0
        and residual.is_contiguous()
        and pending.is_contiguous()
    )


def post_collapse_streams(pending, residual, post_mix, combination, pre_mix=None):
    """`settle()` then `collapse_streams` (or, with pre_mix=None, the BF16
    mean over the streams) in one launch, without writing the settled
    residual. Bit-exact with mhc_post followed by either reduction."""
    hidden_size = residual.shape[-1]
    flat = residual.view(-1, 4, hidden_size)
    num_tokens = flat.shape[0]
    out = torch.empty(
        num_tokens, hidden_size, dtype=residual.dtype, device=residual.device
    )
    if num_tokens == 0:
        return out
    post = post_mix.reshape(num_tokens, 4).contiguous()
    comb = combination.reshape(num_tokens, 16).contiguous()
    pre = post if pre_mix is None else pre_mix.reshape(num_tokens, 4).contiguous()
    block_h = 1024
    _post_collapse_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pending.view(num_tokens, hidden_size),
        flat,
        post,
        comb,
        pre,
        out,
        hidden_size,
        MEAN=pre_mix is None,
        BLOCK_H=block_h,
        num_warps=4,
        # Explicit FMAs only, as mhc_post and collapse_streams round them.
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
        """Every stream the embedding, the first stream's pre gate 1: in one
        launch (``embed_state``) where it serves, else in torch."""
        fused = embed_state(hidden, hc_mult)
        if fused is not None:
            return cls(*fused)
        residual = (
            hidden.unsqueeze(-2)
            .expand(*hidden.shape[:-1], hc_mult, hidden.shape[-1])
            .contiguous()
        )
        return cls(residual, initial_pre_mix(hidden.shape[:-1], hc_mult, hidden.device))

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
        torch one stays for callers off the GPU. An owed post fuses into the
        same launch.
        """
        if self.pending is not None and post_collapse_supported(
            self.residual, self.pending
        ):
            *outer, _, hidden = self.residual.shape
            return post_collapse_streams(
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
