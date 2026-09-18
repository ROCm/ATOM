# SPDX-License-Identifier: MIT
"""V4.1 draft-block math: positions, bidirectional attention, and the KV tail.

The first three helpers are the block's own geometry -- where a draft row sits,
what it may attend to, how a ragged batch reaches the RoPE interface. The
fourth, `fused_draft_kv_tail`, is the epilogue of the *context* KV write: the
target rows the drafter absorbs after every target forward.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import _ceil_pow2_code
from atom.model_ops.sparse_attn_v4 import sparse_attn


@dataclass(frozen=True)
class DraftStep:
    positions: torch.Tensor
    indices: torch.Tensor
    decode: bool = False


def draft_step(context_positions, anchors, width, window):
    """Anchors locate the last processed target token, not the next input ID."""
    positions = anchors[:, None] + torch.arange(
        1, width + 1, device=anchors.device, dtype=anchors.dtype
    )
    valid = (
        (context_positions >= 0)
        & (context_positions <= anchors[:, None])
        & (context_positions > anchors[:, None] - window)
    )
    slots = torch.arange(context_positions.shape[1], device=anchors.device)
    history = torch.where(valid, slots[None], -1)
    draft = context_positions.shape[1] + torch.arange(width, device=anchors.device)
    indices = torch.cat((history, draft[None].expand(anchors.shape[0], -1)), dim=-1)
    return DraftStep(
        positions, indices[:, None].expand(-1, width, -1).int().contiguous()
    )


def rotate_rows(rope, hidden, positions, *, inverse=False):
    """Flatten ragged-request positions onto the existing V4.1 RoPE interface."""
    shape = hidden.shape
    return rope(
        hidden.reshape(1, -1, *shape[2:]), positions.flatten(), inverse=inverse
    ).view(shape)


def draft_attention(query, context_kv, draft_kv, sink, step, scale):
    """All draft rows attend to the whole draft block, plus valid target rows."""
    keys = torch.cat((context_kv, draft_kv), dim=1)
    return sparse_attn(query, keys, sink, step.indices, scale)


_GROUP = 32  # quantize_fp8's group, fixed by the V4.1 QAT the cache stores


@triton.jit
def _fused_draft_kv_tail_kernel(
    kv_ptr,  # [W, L, D] fused-GEMM output, rows contiguous
    norm_weight_ptr,  # [L, D] the stages' kv_norm weights, stacked
    positions_ptr,  # [W]
    cos_ptr,  # [max_position, PE_DIM // 2]
    sin_ptr,  # [max_position, PE_DIM // 2]
    out_ptr,  # [L, W, D] fp8 codes, or QAT activations when DEQUANT
    scale_ptr,  # [L, W, D // 32] ue8m0 exponents; unused when DEQUANT
    width,  # W, a runtime value: it changes every step
    max_position,
    stride_kv_w,
    stride_kv_l,
    stride_cos_p,
    eps,
    D: tl.constexpr,
    PE_DIM: tl.constexpr,
    GROUPS: tl.constexpr,
    DEQUANT: tl.constexpr,
):
    # One program per output row. The grid is flat over (stage, token) in that
    # order, so `row` IS the stage-major output row: the transpose the cat used
    # to do is this indexing, and the store below needs no stride.
    row = tl.program_id(0)
    stage = row // width
    token = row % width

    offs = tl.arange(0, D)
    src = kv_ptr + token.to(tl.int64) * stride_kv_w + stage.to(tl.int64) * stride_kv_l
    stage_weight = norm_weight_ptr + stage.to(tl.int64) * D

    x = tl.load(src + offs).to(tl.float32)
    w = tl.load(stage_weight + offs).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    y = x * rstd * w

    # RoPE over the tail lanes only, GPT-J interleaved: (2i, 2i+1) share
    # frequency i and lane 2i takes its partner negated. `pe_lane` is negative
    # outside the tail and Triton's `%` keeps the sign of the DIVIDEND, so it is
    # `is_pe` -- not the parity -- that decides whether a lane rotates.
    pe_lane = offs - (D - PE_DIM)
    is_pe = pe_lane >= 0
    even = pe_lane % 2 == 0
    pair = tl.where(even, offs + 1, offs - 1)
    # A masked lane still forms its address, so clamp rather than trust the mask.
    pair = tl.where(is_pe, pair, offs)
    freq = tl.where(is_pe, pe_lane // 2, 0)

    # The partner is the *normed* value, so unlike the MLA context kernel this
    # cannot re-read it from memory. Recomputing it from the same x, w and rstd
    # is exact (identical fp32 expression) and keeps the kernel free of
    # cross-lane ops; under the mask it is 64 lanes off a line already resident.
    x_pair = tl.load(src + pair, mask=is_pe, other=0.0).to(tl.float32)
    w_pair = tl.load(stage_weight + pair, mask=is_pe, other=0.0).to(tl.float32)
    y_pair = x_pair * rstd * w_pair

    pos = tl.load(positions_ptr + token).to(tl.int64)
    pos = tl.minimum(tl.maximum(pos, 0), max_position - 1)
    cos = tl.load(cos_ptr + pos * stride_cos_p + freq).to(tl.float32)
    sin = tl.load(sin_ptr + pos * stride_cos_p + freq).to(tl.float32)
    rot = tl.where(even, -y_pair, y_pair)
    y = tl.where(is_pe, y * cos + rot * sin, y)

    # quantize_fp8's group-32 ue8m0 ceil scale, sharing its exponent helper so
    # the two cannot drift. The reshape is free: `offs` already runs group-major.
    groups = tl.reshape(y, (GROUPS, 32))
    amax = tl.maximum(tl.max(tl.abs(groups), 1), 1e-4)
    code = _ceil_pow2_code(amax * (1.0 / 448.0))
    scale = (code << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(groups / scale[:, None], -448.0), 448.0).to(tl.float8e4nv)

    group_offs = tl.arange(0, GROUPS)
    dst = out_ptr + row.to(tl.int64) * D + group_offs[:, None] * 32 + tl.arange(0, 32)
    if DEQUANT:
        tl.store(dst, (q.to(tl.float32) * scale[:, None]).to(out_ptr.dtype.element_ty))
    else:
        tl.store(dst, q)
        tl.store(scale_ptr + row.to(tl.int64) * GROUPS + group_offs, code.to(tl.uint8))


def fused_draft_kv_tail(
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    eps: float,
    *,
    packed: bool,
):
    """Per-stage RMSNorm + RoPE + FP8 quantize, emitted stage-major, in one launch.

    The epilogue of `write_context_kv`. Once the stages' `wkv` weights are one
    GEMM, its tail is six kernels -- three RMSNorms, a cat, a RoPE, a quantize --
    each re-reading the D-wide row from HBM only to hand it to the next. At 4.5us
    of launch floor apiece that is ~39us of a 163us drafting step doing no
    arithmetic worth the traffic. The cat does not survive as a kernel at all:
    one program owns one (stage, token) row and writes it straight to its
    stage-major slot, which is the layout both window writers need since they
    address rows by width.

    Numerics are not the op chain's and cannot be from Triton -- the fp32
    reduction tree and `rsqrt` differ from the HIP kernel's -- and this path
    also rounds less, carrying fp32 where the chain lands in bf16 three times.
    So ~3.4% of E4M3 codes move, essentially all of them toward an fp64
    reference: over 7.3M elements, 7 come out nearer under the chain and all 7
    straddle a code boundary. Per quantization group the fused value is never
    the further one, which is what `tests/model_ops/test_fused_dspark_ctx_kv.py`
    asserts.

    aiter has nothing to reuse here: its fused RMSNorm+quant kernels emit a
    per-token or block scale rather than the group-32 ue8m0 ceil scale V4.1's
    QAT is defined against, and none of them carry a rotation.

    Args:
        kv: ``[..., W, L, D]`` fused-``wkv`` output; leading dims flatten into
            ``W``. Rows must be contiguous -- a stage's D lanes are what one
            program reads.
        norm_weight: ``[L, D]`` the stages' ``kv_norm.weight``, stacked.
        positions: ``[W]`` absolute positions indexing ``cos_cache``; every
            stage of a token rotates by the same one, which is what
            ``RotaryEmbedding._rotate_cuda``'s ``positions.repeat(batch)``
            spells out for the per-op path.
        cos_cache, sin_cache: ``RotaryEmbedding``'s own buffers, so YaRN scaling
            and cache dtype come along unchanged instead of being recomputed.
        eps: the stages' shared ``kv_norm.eps``.
        packed: the cache layout. ``True`` returns ``(fp8 codes, ue8m0 scales)``
            for ``write_packed_window``; ``False`` returns QAT activations in
            ``kv.dtype``. This is ``quantize_fp8(..., dequantize=not packed)``.

    Returns:
        Stage-major ``[L, W, D]`` (and ``[L, W, D // 32]`` scales when packed),
        so that each ``keys[i : i + 1]`` is contiguous.
    """
    stages, dim = norm_weight.shape
    kv = kv.reshape(-1, stages, dim)
    tokens = kv.shape[0]
    pe_dim = cos_cache.shape[-1] * 2
    if kv.stride(-1) != 1 or norm_weight.stride(-1) != 1:
        raise ValueError("Fused draft KV needs contiguous rows")
    # One position per token, checked rather than masked: a short `positions`
    # is a caller contract violation, and the kernel indexes it by token with
    # no bound of its own -- so the alternative to this line is a silent
    # out-of-bounds read that faults only once the shapes line up to make it.
    if positions.numel() < tokens:
        raise ValueError(
            f"Fused draft KV needs {tokens} positions, got {positions.numel()}"
        )
    if dim % _GROUP or dim & (dim - 1) or pe_dim > dim or pe_dim % 2:
        raise ValueError(f"Fused draft KV needs a power-of-two {dim} >= {pe_dim}")

    values = torch.empty(
        (stages, tokens, dim),
        device=kv.device,
        dtype=torch.float8_e4m3fn if packed else kv.dtype,
    )
    scales = (
        torch.empty(
            (stages, tokens, dim // _GROUP),
            device=kv.device,
            dtype=torch.float8_e8m0fnu,
        )
        if packed
        else None
    )
    if tokens:
        _fused_draft_kv_tail_kernel[(stages * tokens,)](
            kv,
            norm_weight,
            positions,
            cos_cache,
            sin_cache,
            values,
            None if scales is None else scales.view(torch.uint8),
            tokens,
            cos_cache.shape[0],
            kv.stride(0),
            kv.stride(1),
            cos_cache.stride(0),
            eps,
            D=dim,
            PE_DIM=pe_dim,
            GROUPS=dim // _GROUP,
            DEQUANT=not packed,
            # 4 waves over a 512-wide row leave each lane a handful of elements,
            # which is where the fp32 reduction stops dominating.
            num_warps=4,
            num_stages=2,
        )
    return (values, scales) if packed else values
