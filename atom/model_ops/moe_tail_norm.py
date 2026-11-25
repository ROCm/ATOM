# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MoE block's gated shared-expert add folded into the next Gemma RMSNorm
(ATOM_FUSED_MOE_TAIL_NORM).

Qwen3-Next / Qwen3.5 end every MoE layer with two row kernels:

    moe      = routed + sigmoid(gate_logit) * shared        (sigmoid_gate.py)
    residual = residual + moe;  normed = rmsnorm(residual)  (the next norm)

where "the next norm" is the next layer's input_layernorm, the model's final
norm, or (MTP drafter) the predictor's final norm. Each is a node of its own in
the decode CUDA graph and a launch of its own in eager prefill, and `moe` makes
a round trip through memory between them. Here they are one kernel: the MoE
block hands its three addends to the norm unsummed (`MoEGatedSum`) and the norm
adds them while it loads its input.

Rounding is the unfused ops' step for step: the sigmoid rounds to the logit
dtype, the gated product to the activation dtype, `routed + product` is added in
fp32 and rounded to the activation dtype (what `sigmoid_mul` stores), and only
then does the norm add the residual in fp32 (aiter's fused_qk_rmsnorm_group_quant
adds and normalizes in fp32, storing the rounded sum as the new residual). The
new residual is therefore bitwise the unfused one; the normed output can differ
in the last bit where the fp32 sum of squares is reduced in another order.

`rows` (drafter): normalize only these rows, out[i] = norm(row rows[i]), and
skip the residual store. The MTP head reads one row per sequence, so the
drafter's final norm then covers `bs` rows instead of the whole token stream
(4 x bs at decode, the whole prefill chunk at prefill) and the row gather
before its LM head disappears.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard
from torch import Tensor

from atom.model_ops.mxfp8_1x32 import alloc as mx32_alloc
from atom.model_ops.mxfp8_1x32 import (
    mx32_quant_groups,
    mx32_quant_groups_bits,
    mx32_scale_offset,
)
from atom.model_ops.mxfp8_1x32 import note as mx32_note
from atom.utils import envs


class MoEGatedSum(NamedTuple):
    """A sparse MoE block's output left unsummed: the norm that reads it
    computes `routed + sigmoid(gate_logits) * shared` itself.

    `gate_logits` is `[tokens, 1]` (rows may be strided, e.g. the last column
    of the router GEMM's output). `None` means `shared` is already gated.
    """

    routed: Tensor
    shared: Tensor
    gate_logits: Tensor | None


@triton.jit
def _moe_tail_rmsnorm_kernel(
    shared_ptr,
    gate_ptr,
    routed_ptr,
    residual_ptr,
    weight_ptr,
    rows_ptr,
    out_ptr,
    res_out_ptr,
    stride_shared,
    stride_gate,
    stride_routed,
    stride_residual,
    stride_out,
    stride_res_out,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_GATE: tl.constexpr,
    GATHER: tl.constexpr,
    WRITE_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if GATHER:
        src = tl.load(rows_ptr + row).to(tl.int64)
    else:
        src = row
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    shared = tl.load(shared_ptr + src * stride_shared + cols, mask, other=0.0).to(
        tl.float32
    )
    if HAS_GATE:
        # sigmoid_mul's steps: sigmoid in fp32 rounded to the logit dtype, the
        # product rounded to the activation dtype.
        gate = tl.load(gate_ptr + src * stride_gate)
        gate = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype).to(tl.float32)
        shared = (shared * gate).to(shared_ptr.dtype.element_ty).to(tl.float32)
    routed = tl.load(routed_ptr + src * stride_routed + cols, mask, other=0.0).to(
        tl.float32
    )
    # The MoE block's output as sigmoid_mul stores it, then the norm's fp32 add.
    moe = (shared + routed).to(out_ptr.dtype.element_ty).to(tl.float32)
    x = moe + tl.load(residual_ptr + src * stride_residual + cols, mask, other=0.0).to(
        tl.float32
    )
    if WRITE_RESIDUAL:
        tl.store(
            res_out_ptr + row * stride_res_out + cols,
            x.to(res_out_ptr.dtype.element_ty),
            mask,
        )
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    weight = tl.load(weight_ptr + cols, mask, other=0.0).to(tl.float32) + 1.0
    tl.store(
        out_ptr + row * stride_out + cols,
        ((x * rstd) * weight).to(out_ptr.dtype.element_ty),
        mask,
    )


def can_fuse_moe_tail(
    moe: MoEGatedSum, residual: Tensor, weight: Tensor, rows: Tensor | None = None
) -> bool:
    """Whether the kernel reproduces the unfused ops for these inputs: 2-D rows
    of one 16-bit float dtype, last dim dense, a `[tokens, 1]` gate of the same
    dtype, and a norm weight of the hidden width. Shape / stride / dtype only,
    so a capture's answer holds for every replay of that shape."""
    routed, shared, gate = moe
    dtype = routed.dtype
    if dtype not in (torch.bfloat16, torch.half) or routed.dim() != 2:
        return False
    tokens, hidden = routed.shape
    for t in (shared, residual):
        if t.dtype != dtype or t.shape != routed.shape:
            return False
    if any(t.stride(1) != 1 for t in (routed, shared, residual)):
        return False
    if gate is not None and (gate.dtype != dtype or gate.shape != (tokens, 1)):
        return False
    if weight.shape != (hidden,) or weight.stride(0) != 1:
        return False
    return rows is None or (
        rows.dim() == 1 and rows.dtype in (torch.int32, torch.int64)
    )


# Compiled kernels by launch key (see `_launch`).
_COMPILED: dict[tuple, object] = {}
_INT32_MAX = 2**31 - 1


def _launch(
    routed: Tensor,
    shared: Tensor,
    gate: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    rows: Tensor | None,
    out: Tensor,
    res_out: Tensor | None,
) -> None:
    n_out, hidden = out.shape
    if n_out == 0:
        return
    block = triton.next_power_of_2(hidden)
    ptrs = (
        shared,
        gate if gate is not None else shared,
        routed,
        residual,
        weight,
        rows if rows is not None else routed,
        out,
        res_out if res_out is not None else out,
    )
    strides = (
        shared.stride(0),
        gate.stride(0) if gate is not None else 0,
        routed.stride(0),
        residual.stride(0),
        out.stride(0),
        res_out.stride(0) if res_out is not None else 0,
    )
    # Eager prefill is host-bound, and Triton's own dispatch (argument binding,
    # specialization, cache lookup) is most of this launch's host time. So the
    # compiled kernel is kept and launched directly, under a key at least as
    # fine as what Triton specializes on: dtypes, int args == 1 / % 16, and --
    # only cached when every pointer is 16-byte aligned and reaches no further
    # than 2 GiB from itself (the AMD buffer-op attribute) -- pointer alignment
    # and reach. Anything else, and a new key, takes Triton's own path.
    reach = (max(n_out, routed.shape[0]) * max(strides) + hidden) * out.element_size()
    cacheable = reach <= _INT32_MAX and not any(t.data_ptr() % 16 for t in ptrs)
    key = (
        hidden,
        block,
        routed.dtype,
        None if gate is None else gate.dtype,
        None if rows is None else rows.dtype,
        res_out is not None,
        tuple((v == 1, v % 16 == 0) for v in strides),
    )
    kernel = _COMPILED.get(key) if cacheable else None
    if kernel is not None:
        kernel[(n_out, 1, 1)](*ptrs, *strides, eps)
        return
    kernel = _moe_tail_rmsnorm_kernel[(n_out,)](
        *ptrs,
        *strides,
        eps,
        N=hidden,
        BLOCK=block,
        HAS_GATE=gate is not None,
        GATHER=rows is not None,
        WRITE_RESIDUAL=res_out is not None,
        num_warps=4 if block <= 2048 else 8,
        # The products and sums round where the unfused kernels round; no FMA
        # contraction across them.
        enable_fp_fusion=False,
    )
    if cacheable:
        _COMPILED[key] = kernel


def _moe_tail_rmsnorm_impl(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    out = torch.empty(routed.shape, dtype=routed.dtype, device=routed.device)
    res_out = torch.empty(routed.shape, dtype=routed.dtype, device=routed.device)
    _launch(routed, shared, gate_logits, residual, weight, eps, None, out, res_out)
    return out, res_out


def _moe_tail_rmsnorm_fake(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(routed), torch.empty_like(routed)


# Custom ops, so Dynamo sees one opaque call rather than tracing the launch.
@torch_compile_guard(gen_fake=_moe_tail_rmsnorm_fake, mutates_args=[])
def moe_tail_rmsnorm_op(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    return _moe_tail_rmsnorm_impl(routed, shared, gate_logits, residual, weight, eps)


def moe_tail_rmsnorm(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """`(rmsnorm(r) * (1 + weight), r)` with
    `r = residual + (routed + sigmoid(gate_logits) * shared)`.

    The registered op under torch.compile; eagerly the launch itself (the op
    dispatcher is a quarter of this call's host time, and eager prefill is
    host-bound)."""
    if torch.compiler.is_compiling():
        return moe_tail_rmsnorm_op(routed, shared, gate_logits, residual, weight, eps)
    return _moe_tail_rmsnorm_impl(routed, shared, gate_logits, residual, weight, eps)


def _moe_tail_rmsnorm_rows_impl(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    rows: Tensor | None,
) -> Tensor:
    n = routed.shape[0] if rows is None else rows.shape[0]
    out = torch.empty((n, routed.shape[1]), dtype=routed.dtype, device=routed.device)
    _launch(routed, shared, gate_logits, residual, weight, eps, rows, out, None)
    return out


def _moe_tail_rmsnorm_rows_fake(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    rows: Tensor | None,
) -> Tensor:
    n = routed.shape[0] if rows is None else rows.shape[0]
    return routed.new_empty((n, routed.shape[1]))


@torch_compile_guard(gen_fake=_moe_tail_rmsnorm_rows_fake, mutates_args=[])
def moe_tail_rmsnorm_rows_op(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    rows: Tensor | None,
) -> Tensor:
    return _moe_tail_rmsnorm_rows_impl(
        routed, shared, gate_logits, residual, weight, eps, rows
    )


def moe_tail_rmsnorm_rows(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    rows: Tensor | None,
) -> Tensor:
    """The normed output alone (a final norm: the new residual is not read),
    for rows `rows` only when given: `out[i]` is source row `rows[i]`."""
    if torch.compiler.is_compiling():
        return moe_tail_rmsnorm_rows_op(
            routed, shared, gate_logits, residual, weight, eps, rows
        )
    return _moe_tail_rmsnorm_rows_impl(
        routed, shared, gate_logits, residual, weight, eps, rows
    )


# ---------------------------------------------------------------------------
# ATOM_MXFP8_1X32: the normed rows written as 1x32 MXFP8 for the next projection
# ---------------------------------------------------------------------------
@triton.jit
def _moe_tail_rmsnorm_mx32_kernel(
    shared_ptr,
    gate_ptr,
    routed_ptr,
    residual_ptr,
    weight_ptr,
    res_out_ptr,
    q_ptr,
    s_ptr,
    stride_shared,
    stride_gate,
    stride_routed,
    stride_residual,
    stride_res_out,
    eps,
    M,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_GATE: tl.constexpr,
    N32K4: tl.constexpr,
):
    """`_moe_tail_rmsnorm_kernel` (no gather, residual written) whose normed row goes out as fp8
    e4m3 [M, N] plus one e8m0 scale per 32 columns in the 1x32 GEMM's x-scale layout
    (mxfp8_1x32: n32k4 for M >= 32, row-major below) instead of bf16. The row math is the
    default kernel's line for line, so the quantized values are those of its bf16 output.
    Programs past M (the n32k4 pad rows, up to a multiple of 32) write scale byte 0x7F.
    """
    row = tl.program_id(0).to(tl.int64)
    KG: tl.constexpr = N // 32
    g = tl.arange(0, KG)
    if row < M:
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        shared = tl.load(shared_ptr + row * stride_shared + cols, mask, other=0.0).to(
            tl.float32
        )
        if HAS_GATE:
            gate = tl.load(gate_ptr + row * stride_gate)
            gate = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype).to(tl.float32)
            shared = (shared * gate).to(shared_ptr.dtype.element_ty).to(tl.float32)
        routed = tl.load(routed_ptr + row * stride_routed + cols, mask, other=0.0).to(
            tl.float32
        )
        moe = (shared + routed).to(routed_ptr.dtype.element_ty).to(tl.float32)
        x = moe + tl.load(
            residual_ptr + row * stride_residual + cols, mask, other=0.0
        ).to(tl.float32)
        tl.store(
            res_out_ptr + row * stride_res_out + cols,
            x.to(res_out_ptr.dtype.element_ty),
            mask,
        )
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
        weight = tl.load(weight_ptr + cols, mask, other=0.0).to(tl.float32) + 1.0
        normed = ((x * rstd) * weight).to(routed_ptr.dtype.element_ty)
        q, e = mx32_quant_groups(tl.reshape(normed.to(tl.float32), (KG, 32)))
        qcols = g[:, None] * 32 + tl.arange(0, 32)[None, :]
        tl.store(q_ptr + row * N + qcols, q)
        tl.store(s_ptr + mx32_scale_offset(row, g, KG, N32K4), e)
    else:
        tl.store(
            s_ptr + mx32_scale_offset(row, g, KG, N32K4),
            tl.full([KG], 0x7F, dtype=tl.uint8),
        )


def can_fuse_moe_tail_mx32(moe: MoEGatedSum, residual: Tensor, weight: Tensor) -> bool:
    """`can_fuse_moe_tail` plus what the 1x32 epilogue needs: a power-of-2 hidden, K % 128.
    A deferred gather (ATOM_MOE_GATHER_TAIL_NORM's GatheredRouted) is not this kernel's: it
    takes the gather + tail kernel's epilogue (`can_fuse_gather_tail_mx32`), and where that
    does not apply runs in bf16 and the Linear quantizes (the same bytes)."""
    if isinstance(moe.routed, GatheredRouted):
        return False
    if not can_fuse_moe_tail(moe, residual, weight, None):
        return False
    hidden = moe.routed.shape[1]
    return hidden == triton.next_power_of_2(hidden) and hidden % 128 == 0


def _moe_tail_rmsnorm_mx32_impl(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    m, hidden = routed.shape
    mx32_note("fused: moe_tail_rmsnorm (qkv_proj / GDN in_proj input)")
    q, s = mx32_alloc(m, hidden, routed.device)
    res_out = torch.empty(routed.shape, dtype=routed.dtype, device=routed.device)
    if m:
        grid = (m + 31) // 32 * 32 if m >= 32 else m
        _moe_tail_rmsnorm_mx32_kernel[(grid,)](
            shared,
            gate_logits if gate_logits is not None else shared,
            routed,
            residual,
            weight,
            res_out,
            q,
            s,
            shared.stride(0),
            gate_logits.stride(0) if gate_logits is not None else 0,
            routed.stride(0),
            residual.stride(0),
            res_out.stride(0),
            eps,
            m,
            N=hidden,
            BLOCK=triton.next_power_of_2(hidden),
            HAS_GATE=gate_logits is not None,
            N32K4=m >= 32,
            num_warps=4 if hidden <= 2048 else 8,
            enable_fp_fusion=False,
        )
    return q, s, res_out


def _moe_tail_rmsnorm_mx32_fake(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    m, hidden = routed.shape
    q, s = mx32_alloc(m, hidden, routed.device)
    return q, s, torch.empty_like(routed)


@torch_compile_guard(gen_fake=_moe_tail_rmsnorm_mx32_fake, mutates_args=[])
def moe_tail_rmsnorm_mx32_op(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return _moe_tail_rmsnorm_mx32_impl(
        routed, shared, gate_logits, residual, weight, eps
    )


def moe_tail_rmsnorm_mx32(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """`moe_tail_rmsnorm` with the normed rows as 1x32 MXFP8 for the next projection:
    `(x_fp8 [M, hidden], x_scale (flat uint8, mxfp8_1x32 layout), new_residual)`. Bitwise
    `moe_tail_rmsnorm`'s normed rows followed by mxfp8_1x32.quant_standalone (scale pad rows 0x7F);
    the residual is bitwise `moe_tail_rmsnorm`'s. Callers check `can_fuse_moe_tail_mx32`.
    """
    if torch.compiler.is_compiling():
        return moe_tail_rmsnorm_mx32_op(
            routed, shared, gate_logits, residual, weight, eps
        )
    return _moe_tail_rmsnorm_mx32_impl(
        routed, shared, gate_logits, residual, weight, eps
    )


# ---------------------------------------------------------------------------
# ATOM_MOE_GATHER_TAIL_NORM: the routed experts' top-k gather-reduce in the tail
# ---------------------------------------------------------------------------
#
# The routed sum normally comes out of aiter's moe_gather_reduce (MoE2's output
# rows, `topk` of them a token, weighted and summed in fp32, rounded to bf16)
# and goes through memory to the kernel above. With the gather deferred
# (`GatheredRouted`), one kernel does both: a program a token loads its `topk`
# MoE2 rows by the route -> row map, sums them exactly as moe_gather_reduce does
# (fp32, `acc + w_k * row_k` for k = 0 .. topk - 1, no FMA, rounded to the
# activation dtype), and continues with `_moe_tail_row`'s steps.


class GatheredRouted(NamedTuple):
    """MoE2's output rows before the top-k weighted sum: `routed[t] = sum_k
    weights[t, k] * grouped[route_rows[t, k]]` (aiter's
    `flydsl_moe_gather_reduce`). Slots holding a negative row contribute 0;
    tokens at or past `num_valid_tokens` (a (1,) int32, padding) have no routed
    sum (the fused kernel takes it as 0, the reduce leaves the row unwritten)."""

    grouped: Tensor  # [rows, hidden], last dim dense
    route_rows: Tensor  # [tokens, topk] int32
    weights: Tensor  # [tokens, topk] f32 / bf16 / f16
    num_valid_tokens: Tensor | None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.route_rows.shape[0], self.grouped.shape[1])

    def materialize(self) -> Tensor:
        """The routed sum as moe_gather_reduce writes it (the unfused path)."""
        from aiter.ops.flydsl.grouped_moe_gfx1250 import flydsl_moe_gather_reduce

        return flydsl_moe_gather_reduce(
            self.grouped.view(1, *self.grouped.shape),
            self.route_rows,
            self.weights,
            num_valid_tokens=self.num_valid_tokens,
        )


def resolve_gathered(moe: MoEGatedSum) -> MoEGatedSum:
    """`moe` with a deferred gather (`GatheredRouted`) reduced to a tensor, for
    every consumer but the fused gather + tail kernel."""
    if isinstance(moe.routed, GatheredRouted):
        return moe._replace(routed=moe.routed.materialize())
    return moe


@triton.jit
def _gather_tail_row(
    row,
    src,
    grouped_ptr,
    route_rows_ptr,
    route_w_ptr,
    nvt_ptr,
    shared_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    res_out_ptr,
    stride_grouped,
    stride_shared,
    stride_gate,
    stride_residual,
    stride_res_out,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_GATE: tl.constexpr,
    WRITE_RESIDUAL: tl.constexpr,
    HAS_NVT: tl.constexpr,
):
    """Output row `row` (source row `src`): moe_gather_reduce's sum of the
    token's TOPK MoE2 rows, then `_moe_tail_row`'s steps; stores the new
    residual (WRITE_RESIDUAL) and returns the normed row in the activation
    dtype, [BLOCK] (bitwise both: the row layout, and with it the norm's
    reduction order, is the tail kernel's)."""
    act = shared_ptr.dtype.element_ty
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    if HAS_NVT:
        tok_valid = src < tl.load(nvt_ptr)
    else:
        tok_valid = True
    # Every load that does not wait on the route map goes first, then all TOPK
    # row loads before the first add: the reduction only waits on the slowest.
    shared = tl.load(shared_ptr + src * stride_shared + cols, mask, other=0.0).to(
        tl.float32
    )
    if HAS_GATE:
        gate = tl.load(gate_ptr + src * stride_gate)
    resid = tl.load(residual_ptr + src * stride_residual + cols, mask, other=0.0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        r = tl.load(route_rows_ptr + src * TOPK + k)
        w = tl.load(route_w_ptr + src * TOPK + k).to(tl.float32)
        live = tok_valid & (r >= 0)
        x = tl.load(
            grouped_ptr + r.to(tl.int64) * stride_grouped + cols,
            mask & live,
            other=0.0,
        ).to(tl.float32)
        w = tl.where(tok_valid, w, 0.0)
        # moe_gather_reduce's step: a product, then a sum, each rounded (an
        # FMA here differs from it in the last bit).
        acc = acc + w * x
    routed = acc.to(act).to(tl.float32)
    if HAS_GATE:
        gate = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype).to(tl.float32)
        shared = (shared * gate).to(act).to(tl.float32)
    moe = (shared + routed).to(act).to(tl.float32)
    x = moe + resid.to(tl.float32)
    if WRITE_RESIDUAL:
        tl.store(
            res_out_ptr + row * stride_res_out + cols,
            x.to(res_out_ptr.dtype.element_ty),
            mask,
        )
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    weight = tl.load(weight_ptr + cols, mask, other=0.0).to(tl.float32) + 1.0
    return ((x * rstd) * weight).to(act)


@triton.jit
def _moe_gather_tail_rmsnorm_kernel(
    grouped_ptr,
    route_rows_ptr,
    route_w_ptr,
    nvt_ptr,
    shared_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    rows_ptr,
    out_ptr,
    res_out_ptr,
    stride_grouped,
    stride_shared,
    stride_gate,
    stride_residual,
    stride_out,
    stride_res_out,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_GATE: tl.constexpr,
    GATHER: tl.constexpr,
    WRITE_RESIDUAL: tl.constexpr,
    HAS_NVT: tl.constexpr,
):
    """One program an output row: `_gather_tail_row`, the normed row stored
    in the activation dtype."""
    row = tl.program_id(0).to(tl.int64)
    if GATHER:
        src = tl.load(rows_ptr + row).to(tl.int64)
    else:
        src = row
    normed = _gather_tail_row(
        row,
        src,
        grouped_ptr,
        route_rows_ptr,
        route_w_ptr,
        nvt_ptr,
        shared_ptr,
        gate_ptr,
        residual_ptr,
        weight_ptr,
        res_out_ptr,
        stride_grouped,
        stride_shared,
        stride_gate,
        stride_residual,
        stride_res_out,
        eps,
        N,
        BLOCK,
        TOPK,
        HAS_GATE,
        WRITE_RESIDUAL,
        HAS_NVT,
    )
    cols = tl.arange(0, BLOCK)
    tl.store(out_ptr + row * stride_out + cols, normed, cols < N)


@triton.jit
def _moe_gather_tail_rmsnorm_mx32_kernel(
    grouped_ptr,
    route_rows_ptr,
    route_w_ptr,
    nvt_ptr,
    shared_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    res_out_ptr,
    q_ptr,
    s_ptr,
    stride_grouped,
    stride_shared,
    stride_gate,
    stride_residual,
    stride_res_out,
    eps,
    M,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_GATE: tl.constexpr,
    HAS_NVT: tl.constexpr,
    N32K4: tl.constexpr,
    NUM_XCD: tl.constexpr,
):
    """`_moe_gather_tail_rmsnorm_kernel` (no row subset, residual written)
    whose normed row goes out as 1x32 MXFP8 (`_moe_tail_rmsnorm_mx32_kernel`'s
    epilogue): fp8 e4m3 [M, N] plus one e8m0 byte per 32 columns in the GEMM's
    x-scale layout. One program a row; program r < pad32(M) - M also writes
    n32k4 pad row M + r (byte 0x7F).

    n32k4 puts a row's 64 bytes in 16 pieces of 4, 128 bytes apart: a 128-byte
    line holds 32 rows' pieces. Programs go to the XCDs round robin, so row
    after row would land on another XCD's L2 and no line would be assembled in
    one (measured: +30% kernel time at 8k rows). Rows are therefore remapped so
    that each XCD runs a contiguous range of them."""
    pid = tl.program_id(0)
    if N32K4 and NUM_XCD > 1:
        per = (M + NUM_XCD - 1) // NUM_XCD  # rows of the first `tall` XCDs
        tall = M - (per - 1) * NUM_XCD
        xcd = pid % NUM_XCD
        loc = pid // NUM_XCD
        pid = tl.where(
            xcd < tall,
            xcd * per + loc,
            tall * per + (xcd - tall) * (per - 1) + loc,
        )
    row = pid.to(tl.int64)
    KG: tl.constexpr = N // 32
    g = tl.arange(0, KG)
    normed = _gather_tail_row(
        row,
        row,
        grouped_ptr,
        route_rows_ptr,
        route_w_ptr,
        nvt_ptr,
        shared_ptr,
        gate_ptr,
        residual_ptr,
        weight_ptr,
        res_out_ptr,
        stride_grouped,
        stride_shared,
        stride_gate,
        stride_residual,
        stride_res_out,
        eps,
        N,
        N,
        TOPK,
        HAS_GATE,
        True,
        HAS_NVT,
    )
    # The fp8 codes as bytes (q_ptr is uint8): no fp8 conversion, so the route
    # map / weight / gate loads stay scalar loads (see mx32_e4m3_bits).
    q, e = mx32_quant_groups_bits(tl.reshape(normed.to(tl.float32), (KG, 32)))
    # 8 contiguous codes a thread, the reshape's own layout: a wider store
    # would go through an LDS layout conversion.
    qcols = tl.max_contiguous(g[:, None] * 32 + tl.arange(0, 32)[None, :], [1, 8])
    tl.store(q_ptr + row * N + qcols, q)
    tl.store(s_ptr + mx32_scale_offset(row, g, KG, N32K4), e)
    if N32K4:
        pad_row = M + row
        tl.store(
            s_ptr + mx32_scale_offset(pad_row, g, KG, N32K4),
            tl.full([KG], 0x7F, dtype=tl.uint8),
            (g < KG) & (pad_row < (M + 31) // 32 * 32),
        )


def can_fuse_gather_tail(
    moe: MoEGatedSum, residual: Tensor, weight: Tensor, rows: Tensor | None = None
) -> bool:
    """`can_fuse_moe_tail` for a deferred gather: the same row conditions, the
    MoE2 rows of the activation dtype and width, dense."""
    g = moe.routed
    if not isinstance(g, GatheredRouted):
        return False
    shared, gate = moe.shared, moe.gate_logits
    dtype = shared.dtype
    if dtype not in (torch.bfloat16, torch.half) or shared.dim() != 2:
        return False
    tokens, hidden = shared.shape
    if g.grouped.dtype != dtype or g.grouped.dim() != 2 or g.grouped.shape[1] != hidden:
        return False
    if g.route_rows.shape[0] != tokens or g.route_rows.dtype != torch.int32:
        return False
    if g.weights.shape != g.route_rows.shape or g.weights.dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.half,
    ):
        return False
    if not (g.route_rows.is_contiguous() and g.weights.is_contiguous()):
        return False
    if residual.dtype != dtype or residual.shape != shared.shape:
        return False
    if any(t.stride(1) != 1 for t in (g.grouped, shared, residual)):
        return False
    if gate is not None and (gate.dtype != dtype or gate.shape != (tokens, 1)):
        return False
    if weight.shape != (hidden,) or weight.stride(0) != 1:
        return False
    return rows is None or (
        rows.dim() == 1 and rows.dtype in (torch.int32, torch.int64)
    )


# Compiled fused kernels by launch key (see `moe_gather_tail_rmsnorm`).
_GATHER_COMPILED: dict[tuple, object] = {}


def moe_gather_tail_rmsnorm(
    moe: MoEGatedSum,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    rows: Tensor | None = None,
    write_residual: bool = True,
    out: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """`moe_tail_rmsnorm` (write_residual) / `moe_tail_rmsnorm_rows` (not) of
    `MoEGatedSum(gather_reduce(moe.routed), shared, gate)` in one launch,
    bitwise the two kernels in sequence on every token below
    `num_valid_tokens` (past it the routed sum is 0, where the reduce leaves the
    row unwritten). `moe.routed` is a `GatheredRouted` (`can_fuse_gather_tail`)."""
    g, shared, gate = moe
    tokens, hidden = shared.shape
    n_out = tokens if rows is None else rows.shape[0]
    assert rows is None or not write_residual, "a row subset has no new residual"
    if out is None or tuple(out.shape) != (n_out, hidden) or out.dtype != shared.dtype:
        # (`out`: ATOM_FINAL_NORM_INTO_OUTPUT's caller-owned rows, model_out_buffer)
        out = torch.empty((n_out, hidden), dtype=shared.dtype, device=shared.device)
    res_out = torch.empty_like(out) if write_residual else None
    if n_out == 0:
        return (out, res_out) if write_residual else out
    if envs.ATOM_MOE_GATHER_TAIL_NORM_PIPE:
        # Prefill sizes: the software-pipelined TDM kernel (same bits).
        from atom.model_ops import moe_gather_tail_gluon as tdm

        if tdm.gather_tail_tdm_takes(g, shared, residual, n_out):
            tdm.launch_gather_tail_tdm(
                g, shared, gate, residual, weight, eps, rows, out, res_out
            )
            return (out, res_out) if write_residual else out
    nvt = g.num_valid_tokens
    ptrs = (
        g.grouped,
        g.route_rows,
        g.weights,
        nvt if nvt is not None else g.route_rows,
        shared,
        gate if gate is not None else shared,
        residual,
        weight,
        rows if rows is not None else g.route_rows,
        out,
        res_out if res_out is not None else out,
    )
    strides = (
        g.grouped.stride(0),
        shared.stride(0),
        gate.stride(0) if gate is not None else 0,
        residual.stride(0),
        out.stride(0),
        res_out.stride(0) if res_out is not None else 0,
    )
    block = triton.next_power_of_2(hidden)
    topk = g.route_rows.shape[1]
    # The compiled kernel launched directly, as `_launch` does (eager prefill is
    # host-bound): only where every pointer is 16-byte aligned and every row
    # offset stays within 2 GiB (Triton's AMD buffer-op assumption), under a key
    # at least as fine as Triton's specialization.
    reach = (
        max(g.grouped.shape[0] * strides[0], max(tokens, n_out) * max(strides[1:]))
        + hidden
    ) * shared.element_size()
    cacheable = reach <= _INT32_MAX and not any(t.data_ptr() % 16 for t in ptrs)
    key = (
        hidden,
        block,
        topk,
        shared.dtype,
        g.weights.dtype,
        None if gate is None else gate.dtype,
        None if rows is None else rows.dtype,
        nvt is not None,
        write_residual,
        tuple((v == 1, v % 16 == 0) for v in strides),
    )
    kernel = _GATHER_COMPILED.get(key) if cacheable else None
    if kernel is not None:
        kernel[(n_out, 1, 1)](*ptrs, *strides, eps)
        return (out, res_out) if write_residual else out
    kernel = _moe_gather_tail_rmsnorm_kernel[(n_out,)](
        *ptrs,
        *strides,
        eps,
        N=hidden,
        BLOCK=block,
        TOPK=topk,
        HAS_GATE=gate is not None,
        GATHER=rows is not None,
        WRITE_RESIDUAL=write_residual,
        HAS_NVT=nvt is not None,
        # The tail kernel's layout (and so its sum-of-squares order).
        num_warps=4 if block <= 2048 else 8,
        enable_fp_fusion=False,
    )
    if cacheable:
        _GATHER_COMPILED[key] = kernel
    return (out, res_out) if write_residual else out


def can_fuse_gather_tail_mx32(
    moe: MoEGatedSum, residual: Tensor, weight: Tensor
) -> bool:
    """`can_fuse_gather_tail` (no row subset) plus what the 1x32 epilogue
    needs: a power-of-2 hidden, K % 128 (`can_fuse_moe_tail_mx32`'s)."""
    if not can_fuse_gather_tail(moe, residual, weight, None):
        return False
    hidden = moe.shared.shape[1]
    return hidden == triton.next_power_of_2(hidden) and hidden % 128 == 0


@functools.cache
def num_xcds() -> int:
    """XCDs of the device (aiter's table; 8 where it has no entry)."""
    try:
        from aiter.ops.triton.utils.device_info import get_num_xcds
    except ImportError:
        return 8
    return get_num_xcds()


# Compiled 1x32 fused kernels by launch key (see `moe_gather_tail_rmsnorm_mx32`).
_GATHER_MX32_COMPILED: dict[tuple, object] = {}


def moe_gather_tail_rmsnorm_mx32(
    moe: MoEGatedSum, residual: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    """`moe_gather_tail_rmsnorm` (residual written) with the normed rows as
    1x32 MXFP8 for the next projection: `(x_fp8 [M, hidden], x_scale (flat
    uint8, mxfp8_1x32 layout), new_residual)`, as `moe_tail_rmsnorm_mx32`
    returns. Bitwise `moe_gather_tail_rmsnorm`'s normed rows followed by
    mxfp8_1x32.quant_standalone (scale pad rows 0x7F); the residual is bitwise
    its residual. Callers check `can_fuse_gather_tail_mx32`. Eager / CUDA-graph
    only, like the deferred gather itself."""
    g, shared, gate = moe
    m, hidden = shared.shape
    mx32_note("fused: moe_gather_tail_rmsnorm (qkv_proj / GDN in_proj input)")
    q, s = mx32_alloc(m, hidden, shared.device)
    res_out = torch.empty_like(shared)
    if m == 0:
        return q, s, res_out
    if envs.ATOM_MOE_GATHER_TAIL_NORM_PIPE and m >= 32:  # its epilogue is n32k4
        from atom.model_ops import moe_gather_tail_gluon as tdm

        if tdm.gather_tail_tdm_takes(g, shared, residual, m):
            tdm.launch_gather_tail_tdm(
                g,
                shared,
                gate,
                residual,
                weight,
                eps,
                None,
                res_out,
                res_out,
                mx32=(q, s),
            )
            return q, s, res_out
    nvt = g.num_valid_tokens
    ptrs = (
        g.grouped,
        g.route_rows,
        g.weights,
        nvt if nvt is not None else g.route_rows,
        shared,
        gate if gate is not None else shared,
        residual,
        weight,
        res_out,
        q.view(torch.uint8),
        s,
    )
    strides = (
        g.grouped.stride(0),
        shared.stride(0),
        gate.stride(0) if gate is not None else 0,
        residual.stride(0),
        res_out.stride(0),
    )
    grid = m
    topk = g.route_rows.shape[1]
    # Launched directly where cacheable, as `moe_gather_tail_rmsnorm` does; M
    # is an int argument too (== 1 / % 16 in the key).
    reach = (
        max(g.grouped.shape[0] * strides[0], m * max(strides[1:])) + hidden
    ) * shared.element_size()
    cacheable = reach <= _INT32_MAX and not any(t.data_ptr() % 16 for t in ptrs)
    key = (
        hidden,
        topk,
        shared.dtype,
        g.weights.dtype,
        None if gate is None else gate.dtype,
        nvt is not None,
        m >= 32,
        tuple((v == 1, v % 16 == 0) for v in (*strides, m)),
    )
    kernel = _GATHER_MX32_COMPILED.get(key) if cacheable else None
    if kernel is not None:
        kernel[(grid, 1, 1)](*ptrs, *strides, eps, m)
        return q, s, res_out
    kernel = _moe_gather_tail_rmsnorm_mx32_kernel[(grid,)](
        *ptrs,
        *strides,
        eps,
        m,
        N=hidden,
        TOPK=topk,
        HAS_GATE=gate is not None,
        HAS_NVT=nvt is not None,
        N32K4=m >= 32,
        NUM_XCD=num_xcds(),
        # The tail kernel's layout (and so its sum-of-squares order).
        num_warps=4 if hidden <= 2048 else 8,
        enable_fp_fusion=False,
    )
    if cacheable:
        _GATHER_MX32_COMPILED[key] = kernel
    return q, s, res_out


def gated_sum(moe: MoEGatedSum) -> Tensor:
    """`routed + sigmoid(gate_logits) * shared` with the unfused ops' rounding
    (the fallback when the fused kernel cannot take the inputs)."""
    from atom.model_ops.sigmoid_gate import shared_expert_gate_add

    routed, shared, gate = resolve_gathered(moe)
    if gate is None:
        return shared + routed
    return shared_expert_gate_add(shared, gate, routed, True)


# ---------------------------------------------------------------------------
# ATOM_MEGA_COMBINE_TAIL: the tail inside aiter's MegaMoE combine
# ---------------------------------------------------------------------------
#
# With mori EP on aiter's MegaMoEGfx1250, the routed sum comes out of a combine
# kernel that already owns whole token rows. aiter's one-kernel combine
# (`MegaMoEGfx1250.forward(combine_tail=)`) then does the cross-device wait,
# the top-k reduce, this module's gated add, the residual add and the norm in
# ONE launch (instead of combine_sync + combine + the kernel above). The MoE
# block needs the norm's weight and the residual for that, which live in the
# decoder layer; it hands them down through `combine_tail_request` (a scope
# around its `self.experts(...)` call, read by the mori MegaMoE call site) and
# gets back `MoENormed`. Eager and CUDA-graph capture only: under torch.compile
# the FusedMoE custom op hides the call, so the block keeps MoEGatedSum.


class MoENormed(NamedTuple):
    """The next norm's outputs, already computed by the MoE combine:
    `(normed, new_residual)`; `residual` is None for a final norm."""

    out: Tensor
    residual: Tensor | None


class CombineTailRequest:
    """What the combine needs to finish the layer, and what it produced."""

    def __init__(self, shared, gate, residual, weight, eps, write_residual):
        self.shared = shared
        self.gate = gate
        self.residual = residual
        self.weight = weight
        self.eps = eps
        self.write_residual = write_residual
        self.result: tuple[Tensor, Tensor | None] | None = None


_COMBINE_TAIL_REQUEST: CombineTailRequest | None = None


class combine_tail_request:  # a context manager, named as one
    def __init__(self, req: CombineTailRequest):
        self.req = req

    def __enter__(self):
        global _COMBINE_TAIL_REQUEST
        self.prev, _COMBINE_TAIL_REQUEST = _COMBINE_TAIL_REQUEST, self.req
        return self.req

    def __exit__(self, *exc):
        global _COMBINE_TAIL_REQUEST
        _COMBINE_TAIL_REQUEST = self.prev


def current_combine_tail_request() -> CombineTailRequest | None:
    return _COMBINE_TAIL_REQUEST


# ---------------------------------------------------------------------------
# ATOM_SHARED_EXPERT_GATED_GEMM: the shared expert's SiLU-and-mul in its GEMM
# ---------------------------------------------------------------------------


def gated_gemm_available() -> bool:
    """aiter's gfx1250 gluon GEMM with the gate / up SiLU epilogue."""
    try:
        from aiter.jit.utils.chip_info import get_gfx
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import (  # noqa: F401
            gemm_a16w16_gated,
        )
    except ImportError:
        return False
    return get_gfx() == "gfx1250"


def interleave_gate_up(w: Tensor) -> Tensor:
    """[gate; up] rows -> g0, u0, g1, u1, ...: the layout the gated GEMM reads."""
    half = w.shape[0] // 2
    return torch.stack((w[:half], w[half:]), dim=1).reshape(w.shape).contiguous()


def _silu_gate_up_gemm_impl(x: Tensor, w_interleaved: Tensor) -> Tensor:
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16_gated

    return gemm_a16w16_gated(x, w_interleaved, dtype=x.dtype)


def _silu_gate_up_gemm_fake(x: Tensor, w_interleaved: Tensor) -> Tensor:
    return x.new_empty((x.shape[0], w_interleaved.shape[0] // 2))


@torch_compile_guard(gen_fake=_silu_gate_up_gemm_fake, mutates_args=[])
def silu_gate_up_gemm_op(x: Tensor, w_interleaved: Tensor) -> Tensor:
    return _silu_gate_up_gemm_impl(x, w_interleaved)


def silu_gate_up_gemm(x: Tensor, w_interleaved: Tensor) -> Tensor:
    """`silu(x @ gate.T) * (x @ up.T)` in one GEMM (bitwise the GEMM +
    act_and_mul pair at the Qwen3.5 shared-expert shapes up to 4k rows, within
    one bf16 ulp above)."""
    if torch.compiler.is_compiling():
        return silu_gate_up_gemm_op(x, w_interleaved)
    return _silu_gate_up_gemm_impl(x, w_interleaved)


# ---------------------------------------------------------------------------
# ATOM_FUSED_ROUTER_SHARED_EXPERT: the router GEMM and the shared expert in one
# kernel
# ---------------------------------------------------------------------------

# The fused kernel's last router tile reads whole 16-row groups fast; a
# 257-row weight makes its TDM loads slow. The weight is padded with zero rows
# to this multiple once, and the module keeps a view of the real rows.
_ROUTER_ROW_ALIGN = 16


def fused_router_shared_available() -> bool:
    """aiter's gfx1250 gluon router + shared expert kernel."""
    try:
        from aiter.jit.utils.chip_info import get_gfx
        from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_router_shared_expert import (  # noqa: F401
            fused_gemm_a16w16_router_shared_expert,
        )
    except ImportError:
        return False
    return get_gfx() == "gfx1250"


def fused_router_shared_prefill_available() -> bool:
    """aiter's prefill path of the fused router + shared expert (the router +
    gate_up GEMM)."""
    if not fused_router_shared_available():
        return False
    from aiter.ops.triton.gemm.fused import (
        fused_gemm_a16w16_router_shared_expert as k,
    )

    return hasattr(k, "fused_gemm_a16w16_router_shared_expert_prefill")


def pad_router_weight(weight: Tensor) -> Tensor:
    """`weight` [N, K] in a buffer of whole 16-row groups (zero rows past N):
    returns the padded buffer. Set the module weight to `padded[:N]`."""
    n, k = weight.shape
    rows = -(-n // _ROUTER_ROW_ALIGN) * _ROUTER_ROW_ALIGN
    padded = weight.new_zeros((rows, k))
    padded[:n].copy_(weight)
    return padded


def router_logits_gemm(x: Tensor, w_router_padded: Tensor, n_router: int) -> Tensor:
    """x @ w_router.T [M, n_router] by the a16w16 GEMM: bitwise the logits of
    `router_shared_expert` (both kernels run the a16w16 K chain)."""
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    return gemm_a16w16(x, w_router_padded[:n_router], None, x.dtype)


def shared_expert_gemms(x: Tensor, w_gate_up: Tensor, w_down: Tensor) -> Tensor:
    """The bf16 shared expert by two GEMMs (gated gate_up, down): bitwise the
    shared output of `router_shared_expert`. `w_gate_up` is interleaved."""
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    h = _silu_gate_up_gemm_impl(x, w_gate_up)
    return gemm_a16w16(h, w_down, None, x.dtype)


def router_shared_expert(
    x: Tensor,
    w_router_padded: Tensor,
    n_router: int,
    w_gate_up: Tensor,
    w_down: Tensor,
    prefill: bool = False,
) -> tuple[Tensor, Tensor]:
    """(x @ w_router.T [M, n_router], the shared expert's output [M, hidden]),
    bitwise the router GEMM and the gated gate_up + down GEMMs. `w_gate_up`
    is interleaved gate / up (`interleave_gate_up`). Decode sizes run one
    persistent kernel; with `prefill`, the router and gate_up run as one GEMM
    over both weights, then the down GEMM."""
    from aiter.ops.triton.gemm.fused import fused_gemm_a16w16_router_shared_expert as k

    if prefill:
        return k.fused_gemm_a16w16_router_shared_expert_prefill(
            x, w_router_padded, w_gate_up, w_down, None, None, None, n_router
        )
    return k.fused_gemm_a16w16_router_shared_expert(
        x, w_router_padded, w_gate_up, w_down, n_router=n_router
    )
