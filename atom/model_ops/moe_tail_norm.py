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

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard
from torch import Tensor


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


def gated_sum(moe: MoEGatedSum) -> Tensor:
    """`routed + sigmoid(gate_logits) * shared` with the unfused ops' rounding
    (the fallback when the fused kernel cannot take the inputs)."""
    from atom.model_ops.sigmoid_gate import shared_expert_gate_add

    routed, shared, gate = moe
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
