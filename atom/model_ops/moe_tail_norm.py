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

from atom.model_ops.mxfp8_act_quant import GROUP as _MX_GROUP
from atom.model_ops.mxfp8_act_quant import e8m0_quant_group, scale_view
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
def _moe_tail_row(
    row,
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
    WRITE_OUT: tl.constexpr,
):
    """One output row: stores the new residual / bf16 normed row as asked and
    returns the normed row (activation dtype)."""
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
    normed = ((x * rstd) * weight).to(out_ptr.dtype.element_ty)
    if WRITE_OUT:
        tl.store(out_ptr + row * stride_out + cols, normed, mask)
    return normed


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
    q_ptr,
    s_ptr,
    M,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_GATE: tl.constexpr,
    GATHER: tl.constexpr,
    WRITE_RESIDUAL: tl.constexpr,
    WRITE_OUT: tl.constexpr = True,
    QUANT_OUT: tl.constexpr = False,
    QUANT_ROWS: tl.constexpr = 1,
):
    """Default: one program per output row. QUANT_OUT: also the normed rows as
    MXFP8 (mxfp8_act_quant) for the next per_1x128 e8m0 projection -- fp8
    [M, N] dense, e8m0 byte of (row m, group g) at g * M + m; QUANT_ROWS rows a
    program, so each group's scales go out as QUANT_ROWS contiguous bytes
    instead of one byte M apart (a single-byte transposed store per row cost
    more than the rest of the quant). Every row is still the default kernel's
    1-D row computation, so the normed values (and with them the quant) are
    bitwise the default kernel's."""
    if not QUANT_OUT:
        row = tl.program_id(0).to(tl.int64)
        _moe_tail_row(
            row, shared_ptr, gate_ptr, routed_ptr, residual_ptr, weight_ptr, rows_ptr,
            out_ptr, res_out_ptr, stride_shared, stride_gate, stride_routed,
            stride_residual, stride_out, stride_res_out, eps,
            N, BLOCK, HAS_GATE, GATHER, WRITE_RESIDUAL, WRITE_OUT,
        )
    else:
        NG: tl.constexpr = N // 128
        cols = tl.arange(0, BLOCK)
        # Offsets provably contiguous only in runs of 8: a 16-wide fp8 store
        # would pull the row into a 16-per-thread layout and change the order
        # of the sum of squares (the default kernel reduces 8 per thread).
        qcols = (cols // 8) * 8 + cols % 8
        rid = tl.arange(0, QUANT_ROWS)
        row0 = tl.program_id(0).to(tl.int64) * QUANT_ROWS
        scales = tl.zeros((NG, QUANT_ROWS), dtype=tl.uint8)
        for r in tl.static_range(QUANT_ROWS):
            row = row0 + r
            if row < M:
                normed = _moe_tail_row(
                    row, shared_ptr, gate_ptr, routed_ptr, residual_ptr, weight_ptr,
                    rows_ptr, out_ptr, res_out_ptr, stride_shared, stride_gate,
                    stride_routed, stride_residual, stride_out, stride_res_out, eps,
                    N, BLOCK, HAS_GATE, GATHER, WRITE_RESIDUAL, WRITE_OUT,
                )
                q, e = e8m0_quant_group(tl.reshape(normed.to(tl.float32), (NG, 128)))
                tl.store(q_ptr + row * N + qcols, tl.reshape(q, (BLOCK,)))
                scales = tl.where(rid[None, :] == r, e[:, None], scales)
        g = tl.arange(0, NG)
        rows_v = row0 + rid
        tl.store(s_ptr + g[:, None] * M + rows_v[None, :], scales, rows_v[None, :] < M)


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
    out: Tensor | None,
    res_out: Tensor | None,
    q_out: Tensor | None = None,
    s_out: Tensor | None = None,
) -> None:
    """`out` None: no bf16 normed store (then `q_out` / `s_out` must be given).
    `q_out` [n_out, hidden] fp8 e4m3 and `s_out` (n_out * hidden / 128 bytes):
    also write the normed rows as MXFP8 (see `moe_tail_rmsnorm_mxfp8`)."""
    quant = q_out is not None
    assert out is not None or quant
    n_out = rows.shape[0] if rows is not None else routed.shape[0]
    hidden = routed.shape[1]
    if n_out == 0:
        return
    block = triton.next_power_of_2(hidden)
    if quant:
        assert hidden == block and hidden % _MX_GROUP == 0, hidden
        assert q_out.shape == (n_out, hidden) and q_out.is_contiguous(), q_out.shape
        assert s_out.numel() == n_out * (hidden // _MX_GROUP) and s_out.is_contiguous()
    ptrs = (
        shared,
        gate if gate is not None else shared,
        routed,
        residual,
        weight,
        rows if rows is not None else routed,
        # without a bf16 output the pointer only carries the activation dtype
        out if out is not None else routed,
        res_out if res_out is not None else (out if out is not None else routed),
    )
    strides = (
        shared.stride(0),
        gate.stride(0) if gate is not None else 0,
        routed.stride(0),
        residual.stride(0),
        out.stride(0) if out is not None else 0,
        res_out.stride(0) if res_out is not None else 0,
    )
    # Rows a program in the MXFP8 mode. A program's rows run one after another,
    # so more rows cost latency, but each group's scale bytes then go out
    # together instead of one byte M apart (gfx1250, 2048 hidden, measured:
    # one row is best up to 1k rows, 2 / 4 / 8 at 2k / 4k-8k / 16k).
    if not quant:
        qrows = 1
    elif n_out <= 1024:
        qrows = 1
    elif n_out <= 2048:
        qrows = 2
    elif n_out <= 8192:
        qrows = 4
    else:
        qrows = 8
    grid = triton.cdiv(n_out, qrows)
    # q, s and the row count are runtime args only in the MXFP8 mode (None, a
    # constexpr, otherwise -- the default kernel keeps its signature).
    qargs = (q_out, s_out, n_out) if quant else ()
    # Eager prefill is host-bound, and Triton's own dispatch (argument binding,
    # specialization, cache lookup) is most of this launch's host time. So the
    # compiled kernel is kept and launched directly, under a key at least as
    # fine as what Triton specializes on: dtypes, int args == 1 / % 16, and --
    # only cached when every pointer is 16-byte aligned and reaches no further
    # than 2 GiB from itself (the AMD buffer-op attribute) -- pointer alignment
    # and reach. Anything else, and a new key, takes Triton's own path.
    reach = (max(n_out, routed.shape[0]) * max(strides) + hidden) * routed.element_size()
    cacheable = reach <= _INT32_MAX and not any(t.data_ptr() % 16 for t in ptrs)
    key = (
        hidden,
        block,
        routed.dtype,
        None if gate is None else gate.dtype,
        None if rows is None else rows.dtype,
        res_out is not None,
        tuple((v == 1, v % 16 == 0) for v in strides),
        out is not None,
        quant,
        qrows,
        # Triton specializes the row count (== 1: a constexpr, not a runtime
        # arg; % 16: an alignment hint), so the key carries both
        quant and n_out == 1,
        quant and n_out % 16 == 0,
    )
    if quant:
        cacheable = cacheable and not any(t.data_ptr() % 16 for t in (q_out, s_out))
        if n_out == 1:
            qargs = (q_out, s_out)
    kernel = _COMPILED.get(key) if cacheable else None
    if kernel is not None:
        kernel[(grid, 1, 1)](*ptrs, *strides, eps, *qargs)
        return
    kernel = _moe_tail_rmsnorm_kernel[(grid,)](
        *ptrs,
        *strides,
        eps,
        q_out,
        s_out,
        n_out if quant else None,
        N=hidden,
        BLOCK=block,
        HAS_GATE=gate is not None,
        GATHER=rows is not None,
        WRITE_RESIDUAL=res_out is not None,
        WRITE_OUT=out is not None,
        QUANT_OUT=quant,
        QUANT_ROWS=qrows,
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


def _moe_tail_rmsnorm_mxfp8_impl(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    write_normed: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    m, hidden = routed.shape
    dev = routed.device
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty((hidden // _MX_GROUP) * m, dtype=torch.uint8, device=dev)
    res_out = torch.empty(routed.shape, dtype=routed.dtype, device=dev)
    out = torch.empty(routed.shape, dtype=routed.dtype, device=dev) if write_normed else None
    _launch(routed, shared, gate_logits, residual, weight, eps, None, out, res_out, q, s)
    if out is None:
        out = routed.new_empty((0, hidden))
    return q, scale_view(s, m, hidden), res_out, out


def _moe_tail_rmsnorm_mxfp8_fake(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    write_normed: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    m, hidden = routed.shape
    return (
        routed.new_empty((m, hidden), dtype=torch.float8_e4m3fn),
        routed.new_empty((m, hidden // _MX_GROUP), dtype=torch.float8_e8m0fnu),
        torch.empty_like(routed),
        routed.new_empty((m if write_normed else 0, hidden)),
    )


@torch_compile_guard(gen_fake=_moe_tail_rmsnorm_mxfp8_fake, mutates_args=[])
def moe_tail_rmsnorm_mxfp8_op(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    write_normed: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _moe_tail_rmsnorm_mxfp8_impl(
        routed, shared, gate_logits, residual, weight, eps, write_normed
    )


def moe_tail_rmsnorm_mxfp8(
    routed: Tensor,
    shared: Tensor,
    gate_logits: Tensor | None,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    write_normed: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`moe_tail_rmsnorm` with the normed rows also written as MXFP8 for the next
    layer's per_1x128 e8m0 projection: returns `(x_fp8, x_scale, residual, normed)`.

    `x_fp8` [M, hidden] fp8 e4m3 and `x_scale` [M, hidden / 128] e8m0 (the
    transposed layout: storage byte of row m, group g at g * M + m, viewed as
    (M, hidden / 128)) are bitwise `per_group_quant_hip(normed, group_size=128,
    transpose_scale=True, scale_type=fp8_e8m0)` of the bf16 normed rows, so
    `linear(x_fp8, x_scale=x_scale)` matches the unfused Linear. `residual` is
    the new residual, bitwise `moe_tail_rmsnorm`'s. `normed` (the bf16 rows,
    bitwise `moe_tail_rmsnorm`'s) is only written when `write_normed`; otherwise
    an empty [0, hidden] tensor. hidden: a power of two, a multiple of 128.
    Only for a norm whose sole consumer is such a projection -- not the final
    norm / MTP drafter norm (`moe_tail_rmsnorm_rows`)."""
    if torch.compiler.is_compiling():
        return moe_tail_rmsnorm_mxfp8_op(
            routed, shared, gate_logits, residual, weight, eps, write_normed
        )
    return _moe_tail_rmsnorm_mxfp8_impl(
        routed, shared, gate_logits, residual, weight, eps, write_normed
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
    """One program an output row: moe_gather_reduce's sum of the token's TOPK
    MoE2 rows, then `_moe_tail_row` (bitwise both: the row layout, and with it
    the norm's reduction order, is the tail kernel's)."""
    row = tl.program_id(0).to(tl.int64)
    if GATHER:
        src = tl.load(rows_ptr + row).to(tl.int64)
    else:
        src = row
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
    routed = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    if HAS_GATE:
        gate = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype).to(tl.float32)
        shared = (shared * gate).to(shared_ptr.dtype.element_ty).to(tl.float32)
    moe = (shared + routed).to(out_ptr.dtype.element_ty).to(tl.float32)
    x = moe + resid.to(tl.float32)
    if WRITE_RESIDUAL:
        tl.store(
            res_out_ptr + row * stride_res_out + cols,
            x.to(res_out_ptr.dtype.element_ty),
            mask,
        )
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    weight = tl.load(weight_ptr + cols, mask, other=0.0).to(tl.float32) + 1.0
    normed = ((x * rstd) * weight).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_out + cols, normed, mask)


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
