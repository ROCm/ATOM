# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`x * sigmoid(gate) [+ residual]` as one launch (ATOM_FUSED_SIGMOID_GATE).

Qwen3-Next / Qwen3.5 run it eagerly as two or three elementwise kernels per
layer: the MoE shared-expert combine `routed + sigmoid(gate_logit) * shared`
and the attention output gate `o * sigmoid(gate)`. Inside a decode CUDA graph
each of those is a node of its own, so the dispatch cost, not the math, is
what they spend.

The kernel is qwen4_exp's `sigmoid_mul`, which keeps the BF16 rounding of each
eager step: the sigmoid rounds to the gate dtype, the product to the activation
dtype, and only then is the residual added in fp32 and rounded once more. With
one dtype throughout that is exactly what the eager ops compute.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard
from torch import Tensor

from atom.model_ops.mxfp8_1x32 import alloc as mx32_alloc
from atom.model_ops.mxfp8_1x32 import mx32_quant_groups, mx32_scale_offset
from atom.model_ops.mxfp8_1x32 import note as mx32_note
from atom.model_ops.qwen4_exp.ops.gated import sigmoid_mul


def can_fuse_sigmoid_gate(x: Tensor, gate: Tensor, residual: Tensor | None) -> bool:
    """Whether `fused_sigmoid_gate` reproduces the eager ops for these inputs.

    Shape, stride and dtype only, so the answer a capture records is the one
    every replay of that shape gets. One 16-bit float dtype throughout: with a
    wider gate or residual, eager promotes and skips a rounding the kernel
    keeps. The kernel reads rows by stride but each row densely.
    """
    if x.dim() != 2 or gate.dim() != 2 or x.dtype not in (torch.bfloat16, torch.half):
        return False
    rows, width = x.shape
    if gate.dtype != x.dtype or gate.shape not in ((rows, 1), (rows, width)):
        return False
    if x.stride(1) != 1 or (gate.shape[1] > 1 and gate.stride(1) != 1):
        return False
    if residual is None:
        return True
    return (
        residual.dtype == x.dtype
        and residual.shape == x.shape
        and residual.stride(1) == 1
    )


def _fused_sigmoid_gate_fake(
    x: Tensor, gate: Tensor, residual: Tensor | None = None
) -> Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


# A custom op so Dynamo sees one opaque call rather than tracing the launch.
@torch_compile_guard(gen_fake=_fused_sigmoid_gate_fake, mutates_args=[])
def fused_sigmoid_gate(
    x: Tensor, gate: Tensor, residual: Tensor | None = None
) -> Tensor:
    """`x * sigmoid(gate) [+ residual]`, contiguous, in x's dtype.

    `gate` is `[rows, 1]` (one gate per row) or `x`'s shape; callers check
    `can_fuse_sigmoid_gate` first.
    """
    return sigmoid_mul(x, gate, residual)


def shared_expert_gate_add(
    shared: Tensor, gate_logits: Tensor, routed: Tensor, fused: bool
) -> Tensor:
    """The MoE block's `sigmoid(gate_logits) * shared + routed`, one launch
    when `fused` and the inputs allow it, else the eager ops."""
    if fused and can_fuse_sigmoid_gate(shared, gate_logits, routed):
        return fused_sigmoid_gate(shared, gate_logits, routed)
    shared = F.sigmoid(gate_logits) * shared
    return shared + routed


def output_gate_mul(attn_output: Tensor, gate: Tensor, fused: bool) -> Tensor:
    """The attention output gate `attn_output * sigmoid(gate)`, one launch
    when `fused` and the inputs allow it, else the eager ops."""
    if fused and can_fuse_sigmoid_gate(attn_output, gate, None):
        return fused_sigmoid_gate(attn_output, gate)
    gate = torch.sigmoid(gate)
    return attn_output * gate


# --------------------------------------------------------------------------------------------------
# ATOM_MXFP8_1X32: the attention output gate writing o_proj's 1x32 MXFP8 input (one launch instead
# of the gate, then the standalone quant).
# --------------------------------------------------------------------------------------------------
@triton.jit
def _gated_mx32_kernel(
    x_ptr,
    gate_ptr,
    q_ptr,
    s_ptr,
    M,
    stride_x,
    stride_gate,
    WIDTH: tl.constexpr,
    GATE_WIDTH: tl.constexpr,
    COLS: tl.constexpr,
    N32K4: tl.constexpr,
):
    """`x * sigmoid(gate)` with `_gated_pointwise_kernel`'s roundings (sigmoid to the gate dtype,
    product to x's dtype), then 1x32 e8m0 quant of that bf16 product into the GEMM's x-scale layout
    (mxfp8_1x32). One row x COLS columns a program; programs on rows past M (n32k4 pad rows) write
    scale byte 0x7F."""
    row = tl.program_id(0).to(tl.int64)
    KG: tl.constexpr = WIDTH // 32
    g = tl.program_id(1) * (COLS // 32) + tl.arange(0, COLS // 32)
    if row < M:
        cols = g[:, None] * 32 + tl.arange(0, 32)[None, :]
        x = tl.load(x_ptr + row * stride_x + cols).to(tl.float32)
        gate_cols = cols * 0 if GATE_WIDTH == 1 else cols
        gate = tl.load(gate_ptr + row * stride_gate + gate_cols)
        gate = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype).to(tl.float32)
        result = (x * gate).to(x_ptr.dtype.element_ty).to(tl.float32)
        q, e = mx32_quant_groups(result)
        tl.store(q_ptr + row * WIDTH + cols, q)
        tl.store(s_ptr + mx32_scale_offset(row, g, KG, N32K4), e)
    else:
        tl.store(
            s_ptr + mx32_scale_offset(row, g, KG, N32K4),
            tl.full([COLS // 32], 0x7F, dtype=tl.uint8),
        )


def can_fuse_output_gate_mx32(attn_output: Tensor, gate: Tensor) -> bool:
    return (
        can_fuse_sigmoid_gate(attn_output, gate, None)
        and attn_output.shape[1] % 128 == 0
    )


def _output_gate_mul_mx32_impl(
    attn_output: Tensor, gate: Tensor
) -> tuple[Tensor, Tensor]:
    m, k = attn_output.shape
    mx32_note("fused: output gate (o_proj input)")
    q, s = mx32_alloc(m, k, attn_output.device)
    if m:
        cols = 1024 if k % 1024 == 0 else 128
        rows = (m + 31) // 32 * 32 if m >= 32 else m
        _gated_mx32_kernel[(rows, k // cols)](
            attn_output,
            gate,
            q,
            s,
            m,
            attn_output.stride(0),
            gate.stride(0),
            WIDTH=k,
            GATE_WIDTH=gate.shape[1],
            COLS=cols,
            N32K4=m >= 32,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, s


def _output_gate_mul_mx32_fake(
    attn_output: Tensor, gate: Tensor
) -> tuple[Tensor, Tensor]:
    m, k = attn_output.shape
    return mx32_alloc(m, k, attn_output.device)


@torch_compile_guard(gen_fake=_output_gate_mul_mx32_fake, mutates_args=[])
def output_gate_mul_mx32_op(attn_output: Tensor, gate: Tensor) -> tuple[Tensor, Tensor]:
    return _output_gate_mul_mx32_impl(attn_output, gate)


def output_gate_mul_mx32(attn_output: Tensor, gate: Tensor) -> tuple[Tensor, Tensor]:
    """`attn_output * sigmoid(gate)` as o_proj's 1x32 MXFP8 input: (fp8 [M, K], flat uint8 x_scale
    in the mxfp8_1x32 layout). Bitwise `fused_sigmoid_gate(attn_output, gate)` followed by
    mxfp8_1x32.quant_standalone. Callers check `can_fuse_output_gate_mx32`."""
    if torch.compiler.is_compiling():
        return output_gate_mul_mx32_op(attn_output, gate)
    return _output_gate_mul_mx32_impl(attn_output, gate)
