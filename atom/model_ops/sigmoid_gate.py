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


@triton.jit
def _gated_rows_kernel(
    x_ptr,
    gate_ptr,
    rows_ptr,
    out_ptr,
    stride_x,
    stride_gate,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """out[r] = x[rows[r]] * sigmoid(gate[rows[r]]), rounded as
    `_gated_pointwise_kernel`: the sigmoid to the gate dtype, the product to x's."""
    out_row = tl.program_id(0)
    row = tl.load(rows_ptr + out_row).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < WIDTH
    x = tl.load(x_ptr + row * stride_x + cols, mask, 0).to(tl.float32)
    gate = tl.load(gate_ptr + row * stride_gate + cols, mask, 0)
    gate = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype).to(tl.float32)
    result = (x * gate).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + out_row.to(tl.int64) * WIDTH + cols, result, mask)


def _output_gate_rows_fake(attn_output: Tensor, gate: Tensor, rows: Tensor) -> Tensor:
    return attn_output.new_empty((rows.shape[0], attn_output.shape[1]))


@torch_compile_guard(gen_fake=_output_gate_rows_fake, mutates_args=[])
def output_gate_rows(attn_output: Tensor, gate: Tensor, rows: Tensor) -> Tensor:
    """`(attn_output * sigmoid(gate))[rows]` in one launch, bitwise
    `output_gate_mul` then the row gather (the MTP drafter's step 0,
    ATOM_MTP_SELECT_ROWS_BEFORE_O_PROJ). The rows must be in range."""
    if not can_fuse_sigmoid_gate(attn_output, gate, None) or gate.shape[1] == 1:
        out = output_gate_mul(attn_output, gate, True)
        return torch.index_select(out, 0, rows)
    width = attn_output.shape[1]
    out = attn_output.new_empty((rows.shape[0], width))
    if rows.shape[0]:
        block = min(1024, triton.next_power_of_2(width))
        _gated_rows_kernel[(rows.shape[0], triton.cdiv(width, block))](
            attn_output,
            gate,
            rows,
            out,
            attn_output.stride(0),
            gate.stride(0),
            WIDTH=width,
            BLOCK=block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
