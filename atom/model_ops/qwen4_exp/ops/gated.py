# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen-local gated operators: HyperConnection, sigmoid RMSNorm, and output gates.

Each of the 97 hyper-connections runs a `mix` and (bar the final mixer) a
`combine`, and the parts of them that are not a GEMM are all cheap elementwise
work over a `[tokens, hc_count * hidden]` tensor. Fusing them avoids separate
elementwise launches and intermediate tensors.

The sigmoid norm and output gates retain the reference's intermediate BF16
rounding. They cannot use SiLU-gated or FP8-quantizing norm implementations.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mix_gated_mean_kernel(
    normed_ptr,
    gate_ptr,
    out_ptr,
    stride_row,
    stride_out_row,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """`mean_over_streams(sigmoid(gate) * normed)` in one pass."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for stream in tl.static_range(HC):
        column = stream * HIDDEN + offsets
        normed = tl.load(
            normed_ptr + row * stride_row + column, mask=mask, other=0.0
        ).to(tl.float32)
        gate = tl.load(gate_ptr + row * stride_row + column, mask=mask, other=0.0).to(
            tl.float32
        )
        total += tl.sigmoid(gate) * normed
    tl.store(
        out_ptr + row * stride_out_row + offsets,
        (total / HC).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _combine_inject_kernel(
    hyper_ptr,
    block_ptr,
    injection_ptr,
    out_ptr,
    stride_row,
    stride_block_row,
    stride_inject_row,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """`residual + block_out * 2*sigmoid(injection)`, broadcast over streams."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    block_out = tl.load(
        block_ptr + row * stride_block_row + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    for stream in tl.static_range(HC):
        column = stream * HIDDEN + offsets
        residual = tl.load(
            hyper_ptr + row * stride_row + column, mask=mask, other=0.0
        ).to(tl.float32)
        raw = tl.load(injection_ptr + row * stride_inject_row + stream).to(tl.float32)
        raw = (raw / HC).to(injection_ptr.dtype.element_ty).to(tl.float32)
        value = residual + block_out * (2.0 * tl.sigmoid(raw))
        tl.store(
            out_ptr + row * stride_row + column,
            value.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def mix_gated_mean(
    normed: torch.Tensor, gate: torch.Tensor, hc_count: int
) -> torch.Tensor:
    """`(sigmoid(gate) * normed).unflatten(-1, (hc, H)).mean(-2)`."""
    width = normed.shape[-1]
    hidden = width // hc_count
    flat_normed = normed.reshape(-1, width).contiguous()
    flat_gate = gate.reshape(-1, width).contiguous()
    rows = flat_normed.shape[0]
    out = torch.empty((rows, hidden), dtype=normed.dtype, device=normed.device)
    if rows:
        _mix_gated_mean_kernel[(rows,)](
            flat_normed,
            flat_gate,
            out,
            flat_normed.stride(0),
            out.stride(0),
            HIDDEN=hidden,
            HC=hc_count,
            BLOCK=triton.next_power_of_2(hidden),
            num_warps=8,
        )
    return out.reshape(*normed.shape[:-1], hidden)


def combine_inject(
    hyper_input: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """`hyper + block_out * 2*sigmoid(injection / hc_count)` over every stream.

    `injection` is the unscaled `[tokens, hc_count]` projection. Scaling,
    sigmoid and doubling happen here; scaling still rounds to its input dtype.
    """
    width = hyper_input.shape[-1]
    hidden = width // hc_count
    flat_hyper = hyper_input.reshape(-1, width).contiguous()
    flat_block = block_output.reshape(-1, hidden).contiguous()
    flat_inject = injection.reshape(-1, hc_count).contiguous()
    rows = flat_hyper.shape[0]
    out = torch.empty_like(flat_hyper)
    if rows:
        _combine_inject_kernel[(rows,)](
            flat_hyper,
            flat_block,
            flat_inject,
            out,
            flat_hyper.stride(0),
            flat_block.stride(0),
            flat_inject.stride(0),
            HIDDEN=hidden,
            HC=hc_count,
            BLOCK=triton.next_power_of_2(hidden),
            num_warps=8,
        )
    return out.reshape(hyper_input.shape)


@triton.jit
def _gated_pointwise_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    out_ptr,
    stride_x,
    stride_gate,
    stride_residual,
    WIDTH: tl.constexpr,
    GATE_WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    SILU: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride_x + cols, cols < WIDTH, 0).to(tl.float32)
    if SILU:
        x = (x / SCALE).to(x_ptr.dtype.element_ty).to(tl.float32)
        result = x * tl.sigmoid(x)
    else:
        gate_cols = tl.full((BLOCK,), 0, tl.int32) if GATE_WIDTH == 1 else cols
        gate = tl.load(gate_ptr + row * stride_gate + gate_cols, cols < WIDTH, 0)
        gate = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype).to(tl.float32)
        result = (x * gate).to(x_ptr.dtype.element_ty).to(tl.float32)
        if ADD_RESIDUAL:
            residual = tl.load(
                residual_ptr + row * stride_residual + cols, cols < WIDTH, 0
            ).to(tl.float32)
            result += residual
    tl.store(out_ptr + row * WIDTH + cols, result, cols < WIDTH)


def scaled_silu(x: torch.Tensor, scale: int) -> torch.Tensor:
    """SiLU(x / scale), retaining the intermediate activation dtype."""
    rows, width = x.shape
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if rows:
        block = min(1024, triton.next_power_of_2(width))
        _gated_pointwise_kernel[(rows, triton.cdiv(width, block))](
            x,
            x,
            x,
            out,
            x.stride(0),
            0,
            0,
            width,
            width,
            scale,
            True,
            False,
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def sigmoid_mul(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """x * sigmoid(gate) [+ residual], with a per-row or per-element gate."""
    rows, width = x.shape
    if gate.shape not in ((rows, 1), (rows, width)):
        raise ValueError("gate must be [tokens, 1] or match the activation")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if rows:
        block = min(1024, triton.next_power_of_2(width))
        _gated_pointwise_kernel[(rows, triton.cdiv(width, block))](
            x,
            gate,
            residual if residual is not None else x,
            out,
            x.stride(0),
            gate.stride(0),
            residual.stride(0) if residual is not None else 0,
            width,
            gate.shape[1],
            1,
            False,
            residual is not None,
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _sigmoid_rmsnorm_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    stride_x_token,
    stride_x_head,
    stride_gate_token,
    stride_gate_head,
    ROWS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_DIM)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < DIM)
    tokens, heads = rows // HEADS, rows % HEADS
    x = tl.load(
        x_ptr
        + tokens[:, None] * stride_x_token
        + heads[:, None] * stride_x_head
        + cols[None, :],
        mask,
        0,
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / DIM
    normalized = x * tl.rsqrt(variance[:, None] + EPS)
    # Transformers rounds normalized values BEFORE affine multiplication, and
    # rounds the affine result BEFORE multiplying by the FP32 sigmoid gate.
    normalized = normalized.to(x_ptr.dtype.element_ty).to(tl.float32)
    weight = tl.load(weight_ptr + cols, cols < DIM, 0).to(tl.float32)
    affine = (normalized * weight[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
    gate = tl.load(
        gate_ptr
        + tokens[:, None] * stride_gate_token
        + heads[:, None] * stride_gate_head
        + cols[None, :],
        mask,
        0,
    ).to(tl.float32)
    tl.store(
        out_ptr + rows[:, None] * DIM + cols[None, :], affine * tl.sigmoid(gate), mask
    )


def sigmoid_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Qwen GDN's norm-before-sigmoid contract; returns [tokens, heads * dim]."""
    if x.ndim != 3 or x.shape != gate.shape or x.shape[-1] != weight.numel():
        raise ValueError("sigmoid RMSNorm expects matching [tokens, heads, dim] inputs")
    if x.stride(-1) != 1 or gate.stride(-1) != 1 or weight.stride(0) != 1:
        raise ValueError("sigmoid RMSNorm requires contiguous head dimensions")
    tokens, heads, dim = x.shape
    out = x.new_empty((tokens, heads * dim))
    if tokens:
        _sigmoid_rmsnorm_kernel[(triton.cdiv(tokens * heads, 4),)](
            x,
            gate,
            weight,
            out,
            x.stride(0),
            x.stride(1),
            gate.stride(0),
            gate.stride(1),
            tokens * heads,
            heads,
            dim,
            eps,
            4,
            triton.next_power_of_2(dim),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
