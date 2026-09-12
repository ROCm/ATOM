# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused tails for Qwen3.8-Flash-Next's hyper-connections.

Each of the 97 hyper-connections runs a `mix` and (bar the final mixer) a
`combine`, and the parts of them that are not a GEMM are all cheap elementwise
work over a `[tokens, hc_count * hidden]` tensor. Eagerly that is a dozen
launches per call and roughly a fifth of all GPU time; the arithmetic itself
is memory-bound and finishes in one pass.

Both kernels keep the eager operation order so the only difference is the
order of the mean over the `hc_count` streams, which is a fixed four-term sum
here.
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
    """`hyper + block_out * 2*sigmoid(injection)` over every stream.

    `injection` is the RAW `[tokens, hc_count]` projection; the sigmoid and the
    doubling happen here so the intermediate never reaches memory.
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
