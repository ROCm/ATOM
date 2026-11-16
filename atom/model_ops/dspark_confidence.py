# SPDX-License-Identifier: MIT
"""DSpark confidence head as one launch: concatenation, N=1 GEMV and sigmoid.

`DSparkConfidenceHead` concatenates ``h_k`` with the Markov embedding and runs
a GEMM with a single output column. At draft widths that GEMM tiles M by 128
and N by 128, so a handful of workgroups stream the activations while 127 of
every 128 output lanes are padding; the concatenated copy costs as much again.
This reads both operands where they are and reduces each row in FP32.
"""

from functools import cache

import torch
import triton
import triton.language as tl

_BLOCK_ROWS = 4
_BLOCK_K = 512


@triton.jit
def _confidence_kernel(
    hidden_ptr,
    embed_ptr,
    weight_ptr,
    out_ptr,
    rows,
    stride_hidden,
    stride_embed,
    HIDDEN: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    live = row < rows
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for start in tl.static_range(0, HIDDEN, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        x = tl.load(
            hidden_ptr + row[:, None].to(tl.int64) * stride_hidden + k[None, :],
            mask=live[:, None],
            other=0.0,
        )
        w = tl.load(weight_ptr + k)
        acc += tl.sum(x.to(tl.float32) * w.to(tl.float32)[None, :], axis=1)
    k = tl.arange(0, RANK)
    x = tl.load(
        embed_ptr + row[:, None].to(tl.int64) * stride_embed + k[None, :],
        mask=live[:, None],
        other=0.0,
    )
    w = tl.load(weight_ptr + HIDDEN + k)
    acc += tl.sum(x.to(tl.float32) * w.to(tl.float32)[None, :], axis=1)
    tl.store(out_ptr + row, tl.sigmoid(acc), mask=live)


@cache
def _is_gfx1250(device):
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return arch.split(":")[0] == "gfx1250"


def can_fuse_confidence(hidden, embed, weight, bias) -> bool:
    """Static gate: bias-free BF16 head whose widths tile exactly."""
    hidden_size, rank = hidden.shape[-1], embed.shape[-1]
    return (
        bias is None
        and hidden.is_cuda
        and hidden.dtype == embed.dtype == weight.dtype == torch.bfloat16
        and weight.shape == (1, hidden_size + rank)
        and weight.is_contiguous()
        and hidden.shape[:-1] == embed.shape[:-1]
        and hidden.stride(-1) == embed.stride(-1) == 1
        and hidden_size % _BLOCK_K == 0
        and rank == triton.next_power_of_2(rank)
        and _is_gfx1250(hidden.device)
    )


def fused_confidence(hidden, embed, weight):
    """sigmoid([hidden, embed] @ weight.T) in FP32, shaped like ``hidden[..., 0]``."""
    lead = hidden.shape[:-1]
    hidden = hidden.reshape(-1, hidden.shape[-1])
    embed = embed.reshape(-1, embed.shape[-1])
    rows = hidden.shape[0]
    out = torch.empty(rows, device=hidden.device, dtype=torch.float32)
    if rows:
        _confidence_kernel[(triton.cdiv(rows, _BLOCK_ROWS),)](
            hidden,
            embed,
            weight,
            out,
            rows,
            hidden.stride(0),
            embed.stride(0),
            HIDDEN=hidden.shape[-1],
            RANK=embed.shape[-1],
            BLOCK_ROWS=_BLOCK_ROWS,
            BLOCK_K=_BLOCK_K,
            num_warps=4,
        )
    return out.view(lead)
