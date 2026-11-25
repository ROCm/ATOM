# SPDX-License-Identifier: MIT
"""DSpark confidence head as one launch: concatenation, N=1 GEMV and sigmoid.

`DSparkConfidenceHead` concatenates ``h_k`` with the Markov embedding and runs
a GEMM with a single output column. At draft widths that GEMM tiles M by 128
and N by 128, so a handful of workgroups stream the activations while 127 of
every 128 output lanes are padding; the concatenated copy costs as much again.
This reads both operands where they are and reduces each row in FP32.
"""

import torch
import triton
import triton.language as tl

# Rows a program reduces: each weight chunk it loads serves four rows, and a
# draft batch still spreads over enough programs to fill the GPU.
_BLOCK_ROWS = 4
# Hidden columns a loop step reads: the DSpark heads' hidden widths are
# multiples of it (the gate requires that, so the loop needs no mask).
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


def _fits_kernel(hidden, embed, weight) -> bool:
    """What `_confidence_kernel` reads without masks: ``hidden`` in whole
    `_BLOCK_K` chunks, the rank as one power-of-two `tl.arange`, unit-stride
    rows of matching lead shape and a contiguous ``[1, hidden + rank]`` head."""
    hidden_size, rank = hidden.shape[-1], embed.shape[-1]
    return (
        hidden.shape[:-1] == embed.shape[:-1]
        and hidden.stride(-1) == embed.stride(-1) == 1
        and weight.shape == (1, hidden_size + rank)
        and weight.is_contiguous()
        and hidden_size % _BLOCK_K == 0
        and rank > 0
        and rank & (rank - 1) == 0
    )


def can_fuse_confidence(hidden, embed, weight, bias) -> bool:
    """Static gate: a bias-free BF16 head on the GPU that the kernel tiles.

    No arch gate: the kernel is plain Triton, and both gfx1250 and gfx950
    measured it faster than cat + N=1 GEMM + sigmoid at every draft batch
    tried, the widest as well as a single request.
    """
    return (
        bias is None
        and hidden.is_cuda
        and hidden.dtype == embed.dtype == weight.dtype == torch.bfloat16
        and _fits_kernel(hidden, embed, weight)
    )


def fused_confidence(hidden, embed, weight):
    """sigmoid([hidden, embed] @ weight.T) in FP32, shaped like ``hidden[..., 0]``.

    The kernel loads without masks, so a shape it does not tile
    (`_fits_kernel`) raises here rather than reading out of bounds.
    """
    if not _fits_kernel(hidden, embed, weight):
        raise ValueError(
            "fused_confidence needs hidden a multiple of "
            f"{_BLOCK_K}, a power-of-two rank, unit-stride rows and a "
            "contiguous [1, hidden + rank] weight; got hidden "
            f"{tuple(hidden.shape)}, embed {tuple(embed.shape)}, weight "
            f"{tuple(weight.shape)}"
        )
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
