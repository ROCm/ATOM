# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The sampler's `softmax(logits / temperature)` split across many programs a
row (ATOM_SAMPLER_SPLIT_SOFTMAX).

torch's softmax over a row longer than shared memory (`cunn_SoftMaxForwardGmem`)
runs one block a row. A decode step samples 5-40 rows of the 248k-entry Qwen3.5
vocab, so most CUs sit idle and the call takes ~95 us on gfx1250 whatever the
row count, with the temperature division a further full pass before it.

Here the row is cut into chunks. Pass 1, one program a (row, chunk), reduces its
chunk of `logits / t` to an online (max, sum of exp) pair; pass 2, on the same
grid, combines the row's pairs and writes its chunk's probabilities. 248k
columns at 4096 a chunk are 61 programs a row: 5-13 us for 1-35 rows. Both
passes divide by the temperature as the eager `logits / t` does, in fp32.
"""

import torch
import triton
import triton.language as tl

_CHUNK = 4096
_BLOCK = 4096
_NUM_WARPS = 4


@triton.jit
def _split_softmax_stats_kernel(
    x_ptr, t_ptr, m_ptr, s_ptr, stride_x, n_cols, chunk, n_chunks, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.load(t_ptr + row).to(tl.float32)
    start = c * chunk
    end = tl.minimum(start + chunk, n_cols)
    m = -float("inf")
    s = 0.0
    for off in range(start, end, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * stride_x + cols, mask=cols < end, other=-float("inf"))
        x = x.to(tl.float32) / t
        new_m = tl.maximum(m, tl.max(x, 0))
        # A chunk that is all -inf so far keeps (m, s) = (-inf, 0).
        ref = tl.where(new_m == -float("inf"), 0.0, new_m)
        s = s * tl.exp(m - ref) + tl.sum(tl.exp(x - ref), 0)
        m = new_m
    tl.store(m_ptr + row * n_chunks + c, m)
    tl.store(s_ptr + row * n_chunks + c, s)


@triton.jit
def _split_softmax_write_kernel(
    x_ptr,
    t_ptr,
    m_ptr,
    s_ptr,
    y_ptr,
    stride_x,
    stride_y,
    n_cols,
    chunk,
    n_chunks,
    CHUNKS_P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.load(t_ptr + row).to(tl.float32)
    idx = tl.arange(0, CHUNKS_P2)
    ms = tl.load(m_ptr + row * n_chunks + idx, mask=idx < n_chunks, other=-float("inf"))
    ss = tl.load(s_ptr + row * n_chunks + idx, mask=idx < n_chunks, other=0.0)
    mx = tl.max(ms, 0)
    total = tl.sum(tl.where(ms == -float("inf"), 0.0, ss * tl.exp(ms - mx)), 0)
    inv = 1.0 / total
    start = c * chunk
    end = tl.minimum(start + chunk, n_cols)
    for off in range(start, end, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * stride_x + cols, mask=cols < end, other=-float("inf"))
        x = x.to(tl.float32) / t
        tl.store(y_ptr + row * stride_y + cols, tl.exp(x - mx) * inv, mask=cols < end)


def can_split_softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> bool:
    """fp32 logits [rows, cols] read densely along the row, one temperature a
    row. Other dtypes keep the eager path, whose division rounds to theirs."""
    return (
        logits.is_cuda
        and logits.dim() == 2
        and logits.dtype == torch.float32
        and logits.shape[0] > 0
        and logits.stride(1) == 1
        and temperatures.dim() == 1
        and temperatures.shape[0] == logits.shape[0]
    )


def split_softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """`(logits / temperatures[:, None]).softmax(dim=-1, dtype=torch.float32)`,
    contiguous, for inputs `can_split_softmax` accepts."""
    rows, cols = logits.shape
    n_chunks = triton.cdiv(cols, _CHUNK)
    temperatures = temperatures.contiguous()
    m = torch.empty(rows, n_chunks, device=logits.device, dtype=torch.float32)
    s = torch.empty_like(m)
    probs = torch.empty(rows, cols, device=logits.device, dtype=torch.float32)
    grid = (rows, n_chunks)
    _split_softmax_stats_kernel[grid](
        logits,
        temperatures,
        m,
        s,
        logits.stride(0),
        cols,
        _CHUNK,
        n_chunks,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    _split_softmax_write_kernel[grid](
        logits,
        temperatures,
        m,
        s,
        probs,
        logits.stride(0),
        probs.stride(0),
        cols,
        _CHUNK,
        n_chunks,
        CHUNKS_P2=triton.next_power_of_2(n_chunks),
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return probs
