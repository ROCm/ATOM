# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The whole embedding table's host arithmetic (``embed_local``): its canonical
form, its digest and the ranks' agreement. Torch only -- no aiter, FlyDSL or
Triton -- so it is tested wherever the tests run."""

import torch

# rows a pass over the table touches
PASS_ROWS = 4096


def canonicalize_(table: torch.Tensor) -> torch.Tensor:
    """``table`` (BF16) in place as production's all-reduce returns a row
    summed with +0.0: zeros and denormals to +0.0, NaNs quiet, anything else
    as it is."""
    if table.dtype != torch.bfloat16:
        raise TypeError(f"a {table.dtype} table")
    raw = table.view(torch.int16)
    flat = raw.view(-1, raw.shape[-1]) if raw.dim() > 1 else raw.view(1, -1)
    for start in range(0, flat.shape[0], PASS_ROWS):
        part = flat[start : start + PASS_ROWS]
        exponent = (part >> 7) & 0xFF
        nan = (exponent == 0xFF) & ((part & 0x7F) != 0)
        part.masked_fill_(exponent == 0, 0)
        part.bitwise_or_(nan.to(torch.int16) << 6)
    return table


def checksum(table: torch.Tensor) -> int:
    """A 62-bit digest of a BF16 table's bits (sums of its int16 values,
    plain and weighted by row), for the ranks to agree on."""
    raw = table.view(torch.int16)
    plain = weighted = 0
    for start in range(0, raw.shape[0], PASS_ROWS):
        rows = raw[start : start + PASS_ROWS].to(torch.int64).sum(dim=1)
        index = torch.arange(start, start + rows.numel(), device=rows.device)
        plain += int(rows.sum())
        weighted += int((rows * (index % 8191 + 1)).sum())
    return (plain * 1000003 + weighted) & ((1 << 62) - 1)


def group_min(values, group) -> tuple[int, ...]:
    """For each position of ``values``, the smallest any rank of the CPU
    group passed there, in one host collective (``group`` None: one rank)."""
    mine = [int(value) for value in values]
    if group is None:
        return tuple(mine)
    import torch.distributed as dist

    flags = torch.tensor(mine, dtype=torch.int64)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
    return tuple(int(value) for value in flags.tolist())
