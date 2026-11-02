# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The index query as the scorer reads it: RoPE and quantization in one
launch (aiter `rope_rotate_activation`), the rotated values never written
back in bf16.

FP8 quantizes each `(token, head)` row with one fp32 scale, which the scorer
can only dequantize by folding it into that head's weight (`fold_query_scale`).
FP4 is E2M1 with one e8m0 scale per 32 dims, its scales in the order the
cache's layout's kernel reads them; the kernel dequantizes them itself."""

import torch
from aiter import dtypes, rope_rotate_activation
from aiter.ops.quant import dynamic_per_token_scaled_quant

from atom.model_ops.v4_kernels import scale_indexer_weights

from .rows import NATURAL, PRESHUFFLE


def quantize_query(
    query: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    *,
    fp4: bool,
    layout: str = PRESHUFFLE,
    round_rope: bool = False,
):
    """`(values, scales)` of `query` [rows, heads, dim] after RoPE on its last
    `rope_dim` dims. FP8: values [rows, heads, dim] e4m3, scales [rows,
    heads, 1] fp32. FP4: values [rows, 1, heads, dim / 2] packed, scales
    [rows, 1, ...] e8m0 in `layout`'s order. `round_rope` rounds the rotated
    values to `query`'s dtype before quantizing, as a model whose RoPE writes
    them does."""
    rows, heads, dim = query.shape
    if not fp4:
        values = torch.empty_like(query, dtype=dtypes.fp8)
        scales = torch.empty(rows * heads, 1, dtype=torch.float32, device=query.device)
        rope_rotate_activation(
            values,
            query,
            cos_cache,
            sin_cache,
            positions,
            rope_dim,
            out_scale=scales,
            group_size=dim,
            do_rotate_act=False,
            round_rope=round_rope,
        )
        return values, scales.view(rows, heads, 1)
    values = torch.empty(rows, heads, dim // 2, dtype=torch.uint8, device=query.device)
    if layout == NATURAL:
        scales = torch.empty(
            rows, heads, dim // 32, dtype=torch.uint8, device=query.device
        )
    else:
        scales = torch.empty(
            rows, dim // 128, 4, 16, -(-(heads // 16) // 4) * 4,
            dtype=torch.uint8, device=query.device,
        )  # fmt: skip
    rope_rotate_activation(
        values.view(dtypes.fp4x2),
        query,
        cos_cache,
        sin_cache,
        positions,
        rope_dim,
        out_scale=scales,
        group_size=32,
        shuffle_scale=layout != NATURAL,
        do_rotate_act=False,
        round_rope=round_rope,
    )
    return values.unsqueeze(1), scales.unsqueeze(1)


def quantize_query_rows(query: torch.Tensor):
    """FP8 `quantize_query` of a query already rotated: E4M3 with one fp32
    scale per `(token, head)` row, `(values [..., dim], scales [rows, 1])`."""
    rows = query.reshape(-1, query.shape[-1])
    values = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows.shape[0], 1), dtype=torch.float32, device=rows.device)
    dynamic_per_token_scaled_quant(values, rows, scales)
    return values.view_as(query), scales


def fold_query_scale(
    weights: torch.Tensor, scales: torch.Tensor, weights_scale: float, prefix=""
) -> torch.Tensor:
    """FP8: the head weights the scorer reads, `weights * scales *
    weights_scale` in fp32 -- where an FP8 query's scale is dequantized."""
    return scale_indexer_weights(
        weights.contiguous(), scales, weights_scale, prefix=prefix
    )
