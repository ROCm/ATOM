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
from atom.utils import envs

from .layout import MFMA, NATURAL, PRESHUFFLE


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
    if layout == MFMA:
        if heads not in (32, 64) or dim != 128:
            raise ValueError("gfx950 OPUS queries require heads=32 or 64, dim=128")
        scales = torch.empty(
            rows, 2, 32, heads // 16, dtype=torch.uint8, device=query.device
        )
    elif layout == NATURAL:
        scales = torch.empty(
            rows, heads, dim // 32, dtype=torch.uint8, device=query.device
        )
    elif layout == PRESHUFFLE:
        scales = torch.empty(
            rows, dim // 128, 4, 16, -(-(heads // 16) // 4) * 4,
            dtype=torch.uint8, device=query.device,
        )  # fmt: skip
    else:
        raise ValueError(f"unknown FP4 query layout {layout!r}")
    rope_rotate_activation(
        values.view(dtypes.fp4x2),
        query,
        cos_cache,
        sin_cache,
        positions,
        rope_dim,
        out_scale=scales,
        group_size=32,
        scale_layout={NATURAL: "none", MFMA: "opus", PRESHUFFLE: "flydsl"}[layout],
        do_rotate_act=False,
        round_rope=round_rope,
    )
    return values.unsqueeze(1), scales.unsqueeze(1)


def quantize_query_rows(query: torch.Tensor):
    """FP8 `quantize_query` of a query already rotated: E4M3 with one fp32
    scale per `(token, head)` row, `(values [..., dim], scales [rows, 1])`."""
    rows = query.reshape(-1, query.shape[-1])
    values = torch.empty_like(rows, dtype=dtypes.fp8)
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


# gfx1250 FP4 indexer: the per-layer `weights_proj` GEMM folded into the query
# RoPE/MXFP4-quant launch (aiter `q_rope_mxfp4_quant`). Above this many decode
# rows the standalone bf16 GEMM is the faster kernel (it is not launch-bound),
# matching the vLLM reference's 128-row gate; the aiter projection K-loop also
# tiles the hidden width by _PROJECT_BLOCK_K, which must divide it.
_MAX_FUSED_PROJECT_ROWS = 128
_PROJECT_BLOCK_K = 256

# `[max_pos, rope_dim]` cos|sin tables, built once per rope buffer set rather
# than concatenated every decode step, keyed on the source tables' identity.
_cos_sin_cache: dict = {}


def _gfx1250() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime() == "gfx1250"
    except Exception:
        return False


def _combined_cos_sin(cos_cache: torch.Tensor, sin_cache: torch.Tensor):
    """`[max_pos, rope_dim]`, the cos half then the sin half, as
    `q_rope_mxfp4_quant` reads its table -- the same `cos_cache`/`sin_cache`
    halves `quantize_query` hands the HIP op, concatenated."""
    key = (cos_cache.data_ptr(), sin_cache.data_ptr(), str(cos_cache.device))
    hit = _cos_sin_cache.get(key)
    if hit is None:
        hit = torch.cat((cos_cache, sin_cache), dim=-1).contiguous()
        _cos_sin_cache[key] = hit
    return hit


def can_fuse_weights_proj(layout, query, hidden, weights_proj_weight, rope) -> bool:
    """Whether `project_query_fp4` can fold the `weights_proj` GEMM into the
    query-quant launch for this step: the gfx1250 FP4 NATURAL plane, at most
    128 decode rows, a unit-stride query, a 2-D hidden one row per query row,
    and a plain bf16/fp16 projection `[heads, hidden]` whose width the K-loop
    tiles evenly. Any unmet precondition falls back to the separate GEMM, which
    is always correct."""
    return (
        envs.ATOM_DSV41_FUSE_WEIGHTS_PROJ
        and layout == NATURAL
        and query.ndim == 3
        and query.stride(-1) == 1
        and 0 < query.shape[0] <= _MAX_FUSED_PROJECT_ROWS
        and hidden.ndim == 2
        and hidden.shape[0] == query.shape[0]
        and hidden.stride(-1) == 1
        and hidden.dtype in (torch.bfloat16, torch.float16)
        and weights_proj_weight.ndim == 2
        and weights_proj_weight.dtype in (torch.bfloat16, torch.float16)
        and weights_proj_weight.stride(-1) == 1
        and weights_proj_weight.shape[0] == query.shape[1]
        and weights_proj_weight.shape[1] == hidden.shape[1]
        and weights_proj_weight.shape[1] % _PROJECT_BLOCK_K == 0
        and hasattr(rope, "cos_cache")
        and hasattr(rope, "sin_cache")
        and _gfx1250()
    )


def project_query_fp4(query, rope, positions, hidden, weights_proj_weight):
    """`((values, scales), weights)` for the FP4 NATURAL plane with the head
    weights projected inside the quant launch, so the per-layer `weights_proj`
    GEMM is not its own launch. `query` is [rows, heads, dim], `hidden` [rows,
    hidden], `weights_proj_weight` [heads, hidden]. `(values, scales)` match
    `quantize_query`'s NATURAL FP4 branch; `weights` [rows, heads] is the raw
    bf16 projection `Indexer.project` returns, which `score_quantized` then
    folds `weights_scale` into (so no scale is applied here).

    Caller guards the shapes with `can_fuse_weights_proj`."""
    from aiter.ops.triton.rope.q_rope_mxfp4_quant import q_rope_mxfp4_quant

    q_fp4, q_scale, weights = q_rope_mxfp4_quant(
        query,
        positions,
        _combined_cos_sin(rope.cos_cache, rope.sin_cache),
        weights_scale=1.0,
        hidden=hidden.contiguous(),
        weights_proj=weights_proj_weight,
        proj_block_k=_PROJECT_BLOCK_K,
    )
    # The OPUS FP4 MQA scorer reads BF16 head weights, as the baseline
    # `weights_proj` GEMM emits; the aiter op returns them f32, so round back
    # (lossless -- the kernel already rounded the projection through bf16).
    return (q_fp4.unsqueeze(1), q_scale.unsqueeze(1)), weights.to(torch.bfloat16)
