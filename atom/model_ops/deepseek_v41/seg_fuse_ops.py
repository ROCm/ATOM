# SPDX-License-Identifier: MIT
"""Guarded ops of `ATOM_DSV41_SEG_FUSE` (see `atom.models.deepseek_v41.seg_fuse`).

Imported only when an item that needs one is on: importing this file registers
two custom ops and binds AITER's `silu_mul_mxfp8` (an AITER without it raises
ImportError here, which `seg_fuse.act_quant_op` turns into "item declined").
"""

import torch
from aiter import dtypes
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.dsv41_seg_fuse import silu_mul_mxfp8

from atom.models.deepseek_v41 import seg_fuse


def _silu_mul_mxfp8_fake(combined: torch.Tensor, limit: float) -> list[torch.Tensor]:
    width = combined.shape[-1] // 2
    lead = combined.shape[:-1]
    return [
        combined.new_empty((*lead, width), dtype=dtypes.fp8),
        combined.new_empty((*lead, width // 32), dtype=torch.float8_e8m0fnu),
    ]


@torch_compile_guard(mutates_args=[], gen_fake=_silu_mul_mxfp8_fake)
def v41_silu_mul_mxfp8(combined: torch.Tensor, limit: float) -> list[torch.Tensor]:
    """Item 7: `[gate | up]` bf16 -> `[x_fp8, x_scale]`, the pair `w2` reads:
    clamped SiLU-mul rounded to BF16, then the group-32 MXFP8 quant, in one
    launch."""
    x_fp8, x_scale = silu_mul_mxfp8(combined, limit)
    return [x_fp8, x_scale]


def _down_topk_fake(
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scale: float,
) -> list[torch.Tensor]:
    rows = logits.shape[0]
    return [
        act_fp8.new_empty((act_fp8.shape[0], weight.shape[0]), dtype=dtypes.bf16),
        logits.new_empty((rows, top_k), dtype=torch.float32),
        logits.new_empty((rows, top_k), dtype=torch.int32),
    ]


@torch_compile_guard(mutates_args=[], gen_fake=_down_topk_fake)
def v41_down_topk(
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scale: float,
) -> list[torch.Tensor]:
    """Item 10: `[w2(act_fp8, act_scale), topk_weights, topk_ids]`. One launch
    where AITER's tuned `w2` row at this row count can host the selection
    (`gemm_a8w8_blockscale_topk_supported`), else the two launches as before:
    `w2`'s GEMM, then `topk_gating` -- the same outputs either way."""
    from aiter import topk_gating
    from aiter.ops.gemm_topk_tail import (
        gemm_a8w8_blockscale_topk,
        gemm_a8w8_blockscale_topk_supported,
    )

    from atom.model_ops.blockscale import native_quant_linear
    from atom.utils.forward_context import pass_scheduled_tokens_gpu

    rows = logits.shape[0]
    ids = torch.empty(rows, top_k, dtype=torch.int32, device=logits.device)
    weights = torch.empty(rows, top_k, dtype=torch.float32, device=logits.device)
    num_valid_rows = pass_scheduled_tokens_gpu(rows)
    host = gemm_a8w8_blockscale_topk_supported(
        act_fp8, weight, act_scale, weight_scale, weights, ids, logits, bias,
        renormalize, scale, num_valid_rows,
    )  # fmt: skip
    if host is not None:
        out = gemm_a8w8_blockscale_topk(
            act_fp8, weight, act_scale, weight_scale, weights, ids, logits, bias,
            renormalize, scale, num_valid_rows, kernel_name=host,
        )  # fmt: skip
    else:
        out = native_quant_linear(
            act_fp8,
            weight,
            weight_scale,
            x_scale=act_scale,
            weight_group_rows=32,
            weight_preshuffled=True,
        )
        topk_gating(
            weights,
            ids,
            logits,
            bias,
            renormalize,
            scale,
            score_func="sqrtsoftplus",
            num_valid_rows=num_valid_rows,
        )
    seg_fuse.note_tail_rows(logits.shape[1], rows, host)
    return [out, weights, ids]
