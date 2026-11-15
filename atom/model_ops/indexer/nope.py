# SPDX-License-Identifier: MIT
"""NoPE index Q quantization and normalized FP8 index-cache publication.

Q retains the separate per-128 quantizer's FP32 scale. K retains the cache
writer's upward-rounded power-of-two scale and MFMA 16x16 byte layout.
Only activations and the index cache are quantized; projection weights stay BF16.
"""

import torch
import triton
import triton.language as tl
from aiter import dtypes


@triton.jit
def _indexer_nope_quant_and_cache(
    Q,
    K,
    Weights,
    NormWeight,
    NormBias,
    Slots,
    Cache,
    QOut,
    WeightsOut,
    Q_STRIDE_T: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    K_STRIDE_T: tl.constexpr,
    W_STRIDE_T: tl.constexpr,
    slot_count,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHTS_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head_group = tl.program_id(1)
    heads = head_group * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    cols = tl.arange(0, 128)
    q = tl.load(
        Q + token * Q_STRIDE_T + heads[:, None] * Q_STRIDE_H + cols[None, :],
        mask=heads[:, None] < HEADS,
        other=0.0,
    ).to(tl.float32)
    # AITER's per-128 Q quantizer uses a regular FP32 scale, not UE8M0.
    q_amax = tl.maximum(tl.max(tl.abs(q), axis=1), 1.0e-10)
    q_scale = q_amax * (1.0 / FP8_MAX)
    q_scaled = q * tl.div_rn(1.0, q_scale[:, None])
    q_scaled = tl.minimum(tl.maximum(q_scaled, -FP8_MAX), FP8_MAX)
    tl.store(
        QOut + token * HEADS * 128 + heads[:, None] * 128 + cols[None, :],
        q_scaled.to(QOut.dtype.element_ty),
        mask=heads[:, None] < HEADS,
    )
    weights = tl.load(
        Weights + token * W_STRIDE_T + heads, mask=heads < HEADS, other=0.0
    ).to(tl.float32)
    tl.store(
        WeightsOut + token * HEADS + heads,
        (weights * q_scale) * WEIGHTS_SCALE,
        mask=heads < HEADS,
    )

    # Only one head group publishes K for this token. Q is produced for padded
    # graph rows as well; padded/absent slots must never overwrite the cache.
    if head_group == 0:
        slot = tl.load(Slots + token, mask=token < slot_count, other=-1).to(tl.int64)
        if slot >= 0:
            k = tl.load(K + token * K_STRIDE_T + cols).to(tl.float32)
            mean = tl.sum(k, axis=0) * (1.0 / 128)
            centered = k - mean
            variance = tl.sum(centered * centered, axis=0) * (1.0 / 128)
            # LayerNorm's unfused path casts affine params to its BF16 input
            # dtype, then materializes a BF16 result before cache quantization.
            gamma = tl.load(NormWeight + cols).to(tl.bfloat16).to(tl.float32)
            beta = tl.load(NormBias + cols).to(tl.bfloat16).to(tl.float32)
            normalized = centered * tl.rsqrt(variance + EPS)
            k = (normalized * gamma + beta).to(tl.bfloat16).to(tl.float32)
            k_amax = tl.maximum(tl.max(tl.abs(k), axis=0), 1.0e-4)
            k_scale = tl.exp2(tl.ceil(tl.log2(k_amax / FP8_MAX)))
            k_scaled = k * tl.div_rn(1.0, k_scale)
            k_scaled = tl.minimum(tl.maximum(k_scaled, -FP8_MAX), FP8_MAX)
            k_bytes = k_scaled.to(QOut.dtype.element_ty).to(tl.uint8, bitcast=True)

            block = slot // PAGE_SIZE
            offset = slot % PAGE_SIZE
            base = block * PAGE_SIZE * CACHE_STRIDE
            # Match indexer_k_quant_and_cache(preshuffle=True): 16-token tiles,
            # each containing eight 16-column tiles, followed by FP32 scales.
            data_offset = (
                base
                + (offset // 16) * (16 * 128)
                + (cols // 16) * (16 * 16)
                + (offset % 16) * 16
                + cols % 16
            )
            tl.store(Cache + data_offset, k_bytes)
            scale_ptr = (Cache + base + PAGE_SIZE * 128 + offset * 4).to(
                tl.pointer_type(tl.float32)
            )
            tl.store(scale_ptr, k_scale)


def indexer_nope_quant_and_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    weights_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 Q [tokens, heads, 128], fold its scales into weights, and
    publish NoPE K.

    K and head weights may be strided views of a fused projection's output.
    The cache is the contiguous uint8 allocation with one FP32 scale per token;
    page padding beyond those bytes is untouched. The caller supplies unique,
    in-range physical slots, with -1 for rows without a cache write.
    Under DCP each rank receives all query rows and only its own K slots;
    non-owned slots are -1, so query quantization is independent of K ownership.
    """
    rows, heads, dim = q.shape
    assert dim == 128 and k.shape == (rows, dim) and weights.shape == (rows, heads)
    # The kernel indexes the cache as packed bytes of whole 16-token tiles.
    assert cache.dtype == torch.uint8 and cache.is_contiguous()
    assert cache.shape[1] % 16 == 0 and cache.shape[2] >= dim + 4
    q_out = torch.empty(q.shape, dtype=dtypes.fp8, device=q.device)
    weights_out = torch.empty(weights.shape, dtype=torch.float32, device=q.device)
    if rows:
        _indexer_nope_quant_and_cache[(rows, triton.cdiv(heads, 4))](
            q,
            k,
            weights,
            norm_weight,
            norm_bias,
            slot_mapping,
            cache,
            q_out,
            weights_out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            weights.stride(0),
            slot_mapping.numel(),
            heads,
            cache.shape[1],
            cache.shape[2],
            eps,
            weights_scale,
            torch.finfo(dtypes.fp8).max,
            HEADS_PER_PROGRAM=4,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, weights_out
