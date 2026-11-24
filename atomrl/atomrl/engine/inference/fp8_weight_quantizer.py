# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Actor-side FP8 per-block quantization for weight sync.

Wraps the HF-named BF16 weight generator from ``get_per_tensor_param`` and
yields FP8-quantized weights + scale tensors. This halves the RDMA transfer
size while keeping training in full BF16 precision.

The quantized output is compatible with vLLM's ``fp8_per_block`` quantization
format: each weight tensor is cast to ``float8_e4m3fn`` with per-128×128-block
FP32 inverse-scale tensors.

Usage::

    params, _ = engine.get_per_tensor_param()
    quantized = quantize_weights_fp8_per_block(params)
    send_weight_stream(group, quantized, ...)
"""

from __future__ import annotations

from collections.abc import Generator, Iterable

import torch

# Parameters that should NOT be quantized to FP8 (norms, biases, scalars, HC, etc.)
_FP8_SKIP_SUFFIXES = (
    ".bias",
    "_layernorm.weight",
    "layernorm.weight",
    "norm.weight",
    ".attn_sink",
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
    ".ape",
    "embed_tokens.weight",
    "embed.weight",
    "lm_head.weight",
    "head.weight",
    "model.norm.weight",
    ".e_score_correction_bias",
    ".tid2eid",
    "mlp.gate.weight",  # router weight
    "ffn.gate.weight",  # RedHat/vLLM router weight
    # DSV4 compressor projections are explicitly constructed with
    # quant_config=None in vLLM and must remain BF16.
    "compressor.wkv.weight",
    "compressor.wgate.weight",
    # DSV4 indexer score projection is also explicitly unquantized in vLLM.
    "indexer.weights_proj.weight",
)

# Parameters that SHOULD be quantized (linear weights in attention + MLP + experts)
_FP8_QUANTIZE_PATTERNS = (".weight",)


def _should_quantize(name: str) -> bool:
    """Determine if a parameter should be FP8 quantized."""
    if not name.endswith(".weight"):
        return False
    for skip in _FP8_SKIP_SUFFIXES:
        if name.endswith(skip) or skip in name:
            return False
    return True


def _is_rocm() -> bool:
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def _is_fp8_fnuz() -> bool:
    """True only where the platform FP8 format is e4m3*fnuz* (gfx942 and earlier).

    Not all of ROCm is fnuz: gfx950 (MI355X) uses OCP e4m3 with max 448, the same
    format as CUDA. Gating on "is ROCm" quantised gfx950 weights as fnuz with a
    224 cap -- a 2x-too-small range, and a different format from the one TE and
    vLLM use on that arch.
    """
    if not _is_rocm():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except RuntimeError:
        return False
    return any(a in arch for a in ("gfx940", "gfx941", "gfx942"))


# Elements per FP32 chunk in _per_block_cast_to_fp8. 16M bf16 elements -> a 64 MiB
# FP32 temporary, small enough to stay under the BF16 path's peak on every tensor
# Qwen3-8B/DSV4 produce, large enough that the loop runs a handful of iterations.
_FP32_CHUNK_ELEMS = 1 << 24


def _scale_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"FP8 scale source must end with '.weight': {name}")
    return f"{name[:-len('.weight')]}.weight_scale_inv"


@torch.no_grad()
def _per_block_cast_to_fp8(
    tensor: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
    pow2_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast a BF16/FP32 tensor to FP8 e4m3 with per-block scaling.

    Returns ``(fp8_weight, scale_inv)`` where:
    - ``fp8_weight`` has the same shape, dtype ``torch.float8_e4m3fn``
      (on ROCm, converted to ``float8_e4m3fnuz`` to match the platform FP8 format)
    - ``scale_inv`` has shape ``(ceil(M/block_m), ceil(N/block_n))``, dtype FP32
    """
    assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D"
    M, N = tensor.shape
    block_m, block_n = block_size

    fp8_dtype = torch.float8_e4m3fnuz if _is_fp8_fnuz() else torch.float8_e4m3fn
    if fp8_dtype is torch.float8_e4m3fnuz:
        # Match vLLM.utils.deep_gemm.get_fp8_min_max(). PyTorch reports
        # +/-240 for fnuz, but vLLM uses +/-224 on ROCm for accuracy.
        FP8_MAX, FP8_MIN = 224.0, -224.0
    else:
        FP8_MAX = torch.finfo(fp8_dtype).max
        FP8_MIN = torch.finfo(fp8_dtype).min

    # Pad to multiple of block size (stay in bf16 to avoid FP32 copy)
    pad_m = (block_m - M % block_m) % block_m
    pad_n = (block_n - N % block_n) % block_n
    if pad_m or pad_n:
        padded = torch.nn.functional.pad(tensor, (0, pad_n, 0, pad_m))
    else:
        padded = tensor

    PM, PN = padded.shape
    blocks = padded.reshape(PM // block_m, block_m, PN // block_n, block_n)
    blocks = blocks.permute(0, 2, 1, 3)  # (bm, bn, block_m, block_n)

    # Per-block amax in bf16 -> FP32 scale (only the scale grid is FP32, not the full tensor)
    amax = blocks.abs().amax(dim=(2, 3)).float().clamp(min=1e-12)  # (bm, bn) FP32
    scale = amax / FP8_MAX
    if pow2_scales:
        # Round the dequant multiplier UP to a power of two, matching TE. TE's
        # compute_scale_from_amax() truncates the mantissa of FP8_MAX/amax, so its
        # reciprocal -- the scale_inv the GEMM consumes -- is rounded up. ceil here,
        # not round-to-nearest, or roughly half the blocks disagree with TE by 2x.
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    weight_scale_inv = scale.to(torch.float32)

    # Quantize through an FP32 reciprocal, in chunks of block-rows.
    #
    # Dividing in BF16 first (which this used to do, to avoid an FP32 copy) rounds
    # the quotient to 8 mantissa bits before the cast rounds it again to e4m3's 3,
    # and that double rounding is never in our favour: measured against vLLM's
    # per_block_cast_to_fp8 on Qwen3-8B, 3.01% of elements came out one ULP worse
    # and none came out better (RMSE 0.8% higher). vLLM's online fp8_per_block
    # quantizer -- the thing the rollout uses when the actor does NOT pre-quantize
    # -- multiplies by an FP32 reciprocal, so matching it here makes pre-quantized
    # and online weight sync bit-identical rather than merely close.
    #
    # Chunking keeps that accuracy without paying for it: one FP32 temp per chunk
    # peaks at 96 MiB on a 12288x4096 tensor, against 192 MiB for the BF16 divide
    # it replaces (and 384 MiB if the whole tensor goes through FP32 at once).
    # Keep the block-grid dims on the reciprocal. A 0-dim tensor is a "wrapped
    # number" for type promotion, so ``bf16 * zero_dim_fp32`` comes back BF16 and
    # would silently restore the double rounding.
    recip = scale.reciprocal().unsqueeze(-1).unsqueeze(-1)
    elems_per_block_row = max(1, blocks[0].numel())
    step = max(1, _FP32_CHUNK_ELEMS // elems_per_block_row)
    blocks_fp8 = torch.empty(blocks.shape, dtype=fp8_dtype, device=blocks.device)
    for i in range(0, blocks.shape[0], step):
        sl = slice(i, i + step)
        blocks_fp8[sl] = (blocks[sl] * recip[sl]).clamp(FP8_MIN, FP8_MAX).to(fp8_dtype)

    # Reshape back and trim padding
    fp8_padded = blocks_fp8.permute(0, 2, 1, 3).reshape(PM, PN)
    fp8_weight = fp8_padded[:M, :N].contiguous()

    return fp8_weight, weight_scale_inv


def quantize_weights_fp8_per_block(
    weights: Iterable[tuple[str, torch.Tensor]],
    block_size: tuple[int, int] = (128, 128),
    pow2_scales: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Wrap a BF16 weight generator to yield FP8-quantized weights.

    For each weight that should be quantized:
    - Yields ``(name, fp8_weight)`` — the quantized weight
    - Yields ``(name.replace('.weight', '.weight_scale_inv'), scale)`` — the scale

    For weights that should NOT be quantized (norms, biases, embeddings, etc.):
    - Yields ``(name, tensor)`` unchanged

    This is compatible with vLLM's ``fp8_per_block`` loading path, which
    expects ``.weight`` as FP8 and ``.weight_scale_inv`` as FP32.
    """
    for name, tensor in weights:
        if _should_quantize(name) and tensor.ndim == 2 and tensor.is_floating_point():
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            fp8_weight, scale_inv = _per_block_cast_to_fp8(
                tensor, block_size, pow2_scales=pow2_scales
            )
            yield name, fp8_weight
            scale_name = _scale_name_for_weight(name)
            yield scale_name, scale_inv
        else:
            yield name, tensor
