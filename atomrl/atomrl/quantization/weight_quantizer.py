# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Blockwise FP8 weight packing for checkpoints and offline transforms."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from atomrl.quantization.fp8_config import FP8Config

logger = logging.getLogger(__name__)

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def fp8_e4m3_max() -> float:
    if _FP8_DTYPE is not None:
        return float(torch.finfo(_FP8_DTYPE).max)
    return 448.0


def _round_scale_pow2(scale: Tensor) -> Tensor:
    """Round scales UP to a power of two (IEEE-style exponent-only scaling).

    Must be ceil, not round-to-nearest, to agree with TransformerEngine: TE's
    compute_scale_from_amax() truncates the mantissa of FP8_MAX/amax, so the
    reciprocal it hands the GEMM -- the dequant multiplier, which is what ``scale``
    is here -- is rounded up. Round-to-nearest disagrees with TE on roughly half of
    all blocks, by exactly one power of two, and vLLM also use ceil
    (exp2(ceil(log2(s)))). See test_round_scale_pow2_rounds_up.
    """
    log2 = torch.log2(scale.clamp(min=torch.finfo(scale.dtype).tiny))
    return torch.pow(2.0, torch.ceil(log2))


class WeightQuantizer:
    """BF16 ↔ FP8 (E4M3) blockwise transforms for ``state_dict`` tensors."""

    def __init__(self, config: FP8Config) -> None:
        self._config = config
        if _FP8_DTYPE is None:
            logger.warning(
                "torch.float8_e4m3fn is unavailable; quantize/dequantize uses bfloat16 surrogate storage."
            )

    def quantize_tensor(
        self, tensor: Tensor, block_size: int = 128
    ) -> tuple[Tensor, Tensor]:
        """Blockwise FP8 quantization along the last dimension.

        Returns:
            Tuple of ``(quantized, scales)`` where ``quantized`` matches the input shape
            (dtype ``float8_e4m3fn`` when supported, else ``bfloat16`` surrogate), and ``scales`` has
            shape ``tensor.shape[:-1] + (tensor.shape[-1] // block_size,)``.
        """
        if tensor.dim() < 1:
            raise ValueError("quantize_tensor expects tensor.dim() >= 1")

        last = tensor.shape[-1]
        if last % block_size != 0:
            raise ValueError(
                f"Last dim {last} must be divisible by block_size={block_size}"
            )

        leading = 1
        for s in tensor.shape[:-1]:
            leading *= int(s)
        flat = tensor.reshape(leading, last)
        blocks = flat.unfold(-1, block_size, block_size)
        amax = blocks.abs().amax(dim=-1)
        fp8_max = tensor.new_tensor(fp8_e4m3_max())
        scales = amax / fp8_max
        scales = torch.clamp(scales, min=torch.finfo(amax.dtype).tiny)
        if self._config.use_weight_pow2_scale:
            scales = _round_scale_pow2(scales)

        expanded = scales.unsqueeze(-1).expand_as(blocks)
        quantized_blocks = (blocks / expanded).to(dtype=torch.float32)

        if _FP8_DTYPE is not None:
            q = quantized_blocks.to(_FP8_DTYPE)
        else:
            q = torch.clamp(quantized_blocks, -fp8_e4m3_max(), fp8_e4m3_max()).to(
                torch.bfloat16
            )

        q_flat = q.reshape(leading, last)
        q_view = q_flat.reshape(tensor.shape)
        return q_view, scales.reshape(tensor.shape[:-1] + (last // block_size,))

    def dequantize_tensor(
        self, quantized: Tensor, scales: Tensor, block_size: int = 128
    ) -> Tensor:
        """Inverse of :meth:`quantize_tensor` for a single tensor."""
        last = quantized.shape[-1]
        leading = 1
        for s in quantized.shape[:-1]:
            leading *= int(s)
        q_flat = quantized.reshape(leading, last).to(dtype=torch.float32)
        blocks = q_flat.unfold(-1, block_size, block_size)
        expanded = scales.unsqueeze(-1).expand_as(blocks).to(dtype=torch.float32)
        dq_blocks = blocks * expanded
        return dq_blocks.reshape(quantized.shape)

    def quantize_state_dict(
        self, state_dict: dict[str, Tensor]
    ) -> dict[str, Tensor | Any]:
        """Return a new dict with 2-D float weights converted to FP8 + sidecar scales."""
        out: dict[str, Tensor | Any] = {}
        for key, value in state_dict.items():
            if not isinstance(value, Tensor):
                out[key] = value
                continue
            if value.dtype not in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
                torch.float64,
            ):
                out[key] = value
                continue
            if value.ndim != 2:
                out[key] = value
                continue
            out_features, in_features = value.shape
            if in_features % 128 != 0:
                logger.debug(
                    "Skipping %s: in_features=%s not aligned to 128", key, in_features
                )
                out[key] = value
                continue
            q, s = self.quantize_tensor(value.to(torch.float32), block_size=128)
            out[key] = q
            out[f"{key}_fp8_scales"] = s.to(torch.bfloat16)
            out[f"{key}_fp8_meta"] = torch.tensor(
                [out_features, in_features], dtype=torch.int64, device="cpu"
            )
        return out

    def dequantize_state_dict(
        self, state_dict: dict[str, Tensor]
    ) -> dict[str, Tensor | Any]:
        """Strip FP8 tensors back to BF16 weights using stored scales."""
        consumed_suffixes = {"_fp8_scales", "_fp8_meta"}
        out: dict[str, Tensor | Any] = {}
        for key, value in state_dict.items():
            if any(key.endswith(sfx) for sfx in consumed_suffixes):
                continue
            if not isinstance(value, Tensor):
                out[key] = value
                continue
            scale_key = f"{key}_fp8_scales"
            if scale_key not in state_dict:
                out[key] = value
                continue
            scales = state_dict[scale_key]
            dq = self.dequantize_tensor(value, scales.to(value.device), block_size=128)
            out[key] = dq.to(torch.bfloat16)
        return out
