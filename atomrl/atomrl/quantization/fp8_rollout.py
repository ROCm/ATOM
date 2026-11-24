# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 W8A8 rollout quantization applied to linear layers."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from atomrl.quantization.fp8_config import FP8Config
from atomrl.quantization.weight_quantizer import (
    WeightQuantizer,
    _round_scale_pow2,
    fp8_e4m3_max,
)

logger = logging.getLogger(__name__)


def _leading_prod(shape: tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p


class _FP8W8A8Linear(nn.Module):
    """Linear layer emulating blockwise FP8 W8A8 via quantize → matmul → dequant."""

    def __init__(
        self,
        linear: nn.Linear,
        weight_quantizer: WeightQuantizer,
        fp8_config: FP8Config,
        activation_block_size: int = 128,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.activation_block_size = activation_block_size
        self._fp8_config = fp8_config
        if linear.in_features % activation_block_size != 0:
            raise ValueError(
                f"in_features={linear.in_features} must be divisible by "
                f"activation_block_size={activation_block_size}"
            )
        w = linear.weight.data.to(torch.float32)
        self.bias = (
            nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        )
        q_w, s_w = weight_quantizer.quantize_tensor(w, block_size=activation_block_size)
        self.register_buffer("weight_fp8", q_w)
        self.register_buffer("weight_scale", s_w.to(torch.bfloat16))
        self._wq = weight_quantizer

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.to(torch.float32)
        *batch_dims, last = x32.shape
        leading = _leading_prod(tuple(batch_dims)) if batch_dims else 1
        flat = x32.reshape(leading, last)
        blk = self.activation_block_size
        if last % blk != 0:
            raise RuntimeError(f"Token hidden size {last} not divisible by {blk}")

        blocks = flat.unfold(-1, blk, blk)
        amax = blocks.abs().amax(dim=-1)
        fp8_max = blocks.new_tensor(fp8_e4m3_max())

        x_scales = amax / fp8_max
        x_scales = torch.clamp(x_scales, min=torch.finfo(amax.dtype).tiny)
        if self._fp8_config.use_activation_pow2_scale:
            x_scales = _round_scale_pow2(x_scales)

        x_blocks_f = (blocks / x_scales.unsqueeze(-1)).to(torch.float32)
        x_dq = (x_blocks_f * x_scales.unsqueeze(-1)).reshape(leading, last)

        w_dq = self._wq.dequantize_tensor(
            self.weight_fp8,
            self.weight_scale.to(self.weight_fp8.device),
            block_size=blk,
        ).to(x_dq.dtype)
        y = F.linear(
            x_dq, w_dq, self.bias.to(x_dq.dtype) if self.bias is not None else None
        )
        return y.to(x.dtype)


class FP8RolloutQuantizer:
    """Installs blockwise FP8 W8A8 linear replacements for rollout inference."""

    def __init__(self, config: FP8Config) -> None:
        self._config = config
        self._weight_quantizer = WeightQuantizer(config)
        self._originals: dict[str, nn.Linear] = {}
        self._patched: dict[str, _FP8W8A8Linear] = {}

    def should_skip_layer(self, name: str, layer_idx: int, total_layers: int) -> bool:
        """Keep first/last transformer blocks in BF16 when configured."""
        _ = name
        n_first = self._config.num_first_layers_in_bf16
        n_last = self._config.num_last_layers_in_bf16
        if layer_idx < n_first:
            return True
        return bool(total_layers > 0 and layer_idx >= total_layers - n_last)

    def quantize_model(self, model: nn.Module) -> None:
        """Replace eligible ``nn.Linear`` layers with FP8 W8A8 modules."""
        if not self._config.is_fp8_enabled():
            logger.info("FP8 disabled in FP8Config; skipping quantize_model.")
            return

        if self._config.use_deep_gemm:
            try:
                import deep_gemm  # noqa: F401

                logger.debug(
                    "deep_gemm import succeeded; fused FP8 matmul may be wired externally."
                )
            except ImportError:
                logger.warning(
                    "use_deep_gemm=True but deep_gemm is not installed; using torch reference path."
                )

        if self._config.recipe != "blockwise":
            logger.warning(
                "Only blockwise recipe is implemented; got %s", self._config.recipe
            )

        linears: list[tuple[str, nn.Linear]] = [
            (n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)
        ]
        total = len(linears)
        wq = self._weight_quantizer
        for layer_idx, (name, module) in enumerate(linears):
            if self.should_skip_layer(name, layer_idx, total):
                continue
            if module.in_features % 128 != 0:
                logger.debug("Skip %s: in_features=%s", name, module.in_features)
                continue
            try:
                replacement = _FP8W8A8Linear(
                    module, wq, self._config, activation_block_size=128
                )
            except (
                RuntimeError
            ) as exc:  # pragma: no cover - defensive for exotic layers
                logger.warning("Failed to FP8-wrap %s: %s", name, exc)
                continue
            self._originals[name] = module
            self._patched[name] = replacement
            _replace_child(model, name, replacement)

        logger.info("FP8 rollout: patched %d linear layers.", len(self._patched))

    def restore_model(self, model: nn.Module) -> None:
        """Restore original linear modules if this quantizer patched them."""
        for name, orig in self._originals.items():
            _replace_child(model, name, orig)
        self._originals.clear()
        self._patched.clear()


def _replace_child(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)
