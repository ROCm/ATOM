# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 KV cache quantization helpers (scale recalibration per RL step)."""

from __future__ import annotations

import logging

import torch
from torch import nn

from atom import quant_spec as atom_quant_spec
from atomrl.quantization.fp8_config import FP8Config
from atomrl.quantization.weight_quantizer import fp8_e4m3_max

logger = logging.getLogger(__name__)


class FP8KVCacheQuantizer:
    """Maintains per-step Q/K/V FP8 scales for inference KV caches."""

    def __init__(self, config: FP8Config) -> None:
        self._config = config
        self._enabled = False
        self._atom_spec = atom_quant_spec

    def recalibrate_scales(self, model: nn.Module) -> None:
        """Recompute Q/K/V FP8 scales from current weights (call each RL step)."""
        if not self._config.is_fp8_enabled():
            logger.debug("FP8 disabled; skipping KV scale recalibration.")
            return

        fp8_max = float(fp8_e4m3_max())
        updated = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            short = name.split(".")[-1]
            if short not in {"q_proj", "k_proj", "v_proj"}:
                continue
            w = module.weight.detach().to(torch.float32)
            amax = w.abs().max().clamp(min=torch.finfo(torch.float32).tiny)
            scale = (amax / fp8_max).to(torch.float32)
            buf_name = f"_atomrl_fp8_kv_scale_{short}"
            if hasattr(module, buf_name):
                getattr(module, buf_name).copy_(
                    scale.to(getattr(module, buf_name).device)
                )
            else:
                module.register_buffer(buf_name, scale.to(module.weight.device))
            updated += 1

        if updated == 0:
            logger.warning(
                "recalibrate_scales: no q_proj/k_proj/v_proj Linear layers found on model."
            )
        else:
            logger.debug("recalibrate_scales: updated %d projection layers.", updated)

    def enable(self, model: nn.Module) -> None:
        """Mark the model as using FP8 KV caches and ensure scale buffers exist."""
        if not self._config.is_fp8_enabled():
            logger.info("FP8 disabled; FP8KVCacheQuantizer.enable is a no-op.")
            return
        model._atomrl_fp8_kv_cache_enabled = True
        self._enabled = True
        self.recalibrate_scales(model)
        logger.debug(
            "ATOM engine may consume the recalibrated _atomrl_fp8_kv_scale_* buffers."
        )

    @property
    def enabled(self) -> bool:
        return self._enabled
