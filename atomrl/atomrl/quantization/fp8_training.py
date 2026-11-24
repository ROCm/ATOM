# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Training-time FP8 integration via Lumen ``quantize`` APIs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from torch import nn
from torch.optim import Optimizer

from atomrl.core.config import AtomRLConfig, QuantizationConfig
from atomrl.quantization.fp8_config import FP8Config

logger = logging.getLogger(__name__)


def _try_lumen_quantize() -> tuple[Any, Any] | tuple[None, None]:
    try:
        from atomrl.lumen import (
            quantize as lumen_quantize,  # type: ignore[import-not-found]
        )

        return lumen_quantize, getattr(lumen_quantize, "QuantConfig", None)
    except ImportError:
        return None, None


def _try_lumen_reset() -> Callable[[nn.Module], None] | None:
    try:
        from atomrl.lumen.quantize import (
            reset_fp8_state as lumen_reset_fp8_state,  # type: ignore[import-not-found]
        )

        return lumen_reset_fp8_state
    except ImportError:
        pass
    try:
        from atomrl.lumen import quantize as lq  # type: ignore[import-not-found]

        return getattr(lq, "reset_fp8_state", None)
    except ImportError:
        return None


class FP8TrainingManager:
    """Bridges AtomRL quantization config to ``atomrl.lumen.quantize`` training hooks."""

    def __init__(self, config: QuantizationConfig | AtomRLConfig) -> None:
        if isinstance(config, AtomRLConfig):
            self._quant = config.quantization
        elif isinstance(config, QuantizationConfig):
            self._quant = config
        else:
            raise TypeError(
                "FP8TrainingManager config must be QuantizationConfig or AtomRLConfig, "
                f"got {type(config)!r}"
            )
        self._fp8 = FP8Config.from_config(self._quant)
        self._weight_cache = self._quant.training.fp8_weight_cache
        self._optimizer_hook_handles: list[Any] = []

    def enable(self, model: nn.Module) -> None:
        """Enable FP8 training kernels on ``model`` via ``atomrl.lumen.quantize.enable``."""
        lumen_quantize, QuantConfigCls = _try_lumen_quantize()
        if lumen_quantize is None or QuantConfigCls is None:
            logger.warning(
                "atomrl.lumen.quantize is not installed; FP8TrainingManager.enable skipped."
            )
            return
        if not self._fp8.is_fp8_enabled():
            logger.info(
                "FP8 not enabled in config; skipping atomrl.lumen.quantize.enable."
            )
            return

        try:
            quant_config = QuantConfigCls(
                enabled=True,
                recipe=self._fp8.recipe,
                use_deep_gemm=self._fp8.use_deep_gemm,
                use_weight_pow2_scale=self._fp8.use_weight_pow2_scale,
                use_activation_pow2_scale=self._fp8.use_activation_pow2_scale,
            )
        except TypeError:
            quant_config = QuantConfigCls()
            for key, val in {
                "enabled": True,
                "recipe": self._fp8.recipe,
                "use_deep_gemm": self._fp8.use_deep_gemm,
                "use_weight_pow2_scale": self._fp8.use_weight_pow2_scale,
                "use_activation_pow2_scale": self._fp8.use_activation_pow2_scale,
            }.items():
                if hasattr(quant_config, key):
                    setattr(quant_config, key, val)

        enable_fn = getattr(lumen_quantize, "enable", None)
        if enable_fn is None:
            logger.error("atomrl.lumen.quantize.enable is missing.")
            return
        enable_fn(model, quant_config)
        logger.info(
            "atomrl.lumen.quantize.enable applied with recipe=%s.", self._fp8.recipe
        )

    def register_optimizer_hooks(self, optimizer: Optimizer) -> None:
        """Register FP8 weight-cache hooks after optimizer steps when enabled."""
        if not self._weight_cache:
            logger.debug("fp8_weight_cache disabled; no optimizer hooks registered.")
            return
        if not self._fp8.is_fp8_enabled():
            return

        def _post_step(
            opt: Optimizer,
        ) -> None:  # pragma: no cover - runtime side effect
            _ = opt
            logger.debug("FP8 weight cache post-step hook fired.")

        register = getattr(optimizer, "register_step_post_hook", None)
        if register is None:
            logger.warning(
                "Optimizer does not support register_step_post_hook; FP8 cache hook not installed."
            )
            return
        handle = register(_post_step)
        self._optimizer_hook_handles.append(handle)
        logger.info(
            "Registered FP8 weight-cache post-step hook on %s.",
            type(optimizer).__name__,
        )

    def reset_fp8_state(self, model: nn.Module) -> None:
        """Reset FP8 runtime state on ``model`` after weight swaps or optimizer resharding."""
        reset_fn = _try_lumen_reset()
        if reset_fn is not None:
            reset_fn(model)
            logger.debug("reset_fp8_state: delegated to atomrl.lumen.")
            return
        local = False
        for module in model.modules():
            reset = getattr(module, "reset_fp8_state", None)
            if callable(reset):
                reset()
                local = True
        if not local:
            logger.warning(
                "reset_fp8_state: atomrl.lumen.quantize.reset_fp8_state unavailable and no "
                "module.reset_fp8_state hooks were found."
            )
