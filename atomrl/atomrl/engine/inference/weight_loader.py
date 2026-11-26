# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Dynamic weight loading with optional FP8 quantization."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


class _SupportsWeightUpdate(Protocol):
    """Minimal protocol for engines that accept a PyTorch ``state_dict``."""

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None: ...


def _maybe_fp8_quantize_tensor(
    tensor: torch.Tensor,
    fp8_config: dict[str, Any] | None,
) -> torch.Tensor:
    """Optionally cast floating weights to a reduced dtype for FP8-style loads."""
    if fp8_config is None or not fp8_config.get("enabled", False):
        return tensor
    target_dtype_name = str(fp8_config.get("dtype", "float8_e4m3fn")).lower()
    try:
        if hasattr(torch, "float8_e4m3fn") and target_dtype_name in (
            "float8_e4m3fn",
            "fp8",
            "fp8_e4m3",
        ):
            fp8_dtype = torch.float8_e4m3fn
            # dequantize path for load: represent as fp8 storage where supported
            return tensor.to(dtype=fp8_dtype)
        logger.warning(
            "FP8 dtype %s requested but this PyTorch build has no float8_e4m3fn; "
            "keeping original dtype.",
            target_dtype_name,
        )
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - defensive
        logger.warning("FP8 quantization during load failed: %s", exc)
    return tensor


class DynamicWeightLoader:
    """Loads PyTorch checkpoints into inference engines with optional FP8 paths."""

    @staticmethod
    def load_weights(
        engine: _SupportsWeightUpdate,
        state_dict: dict[str, torch.Tensor],
        fp8_config: dict[str, Any] | None = None,
    ) -> None:
        """Load ``state_dict`` into ``engine`` with optional FP8 on-the-fly conversion.

        Parameters
        ----------
        engine:
            Object implementing ``update_weights(state_dict)`` (for example
            :class:`~atomrl.engine.inference.atom_engine.AtomEngine`).
        state_dict:
            PyTorch state dictionary keyed by parameter name.
        fp8_config:
            Optional dict with keys such as ``enabled`` (bool) and ``dtype`` (str).
        """
        if fp8_config and fp8_config.get("enabled"):
            converted: dict[str, torch.Tensor] = {}
            for name, tensor in state_dict.items():
                if tensor.dtype.is_floating_point:
                    converted[name] = _maybe_fp8_quantize_tensor(tensor, fp8_config)
                else:
                    converted[name] = tensor
            state_dict = converted
            logger.info(
                "DynamicWeightLoader: applied FP8 conversion for %d tensors.",
                len(state_dict),
            )
        engine.update_weights(state_dict)
