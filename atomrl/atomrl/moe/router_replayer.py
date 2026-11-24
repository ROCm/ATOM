# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Replay recorded MoE router logits during training forward passes."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

from atomrl.moe.moe_utils import _extract_router_logits, iter_moe_modules

logger = logging.getLogger(__name__)


def _broadcast_like(recorded: Tensor, reference: Tensor) -> Tensor:
    r = recorded.to(device=reference.device, dtype=reference.dtype)
    if r.shape == reference.shape:
        return r
    if r.numel() == reference.numel():
        return r.reshape(reference.shape)
    logger.warning(
        "RouterReplayer: shape mismatch recorded=%s reference=%s; returning reference.",
        tuple(r.shape),
        tuple(reference.shape),
    )
    return reference


def _replace_router_logits(output: Any, recorded: Tensor) -> Any:
    if isinstance(output, Tensor):
        return _broadcast_like(recorded, output)
    if isinstance(output, tuple):
        logits = _extract_router_logits(output)
        if logits is None:
            return output
        new_logits = _broadcast_like(recorded, logits)
        parts = list(output)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] is logits:
                parts[i] = new_logits
                return tuple(parts)
        parts[-1] = new_logits
        return tuple(parts)
    return output


class RouterReplayer:
    """Replace MoE router logits with recorded distributions via forward hooks."""

    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._distributions: dict[int, Tensor] = {}

    def install_hooks(self, model: nn.Module, distributions: dict[int, Tensor]) -> None:
        """Install hooks that overwrite router logits with ``distributions``."""
        self.remove_hooks()
        self._distributions = {
            int(k): v.float().cpu() for k, v in distributions.items()
        }

        def make_hook(layer_idx: int) -> Any:
            def _hook(_module: nn.Module, _inp: Any, out: Any) -> Any:
                recorded = self._distributions.get(layer_idx)
                if recorded is None:
                    return out
                return _replace_router_logits(out, recorded)

            return _hook

        for layer_idx, _name, module in iter_moe_modules(model):
            h = module.register_forward_hook(make_hook(layer_idx))
            self._handles.append(h)
        if not self._handles:
            logger.warning("RouterReplayer: no MoE modules detected on model.")

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._distributions.clear()
