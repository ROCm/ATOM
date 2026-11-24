# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Capture MoE router logits during inference using temporary forward hooks."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

from atomrl.moe.moe_utils import _extract_router_logits, iter_moe_modules

logger = logging.getLogger(__name__)


class RouterRecorder:
    """Install forward hooks to record router logits keyed by MoE layer index."""

    def __init__(self) -> None:
        self._distributions: dict[int, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def install_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on detected MoE layers."""

        def make_hook(layer_idx: int) -> Any:
            def _hook(_module: nn.Module, _inp: Any, out: Any) -> None:
                logits = _extract_router_logits(out)
                if logits is None:
                    logger.debug(
                        "RouterRecorder: could not parse router output on layer %s.",
                        layer_idx,
                    )
                    return
                self._distributions[layer_idx] = logits.detach().float().cpu()

            return _hook

        self.remove_hooks()
        for layer_idx, _name, module in iter_moe_modules(model):
            h = module.register_forward_hook(make_hook(layer_idx))
            self._handles.append(h)
        if not self._handles:
            logger.warning("RouterRecorder: no MoE modules detected on model.")

    def remove_hooks(self) -> None:
        """Remove all hooks installed by this recorder."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_distributions(self) -> dict[int, Tensor]:
        """Return recorded router logits (CPU float32) keyed by layer index."""
        return dict(self._distributions)

    def clear(self) -> None:
        """Drop captured logits without removing hooks."""
        self._distributions.clear()

    @contextmanager
    def recording(self, model: nn.Module) -> Iterator[RouterRecorder]:
        """Context-manager style install/remove around a forward region."""
        self.install_hooks(model)
        try:
            yield self
        finally:
            self.remove_hooks()
