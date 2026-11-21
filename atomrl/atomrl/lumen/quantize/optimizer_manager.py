# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Optimizer utilities for FP8 training in Lumen.

FP8 parameter lifecycle management is handled by
:class:`~atomrl.lumen.quantize.ScalingManager`.  This module provides:

- :class:`FP32MasterWeightOptimizer` — optimizer wrapper that maintains
  FP32 master copies of BF16 parameters for numerical stability.
- :func:`get_scaling_manager` — retrieve the :class:`ScalingManager`
  attached to a model (handles FSDP / DDP wrapper unwrapping).

Usage::

    from atomrl.lumen.quantize.optimizer_manager import FP32MasterWeightOptimizer, get_scaling_manager

    optimizer = FP32MasterWeightOptimizer(
        model.parameters(), torch.optim.AdamW,
        lr=1e-4, weight_decay=0.01,
    )
    mgr = get_scaling_manager(model)
    mgr.register_fp8_optimizer_hook(optimizer)
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def get_scaling_manager(model: nn.Module):
    """Retrieve the :class:`~atomrl.lumen.quantize.ScalingManager` attached to *model*.

    Handles FSDP / DDP wrappers by unwrapping ``.module`` attributes.
    Returns ``None`` if no manager is found.
    """
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return getattr(unwrapped, "_scaling_manager", None)


# ---------------------------------------------------------------------------
# FP32 Master Weight Optimizer
# ---------------------------------------------------------------------------


class FP32MasterWeightOptimizer(torch.optim.Optimizer):
    """Optimizer wrapper that maintains FP32 master copies of BF16 parameters.

    Aligns with Megatron-LM's ``shard_fp32_from_float16_groups`` in
    DistributedOptimizer: gradient updates are accumulated in FP32 for
    numerical stability, then synced back to BF16 model parameters after
    each step.

    Data flow per step::

        BF16 model grads ──(upcast)──► FP32 master grads
                                            │
                                        AdamW step (FP32)
                                            │
        BF16 model params ◄──(downcast)── FP32 master params

    This matches Megatron's quantization path where FP32 main params are
    cast to ``model_param.dtype`` (BF16) before FP8 quantization, ensuring
    numerical consistency regardless of ``--fp8-param-gather``.

    The wrapper inherits from :class:`torch.optim.Optimizer` so that
    learning-rate schedulers and hooks work transparently.
    """

    def __init__(self, params, optimizer_cls, **optimizer_kwargs):
        model_params = list(params)
        self._model_params: list[nn.Parameter] = []
        self._fp32_masters: list[torch.Tensor | None] = []

        opt_params: list[torch.Tensor] = []
        for p in model_params:
            self._model_params.append(p)
            if p.requires_grad and p.is_floating_point() and p.dtype != torch.float32:
                fp32 = p.detach().float().clone().requires_grad_(True)
                self._fp32_masters.append(fp32)
                opt_params.append(fp32)
            else:
                self._fp32_masters.append(None)
                opt_params.append(p)

        self._inner = optimizer_cls(opt_params, **optimizer_kwargs)

        # Initialize Optimizer base class so that LR schedulers pass
        # the ``isinstance(optimizer, Optimizer)`` check.
        super().__init__(opt_params, dict(self._inner.defaults))
        # Share param_groups / state with the inner optimizer so that
        # schedulers adjust LR on the actual optimizer.
        self.param_groups = self._inner.param_groups
        self.state = self._inner.state

        n_masters = sum(1 for m in self._fp32_masters if m is not None)
        logger.info(
            "FP32MasterWeightOptimizer: created %d FP32 master copies "
            "(%d params total)",
            n_masters,
            len(self._model_params),
        )

    # ---- Core interface --------------------------------------------------

    def step(self, closure=None):
        """Copy BF16 grads → FP32, step, sync FP32 → BF16 model params."""
        with torch.no_grad():
            for model_p, fp32_p in zip(self._model_params, self._fp32_masters):
                if fp32_p is not None and model_p.grad is not None:
                    if fp32_p.grad is None:
                        fp32_p.grad = model_p.grad.float()
                    else:
                        fp32_p.grad.copy_(model_p.grad)

        loss = self._inner.step(closure=closure)

        with torch.no_grad():
            for model_p, fp32_p in zip(self._model_params, self._fp32_masters):
                if fp32_p is not None:
                    model_p.data.copy_(fp32_p.data)

        return loss

    def zero_grad(self, set_to_none=True):
        self._inner.zero_grad(set_to_none=set_to_none)
        for p in self._model_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    # ---- State persistence -----------------------------------------------

    def state_dict(self):
        sd = self._inner.state_dict()
        sd["_fp32_masters"] = {
            str(i): p.data.cpu()
            for i, p in enumerate(self._fp32_masters)
            if p is not None
        }
        return sd

    def load_state_dict(self, state_dict):
        masters = state_dict.pop("_fp32_masters", {})
        self._inner.load_state_dict(state_dict)
        with torch.no_grad():
            for i_str, data in masters.items():
                i = int(i_str)
                if i < len(self._fp32_masters) and self._fp32_masters[i] is not None:
                    device = self._fp32_masters[i].device
                    self._fp32_masters[i].data.copy_(data.to(device))
                    self._model_params[i].data.copy_(self._fp32_masters[i].data)
        # Re-sync shared references after inner load
        self.param_groups = self._inner.param_groups
        self.state = self._inner.state
