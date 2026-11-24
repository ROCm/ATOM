# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

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


def build_training_scheduler(optimizer, policy, total_steps: int):
    """Build a Transformers scheduler from AtomRL's policy fields."""
    from transformers import get_scheduler

    warmup_steps = int(policy.warmup_ratio * total_steps)
    style = policy.lr_decay_style.lower()
    scheduler_kwargs: dict[str, object] = {}
    if style == "wsd":
        decay_type = {
            "minus_sqrt": "1-sqrt",
        }.get(policy.wsd_decay_style, policy.wsd_decay_style)
        if decay_type not in {"linear", "cosine", "1-sqrt"}:
            raise ValueError(
                "WSD decay style must be linear, cosine, or minus_sqrt; "
                f"got {policy.wsd_decay_style!r}"
            )
        scheduler_name = "warmup_stable_decay"
        scheduler_kwargs = {
            "num_decay_steps": int(policy.wsd_decay_ratio * total_steps),
            "decay_type": decay_type,
            "min_lr_ratio": policy.min_lr / policy.learning_rate,
        }
    elif style == "cosine" and policy.min_lr > 0:
        scheduler_name = "cosine_with_min_lr"
        scheduler_kwargs = {"min_lr": policy.min_lr}
    elif style == "constant" and warmup_steps > 0:
        scheduler_name = "constant_with_warmup"
    else:
        scheduler_name = style
    return get_scheduler(
        scheduler_name,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs=scheduler_kwargs,
    )


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
        self._fp32_masters: list[nn.Parameter | None] = []

        opt_params: list[torch.Tensor] = []
        for p in model_params:
            self._model_params.append(p)
            if p.requires_grad and p.is_floating_point() and p.dtype != torch.float32:
                fp32 = nn.Parameter(p.detach().float().clone(), requires_grad=True)
                self._fp32_masters.append(fp32)
                opt_params.append(fp32)
            else:
                self._fp32_masters.append(None)
                opt_params.append(p)

        self._inner = optimizer_cls(opt_params, **optimizer_kwargs)
        self._optimizer_params = opt_params

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

    def _copy_model_grads_to_masters(self) -> None:
        with torch.no_grad():
            for model_p, fp32_p in zip(self._model_params, self._fp32_masters):
                if fp32_p is not None and model_p.grad is not None:
                    if fp32_p.grad is None:
                        fp32_p.grad = model_p.grad.float()
                    else:
                        fp32_p.grad.copy_(model_p.grad)

    def _copy_masters_to_model(self) -> None:
        with torch.no_grad():
            for model_p, fp32_p in zip(self._model_params, self._fp32_masters):
                if fp32_p is not None:
                    model_p.data.copy_(fp32_p.data)

    def step(self, closure=None):
        """Copy BF16 grads → FP32, step, sync FP32 → BF16 model params."""
        self._copy_model_grads_to_masters()
        loss = self._inner.step(closure=closure)
        self._copy_masters_to_model()
        return loss

    def step_with_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        """Clip FP32 gradients, update finite steps, and return their norm."""
        self._copy_model_grads_to_masters()
        params = [
            param
            for group in self._inner.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        if torch.isfinite(grad_norm) and grad_norm > 0:
            self._inner.step()
            self._copy_masters_to_model()
        elif not torch.isfinite(grad_norm):
            logger.warning(
                "FP32MasterWeightOptimizer: skipping non-finite gradient step (%s)",
                grad_norm,
            )
        return grad_norm

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

    def sync_master_params_from_model(self) -> None:
        """Refresh FP32 master parameters after loading model-only weights."""
        with torch.no_grad():
            for model_p, fp32_p in zip(self._model_params, self._fp32_masters):
                if fp32_p is not None:
                    fp32_p.copy_(model_p.float())

    def get_learning_rate(self) -> float:
        """Return the learning rate used by the first parameter group."""
        return float(self.param_groups[0]["lr"])

    @property
    def fp32_params(self) -> list[torch.Tensor]:
        """FP32 masters exposed for legacy AtomRL checkpoint compatibility."""
        return [param for param in self._fp32_masters if param is not None]

    def _master_parameter_model(self, model: nn.Module) -> nn.Module:
        """Expose optimizer parameters under the model's canonical names."""
        names = {id(param): name for name, param in model.named_parameters()}
        proxy = nn.Module()
        for model_param, optimizer_param in zip(
            self._model_params, self._optimizer_params
        ):
            name = names[id(model_param)]
            parent = proxy
            parts = name.split(".")
            for part in parts[:-1]:
                child = parent._modules.get(part)
                if child is None:
                    child = nn.Module()
                    parent.add_module(part, child)
                parent = child
            parent.register_parameter(parts[-1], optimizer_param)
        return proxy

    def distributed_state_dict(self, model, options) -> dict:
        """Serialize optimizer state with model parameter names for FSDP2."""
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        master_model = self._master_parameter_model(model)
        return get_optimizer_state_dict(master_model, self._inner, options=options)

    def load_distributed_state_dict(self, model, state_dict: dict, options) -> None:
        """Restore a model-parameter-named FSDP2 optimizer state."""
        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

        master_model = self._master_parameter_model(model)
        set_optimizer_state_dict(
            master_model,
            self._inner,
            state_dict,
            options=options,
        )
        self.state = self._inner.state
