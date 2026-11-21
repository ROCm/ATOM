# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Adapted from TorchSpec's training optimizer and scheduler. The upstream
# attribution and license must be retained; see atomrl/THIRD_PARTY_NOTICES.md.

"""BF16Optimizer: FP32 master-copy optimizer for BF16 model training.

Maintains FP32 copies of all trainable parameters. Gradients computed in BF16
are upcast to FP32 before AdamW update, then the updated FP32 params are
copied back to the BF16 model. This prevents precision truncation when
gradients are small (e.g. forward KL loss ~1e-5 vs BF16 ULP ~6e-5).

Ported from TorchSpec (torchspec/training/optimizer.py + lr_scheduler.py).
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

DecayStyle = Literal["constant", "linear", "cosine", "WSD"]
WSDDecayStyle = Literal["linear", "cosine", "exponential", "minus_sqrt"]


class LRSchedulerWithWarmup(LRScheduler):
    """LR scheduler with linear warmup and multiple decay styles.

    Supports: constant, linear, cosine, WSD (Warmup-Stable-Decay).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        init_lr: float = 0.0,
        decay_style: DecayStyle = "cosine",
        wsd_decay_steps: int | None = None,
        wsd_decay_style: WSDDecayStyle | None = None,
        last_epoch: int = -1,
    ) -> None:
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.init_lr = float(init_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.decay_style = decay_style
        self.wsd_decay_steps = wsd_decay_steps
        self.wsd_decay_style = wsd_decay_style

        if self.decay_style == "WSD":
            assert self.wsd_decay_steps is not None, "WSD requires wsd_decay_steps"

        super().__init__(optimizer, last_epoch)

    def _get_lr_for_group(self, param_group: dict) -> float:
        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)
        step = self.last_epoch

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.init_lr + (max_lr - self.init_lr) * step / self.warmup_steps

        if self.decay_style == "constant":
            return max_lr

        if step > self.total_steps:
            return min_lr

        num_steps_ = step - self.warmup_steps
        decay_steps_ = self.total_steps - self.warmup_steps
        decay_ratio = max(
            0.0, min(1.0, float(num_steps_) / float(max(decay_steps_, 1)))
        )
        delta_lr = max_lr - min_lr

        if self.decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.decay_style == "WSD":
            coeff = self._compute_wsd_coeff(step)
        else:
            raise ValueError(f"Unknown decay style: {self.decay_style}")

        return min_lr + coeff * delta_lr

    def _compute_wsd_coeff(self, step: int) -> float:
        wsd_anneal_start = self.total_steps - self.wsd_decay_steps
        if step <= wsd_anneal_start:
            return 1.0

        wsd_ratio = float(step - wsd_anneal_start) / float(self.wsd_decay_steps)

        if self.wsd_decay_style == "linear":
            return 1.0 - wsd_ratio
        elif self.wsd_decay_style == "cosine":
            return 0.5 * (math.cos(math.pi * wsd_ratio) + 1.0)
        elif self.wsd_decay_style == "exponential":
            return (2.0 * math.pow(0.5, wsd_ratio)) - 1.0
        elif self.wsd_decay_style == "minus_sqrt":
            return 1.0 - math.sqrt(wsd_ratio)
        else:
            raise ValueError(f"Unknown WSD decay style: {self.wsd_decay_style}")

    def get_lr(self) -> list[float]:
        return [self._get_lr_for_group(group) for group in self.optimizer.param_groups]


class BF16Optimizer:
    """FP32 master-copy optimizer for BF16 models.

    Maintains FP32 copies of all trainable parameters. On each step:
    1. Zero NaN gradients on BF16 model params
    2. Copy BF16 gradients to FP32 buffers
    3. Clip FP32 gradients (max_grad_norm)
    4. AdamW step on FP32 params
    5. LR scheduler step
    6. Copy FP32 params back to BF16 model
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        total_steps: int = 800_000,
        warmup_ratio: float = 0.015,
        decay_style: str = "cosine",
        min_lr: float = 0.0,
        wsd_decay_ratio: float = 0.2,
        wsd_decay_style: str = "cosine",
    ) -> None:
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm

        self.fp32_params = [p.detach().clone().float() for p in self.model_params]
        self.fp32_grads = [torch.zeros_like(mp) for mp in self.fp32_params]
        for mp in self.fp32_params:
            mp.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            self.fp32_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        warmup_steps = int(warmup_ratio * total_steps)
        wsd_decay_steps = None
        if decay_style == "WSD":
            wsd_decay_steps = int(wsd_decay_ratio * total_steps)

        self.scheduler = LRSchedulerWithWarmup(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_style=decay_style,
            min_lr=min_lr,
            wsd_decay_steps=wsd_decay_steps,
            wsd_decay_style=wsd_decay_style if decay_style == "WSD" else None,
        )

        num_params = sum(p.numel() for p in self.model_params)
        logger.info(
            "BF16Optimizer: %d trainable params, FP32 master copy, "
            "max_grad_norm=%.2f, lr=%.2e, decay=%s, warmup=%d/%d steps",
            num_params,
            max_grad_norm,
            lr,
            decay_style,
            warmup_steps,
            total_steps,
        )

    def step(self) -> torch.Tensor:
        """Perform one optimizer step. Returns grad_norm for logging."""
        # 1. Zero NaN gradients on BF16 model params
        with torch.no_grad():
            nan_count = 0
            for p in self.model_params:
                if p.grad is not None and torch.isnan(p.grad).any():
                    p.grad.zero_()
                    nan_count += 1
            if nan_count > 0:
                logger.warning(
                    "BF16Optimizer: zeroed NaN grads in %d params", nan_count
                )

        # 2. Copy BF16 grads → FP32
        with torch.no_grad():
            for p, mp, g in zip(self.model_params, self.fp32_params, self.fp32_grads):
                if p.grad is not None:
                    g.copy_(p.grad)
                    mp.grad = g
                else:
                    mp.grad = None

        # 3. Clip FP32 gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)

        # 4. AdamW step (skip if grad_norm is zero or non-finite)
        if torch.isfinite(grad_norm) and grad_norm > 0.0:
            self.optimizer.step()
        elif not torch.isfinite(grad_norm):
            logger.warning("BF16Optimizer: skipping step (grad_norm=%s)", grad_norm)

        # 5. Scheduler step
        self.optimizer.zero_grad()
        self.scheduler.step()

        # 6. Copy FP32 → BF16
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None

        return grad_norm

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.model_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def sync_fp32_params_from_model(self) -> None:
        """Re-sync FP32 master copies from model. Call after loading a checkpoint."""
        with torch.no_grad():
            for mp, p in zip(self.fp32_params, self.model_params):
                mp.data.copy_(p.data.float())

    def get_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups
