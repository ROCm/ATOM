# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Allocation-conscious CPU Adam updates over parameter chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch


class AdamState(TypedDict):
    """Standard eager PyTorch Adam state for one parameter."""

    step: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


@dataclass(frozen=True)
class AdamChunkOptions:
    """Adam options shared by every chunk in a parameter group."""

    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    maximize: bool
    decoupled_weight_decay: bool


def _require_cpu_float32(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype float32")


def _validate_options(options: AdamChunkOptions) -> None:
    for name in ("lr", "eps", "weight_decay"):
        value = getattr(options, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name in ("beta1", "beta2"):
        value = getattr(options, name)
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be finite and in [0, 1)")


def _contiguous_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    start = (
        tensor.untyped_storage().data_ptr()
        + tensor.storage_offset() * tensor.element_size()
    )
    return start, start + tensor.numel() * tensor.element_size()


def _validate_disjoint_storage(
    tensors: tuple[tuple[torch.Tensor, str], ...],
) -> None:
    ranges = []
    for tensor, name in tensors:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        ranges.append((*_contiguous_byte_range(tensor), name))

    for index, (left_start, left_end, left_name) in enumerate(ranges):
        for right_start, right_end, right_name in ranges[index + 1 :]:
            if left_start < right_end and right_start < left_end:
                raise ValueError(f"{left_name} and {right_name} storage ranges overlap")


def initialize_adam_state(
    parameter: torch.Tensor,
    *,
    moment_dtype: torch.dtype = torch.float32,
) -> AdamState:
    """Create PyTorch-compatible Adam state for a CPU FP32 parameter."""

    _require_cpu_float32(parameter, "parameter")
    if moment_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("moment_dtype must be float32 or bfloat16")
    return {
        "step": torch.tensor(0.0, dtype=torch.float32, device="cpu"),
        "exp_avg": torch.zeros_like(
            parameter,
            dtype=moment_dtype,
            memory_format=torch.preserve_format,
        ),
        "exp_avg_sq": torch.zeros_like(
            parameter,
            dtype=moment_dtype,
            memory_format=torch.preserve_format,
        ),
    }


@torch.no_grad()
def adam_step_chunk_(
    parameter: torch.Tensor,
    gradient_scratch: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    exp_avg_workspace: torch.Tensor | None = None,
    exp_avg_sq_workspace: torch.Tensor | None = None,
    step: int,
    options: AdamChunkOptions,
) -> None:
    """Apply one Adam/AdamW update, reusing the gradient as denominator scratch."""

    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    _validate_options(options)

    tensors = (
        (parameter, "parameter"),
        (gradient_scratch, "gradient_scratch"),
    )
    for tensor, name in tensors:
        _require_cpu_float32(tensor, name)
        if tensor.shape != parameter.shape:
            raise ValueError(f"{name} must have the same shape as parameter")
    for tensor, name in ((exp_avg, "exp_avg"), (exp_avg_sq, "exp_avg_sq")):
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if tensor.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"{name} must have dtype float32 or bfloat16")
        if tensor.shape != parameter.shape:
            raise ValueError(f"{name} must have the same shape as parameter")
    if exp_avg.dtype != exp_avg_sq.dtype:
        raise ValueError("exp_avg and exp_avg_sq must have the same dtype")

    persistent_tensors = tensors + (
        (exp_avg, "exp_avg"),
        (exp_avg_sq, "exp_avg_sq"),
    )
    if exp_avg.dtype == torch.bfloat16:
        if exp_avg_workspace is None or exp_avg_sq_workspace is None:
            raise ValueError("bfloat16 moments require float32 workspaces")
        workspaces = (
            (exp_avg_workspace, "exp_avg_workspace"),
            (exp_avg_sq_workspace, "exp_avg_sq_workspace"),
        )
        for tensor, name in workspaces:
            _require_cpu_float32(tensor, name)
            if tensor.shape != parameter.shape:
                raise ValueError(f"{name} must have the same shape as parameter")
        _validate_disjoint_storage(persistent_tensors + workspaces)
        exp_avg_workspace.copy_(exp_avg)
        exp_avg_sq_workspace.copy_(exp_avg_sq)
        exp_avg_work = exp_avg_workspace
        exp_avg_sq_work = exp_avg_sq_workspace
    else:
        _validate_disjoint_storage(persistent_tensors)
        exp_avg_work = exp_avg
        exp_avg_sq_work = exp_avg_sq

    if options.decoupled_weight_decay:
        parameter.mul_(1 - options.lr * options.weight_decay)

    if options.maximize:
        gradient_scratch.neg_()
    if options.weight_decay and not options.decoupled_weight_decay:
        gradient_scratch.add_(parameter, alpha=options.weight_decay)

    exp_avg_work.lerp_(gradient_scratch, 1 - options.beta1)
    exp_avg_sq_work.mul_(options.beta2).addcmul_(
        gradient_scratch,
        gradient_scratch,
        value=1 - options.beta2,
    )

    bias_correction1 = 1 - options.beta1**step
    bias_correction2 = 1 - options.beta2**step
    gradient_scratch.copy_(exp_avg_sq_work)
    gradient_scratch.sqrt_()
    gradient_scratch.div_(math.sqrt(bias_correction2))
    gradient_scratch.add_(options.eps)
    parameter.addcdiv_(
        exp_avg_work,
        gradient_scratch,
        value=-(options.lr / bias_correction1),
    )
    if exp_avg.dtype == torch.bfloat16:
        exp_avg.copy_(exp_avg_work)
        exp_avg_sq.copy_(exp_avg_sq_work)
