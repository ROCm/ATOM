# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 AtomRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code adapted from the verl project
# (https://github.com/verl-project/verl).
# Original: verl/utils/fsdp_utils.py

"""FSDP model/optimizer offload utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _get_device_id() -> int:
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    return local_rank


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/fsdp_utils.py::offload_fsdp2_model_to_cpu
@torch.no_grad()
def offload_fsdp_model_to_cpu(model: nn.Module, empty_cache: bool = True) -> None:
    """Move an FSDP2-wrapped model to CPU."""
    model.cpu()
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/fsdp_utils.py::load_fsdp2_model_to_gpu
@torch.no_grad()
def load_fsdp_model_to_gpu(model: nn.Module) -> None:
    """Move an FSDP2-wrapped model back to GPU."""
    device = _get_device_id()
    model.to(device)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/fsdp_utils.py::offload_fsdp_optimizer
@torch.no_grad()
def offload_fsdp_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Move all optimizer state tensors to CPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/fsdp_utils.py::load_fsdp_optimizer
@torch.no_grad()
def load_fsdp_optimizer(optimizer: torch.optim.Optimizer, device: Any) -> None:
    """Move all optimizer state tensors back onto the compute device.

    Moves each parameter's state to that parameter's own device rather than a
    generic ``"cuda"`` string. This is important for FSDP2: params are DTensors
    living on ``cuda:{local_rank}`` and the Adam state must land on the exact
    same device, otherwise ``optimizer.step()`` raises a cuda-vs-cpu (or
    cuda:i-vs-cuda:j) device mismatch. Using the param device also makes
    CPU<->GPU offload reload symmetric and robust.
    """
    if not optimizer.state:
        return
    want_gpu = (isinstance(device, str) and device in ("cuda", "gpu")) or isinstance(
        device, (int, torch.device)
    )
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            target = param.device if want_gpu else torch.device("cpu")
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device != target:
                    state[key] = value.to(target, non_blocking=True)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/fsdp_utils.py::fsdp2_clip_grad_norm_
def fsdp2_clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach=None,
) -> torch.Tensor:
    """``clip_grad_norm_`` that works with FSDP2 DTensor parameters on CPU."""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(_get_device_id(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
