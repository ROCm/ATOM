# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/utils/torch_functional.py

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
# Original: verl/utils/torch_functional.py

"""Learning-rate schedulers with linear warmup."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::get_cosine_schedule_with_warmup
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Cosine schedule with linear warmup.

    After warmup, the LR follows a cosine curve from the initial value down to
    ``min_lr_ratio * initial_lr``.
    """
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    assert 0.0 <= min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step: int) -> float:
        # +1 so the first optimizer step uses base_lr * 1/num_warmup (e.g. 1e-7),
        # matching verl's warmup (step1=1e-7 ... step10=1e-6) instead of starting
        # at 0. Compensates for Lumen calling scheduler.step() AFTER optimizer.step().
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::get_constant_schedule_with_warmup
def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Constant LR schedule with a linear warmup phase."""

    def lr_lambda(current_step: int) -> float:
        # +1 so the first optimizer step uses base_lr * 1/num_warmup (e.g. 1e-7),
        # matching verl's warmup (step1=1e-7 ... step10=1e-6) instead of starting
        # at 0. Compensates for Lumen calling scheduler.step() AFTER optimizer.step().
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)
