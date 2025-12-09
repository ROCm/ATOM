# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import nn
import torch.nn.functional as F
from aiter import silu_and_mul


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty([*x.shape[:-1], x.shape[-1] // 2], device=x.device, dtype=x.dtype)
        silu_and_mul(out, x)
        return out

