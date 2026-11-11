# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The native final draft RMSNorm with its existing read-only AITER schema."""

import aiter
import torch

from atom.model_ops.layernorm import RMSNorm
from atom.utils import envs
from atom.utils.decorators import mark_trace


class MonoRMSNorm(torch.nn.Module):
    """Keep the loaded norm weight without the outer mutable op declaration."""

    def __init__(self, native):
        super().__init__()
        if (
            type(native) is not RMSNorm
            or native.x_pad_to_multiple
            or native.fused_allreduce
            or native.use_fused_quant
        ):
            raise ValueError("V4 draft output requires an unfused, unpadded RMSNorm")
        self.native = native

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.native, name)

    @mark_trace
    def forward(self, x: torch.Tensor):
        # Reuse the exact call inside native rmsnorm2d_fwd_. That outer op
        # marks both operands mutable; AITER already declares this one read-only.
        output = aiter.rmsnorm2d_fwd(
            x.reshape(-1, self.dim),
            self.weight,
            self.eps,
            use_model_sensitive_rmsnorm=int(envs.ATOM_USE_MODEL_SENSITIVE_RMSNORM),
        )
        return output.view(x.shape)
