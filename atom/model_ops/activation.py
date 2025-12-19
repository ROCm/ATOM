# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Optional
from torch import nn
import torch.nn.functional as F
from aiter import silu_and_mul
from atom.config import QuantizationConfig

from aiter import (
    QuantType,
    dtypes,
)
from atom.utils import envs
from aiter.ops.triton.fused_mxfp4_quant import (
    fused_reduce_act_mul_and_mxfp4_quant,
)


ATOM_USE_AITER_TRITON_FUSED_SILU_MUL_FP8_QUANT = (
    envs.ATOM_USE_AITER_TRITON_FUSED_SILU_MUL_FP8_QUANT
)
if ATOM_USE_AITER_TRITON_FUSED_SILU_MUL_FP8_QUANT:
    from aiter.ops.triton.fused_fp8_quant import (
        fused_silu_mul_fp8_per_tensor_static_quant,
    )

if ATOM_USE_AITER_TRITON_FUSED_SILU_MUL_FP8_QUANT:
    import aiter as rocm_aiter

    rocm_aiter_fp8_dtype = rocm_aiter.dtypes.fp8


class SiluAndMul(nn.Module):
    def __init__(
        self,
        quant_config: Optional[QuantizationConfig] = None,
    )->None: 
        super().__init__()
        quant_type = quant_config["quant_type"]
        params_dtype = quant_config["quant_dtype"]
        self.quant_type = quant_type
        self.params_dtype = params_dtype

    def forward_native(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x_scale is not None and ATOM_USE_AITER_TRITON_FUSED_SILU_MUL_FP8_QUANT:
            x = fused_silu_mul_fp8_per_tensor_static_quant(
                x, x_scale, dtype_quant=rocm_aiter_fp8_dtype
            )
            return x, x_scale
        elif x_scale is None and self.quant_type.value == QuantType.per_1x32.value:
            (x, x_scale), _ = fused_reduce_act_mul_and_mxfp4_quant(x, "silu")
            return x, x_scale
        else:
            out = torch.empty(
                [*x.shape[:-1], x.shape[-1] // 2], device=x.device, dtype=x.dtype
            )
            silu_and_mul(out, x)
            return x
