# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.


import torch
import torch.nn.functional as F
from aiter import (
    QuantType,
    silu_and_mul,
)
from aiter.jit.utils.torch_guard import torch_compile_guard
from torch import nn

from atom.config import QuantizationConfig
from atom.quant_spec import LayerQuantConfig


def mxfp4_act_mul_quant_fuse_fake(
    x: torch.Tensor,
    shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N1 = x.shape
    N_half = N1 // 2
    out = torch.empty((M, N_half // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
    MXFP4_QUANT_BLOCK_SIZE = 32
    SCALE_N_valid = (N_half + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    use_scale_shuffle_padding = shuffle
    if use_scale_shuffle_padding:
        SCALE_M = ((M + 255) // 256) * 256
        SCALE_N = ((SCALE_N_valid + 7) // 8) * 8
    else:
        SCALE_M = M
        SCALE_N = SCALE_N_valid
    scale = torch.empty(
        (SCALE_M, SCALE_N),
        dtype=torch.float8_e8m0fnu,
        device=x.device,
    )

    return out, scale


# It's important to use mutates_args=[] to avoid functionized_v2 op generation
@torch_compile_guard(gen_fake=mxfp4_act_mul_quant_fuse_fake, mutates_args=[])
def mxfp4_act_mul_quant_fuse(
    x: torch.Tensor,
    shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fused_mxfp4_quant import (
        fused_reduce_act_mul_and_mxfp4_quant,
    )

    (x, x_scale), _ = fused_reduce_act_mul_and_mxfp4_quant(x, "silu", shuffle=shuffle)

    return x, x_scale


class SiluAndMul(nn.Module):
    def __init__(
        self,
        fused_quant: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fused_quant = fused_quant
        layer_quant_config = (
            LayerQuantConfig()
            if quant_config is None
            else quant_config.get_layer_quant_config(prefix)
        )

        quant_type = layer_quant_config.quant_type
        params_dtype = layer_quant_config.quant_dtype
        self.quant_type = quant_type
        self.params_dtype = params_dtype

    def forward_native(
        self, x: torch.Tensor, x_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(
        self, x: torch.Tensor, x_scale: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # fp8 quantization
        if x_scale is not None and self.fused_quant:
            import aiter as rocm_aiter
            from aiter.ops.triton.fused_fp8_quant import (
                fused_silu_mul_fp8_per_tensor_static_quant,
            )

            rocm_aiter_fp8_dtype = rocm_aiter.dtypes.fp8

            x = fused_silu_mul_fp8_per_tensor_static_quant(
                x, x_scale, dtype_quant=rocm_aiter_fp8_dtype
            )
            return x, x_scale
        # mxfp4 quantization
        elif (
            x_scale is None
            and self.fused_quant
            and self.quant_type.value == QuantType.per_1x32.value
        ):
            return mxfp4_act_mul_quant_fuse(x, shuffle=True)
        else:
            out = torch.empty(
                [*x.shape[:-1], x.shape[-1] // 2], device=x.device, dtype=x.dtype
            )
            silu_and_mul(out, x)
            return out
