# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Optional
from torch import nn
import torch.nn.functional as F
from aiter import silu_and_mul
from atom.config import QuantizationConfig
from atom.quant_spec import LayerQuantConfig
from aiter.jit.utils.torch_guard import torch_compile_guard

# --- gfx1201 fallback: triton SiLU + Mul (replaces forward_native) ---------
import triton as _triton
import triton.language as _tl


@_triton.jit
def _silu_mul_kernel(
    X_PTR, OUT_PTR,
    stride_x_row, stride_out_row,
    HALF_D: _tl.int32,
    BLOCK_D: _tl.constexpr,
):
    """For each row: out = silu(x[..., :HALF_D]) * x[..., HALF_D:]. Iterates
    over D in BLOCK_D chunks so HALF_D need not be a power of two."""
    row = _tl.program_id(0)
    block_start = _tl.program_id(1) * BLOCK_D
    cols = block_start + _tl.arange(0, BLOCK_D)
    mask = cols < HALF_D
    a = _tl.load(X_PTR + row * stride_x_row + cols, mask=mask, other=0.0).to(_tl.float32)
    b = _tl.load(X_PTR + row * stride_x_row + HALF_D + cols, mask=mask, other=0.0).to(_tl.float32)
    silu_a = a * (1.0 / (1.0 + _tl.exp(-a)))
    out = (silu_a * b).to(OUT_PTR.dtype.element_ty)
    _tl.store(OUT_PTR + row * stride_out_row + cols, out, mask=mask)


def _silu_mul_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton SiLU+Mul. x: [N, 2*HALF_D]; output: [N, HALF_D]. HALF_D can be
    arbitrary (kernel uses masked block iteration)."""
    N, full_d = x.shape
    half = full_d // 2
    out = torch.empty((N, half), dtype=x.dtype, device=x.device)
    BLOCK_D = 1024
    grid = (N, _triton.cdiv(half, BLOCK_D))
    _silu_mul_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        HALF_D=half,
        BLOCK_D=BLOCK_D,
    )
    return out


def _is_gfx1201_act() -> bool:
    if not hasattr(_is_gfx1201_act, "_cached"):
        try:
            _is_gfx1201_act._cached = (
                torch.cuda.get_device_properties(0).gcnArchName or ""
            ).startswith("gfx1201")
        except Exception:
            _is_gfx1201_act._cached = False
    return _is_gfx1201_act._cached


from aiter import (
    QuantType,
)


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
        quant_config: Optional[QuantizationConfig] = None,
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
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # gfx1201 (RDNA4): aiter prebuilt silu_and_mul HIP kernel has no gfx1201
        # code object. Prefer the triton kernel; fall back to torch forward_native
        # if the input HALF_D is not a power of two (triton kernel limitation).
        if _is_gfx1201_act():
            return _silu_mul_triton(x)
        # fp8 quantization
        if x_scale is not None and self.fused_quant:
            from aiter.ops.triton.fused_fp8_quant import (
                fused_silu_mul_fp8_per_tensor_static_quant,
            )
            import aiter as rocm_aiter

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
