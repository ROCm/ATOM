# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import List, Tuple, Optional, Union
import torch
from torch import Tensor
from torch.overrides import (
    has_torch_function_unary,
    handle_torch_function,
)
from atom.config import QuantizationConfig
from torch import nn
from aiter import (
    rmsnorm2d_fwd,
    rmsnorm2d_fwd_with_add,
    rms_norm,
    layernorm2d_fwd,
    layernorm2d_fwd_with_add,
)
from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.ops.triton.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad
from aiter.jit.utils.torch_guard import torch_compile_guard

from aiter import (
    QuantType,
    dtypes,
)


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(silu, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.silu_(input)
    return torch._C._nn.silu(input)


@torch_compile_guard()
def rmsnorm2d_fwd_(
    x: torch.Tensor, weight: torch.Tensor, eps: float, dim: int
) -> torch.Tensor:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    return rmsnorm2d_fwd(x, weight, eps).view(ori_shape)


@torch_compile_guard()
def rmsnorm2d_fwd_with_add_(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, eps: float, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    rmsnorm2d_fwd_with_add(out, x, residual, residual_out, weight, eps)
    return out.view(ori_shape), residual_out.view(ori_shape)


def fused_rmsnorm_pad_fake_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    x_pad_to_multiple: int = 0,
) -> torch.Tensor:
    M, N = x.shape
    N_out = (N + x_pad_to_multiple - 1) // x_pad_to_multiple * x_pad_to_multiple
    out = torch.empty((M, N_out), dtype=x.dtype, device=x.device)
    return out


@torch_compile_guard(gen_fake=fused_rmsnorm_pad_fake_tensors)
def fused_rmsnorm_pad_(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    x_pad_to_multiple: int = 0,
) -> torch.Tensor:
    return fused_add_rmsnorm_pad(x, weight, epsilon, None, x_pad_to_multiple)


def fused_add_rmsnorm_pad_fake_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor,
    x_pad_to_multiple: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    N_out = (N + x_pad_to_multiple - 1) // x_pad_to_multiple * x_pad_to_multiple
    out = torch.empty((M, N_out), dtype=x.dtype, device=x.device)
    res_out = torch.empty((M, N), dtype=res.dtype, device=res.device)
    return out, res_out


@torch_compile_guard(gen_fake=fused_add_rmsnorm_pad_fake_tensors)
def fused_add_rmsnorm_pad_(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor,
    x_pad_to_multiple: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return fused_add_rmsnorm_pad(x, weight, epsilon, res, x_pad_to_multiple)


def mxfp4_rms_quant_fuse_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    shuffle: bool = False,
    res1: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    M, N = x.shape
    out = torch.empty((M, N // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
    MXFP4_QUANT_BLOCK_SIZE = 32
    SCALE_N_valid = (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
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

    if res1 is None:
        return (out, scale, None)
    else:
        res = torch.empty_like(res1)
        return (out, scale, res)


# It's important to use mutates_args=[] to avoid functionized_v2 op generation
@torch_compile_guard(gen_fake=mxfp4_rms_quant_fuse_fake, mutates_args=[])
def mxfp4_rms_quant_fuse(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    shuffle: bool = False,
    res1: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fused_mxfp4_quant import (
        fused_rms_mxfp4_quant,
    )

    if res1 is None:
        (x, x_scale), _, _, _ = fused_rms_mxfp4_quant(x, weight, eps, shuffle=True)
        return (x, x_scale, None)
    else:
        (x, x_scale), _, _, residual = fused_rms_mxfp4_quant(
            x, weight, eps, shuffle=True, res1=res1
        )
        return (x, x_scale, residual)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        x_pad_to_multiple: int = 0,
        fused_allreduce: bool = False,
        fused_quant: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.x_pad_to_multiple = x_pad_to_multiple
        self.fused_allreduce = fused_allreduce
        self.use_fused_quant = fused_quant
        self.tp_size = get_tensor_model_parallel_world_size()

        if quant_config is None:
            quant_config = QuantizationConfig()
        quant_type = quant_config["quant_type"]
        params_dtype = quant_config["quant_dtype"]
        self.quant_type = quant_type
        self.params_dtype = params_dtype

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.x_pad_to_multiple > 0:
            assert (
                not self.fused_allreduce
            ), "fused_allreduce_rmsnorm is not supported with rms_norm padding!"
            if residual is None:
                x = fused_rmsnorm_pad_(x, self.weight, self.eps, self.x_pad_to_multiple)
                return x
            else:
                x, residual = fused_add_rmsnorm_pad_(
                    x, self.weight, self.eps, residual, self.x_pad_to_multiple
                )
                return x, residual
        if self.fused_allreduce and self.tp_size > 1:
            assert (
                residual is not None
            ), "fused_allreduce_rmsnorm requires residual input!"
            x, residual = tensor_model_parallel_fused_allreduce_rmsnorm(
                x,
                residual,
                self.weight,
                self.eps,
            )
            return x, residual
        else:
            if x_scale is not None and self.use_fused_quant:
                from aiter.ops.triton.fused_fp8_quant import (
                    fused_rms_fp8_per_tensor_static_quant,
                )
                import aiter as rocm_aiter

                rocm_aiter_fp8_dtype = rocm_aiter.dtypes.fp8

                # static FP8 quantization
                if residual is None:
                    x, _, _, _ = fused_rms_fp8_per_tensor_static_quant(
                        x,
                        self.weight,
                        self.eps,
                        x_scale,
                        None,
                        None,
                        self.eps,
                        dtype_quant=rocm_aiter_fp8_dtype,
                        res1=None,
                    )
                    return (x, x_scale)
                else:
                    x, _, _, residual = fused_rms_fp8_per_tensor_static_quant(
                        x,
                        self.weight,
                        self.eps,
                        x_scale,
                        None,
                        None,
                        self.eps,
                        dtype_quant=rocm_aiter_fp8_dtype,
                        res1=residual,
                    )
                    return (x, x_scale), residual
            elif self.use_fused_quant and (
                x_scale is None and self.quant_type.value == QuantType.per_1x32.value
            ):
                if residual is None:
                    x, x_scale, _ = mxfp4_rms_quant_fuse(
                        x, self.weight, self.eps, shuffle=True
                    )
                    return x, x_scale
                else:
                    x, x_scale, residual = mxfp4_rms_quant_fuse(
                        x, self.weight, self.eps, shuffle=True, res1=residual
                    )
                    return (x, x_scale), residual
            else:
                if residual is None:
                    # return rmsnorm2d_fwd(x, self.weight, self.eps).view(ori_shape)
                    x = rmsnorm2d_fwd_(x, self.weight, self.eps, self.dim)
                    return x
                else:
                    # return self.add_rms_forward(x, residual)
                    x, residual = rmsnorm2d_fwd_with_add_(
                        x, self.weight, residual, self.eps, self.dim
                    )
                    return x, residual


class RMSNormGated(nn.Module):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize RMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
                        group_size=None is equivalent to group_size=hidden_size
                        (i.e. there's only 1 group).
            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
                              If False and z is provided: out = norm(x * silu(z))
            dtype: Data type for parameters
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Native PyTorch implementation of RMS normalization with gating.

        Args:
            x: Input tensor
            z: Optional gating tensor

        Returns:
            Normalized (and optionally gated) tensor

        If z is not None:
            - norm_before_gate=True: out = norm(x) * silu(z)
            - norm_before_gate=False: out = norm(x * silu(z))
        """
        # Apply gating before normalization if needed
        if z is not None and not self.norm_before_gate:
            x = x * silu(z)

        # RMS Normalization
        if self.group_size is None:
            # Standard RMS norm across the last dimension
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps)
            out = x_normed * self.weight
        else:
            # Group RMS norm
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=self.group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + self.eps)
            out = rearrange(x_normed, "... g d -> ... (g d)") * self.weight

        # Apply gating after normalization if needed
        if z is not None and self.norm_before_gate:
            out = out * silu(z)

        return out

    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        if torch.compiler.is_compiling():
            return self.forward_native(x, z)
        return self.forward_native(x, z)

        # from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

        # return rmsnorm_fn(
        #     x,
        #     self.weight,
        #     self.bias,
        #     z=z,
        #     eps=self.eps,
        #     group_size=self.group_size,
        #     norm_before_gate=self.norm_before_gate,
        # )
    
    def forward(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        return self.forward_cuda(x, z)

class GemmaRMSNorm(nn.Module):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def forward_static(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        if residual is not None:
            x = (
                x.float() + residual.float()
                if orig_dtype == torch.float16
                else x + residual
            )
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(self.weight.data, self.variance_epsilon, x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self.forward_static = torch.compile(  # type: ignore
                self.forward_static
            )
            self._is_compiled = True
        return self.forward_native(x, residual)
    
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)


@torch_compile_guard()
def layernorm2d_fwd_(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, dim: int
) -> torch.Tensor:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    return layernorm2d_fwd(x, weight, bias, eps).view(ori_shape)


@torch_compile_guard()
def layernorm2d_fwd_with_add_(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    layernorm2d_fwd_with_add(out, x, residual, residual_out, weight, bias, eps)
    return out.view(ori_shape), residual_out.view(ori_shape)


class LayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return layernorm2d_fwd_(x, self.weight, self.bias, self.eps, self.dim)
        else:
            return layernorm2d_fwd_with_add_(
                x, self.weight, residual, self.bias, self.eps, self.dim
            )
