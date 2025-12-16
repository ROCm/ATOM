# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import List, Tuple, Optional, Union
import torch
from torch import nn
from aiter import (
    rmsnorm2d_fwd,
    rmsnorm2d_fwd_with_add,
    rms_norm,
    layernorm2d_fwd,
    layernorm2d_fwd_with_add,
)
from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size, get_tp_group
from aiter.dist.device_communicators.quick_all_reduce import QuickAllReduce
from aiter.ops.triton.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad
from aiter.jit.utils.torch_guard import torch_compile_guard
from atom.utils.forward_context import (
    ForwardContext,
    get_forward_context,
)

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


def quick_ar_or_fused_arnorm_fake_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
    dim: int,
    use_qr_when_possible,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty_like(x), torch.empty_like(residual))


@torch_compile_guard(gen_fake=quick_ar_or_fused_arnorm_fake_tensors)
def quick_ar_or_fused_arnorm_(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
    dim: int,
    use_qr_when_possible: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    context = forward_context.context
    if use_qr_when_possible and context.is_prefill:
        qr_comm = get_tp_group().device_communicator.qr_comm
        assert qr_comm is not None, "Quick all-reduce communication is not initialized!"
        ar_out = qr_comm.quick_all_reduce(x)
        return rmsnorm2d_fwd_with_add_(ar_out, weight, residual, eps, dim)
    return tensor_model_parallel_fused_allreduce_rmsnorm(
        x, 
        residual, 
        weight, 
        eps,
        )


class RMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        x_pad_to_multiple: int = 0,
        fused_allreduce: bool = False,
        use_qr_when_possible: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.x_pad_to_multiple = x_pad_to_multiple
        self.fused_allreduce = fused_allreduce
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_qr_when_possible = use_qr_when_possible

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.x_pad_to_multiple > 0:
            assert not self.fused_allreduce, "fused_allreduce_rmsnorm is not supported with rms_norm padding!"
            if residual is None:
                return fused_rmsnorm_pad_(
                    x, self.weight, self.eps, self.x_pad_to_multiple
                )
            else:
                return fused_add_rmsnorm_pad_(
                    x, self.weight, self.eps, residual, self.x_pad_to_multiple
                )            
        if self.fused_allreduce and self.tp_size > 1:
            # use quick allreduce for prefill, will not fuse rmsnorm
            # use fused allreduce rmsnorm for decode
            assert residual is not None, "residual input must be provided."
            return quick_ar_or_fused_arnorm_(
                x,
                self.weight,
                residual,
                self.eps,
                self.dim,
                self.use_qr_when_possible,
            )
        else:
            if residual is None:
                # return rmsnorm2d_fwd(x, self.weight, self.eps).view(ori_shape)
                return rmsnorm2d_fwd_(x, self.weight, self.eps, self.dim)
            else:
                # return self.add_rms_forward(x, residual)
                return rmsnorm2d_fwd_with_add_(x, self.weight, residual, self.eps, self.dim)


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
