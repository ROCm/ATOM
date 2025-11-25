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
from aiter.ops.triton.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad
from aiter.jit.utils.torch_guard import torch_compile_guard


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


@torch_compile_guard()
def fused_rmsnorm_pad_(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    x_pad_to_multiple: int = 0,
) -> torch.Tensor:
    return fused_add_rmsnorm_pad(x, weight, epsilon, None, x_pad_to_multiple)


@torch_compile_guard()
def fused_add_rmsnorm_pad_(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor,
    x_pad_to_multiple: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return fused_add_rmsnorm_pad(x, weight, epsilon, res, x_pad_to_multiple)


class RMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        x_pad_to_multiple: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.x_pad_to_multiple = x_pad_to_multiple

    # def rms_forward(
    #     self,
    #     x: torch.Tensor,
    # ) -> torch.Tensor:
    #     orig_dtype = x.dtype
    #     x = x.to(torch.float32)
    #     var = x.pow(2).mean(dim=-1, keepdim=True)
    #     x.mul_(torch.rsqrt(var + self.eps))
    #     x = x.to(orig_dtype).mul_(self.weight)
    #     return x

    # def add_rms_forward(
    #     self,
    #     x: torch.Tensor,
    #     residual: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     orig_dtype = x.dtype
    #     x = x.to(torch.float32).add_(residual.to(torch.float32))
    #     residual = x.to(orig_dtype)
    #     var = x.pow(2).mean(dim=-1, keepdim=True)
    #     x.mul_(torch.rsqrt(var + self.eps))
    #     x = x.to(orig_dtype).mul_(self.weight)
    #     return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.x_pad_to_multiple > 0:
            if residual is None:
                return fused_rmsnorm_pad_(
                    x, self.weight, self.eps, self.x_pad_to_multiple
                )
            else:
                return fused_add_rmsnorm_pad_(
                    x, self.weight, self.eps, residual, self.x_pad_to_multiple
                )
        elif residual is None:
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
