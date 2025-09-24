import torch
from torch import nn
from aiter import rmsnorm2d_fwd, rmsnorm2d_fwd_with_add, rms_norm


class RMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

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
        ori_shape = x.shape
        x = x.reshape(-1, self.dim)
        if residual is None:
            return rmsnorm2d_fwd(x, self.weight, self.eps).view(ori_shape)
        else:
            # return self.add_rms_forward(x, residual)
            residual_out = torch.empty_like(x)
            out = torch.empty_like(x)
            rmsnorm2d_fwd_with_add(
                out, x, residual, residual_out, self.weight, self.eps
            )
            return out.view(ori_shape), residual_out.view(ori_shape)
