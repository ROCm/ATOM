"""Split-K GEMM "prezero" helpers.

A split-K GEMM can skip its in-kernel output zero-init (NOZINIT) and atomic-add
into a buffer the preceding fused allreduce-rmsnorm zeroed for free, when that
pays off for the current batch. The decision lives inside opaque custom ops so
it re-runs at each per-bs cudagraph capture instead of being frozen by the one
prefill-time compile.
"""

from typing import Optional

import torch
from aiter import is_prezero_free
from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.tuned_gemm import tgemm

from atom.utils import envs
from atom.utils.forward_context import get_forward_context


def prezero_active(n_total: int) -> bool:
    if not envs.ATOM_ENABLE_SPLITK_PREZERO:
        return False
    ctx = get_forward_context().context
    return (not ctx.is_prefill) and is_prezero_free(ctx.graph_bs, n_total)


@torch_compile_guard(
    mutates_args=["out"],
    gen_fake=lambda out, a, weight, n_total, bias=None: None,
)
def mm_maybe_prezero_(
    out: torch.Tensor,
    a: torch.Tensor,
    weight: torch.Tensor,
    n_total: int,
    bias: Optional[torch.Tensor] = None,
) -> None:
    active = prezero_active(n_total)
    tgemm.mm(a, weight, bias, zero_init=not active, out=out)


@torch_compile_guard(
    mutates_args=["zero_buf"],
    gen_fake=lambda x, residual, weight, eps, zero_buf, n_total: (
        torch.empty_like(x),
        torch.empty_like(residual),
    ),
)
def ar_rmsnorm_maybe_prezero_(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_buf: torch.Tensor,
    n_total: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = prezero_active(n_total)
    return tensor_model_parallel_fused_allreduce_rmsnorm(
        x.contiguous(),
        residual,
        weight,
        eps,
        zero_fill=zero_buf if active else None,
    )
