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

from atom.config import get_current_atom_config
from atom.utils.forward_context import get_forward_context


def _prezero_hidden() -> int:
    # AR out_hidden_dim == hidden_size; sets the zero-fill CTA width.
    return get_current_atom_config().hf_config.hidden_size


def prezero_active(n_total: int) -> bool:
    import os

    if not os.getenv("ATOM_ENABLE_PREZERO"):
        return False
    ctx = get_forward_context().context
    return (not ctx.is_prefill) and is_prezero_free(
        ctx.graph_bs, n_total, _prezero_hidden()
    )


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
    gen_fake=lambda x, residual, weight, eps, zero_buf, n_total, n_base=0: (
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
    n_base: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # zero the largest free cumulative prefix; drop only the overflowing tail.
    import os

    zf = None
    if os.getenv("ATOM_ENABLE_PREZERO"):
        ctx = get_forward_context().context
        if not ctx.is_prefill:
            bs = ctx.graph_bs
            h = x.shape[-1]
            if is_prezero_free(bs, n_total, h):
                zf = zero_buf
            elif n_base and is_prezero_free(bs, n_base, h):
                m = x.shape[0]
                zf = zero_buf.view(-1)[: m * n_base].view(m, n_base)
    return tensor_model_parallel_fused_allreduce_rmsnorm(
        x.contiguous(),
        residual,
        weight,
        eps,
        zero_fill=zf,
    )
