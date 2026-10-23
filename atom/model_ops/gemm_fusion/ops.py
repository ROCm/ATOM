# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opaque custom op: QKV GEMM [+ SP exchange [+ norm/rope/cache]] + attention.

Follows ``minimax_m3_attention_fp8``: the GEMM moves inside the op, and the
fused-vs-native decision is made at runtime inside it, so the compiled graph
sees one stable schema and piecewise compilation never bakes in a branch.
"""

import torch

from atom.config import get_current_atom_config
from atom.utils import mark_spliting_op

from . import hooks
from .context import AttnFusionCtx


def _layer(layer_name):
    return get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]


def run_qkv_attention(attn, x, x_scale, positions):
    """Real implementation behind the custom op.

    Returns ``((output_fp8, scale), Decision)``. Any failure inside a fused op
    triggers ``hooks.recover()`` before re-raising.
    """
    from atom.distributed.ulysses_sp import get_sp_world_size

    site = attn.qkv_site
    ctx = AttnFusionCtx.from_forward_context(
        attn, positions, x.shape[0], get_sp_world_size()
    )
    try:
        return site.run(x, x_scale, ctx)
    except Exception:
        # A host exception on THIS rank is a one-rank event while recover() is
        # collective. It is bounded: peers that do not arrive make this call raise
        # GemmFusionRecoveryTimeout (chained to the original error) instead of
        # hanging. The engine must treat the worker as failed either way.
        hooks.recover()
        raise


def _fake(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    positions: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from atom.distributed.ulysses_sp import get_sp_world_size

    attn = _layer(layer_name)
    q_cols = attn.impl.num_heads * attn.impl.head_dim * get_sp_world_size()
    from aiter import dtypes

    return (
        torch.empty((x.shape[0], q_cols), device=x.device, dtype=dtypes.fp8),
        torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
    )


@mark_spliting_op(is_custom=True, gen_fake=_fake, mutates_args=[])
def gemm_fusion_qkv_attention(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    positions: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, _decision = run_qkv_attention(_layer(layer_name), x, x_scale, positions)
    return out
