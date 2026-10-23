# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Call-site installation of fused QKV+attention on MiniMax-M3 attention modules.

Why not edit ``atom/models/minimax_m3/model.py``: ``MiniMaxM3Model`` is
``@support_torch_compile`` and ATOM's rules forbid modifying such files. The
runner calls ``install_m3_sites(model)`` right after the model is built (before
weights are loaded, before ``torch.compile``/Dynamo or any forward) ONLY when
``config.gemm_fusion`` is ON. With OFF this module is never imported and
``model.py`` is byte-identical to the base.

How Dynamo sees it: ``install`` sets an instance attribute ``forward`` (a bound
method of a module-level function) on each eligible attention module. Dynamo
traces ``MiniMaxM3Model.forward`` -> ``decoder_layer`` -> ``self_attn(...)``;
``nn.Module.__call__`` resolves ``self.forward`` to the instance attribute, which
Dynamo inlines like any user method. The inlined body is a single call to the
registered splitting op ``torch.ops.aiter.gemm_fusion_qkv_attention`` followed by
``o_proj`` -- no Python branch on runtime values, so no guards beyond the
module-attribute ones Dynamo already installs. The op is marked with
``mark_spliting_op`` so piecewise compilation splits around it exactly as it does
around ``minimax_m3_attention_fp8``. The install runs before ``self.model`` is
wrapped or compiled and before the first call of the decorated ``__call__``, so
the swapped forward is what the first trace sees.
"""

import types

import torch


def _fused_forward(self, positions, hidden_states, hidden_states_scale=None):
    """Replacement forward for an eligible M3 attention module (site installed)."""
    output, scale = torch.ops.aiter.gemm_fusion_qkv_attention(
        hidden_states, hidden_states_scale, positions, self.attn.layer_name
    )
    return self.o_proj(output, x_scale=scale)


def install_m3_sites(model) -> int:
    """Declare sites and swap forwards on every eligible M3 attention module.

    Returns the number of modules fused. Idempotent per module.
    """
    from atom.models.minimax_m3.model import (
        MiniMaxM3Attention,
        MiniMaxM3SparseAttention,
    )

    from . import declare_m3_qkv_site

    n = 0
    for module in model.modules():
        if isinstance(module, MiniMaxM3SparseAttention):
            sparse = True
        elif isinstance(module, MiniMaxM3Attention):
            sparse = False
        else:
            continue
        if getattr(module.attn, "qkv_site", None) is not None:
            continue
        # layer_name == f"{prefix}.attn"
        prefix = module.attn.layer_name[: -len(".attn")]
        site = declare_m3_qkv_site(module, prefix=prefix, sparse=sparse)
        if site is None:
            continue
        module.forward = types.MethodType(_fused_forward, module)
        n += 1
    return n
