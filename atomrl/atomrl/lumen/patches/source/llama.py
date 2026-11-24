# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""LLaMA / GPT Megatron SOURCE patches (RMSNorm + apex FusedLayerNorm fix).

FusedLayerNorm (apex) does not support RMSNorm, but several Megatron files
default to FusedLayerNorm when apex is present. These patches:

1. Add ``megatron_fused_rmsnorm.py`` (MegatronFusedRMSNorm wrapper).
2. Patch ``gpt_layer_specs.py`` to select MegatronFusedRMSNorm for RMSNorm.
3. Patch ``transformer_block.py`` to use MegatronFusedRMSNorm instead of FusedLayerNorm.

Filter with ``--tag llama`` (included in the default SOURCE apply set).
"""

from __future__ import annotations

import os

from atomrl.lumen.patches.registry import PatchPhase, register_patch

_MARKER = "# patched-lnimpl"

_MEGATRON_FUSED_RMSNORM_SOURCE = '''\
"""Megatron-compatible wrapper around apex FusedRMSNorm.

Supports the TransformerConfig constructor signature and
sequence_parallel attribute on weights.
"""

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init

from apex.normalization.fused_layer_norm import (
    fused_rms_norm_affine,
    fused_rms_norm,
    manual_rms_norm,
)

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import make_viewless_tensor


class MegatronFusedRMSNorm(torch.nn.Module):
    """RMSNorm using apex fused kernels, compatible with Megatron layer specs.

    Accepts the same (config, hidden_size, eps, ...) signature that Megatron
    layer-spec build_module() passes, and sets sequence_parallel on weights
    so gradient all-reduce works correctly with SP.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ):
        super().__init__()
        self.config = config
        self.zero_centered_gamma = config.layernorm_zero_centered_gamma

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps

        self.weight = Parameter(torch.empty(*self.hidden_size))
        self.reset_parameters()

        self.sequence_parallel = config.sequence_parallel
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self):
        if self.zero_centered_gamma:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        if not input.is_cuda:
            return manual_rms_norm(input, self.hidden_size, weight, self.eps)

        output = fused_rms_norm_affine(input, weight, self.hidden_size, self.eps, False)
        return make_viewless_tensor(
            inp=output, requires_grad=input.requires_grad, keep_graph=True
        )
'''


def patch_megatron_fused_rmsnorm_module(megatron_root: str) -> bool:
    """Create ``megatron/core/transformer/megatron_fused_rmsnorm.py``."""
    path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "megatron_fused_rmsnorm.py"
    )
    if os.path.isfile(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(_MEGATRON_FUSED_RMSNORM_SOURCE)
    return True


def patch_gpt_layer_specs_rmsnorm(megatron_root: str) -> bool:
    """Use MegatronFusedRMSNorm in gpt_layer_specs when normalization is RMSNorm."""
    path = os.path.join(
        megatron_root, "megatron", "core", "models", "gpt", "gpt_layer_specs.py"
    )
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        content = f.read()
    if _MARKER in content:
        return False
    old = "        layer_norm_impl = LNImpl"
    if old not in content:
        return False
    new = (
        "        from megatron.core.transformer.megatron_fused_rmsnorm import MegatronFusedRMSNorm as _MFRN\n"
        f'        layer_norm_impl = _MFRN if normalization == "RMSNorm" else LNImpl  {_MARKER}'
    )
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    return True


def patch_transformer_block_rmsnorm(megatron_root: str) -> bool:
    """Use MegatronFusedRMSNorm as LayerNormImpl in transformer_block."""
    path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "transformer_block.py"
    )
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        content = f.read()
    if _MARKER in content:
        return False
    old = "elif HAVE_APEX:\n    LayerNormImpl = FusedLayerNorm"
    new = (
        "elif HAVE_APEX:\n"
        f"    from megatron.core.transformer.megatron_fused_rmsnorm import MegatronFusedRMSNorm  {_MARKER}\n"
        "    LayerNormImpl = MegatronFusedRMSNorm"
    )
    if old not in content:
        return False
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    return True


register_patch(
    "llama_megatron_fused_rmsnorm",
    PatchPhase.SOURCE,
    description="Add MegatronFusedRMSNorm wrapper for apex FusedRMSNorm + SP",
    tags=frozenset({"llama", "norm", "megatron"}),
)(patch_megatron_fused_rmsnorm_module)

register_patch(
    "llama_gpt_layer_specs_rmsnorm",
    PatchPhase.SOURCE,
    description="Select MegatronFusedRMSNorm in gpt_layer_specs for RMSNorm",
    depends_on=("llama_megatron_fused_rmsnorm",),
    tags=frozenset({"llama", "norm", "megatron"}),
)(patch_gpt_layer_specs_rmsnorm)

register_patch(
    "llama_transformer_block_rmsnorm",
    PatchPhase.SOURCE,
    description="Use MegatronFusedRMSNorm as LayerNormImpl in transformer_block",
    depends_on=("llama_megatron_fused_rmsnorm",),
    tags=frozenset({"llama", "norm", "megatron"}),
)(patch_transformer_block_rmsnorm)
