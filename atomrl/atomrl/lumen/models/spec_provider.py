# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen spec provider for Megatron-Core layer specs.

Every ``BackendSpecProvider`` method returns a Lumen module class.
Plug this into ``get_gpt_layer_with_transformer_engine_spec`` or compose
your own ``ModuleSpec`` tree to get Lumen-accelerated attention, norms,
and (optionally FP8) linear layers.
"""

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP

from atomrl.lumen.modules.attention_megatron import LumenDotProductAttention
from atomrl.lumen.modules.grouped_linear import (
    LumenColumnParallelGroupedLinear,
    LumenRowParallelGroupedLinear,
)
from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear
from atomrl.lumen.modules.parallel_linear import (
    LumenColumnParallelLinear,
    LumenRowParallelLinear,
)


class _LumenNorm:
    """Factory that returns the correct Lumen norm wrapper class based on
    the ``normalization`` config attribute (``"RMSNorm"`` or ``"LayerNorm"``)."""

    def __new__(cls, config, hidden_size, eps=1e-6, **kwargs):
        norm_type = getattr(config, "normalization", "LayerNorm")
        sequence_parallel = bool(getattr(config, "sequence_parallel", False))
        if norm_type == "RMSNorm":
            from atomrl.lumen.ops.normalization import LumenRMSNorm

            return LumenRMSNorm(
                hidden_size, eps=eps, sequence_parallel=sequence_parallel
            )
        else:
            from atomrl.lumen.ops.normalization import LumenLayerNorm

            return LumenLayerNorm(
                hidden_size, eps=eps, sequence_parallel=sequence_parallel
            )


class LumenSpecProvider(BackendSpecProvider):
    """Provides Lumen module classes for Megatron-Core layer specs."""

    def linear(self):
        return LumenColumnParallelLinear

    def column_parallel_linear(self):
        return LumenColumnParallelLinear

    def row_parallel_linear(self):
        return LumenRowParallelLinear

    def fuse_layernorm_and_linear(self):
        return True

    def column_parallel_layer_norm_linear(self):
        return LumenLayerNormLinear

    def layer_norm(self, rms_norm=False, for_qk=False):
        return _LumenNorm

    def core_attention(self):
        return LumenDotProductAttention

    def grouped_mlp_modules(
        self,
        moe_use_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=False,
    ) -> tuple[type, MLPSubmodules | None]:
        try:
            from megatron.core.transformer.moe.experts import TEGroupedMLP

            if moe_use_grouped_gemm and not moe_use_legacy_grouped_gemm:
                return TEGroupedMLP, MLPSubmodules(
                    linear_fc1=LumenColumnParallelGroupedLinear,
                    linear_fc2=LumenRowParallelGroupedLinear,
                )
        except ImportError:
            pass

        return SequentialMLP, MLPSubmodules(
            linear_fc1=LumenColumnParallelLinear,
            linear_fc2=LumenRowParallelLinear,
        )

    def activation_func(self):
        return None
