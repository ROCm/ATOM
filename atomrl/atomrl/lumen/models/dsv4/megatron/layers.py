# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Megatron-compatible linear/norm modules without Transformer Engine."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.layers import (
    set_tensor_model_parallel_attributes,
)
from megatron.core.transformer.module import MegatronModule
from torch import nn

from atomrl.lumen.modules.grouped_linear import (
    LumenColumnParallelGroupedLinear as _LumenColumnParallelGroupedLinear,
)
from atomrl.lumen.modules.grouped_linear import (
    LumenRowParallelGroupedLinear as _LumenRowParallelGroupedLinear,
)
from atomrl.lumen.modules.parallel_linear import (
    LumenColumnParallelLinear as _LumenColumnParallelLinear,
)
from atomrl.lumen.modules.parallel_linear import (
    LumenRowParallelLinear as _LumenRowParallelLinear,
)


def _dsv4_use_gemm_bf16() -> bool:
    return os.environ.get("LUMEN_DSV4_GEMM_BF16", "1") != "0"


class LumenNorm(torch.nn.Module):
    """Megatron-compatible RMSNorm/LayerNorm without Transformer Engine."""

    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        del kwargs
        self.eps = eps
        norm_type = getattr(config, "normalization", "LayerNorm")
        self._norm_type = norm_type
        # Own weight at ``*.weight`` so torch_dist ckpt keys match mbridge/HF.
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.weight.sequence_parallel = bool(
            getattr(config, "sequence_parallel", False)
        )

    def forward(self, x):
        if self._norm_type == "RMSNorm":
            from atomrl.lumen.ops.normalization import rmsnorm

            return rmsnorm(x, self.weight, self.eps)
        from atomrl.lumen.ops.normalization import layernorm

        return layernorm(x, self.weight, self.eps)


class _Dsv4GemmBf16Mixin:
    """Enable AITER tuned BF16 GEMM on DSV4 linear subclasses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gemm_bf16 = _dsv4_use_gemm_bf16()


class LumenColumnParallelLinear(_Dsv4GemmBf16Mixin, _LumenColumnParallelLinear):
    pass


class LumenRowParallelLinear(_Dsv4GemmBf16Mixin, _LumenRowParallelLinear):
    pass


class LumenColumnParallelGroupedLinear(
    _Dsv4GemmBf16Mixin, _LumenColumnParallelGroupedLinear
):
    pass


class LumenRowParallelGroupedLinear(_Dsv4GemmBf16Mixin, _LumenRowParallelGroupedLinear):
    pass


__all__ = [
    "LumenColumnParallelGroupedLinear",
    "LumenColumnParallelLinear",
    "LumenDuplicatedLinear",
    "LumenNorm",
    "LumenRowParallelGroupedLinear",
    "LumenRowParallelLinear",
]


class LumenDuplicatedLinear(MegatronModule):
    """BF16 linear duplicated on every TP rank (TE ``parallel_mode='duplicated'`` replacement)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        bias: bool = False,
        skip_bias_add: bool = False,
        skip_weight_param_allocation: bool = False,
        parallel_mode: str | None = "duplicated",
        is_expert: bool = False,
        tp_group=None,
        tp_comm_buffer_name: str | None = None,
        **kwargs,
    ):
        super().__init__(config=config)
        del parallel_mode, is_expert, tp_group, tp_comm_buffer_name, kwargs
        if skip_weight_param_allocation:
            raise ValueError(
                "LumenDuplicatedLinear does not support skip_weight_param_allocation"
            )

        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.use_gemm_bf16 = _dsv4_use_gemm_bf16()

        self.scaling_type = "none"
        self.scaling_manager = None
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        self.fp8_dtype = _get_float8_e4m3()
        self.block_size = 128

        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=config.params_dtype)
        )
        # This weight is replicated on every TP rank, so marking it tensor-parallel
        # would make Megatron's grad-norm count it tp_size times.
        set_tensor_model_parallel_attributes(self.weight, False, -1, 1)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(output_size, dtype=config.params_dtype)
            )
        else:
            self.register_parameter("bias", None)

        if config.perform_initialization:
            init_method(self.weight)
            if self.bias is not None:
                with torch.no_grad():
                    self.bias.zero_()

    def forward(self, x: torch.Tensor):
        bias = None if self.skip_bias_add else self.bias
        if self.scaling_type != "none" or self.use_gemm_bf16:
            from atomrl.lumen.ops.quantize.linear import quantized_linear

            out = quantized_linear(
                x,
                self.weight,
                bias,
                scaling_manager=self.scaling_manager,
                scaling_type=self.scaling_type,
                fp8_dtype=self.fp8_dtype,
                block_size=self.block_size,
            )
        else:
            out = F.linear(x, self.weight, bias)
        output_bias = (
            self.bias if self.skip_bias_add and self.bias is not None else None
        )
        return out, output_bias

    def enable_fp8(
        self,
        scaling_manager=None,
        scaling_type="dynamic",
        fp8_dtype=None,
        block_size=None,
    ):
        from atomrl.lumen.quantize import QuantConfig, ScalingManager

        self.scaling_type = scaling_type
        self.scaling_manager = scaling_manager or ScalingManager(QuantConfig())
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if block_size is not None:
            self.block_size = block_size
