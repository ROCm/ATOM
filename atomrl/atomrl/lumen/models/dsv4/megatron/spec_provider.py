# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen backend spec provider for DSV4 transformer blocks (MoE/MLP, no TE)."""

from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

from atomrl.lumen.models.dsv4.megatron.layers import (
    LumenColumnParallelGroupedLinear,
    LumenColumnParallelLinear,
    LumenRowParallelGroupedLinear,
    LumenRowParallelLinear,
)
from atomrl.lumen.models.spec_provider import LumenSpecProvider


class LumenDSV4SpecProvider(LumenSpecProvider):
    """Backend for DSV4 MoE/MLP/dense layers — Lumen linear/norm with tuned BF16 GEMM."""

    def column_parallel_linear(self):
        return LumenColumnParallelLinear

    def row_parallel_linear(self):
        return LumenRowParallelLinear

    def grouped_mlp_modules(
        self,
        moe_use_grouped_gemm: bool = False,
        moe_use_legacy_grouped_gemm: bool = False,
    ) -> tuple[type, MLPSubmodules | None]:
        if moe_use_grouped_gemm and not moe_use_legacy_grouped_gemm:
            return TEGroupedMLP, MLPSubmodules(
                linear_fc1=LumenColumnParallelGroupedLinear,
                linear_fc2=LumenRowParallelGroupedLinear,
            )

        return SequentialMLP, MLPSubmodules(
            linear_fc1=self.column_parallel_linear(),
            linear_fc2=self.row_parallel_linear(),
        )
