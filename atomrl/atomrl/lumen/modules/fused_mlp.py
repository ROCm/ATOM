# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fused MLP modules for Lumen.

Provides drop-in replacements for sequential MLP modules that fuse
up-projection, activation, and down-projection into fewer kernel launches.
"""

import torch
from torch import nn

from atomrl.lumen.ops.mlp.fused_mlp import (
    fused_gated_mlp,
    fused_gated_mlp_fp8_store,
    fused_mlp,
    fused_mlp_fp8_store,
)


class LumenFusedMLP(nn.Module):
    """Fused ungated MLP: down(act(up(x))).

    Args:
        input_size: Input feature dimension.
        hidden_size: Intermediate hidden dimension.
        activation: Activation function name.
        bias: Whether to use bias.
        fp8_activation_store: Store activations in FP8 for backward.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "gelu",
        bias: bool = True,
        fp8_activation_store: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.fp8_activation_store = fp8_activation_store

        self.w_up = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_down = nn.Parameter(torch.empty(input_size, hidden_size))
        self.bias_up = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.bias_down = nn.Parameter(torch.zeros(input_size)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_up)
        nn.init.kaiming_uniform_(self.w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp8_activation_store:
            return fused_mlp_fp8_store(
                x,
                self.w_up,
                self.w_down,
                activation=self.activation,
                bias_up=self.bias_up,
                bias_down=self.bias_down,
            )
        return fused_mlp(
            x,
            self.w_up,
            self.w_down,
            activation=self.activation,
            bias_up=self.bias_up,
            bias_down=self.bias_down,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in={self.input_size}, hidden={self.hidden_size}, "
            f"act={self.activation}, bias={self.bias_up is not None})"
        )


class LumenGatedMLP(nn.Module):
    """Fused gated MLP: down(gate_act(gate(x)) * up(x)).

    Supports SwiGLU, GeGLU, ReGLU activations.

    Args:
        input_size: Input feature dimension.
        hidden_size: Intermediate hidden dimension.
        activation: Gate activation function name.
        bias: Whether to use bias.
        fp8_activation_store: Store activations in FP8 for backward.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "swiglu",
        bias: bool = True,
        fp8_activation_store: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.fp8_activation_store = fp8_activation_store

        self.w_gate = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_up = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_down = nn.Parameter(torch.empty(input_size, hidden_size))
        self.bias_gate = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.bias_up = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.bias_down = nn.Parameter(torch.zeros(input_size)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_gate)
        nn.init.kaiming_uniform_(self.w_up)
        nn.init.kaiming_uniform_(self.w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp8_activation_store:
            return fused_gated_mlp_fp8_store(
                x,
                self.w_up,
                self.w_gate,
                self.w_down,
                activation=self.activation,
                bias_up=self.bias_up,
                bias_gate=self.bias_gate,
                bias_down=self.bias_down,
            )
        return fused_gated_mlp(
            x,
            self.w_up,
            self.w_gate,
            self.w_down,
            activation=self.activation,
            bias_up=self.bias_up,
            bias_gate=self.bias_gate,
            bias_down=self.bias_down,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in={self.input_size}, hidden={self.hidden_size}, "
            f"act={self.activation}, bias={self.bias_gate is not None})"
        )
