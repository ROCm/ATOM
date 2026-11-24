# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.engine.inference.vllm_fp8_utils import (
    prepare_prequantized_fp8_weights_for_loading,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)


class LinearLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4), requires_grad=False)
        self.weight.weight_loader = lambda *args, **kwargs: None
        self.weight_scale_inv = torch.nn.Parameter(
            torch.ones(1, 1),
            requires_grad=False,
        )
        self.weight_block_size = [128, 128]


model = torch.nn.Module()
layer = LinearLayer()
model.add_module("linear", layer)
resident_loader = layer.weight.weight_loader
weight_ptr = layer.weight.data_ptr()
scale_ptr = layer.weight_scale_inv.data_ptr()

template = torch.nn.Parameter(
    torch.empty(4, 4, device="meta"),
    requires_grad=False,
)
template.__class__ = ModelWeightParameter
template.__dict__ = {
    "_output_dim": 0,
    "_input_dim": 1,
    "_weight_loader": lambda *args, **kwargs: None,
    "tp_rank": 0,
    "tp_size": 8,
}
get_layerwise_info(layer).restore_metadata = ({"weight": template}, {})

summary = prepare_prequantized_fp8_weights_for_loading(model)

assert isinstance(layer.weight, ModelWeightParameter)
assert isinstance(layer.weight_scale_inv, BlockQuantScaleParameter)
assert layer.weight.output_dim == 0
assert layer.weight.input_dim == 1
assert layer.weight.weight_loader is resident_loader
assert layer.weight_scale_inv.output_dim == 0
assert layer.weight_scale_inv.input_dim == 1
assert layer.weight_scale_inv.weight_loader is resident_loader
assert layer.weight.data_ptr() == weight_ptr
assert layer.weight_scale_inv.data_ptr() == scale_ptr
assert summary == {"weights": 1, "block_scales": 1, "moe_scales": 0}
print(summary, flush=True)
