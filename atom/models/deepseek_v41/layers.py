# SPDX-License-Identifier: MIT
"""Native linear configuration and FP32 tensor-parallel reductions for V4.1."""

import torch
from aiter import QuantType
from aiter.dist.parallel_state import get_tp_group

from atom.config import QuantizationConfig
from atom.quant_spec import LayerQuantConfig


def native_quant_config(*, fp4=False):
    config = QuantizationConfig()
    config.global_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=torch.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn,
        weight_block_size=(1, 32) if fp4 else (32, 32),
        activation_dtype=torch.float8_e4m3fn,
    )
    return config


def reduce_output(partial):
    group = get_tp_group()
    if group.world_size == 1:
        return partial
    return group.all_reduce(partial.float(), ca_fp8_quant=False).to(partial.dtype)
