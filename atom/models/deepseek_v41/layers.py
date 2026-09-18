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
    """All-reduce a TP partial in FP32, then return it in the caller's dtype.

    The upcast doubles what goes on the wire, which V4 does not do, so it looks
    like an easy saving. It is not: measured over a 4-way sum, reducing in BF16
    instead doubles the error against an FP64 reference while saving about 4 us
    a call, and this runs on every layer, where the error accumulates.
    """
    group = get_tp_group()
    if group.world_size == 1:
        return partial
    return group.all_reduce(partial.float(), ca_fp8_quant=False).to(partial.dtype)
