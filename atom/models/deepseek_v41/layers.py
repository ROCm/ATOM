# SPDX-License-Identifier: MIT
"""Native linear configuration for V4.1."""

import torch
from aiter import QuantType

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
