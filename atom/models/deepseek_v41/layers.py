# SPDX-License-Identifier: MIT
"""Linear quantization configs for V4.1's two checkpoint formats."""

import torch
from aiter import QuantType, dtypes

from atom.config import QuantizationConfig
from atom.quant_spec import LayerQuantConfig


def _native_spec(fp4=False):
    return LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=torch.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn,
        weight_block_size=(1, 32) if fp4 else (32, 32),
        activation_dtype=torch.float8_e4m3fn,
    )


def _as_config(spec):
    config = QuantizationConfig()
    config.global_spec = spec
    return config


def native_quant_config(*, fp4=False):
    return _as_config(_native_spec(fp4))


def converted_linear_spec(config):
    """The spec of a linear the FP4PTPC conversion covers: `wq_a`, `wkv`,
    `wq_b`, `wo_b`, `indexer.wq_b`, `engram.wkv` and the shared expert.

    Per-channel FP8 with dynamic per-token activations in an FP4PTPC
    checkpoint (`config.fp4ptpc`), native 32x32 FP8 otherwise. `wo_a`, the
    engram tables and the draft keep the native format in both.
    """
    if config.fp4ptpc:
        return LayerQuantConfig(quant_type=QuantType.per_Token, quant_dtype=dtypes.fp8)
    return _native_spec()


def converted_linear_config(config):
    """`converted_linear_spec` as a linear's `quant_config`."""
    return _as_config(converted_linear_spec(config))
