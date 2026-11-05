# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5 fused GDN input projection on quantized checkpoints."""

from types import SimpleNamespace

import pytest

pytest.importorskip("triton")
pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from aiter import QuantType, dtypes

from atom.config import QuantizationConfig
from atom.models import qwen3_5
from atom.quant_spec import LayerQuantConfig


def mxfp4_config(exclude_layers):
    config = QuantizationConfig()
    config.global_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2
    )
    config.exclude_layers = exclude_layers
    text_config = SimpleNamespace(
        layer_types=["linear_attention", "linear_attention", "full_attention"]
    )
    return SimpleNamespace(quant_config=config, hf_config=text_config)


def in_proj_excludes(layers):
    prefix = "model.language_model.layers"
    return [
        f"{prefix}.{i}.linear_attn.{proj}"
        for i in layers
        for proj in ("in_proj_qkvz", "in_proj_ba")
    ]


def test_fused_in_proj_needs_every_gdn_layer_excluded():
    assert qwen3_5._gdn_in_proj_is_bf16(mxfp4_config(in_proj_excludes([0, 1])))
    # Layer 1's in_proj_ba stays quantized.
    assert not qwen3_5._gdn_in_proj_is_bf16(mxfp4_config(in_proj_excludes([0, 1])[:-1]))
    # Two entries per projection, but both for layer 0.
    assert not qwen3_5._gdn_in_proj_is_bf16(mxfp4_config(in_proj_excludes([0]) * 2))
