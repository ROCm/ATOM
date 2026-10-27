# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Muse Spark's Quark exclude list must keep every non-expert linear BF16.

CPU-only. The checkpoint quantizes only the routed experts to MXFP4 and names
everything else -- and the experts of layers 0 and 61 -- in ``exclude``. The
model passes the real ``quant_config`` to every linear, so these prefixes,
after ``remap_layer_name`` folds q/k/v into ``qkv_proj``, must resolve to BF16.
"""

import pytest
from transformers import PretrainedConfig

pytest.importorskip("aiter")

from aiter import QuantType

from atom.config import QuantizationConfig
from atom.models.muse_spark import MuseSparkForConditionalGeneration

_MXFP4 = {
    "dtype": "fp4",
    "is_dynamic": False,
    "qscheme": "per_group",
    "ch_axis": -1,
    "group_size": 32,
    "scale_format": "e8m0",
}
_LAYER = "model.language_model.layers.{}."
_NON_EXPERT = [
    "mlp.gate",
    "mlp.post_expert_proj",
    "mlp.pre_expert_proj",
    "self_attn.gate_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
]


def _checkpoint_quant_config() -> dict:
    exclude = ["lm_head"]
    for layer in (0, 1):
        exclude += [_LAYER.format(layer) + name for name in _NON_EXPERT]
    # Layer 0 keeps BF16 experts; the checkpoint lists each one.
    exclude += [
        _LAYER.format(0) + f"mlp.experts.{e}.{proj}"
        for e in range(4)
        for proj in ("down_proj", "gate_proj", "up_proj")
    ]
    return {
        "quant_method": "quark",
        "global_quant_config": {
            "weight": _MXFP4,
            "input_tensors": {**_MXFP4, "is_dynamic": True},
        },
        "layer_quant_config": {},
        "exclude": exclude,
    }


@pytest.fixture
def quant_config() -> QuantizationConfig:
    hf = PretrainedConfig()
    hf.model_type = "muse_spark_text"
    hf.quantization_config = _checkpoint_quant_config()
    qcfg = QuantizationConfig(hf)
    qcfg.remap_layer_name(
        hf,
        packed_modules_mapping=MuseSparkForConditionalGeneration.packed_modules_mapping,
    )
    return qcfg


@pytest.mark.parametrize(
    "module",
    [
        "self_attn.qkv_proj",
        "self_attn.gate_proj",
        "self_attn.o_proj",
        "mlp.pre_expert_proj",
        "mlp.post_expert_proj",
    ],
)
def test_non_expert_linears_stay_bf16(quant_config, module):
    spec = quant_config.get_layer_quant_config(_LAYER.format(1) + module)
    assert spec.quant_type == QuantType.No


def test_routed_experts_are_mxfp4(quant_config):
    spec = quant_config.get_layer_quant_config(
        _LAYER.format(1) + "mlp.experts", check_children=True
    )
    assert spec.quant_type == QuantType.per_1x32


def test_excluded_layer_experts_stay_bf16(quant_config):
    spec = quant_config.get_layer_quant_config(
        _LAYER.format(0) + "mlp.experts", check_children=True
    )
    assert spec.quant_type == QuantType.No
