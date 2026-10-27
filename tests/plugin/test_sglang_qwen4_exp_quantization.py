"""Quantization preparation for the SGLang Flash wrapper."""

from types import SimpleNamespace

import pytest

pytest.importorskip("triton")
pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from aiter import QuantType, dtypes

from atom.config import QuantizationConfig
from atom.models import qwen4_exp
from atom.quant_spec import LayerQuantConfig


def ptpc_config():
    config = QuantizationConfig()
    config.quant_method = "compressed-tensors"
    config.global_spec = LayerQuantConfig(
        quant_type=QuantType.per_Token, quant_dtype=dtypes.fp8
    )
    prefix = "model.language_model.layers.0.linear_attn"
    config.exclude_layers = [
        prefix,
        prefix + ".in_proj_b",
        prefix + ".in_proj_a",
        "re:^mtp.*",
    ]
    return config


def test_sglang_plugin_ptpc_remap_keeps_gdn_shards():
    """Plugin quant remap must not collapse GDN qkv/z with unquantized b/a."""
    from atom.plugin.sglang.models.qwen4_exp import (
        apply_prepare_qwen4_exp_adaptations,
    )

    config = ptpc_config()
    atom_config = SimpleNamespace(
        quant_config=config,
        hf_config=SimpleNamespace(
            model_type="qwen4_exp",
            split_ngram_parts=2,
            text_config=None,
        ),
    )
    apply_prepare_qwen4_exp_adaptations(atom_config, "Qwen4ExpForConditionalGeneration")
    view = qwen4_exp._Qwen4ExpQuantizationConfig(config)
    prefix = "model.layers.0.linear_attn"
    assert view.get_layer_quant_config(prefix + ".in_proj_qkv").is_quantized
    assert view.get_layer_quant_config(prefix + ".in_proj_z").is_quantized
    assert not view.get_layer_quant_config(prefix + ".in_proj_b").is_quantized
    assert not view.get_layer_quant_config(prefix + ".in_proj_a").is_quantized
    assert not any("in_proj_qkvzba" in name for name in config.exclude_layers)
