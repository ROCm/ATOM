# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen checkpoint storage, mixed GDN projections, and scale preservation."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from aiter import QuantType, dtypes

from atom.config import QuantizationConfig
from atom.model_ops import embed_head
from atom.model_ops.qwen4_exp import ple_layer
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


def test_ptpc_exclusions_do_not_hide_quantized_children():
    config = ptpc_config()
    view = qwen4_exp._Qwen4ExpQuantizationConfig(config)
    prefix = "model.layers.0.linear_attn"
    assert not view.get_layer_quant_config(prefix).is_quantized
    for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
        assert view.get_layer_quant_config(prefix + "." + name).is_quantized
    for name in ("in_proj_b", "in_proj_a"):
        assert not view.get_layer_quant_config(prefix + "." + name).is_quantized
    assert not view.get_layer_quant_config(
        "mtp.layers.0.self_attn.qkv_proj"
    ).is_quantized
    # Other models retain their existing parent-exclusion behavior.
    assert not config.get_layer_quant_config(
        "model.language_model.layers.0.linear_attn.in_proj_qkv"
    ).is_quantized
    config.quant_method = "fp8"
    assert not view.get_layer_quant_config(prefix + ".in_proj_qkv").is_quantized


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.w13_weight",
        ),
        ("mtp.layers.0.mlp.experts.down_proj", "model.layers.0.mlp.experts.w2_weight"),
        (
            "mtp.layers.0.mlp.experts.7.gate_proj.weight_scale_inv",
            "model.layers.0.mlp.experts.7.gate_proj.weight_scale",
        ),
    ],
)
def test_mtp_maps_stacked_and_per_expert_checkpoints(source, target):
    from atom.model_loader.weight_names import CheckpointNameRewriter
    from atom.models.qwen4_exp_mtp import Qwen4ExpMTP

    model = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
    rewriter = CheckpointNameRewriter(
        weights_mapping=model.weights_mapping,
        mtp_remap=model.remap_mtp_weight_name,
        spec_decode=True,
        num_hidden_layers=48,
    )
    assert rewriter.rewrite(source) == target


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("quantized", [False, True])
def test_gdn_projection_storage_and_scales(monkeypatch, tp, rank, quantized):
    from atom.model_ops import linear

    group = SimpleNamespace(world_size=tp, rank_in_group=rank)
    monkeypatch.setattr(linear, "get_tp_group", lambda: group)
    monkeypatch.setattr(qwen4_exp, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(qwen4_exp, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(
        qwen4_exp, "LinearAttention", lambda *args, **kwargs: torch.nn.Identity()
    )
    config = SimpleNamespace(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    quant_config = ptpc_config()
    if not quantized:
        quant_config.quant_method = "fp8"
    layer = qwen4_exp.Qwen4ExpLinearAttention(
        SimpleNamespace(torch_dtype=torch.bfloat16),
        config,
        qwen4_exp._Qwen4ExpQuantizationConfig(quant_config),
        prefix="model.layers.0.linear_attn",
    )
    assert layer.quantized_inputs == quantized
    if not quantized:
        assert layer.in_proj_qkvzba.weight.dtype == torch.bfloat16
        assert layer.in_proj_qkvzba.weight_scale is None
        assert layer.out_proj.weight.dtype == torch.bfloat16
        return
    assert layer.in_proj_qkvz.weight.dtype == dtypes.fp8
    assert layer.in_proj_ba.weight.dtype == layer.conv1d.weight.dtype == torch.bfloat16
    assert layer.out_proj.weight.dtype == dtypes.fp8
    widths = [layer.key_dim, layer.key_dim, layer.value_dim, layer.value_dim]
    weight = (
        torch.arange(sum(widths) * config.hidden_size).reshape(-1, config.hidden_size)
        % 17
    ).to(dtypes.fp8)
    scale = (torch.arange(sum(widths), dtype=torch.float32)[:, None] + 1) / 128
    offset = 0
    for source, rows in ((".in_proj_qkv", sum(widths[:3])), (".in_proj_z", widths[3])):
        target, shard = layer.packed_modules_mapping[source]
        module = getattr(layer, target[1:])
        for name, values in (("weight", weight), ("weight_scale", scale)):
            param = getattr(module, name)
            param.weight_loader(param, values[offset : offset + rows], shard)
        offset += rows
    for name, values in (("weight", weight), ("weight_scale", scale)):
        expected = torch.cat([part.chunk(tp, 0)[rank] for part in values.split(widths)])
        torch.testing.assert_close(
            getattr(layer.in_proj_qkvz, name).float(), expected.float(), rtol=0, atol=0
        )
    out_scale = (
        torch.arange(1, config.hidden_size + 1, dtype=torch.float32)[:, None] / 128
    )
    layer.out_proj.weight_scale.weight_loader(layer.out_proj.weight_scale, out_scale)
    torch.testing.assert_close(layer.out_proj.weight_scale, out_scale)


@pytest.mark.parametrize(
    "unsupported", ["block_scales", "mixed_qkv_z", "quantized_b", "online"]
)
def test_gdn_rejects_unsupported_input_quantization(unsupported):
    config = ptpc_config()
    prefix = "model.language_model.layers.0.linear_attn"
    if unsupported == "block_scales":
        config.global_spec = LayerQuantConfig(
            quant_type=QuantType.per_1x128, quant_dtype=dtypes.fp8
        )
    elif unsupported == "mixed_qkv_z":
        config.exclude_layers.append(prefix + ".in_proj_z")
    elif unsupported == "quantized_b":
        config.exclude_layers.remove(prefix + ".in_proj_b")
    else:
        config.online_quant = True
        config.online_global_spec = config.global_spec
    with pytest.raises(ValueError, match="Qwen GDN"):
        qwen4_exp.Qwen4ExpLinearAttention(
            None,
            None,
            qwen4_exp._Qwen4ExpQuantizationConfig(config),
            prefix="model.layers.0.linear_attn",
        )


@pytest.fixture
def make_embedding(monkeypatch):
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(embed_head, "get_tp_group", lambda: group)
    monkeypatch.setattr(ple_layer, "get_tp_group", lambda: group)

    def make(dtype=None, method=None, quantized=True):
        config = SimpleNamespace(
            ngram_size=3,
            heads_per_ngram=1,
            eos_token_id=0,
            vocab_size=128,
            split_ngram_parts=2,
            ngram_vocab_size_base=17,
            make_ngram_vocab_size_divisible_by=8,
        )
        if dtype is not None:
            config.ple_embedding_dtype = dtype
        quant_config = (
            SimpleNamespace(
                quant_method=method,
                get_layer_quant_config=lambda _: SimpleNamespace(
                    is_quantized=quantized
                ),
            )
            if method is not None
            else None
        )
        result = ple_layer.Qwen4ExpNGramEmbedding(
            config, 8, 0, 16, 4, prefix="ple", quant_config=quant_config
        )
        if quant_config is not None:
            assert quant_config.quant_method == method
        return result

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield make
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize(
    "dtype,method,quantized,expected",
    [
        (None, None, False, torch.bfloat16),
        (None, "fp8", True, torch.float8_e4m3fn),
        (None, "fp8", False, torch.bfloat16),
        ("float8_e4m3fn", "compressed-tensors", True, torch.float8_e4m3fn),
        ("float8_e4m3fn", "compressed-tensors", False, torch.float8_e4m3fn),
        ("float8_e4m3fn", None, False, torch.float8_e4m3fn),
        ("bfloat16", "compressed-tensors", True, torch.bfloat16),
    ],
)
def test_ple_checkpoint_storage(make_embedding, dtype, method, quantized, expected):
    embedding = make_embedding(dtype, method, quantized).ngram_embedding
    assert embedding.weight.dtype == expected
    assert hasattr(embedding, "weight_scale") == (expected == torch.float8_e4m3fn)


def test_ple_does_not_infer_fp8_from_arbitrary_quantization(make_embedding):
    with pytest.raises(ValueError, match="supports BF16 or FP8"):
        make_embedding(method="compressed-tensors")
    with pytest.raises(ValueError, match="unsupported ple_embedding_dtype"):
        make_embedding("float8_e5m2", "compressed-tensors")


def test_ptpc_ple_loads_fp8_shards_and_scalar_scale(make_embedding):
    layer = make_embedding("float8_e4m3fn", "compressed-tensors")
    embedding = layer.ngram_embedding
    with pytest.raises(ValueError, match="missing or is not finite and positive"):
        embedding.process_weights_after_loading()
    weight = (
        torch.arange(layer.table_rows * layer.head_dim)
        .reshape(layer.table_rows, layer.head_dim)
        .to(torch.float8_e4m3fn)
    )
    for index, shard in enumerate(weight.split(layer.checkpoint_shard_rows)):
        embedding.weight.weight_loader(embedding.weight, shard, index)
    torch.testing.assert_close(embedding.weight.float(), weight.float(), rtol=0, atol=0)
    scale = torch.tensor([0.125], dtype=torch.bfloat16)
    embedding.weight_scale.weight_loader(embedding.weight_scale, scale)
    embedding.process_weights_after_loading()
    assert embedding.weight_scale.dtype == torch.float32
    assert embedding.weight_scale.item() == scale.item()
    with pytest.raises(ValueError, match="one global weight_scale"):
        embedding.weight_scale.weight_loader(embedding.weight_scale, torch.ones(2))
    with pytest.raises(ValueError, match="expected torch.float8_e4m3fn"):
        embedding.weight.weight_loader(embedding.weight, weight.to(torch.bfloat16), 0)
