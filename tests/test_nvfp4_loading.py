# SPDX-License-Identifier: MIT
"""Load-only coverage for Quark NVFP4 checkpoint tensors."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")

from aiter import QuantType, dtypes

import atom.model_ops.linear as linear_mod
import atom.model_ops.moe as moe_mod
from atom.model_ops.linear import MergedColumnParallelLinear
from atom.model_ops.moe import FusedMoE, Nvfp4MoEMethod
from atom.models.minimax_m3 import (
    MiniMaxM3SparseForCausalLM,
    _get_sparse_attention_config,
    _is_moe_layer,
    _sparse_attention_layer_ids,
    make_minimax_m3_expert_params_mapping,
)
from atom.quant_spec import LayerQuantConfig, NVFP4_DTYPE
from atom.quantization.quark.utils import dequantize_nvfp4


def _nvfp4_spec() -> LayerQuantConfig:
    return LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=NVFP4_DTYPE,
        is_dynamic=True,
        quant_method="quark",
    )


def _nvfp4_to_mxfp4_config():
    source = _nvfp4_spec()
    target = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp4x2,
        is_dynamic=True,
        quant_method="quark",
    )
    return SimpleNamespace(
        online_quant=True,
        get_layer_quant_config=lambda *_args, use_online_quant=False, **_kwargs: (
            target if use_online_quant else source
        ),
    )


def test_dequantize_nvfp4_applies_block_and_global_scales():
    # Low nibble is the even logical value: byte 0x57 decodes to [6, 3].
    packed = torch.full((1, 8), 0x57, dtype=torch.uint8)
    block_scale = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)

    dequantized = dequantize_nvfp4(
        packed,
        block_scale,
        torch.tensor(0.5),
    )

    assert dequantized.shape == (1, 16)
    assert torch.equal(
        dequantized,
        torch.tensor([[6.0, 3.0] * 8], dtype=torch.float32),
    )


def test_nvfp4_merged_linear_loads_all_checkpoint_tensors(monkeypatch):
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    quant_config = SimpleNamespace(
        get_layer_quant_config=lambda *_args, **_kwargs: _nvfp4_spec(),
        online_quant=False,
    )

    layer = MergedColumnParallelLinear(
        64,
        [32, 32],
        quant_config=quant_config,
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    assert layer.weight.shape == (64, 32)
    assert layer.weight.dtype == torch.uint8
    assert layer.weight_scale.shape == (64, 4)
    assert layer.weight_scale.dtype == torch.float8_e4m3fn

    for shard_id, value in enumerate((3.0, 4.0)):
        layer.weight_loader(layer.weight_scale_2, torch.tensor(value), shard_id)
        layer.weight_loader(
            layer.input_scale_2, torch.tensor(value + 2.0), shard_id
        )
    layer.process_weights_after_loading()

    assert layer.weight_scale_2.tolist() == [3.0, 4.0]
    assert layer.input_scale_2.tolist() == [5.0, 6.0]
    with pytest.raises(RuntimeError, match="direct NVFP4 inference"):
        layer(torch.ones(1, 64, dtype=torch.bfloat16))


def test_nvfp4_merged_linear_online_converts_to_mxfp4(monkeypatch):
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    captured = {}

    def fake_mxfp4_quant(weight):
        captured["weight"] = weight.clone()
        packed = torch.zeros(
            weight.shape[0],
            weight.shape[1] // 2,
            dtype=torch.uint8,
        ).view(dtypes.fp4x2)
        scale = torch.full(
            (weight.shape[0], weight.shape[1] // 32),
            127,
            dtype=torch.uint8,
        ).view(dtypes.fp8_e8m0)
        return packed, scale

    monkeypatch.setattr(linear_mod, "quant_mxfp4_dynamic", fake_mxfp4_quant)
    layer = MergedColumnParallelLinear(
        64,
        [32, 32],
        quant_config=_nvfp4_to_mxfp4_config(),
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    layer.weight.data.fill_(0x22)  # E2M1 value 1 in both nibbles.
    layer.weight_scale.data.fill_(2.0)
    layer.weight_scale_2.data.copy_(torch.tensor([0.5, 1.0]))

    layer.online_quantize_weight()

    assert torch.equal(captured["weight"][:32], torch.ones(32, 64))
    assert torch.equal(captured["weight"][32:], torch.full((32, 64), 2.0))
    assert layer.params_dtype == dtypes.fp4x2
    assert layer.weight.shape == (64, 32)
    assert layer.weight_scale.shape == (64, 2)
    assert layer.weight_scale_2 is None
    assert layer.input_scale_2 is None


def test_nvfp4_linear_rejects_non_group_aligned_input(monkeypatch):
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    quant_config = SimpleNamespace(
        get_layer_quant_config=lambda *_args, **_kwargs: _nvfp4_spec(),
        online_quant=False,
    )

    with pytest.raises(ValueError, match="must be divisible by group_size=16"):
        MergedColumnParallelLinear(
            66,
            [32, 32],
            quant_config=quant_config,
            prefix="model.layers.0.mlp.gate_up_proj",
        )


def test_nvfp4_moe_loads_split_expert_scalars_and_blocks_forward():
    layer = object.__new__(FusedMoE)
    nn.Module.__init__(layer)
    layer.has_bias = False
    layer.quant_config = SimpleNamespace(online_quant=False)
    layer.prefix = "model.layers.3.block_sparse_moe.experts"
    layer.layer_quant_config = _nvfp4_spec()
    layer.moe_parallel_config = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        dp_rank=0,
        use_ep=False,
    )
    layer.expert_map = None

    method = Nvfp4MoEMethod(_nvfp4_spec(), SimpleNamespace())
    layer.quant_method = method
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=dtypes.fp4x2,
        weight_loader=layer.weight_loader,
    )

    for shard_id, value in (("w1", 3.0), ("w3", 4.0)):
        layer.weight_loader(
            layer.w13_weight_scale_2,
            torch.tensor(value),
            "weight_scale_2",
            shard_id,
            1,
        )
        layer.weight_loader(
            layer.w13_input_scale_2,
            torch.tensor(value + 2.0),
            "input_scale_2",
            shard_id,
            1,
        )
    layer.weight_loader(
        layer.w2_weight_scale_2,
        torch.tensor(7.0),
        "weight_scale_2",
        "w2",
        1,
    )
    layer.weight_loader(
        layer.w2_input_scale_2,
        torch.tensor(8.0),
        "input_scale_2",
        "w2",
        1,
    )

    assert layer.w13_weight.shape == (2, 64, 32)
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w13_weight_scale.shape == (2, 64, 4)
    assert layer.w13_weight_scale_2[1].tolist() == [3.0, 4.0]
    assert layer.w13_input_scale_2[1].tolist() == [5.0, 6.0]
    assert layer.w2_weight_scale_2[1].item() == 7.0
    assert layer.w2_input_scale_2[1].item() == 8.0
    method.process_weights_after_loading(layer)
    method.init_prepare_finalize(layer)
    with pytest.raises(RuntimeError, match="direct NVFP4 inference"):
        method.apply(layer, None)


def test_nvfp4_moe_online_conversion_uses_each_projection_global_scale(
    monkeypatch,
):
    source = _nvfp4_spec()
    target = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp4x2,
        is_dynamic=True,
        quant_method="quark",
    )

    class FakeTargetMethod:
        def create_weights(self, layer, **_kwargs):
            layer.w13_weight = nn.Parameter(
                torch.empty(1, 64, 32, dtype=torch.uint8).view(dtypes.fp4x2),
                requires_grad=False,
            )
            layer.w2_weight = nn.Parameter(
                torch.empty(1, 64, 16, dtype=torch.uint8).view(dtypes.fp4x2),
                requires_grad=False,
            )
            layer.w13_weight_scale = nn.Parameter(
                torch.empty(1, 64, 2, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.w2_weight_scale = nn.Parameter(
                torch.empty(1, 64, 1, dtype=torch.uint8),
                requires_grad=False,
            )

    fake_target_method = FakeTargetMethod()
    monkeypatch.setattr(
        moe_mod,
        "_make_mxfp4_moe_method",
        lambda *_args, **_kwargs: fake_target_method,
    )
    converted = []

    def fake_mxfp4_quant(weight):
        converted.append(weight.clone())
        return (
            torch.zeros(
                weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8
            ).view(dtypes.fp4x2),
            torch.full(
                (weight.shape[0], weight.shape[1] // 32),
                127,
                dtype=torch.uint8,
            ).view(dtypes.fp8_e8m0),
        )

    monkeypatch.setattr(moe_mod, "quant_mxfp4_dynamic", fake_mxfp4_quant)

    layer = SimpleNamespace(
        online_quant=True,
        quant_config=SimpleNamespace(
            get_layer_quant_config=lambda *_args, **_kwargs: target
        ),
        layer_name="model.layers.3.block_sparse_moe.experts",
        layer_quant_config=source,
        params_dtype=NVFP4_DTYPE,
        quant_method=object(),
        moe_config=SimpleNamespace(),
        moe_quant_params={"params_dtype": NVFP4_DTYPE},
        _stream_online_quant=False,
        local_num_experts=1,
        intermediate_size_per_partition=32,
        use_ep=False,
        tp_size=1,
        tp_rank=0,
        w13_weight=nn.Parameter(
            torch.full((1, 64, 32), 0x22, dtype=torch.uint8),
            requires_grad=False,
        ),
        w2_weight=nn.Parameter(
            torch.full((1, 64, 16), 0x22, dtype=torch.uint8),
            requires_grad=False,
        ),
        w13_weight_scale=nn.Parameter(
            torch.full(
                (1, 64, 4), 2.0, dtype=torch.float8_e4m3fn
            ),
            requires_grad=False,
        ),
        w2_weight_scale=nn.Parameter(
            torch.full(
                (1, 64, 2), 2.0, dtype=torch.float8_e4m3fn
            ),
            requires_grad=False,
        ),
        w13_weight_scale_2=nn.Parameter(
            torch.tensor([[0.5, 1.0]]), requires_grad=False
        ),
        w2_weight_scale_2=nn.Parameter(
            torch.tensor([1.5]), requires_grad=False
        ),
        w13_input_scale_2=nn.Parameter(torch.ones(1, 2), requires_grad=False),
        w2_input_scale_2=nn.Parameter(torch.ones(1), requires_grad=False),
        _copy_quant_storage=FusedMoE._copy_quant_storage,
        _load_model_weight_or_group_weight_scale=lambda **_kwargs: None,
        _load_quant_weight_scale=lambda **_kwargs: None,
    )

    FusedMoE._online_quant(layer)

    assert len(converted) == 3
    assert torch.equal(converted[0], torch.ones(32, 64))
    assert torch.equal(converted[1], torch.full((32, 64), 2.0))
    assert torch.equal(converted[2], torch.full((64, 32), 3.0))
    assert layer.quant_method is fake_target_method
    assert layer.params_dtype == dtypes.fp4x2
    assert not hasattr(layer, "w13_weight_scale_2")
    assert not hasattr(layer, "w2_input_scale_2")


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size"),
    [(66, 32), (64, 34)],
)
def test_nvfp4_moe_rejects_non_group_aligned_dimensions(
    hidden_size, intermediate_size
):
    layer = nn.Module()
    layer.has_bias = False
    method = Nvfp4MoEMethod(_nvfp4_spec(), SimpleNamespace())

    with pytest.raises(ValueError, match="must be divisible by group_size=16"):
        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=dtypes.fp4x2,
        )


def test_minimax_m3_expert_mapping_names_nvfp4_global_scales_explicitly():
    mapping = {
        checkpoint_name: (param_name, expert_id, shard_id)
        for param_name, checkpoint_name, expert_id, shard_id in (
            make_minimax_m3_expert_params_mapping(1)
        )
    }

    assert mapping["experts.0.w1.weight_scale_2"] == (
        "experts.w13_weight_scale_2",
        0,
        "w1",
    )
    assert mapping["experts.0.w3.input_scale_2"] == (
        "experts.w13_input_scale_2",
        0,
        "w3",
    )
    assert mapping["experts.0.w2.weight_scale_2"] == (
        "experts.w2_weight_scale_2",
        0,
        "w2",
    )
    assert mapping["experts.0.w2.input_scale_2"] == (
        "experts.w2_input_scale_2",
        0,
        "w2",
    )


def test_minimax_m3_current_flat_structure_schema():
    config = SimpleNamespace(
        num_hidden_layers=4,
        layer_types=[
            "full_attention",
            "full_attention",
            "full_attention",
            "minimax_m3_sparse",
        ],
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        index_n_heads=4,
        index_head_dim=128,
        index_block_size=128,
        index_topk_blocks=16,
        index_local_blocks=1,
    )

    assert _sparse_attention_layer_ids(config) == {3}
    assert [_is_moe_layer(config, i) for i in range(4)] == [
        False,
        False,
        False,
        True,
    ]
    assert _get_sparse_attention_config(config) == {
        "sparse_num_index_heads": 4,
        "sparse_index_dim": 128,
        "sparse_block_size": 128,
        "sparse_topk_blocks": 16,
        "sparse_local_block": 1,
        "sparse_init_block": 0,
        "sparse_score_type": "max",
    }


def test_minimax_m3_blocks_nvfp4_before_compiled_backbone():
    model = object.__new__(MiniMaxM3SparseForCausalLM)
    nn.Module.__init__(model)
    model.nvfp4_load_only = True

    with pytest.raises(RuntimeError, match="weights loaded successfully"):
        model(
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
