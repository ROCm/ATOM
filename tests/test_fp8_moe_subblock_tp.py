# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block FP8 expert shards smaller than their checkpoint's quantization block."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.model_ops.fused_moe.expert_layout import block_quantized_intermediate_size


@pytest.fixture
def moe_module():
    with atom_without_aiter() as load:
        yield load("atom.model_ops.moe")


def _layer(
    moe,
    monkeypatch,
    method,
    rank,
    logical=64,
    tp=8,
    experts=2,
    stream=False,
    quant_type=None,
    hidden=256,
):
    parallel = SimpleNamespace(
        tp_size=tp,
        tp_rank=rank,
        dp_size=1,
        dp_rank=0,
        ep_size=1,
        ep_rank=0,
        use_ep=False,
        use_mori_kernels=False,
    )
    monkeypatch.setattr(
        moe, "get_tp_group", lambda: SimpleNamespace(world_size=tp, rank_in_group=rank)
    )
    # The public class's plugin wrapper subclasses the actual module class.
    cls = moe.FusedMoE.__mro__[1]
    layer = object.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.moe_parallel_config = parallel
    layer._stream_online_quant = stream
    layer.intermediate_size_per_partition = logical
    layer.hidden_size = hidden
    layer.local_num_experts = experts
    layer.global_num_experts = experts
    layer.num_redundant_experts = 0
    layer.expert_map = None
    layer.shared_expert_staging = None
    layer.expert_layout = moe.MoEExpertLayout.make(
        num_routed=experts,
        num_fused_shared_experts=0,
        num_configured_redundant=0,
        ep_size=1,
        use_all2all=False,
        eplb_enabled=False,
    )
    if quant_type is None:
        quant_type = moe.QuantType.per_1x128
    layer.layer_quant_config = moe.LayerQuantConfig(
        quant_type=quant_type,
        quant_dtype=torch.float8_e4m3fn,
        weight_block_size=(128, 128) if quant_type == moe.QuantType.per_1x128 else None,
        is_dynamic=True,
    )
    config = moe.FusedMoEConfig(
        num_experts=experts,
        experts_per_token=1,
        hidden_dim=hidden,
        num_local_experts=experts,
        moe_parallel_config=parallel,
        expert_layout=layer.expert_layout,
    )
    layer.quant_method = getattr(moe, method)(layer.layer_quant_config, config)
    layer.quant_method.create_weights(
        layer,
        experts,
        hidden,
        logical,
        torch.float8_e4m3fn,
        weight_loader=layer.weight_loader,
    )
    return layer


def _checkpoint(logical=64, tp=8, experts=2):
    generator = torch.Generator().manual_seed(1944)
    full = logical * tp
    weights = {
        "w1": torch.randint(-3, 4, (experts, full, 256), generator=generator),
        "w3": torch.randint(-3, 4, (experts, full, 256), generator=generator),
        "w2": torch.randint(-3, 4, (experts, 256, full), generator=generator),
    }
    weights = {name: value.to(torch.float8_e4m3fn) for name, value in weights.items()}
    scales = {}
    for offset, (name, weight) in enumerate(weights.items(), start=1):
        shape = (
            experts,
            (weight.shape[1] + 127) // 128,
            (weight.shape[2] + 127) // 128,
        )
        scales[name] = (
            torch.arange(1, 1 + shape[0] * shape[1] * shape[2]).reshape(shape)
            + offset * 19
        ).float() / 256
    return weights, scales


def _load(layer, weights, scales, staged=False, stacked=False):
    for suffix, tensors in (("weight", weights), ("weight_scale", scales)):
        pending = {}
        for shard_id, tensor in tensors.items():
            name = ("w2_" if shard_id == "w2" else "w13_") + suffix
            param = getattr(layer, name)
            if stacked:
                layer.weight_loader(param, tensor, name, shard_id)
                continue
            if staged and name not in pending:
                pending[name] = (torch.zeros_like(param), set())
            for expert in range(tensor.shape[0]):
                if staged:
                    staging, filled = pending[name]
                    assert layer.stage_expert_weight(
                        param, staging, tensor[expert], expert, shard_id, name
                    )
                    filled.add((expert, shard_id))
                else:
                    layer.weight_loader(param, tensor[expert], name, shard_id, expert)
        for name, (staging, filled) in pending.items():
            layer.flush_staged(getattr(layer, name), staging, filled)


def _dequant(weight, scale):
    rows, cols = weight.shape[-2:]
    expanded = scale.repeat_interleave(128, -2).repeat_interleave(128, -1)
    return weight.float() * expanded[..., :rows, :cols]


@pytest.mark.parametrize("method", ["Fp8MoEMethod", "CompressedTensorsFp8MoEMethod"])
@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("load_mode", ["direct", "staged", "stacked"])
def test_half_block_scales_and_weights_match_each_rank_reference(
    moe_module, monkeypatch, method, rank, load_mode
):
    layer = _layer(moe_module, monkeypatch, method, rank)
    weights, scales = _checkpoint()
    assert layer.intermediate_size_per_partition == 64
    assert layer.w13_weight.shape == (2, 256, 256)
    assert layer.w2_weight.shape == (2, 256, 128)
    # Exercise loading over existing storage, as a weight update does.
    layer.w13_weight.data.view(torch.uint8).fill_(64)
    layer.w2_weight.data.view(torch.uint8).fill_(64)
    _load(layer, weights, scales, load_mode == "staged", load_mode == "stacked")

    for expert in range(2):
        source = slice(rank * 64, (rank + 1) * 64)
        gate, up = layer.w13_weight[expert].chunk(2, dim=0)
        assert torch.equal(gate[:64].float(), weights["w1"][expert, source].float())
        assert torch.equal(up[:64].float(), weights["w3"][expert, source].float())
        assert torch.count_nonzero(gate[64:].float()) == 0
        assert torch.count_nonzero(up[64:].float()) == 0
        assert torch.count_nonzero(layer.w2_weight[expert, :, 64:].float()) == 0

        inputs = torch.linspace(-0.5, 0.5, 3 * 256).reshape(3, 256)
        full_gate = _dequant(weights["w1"][expert], scales["w1"][expert])
        full_up = _dequant(weights["w3"][expert], scales["w3"][expert])
        full_down = _dequant(weights["w2"][expert], scales["w2"][expert])
        # Dequantize the whole checkpoint before taking logical TP slices.
        # This oracle does not calculate or reuse the loader's scale-block index.
        gate_scale, up_scale = layer.w13_weight_scale[expert].chunk(2, dim=0)
        loaded_gate = _dequant(gate, gate_scale)
        loaded_up = _dequant(up, up_scale)
        loaded_down = _dequant(layer.w2_weight[expert], layer.w2_weight_scale[expert])
        torch.testing.assert_close(loaded_gate[:64], full_gate[source], rtol=0, atol=0)
        torch.testing.assert_close(loaded_up[:64], full_up[source], rtol=0, atol=0)
        torch.testing.assert_close(
            loaded_down[:, :64], full_down[:, source], rtol=0, atol=0
        )
        expected = (
            torch.nn.functional.silu(inputs @ full_gate[source].T)
            * (inputs @ full_up[source].T)
        ) @ full_down[:, source].T
        actual = (
            torch.nn.functional.silu(inputs @ loaded_gate.T) * (inputs @ loaded_up.T)
        ) @ loaded_down.T
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert layer.quant_method.get_fused_moe_quant_config(layer).block_shape == [
        128,
        128,
    ]


@pytest.mark.parametrize("method", ["Fp8MoEMethod", "CompressedTensorsFp8MoEMethod"])
@pytest.mark.parametrize(
    ("logical", "tp"), [(16, 8), (32, 8), (128, 8), (256, 8), (512, 1), (128, 4)]
)
def test_smaller_subblocks_and_aligned_shards(
    moe_module, monkeypatch, method, logical, tp
):
    layer = _layer(moe_module, monkeypatch, method, tp - 1, logical, tp)
    weights, scales = _checkpoint(logical, tp)
    _load(layer, weights, scales)
    physical = max(128, logical)
    assert layer.w13_weight.shape == (2, physical * 2, 256)
    assert layer.w2_weight.shape == (2, 256, physical)
    assert torch.equal(
        _dequant(layer.w2_weight, layer.w2_weight_scale)[..., :logical],
        _dequant(weights["w2"], scales["w2"])[..., (tp - 1) * logical : tp * logical],
    )


@pytest.mark.parametrize("method", ["Fp8MoEMethod", "CompressedTensorsFp8MoEMethod"])
@pytest.mark.parametrize("logical", [96, 192])
def test_cross_block_unaligned_shards_remain_rejected(
    moe_module, monkeypatch, method, logical
):
    with pytest.raises(ValueError, match="block_n"):
        _layer(moe_module, monkeypatch, method, 0, logical)


def test_group32_alignment_contract_is_unchanged():
    with pytest.raises(ValueError, match="block_k"):
        block_quantized_intermediate_size(16, 1, 32, 8)
    assert block_quantized_intermediate_size(32, 1, 32, 8) == 32


@pytest.mark.parametrize("method", ["Fp8MoEMethod", "CompressedTensorsFp8MoEMethod"])
@pytest.mark.parametrize("logical", [8, 16])
def test_group32_subgroup_tp_shards_remain_rejected(
    moe_module, monkeypatch, method, logical
):
    with pytest.raises(ValueError, match="block_k = 32"):
        _layer(
            moe_module,
            monkeypatch,
            method,
            0,
            logical,
            quant_type=moe_module.QuantType.per_1x32,
        )


@pytest.mark.parametrize(
    ("method", "physical"),
    [("Fp8MoEMethod", 256), ("CompressedTensorsFp8MoEMethod", 32)],
)
def test_group32_aligned_shards_keep_existing_storage_and_kernel_metadata(
    moe_module, monkeypatch, method, physical
):
    layer = _layer(
        moe_module,
        monkeypatch,
        method,
        7,
        logical=32,
        quant_type=moe_module.QuantType.per_1x32,
    )
    # Existing parsers can leave the optional checkpoint block metadata unset.
    assert layer.layer_quant_config.weight_block_size is None
    assert layer.intermediate_size_per_partition == 32
    assert layer.w13_weight.shape == (2, 2 * physical, 256)
    assert layer.w2_weight.shape == (2, 256, physical)
    assert layer.w13_weight_scale.shape == (2, 2 * physical, 8)
    assert layer.w2_weight_scale.shape == (2, 256, physical // 32)
    expected_scale_dtype = (
        moe_module.dtypes.fp8_e8m0 if method == "Fp8MoEMethod" else torch.float32
    )
    assert layer.w13_weight_scale.dtype == expected_scale_dtype
    assert layer.w2_weight_scale.dtype == expected_scale_dtype
    assert layer.quant_method.get_fused_moe_quant_config(layer).block_shape == [1, 32]


@pytest.mark.parametrize("logical", [64, 128])
def test_streaming_source_keeps_checkpoint_shape_until_target_creation(
    moe_module, monkeypatch, logical
):
    layer = _layer(moe_module, monkeypatch, "Fp8MoEMethod", 0, logical, stream=True)
    assert layer.w13_weight.is_meta and layer.w2_weight.is_meta
    assert layer.w13_weight.shape == (2, 2 * logical, 256)
    assert layer.w2_weight.shape == (2, 256, logical)
    layer._stream_online_quant = False
    layer.quant_method.create_weights(layer, 2, 256, logical, torch.float8_e4m3fn)
    assert not layer.w13_weight.is_meta and not layer.w2_weight.is_meta
    assert layer.w13_weight.shape == (2, 256, 256)
    assert layer.w2_weight.shape == (2, 256, 128)
