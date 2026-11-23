# SPDX-License-Identifier: MIT
"""CPU loading regressions for BF16 sources with a padded Qwen MoE policy."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def impl():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            moe=load("atom.model_ops.moe"),
            qwen=load("atom.models.qwen4_exp"),
            config=load("atom.config"),
        )


def _bf16_layer(impl, monkeypatch, rank, streaming, *, excluded=False):
    moe = impl.moe
    config = impl.config.QuantizationConfig()
    config.online_quant = True
    config.online_global_spec = moe.LayerQuantConfig(
        quant_dtype=moe.dtypes.fp8,
        quant_type=moe.QuantType.per_Token,
        quant_method="fp8",
    )
    config.online_exclude_layers = ["mtp.*"] if excluded else []
    prefix = "mtp.layers.0.mlp.experts" if excluded else "model.layers.0.mlp.experts"
    config = impl.qwen._Qwen4ExpQuantizationConfig(config)
    source = config.get_layer_quant_config(prefix)
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", str(int(streaming)))

    layer = object.__new__(moe.FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.prefix = prefix
    layer.quant_config = config
    layer.online_quant = True
    layer.layer_quant_config = source
    layer.params_dtype = torch.bfloat16
    layer.source_is_nvfp4 = False
    layer.local_num_experts = 2
    layer.moe_parallel_config = SimpleNamespace(tp_size=4, tp_rank=rank, use_ep=False)
    layer.expert_map = None
    layer.intermediate_size_per_partition = 160
    layer.moe_config = SimpleNamespace(in_dtype=torch.bfloat16, hidden_dim=32)
    layer.quant_method = moe.UnquantizedFusedMoEMethod(layer.moe_config)
    # Exercise the allocator's handling of an explicit padding request even
    # when model-level policy excludes a source awaiting online conversion.
    layer.quant_method.pad_align = 64
    layer._stream_online_quant = moe.should_stream_online_quant(
        config, prefix, source.quant_type, layer.params_dtype
    )
    layer.moe_quant_params = {
        "num_experts": 2,
        "hidden_size": 32,
        "intermediate_size_per_partition": 160,
        "params_dtype": torch.bfloat16,
        "weight_loader": layer.weight_loader,
    }
    # Production sets prefix before allocation, and layer_name afterwards.
    layer.quant_method.create_weights(layer, **layer.moe_quant_params)
    layer.layer_name = prefix
    return layer


def _cpu_ptpc(weight, dtype):
    """Small CPU quantizer fixture; the conversion/loading code stays real."""
    rows = weight.reshape(-1, weight.shape[-1]).float()
    scale = rows.abs().amax(-1, keepdim=True).clamp_min(1e-6) / torch.finfo(dtype).max
    return (rows / scale).to(dtype), scale


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("rank", range(4))
def test_tp4_bf16_to_online_ptpc_preserves_logical_shards(
    impl, monkeypatch, rank, streaming
):
    moe = impl.moe
    layer = _bf16_layer(impl, monkeypatch, rank, streaming)
    source_w13_shape = layer.w13_weight.shape
    source_w2_shape = layer.w2_weight.shape
    assert layer._stream_online_quant is streaming
    if streaming:
        assert layer.w13_weight.is_meta and layer.w2_weight.is_meta
        # Materialize the temporary checkpoint storage on CPU, as the stream
        # loader does. This test does not exercise its asynchronous lifecycle.
        for name in ("w13_weight", "w2_weight"):
            param = getattr(layer, name)
            replacement = torch.nn.Parameter(
                torch.zeros_like(param, device="cpu"), requires_grad=False
            )
            replacement.weight_loader = layer.weight_loader
            setattr(layer, name, replacement)

    gate = (torch.arange(2 * 640 * 32).reshape(2, 640, 32) % 31 + 1).to(torch.bfloat16)
    up = -gate - 1
    # Distinct values in every rank's W2 columns detect padded shards being
    # concatenated or a physical-width offset being used during re-sharding.
    down = (
        torch.arange(2)[:, None, None]
        + torch.arange(32)[None, :, None] % 5
        + (torch.arange(640)[None, None, :] // 160) * 8
        + 1
    ).to(torch.bfloat16)
    for expert in range(2):
        layer._load_w13(layer.w13_weight.data[expert], 0, "w1", gate[expert], rank)
        layer._load_w13(layer.w13_weight.data[expert], 0, "w3", up[expert], rank)
        layer._load_w2(layer.w2_weight.data[expert], 1, down[expert], rank)

    shard = slice(rank * 160, (rank + 1) * 160)
    gathered = []

    def all_gather(local, dim):
        expert = len(gathered)
        assert not streaming
        assert dim == 1 and local.shape == (32, 160)
        torch.testing.assert_close(local, down[expert, :, shard], rtol=0, atol=0)
        gathered.append(expert)
        return torch.cat(
            [down[expert, :, r * 160 : (r + 1) * 160] for r in range(4)], dim=1
        )

    monkeypatch.setattr(
        moe,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, all_gather=all_gather),
    )

    def quantize(weight, *, online_quant_type, online_quant_dtype):
        assert online_quant_type == moe.QuantType.per_Token
        assert online_quant_dtype == moe.dtypes.fp8
        return _cpu_ptpc(weight, online_quant_dtype)

    monkeypatch.setattr(moe, "quant_weight_online", quantize)
    layer._online_quant()

    assert source_w13_shape == (2, 320, 32)
    assert source_w2_shape == (2, 32, 160)
    assert isinstance(layer.quant_method, moe.Fp8MoEMethod)
    assert layer.w13_weight.shape == (2, 320, 32)
    assert layer.w2_weight.shape == (2, 32, 160)
    assert layer.intermediate_size_per_partition == 160
    assert layer._stream_online_quant is False
    assert gathered == ([] if streaming else [0, 1])
    for expert in range(2):
        q_gate, s_gate = _cpu_ptpc(gate[expert, shard], moe.dtypes.fp8)
        q_up, s_up = _cpu_ptpc(up[expert, shard], moe.dtypes.fp8)
        q_down, s_down = _cpu_ptpc(
            down[expert, :, shard] if streaming else down[expert], moe.dtypes.fp8
        )
        if not streaming:
            q_down = q_down[:, shard]
        torch.testing.assert_close(
            layer.w13_weight[expert].view(torch.uint8),
            torch.cat((q_gate, q_up)).view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer.w2_weight[expert].view(torch.uint8),
            q_down.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer.w13_weight_scale[expert],
            torch.cat((s_gate, s_up)).squeeze(-1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer.w2_weight_scale[expert], s_down.squeeze(-1), rtol=0, atol=0
        )


@pytest.mark.parametrize("streaming", [False, True])
def test_online_excluded_bf16_mtp_keeps_runtime_padding(impl, monkeypatch, streaming):
    layer = _bf16_layer(impl, monkeypatch, rank=0, streaming=streaming, excluded=True)
    assert not layer._stream_online_quant
    assert not layer.w13_weight.is_meta
    assert layer.w13_weight.shape == (2, 384, 32)
    assert layer.w2_weight.shape == (2, 32, 192)
    assert not layer.w13_weight.count_nonzero()
    assert not layer.w2_weight.count_nonzero()
    original = (layer.w13_weight, layer.w2_weight)
    layer._online_quant()
    assert layer.w13_weight is original[0] and layer.w2_weight is original[1]
    assert isinstance(layer.quant_method, impl.moe.UnquantizedFusedMoEMethod)
