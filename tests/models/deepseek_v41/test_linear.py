# SPDX-License-Identifier: MIT
"""Exercise ATOM's actual parallel loaders with nonuniform compact source scales."""

from types import SimpleNamespace

import pytest
import torch

from atom.quant_spec import LayerQuantConfig


@pytest.fixture
def linear_modules(monkeypatch):
    pytest.importorskip("aiter")
    from atom.model_ops import linear

    group = SimpleNamespace(rank_in_group=0, world_size=4)
    monkeypatch.setattr(linear, "get_tp_group", lambda: group)
    return linear, group


def _config(block_rows=32):
    from aiter import QuantType

    spec = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=torch.float8_e4m3fn,
        weight_block_size=(block_rows, 32),
    )
    return SimpleNamespace(get_layer_quant_config=lambda _: spec, online_quant=False)


def _online_quant_config(block_rows=1):
    """A real ``QuantizationConfig``: MXFP8 on disk, ptpc_fp8 online.

    The stub above cannot stand in here -- the decision under test reads the
    *online* side of the config for the layer's own prefix, which only the real
    resolver knows how to answer.
    """
    from transformers import PretrainedConfig

    from atom.config import QuantizationConfig

    hf = PretrainedConfig()
    hf.torch_dtype = torch.bfloat16
    hf.quantization_config = {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [block_rows, 32],
    }
    return QuantizationConfig(
        hf,
        online_quant_config={
            "global_quant_config": "ptpc_fp8",
            "exclude_layer": ["*block_sparse_moe"],
        },
    )


def test_online_requantized_source_drops_the_native_layout(linear_modules):
    """MiniMax-M3-MXFP8's attention linears, which online quant overwrites.

    Reading the model-wide `online_quant` flag instead of this layer's own
    answer made every one of them raise at construction, which is a server that
    never starts rather than a layer that picks the other GEMM.
    """
    linear, group = linear_modules
    group.world_size = 1
    module = linear.ColumnParallelLinear(
        128,
        256,
        quant_config=_online_quant_config(),
        prefix="language_model.model.layers.3.self_attn.qkv_proj",
    )
    assert module.native_a8_group_rows is None
    assert module.weight.shape == (256, 128)
    assert module.weight_scale.shape == (256, 4)


def test_online_excluded_source_keeps_the_native_layout(linear_modules):
    linear, group = linear_modules
    group.world_size = 1
    module = linear.ColumnParallelLinear(
        128,
        256,
        quant_config=_online_quant_config(),
        prefix="language_model.model.layers.3.block_sparse_moe.shared_experts.gate_up_proj",
    )
    assert module.native_a8_group_rows == 1
    assert module.weight_scale.shape == (256, 4)


def test_online_requantized_32x32_source_still_rejected(linear_modules):
    linear, group = linear_modules
    group.world_size = 1
    with pytest.raises(ValueError, match="Native group32 A8"):
        linear.ColumnParallelLinear(
            128,
            256,
            quant_config=_online_quant_config(block_rows=32),
            prefix="language_model.model.layers.3.self_attn.qkv_proj",
        )


@pytest.mark.parametrize("kind", ["ColumnParallelLinear", "RowParallelLinear"])
def test_parallel_source_slices_and_scale_alignment(linear_modules, kind):
    linear, group = linear_modules
    torch.manual_seed(333)
    weight = torch.randn(256, 128).to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.arange(32).remainder(8).reshape(8, 4).float() - 4).to(
        torch.float8_e8m0fnu
    )
    axis = 0 if kind == "ColumnParallelLinear" else 1
    for rank in range(4):
        group.rank_in_group = rank
        module = getattr(linear, kind)(128, 256, quant_config=_config())
        module.weight_loader(module.weight, weight)
        module.weight_loader(module.weight_scale, scales)
        module.process_weights_after_loading()
        assert torch.equal(
            module.weight.view(torch.uint8),
            weight.chunk(4, axis)[rank].view(torch.uint8),
        )
        assert torch.equal(
            module.weight_scale.view(torch.uint8),
            scales.chunk(4, axis)[rank].view(torch.uint8),
        )
        assert not getattr(module.weight, "is_shuffled", False)


@pytest.mark.parametrize("shard_ids", [None, (0, 1)])
def test_merged_column_splits_compact_scale_rows(linear_modules, shard_ids):
    linear, group = linear_modules
    weight = (
        torch.arange(384 * 64, dtype=torch.float32)
        .remainder(64)
        .reshape(384, 64)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.exp2(torch.arange(24).reshape(12, 2).remainder(5).float()).to(
        torch.float8_e8m0fnu
    )
    for rank in range(4):
        group.rank_in_group = rank
        module = linear.MergedColumnParallelLinear(
            64, [256, 128], quant_config=_config()
        )
        module.weight_loader(module.weight, weight, shard_ids)
        module.weight_loader(module.weight_scale, scales, shard_ids)
        expected_w = torch.cat(
            [
                weight.view(torch.uint8)[:256].chunk(4)[rank],
                weight.view(torch.uint8)[256:].chunk(4)[rank],
            ]
        )
        expected_s = torch.cat(
            [
                scales.view(torch.uint8)[:8].chunk(4)[rank],
                scales.view(torch.uint8)[8:].chunk(4)[rank],
            ]
        )
        assert torch.equal(module.weight.view(torch.uint8), expected_w)
        assert torch.equal(module.weight_scale.view(torch.uint8), expected_s)


def test_column_row_views_use_source_scale_groups(linear_modules):
    linear, group = linear_modules
    group.world_size = 1
    module = linear.ColumnParallelLinear(64, 128, quant_config=_config())
    module.weight_scale.view(torch.uint8).copy_(
        torch.arange(8, dtype=torch.uint8).reshape(4, 2) + 120
    )
    view = module.make_row_view(32, 64)
    assert view.weight.shape == (64, 64)
    assert torch.equal(
        view.weight_scale.view(torch.uint8), module.weight_scale.view(torch.uint8)[1:3]
    )
    assert view.weight_scale.data_ptr() == module.weight_scale[1:].data_ptr()
    with pytest.raises(AssertionError, match="32-aligned"):
        module.make_row_view(16, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_native_linear_dispatch_keeps_a8_qat(linear_modules):
    from . import oracle_kernels as oracle

    linear, group = linear_modules
    group.world_size = 1
    torch.manual_seed(991)
    module = linear.ReplicatedLinear(288, 96, quant_config=_config()).cuda()
    weight = (torch.randn(96, 288) * 32).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-8, -1, (3, 9)).float()).to(torch.float8_e8m0fnu)
    module.weight_loader(module.weight, weight)
    module.weight_loader(module.weight_scale, scale)
    module.process_weights_after_loading()
    x = torch.randn(5, 288, dtype=torch.bfloat16)
    q, s = oracle.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    activation = (q.float().reshape(5, 9, 32) * s.float()[:, :, None]).reshape(5, 288)
    full_weight = weight.float() * scale.float().repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    expected = torch.nn.functional.linear(activation, full_weight)
    # Bound tied to the output magnitude; see test_quant_gpu for why the
    # native microscaling MFMA cannot be held to a per-element rtol.
    torch.testing.assert_close(
        module(x.cuda(), otype=torch.float32).cpu(),
        expected,
        rtol=3e-5,
        atol=5e-5 * expected.abs().max().item(),
    )


def test_merged_replicated_splits_v41_qkv_a_scale_rows(linear_modules):
    """The fused `attn.wqkv_a` must land each disk shard on its own rows.

    V4.1's widths are the point: 1280 query-LoRA rows then 512 KV rows, both
    scaled per 32x32, so the scale shard boundary is row 40 and nothing about
    it is checked by a uniform split. A shard offset computed on the weight
    grid instead of the scale grid puts the KV scales 1240 rows too far in and
    the GEMM reads whatever was there.
    """
    linear, group = linear_modules
    group.world_size = 1
    module = linear.MergedReplicatedLinear(5120, [1280, 512], quant_config=_config())
    assert module.weight.shape == (1792, 5120)
    assert module.weight_scale.shape == (56, 160)
    shards = {
        0: (
            torch.full((1280, 5120), 3.0).to(torch.float8_e4m3fn),
            torch.full((40, 160), 4.0).to(torch.float8_e8m0fnu),
        ),
        1: (
            torch.full((512, 5120), 7.0).to(torch.float8_e4m3fn),
            torch.full((16, 160), 64.0).to(torch.float8_e8m0fnu),
        ),
    }
    for shard_id, (weight, scale) in shards.items():
        module.weight_loader(module.weight, weight, shard_id)
        module.weight_loader(module.weight_scale, scale, shard_id)
    assert torch.equal(module.weight[:1280].float(), shards[0][0].float())
    assert torch.equal(module.weight[1280:].float(), shards[1][0].float())
    assert torch.equal(module.weight_scale[:40].float(), shards[0][1].float())
    assert torch.equal(module.weight_scale[40:].float(), shards[1][1].float())
