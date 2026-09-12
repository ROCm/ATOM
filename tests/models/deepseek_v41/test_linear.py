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
    torch.testing.assert_close(
        module(x.cuda(), otype=torch.float32).cpu(), expected, rtol=3e-5, atol=3e-4
    )
