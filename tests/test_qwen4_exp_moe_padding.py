# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Logical TP checkpoint slices in physically padded Qwen expert buffers."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", exc_type=ImportError)

from aiter import QuantType, dtypes

from atom.model_ops.moe import (
    CompressedTensorsFp8MoEMethod,
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from atom.models import qwen4_exp
from atom.quant_spec import LayerQuantConfig


def _layer(mode, tp, rank, experts=4, hidden=32):
    # Bypass the plugin decorator's __new__, which constructs a whole layer.
    # The actual inherited loaders and method allocators are exercised below.
    layer = object.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.moe_parallel_config = SimpleNamespace(tp_size=tp, tp_rank=rank)
    layer.expert_layout = SimpleNamespace(num_routed=experts - 1)
    layer.expert_map = None
    layer.intermediate_size_per_partition = 640 // tp
    moe = SimpleNamespace(in_dtype=torch.bfloat16, hidden_dim=hidden)
    if mode == "fp8":
        method = CompressedTensorsFp8MoEMethod(
            LayerQuantConfig(
                quant_dtype=dtypes.fp8,
                quant_type=QuantType.per_Token,
                quant_method="compressed-tensors",
            ),
            moe,
        )
        dtype = dtypes.fp8
    else:
        method = UnquantizedFusedMoEMethod(moe)
        dtype = torch.bfloat16
    method.pad_align = 64
    method.create_weights(
        layer,
        experts,
        hidden,
        layer.intermediate_size_per_partition,
        dtype,
        weight_loader=lambda *args, **kwargs: None,
    )
    return layer, method


@pytest.mark.parametrize(
    "tp,rank", [(2, r) for r in range(2)] + [(4, r) for r in range(4)]
)
@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_per_expert_weight_and_channel_scale_shards(tp, rank, mode):
    layer, _ = _layer(mode, tp, rank)
    logical = 640 // tp
    physical = 320 if tp == 2 else 192
    assert layer.intermediate_size_per_partition == logical
    assert layer.w13_weight.shape == (4, 2 * physical, 32)
    assert layer.w2_weight.shape == (4, 32, physical)
    # Gate and up have distinct values, including the FP8 byte-copy case.
    dtype = torch.float8_e4m3fn if mode == "fp8" else torch.bfloat16
    for expert in range(4):
        for half, shard_id in enumerate(("w1", "w3")):
            source = (
                (torch.arange(640 * 32).reshape(640, 32) % 13) + half + expert
            ).to(dtype)
            layer._load_w13(layer.w13_weight.data[expert], 0, shard_id, source, rank)
            target = layer.w13_weight.data[
                expert, half * physical : (half + 1) * physical
            ]
            torch.testing.assert_close(
                target[:logical].view(torch.uint8),
                source[rank * logical : (rank + 1) * logical].view(torch.uint8),
                rtol=0,
                atol=0,
            )
            assert not target[logical:].view(torch.uint8).count_nonzero()
            if mode == "fp8":
                scale = torch.arange(640, dtype=torch.float32) + 1 + half * 640
                layer._load_per_channel_weight_scale(
                    layer.w13_weight_scale.data[expert],
                    0,
                    shard_id,
                    scale[:, None],
                    rank,
                )
                target_scale = layer.w13_weight_scale.data[
                    expert, half * physical : (half + 1) * physical, 0
                ]
                torch.testing.assert_close(
                    target_scale[:logical], scale[rank * logical : (rank + 1) * logical]
                )
                assert not target_scale[logical:].count_nonzero()
        down = (torch.arange(32 * 640).reshape(32, 640) % 17).to(dtype)
        layer._load_w2(layer.w2_weight.data[expert], 1, down, rank)
        torch.testing.assert_close(
            layer.w2_weight.data[expert, :, :logical].view(torch.uint8),
            down[:, rank * logical : (rank + 1) * logical]
            .contiguous()
            .view(torch.uint8),
            rtol=0,
            atol=0,
        )
        assert (
            not layer.w2_weight.data[expert, :, logical:]
            .view(torch.uint8)
            .count_nonzero()
        )
        if mode == "fp8":
            scale = torch.arange(32, dtype=torch.float32)[:, None] + 1
            layer._load_per_channel_weight_scale(
                layer.w2_weight_scale.data[expert], 0, "w2", scale, rank
            )
            torch.testing.assert_close(layer.w2_weight_scale.data[expert], scale)


@pytest.mark.parametrize(
    "tp,rank", [(2, r) for r in range(2)] + [(4, r) for r in range(4)]
)
def test_stacked_loader_preserves_shared_slot_and_zero_pads(tp, rank):
    layer, _ = _layer("bf16", tp, rank)
    physical = layer.w2_weight.shape[2]
    logical = 640 // tp
    layer.w13_weight.data.fill_(-7)
    layer.w2_weight.data.fill_(-7)
    qwen4_exp.install_stacked_expert_loaders(layer)
    gate = torch.arange(3 * 640 * 32).reshape(3, 640, 32).to(torch.bfloat16)
    up = -gate - 1
    down = torch.arange(3 * 32 * 640).reshape(3, 32, 640).to(torch.bfloat16)
    layer.w13_weight.weight_loader(layer.w13_weight, torch.cat((gate, up), dim=1))
    layer.w2_weight.weight_loader(layer.w2_weight, down)
    for half, source in enumerate((gate, up)):
        actual = layer.w13_weight[:3, half * physical : (half + 1) * physical]
        torch.testing.assert_close(
            actual[:, :logical], source[:, rank * logical : (rank + 1) * logical]
        )
        assert not actual[:, logical:].count_nonzero()
    torch.testing.assert_close(
        layer.w2_weight[:3, :, :logical],
        down[:, :, rank * logical : (rank + 1) * logical],
    )
    assert not layer.w2_weight[:3, :, logical:].count_nonzero()
    assert (layer.w13_weight[3] == -7).all()
    assert (layer.w2_weight[3] == -7).all()


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_preshuffle_preserves_final_parameter_markers_and_padding(tp, mode):
    from aiter.ops.shuffle import shuffle_weight

    layer, method = _layer(mode, tp, 0)
    layer.w13_weight.data.view(torch.uint8).zero_()
    layer.w2_weight.data.view(torch.uint8).zero_()
    logical = 640 // tp
    physical = layer.w2_weight.shape[-1]
    # Nonzero logical data ensures an omitted shuffle cannot pass this test.
    layer.w13_weight.data[:, :logical] = (
        torch.arange(logical * 32)
        .reshape(logical, 32)
        .remainder(11)
        .to(layer.w13_weight.dtype)
    )
    layer.w13_weight.data[:, physical : physical + logical] = 2
    layer.w2_weight.data[:, :, :logical] = 3
    expected13 = shuffle_weight(layer.w13_weight.data.clone())
    expected2 = shuffle_weight(layer.w2_weight.data.clone())
    if mode == "fp8":
        # Test the layout contract independently of fn -> fnuz normalization.
        method.need_normalize_e4m3fn_to_e4m3fnuz = False
    method.process_weights_after_loading(layer)
    assert layer.w13_weight.is_shuffled and layer.w2_weight.is_shuffled
    torch.testing.assert_close(
        layer.w13_weight.view(torch.uint8), expected13.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        layer.w2_weight.view(torch.uint8), expected2.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_gfx942_alignment_policy(monkeypatch, tp, mode):
    monkeypatch.setattr(qwen4_exp, "get_gfx_runtime", lambda: "gfx942")
    spec = (
        LayerQuantConfig(
            quant_dtype=dtypes.fp8,
            quant_type=QuantType.per_Token,
            quant_method="compressed-tensors",
        )
        if mode == "fp8"
        else LayerQuantConfig()
    )
    assert qwen4_exp.moe_pad_align(640, tp, spec) == 64


def test_gfx942_blockscale_keeps_existing_alignment(monkeypatch):
    monkeypatch.setattr(qwen4_exp, "get_gfx_runtime", lambda: "gfx942")
    blockscale = LayerQuantConfig(
        quant_dtype=dtypes.fp8,
        quant_type=QuantType.per_1x128,
        quant_method="compressed-tensors",
    )
    assert qwen4_exp.moe_pad_align(640, 2, blockscale) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_gpu_padded_moe_matches_torch_reference(tp, mode):
    """Real 513/topk11 shape, both quantization modes, eager and graph replay."""
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe, torch_moe_stage1, torch_moe_stage2
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz

    if get_gfx_runtime() != "gfx942":
        pytest.skip("gfx942 tuned Qwen MoE contract")
    with torch.device("cuda"), torch.inference_mode():
        torch.manual_seed(19)
        layer, method = _layer(mode, tp, 0, experts=513, hidden=2560)
        logical, physical = 640 // tp, layer.w2_weight.shape[-1]
        # A deterministic selected subset keeps the reference small while
        # retaining the actual kernel's expert count and shared id 512.
        ids = torch.tensor(
            [0, 3, 11, 49, 87, 131, 207, 311, 402, 511, 512], dtype=torch.int32
        )
        topk_ids = ids[None].expand(4, -1).contiguous()
        topk_weights = torch.rand(4, 11, dtype=torch.float32)
        topk_weights[:, :10] /= topk_weights[:, :10].sum(-1, keepdim=True)
        hidden = torch.randn(4, 2560, dtype=torch.bfloat16) * 0.1
        layer.w13_weight.data.view(torch.uint8).zero_()
        layer.w2_weight.data.view(torch.uint8).zero_()
        source_dtype = torch.float8_e4m3fn if mode == "fp8" else torch.bfloat16
        for e in ids.tolist():
            gate = (torch.randn(logical, 2560) * 0.02).to(source_dtype)
            up = (torch.randn(logical, 2560) * 0.02).to(source_dtype)
            down = (torch.randn(2560, logical) * 0.02).to(source_dtype)
            layer._copy_quant_storage(layer.w13_weight.data[e, :logical], gate)
            layer._copy_quant_storage(
                layer.w13_weight.data[e, physical : physical + logical], up
            )
            layer._copy_quant_storage(layer.w2_weight.data[e, :, :logical], down)
        qtype = QuantType.per_Token if mode == "fp8" else QuantType.No
        raw13, raw2 = layer.w13_weight.data, layer.w2_weight.data
        s13 = getattr(layer, "w13_weight_scale", None)
        s2 = getattr(layer, "w2_weight_scale", None)
        if mode == "fp8":
            raw13, s13, _ = normalize_e4m3fn_to_e4m3fnuz(raw13.clone(), s13.clone())
            raw2, s2, _ = normalize_e4m3fn_to_e4m3fnuz(raw2.clone(), s2.clone())
        stage1 = torch_moe_stage1(
            hidden,
            raw13,
            raw2,
            topk_weights,
            topk_ids,
            dtype=torch.bfloat16,
            activation=ActivationType.Silu,
            quant_type=qtype,
            a1_scale=torch.ones(1),
            w1_scale=s13,
        )
        reference = torch_moe_stage2(
            stage1.view(4, 11, -1),
            raw13,
            raw2,
            topk_weights,
            topk_ids,
            dtype=torch.bfloat16,
            quant_type=qtype,
            a2_scale=torch.ones(1),
            w2_scale=s2,
        )
        method.process_weights_after_loading(layer)

        def run():
            return fused_moe(
                hidden,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                activation=ActivationType.Silu,
                quant_type=qtype,
                w1_scale=getattr(layer, "w13_weight_scale", None),
                w2_scale=getattr(layer, "w2_weight_scale", None),
            )

        actual = run()
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        x, y = reference.float(), actual.float()
        relative_l2 = (x - y).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)
        logits_diff = 1 - 2 * (x * y).sum() / (x.square().sum() + y.square().sum())
        assert relative_l2.max() < 0.2
        assert logits_diff < 1e-3
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = run()
        graph.replay()
        torch.cuda.synchronize()
        # BF16 route accumulation is atomic; graph scheduling can change its
        # rounding near zero. Compare in token RMS units, as AITER's reference
        # tests do, rather than applying a relative tolerance to cancellations.
        rms = x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        graph_y = replayed.float()
        assert torch.isfinite(graph_y).all()
        assert ((x - graph_y).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).max() < 0.2
        assert (
            1 - 2 * (x * graph_y).sum() / (x.square().sum() + graph_y.square().sum())
            < 1e-3
        )
        torch.testing.assert_close(
            graph_y / rms, actual.float() / rms, rtol=0, atol=0.1
        )
        print(
            f"{mode} TP{tp}: relative_l2={relative_l2.max().item():.6f}, logits_diff={logits_diff.item():.6g}, graph_rms_diff={((graph_y - y).abs() / rms).max().item():.6f}"
        )
