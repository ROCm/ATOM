# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Logical TP checkpoint slices in physically padded Qwen expert buffers."""

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
            mtp=load("atom.models.qwen4_exp_mtp"),
            config=load("atom.config"),
            topk=load("atom.model_ops.topK"),
            loading=load("atom.model_loader.loading_core"),
            names=load("atom.model_loader.weight_names"),
            iterator=load("atom.model_loader.weight_iterator"),
            utils=load("atom.model_ops.utils"),
            shuffle=load("aiter.ops.shuffle"),
        )


def _layer(impl, mode, tp, rank, experts=4, hidden=32):
    # Bypass the plugin decorator's __new__, which constructs a whole layer.
    # The actual inherited loaders and method allocators are exercised below.
    m = impl.moe
    layer = object.__new__(m.FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.mlp.experts"
    layer.quant_config = None
    layer.moe_parallel_config = SimpleNamespace(tp_size=tp, tp_rank=rank)
    layer.expert_layout = SimpleNamespace(num_routed=experts - 1)
    layer.expert_map = None
    layer.intermediate_size_per_partition = 640 // tp
    moe = SimpleNamespace(in_dtype=torch.bfloat16, hidden_dim=hidden)
    if mode == "fp8":
        method = m.CompressedTensorsFp8MoEMethod(
            m.LayerQuantConfig(
                quant_dtype=m.dtypes.fp8,
                quant_type=m.QuantType.per_Token,
                quant_method="compressed-tensors",
            ),
            moe,
        )
        dtype = m.dtypes.fp8
    else:
        method = m.UnquantizedFusedMoEMethod(moe)
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
def test_per_expert_weight_and_channel_scale_shards(impl, tp, rank, mode):
    layer, _ = _layer(impl, mode, tp, rank)
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
def test_stacked_loader_preserves_shared_slot_and_zero_pads(impl, tp, rank):
    layer, _ = _layer(impl, "bf16", tp, rank)
    physical = layer.w2_weight.shape[2]
    logical = 640 // tp
    layer.w13_weight.data.fill_(-7)
    layer.w2_weight.data.fill_(-7)
    impl.qwen.install_stacked_expert_loaders(layer)
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


@pytest.mark.parametrize("fused_shared", [False, True])
def test_mtp_shared_name_rewrite_follows_constructed_layout(impl, fused_shared):
    Qwen4ExpMTP = impl.mtp.Qwen4ExpMTP
    model = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=512)
    model.fused_shared_experts = int(fused_shared)
    model.disable_fused_shared_loading = not fused_shared

    def target_policy(*_):
        raise AssertionError("Draft names must follow its constructed layout")

    rewriter = impl.names.CheckpointNameRewriter(
        weights_mapping=model.weights_mapping,
        mtp_remap=model.remap_mtp_weight_name,
        spec_decode=True,
        n_routed_experts=512,
        fuse_shared_expert=target_policy,
        disable_fused_shared_loading=model.disable_fused_shared_loading,
    )
    expert = "experts.512" if fused_shared else "shared_expert"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert rewriter.rewrite(f"mtp.layers.0.mlp.shared_expert.{proj}.weight") == (
            f"model.layers.0.mlp.{expert}.{proj}.weight"
        )
    assert rewriter.rewrite("mtp.layers.0.mlp.shared_expert_gate.weight") == (
        "model.layers.0.mlp.shared_expert_gate.weight"
    )
    assert rewriter.rewrite("model.layers.0.mlp.shared_expert.gate_proj.weight") is None


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("target_shared_bf16", [False, True])
def test_bf16_mtp_loads_shared_slot_using_draft_policy(
    impl, monkeypatch, tmp_path, tp, target_shared_bf16
):
    """The BF16 draft can fuse even when the FP8 target must keep shared separate."""
    from functools import partial

    from safetensors.torch import save_file

    target_config = impl.config.QuantizationConfig()
    target_config.quant_method = "compressed-tensors"
    target_config.global_spec = impl.moe.LayerQuantConfig(
        quant_dtype=impl.moe.dtypes.fp8, quant_type=impl.moe.QuantType.per_Token
    )
    target_config.exclude_layers = [r"re:^mtp\."]
    if target_shared_bf16:
        target_config.exclude_layers.append(r"re:^model\.layers\.0\.mlp\.shared_expert")
    target_policy = impl.qwen._Qwen4ExpQuantizationConfig(target_config)
    draft_policy = impl.mtp._UnquantizedMTPQuantizationConfig(target_config)
    monkeypatch.setattr(
        impl.topk,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            moe_all2all_backend="none",
        ),
    )
    can_fuse = impl.topk.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config
    assert can_fuse(
        target_policy, "model.layers.0.mlp.shared_expert", "model.layers.0.mlp.experts"
    ) is (not target_shared_bf16)

    # Keep construction CPU-only while using the real MTP model's fusion flags,
    # checkpoint rewriter, stacked loader and per-expert BF16 weight loader.
    routed, hidden, rank = 512, 8, tp - 1
    experts, method = _layer(impl, "bf16", tp, rank, experts=routed + 1, hidden=hidden)
    # A dropped shared weight must fail even if an empty TP2 buffer reuses
    # storage that a previous test populated with the expected values.
    experts.w13_weight.data[routed].zero_()
    experts.w2_weight.data[routed].zero_()
    experts.quant_method = method
    experts.layer_quant_config = impl.moe.LayerQuantConfig(quant_dtype=torch.bfloat16)
    experts.w13_weight.weight_loader = experts.weight_loader
    experts.w2_weight.weight_loader = experts.weight_loader
    impl.qwen.install_stacked_expert_loaders(experts)

    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    block.mlp.experts = experts
    block.mlp.fuse_shared = can_fuse(
        draft_policy, "mtp.layers.0.mlp.shared_expert", "mtp.layers.0.mlp.experts"
    )
    predictor = torch.nn.Module()
    predictor.layers = torch.nn.ModuleList([block])
    monkeypatch.setattr(impl.mtp, "Qwen4ExpMultiTokenPredictor", lambda _: predictor)
    monkeypatch.setattr(
        impl.mtp, "ParallelLMHead", lambda *_args, **_kw: torch.nn.Identity()
    )
    hf = SimpleNamespace(
        num_experts=routed, num_hidden_layers=48, hidden_size=hidden, vocab_size=32
    )
    model = impl.mtp.Qwen4ExpMTP(SimpleNamespace(hf_config=hf))
    assert model.fused_shared_experts == 1
    assert model.disable_fused_shared_loading is False

    gate = (torch.arange(640 * hidden).reshape(640, hidden) % 7 + 1).to(torch.bfloat16)
    up = -gate - 1
    down = (torch.arange(hidden * 640).reshape(hidden, 640) % 11 + 1).to(torch.bfloat16)
    prefix = "mtp.layers.0.mlp"
    weights = {
        f"{prefix}.experts.gate_up_proj": torch.full(
            (routed, 1280, hidden), 5, dtype=torch.bfloat16
        ),
        f"{prefix}.experts.down_proj": torch.full(
            (routed, hidden, 640), 6, dtype=torch.bfloat16
        ),
        f"{prefix}.shared_expert.gate_proj.weight": gate,
        f"{prefix}.shared_expert.up_proj.weight": up,
        f"{prefix}.shared_expert.down_proj.weight": down,
    }
    save_file(weights, tmp_path / "model.safetensors")
    monkeypatch.setenv("ATOM_LOADER_NUM_THREADS", "1")
    loaded = impl.loading.load_weights_into_model(
        model=model,
        model_name_or_path=str(tmp_path),
        hf_config=hf,
        spec_decode=True,
        default_weight_loader=lambda param, weight: param.data.copy_(weight),
        fuse_shared_expert=partial(can_fuse, target_policy),
        is_rank0=lambda: True,
        weights_iterator=impl.iterator.safetensors_weights_iterator,
    )
    assert loaded == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
    }
    logical, physical = 640 // tp, experts.w2_weight.shape[-1]
    shard = slice(rank * logical, (rank + 1) * logical)
    expected13 = torch.zeros((2 * physical, hidden), dtype=torch.bfloat16)
    expected13[:logical] = gate[shard]
    expected13[physical : physical + logical] = up[shard]
    expected2 = torch.zeros((hidden, physical), dtype=torch.bfloat16)
    expected2[:, :logical] = down[:, shard]
    torch.testing.assert_close(experts.w13_weight[routed], expected13, rtol=0, atol=0)
    torch.testing.assert_close(experts.w2_weight[routed], expected2, rtol=0, atol=0)
    assert (experts.w13_weight[:routed, :logical] == 5).all()
    assert (experts.w13_weight[:routed, physical : physical + logical] == 5).all()
    assert (experts.w2_weight[:routed, :, :logical] == 6).all()


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_preshuffle_preserves_final_parameter_markers_and_padding(
    impl, monkeypatch, tp, mode
):
    def shuffle_weight(weight, *args, **kwargs):
        # A non-identity CPU permutation checks ATOM's replacement/marker
        # contract. The GPU suite validates AITER's actual shuffle layout.
        return weight.view(torch.uint8).flip((-2,)).view(weight.dtype)

    monkeypatch.setattr(impl.shuffle, "shuffle_weight", shuffle_weight)
    monkeypatch.setattr(impl.utils, "shuffle_weight", shuffle_weight)
    layer, method = _layer(impl, mode, tp, 0)
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
    pointers = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
    if mode == "fp8":
        # Test the layout contract independently of fn -> fnuz normalization.
        method.need_normalize_e4m3fn_to_e4m3fnuz = False
        scales = (layer.w13_weight_scale.clone(), layer.w2_weight_scale.clone())
    method.process_weights_after_loading(layer)
    assert layer.w13_weight.is_shuffled and layer.w2_weight.is_shuffled
    assert (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr()) == pointers
    torch.testing.assert_close(
        layer.w13_weight.view(torch.uint8), expected13.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        layer.w2_weight.view(torch.uint8), expected2.view(torch.uint8), rtol=0, atol=0
    )
    if mode == "fp8":
        torch.testing.assert_close(layer.w13_weight_scale, scales[0], rtol=0, atol=0)
        torch.testing.assert_close(layer.w2_weight_scale, scales[1], rtol=0, atol=0)
