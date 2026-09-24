# SPDX-License-Identifier: MIT
"""Load-only coverage for Quark and NVIDIA ModelOpt NVFP4 tensors."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

aiter = pytest.importorskip("aiter")
QuantType = aiter.QuantType
dtypes = aiter.dtypes

linear_mod = importlib.import_module("atom.model_ops.linear")
moe_mod = importlib.import_module("atom.model_ops.moe")
OnlineQuantStreamer = importlib.import_module(
    "atom.model_loader.online_quant_streaming"
).OnlineQuantStreamer
MergedColumnParallelLinear = linear_mod.MergedColumnParallelLinear
QKVParallelLinear = linear_mod.QKVParallelLinear
FusedMoE = moe_mod.FusedMoE
Nvfp4MoEMethod = moe_mod.Nvfp4MoEMethod

minimax_m3 = importlib.import_module("atom.models.minimax_m3")
_is_moe_layer = minimax_m3._is_moe_layer
make_minimax_m3_expert_params_mapping = minimax_m3.make_minimax_m3_expert_params_mapping
_normalize_minimax_m3_text_config = importlib.import_module(
    "atom.config"
)._normalize_minimax_m3_text_config

quant_spec = importlib.import_module("atom.quant_spec")
NVFP4_DTYPE = quant_spec.NVFP4_DTYPE
LayerQuantConfig = quant_spec.LayerQuantConfig
should_stream_online_quant = quant_spec.should_stream_online_quant
validate_nvfp4_global_scales = quant_spec.validate_nvfp4_global_scales
dequantize_nvfp4 = importlib.import_module(
    "atom.quantization.quark.utils"
).dequantize_nvfp4


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


def test_streaming_uses_actual_source_dtype_not_checkpoint_spec(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "1")
    source = _nvfp4_spec()
    target = LayerQuantConfig(
        quant_type=QuantType.per_Token,
        quant_dtype=dtypes.fp8,
        is_dynamic=True,
        quant_method="quark",
    )
    config = SimpleNamespace(
        online_quant=True,
        get_layer_quant_config=lambda *_args, use_online_quant=False, **_kwargs: (
            target if use_online_quant else source
        ),
    )

    assert should_stream_online_quant(
        config,
        "model.layers.0.mlp.gate_proj",
        QuantType.No,
        torch.bfloat16,
    )
    assert not should_stream_online_quant(
        config,
        "model.layers.0.mlp.gate_proj",
        QuantType.per_Token,
        dtypes.fp8,
    )


def test_dequantize_nvfp4_applies_block_and_global_scales():
    # Low nibble is the even logical value: byte 0x57 decodes to [6, 3].
    # Keep the two scales non-reciprocal (2.0 * 3.0 = 6.0, not 1.0) so
    # dropping one or both multiplications changes the result.
    packed = torch.full((1, 8), 0x57, dtype=torch.uint8)
    block_scale = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)

    dequantized = dequantize_nvfp4(
        packed,
        block_scale,
        torch.tensor(3.0),
    )

    assert dequantized.shape == (1, 16)
    assert torch.equal(
        dequantized,
        torch.tensor([[36.0, 18.0] * 8], dtype=torch.float32),
    )


def test_dequantize_nvfp4_requires_a_global_scale():
    """NVFP4 is two-level; a missing global scale is a load bug, not a default."""
    packed = torch.full((1, 8), 0x57, dtype=torch.uint8)
    block_scale = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="two-level format"):
        dequantize_nvfp4(packed, block_scale, None)


@pytest.mark.parametrize(
    "bad",
    [
        # What a zeroed (MoE) buffer holds when the checkpoint carried no
        # *_weight_scale_2, and what an uninitialized (Linear) one can hold.
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, -0.5],
        [1.0, float("nan")],
        [1.0, float("inf")],
    ],
)
def test_validate_nvfp4_global_scales_rejects_unloaded_values(bad):
    with pytest.raises(RuntimeError, match="non-positive or non-finite"):
        validate_nvfp4_global_scales(
            torch.tensor(bad), "model.layers.0.mlp.gate_proj", "weight_scale_2"
        )


def test_validate_nvfp4_global_scales_accepts_a_loaded_checkpoint():
    """Guards the check above from rejecting every real NVFP4 layer."""
    validate_nvfp4_global_scales(
        torch.tensor([[0.375, 1.5], [2.0, 0.125]]),
        "model.layers.3.block_sparse_moe.experts",
        "w13_weight_scale_2",
    )
    with pytest.raises(RuntimeError, match="weight_scale_2 is missing"):
        validate_nvfp4_global_scales(
            None, "model.layers.0.mlp.gate_proj", "weight_scale_2"
        )


def test_nvfp4_merged_linear_loads_all_checkpoint_tensors(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )

    layer = MergedColumnParallelLinear(
        64,
        [32, 32],
        quant_config=_nvfp4_to_mxfp4_config(),
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    assert layer.weight.shape == (64, 32)
    assert layer.weight.dtype == torch.uint8
    assert layer.weight_scale.shape == (64, 4)
    assert layer.weight_scale.dtype == torch.float8_e4m3fn

    for shard_id, value in enumerate((3.0, 4.0)):
        layer.weight_scale_2.weight_loader(
            layer.weight_scale_2, torch.tensor(value), shard_id
        )

    assert layer.weight_scale_2.tolist() == [3.0, 4.0]
    assert layer.input_scale_2 is None
    with pytest.raises(RuntimeError, match="direct NVFP4 inference"):
        layer(torch.ones(1, 64, dtype=torch.bfloat16))

    monkeypatch.setattr(layer, "online_quantize_weight", lambda: None)
    with pytest.raises(RuntimeError, match="not converted to MXFP4"):
        layer.process_weights_after_loading()


def test_nvfp4_merged_linear_online_converts_to_mxfp4(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    captured = {}

    def fake_mxfp4_quant(weight, *_args, **_kwargs):
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

    monkeypatch.setattr(linear_mod, "quant_weight_online", fake_mxfp4_quant)
    layer = MergedColumnParallelLinear(
        64,
        [32, 32],
        quant_config=_nvfp4_to_mxfp4_config(),
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    layer.weight.data.fill_(0x22)  # E2M1 value 1 in both nibbles.
    layer.weight_scale.data.fill_(2.0)
    layer.weight_scale_2.weight_loader(layer.weight_scale_2, torch.tensor([0.5, 1.0]))

    layer.online_quantize_weight()

    assert torch.equal(captured["weight"][:32], torch.ones(32, 64))
    assert torch.equal(captured["weight"][32:], torch.full((32, 64), 2.0))
    assert layer.params_dtype == dtypes.fp4x2
    assert layer.weight.shape == (64, 32)
    assert layer.weight_scale.shape == (64, 2)
    assert layer.weight_scale_2 is None
    assert layer.input_scale_2 is None


def test_nvfp4_merged_linear_rejects_an_unloaded_global_scale(monkeypatch):
    """Never running the weight_scale_2 loader must fail, not scale by zero."""
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )

    layer = MergedColumnParallelLinear(
        64,
        [32, 32],
        quant_config=_nvfp4_to_mxfp4_config(),
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    layer.weight.data.fill_(0x22)
    layer.weight_scale.data.fill_(2.0)
    # No weight_scale_2 loader call: the checkpoint did not carry one, so the
    # allocation-time zeros are still there.
    assert not layer.weight_scale_2.data.any()

    with pytest.raises(RuntimeError, match="non-positive or non-finite"):
        layer.online_quantize_weight()


def test_nvfp4_qkv_uses_format_specific_global_scale_loader(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    layer = QKVParallelLinear(
        hidden_size=64,
        head_size=16,
        total_num_heads=2,
        total_num_kv_heads=1,
        quant_config=_nvfp4_to_mxfp4_config(),
        prefix="model.layers.0.self_attn.qkv_proj",
    )

    for shard_id, value in (("q", 2.0), ("k", 3.0), ("v", 4.0)):
        layer.weight_scale_2.weight_loader(
            layer.weight_scale_2, torch.tensor(value), shard_id
        )

    assert layer.weight_scale_2.tolist() == [2.0, 3.0, 4.0]


def test_nvfp4_str_shard_id_without_global_scale_map_is_rejected(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    layer = linear_mod.QKVGParallelLinear(
        hidden_size=64,
        head_size=16,
        total_num_heads=2,
        total_num_kv_heads=1,
        quant_config=_nvfp4_to_mxfp4_config(),
        prefix="model.layers.0.self_attn.qkv_proj",
    )

    with pytest.raises(
        ValueError,
        match="QKVGParallelLinear defines no _nvfp4_global_scale_shard_map",
    ):
        layer.weight_scale_2.weight_loader(layer.weight_scale_2, torch.tensor(2.0), "q")


def test_nvfp4_merged_linear_streaming_completes_from_split_shards(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "1")
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING", "1")
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING_THREADS", "0")
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    monkeypatch.setattr(linear_mod, "use_fp4_non_shuffle_triton_gemm", lambda: True)

    def fake_mxfp4_quant(weight, *_args, **_kwargs):
        return (
            torch.zeros(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8).view(
                dtypes.fp4x2
            ),
            torch.full(
                (weight.shape[0], weight.shape[1] // 32),
                127,
                dtype=torch.uint8,
            ).view(dtypes.fp8_e8m0),
        )

    monkeypatch.setattr(linear_mod, "quant_weight_online", fake_mxfp4_quant)
    layer = MergedColumnParallelLinear(
        64,
        [32, 32],
        quant_config=_nvfp4_to_mxfp4_config(),
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    assert layer._stream_online_quant
    assert layer.weight.is_meta
    assert layer.input_scale_2 is None

    streamer = OnlineQuantStreamer.maybe_create(layer, load_dummy=None)
    assert streamer is not None
    assert layer._stream_expected_numel == 64 * 32 + 64 * 4 + 2

    for shard_id, global_scale in enumerate((0.5, 1.0)):
        arrivals = (
            (layer.weight, torch.full((32, 32), 0x22, dtype=torch.uint8)),
            (
                layer.weight_scale,
                torch.full((32, 4), 2.0, dtype=torch.float8_e4m3fn),
            ),
            (layer.weight_scale_2, torch.tensor(global_scale)),
        )
        for param, value in arrivals:
            streamer.run(param.weight_loader, (param, value, shard_id))

    assert id(layer) in streamer.done_module_ids
    assert layer._stream_loaded_numel == layer._stream_expected_numel
    assert not layer._stream_online_quant
    assert layer.layer_quant_config.quant_dtype == dtypes.fp4x2
    assert layer.weight.dtype == dtypes.fp4x2


def test_nvfp4_qkv_streaming_completes_from_split_shards(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "1")
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING", "1")
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING_THREADS", "0")
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    monkeypatch.setattr(linear_mod, "use_fp4_non_shuffle_triton_gemm", lambda: True)

    def fake_mxfp4_quant(weight, *_args, **_kwargs):
        return (
            torch.zeros(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8).view(
                dtypes.fp4x2
            ),
            torch.full(
                (weight.shape[0], weight.shape[1] // 32),
                127,
                dtype=torch.uint8,
            ).view(dtypes.fp8_e8m0),
        )

    monkeypatch.setattr(linear_mod, "quant_weight_online", fake_mxfp4_quant)
    layer = QKVParallelLinear(
        hidden_size=64,
        head_size=16,
        total_num_heads=2,
        total_num_kv_heads=1,
        quant_config=_nvfp4_to_mxfp4_config(),
        prefix="model.layers.0.self_attn.qkv_proj",
    )
    streamer = OnlineQuantStreamer.maybe_create(layer, load_dummy=None)
    assert streamer is not None
    assert layer._stream_expected_numel == 64 * 32 + 64 * 4 + 3

    for shard_id, rows, global_scale in (
        ("q", 32, 0.5),
        ("k", 16, 1.0),
        ("v", 16, 1.5),
    ):
        arrivals = (
            (layer.weight, torch.full((rows, 32), 0x22, dtype=torch.uint8)),
            (
                layer.weight_scale,
                torch.full((rows, 4), 2.0, dtype=torch.float8_e4m3fn),
            ),
            (layer.weight_scale_2, torch.tensor(global_scale)),
        )
        for param, value in arrivals:
            streamer.run(param.weight_loader, (param, value, shard_id))

    assert id(layer) in streamer.done_module_ids
    assert layer._stream_loaded_numel == layer._stream_expected_numel
    assert layer.layer_quant_config.quant_dtype == dtypes.fp4x2
    assert layer.weight.dtype == dtypes.fp4x2


def test_nvfp4_linear_rejects_non_group_aligned_input(monkeypatch):
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)

    with pytest.raises(ValueError, match="must be divisible by group_size=16"):
        MergedColumnParallelLinear(
            66,
            [32, 32],
            quant_config=_nvfp4_to_mxfp4_config(),
            prefix="model.layers.0.mlp.gate_up_proj",
        )


def test_nvfp4_linear_rejects_mxfp4_misaligned_input_at_construction(monkeypatch):
    """K=48 is valid NVFP4 (3 groups of 16) but not MXFP4 (1.5 blocks of 32)."""
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)

    with pytest.raises(ValueError, match="MXFP4 Linear path.*divisible by 32"):
        MergedColumnParallelLinear(
            48,
            [32, 32],
            quant_config=_nvfp4_to_mxfp4_config(),
            prefix="model.layers.0.mlp.gate_up_proj",
        )


def test_nvfp4_linear_without_mxfp4_target_is_rejected_at_construction(monkeypatch):
    """An NVFP4 layer excluded from the online target is rejected when built."""
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: tp_group)
    excluded = LayerQuantConfig(quant_dtype=torch.bfloat16)
    quant_config = SimpleNamespace(
        online_quant=True,
        get_layer_quant_config=lambda *_args, use_online_quant=False, **_kwargs: (
            excluded if use_online_quant else _nvfp4_spec()
        ),
    )

    with pytest.raises(ValueError, match="excluded from online quantization"):
        MergedColumnParallelLinear(
            64,
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
    layer.layer_name = layer.prefix
    layer.local_num_experts = 2
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=dtypes.fp4x2,
        weight_loader=layer.weight_loader,
    )
    with pytest.raises(RuntimeError, match="expects split per-expert"):
        layer.weight_loader(
            layer.w13_weight,
            torch.empty(1),
            "",
            "w1",
            0,
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
        layer.w2_weight_scale_2,
        torch.tensor(7.0),
        "weight_scale_2",
        "w2",
        1,
    )
    assert layer.w13_weight.shape == (2, 64, 32)
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w13_weight_scale.shape == (2, 64, 4)
    assert layer.w13_weight_scale_2[1].tolist() == [3.0, 4.0]
    assert layer.w2_weight_scale_2[1].item() == 7.0

    assert not hasattr(layer, "w13_input_scale_2")
    assert not hasattr(layer, "w2_input_scale_2")
    with pytest.raises(RuntimeError, match="not converted to MXFP4"):
        method.process_weights_after_loading(layer)
    method.init_prepare_finalize(layer)
    with pytest.raises(RuntimeError, match="direct NVFP4 inference"):
        method.apply(layer, None)


def test_nvfp4_moe_streaming_counts_split_global_scales(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "1")
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING", "1")
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING_THREADS", "0")

    layer = object.__new__(FusedMoE)
    nn.Module.__init__(layer)
    layer.has_bias = False
    layer.quant_config = SimpleNamespace(online_quant=True)
    layer.prefix = "model.layers.3.block_sparse_moe.experts"
    layer.layer_name = layer.prefix
    layer.layer_quant_config = _nvfp4_spec()
    layer.moe_parallel_config = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        use_ep=False,
    )
    layer.expert_layout = SimpleNamespace(
        num_routed_physical=1,
        routed_physical_per_rank=1,
        num_fused_shared_experts=0,
    )
    layer.global_num_experts = 1
    layer.num_redundant_experts = 0
    layer.local_num_experts = 1
    layer.expert_map = None
    layer._stream_online_quant = True
    layer._load_device = torch.device("cpu")
    layer._comm_fused_moe = None

    method = Nvfp4MoEMethod(_nvfp4_spec(), SimpleNamespace())
    layer.quant_method = method
    method.create_weights(
        layer,
        num_experts=1,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=dtypes.fp4x2,
        weight_loader=layer.weight_loader,
    )

    finalized = []

    def fake_online_quant():
        assert layer.w13_weight_scale_2.tolist() == [[0.5, 1.0]]
        assert layer.w2_weight_scale_2.tolist() == [1.5]
        layer._stream_online_quant = False
        finalized.append(True)

    layer._online_quant = fake_online_quant
    layer._validate_moe_backend = lambda: None

    streamer = OnlineQuantStreamer.maybe_create(layer, load_dummy=None)
    assert streamer is not None
    assert layer._stream_expected_numel == 3459

    arrivals = (
        (
            layer.w13_weight,
            torch.full((32, 32), 0x22, dtype=torch.uint8),
            "weight",
            "w1",
        ),
        (
            layer.w13_weight,
            torch.full((32, 32), 0x22, dtype=torch.uint8),
            "weight",
            "w3",
        ),
        (
            layer.w2_weight,
            torch.full((64, 16), 0x22, dtype=torch.uint8),
            "weight",
            "w2",
        ),
        (
            layer.w13_weight_scale,
            torch.full((32, 4), 2.0, dtype=torch.float8_e4m3fn),
            "weight_scale",
            "w1",
        ),
        (
            layer.w13_weight_scale,
            torch.full((32, 4), 2.0, dtype=torch.float8_e4m3fn),
            "weight_scale",
            "w3",
        ),
        (
            layer.w2_weight_scale,
            torch.full((64, 2), 2.0, dtype=torch.float8_e4m3fn),
            "weight_scale",
            "w2",
        ),
        (
            layer.w13_weight_scale_2,
            torch.tensor(0.5),
            "weight_scale_2",
            "w1",
        ),
        (
            layer.w13_weight_scale_2,
            torch.tensor(1.0),
            "weight_scale_2",
            "w3",
        ),
        (
            layer.w2_weight_scale_2,
            torch.tensor(1.5),
            "weight_scale_2",
            "w2",
        ),
    )
    for param, value, weight_name, shard_id in arrivals:
        streamer.run(
            param.weight_loader,
            (param, value, weight_name, shard_id, 0),
        )

    assert finalized == [True]
    assert id(layer) in streamer.done_module_ids
    assert layer._stream_loaded_numel == layer._stream_expected_numel


def _nvfp4_moe_layer(
    source,
    target,
    source_intermediate,
    tp_size,
    w13_scale_2=((0.5, 1.0),),
    w2_scale_2=(1.5,),
    num_local_base_experts=None,
    routed_physical_per_rank=None,
):
    """The subset of FusedMoE state that `_online_quant` reads, as a namespace.

    One local expert slot per `w2_scale_2` entry; by default every slot is a
    routed base expert.
    """
    num_slots = len(w2_scale_2)
    if num_local_base_experts is None:
        num_local_base_experts = num_slots
    if routed_physical_per_rank is None:
        routed_physical_per_rank = num_local_base_experts
    return SimpleNamespace(
        online_quant=True,
        quant_config=SimpleNamespace(
            get_layer_quant_config=lambda *_args, **_kwargs: target
        ),
        layer_name="model.layers.3.block_sparse_moe.experts",
        layer_quant_config=source,
        source_is_nvfp4=True,
        params_dtype=NVFP4_DTYPE,
        quant_method=SimpleNamespace(intermediate_pad=0),
        moe_config=SimpleNamespace(),
        moe_quant_params={"params_dtype": NVFP4_DTYPE},
        _stream_online_quant=False,
        local_num_experts=num_slots,
        num_local_base_experts=num_local_base_experts,
        expert_layout=SimpleNamespace(
            routed_physical_per_rank=routed_physical_per_rank
        ),
        intermediate_size_per_partition=source_intermediate,
        has_bias=False,
        use_ep=False,
        tp_size=tp_size,
        tp_rank=0,
        w13_weight=nn.Parameter(
            torch.full(
                (num_slots, 2 * source_intermediate, 32), 0x22, dtype=torch.uint8
            ),
            requires_grad=False,
        ),
        w2_weight=nn.Parameter(
            torch.full(
                (num_slots, 64, source_intermediate // 2), 0x22, dtype=torch.uint8
            ),
            requires_grad=False,
        ),
        w13_weight_scale=nn.Parameter(
            torch.full(
                (num_slots, 2 * source_intermediate, 4),
                2.0,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        ),
        w2_weight_scale=nn.Parameter(
            torch.full(
                (num_slots, 64, source_intermediate // 16),
                2.0,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        ),
        w13_weight_scale_2=nn.Parameter(torch.tensor(w13_scale_2), requires_grad=False),
        w2_weight_scale_2=nn.Parameter(torch.tensor(w2_scale_2), requires_grad=False),
        _copy_quant_storage=FusedMoE._copy_quant_storage,
        _load_model_weight_or_group_weight_scale=lambda **_kwargs: None,
        _load_quant_weight_scale=lambda **_kwargs: None,
    )


@pytest.mark.parametrize(
    ("source_intermediate", "tp_size"),
    [(32, 1), (32, 2)],
)
def test_nvfp4_moe_online_conversion_uses_local_tp_shards(
    monkeypatch, source_intermediate, tp_size
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
            self.intermediate_pad = 32
            layer.w13_weight = nn.Parameter(
                torch.zeros(1, 128, 32, dtype=torch.uint8).view(dtypes.fp4x2),
                requires_grad=False,
            )
            layer.w2_weight = nn.Parameter(
                torch.zeros(1, 64, 32, dtype=torch.uint8).view(dtypes.fp4x2),
                requires_grad=False,
            )
            layer.w13_weight_scale = nn.Parameter(
                torch.zeros(1, 128, 2, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.w2_weight_scale = nn.Parameter(
                torch.zeros(1, 64, 2, dtype=torch.uint8),
                requires_grad=False,
            )

    fake_target_method = FakeTargetMethod()
    monkeypatch.setattr(
        moe_mod,
        "_make_mxfp4_moe_method",
        lambda *_args, **_kwargs: fake_target_method,
    )
    converted = []

    def fake_mxfp4_quant(weight, *_args, **_kwargs):
        converted.append(weight.clone())
        return (
            torch.full(
                (weight.shape[0], weight.shape[1] // 2),
                0x11,
                dtype=torch.uint8,
            ).view(dtypes.fp4x2),
            torch.full(
                (weight.shape[0], weight.shape[1] // 32),
                127,
                dtype=torch.uint8,
            ).view(dtypes.fp8_e8m0),
        )

    monkeypatch.setattr(moe_mod, "quant_weight_online", fake_mxfp4_quant)

    class NoGatherGroup:
        def all_gather(self, *_args, **_kwargs):
            raise AssertionError("NVFP4-to-MXFP4 must not gather padded TP shards")

    monkeypatch.setattr(moe_mod, "get_tp_group", lambda: NoGatherGroup())

    layer = _nvfp4_moe_layer(source, target, source_intermediate, tp_size)

    FusedMoE._online_quant(layer)

    assert len(converted) == 3
    assert torch.equal(converted[0], torch.ones(source_intermediate, 64))
    assert torch.equal(converted[1], torch.full((source_intermediate, 64), 2.0))
    assert torch.equal(
        converted[2][:, :source_intermediate],
        torch.full((64, source_intermediate), 3.0),
    )
    w2_storage = layer.w2_weight.view(torch.uint8)
    valid_weight_cols = source_intermediate // 2
    assert torch.all(w2_storage[:, :, :valid_weight_cols] == 0x11)
    assert torch.count_nonzero(w2_storage[:, :, valid_weight_cols:]) == 0
    valid_scale_cols = source_intermediate // 32
    assert torch.all(layer.w2_weight_scale[:, :, :valid_scale_cols] == 127)
    assert torch.count_nonzero(layer.w2_weight_scale[:, :, valid_scale_cols:]) == 0
    assert layer.quant_method is fake_target_method
    assert layer.quant_method.intermediate_pad == 0
    assert layer.layer_quant_config is target
    assert layer.params_dtype == dtypes.fp4x2
    assert layer.source_is_nvfp4 is False
    assert layer.w13_weight_scale_2 is None
    assert layer.w2_weight_scale_2 is None
    assert layer._stream_online_quant is False
    merged = []
    layer.mxf4_merged_weight_loader = (
        lambda param, loaded_weight, expert_id: merged.append(
            (param, tuple(loaded_weight.shape), expert_id)
        )
    )
    dummy = nn.Parameter(torch.zeros(1, 2, 4))
    FusedMoE.weight_loader(layer, dummy, dummy, weight_name="")
    assert merged == [(dummy, (1, 2, 4), 0)]


@pytest.mark.parametrize(
    ("w13_scale_2", "w2_scale_2", "missing"),
    [
        (((0.0, 0.0),), (1.5,), "w13_weight_scale_2"),
        (((0.5, 1.0),), (0.0,), "w2_weight_scale_2"),
    ],
)
def test_nvfp4_moe_rejects_unloaded_global_scales(
    monkeypatch, w13_scale_2, w2_scale_2, missing
):
    """The scale_2 buffers are zero-filled, so an absent checkpoint tensor
    would otherwise dequantize every expert to all zeros without an error."""
    monkeypatch.setattr(moe_mod, "get_tp_group", lambda: None)
    target = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp4x2,
        is_dynamic=True,
        quant_method="quark",
    )
    layer = _nvfp4_moe_layer(
        _nvfp4_spec(), target, 32, 1, w13_scale_2=w13_scale_2, w2_scale_2=w2_scale_2
    )

    with pytest.raises(RuntimeError, match=f"{missing} holds non-positive"):
        FusedMoE._online_quant(layer)


class _TargetCreationReached(Exception):
    """Stands in for building the MXFP4 target: the scale_2 check passed."""


@pytest.mark.parametrize(
    ("w13_scale_2", "w2_scale_2", "error", "match"),
    [
        (
            ((0.5, 1.0), (0.0, 0.0), (0.5, 1.0)),
            (1.5, 0.0, 1.5),
            _TargetCreationReached,
            None,
        ),
        (
            ((0.5, 1.0), (0.0, 0.0), (0.5, 1.0)),
            (0.0, 0.0, 1.5),
            RuntimeError,
            "w2_weight_scale_2 holds non-positive",
        ),
        (
            ((0.5, 1.0), (0.0, 0.0), (0.0, 0.0)),
            (1.5, 0.0, 1.5),
            RuntimeError,
            "w13_weight_scale_2 holds non-positive",
        ),
    ],
    ids=["replica-still-empty", "base-unloaded", "fused-shared-unloaded"],
)
def test_nvfp4_moe_global_scale_check_skips_eplb_redundant_replicas(
    monkeypatch, w13_scale_2, w2_scale_2, error, match
):
    """Local slots are [routed base, EPLB redundant replica, fused shared].
    fill_redundant copies the replica from its base expert only after online
    quantization, so it alone may still be zero when the check runs."""
    monkeypatch.setattr(moe_mod, "get_tp_group", lambda: None)

    def reach_target_creation(*_args, **_kwargs):
        raise _TargetCreationReached

    monkeypatch.setattr(moe_mod, "_make_mxfp4_moe_method", reach_target_creation)
    target = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp4x2,
        is_dynamic=True,
        quant_method="quark",
    )
    layer = _nvfp4_moe_layer(
        _nvfp4_spec(),
        target,
        32,
        1,
        w13_scale_2=w13_scale_2,
        w2_scale_2=w2_scale_2,
        num_local_base_experts=1,
        routed_physical_per_rank=2,
    )

    with pytest.raises(error, match=match):
        FusedMoE._online_quant(layer)


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size"),
    [(66, 32), (64, 34)],
)
def test_nvfp4_moe_rejects_non_group_aligned_dimensions(hidden_size, intermediate_size):
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


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size", "axis"),
    [
        (48, 32, "hidden size"),
        # I=1408 at TP8: whole NVFP4 groups, but 5.5 MXFP4 blocks per rank.
        (64, 176, "intermediate size per TP partition"),
    ],
)
def test_nvfp4_moe_rejects_mxfp4_misaligned_dimensions_at_construction(
    hidden_size, intermediate_size, axis
):
    layer = nn.Module()
    layer.has_bias = False
    method = Nvfp4MoEMethod(_nvfp4_spec(), SimpleNamespace())

    with pytest.raises(ValueError, match=f"NVFP4 {axis}.*multiple of 32"):
        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=dtypes.fp4x2,
        )


def test_nvfp4_moe_rejects_expert_biases():
    layer = nn.Module()
    layer.has_bias = True
    method = Nvfp4MoEMethod(_nvfp4_spec(), SimpleNamespace())

    with pytest.raises(ValueError, match="NVFP4 MoE expert biases are not supported"):
        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=64,
            intermediate_size_per_partition=32,
            params_dtype=dtypes.fp4x2,
        )


def _minimax_m3_text_config(**fields):
    return SimpleNamespace(model_type="minimax_m3", num_hidden_layers=4, **fields)


@pytest.mark.parametrize("moe_layer_freq", [None, [0, 0, 0, 1]])
@pytest.mark.parametrize("nested", [True, False])
def test_minimax_m3_uses_checkpoint_mlp_layer_types(nested, moe_layer_freq):
    text_config = _minimax_m3_text_config(
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        moe_layer_freq=moe_layer_freq,
    )
    hf_config = (
        SimpleNamespace(model_type="minimax_m3_vl", text_config=text_config)
        if nested
        else text_config
    )

    _normalize_minimax_m3_text_config(hf_config)

    assert [_is_moe_layer(text_config, i) for i in range(4)] == [
        False,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("root_fields", "text_fields"),
    [
        ({"mlp_layer_types": ["dense", "dense", "dense", "sparse"]}, {}),
        (
            {"mlp_layer_types": ["dense", "dense", "dense", "sparse"]},
            {"mlp_layer_types": ["sparse"] * 4},
        ),
        ({"moe_layer_freq": [0, 0, 0, 1]}, {"moe_layer_freq": [1, 1, 1, 1]}),
    ],
    ids=[
        "text-config-has-neither",
        "overrides-mlp-layer-types",
        "overrides-moe-layer-freq",
    ],
)
def test_minimax_m3_root_layer_layout_reaches_text_config(root_fields, text_fields):
    text_config = _minimax_m3_text_config(**text_fields)
    hf_config = SimpleNamespace(
        model_type="minimax_m3_vl", text_config=text_config, **root_fields
    )

    _normalize_minimax_m3_text_config(hf_config)

    assert [_is_moe_layer(text_config, i) for i in range(4)] == [
        False,
        False,
        False,
        True,
    ]


def test_minimax_m3_root_mlp_layer_types_must_agree_with_text_moe_layer_freq():
    hf_config = SimpleNamespace(
        model_type="minimax_m3_vl",
        text_config=_minimax_m3_text_config(moe_layer_freq=[0, 0, 1, 1]),
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
    )

    with pytest.raises(ValueError, match="disagree on which layers are MoE"):
        _normalize_minimax_m3_text_config(hf_config)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"mlp_layer_types": ["dense", "sparse", "sparse"]}, "has 3 entries"),
        (
            {"mlp_layer_types": ["dense", "dense", "moe", "MoE"]},
            "unknown labels 'MoE', 'moe'",
        ),
        (
            {
                "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
                "moe_layer_freq": [0, 0, 1, 1],
            },
            "disagree on which layers are MoE",
        ),
    ],
)
def test_minimax_m3_rejects_unreadable_mlp_layer_types(fields, match):
    hf_config = SimpleNamespace(
        model_type="minimax_m3_vl", text_config=_minimax_m3_text_config(**fields)
    )

    with pytest.raises(ValueError, match=match):
        _normalize_minimax_m3_text_config(hf_config)


def test_minimax_m3_uses_modelopt_moe_layer_frequency_list():
    config = SimpleNamespace(
        num_hidden_layers=5,
        mlp_layer_types=None,
        moe_layer_freq=[0, 0, 0, 1, 1],
    )

    assert [_is_moe_layer(config, i) for i in range(5)] == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_minimax_m3_input_scales_use_generic_expert_prefix():
    mapping = make_minimax_m3_expert_params_mapping(1)
    prefixes = sorted(mapping, key=lambda entry: len(entry[1]), reverse=True)

    def match(checkpoint_name):
        return next(
            (param_name, checkpoint_prefix, expert_id, shard_id)
            for param_name, checkpoint_prefix, expert_id, shard_id in prefixes
            if checkpoint_prefix in checkpoint_name
        )

    expected = ("experts.w13_", "experts.0.w1.", 0, "w1")
    assert match("experts.0.w1.input_scale") == expected
    assert match("experts.0.w1.input_scale_2") == expected
