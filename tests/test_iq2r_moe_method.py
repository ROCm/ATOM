# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

# The IQ2R config parser tests live in tests/test_quant_config.py, which runs
# without AITER.
try:
    # aiter/triton absent under bare non-GPU pytest
    import atom.model_ops.moe as moe_mod
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")


@pytest.fixture
def aiter_iq2r():
    """AITER's GLM-5.3 IQ2R format code; older AITER builds lack it."""
    return pytest.importorskip("aiter.iq2r_glm53")


def test_iq2r_weight_loader_requires_exact_shapes():
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.tp_size = 1
    parameter = torch.nn.Parameter(
        torch.empty((2, 8), dtype=torch.uint8), requires_grad=False
    )
    parameter.iq2r_name = "w13_weight"
    loaded = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    method.load_weight(parameter, loaded)
    assert torch.equal(parameter, loaded)

    with pytest.raises(ValueError, match="shape mismatch"):
        method.load_weight(parameter, loaded[:, :-1])


def test_iq2r_weight_loader_checks_the_tile_n():
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.tp_size = 4
    parameter = torch.nn.Parameter(
        torch.zeros(1, dtype=torch.int32), requires_grad=False
    )
    parameter.iq2r_name = "w2_iq2r_tile_n"
    method.load_weight(parameter, torch.tensor([128], dtype=torch.int32))
    assert parameter.item() == 128

    for loaded in ([64], [128, 128]):
        with pytest.raises(ValueError, match="kernels need 128"):
            method.load_weight(parameter, torch.tensor(loaded, dtype=torch.int32))


def _glm53_packed_method(tp_size: int, tp_rank: int, intermediate: int):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            tp_size=tp_size, tp_rank=tp_rank, ep_size=1
        ),
        max_num_tokens=8,
    )
    method.num_experts = 257
    method.hidden_size = 6144
    method.intermediate_size = intermediate
    method.tp_size = tp_size
    method.tp_rank = tp_rank
    return method


@pytest.mark.parametrize("layout", [None, "bogus"])
def test_iq2r_rejects_unknown_layout(layout):
    moe = SimpleNamespace()
    with pytest.raises(ValueError, match="unsupported IQ2R layout"):
        moe_mod.Iq2rMoEMethod.__init__(
            object.__new__(moe_mod.Iq2rMoEMethod), None, moe, layout=layout
        )


def _glm53_packed_layer(method):
    layer = torch.nn.Module()
    layer.has_bias = False
    layer.num_fused_shared_experts = 1
    with torch.device("meta"):
        method.create_weights(
            layer,
            num_experts=257,
            hidden_size=6144,
            intermediate_size_per_partition=method.intermediate_size,
            params_dtype=torch.bfloat16,
        )
    return layer


def test_iq2r_glm53_packed_create_weights_at_tp4(aiter_iq2r):
    from aiter.iq2r_glm53 import iq2r_glm53_gate_bytes
    from aiter.ops.iq2r_format import IQ2RMetadata

    method = _glm53_packed_method(tp_size=4, tp_rank=1, intermediate=512)
    layer = _glm53_packed_layer(method)

    gate = IQ2RMetadata(logical_n=1024, logical_k=6144)
    down = IQ2RMetadata(logical_n=6144, logical_k=512)
    assert tuple(layer.w13_weight.shape) == (257, iq2r_glm53_gate_bytes(1024))
    assert tuple(layer.w13_weight_scale.shape) == (257, gate.auxiliary_bytes)
    assert tuple(layer.w2_weight.shape) == (257, down.data_bytes)
    assert tuple(layer.w2_weight_scale.shape) == (257, down.auxiliary_bytes)
    for name in ("w13_iq2r_tile_n", "w2_iq2r_tile_n"):
        parameter = getattr(layer, name)
        assert (tuple(parameter.shape), parameter.dtype) == ((1,), torch.int32)
    assert (method.tp_size, method.tp_rank) == (4, 1)


def test_glm53_weights_mapping_targets_every_compiled_tensor(aiter_iq2r):
    compiler = pytest.importorskip("aiter.iq2r_glm5_compile")
    from atom.model_loader.weight_names import CheckpointNameRewriter
    from atom.models.deepseek_v2 import GlmMoeDsaForCausalLM

    layer = _glm53_packed_layer(_glm53_packed_method(4, 0, 512))
    rewriter = CheckpointNameRewriter(
        weights_mapping=GlmMoeDsaForCausalLM.weights_mapping
    )
    mapped = sorted(
        rewriter.rewrite(key)
        for projection in ("gate_up", "down")
        for key in compiler.iq2r_compiled_tensor_keys(3, projection).values()
    )
    assert mapped == sorted(
        f"model.layers.3.mlp.experts.{name}" for name, _ in layer.named_parameters()
    )


@pytest.mark.parametrize(
    "tp_size,ep_size,fused_shared",
    [(2, 1, 1), (1, 1, 1), (4, 2, 1), (8, 1, 0)],
)
def test_iq2r_glm53_packed_rejects_unsupported_parallelism(
    tp_size, ep_size, fused_shared
):
    method = _glm53_packed_method(tp_size, 0, 2048 // tp_size)
    method.moe.moe_parallel_config.ep_size = ep_size
    layer = torch.nn.Module()
    layer.has_bias = False
    layer.num_fused_shared_experts = fused_shared
    with (
        pytest.raises(NotImplementedError, match="glm53-packed-v1"),
        torch.device("meta"),
    ):
        method.create_weights(
            layer,
            num_experts=256 + fused_shared,
            hidden_size=6144,
            intermediate_size_per_partition=2048 // tp_size,
            params_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize("tp_size,tp_rank", [(4, 3), (8, 5)])
def test_iq2r_glm53_packed_loader_slices_the_rank_shard(aiter_iq2r, tp_size, tp_rank):
    from aiter.iq2r_glm53 import iq2r_glm53_gate_bytes, iq2r_glm53_slice_gate
    from aiter.ops.iq2r_format import (
        IQ2RMetadata,
        iq2r_slice_input_data,
        iq2r_slice_output_auxiliary,
    )

    shard = 2048 // tp_size
    method = _glm53_packed_method(tp_size, tp_rank, shard)
    gate = IQ2RMetadata(logical_n=4096, logical_k=6144)
    down = IQ2RMetadata(logical_n=6144, logical_k=2048)
    generator = torch.Generator().manual_seed(tp_size)

    def full(columns):
        return torch.randint(0, 256, (1, columns), generator=generator).to(torch.uint8)

    weights = {
        "w13_weight": full(iq2r_glm53_gate_bytes(4096)),
        "w13_weight_scale": full(gate.auxiliary_bytes),
        "w2_weight": full(down.data_bytes),
        "w2_weight_scale": full(down.auxiliary_bytes),
    }
    start = 2 * shard * tp_rank
    expected = {
        "w13_weight": iq2r_glm53_slice_gate(weights["w13_weight"], start, 2 * shard),
        "w13_weight_scale": iq2r_slice_output_auxiliary(
            weights["w13_weight_scale"], gate, start, 2 * shard
        ),
        "w2_weight": iq2r_slice_input_data(
            weights["w2_weight"], down, shard * tp_rank, shard
        ),
        "w2_weight_scale": weights["w2_weight_scale"],
    }
    for name, value in expected.items():
        parameter = torch.nn.Parameter(torch.empty_like(value), requires_grad=False)
        parameter.iq2r_name = name
        method.load_weight(parameter, weights[name])
        assert torch.equal(parameter, value), name


def test_iq2r_glm53_packed_apply_runs_the_glm53_moe(aiter_iq2r, monkeypatch):
    method = _glm53_packed_method(tp_size=4, tp_rank=0, intermediate=512)
    layer = SimpleNamespace(
        num_fused_shared_experts=1,
        routed_scaling_factor=2.5,
        w13_weight=torch.empty((257, 3), dtype=torch.uint8),
        w13_weight_scale=torch.empty((257, 5), dtype=torch.uint8),
        w2_weight=torch.empty((257, 2), dtype=torch.uint8),
        w2_weight_scale=torch.empty((257, 7), dtype=torch.uint8),
    )
    hidden = torch.randn(2, 6144, dtype=torch.bfloat16)
    topk_weights = torch.full((2, 9), 0.1, dtype=torch.float32)
    topk_ids = torch.arange(18, dtype=torch.int32).reshape(2, 9)
    calls = []

    monkeypatch.setattr(
        moe_mod.FusedMoE,
        "select_experts",
        staticmethod(lambda **kwargs: (topk_weights, topk_ids)),
    )
    monkeypatch.setattr(
        method, "_glm53_workspace", lambda device: ("workspace", device)
    )
    monkeypatch.setattr(
        aiter_iq2r, "iq2r_glm53_moe_out", lambda *args: calls.append(args)
    )
    arguments = {
        "layer": layer,
        "x": hidden,
        "router_logits": torch.randn(2, 256),
        "top_k": 8,
        "renormalize": True,
        "use_grouped_topk": True,
        "topk_group": 1,
        "num_expert_group": 1,
        "global_num_experts": 256,
        "scoring_func": "sigmoid",
        "e_score_correction_bias": torch.randn(256),
        "activation": moe_mod.ActivationType.Silu,
    }

    output = method.apply(**arguments)

    assert len(calls) == 1
    assert torch.equal(calls[0][0], hidden)
    assert calls[0][1] is layer.w13_weight
    assert calls[0][5] is topk_weights
    assert calls[0][6] is topk_ids
    assert calls[0][7] is output
    assert calls[0][8] == ("workspace", hidden.device)
    with pytest.raises(ValueError, match="does not support"):
        method.apply(**{**arguments, "activation": moe_mod.ActivationType.Swiglu})
