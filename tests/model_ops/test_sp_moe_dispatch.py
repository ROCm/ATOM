# SPDX-License-Identifier: MIT
"""CPU coverage for SP MoE routing, transport, and tiled-sort eligibility."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_comm_fused_moe import atom_modules  # noqa: F401


@pytest.fixture
def sp_modules(request):
    # Preserve the external types used by an already imported MoE module.
    # Replacing AITER then would create a second, incompatible QuantType enum.
    moe = sys.modules.get("atom.model_ops.moe")
    if moe is None:
        moe = request.getfixturevalue("atom_modules").moe

    from aiter.jit.utils import chip_info

    from atom.distributed import ulysses_sp
    from atom.model_ops.sp_moe_sort import supports_m3_sp_tiled_sort
    from atom.plugin import prepare

    return SimpleNamespace(
        moe=moe,
        chip_info=chip_info,
        sp=ulysses_sp,
        plugin=prepare,
        supports_tiled_sort=supports_m3_sp_tiled_sort,
    )


def _layer(moe):
    method = object.__new__(moe.Mxfp4MoEMethod)
    method.quant_type = moe.QuantType.per_1x32
    return SimpleNamespace(
        quant_method=method,
        moe_parallel_config=SimpleNamespace(
            dp_logical_ratio=1, use_all2all_kernels=False
        ),
        expert_layout=SimpleNamespace(
            shared_is_routed=False, uses_dispatch_remap=False
        ),
        use_ep=False,
        use_grouped_topk=False,
        scoring_func="sigmoid",
        hidden_size=6144,
        intermediate_size_per_partition=768,
        global_num_experts=128,
        local_num_experts=129,
        top_k=4,
        num_fused_shared_experts=1,
        renormalize=True,
        expert_map=None,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
        activation=moe.ActivationType.Swiglu,
        apply_router_weight_on_input=False,
        reduce_results=False,
        prefix="experts",
    )


@pytest.mark.parametrize(
    "rows,override,prequantize,local_routing",
    [
        (8192, {}, True, True),
        (8191, {}, True, False),
        (8192, {"use_ep": True}, True, False),
        (8192, {"use_grouped_topk": True}, True, False),
        (8192, {"scoring_func": "softmax"}, True, False),
        (8192, {}, False, False),
    ],
)
def test_forward_selects_local_routing_by_capability(
    monkeypatch, sp_modules, rows, override, prequantize, local_routing
):
    moe = sp_modules.moe
    layer = _layer(moe)
    vars(layer).update(override)
    method = layer.quant_method
    hidden = torch.empty(rows, 2, dtype=torch.bfloat16)
    logits = torch.empty(rows, 4)
    gathered = torch.empty(rows * 4, 2, dtype=torch.bfloat16)
    scale, weights, ids = (object() for _ in range(3))
    method._sp_input_can_prequantize = Mock(return_value=prequantize)
    method.gather_sp_input = Mock(return_value=(gathered, scale))
    method.gather_sp_routed_input = Mock(return_value=(gathered, scale, weights, ids))
    method.apply = Mock(return_value=gathered)
    gather = Mock(side_effect=lambda value: value.repeat(4, 1))
    scatter = Mock(side_effect=lambda value: value[:rows])
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: True)
    monkeypatch.setattr(moe, "sp_moe_gather", gather)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", scatter)

    result = moe.FusedMoE.forward_impl(layer, hidden, logits)

    assert result.shape == hidden.shape
    assert method.apply.call_args.kwargs["x"] is gathered
    assert method.apply.call_args.kwargs["sp_input_scale"] is scale
    scatter.assert_called_once_with(gathered)
    if local_routing:
        method.gather_sp_routed_input.assert_called_once_with(
            layer, hidden, logits, token_group=None
        )
        method.gather_sp_input.assert_not_called()
        gather.assert_not_called()
        assert method.apply.call_args.kwargs["sp_topk"] == (weights, ids)
    else:
        method.gather_sp_routed_input.assert_not_called()
        method.gather_sp_input.assert_called_once_with(layer, hidden, token_group=None)
        gather.assert_called_once_with(logits)
        assert "sp_topk" not in method.apply.call_args.kwargs


@pytest.mark.parametrize("sp_enabled,all2all", [(False, False), (True, True)])
def test_non_sp_and_routed_all2all_skip_sp_gather(
    monkeypatch, sp_modules, sp_enabled, all2all
):
    moe = sp_modules.moe
    layer = _layer(moe)
    layer.moe_parallel_config.use_all2all_kernels = all2all
    hidden, logits = torch.empty(4, 2), torch.empty(4, 4)
    layer.quant_method.apply = Mock(return_value=hidden)
    gather, scatter = Mock(), Mock()
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: sp_enabled)
    monkeypatch.setattr(moe, "sp_moe_gather", gather)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", scatter)

    assert moe.FusedMoE.forward_impl(layer, hidden, logits) is hidden

    gather.assert_not_called()
    scatter.assert_not_called()
    assert "sp_topk" not in layer.quant_method.apply.call_args.kwargs
    assert "sp_input_scale" not in layer.quant_method.apply.call_args.kwargs


@pytest.mark.parametrize("rows", [1, 8192])
def test_tp_token_gather_returns_global_partial_without_reduction(
    monkeypatch, sp_modules, rows
):
    moe = sp_modules.moe
    layer = _layer(moe)
    group = SimpleNamespace(world_size=4)
    hidden = torch.empty(rows, 2, dtype=torch.bfloat16)
    logits = torch.empty(rows, 128)
    gathered = torch.empty(rows * 4, 2, dtype=torch.bfloat16)
    scale, weights, ids = (object() for _ in range(3))
    method = layer.quant_method
    method._sp_input_can_prequantize = Mock(return_value=True)
    method.gather_sp_input = Mock(return_value=(gathered, scale))
    method.gather_sp_routed_input = Mock(return_value=(gathered, scale, weights, ids))
    method.apply = Mock(return_value=gathered)
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(moe, "get_sp_world_size", lambda: 1)
    gather = Mock(side_effect=lambda value, *, group: value.repeat(group.world_size, 1))
    monkeypatch.setattr(moe, "_all_gather_tokens", gather)
    forbidden = Mock(side_effect=AssertionError("unexpected SP collective/reduction"))
    monkeypatch.setattr(moe, "sp_moe_gather", forbidden)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", forbidden)
    monkeypatch.setattr(moe, "get_tp_group", forbidden)

    result = moe.FusedMoE.forward_impl(layer, hidden, logits, token_group=group)

    assert result is gathered and result.shape[0] == rows * 4
    assert method.apply.call_args.kwargs["x"] is gathered
    if rows == 8192:
        method.gather_sp_routed_input.assert_called_once_with(
            layer, hidden, logits, token_group=group
        )
        assert method.apply.call_args.kwargs["sp_topk"] == (weights, ids)
        gather.assert_not_called()
    else:
        method.gather_sp_input.assert_called_once_with(layer, hidden, token_group=group)
        gather.assert_called_once_with(logits, group=group)
    forbidden.assert_not_called()


def test_prequantized_gather_preserves_payload_and_scale_bytes(monkeypatch, sp_modules):
    moe = sp_modules.moe
    layer = _layer(moe)
    method = layer.quant_method
    method._sp_input_can_prequantize = Mock(return_value=True)
    hidden = torch.empty(2, 512, dtype=torch.bfloat16)
    payload_bytes = torch.arange(512).to(torch.uint8).reshape(2, 256)
    scale_bytes = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    quantize = Mock(
        return_value=(
            payload_bytes.view(moe.dtypes.fp4x2),
            scale_bytes.view(moe.dtypes.fp8_e8m0),
        )
    )
    monkeypatch.setattr(moe, "get_hip_quant", lambda _quant_type: quantize)
    monkeypatch.setattr(moe, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(moe, "sp_moe_gather", lambda value: value.repeat(4, 1))

    payload, scale = method.gather_sp_input(layer, hidden)

    quantize.assert_called_once_with(
        hidden, quant_dtype=moe.dtypes.fp4x2, shuffle=False
    )
    assert torch.equal(payload.view(torch.uint8), payload_bytes.repeat(4, 1))
    assert torch.equal(scale.view(torch.uint8), scale_bytes.repeat(4, 1))


def test_routed_gather_preserves_local_route_weights_and_ids(monkeypatch, sp_modules):
    moe = sp_modules.moe
    layer = _layer(moe)
    method = layer.quant_method
    hidden = torch.empty(3, 2, dtype=torch.bfloat16)
    logits = torch.empty(3, 128)
    weights = torch.linspace(0.1, 0.9, 12).reshape(3, 4)
    ids = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    payload, scale = object(), object()
    method.select_experts_with_record = Mock(return_value=(weights, ids))
    method.gather_sp_input = Mock(return_value=(payload, scale))
    monkeypatch.setattr(moe, "sp_moe_gather", lambda value: value.repeat(4, 1))

    gathered_payload, gathered_scale, gathered_weights, gathered_ids = (
        method.gather_sp_routed_input(layer, hidden, logits)
    )

    assert gathered_payload is payload
    assert gathered_scale is scale
    assert torch.equal(
        gathered_weights.view(torch.int32), weights.repeat(4, 1).view(torch.int32)
    )
    assert torch.equal(gathered_ids, ids.repeat(4, 1))
    selection = method.select_experts_with_record.call_args.kwargs
    assert selection["hidden_states"] is hidden
    assert selection["router_logits"] is logits


def test_unsupported_prequantization_gathers_original_input(monkeypatch, sp_modules):
    moe = sp_modules.moe
    layer = _layer(moe)
    method = layer.quant_method
    method._sp_input_can_prequantize = Mock(return_value=False)
    hidden = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    quantize = Mock()
    monkeypatch.setattr(moe, "get_hip_quant", quantize)
    monkeypatch.setattr(moe, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(moe, "sp_moe_gather", lambda value: value.repeat(4, 1))

    gathered, scale = method.gather_sp_input(layer, hidden)

    assert torch.equal(gathered, hidden.repeat(4, 1))
    assert scale is None
    quantize.assert_not_called()


@pytest.mark.parametrize(
    "tiled_sort,prequantized", [(False, False), (False, True), (True, True)]
)
def test_apply_passes_only_supported_sp_options(
    monkeypatch, sp_modules, tiled_sort, prequantized
):
    moe = sp_modules.moe
    layer = _layer(moe)
    layer._sp_tiled_sort_enabled = tiled_sort
    layer.w13_weight = layer.w2_weight = object()
    layer.w13_weight_scale = layer.w2_weight_scale = object()
    layer.w13_input_scale = layer.w2_input_scale = None
    layer.w13_bias = layer.w2_bias = layer.expert_mask = None
    method = layer.quant_method
    method.fused_experts = None
    method.use_triton = method.use_triton_decode = method.use_triton_ep = False
    method.is_guinterleave = False
    weights, ids, hidden, output = (object() for _ in range(4))
    input_scale = object() if prequantized else None
    method.select_experts_with_record = Mock(return_value=(weights, ids))
    run_moe = Mock(return_value=output)
    monkeypatch.setattr(moe, "fused_moe", run_moe)

    result = method.apply(
        layer,
        hidden,
        object(),
        top_k=4,
        renormalize=True,
        sp_input_scale=input_scale,
        sp_topk=(weights, ids) if prequantized else None,
    )

    assert result is output
    assert run_moe.call_args.args == (
        hidden,
        layer.w13_weight,
        layer.w2_weight,
        weights,
        ids,
    )
    options = run_moe.call_args.kwargs
    assert options["a1_scale"] is input_scale
    if prequantized:
        method.select_experts_with_record.assert_not_called()
        assert options["dtype"] == torch.bfloat16
    else:
        method.select_experts_with_record.assert_called_once()
        assert "dtype" not in options
    if tiled_sort:
        assert options["use_tiled_sort"] is True
    else:
        assert "use_tiled_sort" not in options


@pytest.mark.parametrize(
    "override,world,gfx,plugin,expected",
    [
        ({}, 4, "gfx950", False, True),
        ({}, 8, "gfx950", False, False),
        ({}, 4, "gfx942", False, False),
        ({}, 4, "gfx950", True, False),
        ({"use_ep": True}, 4, "gfx950", False, False),
        ({"custom_routing_function": object()}, 4, "gfx950", False, False),
        ({"hidden_size": 7168}, 4, "gfx950", False, False),
        ({"intermediate_size_per_partition": 1536}, 4, "gfx950", False, False),
        ({"num_fused_shared_experts": 0}, 4, "gfx950", False, False),
    ],
)
def test_tiled_sort_uses_hardware_and_layer_contract(
    monkeypatch, sp_modules, override, world, gfx, plugin, expected
):
    layer = _layer(sp_modules.moe)
    vars(layer).update(override)
    monkeypatch.setattr(sp_modules.sp, "get_sp_world_size", lambda: world)
    monkeypatch.setattr(sp_modules.chip_info, "get_gfx_runtime", lambda: gfx)
    monkeypatch.setattr(sp_modules.plugin, "is_plugin_mode", lambda: plugin)
    monkeypatch.setattr(
        sp_modules.moe, "fused_moe", lambda *, use_tiled_sort=False: None
    )

    assert sp_modules.supports_tiled_sort(layer) is expected


def test_older_aiter_falls_back_to_existing_sort(monkeypatch, sp_modules):
    layer = _layer(sp_modules.moe)
    monkeypatch.setattr(sp_modules.sp, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(sp_modules.plugin, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(sp_modules.moe, "fused_moe", lambda hidden: hidden)
    probe = Mock(side_effect=AssertionError("unsupported AITER needs no GPU probe"))
    monkeypatch.setattr(sp_modules.chip_info, "get_gfx_runtime", probe)

    assert not sp_modules.supports_tiled_sort(layer)
    probe.assert_not_called()
