# SPDX-License-Identifier: MIT
"""CPU checks of the real SP dispatch, tiling and shared-runtime lifetime."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_comm_fused_moe import atom_modules  # noqa: F401
from test_sp_moe_dispatch import _layer
from test_sp_moe_dispatch import sp_modules as _sp_modules

from atom.model_ops.sp_mega_moe import (
    MegaMoESettings,
    SPMegaMoE,
    _validate_layer,
    initialize_sp_mega_moe,
    local_token_tile,
)


@pytest.fixture(autouse=True)
def mock_mega_buffer_ops(monkeypatch):
    from atom.model_ops import sp_mega_moe

    # Replace only the GPU copy boundary for CPU orchestration tests. The real
    # copies and launch counts are covered on GPU.
    def copy(source, source_scale, out, out_scale):
        out.view(torch.uint8).copy_(source.view(torch.uint8))
        if source_scale is not None:
            out_scale.view(-1).copy_(source_scale)

    monkeypatch.setattr(sp_mega_moe, "copy_tail", copy)


@pytest.fixture
def sp_modules(request):
    return _sp_modules.__wrapped__(request)


@pytest.mark.parametrize("rows", [0, 1, 16, 32, 63, 64, 65, 2048, 4096, 8192, 8193])
def test_tiles_preserve_token_routing_and_output_ownership(rows):
    # Token IDs encode rank and local row. The fake device boundary verifies
    # each routing slice belongs to the same token, including the last tile.
    x = torch.arange(rows * 2).view(rows, 2).to(torch.bfloat16)
    ids = torch.arange(rows, dtype=torch.int32)[:, None].expand(rows, 5).contiguous()
    weights = ids.float() + 0.25
    method = SimpleNamespace(
        select_experts_with_record=Mock(return_value=(weights, ids))
    )
    layer = SimpleNamespace(
        quant_method=method,
        use_grouped_topk=False,
        top_k=4,
        num_fused_shared_experts=1,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        global_num_experts=128,
        custom_routing_function=None,
        scoring_func="sigmoid",
        e_score_correction_bias=object(),
        shared_expert_scoring_func=None,
        w13_weight=object(),
        w2_weight=object(),
        w13_weight_scale=object(),
        w2_weight_scale=object(),
    )
    calls = []

    def launch(hidden, routing_weights, routing_ids, *, out):
        start = sum(calls)
        end = start + hidden.shape[0]
        torch.testing.assert_close(hidden, x[start:end])
        assert torch.equal(routing_ids, ids[start:end])
        assert torch.equal(routing_weights, weights[start:end])
        calls.append(hidden.shape[0])
        # Stand in for the already reduced result. Returning a reusable buffer
        # instead of honoring out= would overwrite earlier tiles/layers.
        out.copy_(hidden + 2)
        return out

    runtime = SimpleNamespace(
        max_local_tokens=64,
        set_weights=Mock(),
        forward=launch,
    )
    backend = SPMegaMoE(runtime)
    first = backend.forward(layer, x, object())
    torch.testing.assert_close(first, x + 2)
    assert calls == [min(64, rows - n) for n in range(0, rows, 64)]
    bound_weights = runtime.set_weights.call_args.kwargs
    assert bound_weights["w1"] is layer.w13_weight
    assert bound_weights["w2"] is layer.w2_weight
    routed = method.select_experts_with_record.call_args.kwargs
    assert routed["e_score_correction_bias"] is layer.e_score_correction_bias
    assert routed["top_k"] == 4
    assert routed["renormalize"] is True
    calls.clear()
    second = backend.forward(layer, x, object())
    second.zero_()
    torch.testing.assert_close(first, x + 2)


def test_mega_dispatch_owns_gather_and_reduction(monkeypatch, sp_modules):
    moe = sp_modules.moe
    layer = _layer(moe)
    x, logits = torch.empty(16, 2), torch.empty(16, 128)
    result = torch.ones_like(x)
    layer._sp_mega_moe = SimpleNamespace(
        supports=Mock(return_value=True), forward=Mock(return_value=result)
    )
    layer.quant_method.apply = Mock(side_effect=AssertionError("duplicate MoE"))
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: True)
    gather, scatter = Mock(), Mock()
    monkeypatch.setattr(moe, "sp_moe_gather", gather)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", scatter)

    assert moe.FusedMoE.forward_impl(layer, x, logits) is result
    layer._sp_mega_moe.forward.assert_called_once_with(layer, x, logits)
    gather.assert_not_called()
    scatter.assert_not_called()


@pytest.mark.parametrize(
    "rows,use_mega", [(1, False), (16, False), (63, False), (64, True), (65, True)]
)
def test_native_activation_fallback_keeps_sp_collectives(
    monkeypatch, sp_modules, rows, use_mega
):
    from importlib import import_module

    moe = sp_modules.moe
    layer = _layer(moe)
    layer.w13_weight = torch.empty(1, dtype=moe.dtypes.fp4x2)
    hidden = torch.empty(rows, 2, dtype=torch.bfloat16)
    logits = torch.empty(rows, 128)
    gathered = torch.empty(rows * 4, 2, dtype=torch.bfloat16)
    candidate = torch.ones_like(hidden)
    backend = SPMegaMoE(
        SimpleNamespace(cfg=SimpleNamespace(act_dtype="fp4", world_size=4))
    )
    backend.forward = Mock(return_value=candidate)
    layer._sp_mega_moe = backend
    method = layer.quant_method
    method._sp_input_can_prequantize = Mock(return_value=False)
    method.gather_sp_input = Mock(return_value=(gathered, None))
    method.apply = Mock(return_value=gathered)
    gather = Mock(return_value=logits.repeat(4, 1))
    scatter = Mock(return_value=hidden)
    resolver = Mock(return_value=moe.dtypes.fp4x2 if use_mega else torch.bfloat16)
    monkeypatch.setattr(
        import_module("aiter.fused_moe"),
        "resolve_activation_dtype",
        resolver,
        raising=False,
    )
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: True)
    monkeypatch.setattr(moe, "sp_moe_gather", gather)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", scatter)

    result = moe.FusedMoE.forward_impl(layer, hidden, logits)

    if use_mega:
        assert result is candidate
        backend.forward.assert_called_once_with(layer, hidden, logits)
        method.apply.assert_not_called()
        gather.assert_not_called()
        scatter.assert_not_called()
    else:
        assert result is hidden
        backend.forward.assert_not_called()
        method.gather_sp_input.assert_called_once_with(layer, hidden, token_group=None)
        gather.assert_called_once_with(logits)
        scatter.assert_called_once_with(gathered)
        assert method.apply.call_args.kwargs["x"] is gathered
    assert resolver.call_args.kwargs["M"] == rows * 4


def test_disabled_does_not_import_aiter_or_probe_gpu(monkeypatch):
    monkeypatch.setenv("ATOM_SP_MOE_MEGA", "0")
    initialize_sp_mega_moe(None, None)


def test_activation_change_requires_explicit_diagnostic_override():
    backend = SPMegaMoE(None, allow_activation_change=True)
    assert backend.supports(None, None)


def test_native_fallback_consumes_fused_routes_without_logits(monkeypatch, sp_modules):
    moe = sp_modules.moe
    layer = _layer(moe)
    hidden = torch.empty(16, 2, dtype=torch.bfloat16)
    weights = torch.rand(16, 5)
    ids = torch.arange(80, dtype=torch.int32).view(16, 5)
    layer._sp_mega_moe = SimpleNamespace(supports=lambda *_: False)
    method = layer.quant_method
    method.gather_sp_input = Mock(return_value=(hidden.repeat(4, 1), None))
    method.select_experts_with_record = Mock(
        side_effect=AssertionError("duplicate topk")
    )
    method.apply = Mock(return_value=hidden.repeat(4, 1))
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: True)
    gather = Mock(side_effect=lambda t: t.repeat(4, 1))
    monkeypatch.setattr(moe, "sp_moe_gather", gather)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", lambda x: x[:16])
    moe.FusedMoE.forward_impl(layer, hidden, None, sp_topk=(weights, ids))
    assert gather.call_count == 2
    args = method.apply.call_args.kwargs
    assert args["router_logits"] is None
    torch.testing.assert_close(args["sp_topk"][0], weights.repeat(4, 1))
    torch.testing.assert_close(args["sp_topk"][1], ids.repeat(4, 1))


@pytest.mark.parametrize(
    "settings",
    [{"act_dtype": "bf16"}, {"comm_dtype": "int4"}],
)
def test_unsupported_kernel_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        MegaMoESettings(**settings)


@pytest.mark.parametrize(
    ("budget", "world_size", "act_dtype", "tile"),
    [
        (2048, 4, "fp4", 512),
        (4098, 4, "fp4", 1025),
        (32768, 4, "fp4", 8192),
        (65536, 4, "fp4", 8192),
        (2048, 2, "fp4", 1024),
        (16385, 8, "fp4", 2049),
        (65536, 8, "fp4", 8192),
        (16384, 4, "fp8", 64),
        (100, 4, "fp8", 25),
        (16384, 2, "fp8", 128),
        (16384, 8, "fp8", 32),
    ],
)
def test_tile_covers_the_largest_local_shard(budget, world_size, act_dtype, tile):
    assert local_token_tile(budget, world_size, act_dtype) == tile


def _loaded_layer(moe):
    layer = _layer(moe)
    method = layer.quant_method
    method.quant_dtype = moe.dtypes.fp4x2
    method.is_guinterleave = False
    method.use_triton = method.use_triton_decode = method.use_triton_ep = False
    method.fused_experts = None
    method.hidden_pad = method.intermediate_pad = 0
    layer.w13_bias = layer.w2_bias = None
    layer.w13_input_scale = layer.w2_input_scale = None
    for name, shape in (
        ("w13_weight", (129, 1536, 3072)),
        ("w2_weight", (129, 6144, 384)),
        ("w13_weight_scale", (129 * 1536, 192)),
        ("w2_weight_scale", (129 * 6144, 24)),
    ):
        value = torch.empty(shape, dtype=torch.uint8, device="meta")
        value.is_shuffled = True
        setattr(layer, name, value)
    return layer


def test_loaded_m3_weight_contract(sp_modules):
    _validate_layer(_loaded_layer(sp_modules.moe))


@pytest.mark.parametrize(
    "owner,key,value",
    [
        ("method", "is_guinterleave", True),
        ("method", "use_triton", True),
        ("method", "intermediate_pad", 128),
        ("layer", "reduce_results", True),
        ("layer", "use_ep", True),
        ("layer", "num_fused_shared_experts", 0),
        ("layer", "apply_router_weight_on_input", True),
        ("layer", "w13_input_scale", object()),
        ("layer", "w2_bias", object()),
        ("layer", "custom_routing_function", object()),
    ],
)
def test_incompatible_loaded_layers_fail_before_collectives(
    sp_modules, owner, key, value
):
    layer = _loaded_layer(sp_modules.moe)
    setattr(layer.quant_method if owner == "method" else layer, key, value)
    with pytest.raises(ValueError, match="requires M3 SP4"):
        _validate_layer(layer)


def test_unshuffled_weights_are_rejected(sp_modules):
    layer = _loaded_layer(sp_modules.moe)
    layer.w13_weight.is_shuffled = False
    with pytest.raises(ValueError, match="incompatible MegaMoE layout"):
        _validate_layer(layer)


def test_missing_shared_expert_route_is_rejected():
    layer = SimpleNamespace(
        quant_method=SimpleNamespace(
            select_experts_with_record=lambda **kwargs: (
                torch.empty(2, 4),
                torch.empty(2, 4, dtype=torch.int32),
            )
        ),
        use_grouped_topk=False,
        top_k=4,
        num_fused_shared_experts=1,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        global_num_experts=128,
        custom_routing_function=None,
        scoring_func="sigmoid",
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
    )
    with pytest.raises(ValueError, match="include the fused shared expert"):
        SPMegaMoE(None).forward(layer, torch.empty(2, 2, dtype=torch.bfloat16), None)


@pytest.mark.parametrize("lb_mt", [None, "3", "512:3,6"])
def test_initialize_shares_prepared_runtime_across_layers(
    monkeypatch, sp_modules, lb_mt
):
    from importlib import import_module

    from conftest import atom_config_double

    # This test supplies bare MoE layers. Decoder fusion is covered separately;
    # importing its GPU attention stack would defeat this CPU runtime test.
    class Decoder(torch.nn.Module):
        pass

    decoder_module = ModuleType("atom.models.minimax_m3.model")
    decoder_module.MiniMaxM3DecoderLayer = Decoder
    monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)

    state = import_module("aiter.dist.parallel_state")
    api = import_module("aiter.ops.flydsl.mega_moe_tp")
    moe = sp_modules.moe
    layers = []
    for _ in range(3):
        # Keep the real lazy decorator's construction and returned type; only
        # skip weight allocation, which needs distributed GPU initialization.
        with monkeypatch.context() as constructor_patch:
            constructor_patch.setattr(
                moe.FusedMoE.__mro__[1], "__init__", torch.nn.Module.__init__
            )
            layer = moe.FusedMoE()
        for name, value in vars(_loaded_layer(moe)).items():
            if name in ("use_ep", "num_fused_shared_experts"):
                continue
            setattr(layer, name, value)
        layer.moe_parallel_config.use_ep = False
        layer.expert_layout.num_fused_shared_experts = 1
        layer.swiglu_limit = 7.0
        layers.append(layer)
    model = torch.nn.ModuleList(layers)
    group = SimpleNamespace(world_size=4, rank_in_group=2, cpu_group=object())
    monkeypatch.setattr(state, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(state, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(sp_modules.sp, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(sp_modules.sp, "get_sp_group", lambda: group)
    monkeypatch.setattr(sp_modules.chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(sp_modules.plugin, "is_plugin_mode", lambda: False)
    monkeypatch.setenv("ATOM_SP_MOE_MEGA", "1")
    monkeypatch.setenv("ATOM_SP_MOE_MEGA_ACT_DTYPE", "fp8")
    monkeypatch.setenv("ATOM_SP_MOE_MEGA_COMM_DTYPE", "bf16")
    monkeypatch.setenv("ATOM_SP_MOE_MEGA_ALLOW_ACTIVATION_CHANGE", "1")
    if lb_mt is None:
        monkeypatch.delenv("AITER_MEGAMOE_TP_LB_MT", raising=False)
    else:
        monkeypatch.setenv("AITER_MEGAMOE_TP_LB_MT", lb_mt)
    runtime = SimpleNamespace(
        # Emulate the kernel's shape buckets, including a remainder bucket.
        launch_config=lambda rows: "small" if rows <= 8 else "large",
        prepare=Mock(),
    )
    factory = Mock(return_value=runtime)
    monkeypatch.setattr(api, "MegaMoeTPConfig", SimpleNamespace, raising=False)
    monkeypatch.setattr(api, "MegaMoeTP", factory, raising=False)
    config = atom_config_double(torch_dtype=torch.bfloat16, sp_mega_moe_enabled=True)

    initialize_sp_mega_moe(model, config)

    factory.assert_called_once()
    runtime.prepare.assert_called_once_with([1, 9])
    assert all(layer._sp_mega_moe is layers[0]._sp_mega_moe for layer in layers)
    assert layers[0]._sp_mega_moe.runtime is runtime
    assert layers[0]._sp_mega_moe.allow_activation_change
    cfg = factory.call_args.args[0]
    assert (cfg.rank, cfg.world_size, cfg.experts, cfg.topk) == (2, 4, 129, 5)
    assert cfg.comm_mode == "ag_rs"
    assert cfg.act_dtype == "fp8" and cfg.comm_dtype == "bf16"
    assert cfg.max_local_tokens == 64
    assert cfg.swiglu_limit == 7.0
    assert cfg.lb_mt == ("6" if lb_mt is None else lb_mt)
    assert factory.call_args.kwargs["group"] is group.cpu_group

    config.enable_rapidserve = True
    with pytest.raises(ValueError, match="without TBO, RapidServe"):
        initialize_sp_mega_moe(model, config)
    assert factory.call_count == 1
