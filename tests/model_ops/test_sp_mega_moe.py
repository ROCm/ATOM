# SPDX-License-Identifier: MIT
"""CPU checks of SP MegaMoE dispatch, tiling and setup agreement."""

import sys
from functools import partial
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.model_ops.fused_moe import sp_mega_moe as fusion
from atom.model_ops.fused_moe.sp_mega_moe import SPMegaMoE, TPMegaMoE, local_token_tile


def _layer(moe):
    method = object.__new__(moe.Mxfp4MoEMethod)
    method.quant_type = moe.QuantType.per_1x32
    method.fp8_activations = False
    return SimpleNamespace(
        quant_method=method,
        expert_layout=SimpleNamespace(
            shared_is_routed=False, uses_dispatch_remap=False
        ),
        hidden_size=6144,
        intermediate_size_per_partition=768,
        local_num_experts=129,
        top_k=4,
        num_fused_shared_experts=1,
        activation=moe.ActivationType.Swiglu,
        apply_router_weight_on_input=False,
        reduce_results=False,
        prefix="experts",
    )


def test_small_decode_batches_keep_the_native_moe():
    """AITER's own resolver: at SP4, M3's MXFP4 SwiGLU MoE quantizes to FP4,
    so MegaMoE runs, from 256 global tokens, 64 rows a rank. First in the
    module, so that on a GPU host it imports the real MoE before any stub."""
    resolver = pytest.importorskip("aiter.fused_moe")
    if not hasattr(resolver, "resolve_activation_dtype"):
        pytest.skip("AITER without resolve_activation_dtype")
    from atom.model_ops import moe

    layer = _layer(moe)
    layer.quant_method.is_guinterleave = False
    layer.w13_weight = torch.empty(1, dtype=moe.dtypes.fp4x2)
    runtime = SimpleNamespace(cfg=SimpleNamespace(world_size=4))
    backend = SPMegaMoE(runtime, layer, 1, None)
    assert [backend.runs(r) for r in (1, 63, 64, 8192)] == [False, False, True, True]


@pytest.mark.parametrize("use_mega", [False, True])
def test_mega_moe_runs_where_aiter_picks_fp4_activations(
    monkeypatch, sp_modules, use_mega
):
    """AITER's resolver decides, asked about the padded global batch exactly
    as the native MoE would ask it."""
    from importlib import import_module

    moe = sp_modules.moe
    layer = _layer(moe)
    layer.quant_method.is_guinterleave = False
    layer.w13_weight = torch.empty(1, dtype=moe.dtypes.fp4x2)
    resolver = Mock(return_value=moe.dtypes.fp4x2 if use_mega else torch.bfloat16)
    monkeypatch.setattr(
        import_module("aiter.fused_moe"),
        "resolve_activation_dtype",
        resolver,
        raising=False,
    )
    runtime = SimpleNamespace(cfg=SimpleNamespace(world_size=4), forward=Mock())
    backend = SPMegaMoE(runtime, layer, 1, None)
    rows = 64
    assert backend.runs(rows) is use_mega
    resolver.assert_called_once_with(
        moe.QuantType.per_1x32,
        moe.dtypes.fp4x2,
        activation=moe.ActivationType.Swiglu,
        gate_mode=moe.GateMode.SEPARATED,
        M=rows * 4,
        gate_up_interleave=None,
    )
    if not use_mega:
        return
    layer.sp_mega_moe = SimpleNamespace(
        runs=backend.runs, forward=Mock(), serves_block=False
    )
    x, logits = torch.ones(rows, 8), torch.zeros(rows, 128)
    out = moe.FusedMoE.forward_impl(layer, x, logits)
    assert out is layer.sp_mega_moe.forward.return_value
    layer.sp_mega_moe.forward.assert_called_once_with(layer, x, logits)


@pytest.mark.parametrize(
    ("budget", "world_size", "tile"),
    [(2048, 4, 512), (4098, 4, 1025), (65536, 4, 8192), (16385, 8, 2049)],
)
def test_tile_covers_the_largest_local_shard(budget, world_size, tile):
    assert local_token_tile(budget, world_size) == tile


@pytest.mark.parametrize("rows", [8, 65])
def test_large_batches_run_in_local_tiles_into_a_fresh_output(rows):
    launches = []

    def launch(hidden, weights, ids, *, out):
        # Every tile keeps its own rows' routing.
        torch.testing.assert_close(ids[:, :1].bfloat16(), hidden[:, :1])
        launches.append(hidden.shape[0])
        out.copy_(hidden * 2)

    runtime = SimpleNamespace(max_local_tokens=32, set_weights=Mock(), forward=launch)
    backend = SPMegaMoE(runtime, None, 1, None)
    backend.runs = lambda rows: True
    backend._route = lambda layer, hidden, logits: (
        torch.ones(hidden.shape[0], 5),
        hidden[:, :1].int().expand(-1, 5),
    )
    layer = SimpleNamespace(
        w13_weight=object(),
        w13_weight_scale=object(),
        w2_weight=object(),
        w2_weight_scale=object(),
    )
    x = (torch.arange(rows) % 8).bfloat16()[:, None].expand(rows, 8).contiguous()
    before = x.clone()
    out = backend.forward(layer, x, torch.empty(rows, 128))
    torch.testing.assert_close(out, before * 2)
    torch.testing.assert_close(x, before)
    assert launches == ([32, 32, 1] if rows == 65 else [8])
    runtime.set_weights.assert_called_once_with(
        w1=layer.w13_weight,
        w1_scale=layer.w13_weight_scale,
        w2=layer.w2_weight,
        w2_scale=layer.w2_weight_scale,
    )


def _servable_layer(moe):
    layer = _layer(moe)
    method = layer.quant_method
    vars(method).update(
        quant_dtype=moe.dtypes.fp4x2,
        is_guinterleave=False,
        use_triton=False,
        use_triton_decode=False,
        use_triton_ep=False,
        fused_experts=None,
        hidden_pad=0,
        intermediate_pad=0,
    )
    hidden, inter, experts = 256, 128, 5
    vars(layer).update(
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        local_num_experts=experts,
        w13_bias=None,
        w2_bias=None,
        w13_input_scale=None,
        w2_input_scale=None,
    )
    for name, shape in (
        ("w13_weight", (experts, 2 * inter, hidden // 2)),
        ("w2_weight", (experts, hidden, inter // 2)),
        ("w13_weight_scale", (experts * 2 * inter, hidden // 32)),
        ("w2_weight_scale", (experts * hidden, inter // 32)),
    ):
        value = torch.empty(shape, dtype=torch.uint8)
        value.is_shuffled = True
        setattr(layer, name, value)
    layer.sp_mega_moe = None
    return layer


def _pcp_group(monkeypatch, group):
    parallel_state = ModuleType("aiter.dist.parallel_state")
    parallel_state.get_pcp_group = parallel_state.get_tp_group = lambda: group
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)


def _model(layer):
    return SimpleNamespace(modules=lambda: [torch.nn.Identity(), layer])


def _config(backend="mega", mnbt=1024, tp=False):
    return SimpleNamespace(
        moe_backend=backend,
        sequence_parallel_size=1 if tp else 4,
        enable_expert_parallel=False,
        max_num_batched_tokens=mnbt,
    )


@pytest.mark.parametrize("backend", ["standard", "mega"])
def test_only_the_mega_backend_runs_it_and_refuses_ineligible_layers(
    monkeypatch, sp_modules, backend
):
    layer = _servable_layer(sp_modules.moe)
    assert fusion._unsupported_layer(layer) is None
    layer.w2_weight = layer.w2_weight.view(5, 128, 128)
    assert "w2_weight" in fusion._unsupported_layer(layer)

    _pcp_group(monkeypatch, SimpleNamespace(cpu_group=None))
    monkeypatch.setattr(fusion, "_unsupported_config", lambda config: None)
    initialize = partial(fusion.initialize_sp_mega_moe, _model(layer), _config(backend))
    if backend == "standard":
        assert initialize() is None
        return
    with pytest.raises(RuntimeError, match="experts: w2_weight"):
        initialize()


class MegaMoeSetupError(RuntimeError):
    pass


def _initialize(monkeypatch, sp_modules, runtime, mnbt=1024, tp=False):
    layer = _servable_layer(sp_modules.moe)
    mega = ModuleType("aiter.ops.flydsl.mega_moe_tp")
    mega.MegaMoeTPConfig = dict
    mega.MegaMoeTP = runtime
    mega.MegaMoeSetupError = MegaMoeSetupError
    monkeypatch.setitem(sys.modules, mega.__name__, mega)
    _pcp_group(
        monkeypatch, SimpleNamespace(world_size=4, rank_in_group=0, cpu_group=None)
    )
    monkeypatch.setattr(fusion, "_unsupported_config", lambda config: None)
    return layer, partial(
        fusion.initialize_sp_mega_moe, _model(layer), _config(mnbt=mnbt, tp=tp)
    )


def test_agreed_prepare_failure_closes_the_runtime_and_raises(monkeypatch, sp_modules):
    runtime = Mock(max_local_tokens=256, bigm_m=0)
    runtime.prepare_all.side_effect = MegaMoeSetupError(
        "MegaMoeTP.prepare: warmup launches timed out"
    )
    factory = Mock(return_value=runtime)
    _, initialize = _initialize(monkeypatch, sp_modules, factory)
    with pytest.raises(RuntimeError, match="setup failed: MegaMoeTP.prepare"):
        initialize()
    assert factory.call_args.args[0]["max_local_tokens"] == 256
    runtime.close.assert_called_once_with()


def test_epoch_count_covers_every_tile_of_the_largest_shard(monkeypatch, sp_modules):
    # 65536 tokens over SP4 are 16384 local rows: eight 2048-row launches.
    runtime = Mock(
        return_value=Mock(
            max_local_tokens=2048, bigm_m=0, **{"poll_errors.return_value": 0}
        )
    )
    layer, initialize = _initialize(monkeypatch, sp_modules, runtime, mnbt=65536)
    backend = initialize()
    assert runtime.call_args.args[0]["max_local_tokens"] == 8192
    runtime.return_value.prepare_all.assert_called_once_with()
    assert layer.sp_mega_moe is backend
    backend.runs = lambda rows: True
    backend.before_step(16384)
    assert backend.launches == 8


def test_warmup_timeout_on_any_rank_closes_the_runtime_and_raises(
    monkeypatch, sp_modules
):
    # The big-M warmup raises nothing: a timed-out wait is only in the bits.
    runtime = Mock(max_local_tokens=256, bigm_m=0, **{"poll_errors.return_value": 0})
    layer, initialize = _initialize(monkeypatch, sp_modules, Mock(return_value=runtime))
    monkeypatch.setattr(fusion, "group_max", lambda bits, group: max(bits, 32))
    with pytest.raises(RuntimeError, match="warmup wait timed out"):
        initialize()
    runtime.close.assert_called_once_with()
    assert layer.sp_mega_moe is None


@pytest.fixture
def host_copies(monkeypatch):
    # The timeout bits are copied to pinned memory behind an event.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=Mock(), query=lambda: True)
    )


def _counted(runtime, layers=1):
    backend = SPMegaMoE(runtime, None, layers, None)
    # Single-row steps are decode, which keeps the native MoE.
    backend.runs = lambda rows: rows > 1
    return backend


def test_only_megamoe_steps_are_counted(monkeypatch, host_copies):
    monkeypatch.setattr(fusion, "RESET_LAUNCHES", 8)
    error_flag = Mock(return_value=torch.zeros(1, dtype=torch.int32))
    runtime = SimpleNamespace(
        reset=Mock(), error_flag=error_flag, poll_errors=lambda: 0, max_local_tokens=16
    )
    backend = _counted(runtime, layers=3)
    for rows in (1, 16, 1, 1, 1):
        backend.before_step(rows)
    assert backend.launches == 3
    # One host copy covers the MegaMoE step; decode adds none.
    assert error_flag.call_count == 1
    backend.before_step(32)
    runtime.reset.assert_called_once_with()
    assert backend.launches == 0


def test_timed_out_launch_fails_a_later_step_and_skips_the_close(host_copies):
    # The step's output is already invalid; later steps must not run on.
    flag = torch.zeros(1, dtype=torch.int32)
    runtime = SimpleNamespace(
        error_flag=lambda: flag, cfg=Mock(rank=2), max_local_tokens=16, close=Mock()
    )
    backend = _counted(runtime)
    backend.before_step(16)
    flag[0] = 4  # a wait in this step timed out
    backend.before_step(1)
    with pytest.raises(RuntimeError, match="rank 2: a MegaMoeTP wait timed out"):
        backend.before_step(1)
    # Peers still in the step would never reach the collective close().
    backend.close()
    runtime.close.assert_not_called()
    _counted(runtime).close()
    runtime.close.assert_called_once_with()


@pytest.mark.parametrize("local,peer", [(4, 0), (0, 4)])
def test_epoch_reset_first_fails_on_any_rank_timeout(
    monkeypatch, host_copies, local, peer
):
    # reset() clears the bits, which the async copy may not have read yet.
    monkeypatch.setattr(fusion, "RESET_LAUNCHES", 1)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=Mock(), query=lambda: False)
    )
    monkeypatch.setattr(fusion, "group_max", lambda bits, group: max(bits, peer))
    runtime = SimpleNamespace(
        error_flag=lambda: torch.zeros(1, dtype=torch.int32),
        poll_errors=lambda: local,
        reset=Mock(),
        cfg=Mock(rank=2),
        max_local_tokens=16,
    )
    backend = _counted(runtime)
    with pytest.raises(RuntimeError, match="rank 2: a MegaMoeTP wait timed out"):
        backend.before_step(16)
    runtime.reset.assert_not_called()


# -- pure TP (TPMegaMoE) --------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "selected"),
    [
        ({}, True),
        ({"moe_backend": "standard"}, False),
        ({"enable_expert_parallel": True}, False),  # MoRI MegaMoE
        ({"sequence_parallel_size": 4}, False),
    ],
)
def test_mega_backend_selects_the_tp_mode_on_pure_tp_only(override, selected):
    config = _config(tp=True)
    vars(config).update(override)
    assert fusion.tp_mega_moe_selected(config) is selected


def _tp_servable_layer(moe):
    layer = _servable_layer(moe)
    vars(layer).update(
        local_num_experts=4,
        num_fused_shared_experts=0,
        tp_mega_shared={"w1_shared": 5, "w1_shared_scale": 6},
        scoring_func="sigmoid",
        use_grouped_topk=False,
        custom_routing_function=None,
    )
    for name, shape in (
        ("w13_weight", (4, 256, 128)),
        ("w2_weight", (4, 256, 64)),
        ("w13_weight_scale", (4 * 256, 8)),
        ("w2_weight_scale", (4 * 256, 4)),
    ):
        value = torch.empty(shape, dtype=torch.uint8)
        value.is_shuffled = True
        setattr(layer, name, value)
    return layer


def test_tp_needs_the_mxfp8_shared_expert_kept_apart_and_unreduced(sp_modules):
    layer = _tp_servable_layer(sp_modules.moe)
    assert fusion._unsupported_layer(layer, tp=True) is None
    # SP serves the fused shared expert only.
    assert "fused shared expert" in fusion._unsupported_layer(layer)
    layer.num_fused_shared_experts = 1
    assert "kept apart" in fusion._unsupported_layer(layer, tp=True)
    layer.num_fused_shared_experts = 0
    # A step the kernel does not take must stay TP-partial, like the standard MoE.
    layer.reduce_results = True
    assert "needs unreduced" in fusion._unsupported_layer(layer, tp=True)
    layer.reduce_results = False
    layer.tp_mega_shared = None
    assert "tp_mega_shared" in fusion._unsupported_layer(layer, tp=True)
    layer.tp_mega_shared = {}
    layer.scoring_func = "softmax"
    assert "sigmoid" in fusion._unsupported_layer(layer, tp=True)


def _tp_runtime(max_local=16, bigm_m=0, **attrs):
    return SimpleNamespace(
        max_local_tokens=max_local,
        bigm_m=bigm_m,
        cfg=SimpleNamespace(world_size=4, rank=0),
        **attrs,
    )


@pytest.mark.parametrize("rows", [8, 1, 3, 13])
def test_tp_forward_is_one_launch_with_the_shared_route_appended(monkeypatch, rows):
    """Routed weights scaled by the routed scale, then the shared expert (the
    last expert, weight 1); the output buffer is the caller's, the kernel pads
    a row count that is not a multiple of tp itself."""
    import aiter

    from atom.utils import forward_context

    launches = []

    def launch(hidden, weights, ids, *, out):
        launches.append((hidden.shape[0], weights.clone(), ids.clone()))
        out.copy_(hidden * 2)

    def gating(weights, ids, logits, bias, renormalize, scale, *, score_func, **kw):
        weights.copy_(torch.full_like(weights, 0.25) * scale)
        ids.copy_(torch.arange(weights.shape[1], dtype=torch.int32))

    monkeypatch.setattr(aiter, "topk_gating", gating, raising=False)
    monkeypatch.setattr(forward_context, "pass_scheduled_tokens_gpu", lambda r: None)
    runtime = _tp_runtime(set_weights=Mock(), forward=launch)
    backend = TPMegaMoE(runtime, None, 1, None)
    layer = SimpleNamespace(
        w13_weight=1, w13_weight_scale=2, w2_weight=3, w2_weight_scale=4,
        tp_mega_shared={"w1_shared": 5, "w1_shared_scale": 6, "w2_shared": 7,
                        "w2_shared_scale": 8},
        top_k=4, local_num_experts=128, e_score_correction_bias=None,
        renormalize=True, routed_scaling_factor=2.0, scoring_func="sigmoid",
    )  # fmt: skip
    x = (torch.arange(rows) + 1).bfloat16()[:, None].expand(rows, 8).contiguous()
    before = x.clone()
    out = backend.forward(layer, x, torch.zeros(rows, 128))
    torch.testing.assert_close(out, before * 2)
    torch.testing.assert_close(x, before)
    runtime.set_weights.assert_called_once_with(
        w1=1, w1_scale=2, w2=3, w2_scale=4,
        w1_shared=5, w1_shared_scale=6, w2_shared=7, w2_shared_scale=8,
    )  # fmt: skip
    ((launched, weights, ids),) = launches
    assert launched == rows
    assert (weights[:, :4] == 0.5).all() and (weights[:, 4] == 1.0).all()
    assert (ids[:, :4] == torch.arange(4)).all() and (ids[:, 4] == 128).all()


def test_tp_serves_the_small_engine_and_big_m_but_not_between(monkeypatch):
    # 32768 rows a step (8192 local), big-M from 8192 rows (2048 local).
    backend = TPMegaMoE(_tp_runtime(max_local=8192, bigm_m=2048), None, 1, None)
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    top, (low, high) = fusion.TP_SMALL_MAX_ROWS, fusion.TP_STANDARD_ROWS
    served = [0, 1, 256, 257, low - 1, low, high, high + 1, top, top + 1, 8191]
    served += [8192, 32768, 32769]
    expected = [False, True, True, True, True, False, False, True, True, False]
    expected += [False, True, True, False]
    assert [backend.runs(rows) for rows in served] == expected
    assert [backend.tier(r) for r in (16, 257, high, top + 1, 8192)] == [
        "small-M",
        "LB",
        "standard",
        "standard",
        "big-M",
    ]
    # Big-M is not graph-capturable: a recorded step keeps the standard MoE.
    capturing[0] = True
    assert backend.runs(256) and not backend.runs(8192)
    # No big-M path (AITER without it, or a small step budget).
    small = TPMegaMoE(_tp_runtime(max_local=64), None, 1, None)
    assert [small.runs(rows) for rows in (1, 256, 257)] == [True, True, False]


def test_tp_steps_count_one_launch_a_layer_and_their_coverage(monkeypatch, host_copies):
    monkeypatch.setattr(fusion, "RESET_LAUNCHES", 8)
    runtime = _tp_runtime(
        reset=Mock(), error_flag=lambda: torch.zeros(1, dtype=torch.int32),
        poll_errors=lambda: 0,
    )  # fmt: skip
    backend = TPMegaMoE(runtime, None, 3, None)
    for rows in (1, 64, 65, 0):
        backend.before_step(rows)
    assert backend.launches == 6
    assert backend.coverage == {"small-M": [2, 65], "standard": [1, 65]}
    backend.before_step(8)
    runtime.reset.assert_called_once_with()
    assert backend.launches == 0


def test_tp_runtime_is_the_ar_kernel_with_the_shared_expert(monkeypatch, sp_modules):
    runtime = Mock(
        max_local_tokens=8192,
        bigm_m=2048,
        cfg=SimpleNamespace(world_size=4, rank=0),
        **{"poll_errors.return_value": 0},
    )
    factory = Mock(return_value=runtime)
    layer, initialize = _initialize(monkeypatch, sp_modules, factory, 32768, tp=True)
    vars(layer).update(vars(_tp_servable_layer(sp_modules.moe)))
    backend = initialize()
    cfg = factory.call_args.args[0]
    assert (cfg["comm_mode"], cfg["comm_dtype"], cfg["act_dtype"]) == (
        "ar",
        "bf16",
        "auto",
    )
    # 4 routed experts + the shared one, top-4 + 1; AITER's own big-M threshold.
    assert (cfg["experts"], cfg["topk"]) == (5, 5)
    assert "bigm_min_tokens" not in cfg and cfg["max_local_tokens"] == 8192
    assert factory.call_args.kwargs["w1_shared"] == 5
    # The small-batch engine's row counts (4096 rows, 1024 local).
    local = fusion.TP_SMALL_MAX_ROWS // 4
    assert list(runtime.prepare.call_args.args[0]) == list(range(1, local + 1))
    assert isinstance(backend, TPMegaMoE) and layer.sp_mega_moe is backend
    assert backend.bigm_rows == 8192 and backend.max_rows == 32768


def test_tp_refuses_a_fused_shared_expert(monkeypatch, sp_modules):
    _, initialize = _initialize(monkeypatch, sp_modules, Mock(), tp=True)
    with pytest.raises(RuntimeError, match="with TP: experts: needs the MXFP8"):
        initialize()
