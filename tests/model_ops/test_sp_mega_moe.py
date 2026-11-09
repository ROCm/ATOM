# SPDX-License-Identifier: MIT
"""CPU checks of SP MegaMoE dispatch, tiling and setup agreement."""

import sys
from functools import partial
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.model_ops.fused_moe import sp_mega_moe as fusion
from atom.model_ops.fused_moe.sp_mega_moe import SPMegaMoE, local_token_tile


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
    layer.sp_mega_moe = SimpleNamespace(runs=backend.runs, forward=Mock())
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
    parallel_state.get_pcp_group = lambda: group
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)


def _model(layer):
    return SimpleNamespace(modules=lambda: [torch.nn.Identity(), layer])


def _config(backend="mega", mnbt=1024):
    return SimpleNamespace(
        moe_backend=backend, sequence_parallel_size=4, max_num_batched_tokens=mnbt
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


def _initialize(monkeypatch, sp_modules, runtime, mnbt=1024):
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
        fusion.initialize_sp_mega_moe, _model(layer), _config(mnbt=mnbt)
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
