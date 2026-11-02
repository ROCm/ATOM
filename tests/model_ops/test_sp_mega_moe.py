# SPDX-License-Identifier: MIT
"""CPU checks of SP MegaMoE dispatch, tiling and setup agreement."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_comm_fused_moe import atom_modules  # noqa: F401
from test_sp_moe_dispatch import sp_modules as _sp_modules

from atom.model_ops import sp_mega_moe as fusion
from atom.model_ops.sp_mega_moe import SPMegaMoE, local_token_tile


@pytest.fixture
def sp_modules(request):
    return _sp_modules.__wrapped__(request)


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


@pytest.mark.parametrize("rows,use_mega", [(16, False), (63, False), (64, True)])
def test_small_batches_keep_the_native_moe(monkeypatch, sp_modules, rows, use_mega):
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
    assert backend.runs(rows) is use_mega
    assert resolver.call_args.kwargs["M"] == rows * 4
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
    layer.allow_sp_mega_moe = True
    return layer


def _pcp_group(monkeypatch, group):
    parallel_state = ModuleType("aiter.dist.parallel_state")
    parallel_state.get_pcp_group = lambda: group
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)


def _model(layer):
    return SimpleNamespace(modules=lambda: [torch.nn.Identity(), layer])


def test_layer_shapes_follow_the_model_and_ineligible_layers_stay_native(
    monkeypatch, sp_modules
):
    layer = _servable_layer(sp_modules.moe)
    assert fusion._unsupported_layer(layer) is None
    layer.w2_weight = layer.w2_weight.view(5, 128, 128)
    assert "w2_weight" in fusion._unsupported_layer(layer)

    # Enabled by default: an ineligible model must load on the native MoE.
    monkeypatch.setattr(fusion, "get_sp_world_size", lambda: 4)
    _pcp_group(monkeypatch, SimpleNamespace(cpu_group=None))
    monkeypatch.setattr(fusion, "_unsupported_config", lambda config: None)
    assert fusion.initialize_sp_mega_moe(_model(layer), SimpleNamespace()) is None
    assert getattr(layer, "sp_mega_moe", None) is None


def _initialize(monkeypatch, sp_modules, runtime, mnbt=1024, **stubs):
    layer = _servable_layer(sp_modules.moe)
    mega = ModuleType("aiter.ops.flydsl.mega_moe_tp")
    mega.MegaMoeTPConfig = dict
    mega.MegaMoeTP = runtime
    comm = ModuleType("aiter.ops.communication")
    comm._mori = stubs.get("mori", Mock())
    for module in (mega, comm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fusion, "get_sp_world_size", lambda: 4)
    _pcp_group(
        monkeypatch, SimpleNamespace(world_size=4, rank_in_group=0, cpu_group=None)
    )
    monkeypatch.setattr(fusion, "_unsupported_config", lambda config: None)
    config = SimpleNamespace(max_num_batched_tokens=mnbt)
    return layer, lambda: fusion.initialize_sp_mega_moe(_model(layer), config)


@pytest.mark.parametrize(
    "error,falls_back",
    [
        (RuntimeError("SdmaComm: MORI SDMA setup failed on ranks [0, 1, 2, 3]"), True),
        # One rank's compile error, lazy import or OOM leaves its peers inside
        # a collective.
        (RuntimeError("compile failed"), False),
        (ImportError("lazy import failed"), False),
        (torch.OutOfMemoryError("HIP out of memory"), False),
    ],
)
def test_runtime_setup_failure_keeps_the_native_moe(
    monkeypatch, sp_modules, error, falls_back
):
    runtime = Mock(side_effect=error)
    layer, initialize = _initialize(monkeypatch, sp_modules, runtime)
    if not falls_back:
        with pytest.raises(type(error)):
            initialize()
        return
    assert initialize() is None
    assert runtime.call_args.args[0]["max_local_tokens"] == 256
    assert getattr(layer, "sp_mega_moe", None) is None


def test_agreed_prepare_failure_closes_the_runtime(monkeypatch, sp_modules):
    runtime = Mock(max_local_tokens=256, bigm_m=0)
    runtime.prepare.side_effect = RuntimeError(
        "MegaMoeTP.prepare: warmup launches timed out"
    )
    _, initialize = _initialize(monkeypatch, sp_modules, Mock(return_value=runtime))
    assert initialize() is None
    runtime.close.assert_called_once_with()


def test_a_refusing_peer_keeps_every_rank_native(monkeypatch, sp_modules):
    runtime = Mock()
    layer, initialize = _initialize(monkeypatch, sp_modules, runtime)
    votes = []

    def peer_refuses(ok, group):
        votes.append(ok)
        return False

    monkeypatch.setattr(fusion, "group_agree", peer_refuses)
    assert initialize() is None
    # This rank could serve it.
    assert votes == [True]
    assert getattr(layer, "sp_mega_moe", None) is None
    runtime.assert_not_called()


# A 256-row tile: big-M is unreachable, reachable, or past the tile.
@pytest.mark.parametrize("bigm_m,probed", [(0, False), (128, True), (512, False)])
def test_mori_is_required_only_where_big_m_runs(
    monkeypatch, sp_modules, bigm_m, probed
):
    mori = Mock(side_effect=ImportError("no MORI"))
    runtime = Mock(max_local_tokens=256, bigm_m=bigm_m)
    runtime.poll_errors.return_value = 0
    runtime.launch_config.side_effect = lambda rows: rows > 64
    layer, initialize = _initialize(
        monkeypatch, sp_modules, Mock(return_value=runtime), mori=mori
    )
    backend = initialize()
    assert mori.called is probed
    if probed:
        assert backend is None
        runtime.close.assert_called_once_with()
        runtime.prepare.assert_not_called()
    else:
        assert layer.sp_mega_moe is backend
        runtime.prepare.assert_called_once_with([1, 65])


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
    assert layer.sp_mega_moe is backend
    backend.runs = lambda rows: True
    backend.count_step(16384)
    assert backend.launches == 8


def test_warmup_timeout_on_any_rank_keeps_the_native_moe(monkeypatch, sp_modules):
    # The big-M warmup raises nothing: a timed-out wait is only in the bits.
    runtime = Mock(max_local_tokens=256, bigm_m=0, **{"poll_errors.return_value": 0})
    layer, initialize = _initialize(monkeypatch, sp_modules, Mock(return_value=runtime))
    monkeypatch.setattr(fusion, "group_max", lambda bits, group: max(bits, 32))
    assert initialize() is None
    runtime.close.assert_called_once_with()
    assert getattr(layer, "sp_mega_moe", None) is None


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
        backend.count_step(rows)
    assert backend.launches == 3
    # One host copy covers the MegaMoE step; decode adds none.
    assert error_flag.call_count == 1
    backend.count_step(32)
    runtime.reset.assert_called_once_with()
    assert backend.launches == 0


def test_timed_out_launch_fails_a_later_step(host_copies):
    # The step's output is already invalid; later steps must not run on.
    flag = torch.zeros(1, dtype=torch.int32)
    runtime = SimpleNamespace(
        error_flag=lambda: flag, cfg=Mock(rank=2), max_local_tokens=16
    )
    backend = _counted(runtime)
    backend.count_step(16)
    flag[0] = 4  # a wait in this step timed out
    backend.count_step(1)
    with pytest.raises(RuntimeError, match="rank 2: a MegaMoeTP wait timed out"):
        backend.count_step(1)


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
        backend.count_step(16)
    runtime.reset.assert_not_called()
