# SPDX-License-Identifier: MIT
"""CPU checks of the TP MegaMoE backend: layer / deployment eligibility, row
padding, launch accounting and setup agreement. The kernel itself is checked by
``test_tp_mega_moe_gpu.py`` on gfx950."""

import sys
from functools import partial
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.model_ops.fused_moe import tp_mega_moe as fusion
from atom.model_ops.fused_moe.tp_mega_moe import TPMegaMoE


def _layer(moe):
    method = object.__new__(moe.Mxfp4MoEMethod)
    method.quant_type = moe.QuantType.per_1x32
    method.fp8_activations = False
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
    layer = SimpleNamespace(
        quant_method=method,
        expert_layout=SimpleNamespace(
            shared_is_routed=False, uses_dispatch_remap=False
        ),
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        local_num_experts=experts,
        top_k=4,
        num_fused_shared_experts=1,
        activation=moe.ActivationType.Swiglu,
        apply_router_weight_on_input=False,
        # The model's contract for a TP MegaMoE layer.
        reduce_results=True,
        tp_mega_moe_requested=True,
        tp_mega_moe=None,
        w13_bias=None,
        w2_bias=None,
        w13_input_scale=None,
        w2_input_scale=None,
        swiglu_limit=7.0,
        prefix="experts",
    )
    for name, shape in (
        ("w13_weight", (experts, 2 * inter, hidden // 2)),
        ("w2_weight", (experts, hidden, inter // 2)),
        ("w13_weight_scale", (experts * 2 * inter, hidden // 32)),
        ("w2_weight_scale", (experts * hidden, inter // 32)),
    ):
        value = torch.zeros(shape, dtype=torch.uint8)
        value.is_shuffled = True
        setattr(layer, name, value)
    return layer


def _config(data_parallel_size=1, **overrides):
    config = {
        "tensor_parallel_size": 4,
        "tp_world_size": 4,
        "pipeline_parallel_size": 1,
        "parallel_config": SimpleNamespace(data_parallel_size=data_parallel_size),
        "moe_backend": "mega",
        "max_num_batched_tokens": 16384,
        "enable_expert_parallel": False,
        "sequence_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "eplb_enable": False,
        "fake_eplb": False,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


@pytest.mark.parametrize(
    ("batched", "tp", "rows"),
    [(16384, 4, 4096), (8, 4, 8), (1, 4, 4), (10**6, 4, 4096), (10**6, 8, 4096)],
)
def test_rows_are_bounded_by_the_batch_budget_and_the_cap(batched, tp, rows):
    config = _config(max_num_batched_tokens=batched)
    assert fusion.tp_mega_moe_max_rows(config, tp) == rows


@pytest.mark.parametrize(
    ("override", "selected"),
    [
        ({}, True),
        ({"moe_backend": "standard"}, False),
        ({"enable_expert_parallel": True}, False),
        ({"sequence_parallel_size": 2}, False),
    ],
)
def test_moe_backend_mega_means_tp_megamoe_only_on_pure_tp(override, selected):
    # EP runs the MoRI MegaMoE and SP the Ulysses one.
    assert fusion.tp_mega_moe_selected(_config(**override)) is selected


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"tensor_parallel_size": 1, "tp_world_size": 1}, "tensor_parallel_size"),
        ({"tp_world_size": 2}, "tp_world_size"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size"),
        ({"data_parallel_size": 2}, "data_parallel_size"),
        ({"prefill_context_parallel_size": 2}, "prefill_context_parallel_size"),
        ({"eplb_enable": True}, "eplb"),
        ({"fake_eplb": True}, "eplb"),
    ],
)
def test_deployments_outside_pure_tp_are_refused(override, field):
    assert fusion._unsupported_config(_config()) is None
    assert fusion._unsupported_config(_config(**override)).startswith(field)


def test_servable_layer_and_each_way_it_stops_being_one(sp_modules):
    moe = sp_modules.moe
    layer = _layer(moe)
    assert fusion._unsupported_layer(layer) is None

    def refused(mutate, match):
        broken = _layer(moe)
        mutate(broken)
        assert match in fusion._unsupported_layer(broken)

    # The production FlyDSL layout, read in place.
    refused(lambda l: setattr(l.quant_method, "use_triton", True), "MXFP4 FlyDSL")
    refused(lambda l: setattr(l.quant_method, "is_guinterleave", True), "MXFP4 FlyDSL")
    refused(lambda l: setattr(l.quant_method, "intermediate_pad", 256), "MXFP4 FlyDSL")
    refused(lambda l: setattr(l.w13_weight, "is_shuffled", False), "w13_weight")
    refused(
        lambda l: setattr(l, "w2_weight", l.w2_weight.view(5, 128, 128)), "w2_weight"
    )
    # Its output is all-reduced by every path.
    refused(lambda l: setattr(l, "reduce_results", False), "all-reduced contract")
    refused(
        lambda l: setattr(l.expert_layout, "shared_is_routed", True),
        "all-reduced contract",
    )
    # The kernel runs the shared expert as one more routed expert.
    refused(lambda l: setattr(l, "num_fused_shared_experts", 0), "fused shared expert")
    refused(lambda l: setattr(l, "w13_bias", object()), "no biases")
    refused(lambda l: setattr(l, "activation", moe.ActivationType.Silu), "SwiGLU")


class _Runtime:
    """MegaMoeTP stand-in: y = x * 2 where routed, recording what it was given."""

    def __init__(self, tp=4, max_local_tokens=16):
        self.cfg = SimpleNamespace(world_size=tp, rank=0)
        self.max_local_tokens = max_local_tokens
        self.calls = []
        self.set_weights = Mock()

    def forward(self, x, weights, ids, *, out):
        self.calls.append((x.clone(), weights.clone(), ids.clone(), out.shape))
        out.copy_(torch.where((ids[:, :1] >= 0), x * 2, torch.zeros_like(x)))
        return out


def _backend(tp=4, max_rows=64, runtime=None):
    runtime = runtime or _Runtime(tp)
    h, k = 8, 5
    staging = (
        torch.zeros(max_rows, h, dtype=torch.bfloat16),
        torch.zeros(max_rows, k),
        torch.full((max_rows, k), -1, dtype=torch.int32),
        torch.zeros(max_rows, h, dtype=torch.bfloat16),
    )
    backend = TPMegaMoE(runtime, None, 2, None, max_rows, staging)
    backend._route = lambda layer, hidden, logits: (
        torch.ones(hidden.shape[0], k),
        torch.arange(k, dtype=torch.int32).expand(hidden.shape[0], -1).contiguous(),
    )
    return runtime, backend


_WEIGHT_NAMES = {
    "w13_weight": object(),
    "w13_weight_scale": object(),
    "w2_weight": object(),
    "w2_weight_scale": object(),
}


@pytest.mark.parametrize(
    ("rows", "serves"),
    [(0, False), (1, True), (63, True), (64, True), (65, False), (4096, False)],
)
def test_serves_steps_up_to_its_capacity(rows, serves):
    _, backend = _backend()
    assert backend.runs(rows) is serves


def test_aligned_rows_launch_in_place_into_a_fresh_output():
    runtime, backend = _backend()
    x = torch.arange(8, dtype=torch.float32).bfloat16()[:, None].expand(8, 8)
    x = x.contiguous()
    before = x.clone()
    layer = SimpleNamespace(**_WEIGHT_NAMES)
    out = backend.forward(layer, x, torch.empty(8, 128))
    torch.testing.assert_close(out, before * 2)
    torch.testing.assert_close(x, before)
    assert out.data_ptr() != x.data_ptr()
    # One launch of exactly the step's rows, on this layer's weights.
    assert [c[3][0] for c in runtime.calls] == [8]
    runtime.set_weights.assert_called_once_with(
        w1=layer.w13_weight,
        w1_scale=layer.w13_weight_scale,
        w2=layer.w2_weight,
        w2_scale=layer.w2_weight_scale,
    )


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 7, 13])
def test_unaligned_rows_are_padded_with_rows_routed_nowhere(rows):
    runtime, backend = _backend()
    padded = -(-rows // 4) * 4
    x = (torch.arange(rows) + 1).bfloat16()[:, None].expand(rows, 8).contiguous()
    before = x.clone()
    # A previous, bigger step left stale rows in the staging buffers.
    backend._staging[0].fill_(7)
    backend._staging[1].fill_(7)
    backend._staging[2].fill_(3)
    layer = SimpleNamespace(**_WEIGHT_NAMES)
    out = backend.forward(layer, x, torch.empty(rows, 128))
    assert out.shape == x.shape
    torch.testing.assert_close(out, before * 2)
    torch.testing.assert_close(x, before)
    ((sx, sw, si, shape),) = runtime.calls
    # The kernel sees a multiple of tp rows; the pad rows are zero activations
    # with weight 0 and expert id -1, whatever the staging held before.
    assert sx.shape[0] == padded and shape[0] == padded
    assert not sx[rows:].any() and not sw[rows:].any() and (si[rows:] == -1).all()
    torch.testing.assert_close(sx[:rows], before)


def test_forward_allocates_nothing_it_keeps_between_calls():
    # The staging is allocated once (before graph capture); a step only fills it.
    runtime, backend = _backend()
    staging = [t.data_ptr() for t in backend._staging]
    layer = SimpleNamespace(**_WEIGHT_NAMES)
    for rows in (3, 5, 3):
        backend.forward(layer, torch.ones(rows, 8, dtype=torch.bfloat16), None)
    assert [t.data_ptr() for t in backend._staging] == staging
    assert len(runtime.calls) == 3


@pytest.fixture
def host_copies(monkeypatch):
    # The timeout bits are copied to pinned memory behind an event.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=Mock(), query=lambda: True)
    )


def test_each_served_step_counts_one_launch_a_layer(monkeypatch, host_copies):
    monkeypatch.setattr(fusion, "RESET_LAUNCHES", 12)
    error_flag = Mock(return_value=torch.zeros(1, dtype=torch.int32))
    runtime = _Runtime()
    runtime.error_flag = error_flag
    runtime.poll_errors = lambda: 0
    runtime.reset = Mock()
    _, backend = _backend(runtime=runtime)
    # 2 layers: a step is 2 launches; a step beyond the capacity keeps the
    # production MoE and is not counted.
    for rows in (4, 1, 128, 0, 4096):
        backend.before_step(rows)
    assert backend.launches == 4
    assert (backend.steps, backend.steps_served) == (5, 2)
    # The error bits are copied behind the launches only, never host-synced.
    assert error_flag.call_count == 2
    for _ in range(4):
        backend.before_step(8)
    # 12 launches: the epochs are reset on every rank.
    runtime.reset.assert_called_once_with()
    assert backend.launches == 0


def test_timed_out_launch_fails_a_later_step_and_skips_the_close(host_copies):
    flag = torch.zeros(1, dtype=torch.int32)
    runtime = _Runtime()
    runtime.error_flag = lambda: flag
    runtime.cfg.rank = 2
    runtime.close = Mock()
    _, backend = _backend(runtime=runtime)
    backend.before_step(16)
    flag[0] = 4  # a wait in this step timed out
    backend.before_step(8)
    with pytest.raises(RuntimeError, match="rank 2: a MegaMoeTP wait timed out"):
        backend.before_step(8)
    # Peers still in the step would never reach the collective close().
    backend.close()
    runtime.close.assert_not_called()


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
    runtime = _Runtime()
    runtime.cfg.rank = 2
    runtime.error_flag = lambda: torch.zeros(1, dtype=torch.int32)
    runtime.poll_errors = lambda: local
    runtime.reset = Mock()
    _, backend = _backend(runtime=runtime)
    with pytest.raises(RuntimeError, match="rank 2: a MegaMoeTP wait timed out"):
        backend.before_step(16)
    runtime.reset.assert_not_called()


# -- initialization -----------------------------------------------------------


def _model(*layers):
    return SimpleNamespace(modules=lambda: [torch.nn.Identity(), *layers])


def _tp_group(monkeypatch, group):
    parallel_state = ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: group
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)


def _mega_module(monkeypatch, runtime_factory):
    mega = ModuleType("aiter.ops.flydsl.mega_moe_tp")
    mega.MegaMoeTPConfig = dict
    mega.MegaMoeTP = runtime_factory
    monkeypatch.setitem(sys.modules, mega.__name__, mega)


def _runtime(**attrs):
    return Mock(max_local_tokens=1024, **{"poll_errors.return_value": 0}, **attrs)


def test_another_backend_leaves_the_model_alone(sp_modules):
    layer = _layer(sp_modules.moe)
    for override in (
        {"moe_backend": "standard"},
        {"enable_expert_parallel": True},
        {"sequence_parallel_size": 2},
    ):
        assert fusion.initialize_tp_mega_moe(_model(layer), _config(**override)) is None
    assert layer.tp_mega_moe is None


def test_a_model_without_layers_that_serve_it_is_refused(monkeypatch, sp_modules):
    layer = _layer(sp_modules.moe)
    layer.tp_mega_moe_requested = False
    _tp_group(monkeypatch, SimpleNamespace())
    with pytest.raises(RuntimeError, match="no MoE layer of this model serves it"):
        fusion.initialize_tp_mega_moe(_model(layer), _config())
    assert layer.tp_mega_moe is None


def test_an_ineligible_layer_or_deployment_raises_instead_of_falling_back(
    monkeypatch, sp_modules
):
    # Its neighbours' outputs are all-reduced already: no silent fallback.
    layer = _layer(sp_modules.moe)
    _tp_group(monkeypatch, SimpleNamespace())
    with pytest.raises(RuntimeError, match="needs PP 1"):
        fusion.initialize_tp_mega_moe(_model(layer), _config(pipeline_parallel_size=2))
    layer.w2_weight = layer.w2_weight.view(5, 128, 128)
    with pytest.raises(RuntimeError, match="experts: w2_weight"):
        fusion.initialize_tp_mega_moe(_model(layer), _config())
    assert layer.tp_mega_moe is None


def _initialize(monkeypatch, sp_modules, runtime_factory, **config):
    layer = _layer(sp_modules.moe)
    _mega_module(monkeypatch, runtime_factory)
    _tp_group(
        monkeypatch, SimpleNamespace(world_size=4, rank_in_group=1, cpu_group=None)
    )
    return layer, partial(
        fusion.initialize_tp_mega_moe, _model(layer), _config(**config)
    )


def test_runtime_is_built_on_the_ar_kernel_prepared_for_every_local_count(
    monkeypatch, sp_modules
):
    factory = Mock(return_value=_runtime())
    layer, initialize = _initialize(monkeypatch, sp_modules, factory)
    backend = initialize()
    cfg = factory.call_args.args[0]
    assert (cfg["comm_mode"], cfg["comm_dtype"], cfg["act_dtype"]) == (
        "ar",
        "bf16",
        "auto",
    )
    # 128 + 1 shared experts, top-4 + 1, the model's SwiGLU limit, TP rank / size.
    assert (cfg["experts"], cfg["topk"], cfg["swiglu_limit"]) == (5, 5, 7.0)
    assert (cfg["rank"], cfg["world_size"]) == (1, 4)
    # The small engine alone, sized for the default 4096 rows (1024 local).
    assert cfg["bigm_min_tokens"] == 0 and cfg["max_local_tokens"] == 1024
    assert list(factory.return_value.prepare.call_args.args[0]) == list(range(1, 1025))
    kwargs = factory.call_args.kwargs
    assert kwargs["w1"] is layer.w13_weight and kwargs["w2"] is layer.w2_weight
    assert layer.tp_mega_moe is backend
    assert backend.max_rows == 4096 and backend.runs(4096) and not backend.runs(4097)
    # Staging for the unaligned steps exists before any graph is captured.
    assert backend._staging[0].shape == (4096, 256)


def test_the_step_limit_follows_the_batch_budget(monkeypatch, sp_modules):
    factory = Mock(return_value=_runtime())
    factory.return_value.max_local_tokens = 128
    _, initialize = _initialize(
        monkeypatch, sp_modules, factory, max_num_batched_tokens=512
    )
    backend = initialize()
    assert factory.call_args.args[0]["max_local_tokens"] == 128
    assert backend.max_rows == 512


def test_capacity_the_kernel_lowers_the_step_limit_to_is_what_is_served(
    monkeypatch, sp_modules
):
    runtime = _runtime()
    runtime.max_local_tokens = 600  # fit_max_local lowered 1024
    _, initialize = _initialize(monkeypatch, sp_modules, Mock(return_value=runtime))
    assert initialize().max_rows == 2400


def test_agreed_setup_error_closes_the_runtime_and_raises(monkeypatch, sp_modules):
    runtime = _runtime()
    runtime.prepare.side_effect = ValueError("does not fit in LDS")
    _, initialize = _initialize(monkeypatch, sp_modules, Mock(return_value=runtime))
    with pytest.raises(RuntimeError, match="setup failed: does not fit in LDS"):
        initialize()
    runtime.close.assert_called_once_with()


def test_warmup_timeout_raises_without_a_collective_close(monkeypatch, sp_modules):
    # prepare() raises RuntimeError on every rank when a warmup launch timed
    # out, but a peer may not have arrived at a collective close().
    runtime = _runtime()
    runtime.prepare.side_effect = RuntimeError("warmup launches timed out")
    _, initialize = _initialize(monkeypatch, sp_modules, Mock(return_value=runtime))
    with pytest.raises(RuntimeError, match="warmup launches timed out"):
        initialize()
    runtime.close.assert_not_called()


def test_warmup_timeout_on_any_rank_closes_the_runtime_and_raises(
    monkeypatch, sp_modules
):
    runtime = _runtime()
    layer, initialize = _initialize(monkeypatch, sp_modules, Mock(return_value=runtime))
    monkeypatch.setattr(fusion, "group_max", lambda bits, group: max(bits, 32))
    with pytest.raises(RuntimeError, match="warmup wait timed out"):
        initialize()
    runtime.close.assert_called_once_with()
    assert layer.tp_mega_moe is None
