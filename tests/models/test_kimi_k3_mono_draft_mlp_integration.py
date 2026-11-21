# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contracts of the gfx1250 DSpark draft MLP / o_proj kernels'
integration (``ATOM_KIMI_MONO_DRAFT_MLP``, ``ATOM_KIMI_MONO_DRAFT_OPROJ_AR``,
``ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS``): the switches, the model's hooks, the
bindings' refusals, a step's refusals, which kernels a width takes."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.mono.runtime.consensus import MonoUnsupported

aiter = pytest.importorskip("aiter", reason="the draft and MLA modules import aiter")
from aiter import QuantType, dtypes


def test_the_switches_default_off(monkeypatch):
    from atom.utils import envs

    for name in ("ATOM_KIMI_MONO_DRAFT_MLP", "ATOM_KIMI_MONO_DRAFT_OPROJ_AR"):
        monkeypatch.delenv(name, raising=False)
        assert getattr(envs, name) is False
        monkeypatch.setenv(name, "1")
        assert getattr(envs, name) is True
    monkeypatch.delenv("ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS", raising=False)
    assert envs.ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS == 35
    monkeypatch.setenv("ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS", "16")
    assert envs.ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS == 16


def _draft_module():
    try:
        from atom.models import kimi_k3_dspark
    except ImportError as error:
        pytest.skip(f"kimi_k3_dspark needs the GPU stack: {error}")
    return kimi_k3_dspark


class Stop(Exception):
    pass


def _stop(*a, **k):
    raise Stop


# ---------------------------------------------------------------- the hooks
def test_switched_off_the_mlp_never_asks_the_draft_kernels(monkeypatch):
    dspark = _draft_module()
    from atom.models.kimi_k3.mono import draft_mlp

    monkeypatch.setattr(
        draft_mlp, "forward_mlp", lambda *a: pytest.fail("asked with the switch off")
    )
    fake = SimpleNamespace(
        _try_mono_mlp=False, gate_up_proj=_stop, act_fn=None, down_proj=None
    )
    with pytest.raises(Stop):
        dspark.K3DSparkMLP.forward(fake, ("x", "scale"))


def test_switched_on_the_mlp_takes_the_kernels_output_or_production(monkeypatch):
    dspark = _draft_module()
    from atom.models.kimi_k3.mono import draft_mlp

    monkeypatch.setattr(draft_mlp, "forward_mlp", lambda mlp, x: ("out", x))
    fake = SimpleNamespace(_try_mono_mlp=True)
    assert dspark.K3DSparkMLP.forward(fake, "x") == ("out", "x")
    monkeypatch.setattr(draft_mlp, "forward_mlp", lambda *a: None)
    fake = SimpleNamespace(
        _try_mono_mlp=True, gate_up_proj=_stop, act_fn=None, down_proj=None
    )
    with pytest.raises(Stop):  # None: production's MLP runs
        dspark.K3DSparkMLP.forward(fake, ("x", "scale"))


def _o_proj(dspark, attempts=True):
    linear = object.__new__(dspark.K3DSparkOProj)
    object.__setattr__(linear, "_try_mono_proj", attempts)
    return linear


def test_the_o_proj_hook_takes_bf16_decode_calls_only(monkeypatch):
    dspark = _draft_module()
    from atom.models.kimi_k3.mono import draft_mlp

    asked = []
    monkeypatch.setattr(
        draft_mlp, "forward_oproj", lambda linear, x: asked.append(x) or "mono"
    )
    monkeypatch.setattr(
        dspark.RowParallelLinear,
        "forward",
        lambda self, x, x_scale=None, otype=None, out=None: "production",
        raising=False,
    )
    forward = dspark.K3DSparkOProj.forward
    assert forward(_o_proj(dspark), "x") == "mono"
    # a pre-quantized activation, an output buffer, another dtype, or a
    # layer whose attempts are off: the plain linear, never asked
    assert forward(_o_proj(dspark), "x", "scale") == "production"
    assert forward(_o_proj(dspark), "x", None, dtypes.bf16, "out") == "production"
    assert forward(_o_proj(dspark), "x", None, dtypes.fp16) == "production"
    assert forward(_o_proj(dspark, attempts=False), "x") == "production"
    assert asked == ["x"]
    monkeypatch.setattr(draft_mlp, "forward_oproj", lambda linear, x: None)
    assert forward(_o_proj(dspark), "x") == "production"


# -------------------------------------------------------------- the binding
def _linear(rows, cols, e8m0=True, shuffled=True, **change):
    padded = -(-rows // 128) * 128
    weight = nn.Parameter(
        torch.zeros(padded, cols, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    weight.is_shuffled = shuffled
    scale = torch.full((padded // 128, cols // 128), 120, dtype=torch.uint8)
    scale = nn.Parameter(
        scale.view(torch.float8_e8m0fnu) if e8m0 else scale.float(),
        requires_grad=False,
    )
    linear = {
        "weight": weight,
        "weight_scale": scale,
        "bias": None,
        "native_a8_group_rows": None,
        "quant_type": QuantType.per_1x128,
        "blockscale_e8m0_scale": e8m0,
        "params_dtype": dtypes.fp8,
        "input_scale": None,
        "is_output_padded": False,
        "_k_padded": False,
        "tp_dim": 1,
        "reduce_results": True,
        "prefix": "layers.0.self_attn.o_proj",
        "mono_proj": None,
        "_try_mono_proj": True,
    }
    linear.update(change)
    return SimpleNamespace(**linear)


def _norm(**change):
    norm = {
        "use_fused_quant": True,
        "fused_allreduce": False,
        "quant_type": QuantType.per_1x128,
        "params_dtype": dtypes.fp8,
        "_aiter_transpose_scale": True,
        "_aiter_e8m0_scale": True,
    }
    norm.update(change)
    return SimpleNamespace(**norm)


def _mlp(**change):
    from atom.model_ops.activation import SiluAndMul

    mlp = {
        "prefix": "layers.0.mlp",
        "_input_norm": _norm(),
        "gate_up_proj": _linear(3584, 7168, tp_dim=0, reduce_results=False),
        "down_proj": _linear(7168, 1792),
        "act_fn": SiluAndMul(),
        "mono_mlp": None,
        "_try_mono_mlp": True,
    }
    mlp.update(change)
    return SimpleNamespace(**mlp)


@pytest.fixture
def preshuffle(monkeypatch):
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")


def test_a_projection_binds_its_shard_in_place(preshuffle):
    from atom.models.kimi_k3.mono.draft_mlp import bind_proj

    linear = _linear(7168, 1024)
    layer = bind_proj(linear, "layers.0.self_attn.o_proj", 8)
    assert layer.k == 1024 and layer.prefix == "layers.0.self_attn.o_proj"
    assert layer.w.data_ptr() == linear.weight.data_ptr()
    assert layer.s.data_ptr() == linear.weight_scale.data_ptr()


@pytest.mark.parametrize(
    ("linear", "message"),
    [
        (lambda: _linear(7168, 1024, bias=torch.zeros(7168)), "bias"),
        (lambda: _linear(7168, 1024, e8m0=False), "E8M0"),
        (lambda: _linear(7168, 1024, quant_type=QuantType.per_Token), "E8M0"),
        (lambda: _linear(7168, 1024, native_a8_group_rows=32), "E8M0"),
        (lambda: _linear(7168, 1024, params_dtype=torch.bfloat16), "activations"),
        (lambda: _linear(7168, 1024, input_scale=torch.ones(1)), "static"),
        (lambda: _linear(7168, 1024, is_output_padded=True), "padded"),
        (lambda: _linear(7168, 1024, _k_padded=True), "padded"),
        (lambda: _linear(7168, 1024, reduce_results=False), "row-parallel"),
        (lambda: _linear(7168, 1024, tp_dim=0), "row-parallel"),
        (lambda: _linear(7168, 1024, shuffled=False), "preshuffled"),
        (lambda: _linear(7040, 1024), "expected"),
        (lambda: _linear(7168, 1152), "pairs"),
    ],
)
def test_a_projection_the_kernel_would_read_wrong_is_refused(
    preshuffle, linear, message
):
    from atom.models.kimi_k3.mono.draft_mlp import bind_proj

    with pytest.raises(MonoUnsupported, match=message):
        bind_proj(linear(), "o_proj", 8)


def test_without_the_preshuffle_no_projection_binds(monkeypatch):
    from atom.models.kimi_k3.mono.draft_mlp import bind_proj

    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "0")
    with pytest.raises(MonoUnsupported, match="E8M0"):
        bind_proj(_linear(7168, 1024), "o_proj", 8)


def test_an_mlp_binds_gate_up_and_down_in_place(preshuffle):
    from atom.models.kimi_k3.mono.draft_mlp import bind_mlp

    mlp = _mlp()
    layer = bind_mlp(mlp, 8)
    assert layer.inter == 1792 and layer.down.k == 1792
    assert layer.w_gu.data_ptr() == mlp.gate_up_proj.weight.data_ptr()
    assert layer.down.w.data_ptr() == mlp.down_proj.weight.data_ptr()
    assert layer.down.prefix == "layers.0.mlp down_proj"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda: {"_input_norm": None}, "input norm"),
        (lambda: {"_input_norm": _norm(_aiter_e8m0_scale=False)}, "E8M0 FP8 row"),
        (lambda: {"_input_norm": _norm(_aiter_transpose_scale=False)}, "E8M0 FP8 row"),
        (lambda: {"_input_norm": _norm(fused_allreduce=True)}, "E8M0 FP8 row"),
        (lambda: {"gate_up_proj": _linear(3584, 7168, e8m0=False)}, "gate_up_proj"),
        (lambda: {"gate_up_proj": _linear(3712, 7168)}, "rows"),
        (lambda: {"gate_up_proj": _linear(3584, 7040)}, "expected"),
        (lambda: {"act_fn": SimpleNamespace(fused_quant=False)}, "SiLU"),
        (lambda: {"down_proj": _linear(7168, 1024)}, "takes 1024"),
        (lambda: {"down_proj": _linear(7168, 1792, reduce_results=False)}, "row-par"),
    ],
)
def test_an_mlp_the_kernels_would_read_wrong_is_refused(preshuffle, change, message):
    from atom.models.kimi_k3.mono.draft_mlp import bind_mlp

    with pytest.raises(MonoUnsupported, match=message):
        bind_mlp(_mlp(**change()), 8)


def test_a_fused_activation_quant_is_refused(preshuffle):
    from atom.model_ops.activation import SiluAndMul
    from atom.models.kimi_k3.mono.draft_mlp import bind_mlp

    with pytest.raises(MonoUnsupported, match="SiLU"):
        bind_mlp(_mlp(act_fn=SiluAndMul(fused_quant=True)), 8)


# ----------------------------------------------------------------- the step
def _context(monkeypatch, **change):
    import atom.utils.forward_context as fc

    context = {"is_dummy_run": False, "is_prefill": False}
    context.update(change)
    monkeypatch.setattr(
        fc,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(**context)),
    )


def test_a_decode_steps_projection_is_taken(monkeypatch):
    from atom.models.kimi_k3.mono import draft_mlp

    _context(monkeypatch)
    layer = SimpleNamespace(prefix="o_proj", k=1024)
    x = torch.zeros(7, 1024, dtype=torch.bfloat16)
    assert draft_mlp.proj_step_refusal(layer, x) is None


@pytest.mark.parametrize(
    ("context", "x", "message"),
    [
        ({"is_dummy_run": True}, torch.zeros(7, 1024, dtype=torch.bfloat16), "decode"),
        ({"is_prefill": True}, torch.zeros(7, 1024, dtype=torch.bfloat16), "decode"),
        ({}, torch.zeros(7, 1024, dtype=torch.float32), "activation"),
        ({}, torch.zeros(7, 1792, dtype=torch.bfloat16), "activation"),
        ({}, torch.zeros(7, 2048, dtype=torch.bfloat16)[:, ::2], "activation"),
        ({}, torch.zeros(1, 7, 1024, dtype=torch.bfloat16), "activation"),
        ({}, (torch.zeros(7, 1024, dtype=torch.bfloat16), None), "activation"),
        ({}, torch.zeros(65, 1024, dtype=torch.bfloat16), "65 rows"),
    ],
)
def test_a_projection_step_the_kernel_cannot_take_is_refused(
    monkeypatch, context, x, message
):
    from atom.models.kimi_k3.mono import draft_mlp

    _context(monkeypatch, **context)
    layer = SimpleNamespace(prefix="o_proj", k=1024)
    assert message in draft_mlp.proj_step_refusal(layer, x)


def _norm_out(rows=7, dtype=None, scale_dtype=None, groups=56):
    return (
        torch.zeros(rows, 7168, dtype=dtype or dtypes.fp8),
        torch.zeros(rows, groups, dtype=scale_dtype or dtypes.fp8_e8m0),
    )


def test_a_decode_steps_mlp_input_is_taken(monkeypatch):
    from atom.models.kimi_k3.mono import draft_mlp

    _context(monkeypatch)
    x = _norm_out()
    why, found = draft_mlp.mlp_step_refusal(x)
    assert why is None and found[0] is x[0] and found[1] is x[1]


@pytest.mark.parametrize(
    ("context", "x", "message"),
    [
        ({"is_dummy_run": True}, _norm_out, "decode"),
        ({"is_prefill": True}, _norm_out, "decode"),
        ({}, lambda: torch.zeros(7, 7168, dtype=torch.bfloat16), "pair"),
        ({}, lambda: _norm_out(dtype=torch.bfloat16), "input row"),
        ({}, lambda: _norm_out(scale_dtype=torch.float32), "input row"),
        ({}, lambda: _norm_out(groups=55), "input row"),
        ({}, lambda: _norm_out(rows=65), "65 rows"),
    ],
)
def test_an_mlp_step_the_kernels_cannot_take_is_refused(
    monkeypatch, context, x, message
):
    from atom.models.kimi_k3.mono import draft_mlp

    _context(monkeypatch, **context)
    why, found = draft_mlp.mlp_step_refusal(x())
    assert message in why and found is None


# ------------------------------------------------------- binding, collective
class _Runner:
    """A rank's runner with its collectives stubbed out."""

    def __init__(self, kernels=None, banks=True):
        self.tp, self.rank, self.group = 1, 0, None
        self.k_oproj = self.inter = None
        self._kernels = kernels or {}
        self._banks = banks
        self.calls, self.refusals = [], []

    def banks(self):
        return object() if self._banks else None

    def kernel(self, kind, rows):
        return self._kernels.get(kind)

    def _refuse(self, why):
        self.refusals.append(why)

    def launch_gate_up(self, kernel, layer, x8, x_scale):
        self.calls.append(("gate_up", kernel))
        return "act"

    def launch_proj(self, kernel, layer, x):
        self.calls.append(("proj", kernel, layer, x))
        return "reduced"


def _wire(monkeypatch, runner, capturing=False):
    from atom.models.kimi_k3.mono import draft_mlp

    monkeypatch.setattr(draft_mlp, "runner_for", lambda device: runner)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    _context(monkeypatch)
    return draft_mlp


def test_a_refused_layer_turns_its_attempts_off(monkeypatch, preshuffle):
    runner = _Runner({"o_proj": "kernel"})
    draft_mlp = _wire(monkeypatch, runner)
    linear = _linear(7168, 1024, e8m0=False)
    x = torch.zeros(7, 1024, dtype=torch.bfloat16)
    assert draft_mlp.forward_oproj(linear, x) is None
    assert linear._try_mono_proj is False and linear.mono_proj is None
    assert runner.calls == []


def test_without_banks_on_the_tp_group_the_layer_is_refused(monkeypatch, preshuffle):
    runner = _Runner({"o_proj": "kernel"}, banks=False)
    draft_mlp = _wire(monkeypatch, runner)
    linear = _linear(7168, 1024)
    assert (
        draft_mlp.forward_oproj(linear, torch.zeros(7, 1024, dtype=torch.bfloat16))
        is None
    )
    assert linear._try_mono_proj is False


def test_no_layer_binds_while_a_graph_is_captured(monkeypatch, preshuffle):
    runner = _Runner({"o_proj": "kernel"})
    draft_mlp = _wire(monkeypatch, runner, capturing=True)
    monkeypatch.setattr(
        draft_mlp, "bind_proj", lambda *a: pytest.fail("bound during a capture")
    )
    linear = _linear(7168, 1024)
    assert (
        draft_mlp.forward_oproj(linear, torch.zeros(7, 1024, dtype=torch.bfloat16))
        is None
    )
    # still to be bound by an eager step
    assert linear._try_mono_proj is True and linear.mono_proj is None


def test_a_bound_o_proj_launches_the_width_kernel(monkeypatch, preshuffle):
    runner = _Runner({"o_proj": "kernel"})
    draft_mlp = _wire(monkeypatch, runner)
    linear = _linear(7168, 1024)
    x = torch.zeros(7, 1024, dtype=torch.bfloat16)
    assert draft_mlp.forward_oproj(linear, x) == "reduced"
    assert runner.k_oproj == 1024 and linear.mono_proj.k == 1024
    assert runner.calls == [("proj", "kernel", linear.mono_proj, x)]
    # a layer of another width is refused: one build serves every layer
    other = _linear(7168, 2048)
    assert (
        draft_mlp.forward_oproj(other, torch.zeros(7, 2048, dtype=torch.bfloat16))
        is None
    )
    assert other._try_mono_proj is False
    # a width with no kernel, or a step the kernel does not take: production
    runner._kernels = {}
    assert draft_mlp.forward_oproj(linear, x) is None
    runner._kernels = {"o_proj": "kernel"}
    assert draft_mlp.forward_oproj(linear, x.float()) is None
    assert any("activation" in why for why in runner.refusals)


@pytest.mark.parametrize(
    ("kernels", "calls", "result"),
    [
        ({"gate_up": "gu", "down": "dp"}, ["gate_up", "proj"], "reduced"),
        ({"down": "dp"}, ["gate_up_proj", "act_fn", "proj"], "reduced"),
        ({"gate_up": "gu"}, ["gate_up", "down_proj"], "production"),
        ({}, [], None),
    ],
)
def test_an_mlp_step_takes_the_kernels_its_width_has(
    monkeypatch, preshuffle, kernels, calls, result
):
    runner = _Runner(kernels)
    draft_mlp = _wire(monkeypatch, runner)
    seen = []

    def gate_up_proj(x, x_scale):
        seen.append("gate_up_proj")
        return "gu_out"

    def act_fn(gu):
        seen.append("act_fn")
        return "act"

    def down_proj(act):
        seen.append("down_proj")
        return "production"

    layer = SimpleNamespace(
        w_gu=torch.zeros(1), inter=1792, down=SimpleNamespace(k=1792)
    )
    mlp = SimpleNamespace(
        prefix="layers.0.mlp", mono_mlp=layer, _try_mono_mlp=True,
        gate_up_proj=gate_up_proj, act_fn=act_fn, down_proj=down_proj,
    )  # fmt: skip
    assert draft_mlp.forward_mlp(mlp, _norm_out()) == result
    order = [c[0] for c in runner.calls]
    assert sorted(seen + order) == sorted(calls)
    if "proj" in calls:
        assert runner.calls[-1][2] is layer.down


# -------------------------------------------------------- a width's kernels
class _Widths:
    def __init__(self, built):
        self.built, self.prepared = built, []

    def prepare(self, width):
        self.prepared.append(width)
        return width in self.built

    def __getitem__(self, width):
        return self.built[width]


def _real_runner(monkeypatch, built, capturing=False, gate_up_rows=0, ar_rows=48):
    from atom.models.kimi_k3.mono import draft_mlp

    monkeypatch.setenv("ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS", str(gate_up_rows))
    monkeypatch.setenv("ATOM_KIMI_MONO_DRAFT_AR_ROWS", str(ar_rows))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    runner = draft_mlp.DraftMlpRunner(torch.device("cpu"), 1, 0, None)
    runner.widths = _Widths(built)
    return runner


def test_a_width_is_built_once_and_keyed_by_kind(monkeypatch):
    kernel = SimpleNamespace(name="k3_draft_proj_ar")
    runner = _real_runner(monkeypatch, {7: kernel, 1007: kernel})
    assert runner.kernel("o_proj", 7) is kernel
    assert runner.kernel("down", 7) is kernel
    assert runner.kernel("o_proj", 7) is kernel
    assert runner.widths.prepared == [7, 1007]
    # a refused width stays refused without another vote
    assert runner.kernel("o_proj", 14) is None and runner.kernel("o_proj", 14) is None
    assert runner.widths.prepared == [7, 1007, 14]


def test_gate_up_takes_only_the_widths_it_is_given(monkeypatch):
    kernel = SimpleNamespace(name="k3_draft_gate_up")
    runner = _real_runner(monkeypatch, {2007: kernel, 2021: kernel}, gate_up_rows=16)
    assert runner.kernel("gate_up", 7) is kernel
    assert runner.kernel("gate_up", 21) is None
    assert runner.widths.prepared == [2007]  # past the switch: no vote at all
    off = _real_runner(monkeypatch, {2007: kernel})
    assert off.kernel("gate_up", 7) is None and off.widths.prepared == []


def test_the_projections_take_only_the_widths_they_are_given(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_DRAFT_AR_ROWS", raising=False)
    assert envs.ATOM_KIMI_MONO_DRAFT_AR_ROWS == 48
    kernel = SimpleNamespace(name="k3_draft_proj_ar")
    built = {48: kernel, 56: kernel, 1048: kernel, 1056: kernel}
    runner = _real_runner(monkeypatch, built)
    assert runner.kernel("o_proj", 48) is kernel and runner.kernel("down", 48) is kernel
    assert runner.kernel("o_proj", 56) is None and runner.kernel("down", 56) is None
    assert runner.widths.prepared == [48, 1048]  # past the switch: no vote
    wide = _real_runner(monkeypatch, built, ar_rows=64)
    assert wide.kernel("o_proj", 56) is kernel and wide.kernel("down", 56) is kernel


def test_a_width_first_seen_in_a_capture_waits_for_an_eager_step(monkeypatch):
    kernel = SimpleNamespace(name="k3_draft_proj_ar")
    runner = _real_runner(monkeypatch, {7: kernel}, capturing=True)
    assert runner.kernel("o_proj", 7) is None and runner.widths.prepared == []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert runner.kernel("o_proj", 7) is kernel


def test_one_rank_holds_its_banks_locally(monkeypatch):
    from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay

    runner = _real_runner(monkeypatch, {})
    peers = runner.banks()
    assert peers.buffer.numel() == lay.peer_bytes(1, 64)
    assert peers.peers.tolist() == [peers.buffer.data_ptr()]
    assert peers.epochs.numel() == lay.EPOCH_COUNTERS and not peers.epochs.any()
    assert runner.banks() is peers
