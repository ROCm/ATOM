# SPDX-License-Identifier: MIT
"""The V4 shared expert's gate_up GEMM with the SwiGLU and the MXFP8 quant
for w2 in its epilogue.

Whether an expert may take it (`Expert._gate_up_silu`) is decided once at
construction, from the configuration and the gate_up weight shape. Forward
then branches on nothing that depends on the row count: it calls the
functional op `torch.ops.aiter.v4_shared_gate_up_silu_quant`, whose eager body
(`Expert.gate_up_silu_quant`) picks the fused kernel or the unfused chain from
the real row count. The tests check what the body hands AITER and w2, that
everything else keeps the unfused path, and that a fullgraph compile of
`Expert.forward` traced once still makes the right choice for every later row
count."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# Imported before any test swaps in the AITER / Triton stand-ins: torch.compile
# loads it lazily, and it inspects Triton types at import.
import torch.utils.flop_counter

INTER, DIM = 64, 256


class _W2:
    """A traceable stand-in for w2 that records nothing and returns a value
    depending on both of its inputs."""

    native_a8_group_rows = 32

    def __call__(self, act, x_scale=None):
        codes = act.view(torch.uint8).to(torch.float32).sum(-1, keepdim=True)
        scales = x_scale.view(torch.uint8).to(torch.float32).sum(-1, keepdim=True)
        return codes + scales


class _GateUp(SimpleNamespace):
    """gate_up_proj: the weight attributes, called through `fn` (set by a
    test that runs the unfused chain)."""

    def __call__(self, x):
        return self.fn(x)


@pytest.fixture
def atom_config(atom_modules):
    config = importlib.import_module("atom.config")
    custom = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={})
    )
    with config.use_custom_atom_config(custom):
        yield custom


def _expert(
    monkeypatch,
    atom_modules,
    *,
    limit=10.0,
    stored_preshuffled=True,
    group=32,
    fused_act=False,
    w2_group=32,
    prefix="layers.0.ffn.shared_experts",
):
    v4 = atom_modules.deepseek_v4
    expert = v4.Expert.__new__(v4.Expert)
    torch.nn.Module.__init__(expert)
    weight = torch.zeros(2 * INTER, DIM, dtype=torch.float8_e4m3fn)
    expert.gate_up_proj = _GateUp(
        weight=weight,
        weight_scale=torch.zeros(4, 8, dtype=torch.float8_e8m0fnu),
        native_a8_group_rows=group,
        quant_type="per_1x32",
    )
    expert.w2 = _W2() if w2_group == 32 else MagicMock(native_a8_group_rows=None)
    expert.swiglu_limit = limit
    expert.use_fused_clamp_act_mul = fused_act
    expert.prefix = prefix
    monkeypatch.setattr(
        v4, "weight_is_stored_preshuffled", lambda *a, **k: stored_preshuffled
    )
    expert._gate_up_silu = expert._gate_up_silu_applies()
    if expert._gate_up_silu:  # as Expert.__init__ registers it
        config = importlib.import_module("atom.config").get_current_atom_config()
        config.compilation_config.static_forward_context[prefix] = expert
    return expert


@pytest.fixture
def flydsl(monkeypatch, atom_modules):
    """AITER as the expert sees it: tiles up to 4096 rows; the functional op
    records its arguments and returns filled outputs."""
    calls = {"supported": True, "runs": [], "shapes": []}
    module = ModuleType("aiter.ops.flydsl.batched_gemm_a8w8_gfx1250")
    op_module = ModuleType("aiter.ops.batched_gemm_op_a8w8")

    def supported(n2, k):
        calls["supported_shape"] = (n2, k)
        return calls["supported"]

    def kernel_for(m, n2, k):
        calls["shapes"].append((m, n2, k))
        return None if m > 4096 else f"tile{min(m, 4096)}"

    def gate_up_silu_mxfp8_32(xq, wq, xs, ws, limit):
        m, inter = xq.shape[0], wq.shape[0] // 2
        act = torch.full((m, inter), 1, dtype=torch.uint8).view(torch.float8_e4m3fn)
        act_scale = torch.full((m, inter // 32), 2, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        calls["runs"].append((xq, wq, xs, ws, limit, act, act_scale))
        return act, act_scale

    module.gate_up_silu_supported = supported
    module.gate_up_silu_kernel_for = kernel_for
    op_module.gate_up_silu_mxfp8_32 = gate_up_silu_mxfp8_32
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, op_module.__name__, op_module)
    v4 = atom_modules.deepseek_v4
    monkeypatch.setattr(v4, "get_gfx_runtime", lambda: "gfx1250")
    blockscale = importlib.import_module("atom.model_ops.blockscale")

    def quantize_fp8(x):
        calls.setdefault("quantized", []).append(x)
        out = (
            torch.full(x.shape, 3, dtype=torch.uint8).view(torch.float8_e4m3fn),
            torch.full((*x.shape[:-1], x.shape[-1] // 32), 4, dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            ),
        )
        calls.setdefault("quantized_out", []).append(out)
        return out

    monkeypatch.setattr(blockscale, "quantize_fp8", quantize_fp8)
    return calls


@pytest.fixture
def unfused(monkeypatch, atom_modules):
    """The unfused chain's gate_up and silu_and_mul, recorded."""
    calls = {"gate_up": [], "silu": []}

    def gate_up(x):
        calls["gate_up"].append(x.shape[0])
        return torch.zeros(x.shape[0], 2 * INTER, dtype=torch.bfloat16)

    def silu_and_mul(out, x, limit):
        calls["silu"].append((x.shape[0], limit))

    monkeypatch.setattr(atom_modules.deepseek_v4, "aiter_silu_and_mul", silu_and_mul)
    calls["gate_up_fn"] = gate_up
    return calls


@pytest.fixture
def cpu_op(atom_modules):
    """A CPU kernel for the op (production registers CUDA only), dispatching
    to this window's module so it reads this window's forward context."""
    ops = importlib.import_module("atom.model_ops.module_dispatch_ops")
    lib = torch.library.Library("aiter", "IMPL")
    lib.impl(
        "v4_shared_gate_up_silu_quant",
        lambda x, w, name: ops.v4_shared_gate_up_silu_quant(x, w, name),
        "CPU",
    )
    yield
    lib._destroy()


def test_fused_rows_feed_w2_the_epilogue_quant(
    monkeypatch, atom_modules, flydsl, atom_config
):
    expert = _expert(monkeypatch, atom_modules)
    assert expert._gate_up_silu and flydsl["supported_shape"] == (2 * INTER, DIM)
    x = torch.zeros(3, DIM, dtype=torch.bfloat16)
    act, act_scale = expert.gate_up_silu_quant(x)
    xq, wq, xs, ws, limit, out, out_scale = flydsl["runs"][-1]
    # AITER gets quantize_fp8(x): the e4m3 rows and their E8M0 scales.
    assert flydsl["quantized"] == [x]
    x_fp8, x_scale = flydsl["quantized_out"][-1]
    assert xq is x_fp8 and xs is x_scale
    assert xq.dtype == torch.float8_e4m3fn and xq.shape == (3, DIM)
    assert xs.dtype == torch.float8_e8m0fnu and xs.shape == (3, DIM // 32)
    assert wq is expert.gate_up_proj.weight
    assert ws is expert.gate_up_proj.weight_scale
    assert act is out and act_scale is out_scale
    assert act.shape == (3, INTER) and act.dtype == torch.float8_e4m3fn
    assert act_scale.shape == (3, INTER // 32)
    assert act_scale.dtype == torch.float8_e8m0fnu
    assert limit == 10.0
    assert flydsl["shapes"][-1] == (3, 2 * INTER, DIM)


def test_other_rows_run_the_unfused_chain(
    monkeypatch, atom_modules, flydsl, unfused, atom_config
):
    expert = _expert(monkeypatch, atom_modules)
    expert.gate_up_proj.fn = unfused["gate_up_fn"]
    x = torch.zeros(5000, DIM, dtype=torch.bfloat16)
    act, act_scale = expert.gate_up_silu_quant(x)
    assert not flydsl["runs"]
    assert unfused["gate_up"] == [5000] and unfused["silu"] == [(5000, 10.0)]
    # quantize_fp8 of the bf16 activation: what w2 would compute itself.
    (activation,) = flydsl["quantized"]
    assert activation.shape == (5000, INTER) and activation.dtype == torch.bfloat16
    assert act.shape == (5000, INTER) and act_scale.shape == (5000, INTER // 32)


def test_traced_forward_picks_per_runtime_rows(
    monkeypatch, atom_modules, flydsl, unfused, atom_config, cpu_op
):
    """`Expert.forward` compiled once with fullgraph and dynamic shapes, then
    replayed: the first (warmup-sized) row count runs the unfused chain and
    later ones still get their own kernel, all through one graph."""
    expert = _expert(monkeypatch, atom_modules)
    expert.gate_up_proj.fn = unfused["gate_up_fn"]
    assert atom_config.compilation_config.static_forward_context[expert.prefix] is (
        expert
    )
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        expert.forward, fullgraph=True, dynamic=True, backend=backend
    )
    for m in (5000, 3, 1345, 4096):
        y = compiled(torch.zeros(m, DIM, dtype=torch.bfloat16))
        assert y.shape == (m, 1)
        if m > 4096:
            assert unfused["gate_up"][-1] == m and flydsl["runs"] == []
            # The quantize_fp8 stand-in: codes 3, scale bytes 4.
            assert torch.all(y == 3 * INTER + 4 * (INTER // 32))
        else:
            assert flydsl["runs"][-1][0].shape[0] == m
            assert flydsl["shapes"][-1] == (m, 2 * INTER, DIM)
            # The launch stand-in: codes 1, scale bytes 2.
            assert torch.all(y == INTER + 2 * (INTER // 32))
    assert len(graphs) == 1
    torch._dynamo.reset()


def test_gated_forward_keeps_the_unfused_path(
    monkeypatch, atom_modules, flydsl, atom_config
):
    """`forward(x, weights=...)` (a gated expert) never takes the fused path:
    the epilogue has no per-row gate."""
    expert = _expert(monkeypatch, atom_modules)
    assert expert._gate_up_silu
    expert.gate_up_proj = MagicMock(return_value=torch.zeros(3, 2 * INTER))
    monkeypatch.setattr(
        atom_modules.deepseek_v4, "aiter_silu_and_mul", lambda out, x, limit: None
    )
    expert.w2 = MagicMock(return_value="out", native_a8_group_rows=32)
    x = torch.zeros(3, DIM, dtype=torch.bfloat16)
    weights = torch.ones(3, 1, dtype=torch.bfloat16)
    assert expert.forward(x, weights=weights) == "out"
    assert not flydsl["runs"] and not flydsl["shapes"]
    expert.gate_up_proj.assert_called_once_with(x)


@pytest.mark.parametrize(
    "change",
    [
        {"limit": 0.0},
        {"stored_preshuffled": False},
        {"group": None},
        {"fused_act": True},
        {"w2_group": None},
        {"unsupported_shape": True},
    ],
)
def test_ineligible_experts_keep_the_unfused_path(
    monkeypatch, atom_modules, flydsl, atom_config, change
):
    if change.pop("unsupported_shape", False):
        flydsl["supported"] = False  # e.g. a sharded TP4 shared expert
    expert = _expert(monkeypatch, atom_modules, **change)
    assert not expert._gate_up_silu
    assert expert.prefix not in atom_config.compilation_config.static_forward_context


def test_other_inputs_keep_the_unfused_path(
    monkeypatch, atom_modules, flydsl, atom_config
):
    v4 = atom_modules.deepseek_v4
    expert = _expert(monkeypatch, atom_modules)
    assert expert._gate_up_silu
    expert.gate_up_proj = MagicMock(return_value=torch.zeros(3, 2 * INTER))
    monkeypatch.setattr(v4, "aiter_silu_and_mul", lambda out, x, limit: None)
    expert.w2 = MagicMock(return_value="out", native_a8_group_rows=32)
    assert expert.forward(torch.zeros(3, DIM)) == "out"  # fp32 rows
    assert not flydsl["runs"] and not flydsl["shapes"]
    monkeypatch.setattr(v4, "get_gfx_runtime", lambda: "gfx950")
    assert not _expert(monkeypatch, atom_modules)._gate_up_silu


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
def test_init_decides_and_registers(
    monkeypatch, atom_modules, flydsl, atom_config, arch
):
    """`Expert.__init__` queries the architecture and AITER's shape support
    once and registers a fused expert for the op; forward queries neither
    (the op body, not forward, asks the kernel table about the row count)."""
    v4 = atom_modules.deepseek_v4
    queries = []

    def gfx():
        queries.append(arch)
        return arch

    def linear(*args, **kwargs):
        return SimpleNamespace(
            weight=torch.zeros(2 * INTER, DIM, dtype=torch.float8_e4m3fn),
            weight_scale=torch.zeros(4, 8, dtype=torch.float8_e8m0fnu),
            native_a8_group_rows=32,
            quant_type="per_1x32",
            input_size=INTER,
        )

    monkeypatch.setattr(v4, "get_gfx_runtime", gfx)
    monkeypatch.setattr(v4, "MergedColumnParallelLinear", linear)
    monkeypatch.setattr(v4, "RowParallelLinear", linear)
    monkeypatch.setattr(v4, "shared_expert_fused_act_enabled", lambda width: False)
    monkeypatch.setattr(v4, "weight_is_stored_preshuffled", lambda *a, **k: True)
    expert = v4.Expert(DIM, INTER, swiglu_limit=10.0, prefix="m.shared_experts")
    assert queries == [arch]
    fused = arch == "gfx1250"
    assert expert._gate_up_silu == fused
    registry = atom_config.compilation_config.static_forward_context
    assert (registry.get("m.shared_experts") is expert) == fused
    assert ("supported_shape" in flydsl) == fused


@pytest.mark.parametrize("m", [0, 3, 5000])
def test_op_fake_shapes_and_dtypes(atom_modules, m):
    """The op's fake (what torch.compile traces with) returns w2's quantized
    input: e4m3 [m, inter] and E8M0 [m, inter / 32], on x's device; directly
    and through the registered op under FakeTensorMode."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    ops = importlib.import_module("atom.model_ops.module_dispatch_ops")
    x = torch.zeros(m, DIM, dtype=torch.bfloat16)
    weight = torch.zeros(2 * INTER, DIM, dtype=torch.float8_e4m3fn)

    def check(act, act_scale):
        assert act.shape == (m, INTER) and act.dtype == torch.float8_e4m3fn
        assert act_scale.shape == (m, INTER // 32)
        assert act_scale.dtype == torch.float8_e8m0fnu
        assert act.device == x.device and act_scale.device == x.device

    check(*ops._v4_shared_gate_up_silu_quant_fake(x, weight, "unused"))
    with FakeTensorMode() as mode:
        fx = mode.from_tensor(x)
        fw = mode.from_tensor(weight)
        check(*torch.ops.aiter.v4_shared_gate_up_silu_quant(fx, fw, "unused"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
