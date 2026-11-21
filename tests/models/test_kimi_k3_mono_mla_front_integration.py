# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contracts of the gfx1250 MLA front's integration: the switch,
``KimiFullAttention._gated_core``'s precomputed ``front`` (and its unchanged
default path), the layer binding's refusals and the runner's hooks."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.mono.runtime.consensus import MonoUnsupported

aiter = pytest.importorskip("aiter", reason="the K3 model and runner import aiter")
from aiter import QuantType, dtypes

GATE_ROWS = 1536


def test_the_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_FRONT", raising=False)
    assert envs.ATOM_KIMI_MONO_MLA_FRONT is False
    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_FRONT", "1")
    assert envs.ATOM_KIMI_MONO_MLA_FRONT is True


# --------------------------------------------------------------- the model
@pytest.fixture
def model_module():
    try:
        from atom.models.kimi_k3 import model
    except ImportError as error:  # the model module imports the GPU stack
        pytest.skip(f"atom.models.kimi_k3.model needs the GPU stack: {error}")
    return model


def _attention(model_module, monkeypatch, calls, indexer=None):
    """A KimiFullAttention's ``_gated_core`` on recording stand-ins."""

    def record(name, result):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return call

    rows = 2
    projected = torch.arange(rows * 2112, dtype=torch.float32).view(rows, 2112)
    norm_out = ((torch.ones(rows, 1536), None), None, torch.ones(rows, 512), None)
    monkeypatch.setattr(
        "atom.models.deepseek_v2._fuse_rmsnorm_quant", record("norm", norm_out)
    )
    monkeypatch.setattr(
        model_module,
        "fused_sigmoid_mul_maybe_quant",
        record("gating", ("gated", None)),
    )
    fake = SimpleNamespace(
        fused_qkv_a_proj=record("qkv_a", projected),
        projection_split_sizes=[1536, 512, 64],
        qknorm_dtype=torch.bfloat16,
        q_a_layernorm=SimpleNamespace(weight="q_w", eps=1e-6),
        kv_a_layernorm=SimpleNamespace(weight="kv_w", eps=1e-6),
        qknorm_quant_type_value=QuantType.No.value,
        indexer=indexer,
        attn=record("attn", "attn_out"),
        g_proj=record("g_proj", "gate"),
        fuse_sigmoid_mul_quant=False,
        o_proj_quant_type_value=QuantType.No.value,
    )
    return fake, projected


def test_without_a_front_the_core_runs_every_production_projection(
    model_module, monkeypatch
):
    calls = []
    fake, projected = _attention(model_module, monkeypatch, calls)
    out = model_module.KimiFullAttention._gated_core(fake, "pos", "hidden")
    assert out == ("gated", None)
    assert [name for name, *_ in calls] == ["qkv_a", "norm", "attn", "g_proj", "gating"]
    assert calls[0][1] == ("hidden", None)
    norm_args = calls[1][1]
    assert torch.equal(norm_args[0], projected[:, :1536])
    assert torch.equal(norm_args[3], projected[:, 1536:2048])
    _, attn_args, attn_kwargs = calls[2]
    assert torch.equal(attn_args[2], projected[:, 2048:]) and attn_args[3] == "pos"
    assert attn_kwargs == {"q_scale": None}
    assert calls[3][1] == ("hidden", None)
    assert calls[4][1] == ("attn_out", "gate")


def test_a_front_hands_the_attention_and_gating_its_tensors(model_module, monkeypatch):
    calls = []
    fake, _ = _attention(model_module, monkeypatch, calls)
    front = ("q8", "q_scale", "kv", "k_rope", "front_gate")
    out = model_module.KimiFullAttention._gated_core(
        fake, "pos", "hidden", quantize=False, front=front
    )
    assert out == ("gated", None)
    assert [name for name, *_ in calls] == ["attn", "gating"]
    assert calls[0][1] == ("q8", "kv", "k_rope", "pos")
    assert calls[0][2] == {"q_scale": "q_scale"}
    assert calls[1][1] == ("attn_out", "front_gate")
    assert calls[1][2]["quant"] is False


def test_a_dsa_layer_takes_no_front(model_module, monkeypatch):
    fake, _ = _attention(model_module, monkeypatch, [], indexer=object())
    with pytest.raises(ValueError, match="indexer"):
        model_module.KimiFullAttention._gated_core(
            fake, "pos", "hidden", front=(None,) * 5
        )


def test_forward_pre_o_passes_the_front_through(model_module):
    seen = {}

    def gated_core(positions, hidden, quantize=True, front=None):
        seen.update(quantize=quantize, front=front)
        return "core", None

    fake = SimpleNamespace(_gated_core=gated_core)
    attention = model_module.KimiFullAttention
    assert attention.forward_pre_o(fake, None, None, front="f") == "core"
    assert seen == {"quantize": False, "front": "f"}
    attention.forward_pre_o(fake, None, None)
    assert seen["front"] is None


# ------------------------------------------------------------- the binding
def _fp8_linear(rows, padded=None, shuffled=True, dtype=None):
    padded = padded or -(-rows // 128) * 128
    weight = nn.Parameter(
        torch.zeros(padded, 7168, dtype=dtype or torch.float8_e4m3fn),
        requires_grad=False,
    )
    weight.is_shuffled = shuffled
    scale = nn.Parameter(
        torch.zeros(padded // 128, 56, dtype=torch.float8_e8m0fnu),
        requires_grad=False,
    )
    return SimpleNamespace(weight=weight, weight_scale=scale, bias=None)


def _layer(**attn):
    base = {
        "indexer": None,
        "projection_split_sizes": [1536, 512, 64],
        "fuse_qknorm_quant": False,
        "num_heads": 96,
        "v_head_dim": 128,
        "fused_qkv_a_proj": _fp8_linear(2112),
        "g_proj": _fp8_linear(GATE_ROWS),
        "q_a_layernorm": SimpleNamespace(
            weight=torch.ones(1536, dtype=torch.bfloat16), eps=1e-6
        ),
        "kv_a_layernorm": SimpleNamespace(
            weight=torch.ones(512, dtype=torch.bfloat16), eps=1e-5
        ),
        "q_b_proj": SimpleNamespace(
            quant_type=QuantType.per_1x128,
            params_dtype=dtypes.fp8,
            blockscale_e8m0_scale=True,
            native_a8_group_rows=None,
        ),
    }
    base.update(attn)
    return SimpleNamespace(
        is_linear_attn=False,
        self_attention_attn_res=SimpleNamespace(out_quant_dtype=None),
        self_attn=SimpleNamespace(**base),
    )


def _bind(layer):
    from atom.models.kimi_k3.mono.config import Dims
    from atom.models.kimi_k3.mono.weights import bind_mla_front

    return bind_mla_front(layer, 3, Dims(8))


def test_a_full_attention_layer_binds_its_fp8_projections_in_place(monkeypatch):
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    layer = _layer()
    front = _bind(layer)
    attn = layer.self_attn
    assert front.w_a.data_ptr() == attn.fused_qkv_a_proj.weight.data_ptr()
    assert tuple(front.w_a.shape) == (2176, 7168) and tuple(front.s_a.shape) == (17, 56)
    assert tuple(front.w_g.shape) == (GATE_ROWS, 7168)
    assert front.gate_rows == GATE_ROWS and front.key == (GATE_ROWS, 1e-6, 1e-5)
    assert front.q_dtype == dtypes.fp8


def test_q_goes_bf16_where_q_b_proj_takes_another_activation_path(monkeypatch):
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "0")
    assert _bind(_layer()).q_dtype is None
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    fp32_scales = _layer().self_attn.q_b_proj
    fp32_scales.blockscale_e8m0_scale = False
    assert _bind(_layer(q_b_proj=fp32_scales)).q_dtype is None
    bf16 = SimpleNamespace(
        quant_type=QuantType.No, params_dtype=torch.bfloat16, native_a8_group_rows=None
    )
    assert _bind(_layer(q_b_proj=bf16)).q_dtype is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"indexer": object()}, "DSA"),
        ({"fuse_qknorm_quant": True}, "q norm folds"),
        ({"projection_split_sizes": [1536, 512, 64, 128, 32]}, "projection widths"),
        ({"fused_qkv_a_proj": _fp8_linear(2112, dtype=torch.bfloat16)}, "FP8"),
        ({"g_proj": _fp8_linear(GATE_ROWS, shuffled=False)}, "preshuffled"),
        ({"g_proj": _fp8_linear(GATE_ROWS, padded=1664)}, "expected"),
    ],
)
def test_a_layer_the_front_would_read_wrong_is_refused(change, message):
    with pytest.raises(MonoUnsupported, match=message):
        _bind(_layer(**change))


def test_a_layer_whose_input_norm_folds_a_quant_is_refused():
    layer = _layer()
    layer.self_attention_attn_res.out_quant_dtype = dtypes.fp8
    with pytest.raises(MonoUnsupported, match="input norm"):
        _bind(layer)


def test_a_projection_with_a_bias_is_refused():
    linear = _fp8_linear(GATE_ROWS)
    linear.bias = torch.zeros(GATE_ROWS)
    with pytest.raises(MonoUnsupported, match="bias"):
        _bind(_layer(g_proj=linear))


# -------------------------------------------------------------- the runner
@pytest.fixture
def runner_module():
    pytest.importorskip("flydsl", reason="the gfx1250 runner imports FlyDSL")
    from atom.models.kimi_k3.mono import runner_gfx1250

    return runner_gfx1250


def _front(layer_id=3, q_dtype=dtypes.fp8):
    from atom.models.kimi_k3.mono.weights import MlaFrontLayer

    t = torch.zeros(1, dtype=torch.bfloat16)
    return MlaFrontLayer(
        layer_id=layer_id,
        w_a=t,
        s_a=t,
        w_g=torch.zeros(GATE_ROWS, 1),
        s_g=t,
        q_norm=t,
        kv_norm=t,
        q_eps=1e-6,
        kv_eps=1e-6,
        q_dtype=q_dtype,
    )


def _runner(runner_module, **fields):
    runner = object.__new__(runner_module.K3Gfx1250MonoDecodeRunner)
    runner.device = torch.device("cpu")
    runner.max_rows = 64
    for name, value in fields.items():
        setattr(runner, name, value)
    return runner


def test_switched_off_no_layer_binds_a_front(runner_module, monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_FRONT", raising=False)
    monkeypatch.setattr(
        runner_module, "bind_mla_front", lambda *a: pytest.fail("bound with it off")
    )
    runner = _runner(runner_module, ffn_layers={3, 7})
    runner._bind_mla_fronts()
    assert runner.mla_fronts == {} and not hasattr(runner, "mla_qkv")


def test_switched_on_a_refused_layer_keeps_production(runner_module, monkeypatch):
    from atom.models.kimi_k3.mono.kernels.stages import mla_front_layout as lay

    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_FRONT", "1")

    def bind(layer, layer_id, dims):
        if layer_id == 7:
            raise MonoUnsupported("no")
        return _front(layer_id)

    monkeypatch.setattr(runner_module, "bind_mla_front", bind)
    runner = _runner(
        runner_module,
        ffn_layers={3, 7},
        inner=SimpleNamespace(layers={3: None, 7: None}),
        dims=None,
    )
    runner._bind_mla_fronts()
    assert sorted(runner.mla_fronts) == [3]
    # the hand-off words all start unwritten; one scratch for every width
    assert runner.mla_qkv.numel() == lay.qkv_words(64)
    assert bool((runner.mla_qkv == -1).all())
    assert runner.mla_partials.numel() == lay.partial_bytes(64, GATE_ROWS)


def test_a_front_that_does_not_build_clean_leaves_the_width_production(
    runner_module, monkeypatch
):
    built = SimpleNamespace(launcher="launcher", abi="abi")
    seen = []

    def build(rows, gate_rows, q_eps, kv_eps):
        seen.append((rows, gate_rows, q_eps, kv_eps))
        if rows == 65:
            raise ValueError("too wide")
        return built

    monkeypatch.setattr(runner_module.mla_front_gfx1250, "build_mla_front", build)
    monkeypatch.setattr(runner_module, "compile_kernels", lambda k: [k[0][0]])
    hazards = {"value": ()}
    monkeypatch.setattr(runner_module, "_resident_hazards", lambda _l: hazards["value"])
    runner = _runner(runner_module, mla_fronts={3: _front(3), 7: _front(7)})
    assert runner._mla_fronts_for(8) == {(GATE_ROWS, 1e-6, 1e-6): built}
    assert seen == [(8, GATE_ROWS, 1e-6, 1e-6)]  # one build for both layers
    hazards["value"] = ("2 vgpr spill",)
    assert runner._mla_fronts_for(8) == {(GATE_ROWS, 1e-6, 1e-6): None}
    hazards["value"] = ()
    assert runner._mla_fronts_for(65) == {(GATE_ROWS, 1e-6, 1e-6): None}
    assert _runner(runner_module, mla_fronts={})._mla_fronts_for(8) is None


def test_the_width_lists_its_fronts_for_the_resident_checks(runner_module):
    front = SimpleNamespace(launcher=object(), abi="abi")
    build = runner_module.Gfx1250WidthBuild({}, {}, None, {(1,): front, (2,): None})
    assert build.kernels() == [(front.launcher, "abi")]
    assert runner_module.Gfx1250WidthBuild({}, {}).kernels() == []


def test_only_a_layer_with_a_built_front_projects_through_it(runner_module):
    kernel = object()
    front = _front(3)
    runner = _runner(runner_module, mla_fronts={3: front})
    runner._mla_front = lambda k, f, x: (k, f, x)
    build = SimpleNamespace(mla_front={front.key: kernel})
    project = runner._mla_projection(build, 3)
    assert project("normed") == (kernel, front, "normed")
    assert runner._mla_projection(build, 7) is None
    assert (
        runner._mla_projection(SimpleNamespace(mla_front={front.key: None}), 3) is None
    )
    assert runner._mla_projection(SimpleNamespace(), 3) is None
    assert _runner(runner_module)._mla_projection(build, 3) is None


def test_the_launch_reads_the_layer_and_runner_buffers(runner_module, monkeypatch):
    pytest.importorskip("flydsl", reason="builds a FlyDSL kernel")
    from atom.models.kimi_k3.mono.kernels.stages import mla_front_gfx1250 as mf

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    calls = []
    kernel = SimpleNamespace(launcher=lambda *a, **k: calls.append((a, k)))
    runner = _runner(
        runner_module,
        mla_qkv=torch.zeros(4, dtype=torch.int32),
        mla_partials=torch.zeros(4, dtype=torch.uint8),
    )
    front = _front(3)
    x = torch.zeros(5, 7168, dtype=torch.bfloat16)
    q8, q_scale, kv, k_rope, gate = runner._mla_front(kernel, front, x)
    assert (q8.dtype, tuple(q8.shape)) == (dtypes.fp8, (5, 1536))
    assert (q_scale.dtype, tuple(q_scale.shape)) == (dtypes.fp8_e8m0, (5, 12))
    assert tuple(kv.shape) == (5, 512) and tuple(k_rope.shape) == (5, 64)
    assert tuple(gate.shape) == (5, GATE_ROWS)
    ((args, kwargs),) = calls
    assert kwargs == {"stream": "stream"}
    named = dict(zip(mf.MLA_FRONT_ABI.names, args))
    assert named["x"] == x.data_ptr() and named["q"] == 0
    assert named["q_fp8"] == q8.data_ptr() and named["q_scale"] == q_scale.data_ptr()
    assert named["qkv_c"] == runner.mla_qkv.data_ptr()
    assert named["partials"] == runner.mla_partials.data_ptr()
    assert named["qkv_seen"] == 0
    q, none, *_ = runner._mla_front(kernel, _front(3, q_dtype=None), x)
    assert q.dtype == torch.bfloat16 and none is None
    assert dict(zip(mf.MLA_FRONT_ABI.names, calls[-1][0]))["q_fp8"] == 0


@pytest.mark.parametrize(
    "x",
    [
        torch.zeros(5, 7168, dtype=torch.float16),
        torch.zeros(7168, 5, dtype=torch.bfloat16).t(),
        (torch.zeros(5, 7168), torch.zeros(5, 1)),
    ],
)
def test_the_launch_refuses_a_row_it_cannot_read(runner_module, x):
    runner = _runner(runner_module)
    with pytest.raises(MonoUnsupported, match="MLA front"):
        runner._mla_front(None, _front(3), x)


def test_a_built_front_reaches_the_attention_through_pre_o_core(runner_module):
    calls = {}
    layer = SimpleNamespace(self_attn=SimpleNamespace(o_proj=None))
    runner = _runner(
        runner_module,
        inner=SimpleNamespace(layers={7: layer}),
        routed={7: object()},
        ffn_layers={7},
        oproj_weights={},
        mla_fronts={7: _front(7)},
    )
    runner._mla_front = lambda kernel, front, x: ("front of", x)

    def pre_o_core(plan, positions, prefix, blocks, add, add2, project=None):
        calls["project"] = project
        return torch.zeros(2, 4), prefix

    runner._pre_o_core = pre_o_core
    runner._production_k2 = lambda plan, attn, prefix, blocks: (prefix, attn, None)
    runner._write_aux = lambda *_a: None
    prefix = torch.zeros(2, 4)
    build = SimpleNamespace(oproj=None, k2={}, mla_front={_front(7).key: "kernel"})
    layer.self_attn.o_proj = lambda core: core
    runner._layer(
        build, SimpleNamespace(layer_id=7), None, prefix, torch.zeros(2, 1, 4),
        prefix, None, None,
    )  # fmt: skip
    assert calls["project"]("normed") == ("front of", "normed")


def test_pre_o_core_hands_the_projection_of_the_normed_row(monkeypatch):
    pytest.importorskip("flydsl", reason="the K3 mono runner imports FlyDSL")
    pytest.importorskip("triton", reason="the kernel under test needs triton")
    from atom.models.kimi_k3.mono import runner as base
    from atom.models.kimi_k3.mono.config import Dims

    seen = {}
    core = torch.zeros(2, Dims(8).proj, dtype=torch.bfloat16)

    def forward_pre_o(positions, normed, front=None):
        seen.update(positions=positions, normed=normed, front=front)
        return core

    layer = SimpleNamespace(self_attn=SimpleNamespace(forward_pre_o=forward_pre_o))
    runner = object.__new__(base.K3MonoDecodeRunner)
    runner.dims = Dims(8)
    prefix = torch.zeros(2, 4)
    runner._seam = lambda *a: (layer, "normed", prefix, 1, -1)
    got, got_prefix = runner._pre_o_core(
        SimpleNamespace(layer_id=3), "pos", prefix, None, None, None,
        lambda normed: ("projected", normed),
    )  # fmt: skip
    assert got is core and got_prefix is prefix
    assert seen == {"positions": "pos", "normed": "normed",
                    "front": ("projected", "normed")}  # fmt: skip
    runner._pre_o_core(SimpleNamespace(layer_id=3), "pos", prefix, None, None, None)
    assert seen["front"] is None
