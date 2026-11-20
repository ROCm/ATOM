# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contracts of the gfx1250 MLA absorb's integration: the switch,
``KimiFullAttention._gated_core``'s q_out (and the attention's decode from
it), the layer binding's refusals and the runner's per-step gating."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.mono.runtime.consensus import MonoUnsupported

aiter = pytest.importorskip("aiter", reason="the K3 model and runner import aiter")
from aiter import QuantType, dtypes


def test_the_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_ABSORB", raising=False)
    assert envs.ATOM_KIMI_MONO_MLA_ABSORB is False
    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_ABSORB", "1")
    assert envs.ATOM_KIMI_MONO_MLA_ABSORB is True


# --------------------------------------------------------------- the model
@pytest.fixture
def model_module():
    try:
        from atom.models.kimi_k3 import model
    except ImportError as error:  # the model module imports the GPU stack
        pytest.skip(f"atom.models.kimi_k3.model needs the GPU stack: {error}")
    return model


def test_a_q_out_takes_the_attentions_decode_from_it(model_module, monkeypatch):
    calls = []

    def record(name, result):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return call

    monkeypatch.setattr(
        model_module, "fused_sigmoid_mul_maybe_quant", record("gating", ("gated", None))
    )
    attn = record("attn", "attn_out")
    attn.impl = SimpleNamespace(forward_decode_q_out=record("decode", "decoded"))
    fake = SimpleNamespace(
        indexer=None,
        attn=attn,
        fuse_sigmoid_mul_quant=False,
        o_proj_quant_type_value=QuantType.No.value,
    )
    front = ("q8", "q_scale", "kv", "k_rope", "gate", "q_out")
    out = model_module.KimiFullAttention._gated_core(
        fake, "pos", "hidden", quantize=False, front=front
    )
    assert out == ("gated", None)
    assert [name for name, *_ in calls] == ["decode", "gating"]
    assert calls[0][1] == ("q_out",)
    assert calls[1][1] == ("decoded", "gate")
    calls.clear()
    model_module.KimiFullAttention._gated_core(fake, "pos", "hidden", front=front[:5])
    assert [name for name, *_ in calls] == ["attn", "gating"]


def _attention_module():
    try:
        from atom.model_ops import attention_mla
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")
    return attention_mla


@pytest.mark.parametrize("dcp", [1, 8])
def test_the_decode_from_q_out_is_forward_impls_tail(monkeypatch, dcp):
    mla = _attention_module()
    calls = []
    metadata, cache = object(), object()
    context = SimpleNamespace(
        attn_metadata=metadata,
        kv_cache_data={"layer_5": SimpleNamespace(k_cache=cache)},
    )
    monkeypatch.setattr(mla, "get_forward_context", lambda: context)
    fake = SimpleNamespace(
        layer_num=5,
        dcp_world_size=dcp,
        qrep_enabled=True,
        use_seg_mla=False,
        _fused_q_head_pad=True,
        _dcp_decode=lambda *a: calls.append(("dcp", a)) or "dcp_out",
        _forward_decode=lambda *a, **k: calls.append(("local", a, k)) or "out",
    )
    got = mla.MLAAttention.forward_decode_q_out(fake, "q_out")
    if dcp > 1:
        assert got == "dcp_out" and calls == [("dcp", ("q_out", cache, metadata, True))]
    else:
        assert got == "out"
        assert calls == [("local", ("q_out", cache, metadata), {"q_prepadded": True})]


# ------------------------------------------------------------- the binding
def _q_b_proj(heads, dtype=torch.float8_e4m3fn, shuffled=True):
    weight = nn.Parameter(
        torch.zeros(heads * 192, 1536, dtype=dtype), requires_grad=False
    )
    weight.is_shuffled = shuffled
    scale = nn.Parameter(
        torch.zeros(heads * 192 // 128, 12, dtype=torch.float8_e8m0fnu),
        requires_grad=False,
    )
    return SimpleNamespace(weight=weight, weight_scale=scale, bias=None)


def _rope(**kw):
    from atom.model_ops.rotary_embedding import NoPositionalRotaryEmbedding

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    rope = NoPositionalRotaryEmbedding(64, 64, 16, 10000.0)
    torch.set_default_dtype(default)
    for name, value in kw.items():
        setattr(rope, name, value)
    return rope


def _impl(heads=2, dcp=8, qrep=True, **change):
    qrep_heads = heads * dcp
    impl = {
        "forward_decode_q_out": lambda q_out: q_out,
        "is_sparse_mla": False,
        "use_seg_mla": False,
        "kv_shuffled": False,
        "kv_cache_dtype": "fp8",
        "_k_scale": torch.tensor(1.0),
        "_q_scale": torch.tensor(1.0),
        "rotary_emb": _rope(),
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "qrep_enabled": qrep,
        "qrep_num_heads": qrep_heads,
        "W_K_qrep": torch.zeros(qrep_heads, 512, 128, dtype=torch.float8_e4m3fn),
        "W_K_qrep_scale": torch.ones((), dtype=torch.float32),
        "num_heads": heads,
        "W_K": torch.zeros(heads, 512, 128, dtype=torch.float8_e4m3fn),
        "W_K_scale": torch.ones((), dtype=torch.float32),
        "dcp_world_size": dcp,
        "padded_num_heads": 16,
        "_fused_q_head_pad": dcp == 1,
        "layer_num": 3,
    }
    impl.update(change)
    return SimpleNamespace(**impl)


def _layer(impl, q_b_heads):
    return SimpleNamespace(
        self_attn=SimpleNamespace(
            attn=SimpleNamespace(impl=impl), q_b_proj=_q_b_proj(q_b_heads)
        )
    )


def _front(q_dtype=dtypes.fp8):
    from atom.models.kimi_k3.mono.weights import MlaFrontLayer

    t = torch.zeros(1)
    return MlaFrontLayer(3, t, t, torch.zeros(1536, 1), t, t, t, 1e-6, 1e-6, q_dtype)


def _bind(impl, q_b_heads, front=None):
    from atom.models.kimi_k3.mono.weights import bind_mla_absorb

    return bind_mla_absorb(_layer(impl, q_b_heads), 3, front or _front())


def test_qrep_binds_the_dcp_groups_heads_in_place():
    impl = _impl()
    layer = _layer(impl, 16)
    from atom.models.kimi_k3.mono.weights import bind_mla_absorb

    absorb = bind_mla_absorb(layer, 3, _front())
    assert absorb.w_uk is impl.W_K_qrep and absorb.s_uk is impl.W_K_qrep_scale
    assert absorb.w_qb.data_ptr() == layer.self_attn.q_b_proj.weight.data_ptr()
    assert absorb.key == (16, 16, True) and absorb.attention is impl


def test_without_dcp_the_local_heads_go_padded_and_slot_gated():
    absorb = _bind(_impl(heads=12, dcp=1, qrep=False), 12)
    assert absorb.key == (12, 16, False)
    unpadded = _bind(_impl(heads=12, dcp=1, qrep=False, _fused_q_head_pad=False), 12)
    assert unpadded.key == (12, 12, False)
    no_qrep = _bind(_impl(heads=12, dcp=8, qrep=False), 12)
    assert no_qrep.key == (12, 12, True)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"is_sparse_mla": True}, "sparse"),
        ({"use_seg_mla": True}, "segmented"),
        ({"kv_shuffled": True}, "shuffled"),
        ({"kv_cache_dtype": "auto"}, "not FP8"),
        ({"_k_scale": torch.tensor(0.5)}, "scales"),
        ({"rotary_emb": SimpleNamespace(rotary_dim=64, is_neox_style=True)}, "NoPE"),
        ({"rotary_emb": _rope(is_neox_style=False)}, "NoPE"),
        ({"kv_lora_rank": 256}, "widths"),
        ({"W_K_qrep_scale": None}, "BF16 absorb"),
        ({"W_K_qrep": torch.zeros(16, 512, 128, dtype=torch.bfloat16)}, "W_UK"),
        ({"forward_decode_q_out": None}, "MLAAttention"),
    ],
)
def test_a_layer_the_absorb_would_read_wrong_is_refused(change, message):
    with pytest.raises(MonoUnsupported, match=message):
        _bind(_impl(**change), 16)


def test_a_q_b_proj_it_cannot_read_or_a_bf16_front_is_refused():
    with pytest.raises(MonoUnsupported, match="expected"):
        _bind(_impl(), 12)
    with pytest.raises(MonoUnsupported, match="BF16 q"):
        _bind(_impl(), 16, _front(q_dtype=None))


def test_the_fp4_absorb_is_refused(monkeypatch):
    mla = _attention_module()
    monkeypatch.setattr(mla, "is_rocm_aiter_fp4bmm_enabled", lambda: True)
    with pytest.raises(MonoUnsupported, match="FP4"):
        _bind(_impl(), 16)


# -------------------------------------------------------------- the runner
@pytest.fixture
def runner_module():
    pytest.importorskip("flydsl", reason="the gfx1250 runner imports FlyDSL")
    from atom.models.kimi_k3.mono import runner_gfx1250

    return runner_gfx1250


def _absorb(layer_id=3, key=(16, 16, True)):
    from atom.models.kimi_k3.mono.weights import MlaAbsorbLayer

    t = torch.zeros(1)
    return MlaAbsorbLayer(
        layer_id, t, t, t, t, *key, attention=SimpleNamespace(layer_num=layer_id)
    )


def _runner(runner_module, **fields):
    runner = object.__new__(runner_module.K3Gfx1250MonoDecodeRunner)
    runner.device = torch.device("cpu")
    runner.max_rows = 64
    for name, value in fields.items():
        setattr(runner, name, value)
    return runner


def test_switched_off_no_layer_binds_an_absorb(runner_module, monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_ABSORB", raising=False)
    monkeypatch.setattr(
        runner_module, "bind_mla_absorb", lambda *a: pytest.fail("bound with it off")
    )
    runner = _runner(runner_module, mla_fronts={3: _front()})
    runner._bind_mla_absorbs()
    assert runner.mla_absorbs == {} and not hasattr(runner, "mla_q")


def test_switched_on_a_refused_layer_keeps_production(runner_module, monkeypatch):
    from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_layout as lay

    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_ABSORB", "1")

    def bind(layer, layer_id, front):
        if layer_id == 7:
            raise MonoUnsupported("no")
        return _absorb(layer_id, (96, 96, True))

    monkeypatch.setattr(runner_module, "bind_mla_absorb", bind)
    runner = _runner(
        runner_module,
        mla_fronts={3: _front(), 7: _front()},
        inner=SimpleNamespace(layers={3: None, 7: None}),
    )
    runner._bind_mla_absorbs()
    assert sorted(runner.mla_absorbs) == [3]
    assert runner.mla_q.numel() == lay.q_words(64, 96)


def test_an_absorb_that_does_not_build_clean_leaves_the_width_production(
    runner_module, monkeypatch
):
    built = SimpleNamespace(launcher="launcher", abi="abi")
    seen = []

    def build(rows, heads, out_heads, compute_all):
        seen.append((rows, heads, out_heads, compute_all))
        if rows == 65:
            raise ValueError("too wide")
        return built

    monkeypatch.setattr(runner_module.mla_absorb_gfx1250, "build_mla_absorb", build)
    monkeypatch.setattr(runner_module, "compile_kernels", lambda k: [k[0][0]])
    hazards = {"value": ()}
    monkeypatch.setattr(runner_module, "_resident_hazards", lambda _l: hazards["value"])
    runner = _runner(runner_module, mla_absorbs={3: _absorb(3), 7: _absorb(7)})
    assert runner._mla_absorbs_for(8) == {(16, 16, True): built}
    assert seen == [(8, 16, 16, True)]
    hazards["value"] = ("1 vgpr spill",)
    assert runner._mla_absorbs_for(8) == {(16, 16, True): None}
    hazards["value"] = ()
    assert runner._mla_absorbs_for(65) == {(16, 16, True): None}
    assert _runner(runner_module, mla_absorbs={})._mla_absorbs_for(8) is None


def test_the_width_lists_its_absorbs_for_the_resident_checks(runner_module):
    absorb = SimpleNamespace(launcher=object(), abi="abi")
    build = runner_module.Gfx1250WidthBuild({}, {}, None, None, {(1,): absorb, (2,): None})
    assert build.kernels() == [(absorb.launcher, "abi")]


def test_a_built_absorb_rides_behind_its_front(runner_module):
    front, absorb = _front(), _absorb(3)
    runner = _runner(runner_module, mla_fronts={3: front}, mla_absorbs={3: absorb})
    runner._mla_front = lambda *a: a
    build = SimpleNamespace(
        mla_front={front.key: "front_kernel"}, mla_absorb={absorb.key: "absorb_kernel"}
    )
    assert runner._mla_projection(build, 3)("x") == (
        "front_kernel", front, "x", ("absorb_kernel", absorb),
    )  # fmt: skip
    unbuilt = SimpleNamespace(mla_front=build.mla_front, mla_absorb={absorb.key: None})
    assert runner._mla_projection(unbuilt, 3)("x") == ("front_kernel", front, "x")


def _step(monkeypatch, runner_module, rows=4, **change):
    context = {
        "is_dummy_run": False,
        "is_prefill": False,
    }
    metadata = {
        "slot_mapping": torch.arange(rows, dtype=torch.int64),
        "dtype_q": dtypes.fp8,
    }
    cache = torch.zeros(32, 1, 576, dtype=dtypes.fp8)
    for name, value in change.items():
        if name in context:
            context[name] = value
        elif name == "cache":
            cache = value
        else:
            metadata[name] = value
    forward = SimpleNamespace(
        context=SimpleNamespace(**context),
        attn_metadata=SimpleNamespace(**metadata),
        kv_cache_data={"layer_3": SimpleNamespace(k_cache=cache)},
    )
    monkeypatch.setattr(runner_module, "get_forward_context", lambda: forward)
    return forward, cache


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"is_dummy_run": True}, "decode"),
        ({"is_prefill": True}, "decode"),
        ({"slot_mapping": torch.arange(5, dtype=torch.int64)}, "slot mapping"),
        ({"slot_mapping": torch.arange(4, dtype=torch.int32)}, "slot mapping"),
        ({"dtype_q": torch.bfloat16}, "q_out"),
        ({"cache": torch.zeros(32, 1, 576, dtype=torch.bfloat16)}, "KV cache"),
        ({"cache": torch.zeros(16, 2, 576, dtype=dtypes.fp8)}, "KV cache"),
        ({"cache": torch.zeros(0, 1, 576, dtype=dtypes.fp8)}, "KV cache"),
    ],
)
def test_a_step_the_absorb_cannot_take_writes_nothing(
    runner_module, monkeypatch, change, message
):
    _step(monkeypatch, runner_module, **change)
    runner = _runner(runner_module)
    kernel = SimpleNamespace(launcher=lambda *a, **k: pytest.fail("launched"))
    q8 = torch.zeros(4, 1536, dtype=dtypes.fp8)
    assert runner._mla_absorb(kernel, _absorb(3), q8, None, None, None) is None
    assert any(message in why for why in runner._mla_absorb_refused)


def test_the_absorb_reads_the_step_and_the_front(runner_module, monkeypatch):
    from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_gfx1250 as ma

    forward, cache = _step(monkeypatch, runner_module)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    calls = []
    kernel = SimpleNamespace(launcher=lambda *a, **k: calls.append((a, k)))
    runner = _runner(runner_module, mla_q=torch.zeros(4, dtype=torch.int32))
    absorb = _absorb(3, (12, 16, False))
    q8 = torch.zeros(4, 1536, dtype=dtypes.fp8)
    scale, kv, k_rope = (torch.zeros(4, n) for n in (12, 512, 64))
    q_out = runner._mla_absorb(kernel, absorb, q8, scale, kv, k_rope)
    assert (q_out.dtype, tuple(q_out.shape)) == (dtypes.fp8, (4, 16, 576))
    ((args, kwargs),) = calls
    assert kwargs == {"stream": "stream"}
    named = dict(zip(ma.MLA_ABSORB_ABI.names, args))
    assert named["q_fp8"] == q8.data_ptr() and named["q_scale"] == scale.data_ptr()
    assert named["kv"] == kv.data_ptr() and named["k_rope"] == k_rope.data_ptr()
    assert named["slots"] == forward.attn_metadata.slot_mapping.data_ptr()
    assert named["kv_cache"] == cache.data_ptr() and named["kv_stride"] == 576
    assert named["q_out"] == q_out.data_ptr() and named["q"] == runner.mla_q.data_ptr()
    assert named["ql_nope"] == 0 and named["w_uk"] == absorb.w_uk.data_ptr()


def test_the_front_hands_the_absorbs_q_out_as_a_sixth_element(
    runner_module, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    runner = _runner(
        runner_module,
        mla_qkv=torch.zeros(4, dtype=torch.int32),
        mla_partials=torch.zeros(4, dtype=torch.uint8),
    )
    seen = []
    runner._mla_absorb = lambda *a: seen.append(a) or "q_out"
    front_kernel = SimpleNamespace(launcher=lambda *a, **k: None)
    x = torch.zeros(4, 7168, dtype=torch.bfloat16)
    out = runner._mla_front(front_kernel, _front(), x, ("kernel", "absorb"))
    assert len(out) == 6 and out[5] == "q_out"
    assert seen[0][:2] == ("kernel", "absorb") and seen[0][2] is out[0]
    runner._mla_absorb = lambda *a: None
    assert len(runner._mla_front(front_kernel, _front(), x, ("kernel", "absorb"))) == 5
