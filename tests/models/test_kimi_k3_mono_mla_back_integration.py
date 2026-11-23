# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contracts of the gfx1250 MLA back half's integration: the switch,
``KimiFullAttention._gated_core``'s ``back`` (and the attention handing it
the decode's partials), the layer binding's refusals and the runner's
binding, width builds and launch."""

from types import SimpleNamespace

import pytest
import torch

from atom.mono.runtime.consensus import MonoUnsupported

aiter = pytest.importorskip("aiter", reason="the K3 model and runner import aiter")
from aiter import QuantType


def test_the_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_BACK", raising=False)
    assert envs.ATOM_KIMI_MONO_MLA_BACK is False
    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_BACK", "1")
    assert envs.ATOM_KIMI_MONO_MLA_BACK is True


# --------------------------------------------------------------- the model
@pytest.fixture
def model_module():
    try:
        from atom.models.kimi_k3 import model
    except ImportError as error:  # the model module imports the GPU stack
        pytest.skip(f"atom.models.kimi_k3.model needs the GPU stack: {error}")
    return model


def test_a_back_half_takes_the_decodes_partials_past_the_gate(
    model_module, monkeypatch
):
    monkeypatch.setattr(
        model_module,
        "fused_sigmoid_mul_maybe_quant",
        lambda *a, **k: pytest.fail("gated outside the back half"),
    )
    decoded = []

    def decode(q_out, back=None):
        decoded.append(q_out)
        return back("o", "lse")

    fake = SimpleNamespace(
        indexer=None,
        attn=SimpleNamespace(impl=SimpleNamespace(forward_decode_q_out=decode)),
        fuse_sigmoid_mul_quant=False,
        o_proj_quant_type_value=QuantType.No.value,
    )
    front = ("q8", "q_scale", "kv", "k_rope", "gate", "q_out")
    seen = []

    def back(o, lse, gate):
        seen.append((o, lse, gate))
        return "attn_out"

    attention = model_module.KimiFullAttention
    out = attention._gated_core(
        fake, "pos", "hidden", quantize=False, front=front, back=back
    )
    assert out == ("attn_out", None)
    assert decoded == ["q_out"] and seen == [("o", "lse", "gate")]
    fake._gated_core = lambda *a, **k: attention._gated_core(fake, *a, **k)
    assert attention.forward_mono_back(fake, "pos", "hidden", front, back) == "attn_out"
    # without the absorb's q_out there are no partials to take
    with pytest.raises(ValueError, match="q_out"):
        attention._gated_core(fake, "pos", "hidden", front=front[:5], back=back)


def _attention_module():
    try:
        pytest.importorskip("triton", reason="the kernel under test needs triton")
        from atom.model_ops import attention_mla
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")
    return attention_mla


@pytest.mark.parametrize("dcp", [1, 8])
def test_the_decode_from_q_out_hands_its_partials_to_a_back_half(monkeypatch, dcp):
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
        _dcp_decode=lambda *a, **k: calls.append(("dcp", a, k)) or "dcp_out",
        _forward_decode=lambda *a, **k: pytest.fail("decoded without DCP"),
    )

    def back(o, lse):
        return "attn_out"

    if dcp > 1:
        got = mla.MLAAttention.forward_decode_q_out(fake, "q_out", back=back)
        assert got == "dcp_out"
        assert calls == [
            (
                "dcp",
                ("q_out", cache, metadata, True),
                {"q_prepadded": False, "back": back},
            )
        ]
    else:
        with pytest.raises(ValueError, match="DCP"):
            mla.MLAAttention.forward_decode_q_out(fake, "q_out", back=back)


def test_the_dcp_decode_returns_the_back_halfs_output_unmerged():
    mla = _attention_module()
    fake = SimpleNamespace(
        _forward_decode=lambda *a, **k: ("o", "lse"),
        _dcp_project_merge_out=lambda *a, **k: pytest.fail("merged by production"),
        is_sparse_mla=False,
        _cp_triton_ctx=None,
    )
    got = mla.MLAAttention._dcp_decode(
        fake, "q_out", "cache", "metadata", True, back=lambda o, lse: (o, lse, "back")
    )
    assert got == ("o", "lse", "back")


# ------------------------------------------------------------- the binding
def _impl(**change):
    impl = {
        "dcp_world_size": 8,
        "dcp_rank": 3,
        "pbm_enabled": True,
        "is_sparse_mla": False,
        "dcp_comm_backend": "a2a",
        "dcp_group": SimpleNamespace(dcp_lse_merge=lambda *a, **k: None),
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "num_heads": 12,
        "W_V_dcp": torch.zeros(96, 128, 512, dtype=torch.float8_e4m3fn),
        "W_V_dcp_scale": torch.ones((), dtype=torch.float32),
        "layer_num": 3,
    }
    impl.update(change)
    return SimpleNamespace(**impl)


def _bind(monkeypatch, impl, rank=3, aiters_merge=True):
    _attention_module()
    from atom.models.kimi_k3.mono import weights
    from atom.models.kimi_k3.mono.config import Dims

    monkeypatch.setattr(weights, "bind_fp8_o_proj", lambda *a: ("w_o", "s_o"))
    monkeypatch.setattr(weights, "_merge_is_aiters", lambda _ops: aiters_merge)
    absorb = SimpleNamespace(attention=impl)
    return weights.bind_mla_back("layer", 3, Dims(8), absorb, rank)


def test_the_serving_layer_binds_the_groups_w_uv_in_place(monkeypatch):
    impl = _impl()
    back = _bind(monkeypatch, impl)
    assert back.w_uv is impl.W_V_dcp and back.s_uv is impl.W_V_dcp_scale
    assert (back.w_o, back.s_o) == ("w_o", "s_o")
    assert back.key == (8, 12) and back.heads == 96 and back.dcp_rank == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dcp_world_size": 1}, "no DCP"),
        ({"dcp_world_size": 4}, "TP group"),
        ({"dcp_rank": 2}, "TP group"),
        ({"pbm_enabled": False}, "latent"),
        ({"is_sparse_mla": True}, "sparse"),
        ({"dcp_comm_backend": "mailbox"}, "backend"),
        ({"dcp_group": SimpleNamespace()}, "merge kernel"),
        ({"v_head_dim": 256}, "widths"),
        ({"num_heads": 16}, "heads a rank"),
        ({"W_V_dcp_scale": None}, "BF16"),
        ({"W_V_dcp": torch.zeros(96, 128, 512, dtype=torch.bfloat16)}, "W_UV"),
        ({"W_V_dcp": torch.zeros(12, 128, 512, dtype=torch.float8_e4m3fn)}, "W_UV"),
        ({"W_V_dcp_scale": torch.ones(2, dtype=torch.float32)}, "scale"),
    ],
)
def test_a_layer_the_back_half_would_compute_differently_is_refused(
    monkeypatch, change, message
):
    with pytest.raises(MonoUnsupported, match=message):
        _bind(monkeypatch, _impl(**change))


def test_a_tree_whose_merge_is_the_triton_combine_is_refused(monkeypatch):
    with pytest.raises(MonoUnsupported, match="Triton combine"):
        _bind(monkeypatch, _impl(), aiters_merge=False)


def test_the_merge_is_aiters_where_cp_lse_a2a_takes_the_groups():
    _attention_module()
    from atom.models.kimi_k3.mono.weights import _merge_is_aiters

    def fused(out, lse, cp_group):
        merge = getattr(cp_group, "dcp_lse_merge", None)
        return merge(out, lse) if merge is not None else None

    def method(out, lse, cp_group):
        return cp_group.dcp_lse_merge(out, lse)

    def triton_only(out, lse, cp_group):
        return cp_group.world_size

    assert _merge_is_aiters(SimpleNamespace(cp_lse_a2a=fused))
    assert _merge_is_aiters(SimpleNamespace(cp_lse_a2a=method))
    assert not _merge_is_aiters(SimpleNamespace(cp_lse_a2a=triton_only))
    assert not _merge_is_aiters(SimpleNamespace())


def test_the_fp4_v_up_projection_is_refused(monkeypatch):
    mla = _attention_module()
    monkeypatch.setattr(mla, "is_rocm_aiter_fp4bmm_enabled", lambda: True)
    with pytest.raises(MonoUnsupported, match="FP4"):
        _bind(monkeypatch, _impl())


# -------------------------------------------------------------- the runner
@pytest.fixture
def runner_module():
    pytest.importorskip("flydsl", reason="the gfx1250 runner imports FlyDSL")
    from atom.models.kimi_k3.mono import runner_gfx1250

    return runner_gfx1250


def _back(layer_id=3, dcp=8, local_heads=12, rank=3):
    from atom.models.kimi_k3.mono.weights import MlaBackLayer

    t = torch.zeros(1)
    return MlaBackLayer(layer_id, t, t, t, t, local_heads, dcp, rank)


def _runner(runner_module, **fields):
    runner = object.__new__(runner_module.K3Gfx1250MonoDecodeRunner)
    runner.device = torch.device("cpu")
    runner.max_rows = 64
    runner.tp = 8
    runner.rank = 3
    runner.dims = SimpleNamespace(proj=1536, tp=8)
    for name, value in fields.items():
        setattr(runner, name, value)
    return runner


def test_switched_off_no_layer_binds_a_back_half(runner_module, monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_BACK", raising=False)
    monkeypatch.setattr(
        runner_module, "bind_mla_back", lambda *a: pytest.fail("bound with it off")
    )
    runner = _runner(runner_module, mla_absorbs={3: "absorb"})
    runner._bind_mla_backs()
    assert runner.mla_backs == {} and not hasattr(runner, "mla_core")
    assert runner._mla_back_peer_bytes() == 0
    # a runner whose fronts never bound has no absorbs to look at
    assert _runner(runner_module)._mla_back_peer_bytes() == 0


def test_switched_on_a_refused_layer_keeps_production(runner_module, monkeypatch):
    from atom.models.kimi_k3.mono.kernels.stages import mla_back_layout as lay

    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_BACK", "1")
    bound = []

    def bind(layer, layer_id, dims, absorb, rank):
        bound.append((layer_id, absorb, rank))
        if layer_id == 7:
            raise MonoUnsupported("no")
        return _back(layer_id)

    monkeypatch.setattr(runner_module, "bind_mla_back", bind)
    runner = _runner(
        runner_module,
        mla_absorbs={3: "a3", 7: "a7"},
        inner=SimpleNamespace(layers={3: None, 7: None}),
    )
    runner._bind_mla_backs()
    assert bound == [(3, "a3", 3), (7, "a7", 3)]
    assert sorted(runner.mla_backs) == [3]
    widest = lay.BackPlan(64, 8, 8, 12)
    assert tuple(runner.mla_core.shape) == (64, 1536)
    assert runner.mla_core.dtype == torch.bfloat16
    assert runner.mla_flags.numel() == widest.flag_words
    assert not runner.mla_flags.any()
    assert runner._mla_back_peer_bytes() == widest.peer_bytes


def test_a_back_half_that_does_not_build_clean_leaves_the_width_production(
    runner_module, monkeypatch
):
    built = SimpleNamespace(name="k3_mla_back_x", launcher="launcher", abi="abi")
    seen = []

    def build(plan, tp, dcp, local_heads, poll_together=None):
        seen.append((plan, tp, dcp, local_heads))
        if plan == 65:
            raise ValueError("too wide")
        return built

    monkeypatch.setattr(runner_module.mla_back_gfx1250, "build_mla_back", build)
    monkeypatch.setattr(runner_module, "compile_kernels", lambda k: [k[0][0]])
    hazards = {"value": ()}
    monkeypatch.setattr(runner_module, "_resident_hazards", lambda _l: hazards["value"])
    peers = torch.zeros(8, dtype=torch.int64)
    runner = _runner(
        runner_module, mla_backs={3: _back(3), 7: _back(7)}, mla_back_peers=peers
    )
    runner._plan = lambda rows, nblocks: rows
    assert runner._mla_backs_for(8) == {(8, 12): built}
    assert seen == [(8, 8, 8, 12)]
    hazards["value"] = ("1 vgpr spill",)
    assert runner._mla_backs_for(8) == {(8, 12): None}
    hazards["value"] = ()
    assert runner._mla_backs_for(65) == {(8, 12): None}
    # no banks mapped (the all-reduce by aiter), or nothing bound: production
    assert _runner(runner_module, mla_backs={3: _back(3)})._mla_backs_for(8) is None
    assert _runner(runner_module, mla_back_peers=peers)._mla_backs_for(8) is None


def test_the_width_lists_its_back_halves_for_the_resident_checks(runner_module):
    back = SimpleNamespace(launcher=object(), abi="abi")
    build = runner_module.Gfx1250WidthBuild(
        {}, {}, None, None, None, {(1,): back, (2,): None}
    )
    assert build.kernels() == [(back.launcher, "abi")]


def test_a_built_back_half_is_the_cores_way_past_o_proj(runner_module):
    back = _back(3)
    runner = _runner(runner_module, mla_backs={3: back})
    runner._mla_back = lambda *a: a
    build = SimpleNamespace(mla_back={back.key: "kernel"})
    taken = []
    run = runner._mla_back_half(build, 3, taken)
    assert run("o", "lse", "gate") == ("kernel", back, "o", "lse", "gate")
    assert taken == [("kernel", back, "o", "lse", "gate")]
    assert (
        runner._mla_back_half(SimpleNamespace(mla_back={back.key: None}), 3, []) is None
    )
    assert runner._mla_back_half(SimpleNamespace(mla_back=None), 3, []) is None
    assert runner._mla_back_half(build, 4, []) is None


def test_the_back_half_reads_the_partials_the_gate_and_the_banks(
    runner_module, monkeypatch
):
    pytest.importorskip("flydsl", reason="builds a FlyDSL kernel")
    from atom.models.kimi_k3.mono.kernels.stages import mla_back_gfx1250 as mb

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    calls = []
    kernel = SimpleNamespace(
        launcher=lambda *a, **k: calls.append((a, k)), plan=SimpleNamespace(rows=4)
    )
    back = _back(3)
    runner = _runner(
        runner_module,
        mla_core=torch.zeros(64, 1536, dtype=torch.bfloat16),
        mla_flags=torch.zeros(8, dtype=torch.int32),
        mla_back_peers=torch.zeros(8, dtype=torch.int64),
        k2_peers=SimpleNamespace(epochs=torch.zeros(256, dtype=torch.int32)),
    )
    o = torch.zeros(4, 96, 512, dtype=torch.bfloat16)
    lse = torch.zeros(4, 96, dtype=torch.float32)
    gate = torch.zeros(4, 1536, dtype=torch.bfloat16)
    out = runner._mla_back(kernel, back, o, lse, gate)
    assert (out.dtype, tuple(out.shape)) == (torch.bfloat16, (4, 7168))
    ((args, kwargs),) = calls
    assert kwargs == {"stream": "stream"}
    named = dict(zip(mb.MLA_BACK_ABI.names, args))
    assert named["o"] == o.data_ptr() and named["lse"] == lse.data_ptr()
    assert named["gate"] == gate.data_ptr() and named["out"] == out.data_ptr()
    assert named["w_uv"] == back.w_uv.data_ptr() and named["w_o"] == back.w_o.data_ptr()
    assert named["core"] == runner.mla_core.data_ptr()
    assert named["flags"] == runner.mla_flags.data_ptr()
    assert named["peers"] == named["merge_peers"] == runner.mla_back_peers.data_ptr()
    assert named["epochs"] == runner.k2_peers.epochs.data_ptr()
    assert named["rank"] == 3 == named["dcp_rank"]
    # partials of another shape (this rank's heads only): not this kernel's
    with pytest.raises(MonoUnsupported, match="partials"):
        runner._mla_back(kernel, back, o[:, :12].contiguous(), lse[:, :12], gate)
    assert len(calls) == 1


def test_a_layer_with_a_back_half_skips_the_gate_and_o_proj(runner_module):
    from atom.models.kimi_k3.mono.weights import MlaFrontLayer

    t = torch.zeros(1)
    front = MlaFrontLayer(7, t, t, torch.zeros(1536, 1), t, t, t, 1e-6, 1e-6, None)
    back = _back(7)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=lambda *a, **k: pytest.fail("o_proj ran"))
    )
    runner = _runner(
        runner_module,
        inner=SimpleNamespace(layers={7: layer}),
        routed={7: object()},
        ffn_layers={7},
        oproj_weights={7: ("w", "s")},
        mla_fronts={7: front},
        mla_backs={7: back},
    )
    runner._mla_front = lambda kernel, front, x: ("front of", x)
    runner._oproj = lambda *a: pytest.fail("the o_proj all-reduce ran")
    attn_out = torch.zeros(2, 4)
    runner._mla_back = lambda kernel, layer, o, lse, gate: attn_out
    calls = {}

    def pre_o_core(plan, positions, prefix, blocks, add, add2, project=None, back=None):
        calls["back"] = back
        # the absorb ran: the back half takes the partials, there is no core
        assert back("o", "lse", "gate") is attn_out
        return None, prefix

    runner._pre_o_core = pre_o_core
    seen = []
    runner._production_k2 = lambda plan, attn, prefix, blocks: (
        seen.append(attn) or (prefix, attn, None)
    )
    runner._write_aux = lambda *_a: None
    prefix = torch.zeros(2, 4)
    build = SimpleNamespace(
        oproj="oproj", k2={}, mla_front={front.key: "kernel"},
        mla_back={back.key: "back_kernel"},
    )  # fmt: skip
    runner._layer(
        build, SimpleNamespace(layer_id=7), None, prefix, torch.zeros(2, 1, 4),
        prefix, None, None,
    )  # fmt: skip
    assert calls["back"] is not None and seen == [attn_out]


def test_pre_o_core_runs_the_back_half_only_behind_the_absorbs_q_out():
    pytest.importorskip("flydsl", reason="the K3 mono runner imports FlyDSL")
    pytest.importorskip("triton", reason="the kernel under test needs triton")
    from atom.models.kimi_k3.mono import runner as base
    from atom.models.kimi_k3.mono.config import Dims

    seen = []
    core = torch.zeros(2, Dims(8).proj, dtype=torch.bfloat16)

    def forward_pre_o(positions, normed, front=None):
        seen.append(("pre_o", front))
        return core

    def forward_mono_back(positions, normed, front, back):
        seen.append(("back", positions, normed, front, back))
        return "attn_out"

    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            forward_pre_o=forward_pre_o, forward_mono_back=forward_mono_back
        )
    )
    runner = object.__new__(base.K3MonoDecodeRunner)
    runner.dims = Dims(8)
    prefix = torch.zeros(2, 4)
    runner._seam = lambda *a: (layer, "normed", prefix, 1, -1)
    plan = SimpleNamespace(layer_id=3)
    six = ("q8", "s", "kv", "rope", "gate", "q_out")
    got, got_prefix = runner._pre_o_core(
        plan, "pos", prefix, None, None, None, lambda normed: six, "back"
    )
    assert got is None and got_prefix is prefix
    assert seen == [("back", "pos", "normed", six, "back")]
    # the absorb refused the step (five elements): the core, production's way
    seen.clear()
    got, _ = runner._pre_o_core(
        plan, "pos", prefix, None, None, None, lambda normed: six[:5], "back"
    )
    assert got is core and seen == [("pre_o", six[:5])]


def test_a_back_half_whose_polls_do_not_reread_keeps_its_volatile_polls(
    runner_module, monkeypatch
):
    """A build whose polls issue their loads together is read as compiled
    (``poll_isa.bound_build``): one that does not re-read every line is
    replaced by the volatile build, never bound."""
    from atom.utils import poll_isa

    together = SimpleNamespace(name="k3_mla_back_x_pt", launcher="l", abi="a")
    volatile = SimpleNamespace(name="k3_mla_back_x", launcher="l", abi="a")
    asked = []

    def build(plan, tp, dcp, local_heads, poll_together=None):
        asked.append(poll_together)
        return volatile if poll_together is False else together

    monkeypatch.setattr(runner_module.mla_back_gfx1250, "build_mla_back", build)
    monkeypatch.setattr(runner_module, "compile_kernels", lambda k: [k[0][0]])
    monkeypatch.setattr(runner_module, "_resident_hazards", lambda _l: ())
    found = {"value": ()}
    monkeypatch.setattr(poll_isa, "_compiled_hazards", lambda kernel: found["value"])
    peers = torch.zeros(8, dtype=torch.int64)
    runner = _runner(runner_module, mla_backs={3: _back(3)}, mla_back_peers=peers)
    runner._plan = lambda rows, nblocks: rows
    assert runner._mla_backs_for(8) == {(8, 12): together}
    assert asked == [None]
    found["value"] = ("the poll loop at 0x1f00 loads nothing: it re-reads no line",)
    asked.clear()
    assert runner._mla_backs_for(8) == {(8, 12): volatile}
    assert asked == [None, False]
