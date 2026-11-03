# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's DCP merge on Kimi-K3's full-attention layers (``K2Build.dcp``).

Under DCP = TP a full-attention layer hands K2 its unmerged decode partials and
the ungated ``g_proj``; K2 merges and gates them itself. These tests pin the
plain-Python parts: when the merge is K2's, what the attention hands over and
where K2 keeps it. Whether K2's core equals the production merge bit for bit
needs a GPU (``ATOM_MONO_CHECK``).
"""

import itertools
from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono.config import MAX_ROWS, Dims
from atom.mono.runtime.consensus import MonoUnsupported

ROWS, HIDDEN = 4, 8


@pytest.fixture
def runner_module():
    pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
    pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")
    from atom.models.kimi_k3.mono import runner

    return runner


# ------------------------------------------------------------------ layout
def test_the_merge_ctas_start_no_o_proj_first(runner_module):
    from atom.models.kimi_k3.mono.kernels import layer as k2
    from atom.models.kimi_k3.mono.kernels.attention import dcp_merge

    # the merge CTAs are AttnRes CTAs, whose o_proj comes last: a merge CTA
    # past O0 would hold up the first o_proj round behind its merge
    assert dcp_merge.merge_tasks(MAX_ROWS) == 48 <= k2.O0
    assert dcp_merge.merge_tasks(1) == 12


@pytest.mark.parametrize("s", [1, 16, 17, MAX_ROWS])
def test_k2_regions_hold_the_merge_apart_from_every_other(runner_module, s):
    from atom.models.kimi_k3.mono.kernels import layer as k2
    from atom.models.kimi_k3.mono.kernels.attention import dcp_merge

    def disjoint(lay):
        spans = sorted(lay.values())
        return all(o0 + n0 <= o1 for (o0, n0), (o1, _) in itertools.pairwise(spans))

    peer = k2.peer_layout(s, dcp=True)
    assert peer["dcp"][1] == dcp_merge.dcp_pairs(s) * 8 and disjoint(peer)
    scratch = k2.scratch_layout(k2.K2Build(tokens=s, nblocks=2, dcp=True))
    assert scratch["crdy"][1] == dcp_merge.crdy_pairs(s) * 8 and disjoint(scratch)
    # the DSA indexer's regions start past K2's, the merge's included
    assert runner_module.ix_peer_offset(s, True) >= k2.peer_bytes(s, True)


@pytest.mark.parametrize("s", [1, 16, MAX_ROWS])
def test_a_deployment_without_the_merge_pays_nothing_for_it(runner_module, s):
    """Every step zeroes the whole peer data region and the scratch: the merge's
    regions exist only in a deployment that builds it."""
    from atom.models.kimi_k3.mono.kernels import layer as k2

    assert "dcp" not in k2.peer_layout(s)
    assert "crdy" not in k2.scratch_layout(k2.K2Build(tokens=s, nblocks=2))
    assert k2.peer_bytes(s) < k2.peer_bytes(s, True)
    assert runner_module.ix_peer_offset(s) < runner_module.ix_peer_offset(s, True)


def test_a_dcp_build_is_its_own_kernel(runner_module):
    from atom.models.kimi_k3.mono.kernels import layer as k2

    plain = k2.K2Build(tokens=1, nblocks=2)
    dcp = k2.K2Build(tokens=1, nblocks=2, dcp=True)
    assert plain != dcp
    for name in ("part_o", "part_lse", "owned", "gate"):
        assert name in k2.ABI.names


# ------------------------------------------------------------ _dcp_merge_ok
def _impl(**over):
    impl = {
        "dcp_world_size": 8,
        "dcp_group": SimpleNamespace(rank_in_group=3),
        "pbm_enabled": True,
        "dcp_comm_backend": "a2a",
        "num_heads": 12,
        "v_head_dim": 128,
    }
    impl.update(over)
    return SimpleNamespace(**impl)


def _merge_ok(impls, *, rank=3, tp=8):
    from atom.models.kimi_k3.mono.dcp import dcp_merge_ok

    return dcp_merge_ok(impls, rank, tp)


def test_dcp_equal_to_tp_merges_in_k2(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_DCP_MERGE", "1")
    assert _merge_ok([_impl(), _impl()])


def _layer(linear, impl=None):
    attn = SimpleNamespace(attn=SimpleNamespace(impl=impl))
    return SimpleNamespace(is_linear_attn=linear, self_attn=attn)


def test_the_full_attention_layers_are_the_ones_checked():
    from atom.models.kimi_k3.mono.dcp import mla_impls

    a, b = _impl(), _impl()
    layers = [_layer(True), _layer(False, a), _layer(True), _layer(False, b)]
    assert mla_impls(SimpleNamespace(layers=layers)) == [a, b]


@pytest.mark.parametrize(
    "over",
    [
        {"dcp_world_size": 1, "dcp_group": None},
        {"dcp_world_size": 4},
        {"dcp_group": SimpleNamespace(rank_in_group=2)},
        {"pbm_enabled": False},
        {"dcp_comm_backend": "ag_rs"},
        {"num_heads": 16},
        {"v_head_dim": 512},
    ],
)
def test_any_other_deployment_keeps_the_production_merge(monkeypatch, over):
    monkeypatch.setenv("ATOM_MONO_DCP_MERGE", "1")
    # one layer that does not fit turns the merge off for all of them: the
    # build key is the deployment's, not a layer's
    assert not _merge_ok([_impl(), _impl(**over)])


def test_the_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_DCP_MERGE", "0")
    assert not _merge_ok([_impl()])


def _armed(monkeypatch, impls, switch="1"):
    from atom.models.kimi_k3.mono.dcp import arm_dcp_merge

    monkeypatch.setenv("ATOM_MONO_DCP_MERGE", switch)
    for impl in impls:
        impl.dcp_rank_order_merge = False
    layers = [_layer(True)] + [_layer(False, m) for m in impls]
    return arm_dcp_merge(SimpleNamespace(layers=layers), 3, 8, None)


def test_arming_flags_every_full_attention_layer(monkeypatch):
    """Installing mono flags the layers' unserved decode steps for the
    rank-order combine before any graph is captured; off, nothing is flagged."""
    impls = [_impl(), _impl()]
    assert _armed(monkeypatch, impls)
    assert all(m.dcp_rank_order_merge for m in impls)
    assert not _armed(monkeypatch, impls, switch="0")
    assert not any(m.dcp_rank_order_merge for m in impls)


def test_a_check_that_raises_votes_no(monkeypatch):
    """A rank whose check fails must still reach the vote: raising before it
    would leave the other ranks waiting in it."""
    broken = SimpleNamespace(dcp_rank_order_merge=False)  # no DCP attributes
    assert not _armed(monkeypatch, [broken])
    assert not broken.dcp_rank_order_merge


def test_the_installer_arms_before_the_runner_exists(monkeypatch):
    """The decision is made in ``install_mono_decode`` and handed to the runner
    the first eligible step builds, not taken there."""
    from atom.models.kimi_k3.mono import dcp, dispatch

    seen = {}

    def arm(inner, rank, tp, group):
        seen["inner"] = inner
        return True

    monkeypatch.setattr(dcp, "arm_dcp_merge", arm)
    monkeypatch.setattr(dispatch, "config_refusal", lambda cfg: None)
    monkeypatch.setattr(dcp, "tp_context", lambda: (0, 8, None))
    inner = SimpleNamespace(layers=[])
    model = SimpleNamespace(model=inner)
    cfg = SimpleNamespace(tensor_parallel_size=8)
    wrapped = dispatch.install_mono_decode(model, cfg, None)
    assert seen["inner"] is inner
    assert wrapped._dcp_merge is True


def test_dcp_builds_are_the_full_attention_layers_only(runner_module):
    def plan(layer_id, ffn_only):
        seam = SimpleNamespace(eps=1e-6, out_eps=1e-6)
        moe = SimpleNamespace(blocks=1, write_idx=-1, seam=seam, ln_eps=1e-6)
        return SimpleNamespace(layer_id=layer_id, moe=moe, ffn_only=ffn_only)

    runner = object.__new__(runner_module.K3MonoDecodeRunner)
    runner.debug = False
    runner.plans = [plan(1, False), plan(3, True)]
    runner.timeline = None
    runner.dcp_merge = True
    keys = runner._k2_keys(8)
    assert (keys[1].dcp, keys[3].dcp) == (False, True)
    runner.dcp_merge = False
    assert not runner._k2_keys(8)[3].dcp


# --------------------------------------------------------------- _pre_o_dcp
def _sinking_layer(calls, sink_out, gate, raise_in_attn=False):
    impl = SimpleNamespace(dcp_merge_sink=None)

    def seam(prefix, blocks, add, add2):
        return "normed", calls["new_prefix"]

    def attn_and_gate(positions, normed):
        calls["attn"] = (positions, normed)
        calls["sink_during"] = dict(impl.dcp_merge_sink)
        if raise_in_attn:
            raise RuntimeError("attention failed")
        impl.dcp_merge_sink.update(sink_out)
        return "unmerged", gate

    attn = SimpleNamespace(attn=SimpleNamespace(impl=impl), attn_and_gate=attn_and_gate)
    return impl, SimpleNamespace(self_attention_attn_res=seam, self_attn=attn)


def _dcp_inputs(rows=ROWS, owned=True):
    o = torch.zeros(rows, 96, 128, dtype=torch.bfloat16)
    lse = torch.zeros(rows, 96, dtype=torch.float32)
    counts = torch.ones(rows, dtype=torch.int32) if owned else None
    gate = torch.zeros(rows, Dims(8).proj, dtype=torch.bfloat16)
    return {"o": o, "lse": lse, "owned_counts": counts}, gate


def _dcp_runner(runner_module, layer):
    runner = object.__new__(runner_module.K3MonoDecodeRunner)
    runner.dims = Dims(8)
    runner.ones = torch.ones(MAX_ROWS, dtype=torch.int32)
    runner.inner = SimpleNamespace(layers={3: layer})
    return runner


def _pre_o(runner):
    plan = SimpleNamespace(layer_id=3)
    prefix, blocks = torch.ones(ROWS, HIDDEN), torch.zeros(ROWS, 4, HIDDEN)
    return runner._pre_o_dcp(plan, "positions", prefix, blocks, None, None)


def test_the_attention_hands_k2_its_partials_and_gate(runner_module):
    sink_out, gate = _dcp_inputs()
    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)}
    impl, layer = _sinking_layer(calls, sink_out, gate)
    core, prefix, dcp = _pre_o(_dcp_runner(runner_module, layer))

    assert calls["sink_during"] == {}  # armed while the attention ran
    assert impl.dcp_merge_sink is None  # and disarmed after
    o, lse, owned, g = dcp
    assert o is sink_out["o"] and lse is sink_out["lse"]
    assert owned is sink_out["owned_counts"] and g is gate
    assert prefix is calls["new_prefix"]
    assert core.shape == (ROWS, Dims(8).proj) and core.dtype == torch.bfloat16


def test_a_dense_layer_is_handed_counts_of_one(runner_module):
    sink_out, gate = _dcp_inputs(owned=False)
    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)}
    _, layer = _sinking_layer(calls, sink_out, gate)
    runner = _dcp_runner(runner_module, layer)
    _, _, (_, _, owned, _) = _pre_o(runner)
    assert owned is runner.ones


def test_the_sink_is_disarmed_when_the_attention_raises(runner_module):
    sink_out, gate = _dcp_inputs()
    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)}
    impl, layer = _sinking_layer(calls, sink_out, gate, raise_in_attn=True)
    with pytest.raises(RuntimeError, match="attention failed"):
        _pre_o(_dcp_runner(runner_module, layer))
    assert impl.dcp_merge_sink is None


@pytest.mark.parametrize(
    "key,bad",
    [
        ("o", torch.zeros(ROWS, 96, 512, dtype=torch.bfloat16)),
        ("o", torch.zeros(ROWS, 96, 128, dtype=torch.float16)),
        ("lse", torch.zeros(ROWS - 1, 96, dtype=torch.float32)),
        ("lse", torch.zeros(96, ROWS, dtype=torch.float32).t()),
        ("owned_counts", torch.ones(ROWS, dtype=torch.int64)),
        ("gate", torch.zeros(ROWS, Dims(8).proj + 8, dtype=torch.bfloat16)),
    ],
)
def test_inputs_k2_cannot_read_are_refused(runner_module, key, bad):
    sink_out, gate = _dcp_inputs()
    if key == "gate":
        gate = bad
    else:
        sink_out[key] = bad
    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)}
    _, layer = _sinking_layer(calls, sink_out, gate)
    with pytest.raises(MonoUnsupported, match="DCP merge input"):
        _pre_o(_dcp_runner(runner_module, layer))


def test_an_attention_that_merged_anyway_is_refused(runner_module):
    """A decode that never reached ``_dcp_defer_merge`` left the sink empty."""
    _, gate = _dcp_inputs()
    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)}
    _, layer = _sinking_layer(calls, {}, gate)
    with pytest.raises(MonoUnsupported, match="DCP merge input"):
        _pre_o(_dcp_runner(runner_module, layer))


def test_k2_is_handed_the_merge_inputs(runner_module, monkeypatch):
    runner = object.__new__(runner_module.K3MonoDecodeRunner)
    runner.dcp_merge = True
    runner.check = False
    runner.aux = SimpleNamespace(layer_ids=())
    seen = {}
    core = torch.zeros(ROWS, Dims(8).proj, dtype=torch.bfloat16)
    dcp = ("o", "lse", "owned", "gate")
    runner._pre_o_dcp = lambda *a: (core, "prefix", dcp)
    runner._pre_o_core = lambda *a: pytest.fail("the merge is K2's")

    def k2(build, plan, core_in, prefix, blocks, dcp_in=None):
        seen.update(core=core_in, dcp=dcp_in)
        return "k2 out"

    runner._k2 = k2
    plan = SimpleNamespace(layer_id=3, mono=False, ffn_only=True)
    runner._layer("build", plan, "pos", "prefix0", "blocks", None, None, None)
    assert seen["core"] is core and seen["dcp"] is dcp


# ------------------------------------------------------------ the attention
def test_a_deferred_decode_hands_over_the_projected_partial():
    try:
        from atom.model_ops.attention_mla import MLAAttention
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")

    o = torch.zeros(ROWS, 96, 512)
    projected = torch.ones(ROWS, 96, 128)
    lse, counts = torch.zeros(ROWS, 96), torch.ones(ROWS, dtype=torch.int32)
    seen = {}

    def v_up_proj(x, W_V, W_V_scale, num_heads):
        seen.update(x=x, w=(W_V, W_V_scale), heads=num_heads)
        return projected

    fake = SimpleNamespace(
        pbm_enabled=True,
        W_V_dcp="W",
        W_V_dcp_scale="S",
        _v_up_proj=v_up_proj,
        dcp_merge_sink={},
    )
    out = MLAAttention._dcp_defer_merge(fake, o, lse, counts)
    assert out is projected
    assert seen == {"x": o, "w": ("W", "S"), "heads": 96}
    assert fake.dcp_merge_sink == {"o": projected, "lse": lse, "owned_counts": counts}


def test_a_deferred_merge_without_pbm_is_refused():
    try:
        from atom.model_ops.attention_mla import MLAAttention
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")

    fake = SimpleNamespace(pbm_enabled=False, dcp_merge_sink={})
    with pytest.raises(AssertionError, match="project-before-merge"):
        MLAAttention._dcp_defer_merge(fake, None, None, None)


def test_the_rank_order_combine_is_what_the_caller_asks_for(monkeypatch):
    """``rank_order`` reaches the combine, except for a fused-quant combine,
    which has no rank-order form."""
    try:
        from atom.model_ops import dcp_ops
        from atom.model_ops.attention_mla import MLAAttention
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")

    seen = []
    monkeypatch.setattr(
        dcp_ops, "dcp_lse_merge", lambda *a, **k: seen.append(k["rank_order"])
    )
    fake = SimpleNamespace(dcp_group="g", dcp_comm_backend="a2a")
    MLAAttention._dcp_merge(fake, "o", "lse", rank_order=True)
    MLAAttention._dcp_merge(fake, "o", "lse", quant_dtype="fp8", rank_order=True)
    MLAAttention._dcp_merge(fake, "o", "lse")
    assert seen == [True, False, False]


def test_an_armed_sink_takes_the_decode_before_its_merge():
    """``_dcp_decode`` hands a step to ``_dcp_defer_merge`` while the sink is
    armed and merges it otherwise."""
    try:
        from atom.model_ops.attention_mla import MLAAttention
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")

    calls = []
    fake = SimpleNamespace(
        _forward_decode=lambda q, kv, md, return_lse: ("o", "lse"),
        is_sparse_mla=False,
        dcp_owned_counts_buffer=None,
        _cp_triton_ctx=None,
        _dcp_defer_merge=lambda o, lse, owned: calls.append("defer") or "partial",
        _dcp_project_merge_out=lambda *a, **k: calls.append(k["rank_order"])
        or "merged",
        dcp_merge_sink={},
        dcp_rank_order_merge=True,
    )
    q = torch.zeros(ROWS, 96, 8)
    assert MLAAttention._dcp_decode(fake, q, None, None, True) == "partial"
    fake.dcp_merge_sink = None
    assert MLAAttention._dcp_decode(fake, q, None, None, True) == "merged"
    # a flagged layer's decode merge combines in rank order
    assert calls == ["defer", True]


def test_prefill_keeps_the_production_combine():
    """Prefill never runs on mono, so a flagged layer's prefill merge does not
    pay for the rank-order combine."""
    try:
        from atom.model_ops.attention_mla import MLAAttention
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")

    seen = {}
    fake = SimpleNamespace(
        _forward_prefill_mla=lambda *a, **k: ("o", "lse"),
        dcp_prefill_merge_bf16_ok=True,
        dcp_owned_counts_buffer=torch.ones(ROWS, dtype=torch.int32),
        dcp_rank_order_merge=True,
        _dcp_project_merge_out=lambda *a, **k: seen.update(k) or "merged",
    )
    q = torch.zeros(ROWS, 96, 8)
    got = MLAAttention._dcp_sparse_prefill(fake, q, None, None, use_qrep=True)
    assert got == "merged"
    assert not seen.get("rank_order", False)
