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


def test_dcp_equal_to_tp_merges_in_k2():
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
def test_any_other_deployment_keeps_the_production_merge(over):
    # one layer that does not fit turns the merge off for all of them: the
    # build key is the deployment's, not a layer's
    assert not _merge_ok([_impl(), _impl(**over)])


def _armed(impls):
    from atom.models.kimi_k3.mono.dcp import arm_dcp_merge

    for impl in impls:
        impl.dcp_rank_order_merge = False
    layers = [_layer(True)] + [_layer(False, m) for m in impls]
    return arm_dcp_merge(SimpleNamespace(layers=layers), 3, 8, None)


def test_arming_flags_every_full_attention_layer():
    """Installing mono flags the layers' unserved decode steps for the
    rank-order combine before any graph is captured; a deployment K2 does not
    merge flags nothing."""
    impls = [_impl(), _impl()]
    assert _armed(impls)
    assert all(m.dcp_rank_order_merge for m in impls)
    other = [_impl(), _impl(pbm_enabled=False)]
    assert not _armed(other)
    assert not any(m.dcp_rank_order_merge for m in other)


def test_a_check_that_raises_votes_no():
    """A rank whose check fails must still reach the vote: raising before it
    would leave the other ranks waiting in it."""
    broken = SimpleNamespace(dcp_rank_order_merge=False)  # no DCP attributes
    assert not _armed([broken])
    assert not broken.dcp_rank_order_merge


def test_the_installer_arms_before_the_runner_exists(monkeypatch):
    """The decision is made in ``install_mono_decode`` and handed to the runner
    the first eligible step builds, not taken there."""
    pytest.importorskip("aiter", reason="the installer's TP context is aiter's")
    from atom.models.kimi_k3.mono import dcp, dispatch

    seen = {}

    def arm(inner, rank, tp, group):
        seen["inner"] = inner
        return True

    monkeypatch.setattr(dcp, "arm_dcp_merge", arm)
    monkeypatch.setattr(dispatch, "config_refusal", lambda cfg: None)
    from aiter.dist import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(cpu_group=None)
    )
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
    runner.dcp_merge = True
    keys = runner._k2_keys(8)
    assert (keys[1].dcp, keys[3].dcp) == (False, True)
    runner.dcp_merge = False
    assert not runner._k2_keys(8)[3].dcp


# --------------------------------------------------------------- _pre_o_dcp
def _deferring_layer(calls, gate, counts=None, raise_in_attn=False, defers=True):
    impl = SimpleNamespace(
        dcp_defer_merge=False,
        dcp_partial_lse=None,
        is_sparse_mla=counts is not None,
        dcp_owned_counts_buffer=counts,
    )
    calls["lse"] = torch.zeros(ROWS, 96, dtype=torch.float32)

    def seam(prefix, blocks, add, add2):
        return "normed", calls["new_prefix"]

    def attn_and_gate(positions, normed):
        calls["armed"] = impl.dcp_defer_merge
        if raise_in_attn:
            raise RuntimeError("attention failed")
        if defers:
            impl.dcp_partial_lse = calls["lse"]  # what the deferred decode leaves
        return "no data", gate

    attn = SimpleNamespace(attn=SimpleNamespace(impl=impl), attn_and_gate=attn_and_gate)
    return impl, SimpleNamespace(self_attention_attn_res=seam, self_attn=attn)


def _dcp_runner(runner_module, layer):
    runner = object.__new__(runner_module.K3MonoDecodeRunner)
    runner.dims = Dims(8)
    runner.ones = torch.ones(MAX_ROWS, dtype=torch.int32)
    runner.dcp_out = torch.zeros(MAX_ROWS, 96, 128, dtype=torch.bfloat16)
    runner.inner = SimpleNamespace(layers={3: layer})
    return runner


def _pre_o(runner):
    plan = SimpleNamespace(layer_id=3)
    prefix, blocks = torch.ones(ROWS, HIDDEN), torch.zeros(ROWS, 4, HIDDEN)
    return runner._pre_o_dcp(plan, "positions", prefix, blocks, None, None)


def _gate():
    return torch.zeros(ROWS, Dims(8).proj, dtype=torch.bfloat16)


def test_the_attention_hands_k2_its_partials_and_gate(runner_module):
    gate, counts = _gate(), torch.ones(ROWS, dtype=torch.int32)
    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)}
    impl, layer = _deferring_layer(calls, gate, counts)
    runner = _dcp_runner(runner_module, layer)
    core, prefix, dcp = _pre_o(runner)

    assert calls["armed"] and not impl.dcp_defer_merge  # armed while it ran
    assert impl.dcp_partial_lse is None  # handed over, not left behind
    o, lse, owned, g = dcp
    assert o is runner.dcp_out and lse is calls["lse"]
    assert owned is counts and g is gate
    assert prefix is calls["new_prefix"]
    assert core.shape == (ROWS, Dims(8).proj) and core.dtype == torch.bfloat16


def test_a_dense_layer_is_handed_counts_of_one(runner_module):
    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)}
    _, layer = _deferring_layer(calls, _gate())
    runner = _dcp_runner(runner_module, layer)
    _, _, (_, _, owned, _) = _pre_o(runner)
    assert owned is runner.ones


def test_the_deferral_is_disarmed_when_the_attention_raises(runner_module):
    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)}
    impl, layer = _deferring_layer(calls, _gate(), raise_in_attn=True)
    with pytest.raises(RuntimeError, match="attention failed"):
        _pre_o(_dcp_runner(runner_module, layer))
    assert not impl.dcp_defer_merge


def test_a_decode_that_merged_itself_is_caught(runner_module):
    """K2 would merge the previous layer's partial: the deferral is checked,
    on the host, every step."""
    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)}
    _, layer = _deferring_layer(calls, _gate(), defers=False)
    with pytest.raises(AssertionError, match="merged itself"):
        _pre_o(_dcp_runner(runner_module, layer))


def _bind_layers(g_width):
    def full():
        g_proj = SimpleNamespace(output_size=g_width)
        attn = SimpleNamespace(attn=SimpleNamespace(impl=SimpleNamespace()))
        attn.g_proj = g_proj
        return SimpleNamespace(is_linear_attn=False, self_attn=attn)

    kda = SimpleNamespace(is_linear_attn=True, self_attn=None)
    return [kda, full(), kda, full()]


def test_bind_hands_every_full_attention_layer_the_merge_buffers(runner_module):
    runner = object.__new__(runner_module.K3MonoDecodeRunner)
    runner.dims, runner.device = Dims(8), torch.device("cpu")
    runner.inner = SimpleNamespace(layers=_bind_layers(Dims(8).proj))
    runner._bind_dcp_merge()
    assert runner.dcp_out.shape == (MAX_ROWS, 96, 128)
    assert runner.dcp_out.dtype == torch.bfloat16
    for layer in runner.inner.layers[1::2]:
        assert layer.self_attn.attn.impl.dcp_partial_out is runner.dcp_out


def test_bind_refuses_a_gate_k2_cannot_read(runner_module):
    runner = object.__new__(runner_module.K3MonoDecodeRunner)
    runner.dims, runner.device = Dims(8), torch.device("cpu")
    runner.inner = SimpleNamespace(layers=_bind_layers(Dims(8).proj + 8))
    with pytest.raises(MonoUnsupported, match="g_proj width"):
        runner._bind_dcp_merge()


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
def test_a_deferred_decode_writes_the_callers_buffers(monkeypatch):
    try:
        from atom.model_ops import attention_mla
        from atom.model_ops.attention_mla import MLAAttention
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")

    monkeypatch.setattr(
        attention_mla,
        "get_current_atom_config",
        lambda: SimpleNamespace(hf_config=SimpleNamespace(hidden_size=HIDDEN)),
    )
    o = torch.zeros(ROWS, 96, 512)
    lse = torch.arange(ROWS * 96, dtype=torch.float32).view(ROWS, 96)
    seen = {}

    def v_up_proj(x, W_V, W_V_scale, num_heads, out):
        seen.update(x=x, w=(W_V, W_V_scale), heads=num_heads, out=out)
        out.fill_(1.0)
        return out

    fake = SimpleNamespace(
        pbm_enabled=True,
        W_V_dcp="W",
        W_V_dcp_scale="S",
        _v_up_proj=v_up_proj,
        dcp_partial_out=torch.zeros(MAX_ROWS, 96, 128, dtype=torch.bfloat16),
        dcp_partial_lse=None,
        o_proj=torch.nn.Identity(),
        num_heads=12,
        v_head_dim=128,
    )
    out = MLAAttention._dcp_defer_merge(fake, o, lse)
    # the op's declared shape (``_mla_output_width``), no data
    assert out.shape == (ROWS, 12 * 128)
    assert seen["x"] is o and seen["w"] == ("W", "S") and seen["heads"] == 96
    assert seen["out"].data_ptr() == fake.dcp_partial_out.data_ptr()
    assert seen["out"].shape == (ROWS, 96, 128)
    assert fake.dcp_partial_lse is lse  # handed over, not copied
    assert fake.dcp_partial_out[:ROWS].eq(1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runs AITER's fp8 bmm")
def test_the_v_projection_writes_the_buffer_it_is_given():
    """The deferred merge's partial reaches K2 only through ``out``: the bmm
    must write that buffer, with the bits it would return on its own."""
    try:
        from aiter import dtypes

        from atom.model_ops.attention_mla import MLAAttention
        from atom.model_ops.utils import dynamic_per_batched_tensor_quant
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")

    heads, lora, v, rows = 96, 512, 128, 5
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(rows, heads, lora, generator=g, device="cuda").bfloat16()
    w = torch.randn(heads, v, lora, generator=g, device="cuda") * 0.05
    w, scale = dynamic_per_batched_tensor_quant(w, dtype=dtypes.fp8)
    fake = SimpleNamespace(kv_lora_rank=lora, v_head_dim=v, num_heads=heads)

    ref = MLAAttention._v_up_proj(fake, x, w, scale, num_heads=heads)
    buf = torch.full((MAX_ROWS, heads, v), float("nan"), device="cuda").bfloat16()
    got = MLAAttention._v_up_proj(fake, x, w, scale, num_heads=heads, out=buf[:rows])
    assert got.data_ptr() == buf.data_ptr()
    assert torch.equal(buf[:rows].view(torch.int16), ref.view(torch.int16))
    assert buf[rows:].isnan().all()  # nothing past the step's rows


def test_a_deferred_merge_without_pbm_is_refused():
    try:
        from atom.model_ops.attention_mla import MLAAttention
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")

    fake = SimpleNamespace(pbm_enabled=False)
    with pytest.raises(AssertionError, match="project-before-merge"):
        MLAAttention._dcp_defer_merge(fake, None, None)


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


def test_a_deferring_layer_skips_its_merge():
    """``_dcp_decode`` hands a step to ``_dcp_defer_merge`` while the caller
    defers the merge and merges it otherwise."""
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
        _dcp_defer_merge=lambda o, lse: calls.append("defer") or "partial",
        _dcp_project_merge_out=lambda *a, **k: calls.append(k["rank_order"])
        or "merged",
        dcp_defer_merge=True,
        dcp_rank_order_merge=True,
    )
    q = torch.zeros(ROWS, 96, 8)
    assert MLAAttention._dcp_decode(fake, q, None, None, True) == "partial"
    fake.dcp_defer_merge = False
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
