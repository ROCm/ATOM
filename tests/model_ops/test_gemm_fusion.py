# SPDX-License-Identifier: MIT
# CPU-only tests for atom.model_ops.gemm_fusion (fake aiter matcher, no GPU).
from types import SimpleNamespace

import pytest

from atom.config import GemmFusionConfig
from atom.model_ops.gemm_fusion import consensus, hooks, site as site_mod, stages


# ---------------------------------------------------------------- fake aiter
class FakeContractError(RuntimeError):
    pass


class FakePlan:
    vid = "fake.v1"

    def resources(self):
        return []


class FakeResources:
    world = 4
    caps = frozenset()
    bound = object()

    rank = 0
    _reduce_min = None

    def __init__(self, plan_n=(None, 0), outcome="agree", raise_contract=False):
        self.plan_n, self.outcome, self.raise_contract = plan_n, outcome, raise_contract
        self.agree_calls = 0
        self.resets = 0
        self.poison = 0
        self.reset_blocks = None  # set to an Event to make quiesce_and_reset hang
        self.inner = SimpleNamespace(poison_local=lambda: self.poison)

    def local_m(self, tokens):
        return tokens // self.world

    def match_ctx(self, m, graph_mode="eager"):
        return SimpleNamespace(m=m, graph_mode=graph_mode)

    def agree(self, site, tokens, plan, n, errored=False):
        self.agree_calls += 1
        return "error" if errored else self.outcome

    finalized = False

    def can_ensure(self, plan):
        return True

    def ensure(self, plan):
        pass

    def finalize(self):
        self.finalized = True

    def quiesce_and_reset(self):
        if self.reset_blocks is not None:
            self.reset_blocks.wait()
        self.resets += 1

    def release(self):
        pass


def _fake_gf(res):
    def match_prefix(spec, ctx):
        if res.raise_contract:
            raise FakeContractError("covered but contract fails")
        return res.plan_n

    return SimpleNamespace(
        GemmSpec=lambda **kw: SimpleNamespace(**kw),
        FusionSpec=lambda gemm, stages_: SimpleNamespace(gemm=gemm, stages=stages_),
        match_prefix=match_prefix,
        ContractError=FakeContractError,
    )


@pytest.fixture
def make_site(monkeypatch):
    def _make(res):
        hooks.reset_for_tests(res)
        hooks.set_phase("preparing" if res is not None else "pre")
        monkeypatch.setattr(stages, "_aiter_override", _fake_gf(res))
        linear = SimpleNamespace(output_size=10240, input_size=3072)
        stg = [SimpleNamespace(kind="sp_exchange"), SimpleNamespace(kind="head_norm_rope_cache")]
        return site_mod.GemmFusionSite(
            "L0.qkv", linear, stg, native_fns=None, attn=SimpleNamespace(impl=SimpleNamespace(gemm_fusion_prologue_supported=lambda: True)),
            q_size=8192, kv_size=128,
        )

    yield _make
    hooks.reset_for_tests()


# ---------------------------------------------------------------- consensus
def test_decision_key_is_process_independent_and_distinguishes():
    a = consensus.decision_key("s", 64, "v1", 2)
    assert a == consensus.decision_key("s", 64, "v1", 2)
    assert a != consensus.decision_key("s", 64, "v1", 1)
    assert a != consensus.decision_key("s", 128, "v1", 2)
    assert 0 <= a <= 0x7FFFFFFF


def _ranks(keys_err):
    """Simulate an elementwise-MIN reducer over several ranks' inputs."""

    def make(rank_inputs):
        return lambda _: [min(c) for c in zip(*rank_inputs)]

    return make


def _run_ranks(decisions):
    """decisions: list of (vid, n, errored) per rank -> list of outcomes/exceptions."""
    inputs = []
    for vid, n, err in decisions:
        k = consensus.decision_key("s", 64, "ERR" if err else vid, 0 if err else n)
        inputs.append([k, -k, -int(err)])
    reduce = lambda _: [min(c) for c in zip(*inputs)]
    out = []
    for vid, n, err in decisions:
        try:
            out.append(consensus.agree(reduce, "s", 64, vid, n, err))
        except consensus.RankDisagreement as e:
            out.append(e)
    return out


def test_consensus_all_agree():
    assert _run_ranks([("v", 2, False)] * 4) == ["agree"] * 4


def test_consensus_plan_mismatch_falls_back_everywhere():
    r = _run_ranks([("v", 2, False)] * 3 + [("v", 1, False)])
    assert r == ["disagree"] * 4


def test_consensus_all_error_is_error():
    assert _run_ranks([("v", 2, True)] * 4) == ["error"] * 4


def test_consensus_partial_error_raises_on_every_rank():
    r = _run_ranks([("v", 2, False)] * 3 + [("v", 2, True)])
    assert all(isinstance(x, consensus.RankDisagreement) for x in r)


def test_consensus_single_rank():
    assert consensus.agree(None, "s", 8, "v", 2) == "agree"
    assert consensus.agree(None, "s", 8, "v", 2, errored=True) == "error"


# ------------------------------------------------------------------- site
def test_resolve_hit_cached_per_bucket_and_consensus_once(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    r1 = s.resolve(64)
    assert (r1.n_fused, r1.vid, r1.reason) == (2, "fake.v1", "profile_hit")
    assert s.resolve(64) is r1
    assert res.agree_calls == 1  # not repeated outside the first resolution
    s.resolve(128)
    assert res.agree_calls == 2


def test_resolve_prefix_fallback_reports_covered_prefix(make_site):
    s = make_site(FakeResources(plan_n=(FakePlan(), 1)))
    assert s.resolve(2048).n_fused == 1


def test_resolve_uncovered_is_native(make_site):
    r = make_site(FakeResources(plan_n=(None, 0))).resolve(64)
    assert (r.plan, r.n_fused, r.reason) == (None, 0, "uncovered")


def test_resolve_rank_disagreement_is_native(make_site):
    r = make_site(FakeResources(plan_n=(FakePlan(), 2), outcome="disagree")).resolve(64)
    assert (r.plan, r.reason) == (None, "rank_disagree")


def test_resolve_contract_error_raises_and_is_sticky(make_site):
    res = FakeResources(raise_contract=True)
    s = make_site(res)
    with pytest.raises(site_mod.GemmFusionContractError):
        s.resolve(64)
    with pytest.raises(site_mod.GemmFusionContractError):
        s.resolve(64)
    assert res.agree_calls == 1  # no new collective on the repeat


def test_resolve_profile_off_is_native_without_aiter(make_site, monkeypatch):
    s = make_site(None)  # hooks resources None == OFF
    monkeypatch.setattr(stages, "_aiter_override", None)  # aiter must not be needed
    r = s.resolve(64)
    assert (r.plan, r.n_fused, r.reason) == (None, 0, "profile_off")


def test_fallback_reasons_pad_rows_and_input_layout(make_site):
    import torch

    s = make_site(FakeResources(plan_n=(FakePlan(), 2)))
    ctx = SimpleNamespace(is_dummy_run=False, real_tokens=62, padded_tokens=64)
    x = torch.zeros((16, 8), dtype=torch.float8_e4m3fn)
    xs = torch.zeros((16, 1), dtype=torch.float32)
    assert s._fallback_reason(ctx, x, xs) == "pad_rows"
    ctx.real_tokens = 64
    assert s._fallback_reason(ctx, x, xs) is None
    assert s._fallback_reason(ctx, x, None) == "unquantized_input"
    assert s._fallback_reason(ctx, x.to(torch.bfloat16), xs) == "unquantized_input"
    assert s._fallback_reason(ctx, x, xs.view(1, 16)) == "xscale_layout"
    ctx.is_dummy_run = True
    assert s._fallback_reason(ctx, x, xs) == "dummy_run"


def test_one_stage_large_m_has_no_pad_rows_constraint(make_site):
    import torch

    s = make_site(FakeResources(plan_n=(FakePlan(), 1)))
    ctx = SimpleNamespace(is_dummy_run=False, real_tokens=2000, padded_tokens=2048)
    x = torch.zeros((512, 8), dtype=torch.float8_e4m3fn)
    xs = torch.zeros((512, 1), dtype=torch.float32)
    assert s._fallback_reason(ctx, x, xs, n_fused=1) is None
    assert s._fallback_reason(ctx, x, xs, n_fused=2) == "pad_rows"


def _direct_fixture(make_site):
    import torch

    s = make_site(FakeResources(plan_n=(FakePlan(), 2)))
    s.bind_mode = "direct"
    s.resolve(64, "eager")
    t = dict(
        x=torch.zeros((16, 8), dtype=torch.float8_e4m3fn),
        xs=torch.zeros((16, 1), dtype=torch.float32),
        pos=torch.arange(64),
        slot=torch.arange(64),
        kc=torch.zeros(4), vc=torch.zeros(4), ks=torch.zeros(4), vs=torch.zeros(4),
    )
    state = {"slot_mapping": t["slot"], "k_cache": t["kc"], "v_cache": t["vc"],
             "k_scale": t["ks"], "v_scale": t["vs"]}
    ctx = SimpleNamespace(is_dummy_run=False, real_tokens=64, padded_tokens=64,
                          positions=t["pos"], prologue_state=lambda: state)
    return s, t, state, ctx


def test_direct_mode_never_uses_stale_pointers_and_never_falls_back_per_rank(make_site):
    """M-1: a moved x_scale/positions/slot is self-healed by copy (still fused, uniform);
    an unprepared bucket or a moved KV cache raises -- no one-rank native fallback."""
    import torch

    s, t, state, ctx = _direct_fixture(make_site)
    with pytest.raises(site_mod.GemmFusionBindError):  # never lazily bind
        s._prepare(ctx, t["x"], t["xs"])
    st = SimpleNamespace(prepared=True, rebound=False, ptrs=s._live_ptrs(t["xs"], t["pos"], state))
    s._staging[64] = st
    s._by_tokens[(64, "eager")] = site_mod.Resolution(FakePlan(), 2, "fake.v1", "profile_hit")
    r, reason, _, got = s._prepare(ctx, t["x"], t["xs"])
    assert r.plan is not None and got is st and st.rebound is False
    moved = torch.zeros((16, 1), dtype=torch.float32)  # small per-call tensor moved
    r, reason, _, got = s._prepare(ctx, t["x"], moved)
    assert r.plan is not None and got.rebound is True  # fused + copy, not native
    state["k_cache"] = torch.zeros(4)  # KV cache moved: fatal, not a fallback
    with pytest.raises(site_mod.GemmFusionBindError):
        s._prepare(ctx, t["x"], t["xs"])

def test_bind_mode_validated_and_recorded(make_site):
    s = make_site(FakeResources(plan_n=(FakePlan(), 2)))
    assert s.bind_mode == "staged"
    with pytest.raises(ValueError):
        site_mod.GemmFusionSite("x", SimpleNamespace(output_size=1, input_size=1), [], None,
                                attn=SimpleNamespace(), q_size=1, kv_size=1, bind_mode="bogus")
    ctx = SimpleNamespace(padded_tokens=64)
    d = s._decision(ctx, site_mod.Resolution(FakePlan(), 2, "v", ""), "profile_hit", "fused")
    assert d.bind_mode == "staged"
    d1 = s._decision(ctx, site_mod.Resolution(FakePlan(), 1, "v", ""), "profile_hit", "fused")
    assert d1.bind_mode == "raw"


def test_resolve_after_finalize_without_windows_is_native(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    res.can_ensure = lambda plan: False
    r = make_site(res).resolve(64)
    assert (r.plan, r.reason) == (None, "after_finalize")


# ---------------------------------------------------------- hooks / config
def test_hooks_noop_when_off():
    hooks.reset_for_tests()
    cfg = SimpleNamespace(gemm_fusion=GemmFusionConfig())
    hooks.init_gemm_fusion(cfg, sp_group=None)
    assert hooks.resources() is None
    hooks.before_graph_capture([8, 16])
    hooks.after_graph_capture()
    hooks.recover()
    hooks.shutdown()


def test_before_graph_capture_resolves_all_buckets_then_resets(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    hooks.before_graph_capture([64, 128, 256])
    assert sorted(t for t, _ in s._by_tokens) == [64, 128, 256]
    assert res.resets == 1
    hooks.after_graph_capture()
    hooks.recover()
    assert res.resets == 3


def test_config_default_off_and_from_dict():
    assert not GemmFusionConfig().enabled
    assert GemmFusionConfig.from_dict({"profile": "p"}).enabled
    with pytest.raises(ValueError):
        GemmFusionConfig.from_dict({"nope": 1})


# --------------------------------------------- native-path equivalence shape
def test_native_full_matches_todays_call_sequence(monkeypatch):
    """native_attention == qkv_proj -> ulysses_attention(output_transform=gather_heads_fp8)."""
    import atom.distributed.ulysses_sp as U
    from atom.model_ops.gemm_fusion import stages_native as N

    seen = {}
    monkeypatch.setattr(
        U,
        "ulysses_attention",
        lambda attn, q, k, v, pos, qs, qkv, output_transform=None: seen.update(
            attn=attn, q=q, k=k, v=v, pos=pos, qs=qs, qkv=qkv, ot=output_transform
        )
        or ("out", "scale"),
    )
    monkeypatch.setattr(N, "_output_transform", lambda: "gather_heads_fp8")

    class Lin:
        def __call__(self, x, x_scale=None):
            seen["lin"] = (x, x_scale)
            return FakeQKV()

    class FakeQKV:
        def split(self, sizes, dim=-1):
            seen["sizes"] = (sizes, dim)
            return [f"p{i}" for i in range(len(sizes))]

    site = SimpleNamespace(
        linear=Lin(),
        split_qkv=lambda qkv: site_mod.GemmFusionSite.split_qkv(
            SimpleNamespace(sparse=True, q_size=8, kv_size=2, index_q_size=2, index_k_size=1), qkv
        )[:3],
    )
    ctx = SimpleNamespace(attn="ATTN", positions="POS")
    out = N.native_attention(site, "X", "XS", ctx)
    assert out == ("out", "scale")
    assert seen["lin"] == ("X", "XS")
    assert seen["sizes"] == ([8, 2, 2, 2, 1], -1)  # same 5-way split as the model
    assert (seen["q"], seen["k"], seen["v"]) == ("p0", "p1", "p2")
    assert seen["qs"] is None and seen["ot"] == "gather_heads_fp8"
    assert seen["pos"] == "POS" and seen["attn"] == "ATTN"

# ------------------------------------------------- phases (L-3, L-2, M-1)
def test_phase_pre_is_native_not_cached_and_ready_is_late_bucket(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    hooks.set_phase("pre")
    r = s.resolve(64)
    assert (r.plan, r.reason) == (None, "not_prepared") and not s._by_tokens
    assert res.agree_calls == 0  # no collective before prepare
    hooks.set_phase("ready")
    r = s.resolve(64)
    assert (r.plan, r.reason) == (None, "late_bucket")
    assert res.agree_calls == 0  # nor after: decided without a collective in the forward


def test_resolve_cache_is_keyed_by_graph_mode(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    s.resolve(64, "eager")
    s.resolve(64, "capture")
    assert sorted(s._by_tokens) == [(64, "capture"), (64, "eager")]
    assert res.agree_calls == 2


def test_static_prologue_ineligible_is_native_on_agreement(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    s.attn.impl = SimpleNamespace(gemm_fusion_prologue_supported=lambda: False)
    r = s.resolve(64)
    assert (r.plan, r.reason) == (None, "prologue_unsupported")  # row covered, prologue cannot fuse


def test_static_prologue_missing_impl_or_method_is_not_eligible(make_site):
    """Fail safe: an attention without an impl / without the method resolves native."""
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    s.attn = SimpleNamespace()  # no impl at all
    assert s._static_prologue_ok() is False
    s.attn = SimpleNamespace(impl=SimpleNamespace())  # impl without the method
    assert s._static_prologue_ok() is False
    r = s.resolve(64)
    assert (r.plan, r.reason) == (None, "prologue_unsupported")
    s.attn = SimpleNamespace(impl=SimpleNamespace(gemm_fusion_prologue_supported=lambda: True))
    assert s._static_prologue_ok() is True


def test_agree_prepared_downgrades_unprepared_bucket(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    s.resolve(64, "eager")
    s.resolve(64, "capture")
    s.agree_prepared(64, prepared=False)
    assert s._by_tokens[(64, "eager")].reason == "unprepared"
    assert s._by_tokens[(64, "capture")].plan is None


# --------------------------------------------------- S-B2 check_step
def _ready_hooks(res):
    hooks.reset_for_tests(res)
    hooks.register_plan(FakePlan())
    return res


def test_check_step_clean_does_nothing(make_site):
    res = _ready_hooks(FakeResources())
    hooks.check_step()
    assert res.resets == 0


def test_check_step_poison_recovers_then_raises():
    res = _ready_hooks(FakeResources())
    res.poison = 1
    with pytest.raises(hooks.GemmFusionPoisonError):
        hooks.check_step()
    assert res.resets == 1  # collective recovery ran before raising


def test_check_step_any_rank_poison_makes_every_rank_recover():
    """rank 0 clean, rank 1 poisoned: the all-reduce makes BOTH raise and recover."""
    res = _ready_hooks(FakeResources())
    res._reduce_min = lambda v: [min(v[0], -1), v[1], v[2]]  # peer reports poison
    with pytest.raises(hooks.GemmFusionPoisonError):
        hooks.check_step()
    assert res.resets == 1


def test_check_step_decision_mismatch_across_ranks_is_an_error():
    res = _ready_hooks(FakeResources())
    res._reduce_min = lambda v: [v[0], v[1] - 1, v[2]]  # a peer's digest differs
    with pytest.raises(hooks.GemmFusionPoisonError):
        hooks.check_step()


def test_check_step_noop_when_off_or_not_ready():
    hooks.reset_for_tests()
    hooks.check_step()  # OFF: nothing
    res = FakeResources()
    hooks.reset_for_tests(res)
    hooks.set_phase("pre")
    hooks.check_step()
    assert res.resets == 0


def test_negative_control_without_poison_check_is_silent(monkeypatch):
    """E1 negative control: the seam disables the check, poison goes unnoticed."""
    from atom.model_ops.gemm_fusion import faults

    res = _ready_hooks(FakeResources())
    res.poison = 1
    monkeypatch.setenv(faults.ENV, "no_poison_check")
    faults.reload()
    hooks.check_step()  # no raise: this is what the check prevents
    assert res.resets == 0
    monkeypatch.delenv(faults.ENV)
    faults.reload()


def test_poison_rank_fault_seam_targets_one_rank(monkeypatch):
    from atom.model_ops.gemm_fusion import faults

    monkeypatch.setenv(faults.ENV, "poison_rank:1")
    faults.reload()
    assert faults.active("poison_rank", 1) == "" and faults.active("poison_rank", 0) is None
    monkeypatch.delenv(faults.ENV)
    faults.reload()


# ------------------------------------------------------------------ L-1
def test_recover_times_out_instead_of_hanging():
    import threading

    res = _ready_hooks(FakeResources())
    res.reset_blocks = threading.Event()  # a peer never arrives
    with pytest.raises(hooks.GemmFusionRecoveryTimeout):
        hooks.recover(timeout_s=0.2)
    res.reset_blocks.set()
    with pytest.raises(hooks.GemmFusionPoisonError):  # fused path now refuses to run
        hooks.check_step()


# ----------------------------------------------- S-B1 OFF == byte-identical
def test_model_py_has_no_gemm_fusion_reference_and_runner_gates_install():
    """The @support_torch_compile model file must stay untouched; OFF imports nothing."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert "gemm_fusion" not in (root / "atom/models/minimax_m3/model.py").read_text(encoding="utf-8")
    runner = (root / "atom/model_engine/model_runner.py").read_text(encoding="utf-8")
    assert "if config.gemm_fusion.enabled:" in runner and "install_m3_sites" in runner
    # every gemm_fusion import in the runner is inside a guard (OFF never imports it)
    for line in runner.splitlines():
        if "gemm_fusion" in line and "import" in line:
            assert line.startswith(" " * 12) or line.startswith(" " * 16), line


def test_install_swaps_forward_only_when_site_declared(monkeypatch):
    import sys
    import types as _types

    from atom.model_ops.gemm_fusion import model_adapters as MA
    import atom.model_ops.gemm_fusion as G

    class FakeDense:  # stands in for MiniMaxM3Attention
        def __init__(self):
            self.attn = SimpleNamespace(layer_name="model.layers.0.self_attn.attn", qkv_site=None)

        def forward(self):
            return "native"

    # The real model module needs the native aiter build; inject a stand-in so the
    # adapter's `from atom.models.minimax_m3.model import ...` never touches it.
    fake = _types.ModuleType("atom.models.minimax_m3.model")
    fake.MiniMaxM3Attention = FakeDense
    fake.MiniMaxM3SparseAttention = type("FakeSparse", (), {})
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.model", fake)

    calls = []
    monkeypatch.setattr(G, "declare_m3_qkv_site", lambda m, prefix, sparse: calls.append(prefix) or None)
    m = FakeDense()
    model = SimpleNamespace(modules=lambda: [m])
    assert MA.install_m3_sites(model) == 0 and m.forward() == "native"  # ineligible: untouched
    assert calls == ["model.layers.0.self_attn"]
    monkeypatch.setattr(G, "declare_m3_qkv_site", lambda m, prefix, sparse: object())
    assert MA.install_m3_sites(model) == 1 and m.forward.__func__ is MA._fused_forward

# ------------------------- golden adapter: the REAL _site_for, not a hand-built site
def _golden_case(mode):
    nk = (9216, 6144) if mode == "qkv_dense" else (9856, 6144)
    norm = lambda: SimpleNamespace(weight=object(), variance_epsilon=1e-6)
    return SimpleNamespace(
        spec=SimpleNamespace(mode=mode, tokens=64, padded_tokens=64, cache_page_size=16, gemm_nk=nk),
        rotary=SimpleNamespace(rotary_dim=64), q_norm=norm(), k_norm=norm(),
        index_q_norm=norm(), index_k_norm=norm(), caches={"index_cache": object()},
        weight=object(), weight_scale=object(),
    )


@pytest.mark.parametrize("mode,gm", [("qkv_dense", "dense"), ("qkv_indexer", "indexer"),
                                      ("qkv_indexer_reuse", "reuse")])
def test_golden_adapter_site_is_eligible_and_matches_aiter_spec(mode, gm, monkeypatch):
    """Regression: the fail-safe eligibility check made every harness site 'uncovered'
    because the adapter's impl lacked gemm_fusion_prologue_supported. Builds the real
    site via _site_for and compares its 2-stage signature with aiter's own spec builder
    (and the shipped profile) when the aiter worktree is importable."""
    import json
    import sys
    from pathlib import Path

    from atom.model_ops.gemm_fusion import golden_candidate as GC

    aiter_root = Path(__file__).resolve().parents[3] / "aiter-gemm-fusion"
    spec_py = aiter_root / "aiter/gemm_fusion/spec.py"
    specs_py = aiter_root / "aiter/ops/gemm_fusion/sp_qkv/specs.py"
    if not (spec_py.exists() and specs_py.exists()):
        pytest.skip("aiter worktree not available")
    import importlib.util
    import types

    def load(name, path):
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s)
        monkeypatch.setitem(sys.modules, name, m)
        s.loader.exec_module(m)
        return m

    for n in ("aiter", "aiter.gemm_fusion", "aiter.ops", "aiter.ops.gemm_fusion", "aiter.ops.gemm_fusion.sp_qkv"):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
        sys.modules[n].__path__ = []
    spec = load("aiter.gemm_fusion.spec", spec_py)
    sys.modules["aiter.gemm_fusion"].__dict__.update(
        {k: getattr(spec, k) for k in ("FusionSpec", "GemmSpec", "Field", "SPExchange", "HeadNormRopeCache")}
    )
    specs = load("aiter.ops.gemm_fusion.sp_qkv.specs", specs_py)
    monkeypatch.setattr(stages, "_aiter_override", sys.modules["aiter.gemm_fusion"])
    monkeypatch.setattr(stages, "_kv_block_size", lambda: 16)
    hooks.reset_for_tests()
    GC._SITES.clear()
    site, _, _ = GC._site_for(_golden_case(mode), None)
    assert site._static_prologue_ok() is True  # else every golden cell resolves 'uncovered'
    rows = json.loads((aiter_root / "aiter/configs/gemm_fusion/minimax_m3_sp4_gfx950_v1.json").read_text())["rows"]
    # small M: the full 2-stage chain (GEMM + SPExchange + HeadNormRopeCache)
    sig2 = site.spec.prefix(2).signature()
    assert sig2 == specs.sp_qkv_spec(gm).signature()
    assert sig2 in {r["sig"] for r in rows if "small" in r["vid"]}
    # large M: GEMM + SPExchange only (HeadNormRopeCache runs natively in ATOM). Large rows
    # are keyed by this 1-stage signature, so the prefix fallback only works if it matches.
    sig1 = site.spec.prefix(1).signature()
    assert sig1 == specs.sp_qkv_spec(gm).prefix(1).signature()
    large = {r["sig"] for r in rows if "large" in r["vid"]}
    assert sig1 in large, f"{mode}: 1-stage sig {sig1} not among large rows {sorted(large)}"
    assert sig1 != sig2
    GC._SITES.clear()

class _SigMatcher:
    """Fake aiter: returns a plan only when the spec prefix carries a registered signature,
    like match_prefix against an exact-row profile. Records every spec it was asked about."""

    class Plan:
        vid = "sp_qkv.small.v1"
        owner = "o"

        def resources(self):
            return []

    def __init__(self, sigs):
        self.sigs, self.asked = set(sigs), []

    def module(self):
        def match_prefix(spec, ctx):
            for n in range(len(spec.stages), 0, -1):
                sub = spec.prefix(n)
                self.asked.append((n, sub.signature()))
                if sub.signature() in self.sigs:
                    return _SigMatcher.Plan(), n
            return None, 0

        return SimpleNamespace(match_prefix=match_prefix, ContractError=FakeContractError)


def _fake_spec_module(matcher):
    """Minimal stand-ins for the aiter spec classes with a stable, content-derived signature."""
    import hashlib

    class _S:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def key(self):
            return repr(sorted((k, repr(v)) for k, v in self.__dict__.items()))

    class _Spec:
        def __init__(self, gemm, stages_):
            self.gemm, self.stages = gemm, tuple(stages_)

        def prefix(self, n):
            return _Spec(self.gemm, self.stages[:n])

        def signature(self):
            blob = self.gemm.key() + "|" + "|".join(s.key() for s in self.stages)
            return hashlib.sha256(blob.encode()).hexdigest()[:16]

    class _Stage(_S):
        def __init__(self, kind, **kw):
            super().__init__(kind=kind, **kw)

    mod = matcher.module()
    mod.GemmSpec = lambda **kw: _S(**kw)
    mod.FusionSpec = _Spec
    mod.Field = lambda *a: ("F",) + a
    mod.SPExchange = lambda **kw: _Stage("sp_exchange", **kw)
    mod.HeadNormRopeCache = lambda **kw: _Stage("head_norm_rope_cache", **kw)
    return mod


@pytest.mark.parametrize("mode", ["qkv_dense", "qkv_indexer", "qkv_indexer_reuse"])
def test_golden_adapter_sites_resolve_to_a_plan_when_a_row_matches(mode, monkeypatch):
    """The adapter's sites must be ELIGIBLE: resolve through the real site code to a plan
    (2 fused stages) when the matcher has a row for the site's own signature, and must
    carry no 'prologue_unsupported'. Uses only fakes; no aiter, no GPU."""
    from atom.model_ops.gemm_fusion import golden_candidate as GC

    matcher = _SigMatcher([])
    gf = _fake_spec_module(matcher)
    monkeypatch.setattr(stages, "_aiter_override", gf)
    monkeypatch.setattr(stages, "_kv_block_size", lambda: 16)
    res = FakeResources()
    hooks.reset_for_tests(res)
    hooks.set_phase("preparing")
    GC._SITES.clear()
    site, _, _ = GC._site_for(_golden_case(mode), None)
    matcher.sigs.add(site.spec.signature())  # the "profile row" for this 2-stage spec
    r = site.resolve(64, "eager")
    assert r.reason == "profile_hit", r
    assert (r.vid, r.n_fused) == ("sp_qkv.small.v1", 2)
    # and with no row it is a plain profile miss, not an eligibility problem
    matcher.sigs.clear()
    site._by_tokens.clear()
    assert site.resolve(64, "eager").reason == "uncovered"
    GC._SITES.clear()


def test_golden_dense_gate_honours_env_and_stays_in_sync_with_the_real_impl(monkeypatch):
    """The stand-in dense gate mirrors PagedAttentionImpl.gemm_fusion_prologue_supported;
    pin both the behaviour and the source conditions so they cannot drift apart."""
    import re
    from pathlib import Path

    from atom.model_ops.gemm_fusion import golden_candidate as GC

    impl = SimpleNamespace(head_dim=128, sliding_window=-1, rotary_emb=object(), q_norm=object(), k_norm=object())
    monkeypatch.delenv("ATOM_FORCE_ATTN_TRITON", raising=False)
    assert GC._prologue_supported("dense", impl) is True
    monkeypatch.setenv("ATOM_FORCE_ATTN_TRITON", "1")
    assert GC._prologue_supported("dense", impl) is False
    monkeypatch.delenv("ATOM_FORCE_ATTN_TRITON")
    for bad in (dict(head_dim=64), dict(sliding_window=128), dict(rotary_emb=None), dict(k_norm=None)):
        assert GC._prologue_supported("dense", SimpleNamespace(**{**impl.__dict__, **bad})) is False
    # sparse modes: unconditional True, exactly like SparseMHAPagedAttentionImpl
    assert GC._prologue_supported("indexer", SimpleNamespace()) is True
    assert GC._prologue_supported("reuse", SimpleNamespace()) is True

    src = (Path(__file__).resolve().parents[2] / "atom/model_ops/attention_mha.py").read_text(encoding="utf-8")
    body = src.split("def gemm_fusion_prologue_supported", 1)[1].split("def gemm_fusion_prologue_state", 1)[0]
    for needle in ("ATOM_FORCE_ATTN_TRITON", "self.sliding_window != -1", "self.head_dim != 128",
                   "self.rotary_emb is not None", "isinstance(self.q_norm, GemmaRMSNorm)", "self.k_norm is not None"):
        assert needle in body, needle
    sparse = src.split("class SparseMHAPagedAttentionImpl", 1)[1]
    assert re.search(r"def gemm_fusion_prologue_supported\(self\) -> bool:.*?return True", sparse, re.S)