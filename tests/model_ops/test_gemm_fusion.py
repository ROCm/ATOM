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
    def match_prefix(spec, ctx, profile=None):
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
def _golden_case(mode, tokens=64):
    """A case whose ``spec`` is the REAL golden ``CaseSpec`` (page size, gemm_nk, padded
    tokens come from it, never hand-typed). Falls back to the documented values only when
    the golden tree is not on disk, and says so via ``case.spec_is_real``."""
    import importlib.util
    from pathlib import Path

    norm = lambda: SimpleNamespace(weight=object(), variance_epsilon=1e-6)
    gold = Path(__file__).resolve().parents[3] / "sp4-attention-fusion-cases-20261009" / "case_config.py"
    if gold.exists():
        s = importlib.util.spec_from_file_location("golden_case_config", gold)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        spec, real = m.CaseSpec(mode, tokens), True
    else:  # documented golden values (case_config.py: dense page 128, indexer/reuse 16)
        has_idx = mode != "qkv_dense"
        spec = SimpleNamespace(
            mode=mode, tokens=tokens, padded_tokens=(tokens + 3) // 4 * 4,
            cache_page_size=16 if has_idx else 128,
            gemm_nk=((9856 if has_idx else 9216), 6144),
        )
        real = False
    return SimpleNamespace(
        spec=spec, spec_is_real=real,
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
    prof_path = aiter_root / "aiter/configs/gemm_fusion/minimax_m3_sp4_gfx950_v1.json"
    try:
        rows = json.loads(prof_path.read_text())["rows"]
    except (ValueError, OSError):  # aiter worktree mid-regeneration (empty/partial file)
        pytest.skip("aiter profile not readable (being regenerated)")
    # small M: the full 2-stage chain (GEMM + SPExchange + HeadNormRopeCache). The kv_cache
    # string carries the page size, which comes from the REAL golden CaseSpec (dense 128,
    # indexer/reuse 16). aiter >= 27479d5f builds its spec/profile from the same table, so
    # the signatures must be EQUAL and present in the profile for every mode -- strictly.
    page = _golden_case(mode).spec.cache_page_size
    assert page == (128 if gm == "dense" else 16)
    assert f"_page{page}_" in site.spec.stages[1].kv_cache
    assert site.declared_page() == page
    sig2 = site.spec.prefix(2).signature()
    assert sig2 == specs.sp_qkv_spec(gm).signature(), f"{mode}: page-{page} signature differs from aiter's"
    assert sig2 in {r["sig"] for r in rows if "small" in r["vid"]}    # large M: GEMM + SPExchange only (HeadNormRopeCache runs natively in ATOM). Large rows
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
        def match_prefix(spec, ctx, profile=None):
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

# ---------------------- KV page: derived from the real cache, verified at bind time
class _FakeK:
    """Shape-only stand-in for a K cache tensor [blocks, kv_heads, head_dim//x, page, x]."""

    def __init__(self, page):
        self.shape = (4, 1, 8, page, 16)

    def dim(self):
        return 5


def _page_site(make_site, declared_page, sparse):
    s = make_site(FakeResources())
    s.sparse = sparse
    s.stages = (SimpleNamespace(kind="sp_exchange"),
                SimpleNamespace(kind="head_norm_rope_cache", kv_cache=f"shuffle_page{declared_page}_fp8_ptok"))
    s.attn = SimpleNamespace(impl=SimpleNamespace(layer_num=3))
    return s


def _kv(page):
    return {"layer_3": SimpleNamespace(k_cache=_FakeK(page))}


def test_declared_page_is_read_from_the_signature_string(make_site):
    assert _page_site(make_site, 128, False).declared_page() == 128
    assert _page_site(make_site, 16, True).declared_page() == 16


def test_bind_caches_dense_page_must_equal_the_pool_page(make_site):
    s = _page_site(make_site, 128, sparse=False)
    s.bind_caches(_kv(128))  # matches
    with pytest.raises(site_mod.GemmFusionBindError, match="declares KV page 128"):
        s.bind_caches(_kv(16))  # e.g. a hardcoded-16 signature against a page-128 pool
    s16 = _page_site(make_site, 16, sparse=False)
    with pytest.raises(site_mod.GemmFusionBindError):
        s16.bind_caches(_kv(128))  # the opposite mistake


def test_bind_caches_sparse_reviews_the_pool_as_page16(make_site):
    s = _page_site(make_site, 16, sparse=True)
    s.bind_caches(_kv(128))  # page-128 pool re-viewed as 8 x page-16: fine
    s.bind_caches(_kv(16))
    with pytest.raises(site_mod.GemmFusionBindError):
        _page_site(make_site, 128, sparse=True).bind_caches(_kv(16))  # cannot be coarser than the pool


def test_kv_block_size_has_no_silent_default(monkeypatch):
    """A guessed page would silently miss (or wrongly match) a profile row."""
    import sys
    import types as _t

    cfg = _t.ModuleType("atom.config")
    cfg.get_current_atom_config = lambda: (_ for _ in ()).throw(RuntimeError("no config"))
    monkeypatch.setitem(sys.modules, "atom.config", cfg)
    with pytest.raises(RuntimeError):
        stages._kv_block_size()
    cfg.get_current_atom_config = lambda: SimpleNamespace(kv_cache_block_size=128)
    assert stages._kv_block_size() == 128

# ----------------- gate04: eager-prepared plan must be THE plan capture/replay launches
class _BindingPlan:
    """Like aiter's SpQkvSmall: pointer bindings live in a per-INSTANCE dict, and a launch
    that needs a binding it does not have raises (the 'new binding required during capture')."""

    vid = "sp_qkv.small.v1"

    def __init__(self):
        self.owner = "sp_qkv.small/dense/x"
        self._bindings = {}

    def resources(self):
        return []

    def prepare(self, **kw):
        self._bindings[("k",)] = True

    def launch(self, in_capture):
        if ("k",) not in self._bindings:
            if in_capture:
                raise RuntimeError("sp_qkv.small: new binding required during graph capture")
            self._bindings[("k",)] = True


def _fresh_plan_matcher():
    """match_prefix returns a NEW plan object on every call, exactly like the real matcher."""
    made = []

    def match_prefix(spec, ctx, profile=None):
        p = _BindingPlan()
        made.append(p)
        return p, 2

    return match_prefix, made


def test_resolve_shares_one_plan_instance_between_eager_and_capture(make_site, monkeypatch):
    res = FakeResources()
    s = make_site(res)
    match_prefix, made = _fresh_plan_matcher()
    stages._aiter_override.match_prefix = match_prefix
    eager = s.resolve(64, "eager")
    capture = s.resolve(64, "capture")
    assert len(made) == 2  # the matcher really did build two objects...
    assert eager.plan is capture.plan  # ...but the site keeps ONE
    assert eager.plan is made[0]
    # order does not matter
    s2 = make_site(FakeResources())
    stages._aiter_override.match_prefix = match_prefix
    c2 = s2.resolve(128, "capture")
    e2 = s2.resolve(128, "eager")
    assert c2.plan is e2.plan


def test_prepare_then_capture_launch_never_needs_a_new_binding(make_site):
    """The gate04 failure, reproduced: prepare binds the eager plan; capture used to get a
    different, unbound plan. Now the capture-mode resolution IS the prepared plan."""
    s = make_site(FakeResources())
    match_prefix, _ = _fresh_plan_matcher()
    stages._aiter_override.match_prefix = match_prefix
    eager = s.resolve(64, "eager")
    capture = s.resolve(64, "capture")
    eager.plan.prepare()  # what prepare_bucket does, once
    capture.plan.launch(in_capture=True)  # must not raise


def test_regression_without_sharing_capture_would_need_a_new_binding(make_site):
    """Control: with distinct instances (the old behaviour) the same sequence fails, so the
    test above has power."""
    p_eager, p_capture = _BindingPlan(), _BindingPlan()
    p_eager.prepare()
    with pytest.raises(RuntimeError, match="new binding required"):
        p_capture.launch(in_capture=True)


def test_execute_refuses_a_plan_other_than_the_prepared_one(make_site):
    s = make_site(FakeResources())
    prepared = _BindingPlan()
    st = SimpleNamespace(prepared=True, plan=prepared, rebound=False, ptrs=(0,) * 7,
                         xs=None, pos=None, slot=None, outs={}, m=16)
    other = site_mod.Resolution(_BindingPlan(), 2, "sp_qkv.small.v1", "profile_hit")
    ctx = SimpleNamespace(padded_tokens=64, real_tokens=64)
    s.bind_mode = "direct"
    s.build_meta = lambda st, state: {}
    with pytest.raises(site_mod.GemmFusionBindError, match="other than the one prepared"):
        s._execute(other, None, None, ctx, {}, st, None)


def test_binding_key_names_every_pointer_aiter_keys_on(make_site):
    class _T:
        def __init__(self, p):
            self.p = p

        def data_ptr(self):
            return self.p

    s = make_site(FakeResources())
    s.linear.weight_scale = _T(1)
    s.build_meta = lambda st, state: {"q_norm_w": _T(2), "positions": _T(3), "k_cache": _T(4)}
    st = SimpleNamespace(xs=_T(9), outs={"q": _T(5), "k": _T(6)}, m=16)
    key = s.binding_key(st, {})
    assert key == {"x_scale": 9, "w_scale": 1, "q_norm_w": 2, "positions": 3, "k_cache": 4,
                   "out_q": 5, "out_k": 6, "rows": 16}
    st.outs["q"] = _T(55)  # a per-call output buffer would show up as a differing key
    assert s.binding_key(st, {}) != key

def test_m3_smoke_register_hf_config_is_idempotent_and_keeps_architectures(monkeypatch):
    """Logic only, against a stand-in transformers (the real 5.16.1 is not installed here):
    registers a PretrainedConfig subclass for the model_type once, tolerates a second call,
    and re-raises unrelated registration errors."""
    import importlib.util
    import sys
    import types as _t
    from pathlib import Path

    registry = {}

    class _Pre:
        model_type = ""

    class _Auto:
        @staticmethod
        def register(mt, cls):
            if mt in registry:
                raise ValueError(f"'{mt}' is already used by a Transformers config")
            registry[mt] = cls

    fake = _t.ModuleType("transformers")
    fake.AutoConfig, fake.PretrainedConfig = _Auto, _Pre
    monkeypatch.setitem(sys.modules, "transformers", fake)
    path = Path(__file__).resolve().parents[2] / "atom/model_ops/gemm_fusion/m3_smoke.py"
    s = importlib.util.spec_from_file_location("m3_smoke_under_test", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    cls = m.register_hf_config()
    assert registry["minimax_m3"] is cls and cls.model_type == "minimax_m3" and issubclass(cls, _Pre)
    m.register_hf_config()  # second call: already registered -> no error
    _Auto.register = staticmethod(lambda mt, c: (_ for _ in ()).throw(ValueError("something else")))
    with pytest.raises(ValueError, match="something else"):
        m.register_hf_config("other_type")

def _load_m3_smoke(name="m3_smoke_vl_under_test"):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "atom/model_ops/gemm_fusion/m3_smoke.py"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_m3_smoke_vl_layout_follows_the_released_checkpoint_and_keeps_geometry():
    m = _load_m3_smoke()
    c = m.m3_vl_config(layers=4)
    # outer / nested model types and the entry class the recipes' VL checkpoint selects
    assert (c["model_type"], c["text_config"]["model_type"]) == ("minimax_m3_vl", "minimax_m3_text")
    assert c["architectures"] == ["MiniMaxM3SparseForConditionalGeneration"]
    assert c["auto_map"] == {"AutoConfig": "configuration_minimax_m3.MiniMaxM3Config"}
    t = c["text_config"]
    # attention geometry unchanged (the fused site needs q width 8192 on SP4)
    assert t["num_attention_heads"] * t["head_dim"] == 8192 and t["hidden_size"] == 6144
    assert t["num_key_value_heads"] == 4 and t["head_dim"] == 128
    # index-cache controls sit on the OUTER config (ATOM copies them down), not in text_config
    assert c["index_topk_freq"] == 2 and "index_topk_freq" not in t
    assert "index_skip_topk_offset" in c and "index_skip_topk_offset" not in t
    # the three layer types the site declares are present
    assert t["sparse_attention_config"]["sparse_attention_freq"] == [0, 1, 1, 1]
    assert t["moe_layer_freq"][0] == 0


def test_m3_smoke_vl_dir_ships_a_config_class_and_serve_args_trust_it(tmp_path_factory=None):
    import os
    import sys
    import types as _t
    import uuid
    from pathlib import Path

    m = _load_m3_smoke("m3_smoke_vl_dir")
    d = Path(__file__).resolve().parents[2] / f"_tmp_m3_{uuid.uuid4().hex[:8]}"  # workspace scratch
    try:
        m.make_model_dir(str(d))
        assert sorted(os.listdir(d)) == ["config.json", "configuration_minimax_m3.py"]
        cfg_py = (d / "configuration_minimax_m3.py").read_text(encoding="utf-8")
        assert "class MiniMaxM3Config" in cfg_py and 'model_type = "minimax_m3_vl"' in cfg_py
        # the shipped module must execute and nest text_config as a sub-config
        class _P:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        fake = _t.ModuleType("transformers")
        fake.PretrainedConfig = _P
        saved = sys.modules.get("transformers")
        sys.modules["transformers"] = fake
        try:
            ns = {}
            exec(cfg_py, ns)
            import json

            raw = json.loads((d / "config.json").read_text(encoding="utf-8"))
            obj = ns["MiniMaxM3Config"](**{k: v for k, v in raw.items() if k != "auto_map"})
        finally:
            if saved is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = saved
        assert type(obj.text_config).__name__ == "MiniMaxM3TextConfig" and obj.hidden_size == 6144
        assert obj.architectures == ["MiniMaxM3SparseForConditionalGeneration"]
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)
    argv = m.serve_args("D", level=2, profile="p")
    assert "--trust-remote-code" in argv and argv[argv.index("--gemm-fusion-profile") + 1] == "p"
    assert "--trust-remote-code" not in m.serve_args("D", trust_remote_code=False)
    # flat layout is still available for the register_hf_config() route
    assert m.m3_config()["model_type"] == "minimax_m3"

# ----------------------------------------------- shutdown must never hang or mask an earlier failure
def test_shutdown_is_bounded_when_release_hangs():
    import threading
    import time

    res = FakeResources()
    gate = threading.Event()
    res.release = lambda: gate.wait()  # a peer never arrives: the barrier would block forever
    hooks.reset_for_tests(res)
    t0 = time.monotonic()
    hooks.shutdown(timeout_s=0.2)  # must return, not raise
    assert time.monotonic() - t0 < 5
    assert hooks.resources() is None and hooks.phase() == "pre"
    gate.set()


def test_shutdown_swallows_release_errors_and_still_clears_state():
    res = FakeResources()

    def boom():
        raise RuntimeError("HIP error: operation not supported")

    res.release = boom
    hooks.reset_for_tests(res)
    hooks.register_plan(FakePlan())
    hooks.shutdown(timeout_s=2)  # does not raise: it would mask the real start-up failure
    assert hooks.resources() is None and not hooks._PLANS


def test_shutdown_normal_path_still_releases_once():
    res = FakeResources()
    calls = []
    res.release = lambda: calls.append(1)
    hooks.reset_for_tests(res)
    hooks.shutdown()
    assert calls == [1]
    hooks.shutdown()  # idempotent
    assert calls == [1]

def test_m3_smoke_serve_args_use_a_sparse_compatible_block_size():
    """E2 OFF arm: the CLI default --block-size 16 fails the sparse backend's
    `model_runner.block_size % self.block_size == 0` (block 128). serve_args must pass 128."""
    m = _load_m3_smoke("m3_smoke_blocksize")
    argv = m.serve_args("D")
    assert argv[argv.index("--block-size") + 1] == "128"
    assert m.serve_args("D", block_size=256)[m.serve_args("D", block_size=256).index("--block-size") + 1] == "256"
    with pytest.raises(ValueError, match="multiple of 128"):
        m.serve_args("D", block_size=16)
    # the sparse constant is pinned to the real one in the source
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "atom/model_ops/minimax_m3/sparse_attn.py").read_text(encoding="utf-8")
    assert f"SPARSE_BLOCK_SIZE = {m.SPARSE_BLOCK_SIZE}" in src
    backends = (Path(__file__).resolve().parents[2] / "atom/model_ops/attentions/backends.py").read_text(encoding="utf-8")
    assert "model_runner.block_size % self.block_size == 0" in backends

def test_m3_smoke_default_is_all_dense_ffn_but_keeps_all_three_attention_types():
    """E2 OFF arm (job-e2_off03): any MoE layer raises a stock TypeError (x_scale). The smoke
    config therefore uses dense MLPs only, and must still cover dense, sparse-indexed and
    skip-index ATTENTION layers with the real attention geometry."""
    m = _load_m3_smoke("m3_smoke_dense_ffn")
    for cfg in (m.m3_config(layers=4), m.m3_vl_config(layers=4)["text_config"]):
        assert cfg["moe_layer_freq"] == [0, 0, 0, 0]  # no MoE layer anywhere
        assert cfg["sparse_attention_config"]["sparse_attention_freq"] == [0, 1, 1, 1]
        assert cfg["num_attention_heads"] * cfg["head_dim"] == 8192 and cfg["hidden_size"] == 6144
        assert cfg["dense_intermediate_size"] == 6144
    # skip-index layers still arise: sparse ordinals 0,1,2 with index_topk_freq 2 -> [F, T, F]
    outer = m.m3_vl_config(layers=4)
    assert outer["index_topk_freq"] == 2
    freq, off = outer["index_topk_freq"], outer["index_skip_topk_offset"]
    skips = [max(o - off, 0) % freq != 0 for o in range(3)]  # the model's own formula (model.py:105)
    assert skips == [False, True, False] and any(skips) and not all(skips)
    # explicit opt-in restores the old MoE layout
    assert m.m3_config(layers=4, moe=True)["moe_layer_freq"] == [0, 1, 1, 1]
    assert m.m3_vl_config(layers=4, moe=True)["text_config"]["moe_layer_freq"] == [0, 1, 1, 1]
    assert m.m3_vl_config(layers=3)["text_config"]["moe_layer_freq"] == [0, 0, 0]


def test_stock_moe_forward_still_lacks_x_scale_so_the_dense_default_is_still_needed():
    """Pins the reason for moe=False to the real source. When ATOM fixes MiniMaxM3MoE.forward to
    take x_scale this test FAILS, which is the signal to revisit the default."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "atom/models/minimax_m3/model.py").read_text(encoding="utf-8")
    assert "hidden_states = ffn(hidden_states, x_scale=hidden_states_scale)" in src
    moe_cls = src.split("class MiniMaxM3MoE", 1)[1].split("class MiniMaxM3Attention", 1)[0]
    assert re.search(r"def forward\(self, hidden_states: torch\.Tensor\) -> torch\.Tensor:", moe_cls)
    assert "x_scale" not in moe_cls
    mlp_cls = src.split("class MiniMaxM3MLP", 1)[1].split("class MiniMaxM3MoE", 1)[0]
    assert "x_scale: torch.Tensor | None = None" in mlp_cls  # the dense path accepts it