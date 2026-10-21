# SPDX-License-Identifier: MIT
# CPU-only tests for atom.model_ops.gemm_fusion (fake aiter matcher, no GPU).
import sys
from types import SimpleNamespace

import pytest

from atom.config import GemmFusionConfig
from atom.model_ops.gemm_fusion import consensus, hooks, stages
from atom.model_ops.gemm_fusion import site as site_mod
from atom.model_ops.gemm_fusion import test_utils as TU
from atom.model_ops.gemm_fusion.resources import FusionResourcesWrapper


@pytest.fixture(autouse=True)
def _clean_test_utils_patches():
    yield
    TU.uninstall_faults()


def _fake_linear(
    n, k, *, shuffled=True, rows=None, padded=False, ws_rows=None, k_cols=None
):
    """What the site reads off ATOM's QKV linear: logical N/K, a weight that is (or is not)
    16x16-preshuffled with a given shape, and a weight_scale with N rows."""
    w = SimpleNamespace(
        shape=(rows if rows is not None else n, k_cols if k_cols is not None else k)
    )
    if shuffled is not None:
        w.is_shuffled = shuffled
    return SimpleNamespace(
        output_size=n,
        input_size=k,
        weight=w,
        weight_scale=SimpleNamespace(
            shape=(
                ws_rows if ws_rows is not None else (rows if rows is not None else n),
                1,
            )
        ),
        is_output_padded=padded,
    )


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
    arch = "gfx950"
    cu = 256
    _reduce_min = None

    def __init__(self, plan_n=(None, 0), outcome="agree", raise_contract=False):
        self.plan_n, self.outcome, self.raise_contract = plan_n, outcome, raise_contract
        self.agree_calls = 0
        self.resets = 0  # every collective reset, either flavour
        self.error_resets = 0  # reset_after_error
        self.epoch_resets = 0  # clean_epoch
        self.reset_blocks = None  # set to an Event to make a reset hang
        # device error word / STEP model (the adapter API of FusionResourcesWrapper)
        self.error_value = 0  # this rank's device error word
        self.peer_flag = 0  # what the other ranks contribute to reduce_max
        self.readback_ready = True  # False: error_word has to wait for the copy
        self.snapshots = []  # error word value at each error_readback_async
        self.snapshot_steps = []  # the step index each snapshot was enqueued under
        self._ring = {}  # step index -> error word captured then
        self.call_id_report = (
            None  # what check_call_ids returns (None: no call-id module)
        )
        self.call_id_keys = None
        self.reductions = []  # values handed to each reduce_max
        self.blocking_waits = 0
        self.ticks = 0
        self.inner = SimpleNamespace()
        self._step = SimpleNamespace(add_=self._tick)

    def _tick(self, n):
        self.ticks += n

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

    def step_tensor(self):
        return self._step

    def error_readback_async(self, step_idx, stream=None):
        self._ring[step_idx] = self.error_value
        self.snapshots.append(self.error_value)
        self.snapshot_steps.append(step_idx)

    def error_word(self, step_idx):
        if not self.readback_ready:
            self.blocking_waits += 1
        return self._ring[step_idx]

    def error_description(self):
        return "fake error page"

    def check_call_ids(self, keys):
        self.call_id_keys = set(keys)
        return self.call_id_report

    def reduce_max(self, values):
        self.reductions.append(list(values))
        return [max(v, self.peer_flag) for v in values]

    def _reset(self):
        if self.reset_blocks is not None:
            self.reset_blocks.wait()
        self.resets += 1

    def reset_after_error(self):
        self._reset()
        self.error_resets += 1

    def clean_epoch(self):
        self._reset()
        self.epoch_resets += 1

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
    def _make(res, direct=False):
        TU.reset_for_tests(res)
        TU.set_phase("preparing" if res is not None else "pre")
        TU.inject_aiter(monkeypatch, _fake_gf(res))
        linear = _fake_linear(10240, 3072)
        stg = [
            SimpleNamespace(kind="sp_exchange"),
            SimpleNamespace(kind="head_norm_rope_cache"),
        ]
        cls = TU.DirectGemmFusionSite if direct else site_mod.GemmFusionSite
        return cls(
            "L0.qkv",
            linear,
            stg,
            native_fns=None,
            attn=SimpleNamespace(
                impl=SimpleNamespace(gemm_fusion_prologue_supported=lambda: True)
            ),
            q_size=8192,
            kv_size=128,
        )

    yield _make
    TU.reset_for_tests()


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

    def reduce(_):
        return [min(c) for c in zip(*inputs)]

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
    monkeypatch.setitem(
        sys.modules, "aiter.gemm_fusion", None
    )  # aiter must not be needed (import would raise)
    r = s.resolve(64)
    assert (r.plan, r.n_fused, r.reason) == (None, 0, "profile_off")


def test_fallback_reasons_input_layout_and_padded_rows_run_fused(make_site):
    import torch

    s = make_site(FakeResources(plan_n=(FakePlan(), 2)))
    ctx = SimpleNamespace(is_dummy_run=False, real_tokens=62, padded_tokens=64)
    x = torch.zeros((16, 3072), dtype=torch.float8_e4m3fn)
    xs = torch.zeros((16, 1), dtype=torch.float32)
    assert (
        TU.fallback_reason(s, ctx, x, xs) is None
    )  # 62 real of 64: padded rows run the fused kernel
    ctx.real_tokens = 64
    assert TU.fallback_reason(s, ctx, x, xs) is None
    assert TU.fallback_reason(s, ctx, x, None) == "unquantized_input"
    assert TU.fallback_reason(s, ctx, x.to(torch.bfloat16), xs) == "unquantized_input"
    assert TU.fallback_reason(s, ctx, x, xs.view(1, 16)) == "xscale_layout"
    ctx.is_dummy_run = True
    assert TU.fallback_reason(s, ctx, x, xs) == "dummy_run"


def test_padded_rows_do_not_change_the_path_for_either_plan_kind(make_site):
    import torch

    for n_fused in (1, 2):
        s = make_site(FakeResources(plan_n=(FakePlan(), n_fused)))
        ctx = SimpleNamespace(is_dummy_run=False, real_tokens=2000, padded_tokens=2048)
        x = torch.zeros((512, 3072), dtype=torch.float8_e4m3fn)
        xs = torch.zeros((512, 1), dtype=torch.float32)
        assert TU.fallback_reason(s, ctx, x, xs, n_fused=n_fused) is None


def _direct_fixture(make_site, res=None):
    import torch

    s = make_site(res or FakeResources(plan_n=(FakePlan(), 2)), direct=True)
    s.resolve(64, "eager")
    t = {
        "x": torch.zeros((16, 3072), dtype=torch.float8_e4m3fn),
        "xs": torch.zeros((16, 1), dtype=torch.float32),
        "pos": torch.arange(64),
        "slot": torch.arange(64),
        "kc": torch.zeros(4),
        "vc": torch.zeros(4),
        "ks": torch.zeros(4),
        "vs": torch.zeros(4),
    }
    state = {
        "slot_mapping": t["slot"],
        "k_cache": t["kc"],
        "v_cache": t["vc"],
        "k_scale": t["ks"],
        "v_scale": t["vs"],
    }
    ctx = SimpleNamespace(
        is_dummy_run=False,
        real_tokens=64,
        padded_tokens=64,
        positions=t["pos"],
        prologue_state=lambda: state,
    )
    return s, t, state, ctx


def test_decision_bind_label_is_staged_or_raw_for_the_runtime_site(make_site):
    s = make_site(FakeResources(plan_n=(FakePlan(), 2)))
    assert not hasattr(s, "bind_mode") and not hasattr(
        s, "direct_provider"
    )  # runtime site only stages
    ctx = SimpleNamespace(padded_tokens=64)
    d = s._decision(
        ctx, site_mod.Resolution(FakePlan(), 2, "v", ""), "profile_hit", "fused"
    )
    assert d.bind_mode == "staged"
    d1 = s._decision(
        ctx, site_mod.Resolution(FakePlan(), 1, "v", ""), "profile_hit", "fused"
    )
    assert d1.bind_mode == "raw"


def test_direct_site_labels_its_decisions_and_marks_a_moved_input_as_copy(make_site):
    s = make_site(FakeResources(plan_n=(FakePlan(), 2)), direct=True)
    assert s.bind_mode == "direct"
    ctx = SimpleNamespace(padded_tokens=64)
    r2 = site_mod.Resolution(FakePlan(), 2, "v", "")
    assert s._decision(ctx, r2, "profile_hit", "fused").bind_mode == "direct"
    assert (
        s._decision(
            ctx, r2, "profile_hit", "fused", SimpleNamespace(rebound=True)
        ).bind_mode
        == "direct+copy"
    )
    assert (
        s._decision(
            ctx, site_mod.Resolution(FakePlan(), 1, "v", ""), "profile_hit", "fused"
        ).bind_mode
        == "raw"
    )
    assert s._decision(ctx, r2, "dummy_run", "native").bind_mode == "direct"


def test_resolve_after_finalize_without_windows_is_native(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    res.can_ensure = lambda plan: False
    r = make_site(res).resolve(64)
    assert (r.plan, r.reason) == (None, "after_finalize")


# ---------------------------------------------------------- hooks / config
def test_hooks_noop_when_off():
    TU.reset_for_tests()
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
            SimpleNamespace(
                sparse=True, q_size=8, kv_size=2, index_q_size=2, index_k_size=1
            ),
            qkv,
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
    TU.set_phase("pre")
    r = s.resolve(64)
    assert (r.plan, r.reason) == (None, "not_prepared") and not s._by_tokens
    assert res.agree_calls == 0  # no collective before prepare
    TU.set_phase("ready")
    r = s.resolve(64)
    assert (r.plan, r.reason) == (None, "late_bucket")
    assert (
        res.agree_calls == 0
    )  # nor after: decided without a collective in the forward


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
    assert (r.plan, r.reason) == (
        None,
        "prologue_unsupported",
    )  # row covered, prologue cannot fuse


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
    s.attn = SimpleNamespace(
        impl=SimpleNamespace(gemm_fusion_prologue_supported=lambda: True)
    )
    assert s._static_prologue_ok() is True


def test_agree_prepared_downgrades_unprepared_bucket(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    s.resolve(64, "eager")
    s.resolve(64, "capture")
    s.agree_prepared(64, prepared=False)
    assert s._by_tokens[(64, "eager")].reason == "unprepared"
    assert s._by_tokens[(64, "capture")].plan is None


# --------------------------------------------------- R12 device error check
def _ready_hooks(res, period=14):
    TU.reset_for_tests(res)
    hooks._PERIOD = period
    hooks.register_plan(FakePlan())
    return res


def _wrapper_error_word(local=0):
    """The adapter's real ``error_word`` (as patched by a fault seam, if any) on a stand-in self."""
    return FusionResourcesWrapper.error_word(
        SimpleNamespace(
            inner=SimpleNamespace(readback_consume=lambda step, wait: local)
        ),
        0,
    )


def test_check_step_clean_async_schedule_snapshots_every_period_and_evaluates_two_steps_later():
    res = _ready_hooks(FakeResources())
    snapshot_at, evaluated_at = [], []
    for s in range(1, 31):
        n_snap, n_red = len(res.snapshots), len(res.reductions)
        assert hooks.check_step((s,)) is None
        if len(res.snapshots) > n_snap:
            snapshot_at.append(s)
        if len(res.reductions) > n_red:
            evaluated_at.append(s)
    assert snapshot_at == [14, 28] and evaluated_at == [16, 30]
    assert res.snapshot_steps == [14, 28]  # the ring is keyed by the step index
    assert res.resets == 0
    # once the snapshot of step 28 is proven clean (step 30) only the steps after it stay unchecked
    assert [s for s, _ in hooks._WINDOW] == [29, 30]
    assert res.blocking_waits == 0


def test_an_error_before_the_first_snapshot_aborts_the_first_sixteen_steps():
    res = _ready_hooks(FakeResources())
    res.error_value = 1
    results = {s: hooks.check_step((s,)) for s in range(1, 17)}
    assert {s: r for s, r in results.items() if r is not None} == {
        16: list(range(1, 17))
    }
    assert res.error_resets == 1 and res.epoch_resets == 0
    # the step counter keeps counting (aiter keys its readback ring by step); only the window and
    # the snapshot in flight are dropped
    assert hooks._STEP_IDX == 16 and not hooks._WINDOW and hooks._PENDING is None
    assert res.snapshot_steps == [14]  # snapshot enqueued under step 14, consumed at 16


def test_an_error_after_a_clean_snapshot_aborts_exactly_the_last_sixteen_steps():
    """period 14 + evaluation 2 steps later = a window of 16: the steps after the last snapshot
    proven clean (28) up to the step that evaluates the next one (44)."""
    res = _ready_hooks(FakeResources())
    for s in range(1, 31):
        assert hooks.check_step((s,)) is None
    res.error_value = 1  # the device error appears during step 31
    verdicts = {}
    for s in range(31, 45):
        r = hooks.check_step((s,))
        if r is not None:
            verdicts[s] = r
    assert verdicts == {44: list(range(29, 45))}
    assert len(verdicts[44]) == 16 == hooks._PERIOD + hooks.SNAPSHOT_EVAL_DELAY
    assert res.error_resets == 1


def test_check_step_any_rank_error_makes_every_rank_reset_and_abort():
    """This rank is clean; a peer reports: the MAX all-reduce makes THIS rank reset too."""
    res = _ready_hooks(FakeResources())
    res.peer_flag = 1
    out = [hooks.check_step((s,)) for s in range(1, 17)]
    assert out[15] == list(range(1, 17)) and all(r is None for r in out[:15])
    assert res.reductions == [[0]]  # ONE all-reduce of this rank's flag
    assert res.error_resets == 1


def test_a_snapshot_not_yet_complete_is_waited_for_not_skipped():
    """Skipping on 'not ready' would make ranks disagree about whether to all-reduce."""
    res = _ready_hooks(FakeResources())
    res.readback_ready = False
    for s in range(1, 17):
        assert hooks.check_step((s,)) is None
    assert res.blocking_waits == 1 and len(res.reductions) == 1


def test_check_step_period_one_is_synchronous_and_precedes_publication():
    """KV publishers: snapshot, wait and agree in the SAME step (before the KV is announced)."""
    res = _ready_hooks(FakeResources(), period=1)
    assert hooks.check_step((1, 2)) is None
    assert (len(res.snapshots), len(res.reductions), list(hooks._WINDOW)) == (1, 1, [])
    res.error_value = 1
    assert hooks.check_step((3, 4)) == [3, 4]  # same step, only this step's requests
    assert res.error_resets == 1


def test_a_verdict_with_nothing_to_abort_is_still_a_verdict():
    res = _ready_hooks(FakeResources(), period=1)
    res.error_value = 1
    assert (
        hooks.check_step(()) == []
    )  # e.g. a dummy batch: reset ran, caller must not publish KV
    assert res.error_resets == 1


def test_check_step_noop_when_off_or_not_ready():
    TU.reset_for_tests()
    assert hooks.check_step((1,)) is None and hooks.check_now() is None  # OFF
    res = FakeResources()
    TU.reset_for_tests(res)
    TU.set_phase("pre")
    assert hooks.check_step((1,)) is None and hooks.check_now() is None
    assert (res.snapshots, res.reductions, res.resets) == ([], [], 0)


def test_after_a_failed_recovery_the_fused_path_refuses_to_run():
    _ready_hooks(FakeResources())
    hooks._BROKEN = "recover timeout after 1s"
    with pytest.raises(hooks.GemmFusionDeviceError, match="failed recovery"):
        hooks.check_step((1,))
    with pytest.raises(hooks.GemmFusionDeviceError, match="failed recovery"):
        hooks.check_now()


def test_check_now_is_synchronous_and_has_the_check_step_contract():
    res = _ready_hooks(FakeResources())
    assert hooks.check_now() is None and len(res.snapshots) == 1
    res.error_value = 1
    assert hooks.check_now() == []  # nothing recorded yet, but the reset ran
    assert res.error_resets == 1


@pytest.mark.parametrize("hook", ["before_graph_capture", "after_graph_capture"])
def test_an_error_at_a_graph_capture_point_is_fatal_not_an_abort(hook):
    """No request exists there, so there is nothing to abort: reset, then raise."""
    res = _ready_hooks(FakeResources())
    call = (
        (lambda: hooks.before_graph_capture([]))
        if hook == "before_graph_capture"
        else hooks.after_graph_capture
    )
    call()  # clean: a clean epoch, no error reset
    assert (res.epoch_resets, res.error_resets) == (1, 0)
    res.error_value = 1
    with pytest.raises(hooks.GemmFusionDeviceError, match="plans re-initialised"):
        call()
    assert res.error_resets == 1


def test_the_first_registered_site_ticks_and_only_it():
    TU.reset_for_tests()
    a, b = SimpleNamespace(name="a"), SimpleNamespace(name="b")
    hooks.register_site(a)
    hooks.register_site(b)
    hooks.register_site(a)  # re-registration keeps the decision
    assert (a.ticks_step, b.ticks_step) == (True, False)


def test_tick_adds_one_to_the_step_tensor_only_when_ready():
    res = FakeResources()
    TU.reset_for_tests(res)
    hooks._STEP_T = res.step_tensor()
    hooks.tick()
    assert res.ticks == 1
    TU.set_phase("preparing")
    hooks.tick()  # the counter does not exist before ready; the phase is rank-uniform
    assert res.ticks == 1
    hooks._STEP_T = None
    TU.set_phase("ready")
    hooks.tick()
    assert res.ticks == 1


def test_run_ticks_before_anything_that_can_return_or_raise(make_site):
    """Once per forward on EVERY path: the tick is the first thing run() does."""
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    hooks._STEP_T = res.step_tensor()
    TU.set_phase("ready")

    def boom(*a, **k):
        raise RuntimeError("stop")

    s._prepare = boom
    s.ticks_step = True
    with pytest.raises(RuntimeError, match="stop"):
        s.run(None, None, None)
    assert res.ticks == 1
    s.ticks_step = False
    with pytest.raises(RuntimeError, match="stop"):
        s.run(None, None, None)
    assert res.ticks == 1  # a later site never ticks


def test_prepare_buckets_creates_the_step_view_and_starts_the_check_clean():
    res = FakeResources()
    TU.reset_for_tests(res)
    TU.set_phase("preparing")
    hooks._WINDOW.append((1, (9,)))
    hooks._STEP_IDX, hooks._PENDING = 5, 3
    hooks.prepare_buckets([])
    assert hooks.phase() == "ready" and hooks._STEP_T is res.step_tensor()
    assert res.epoch_resets == 1
    assert (hooks._STEP_IDX, hooks._PENDING, len(hooks._WINDOW)) == (0, None, 0)


def test_shutdown_drops_the_step_view_and_the_check_state():
    res = FakeResources()
    TU.reset_for_tests(res)
    hooks._STEP_T = res.step_tensor()
    hooks._WINDOW.append((1, (9,)))
    hooks._STEP_IDX = 4
    hooks.shutdown()
    assert hooks._STEP_T is None and hooks._STEP_IDX == 0 and not hooks._WINDOW


def test_error_word_fault_seam_targets_one_rank():
    res = _ready_hooks(FakeResources())
    res.rank = 1
    TU.install_faults("error_word:1")
    assert _wrapper_error_word() == 1  # rank 1 reads its error word as set
    res.rank = 0
    TU.install_faults("error_word:1")  # idempotent
    assert _wrapper_error_word() == 0  # rank 0 is untouched


def test_negative_control_without_error_check_is_silent():
    """E-1 negative control: the seam disables the check, the error goes unnoticed."""
    res = _ready_hooks(FakeResources(), period=1)
    res.error_value = 1
    assert hooks.check_step((1,)) == [1]  # without the seam the same error is a verdict
    res.resets = res.error_resets = 0
    TU.install_faults("no_error_check")
    assert hooks.check_step((2,)) is None  # no verdict: this is what the check prevents
    assert res.resets == 0


def test_recover_skip_seam_skips_the_collective_recovery_on_the_target_rank():
    res = _ready_hooks(FakeResources())
    TU.install_faults("recover_skip")
    hooks.recover()
    assert res.resets == 0
    TU.uninstall_faults()
    hooks.recover()
    assert res.resets == 1


def test_rank_native_seam_makes_only_the_target_rank_decide_native():
    kw = {
        "resolution": site_mod.Resolution(FakePlan(), 2, "fake.v1", "profile_hit"),
        "is_dummy_run": False,
        "x_dtype_is_fp8": True,
        "x_scale_ok": True,
    }
    orig = site_mod.GemmFusionSite.__dict__["decide_path"]
    res = _ready_hooks(FakeResources())
    res.rank = 1
    TU.install_faults("rank_native:1")
    assert site_mod.GemmFusionSite.decide_path(**kw) == ("native", "fault_rank_native")
    res.rank = 0
    assert site_mod.GemmFusionSite.decide_path(**kw) == (
        "fused",
        "profile_hit",
    )  # other ranks fuse
    TU.uninstall_faults()
    assert site_mod.GemmFusionSite.__dict__["decide_path"] is orig  # restored exactly


def test_slow_remainder_seam_spins_after_the_native_remainder(monkeypatch):
    spun = []
    monkeypatch.setattr(TU, "_spin_gpu_us", lambda us: spun.append(us))
    monkeypatch.setattr(
        site_mod,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: "s")),
        raising=False,
    )
    s = _ready_hooks(FakeResources()) and site_mod.GemmFusionSite.__new__(
        site_mod.GemmFusionSite
    )
    TU.install_faults("slow_remainder:*:750")
    out = s._remainder_after_exchange(
        None, {"packed": 1}, None, lambda packed: packed + 1
    )
    assert out == 2 and spun == [750]


def test_op_raise_seam_recovers_then_raises(monkeypatch):
    import types

    res = _ready_hooks(FakeResources())
    fake_ops = types.ModuleType("atom.model_ops.gemm_fusion.ops")
    fake_ops.run_qkv_attention = lambda attn, x, xs, pos: ("real-op-ran",)
    monkeypatch.setitem(sys.modules, "atom.model_ops.gemm_fusion.ops", fake_ops)
    TU.install_faults("op_raise")
    with pytest.raises(RuntimeError, match="op_raise"):
        fake_ops.run_qkv_attention(None, None, None, None)
    assert res.resets == 1  # the real op's except path: recover() before re-raising
    TU.uninstall_faults()
    assert fake_ops.run_qkv_attention(None, None, None, None) == ("real-op-ran",)


def test_two_fault_kinds_installed_together_each_keep_their_own_original():
    """Regression for a late-binding closure bug: with several kinds in one spec each wrapper must
    call ITS original, not whichever was assigned last."""
    res = _ready_hooks(FakeResources(), period=1)
    TU.install_faults("error_word:0,recover_skip:0,no_error_check:7")
    assert _wrapper_error_word() == 1  # error_word wrapper, rank 0
    hooks.recover()
    assert res.resets == 0  # recover_skip wrapper
    res.rank = 7
    res.error_value = 1
    assert hooks.check_step((1,)) is None  # no_error_check on rank 7: silent
    assert res.error_resets == 0
    TU.uninstall_faults()
    assert _wrapper_error_word() == 0


def test_unknown_fault_kind_is_rejected_and_the_env_spec_parses():
    with pytest.raises(ValueError, match="unknown gemm_fusion fault"):
        TU.install_faults("bogus")
    assert TU.parse_faults("error_word:1:x,no_error_check,op_raise:*") == {
        "error_word": (1, "x"),
        "no_error_check": (None, None),
        "op_raise": (None, None),
    }


def test_install_from_env_applies_faults_and_timeouts_and_uninstall_restores(
    monkeypatch,
):
    monkeypatch.setenv(TU.FAULT_ENV, "error_word")
    monkeypatch.setenv(TU.RECOVER_TIMEOUT_ENV, "7")
    monkeypatch.setenv(TU.SHUTDOWN_TIMEOUT_ENV, "3")
    before = (hooks.RECOVER_TIMEOUT_S, hooks.SHUTDOWN_TIMEOUT_S)
    TU.install_from_env()
    assert TU.fault_active("error_word") == ""
    assert (hooks.RECOVER_TIMEOUT_S, hooks.SHUTDOWN_TIMEOUT_S) == (7.0, 3.0)
    TU.uninstall_faults()
    assert (hooks.RECOVER_TIMEOUT_S, hooks.SHUTDOWN_TIMEOUT_S) == before
    assert TU.fault_active("error_word") is None


def test_the_runtime_package_has_no_test_hooks():
    """The runtime keeps only what serving needs: no module imports test_utils, and none carries a
    fault seam, a debug env branch, a test injection point, or a reset-for-tests helper.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "atom/model_ops/gemm_fusion"
    runtime = [f for f in root.glob("*.py") if f.name != "test_utils.py"]
    assert {f.name for f in runtime} >= {
        "hooks.py",
        "site.py",
        "ops.py",
        "resources.py",
        "stages.py",
    }
    forbidden = (
        "test_utils",
        "faults",
        "ATOM_GEMM_FUSION_FAULT",
        "ATOM_GEMM_FUSION_DEBUG_BIND",
        "ATOM_GEMM_FUSION_CHECK_ALIAS",
        "_aiter_override",
        "reset_for_tests",
        "golden_candidate",
        "m3_smoke",
        "hip_triage",
        "bind_mode=",
        "direct_provider",
        "gemm_fusion_kv_page",
        "ATOM_GEMM_FUSION_RECOVER_TIMEOUT_S",
        "ATOM_GEMM_FUSION_SHUTDOWN_TIMEOUT_S",
    )
    for f in runtime:
        src = f.read_text(encoding="utf-8")
        for word in forbidden:
            assert word not in src, f"{f.name} contains test-only {word!r}"
    for gone in ("faults.py", "golden_candidate.py", "m3_smoke.py", "hip_triage.py"):
        assert not (root / gone).exists()


# ------------------------------------------------------------------ L-1
def test_recover_times_out_instead_of_hanging():
    import threading

    res = _ready_hooks(FakeResources())
    res.reset_blocks = threading.Event()  # a peer never arrives
    with pytest.raises(hooks.GemmFusionRecoveryTimeout):
        hooks.recover(timeout_s=0.2)
    res.reset_blocks.set()
    with pytest.raises(hooks.GemmFusionDeviceError):  # fused path now refuses to run
        hooks.check_step((1,))


# ----------------------------------------------- S-B1 OFF == byte-identical
def test_model_py_has_no_gemm_fusion_reference_and_runner_gates_install():
    """The @support_torch_compile model file must stay untouched; OFF imports nothing."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert "gemm_fusion" not in (root / "atom/models/minimax_m3/model.py").read_text(
        encoding="utf-8"
    )
    runner = (root / "atom/model_engine/model_runner.py").read_text(encoding="utf-8")
    assert "if config.gemm_fusion.enabled:" in runner and "install_m3_sites" in runner
    # every gemm_fusion import in the runner is inside a guard (OFF never imports it)
    for line in runner.splitlines():
        if "gemm_fusion" in line and "import" in line:
            assert line.startswith((" " * 12, " " * 16)), line


def test_install_swaps_forward_only_when_site_declared(monkeypatch):
    import sys
    import types as _types

    import atom.model_ops.gemm_fusion as G
    from atom.model_ops.gemm_fusion import model_adapters as MA

    class FakeDense:  # stands in for MiniMaxM3Attention
        def __init__(self):
            self.attn = SimpleNamespace(
                layer_name="model.layers.0.self_attn.attn", qkv_site=None
            )

        def forward(self):
            return "native"

    # The real model module needs the native aiter build; inject a stand-in so the
    # adapter's `from atom.models.minimax_m3.model import ...` never touches it.
    fake = _types.ModuleType("atom.models.minimax_m3.model")
    fake.MiniMaxM3Attention = FakeDense
    fake.MiniMaxM3SparseAttention = type("FakeSparse", (), {})
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.model", fake)

    calls = []
    monkeypatch.setattr(
        G, "declare_m3_qkv_site", lambda m, prefix, sparse: calls.append(prefix) or None
    )
    m = FakeDense()
    model = SimpleNamespace(modules=lambda: [m])
    assert (
        MA.install_m3_sites(model) == 0 and m.forward() == "native"
    )  # ineligible: untouched
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

    def norm():
        return SimpleNamespace(weight=object(), variance_epsilon=1e-6)

    gold = (
        Path(__file__).resolve().parents[3]
        / "sp4-attention-fusion-cases-20261009"
        / "case_config.py"
    )
    if gold.exists():
        s = importlib.util.spec_from_file_location("golden_case_config", gold)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        spec, real = m.CaseSpec(mode, tokens), True
    else:  # documented golden values (case_config.py: dense page 128, indexer/reuse 16)
        has_idx = mode != "qkv_dense"
        spec = SimpleNamespace(
            mode=mode,
            tokens=tokens,
            padded_tokens=(tokens + 3) // 4 * 4,
            cache_page_size=16 if has_idx else 128,
            gemm_nk=((9856 if has_idx else 9216), 6144),
        )
        real = False
    return SimpleNamespace(
        spec=spec,
        spec_is_real=real,
        rotary=SimpleNamespace(rotary_dim=64),
        q_norm=norm(),
        k_norm=norm(),
        index_q_norm=norm(),
        index_k_norm=norm(),
        caches={"index_cache": object()},
        weight=SimpleNamespace(shape=((9856 if has_idx else 9216), 6144)),
        weight_scale=SimpleNamespace(shape=((9856 if has_idx else 9216), 1)),
    )


@pytest.mark.parametrize(
    "mode,gm",
    [
        ("qkv_dense", "dense"),
        ("qkv_indexer", "indexer"),
        ("qkv_indexer_reuse", "reuse"),
    ],
)
def test_golden_adapter_site_is_eligible_and_matches_aiter_spec(mode, gm, monkeypatch):
    """Regression: the fail-safe eligibility check made every harness site 'uncovered'
    because the adapter's impl lacked gemm_fusion_prologue_supported. Builds the real
    site via _site_for and compares its 2-stage signature with aiter's own spec builder
    (and the shipped profile) when the aiter worktree is importable."""
    import json
    import sys
    from pathlib import Path

    from atom.model_ops.gemm_fusion import test_utils as GC

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

    for n in (
        "aiter",
        "aiter.gemm_fusion",
        "aiter.ops",
        "aiter.ops.gemm_fusion",
        "aiter.ops.gemm_fusion.sp_qkv",
    ):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
        sys.modules[n].__path__ = []
    spec = load("aiter.gemm_fusion.spec", spec_py)
    sys.modules["aiter.gemm_fusion"].__dict__.update(
        {
            k: getattr(spec, k)
            for k in (
                "FusionSpec",
                "GemmSpec",
                "Field",
                "SPExchange",
                "HeadNormRopeCache",
            )
        }
    )
    specs = load("aiter.ops.gemm_fusion.sp_qkv.specs", specs_py)
    TU.inject_aiter(monkeypatch, sys.modules["aiter.gemm_fusion"])
    monkeypatch.setattr(stages, "_kv_block_size", lambda: 16)
    TU.reset_for_tests()
    GC._SITES.clear()
    site, _, _ = GC._site_for(_golden_case(mode), None)
    assert (
        site._static_prologue_ok() is True
    )  # else every golden cell resolves 'uncovered'
    prof_path = aiter_root / "aiter/configs/gemm_fusion/minimax_m3_sp4_gfx950_v1.json"
    try:
        rows = json.loads(prof_path.read_text())["rows"]
    except (
        ValueError,
        OSError,
    ):  # aiter worktree mid-regeneration (empty/partial file)
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
    assert (
        sig2 == specs.sp_qkv_spec(gm).signature()
    ), f"{mode}: page-{page} signature differs from aiter's"
    assert sig2 in {
        r["sig"] for r in rows if "small" in r["vid"]
    }  # large M: GEMM + SPExchange only (HeadNormRopeCache runs natively in ATOM). Large rows
    # are keyed by this 1-stage signature, so the prefix fallback only works if it matches.
    sig1 = site.spec.prefix(1).signature()
    assert sig1 == specs.sp_qkv_spec(gm).prefix(1).signature()
    large = {r["sig"] for r in rows if "large" in r["vid"]}
    assert (
        sig1 in large
    ), f"{mode}: 1-stage sig {sig1} not among large rows {sorted(large)}"
    assert sig1 != sig2
    GC._SITES.clear()


class _SigMatcher:
    """Fake aiter: returns a plan only when the spec prefix carries a registered signature,
    like match_prefix against an exact-row profile. Records every spec it was asked about.
    """

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

        return SimpleNamespace(
            match_prefix=match_prefix, ContractError=FakeContractError
        )


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
    from atom.model_ops.gemm_fusion import test_utils as GC

    matcher = _SigMatcher([])
    gf = _fake_spec_module(matcher)
    TU.inject_aiter(monkeypatch, gf)
    monkeypatch.setattr(stages, "_kv_block_size", lambda: 16)
    res = FakeResources()
    TU.reset_for_tests(res)
    TU.set_phase("preparing")
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


def test_golden_dense_gate_is_independent_of_the_decode_flag_and_pinned_to_the_real_impl(
    monkeypatch,
):
    """The stand-in dense gate mirrors PagedAttentionImpl.gemm_fusion_prologue_supported: rotary +
    Gemma q_norm + k_norm, and NOTHING about the decode kernel choice (ATOM_FORCE_ATTN_TRITON,
    sliding window, head_dim). Pin both the behaviour and the source so they cannot drift.
    """
    import re
    from pathlib import Path

    from atom.model_ops.gemm_fusion import test_utils as GC

    impl = SimpleNamespace(
        head_dim=128,
        sliding_window=-1,
        rotary_emb=object(),
        q_norm=object(),
        k_norm=object(),
    )
    for env in (
        None,
        "0",
        "1",
    ):  # the production M3 env sets it to 1: eligibility must not change
        if env is None:
            monkeypatch.delenv("ATOM_FORCE_ATTN_TRITON", raising=False)
        else:
            monkeypatch.setenv("ATOM_FORCE_ATTN_TRITON", env)
        assert GC._prologue_supported("dense", impl) is True
    monkeypatch.delenv("ATOM_FORCE_ATTN_TRITON", raising=False)
    # decode-kernel inputs do not matter either
    for ok in ({"head_dim": 64}, {"sliding_window": 128}):
        assert (
            GC._prologue_supported("dense", SimpleNamespace(**{**impl.__dict__, **ok}))
            is True
        )
    # what DOES matter: the prologue's own inputs
    for bad in ({"rotary_emb": None}, {"q_norm": None}, {"k_norm": None}):
        assert (
            GC._prologue_supported("dense", SimpleNamespace(**{**impl.__dict__, **bad}))
            is False
        )
    # sparse modes: unconditional True, exactly like SparseMHAPagedAttentionImpl
    assert GC._prologue_supported("indexer", SimpleNamespace()) is True
    assert GC._prologue_supported("reuse", SimpleNamespace()) is True

    src = (
        Path(__file__).resolve().parents[2] / "atom/model_ops/attention_mha.py"
    ).read_text(encoding="utf-8")
    body = src.split("def gemm_fusion_prologue_supported", 1)[1].split(
        "def gemm_fusion_prologue_state", 1
    )[0]
    code = re.sub(r'""".*?"""', "", body, flags=re.DOTALL)
    for needle in (
        "self.rotary_emb is not None",
        "isinstance(self.q_norm, GemmaRMSNorm)",
        "self.k_norm is not None",
    ):
        assert needle in code, needle
    for gone in (
        "ATOM_FORCE_ATTN_TRITON",
        "sliding_window",
        "head_dim",
        "use_triton_attn",
    ):
        assert gone not in code, f"eligibility must not read {gone}"
    sparse = src.split("class SparseMHAPagedAttentionImpl", 1)[1]
    assert re.search(
        r"def gemm_fusion_prologue_supported\(self\) -> bool:.*?return True",
        sparse,
        re.DOTALL,
    )


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
    s.stages = (
        SimpleNamespace(kind="sp_exchange"),
        SimpleNamespace(
            kind="head_norm_rope_cache",
            kv_cache=f"shuffle_page{declared_page}_fp8_ptok",
        ),
    )
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
        _page_site(make_site, 128, sparse=True).bind_caches(
            _kv(16)
        )  # cannot be coarser than the pool


def test_kv_block_size_has_no_silent_default(monkeypatch):
    """A guessed page would silently miss (or wrongly match) a profile row."""
    import sys
    import types as _t

    cfg = _t.ModuleType("atom.config")
    cfg.get_current_atom_config = lambda: (_ for _ in ()).throw(
        RuntimeError("no config")
    )
    monkeypatch.setitem(sys.modules, "atom.config", cfg)
    with pytest.raises(RuntimeError):
        stages._kv_block_size()
    cfg.get_current_atom_config = lambda: SimpleNamespace(kv_cache_block_size=128)
    assert stages._kv_block_size() == 128


# ----------------- gate04: eager-prepared plan must be THE plan capture/replay launches
class _BindingPlan:
    """Like aiter's SpQkvSmall: pointer bindings live in a per-INSTANCE dict, and a launch
    that needs a binding it does not have raises (the 'new binding required during capture').
    """

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
                raise RuntimeError(
                    "sp_qkv.small: new binding required during graph capture"
                )
            self._bindings[("k",)] = True


def _fresh_plan_matcher():
    """match_prefix returns a NEW plan object on every call, exactly like the real matcher."""
    made = []

    def match_prefix(spec, ctx, profile=None):
        p = _BindingPlan()
        made.append(p)
        return p, 2

    return match_prefix, made


def test_resolve_shares_one_plan_instance_between_eager_and_capture(
    make_site, monkeypatch
):
    res = FakeResources()
    s = make_site(res)
    match_prefix, made = _fresh_plan_matcher()
    sys.modules["aiter.gemm_fusion"].match_prefix = match_prefix
    eager = s.resolve(64, "eager")
    capture = s.resolve(64, "capture")
    assert len(made) == 2  # the matcher really did build two objects...
    assert eager.plan is capture.plan  # ...but the site keeps ONE
    assert eager.plan is made[0]
    # order does not matter
    s2 = make_site(FakeResources())
    sys.modules["aiter.gemm_fusion"].match_prefix = match_prefix
    c2 = s2.resolve(128, "capture")
    e2 = s2.resolve(128, "eager")
    assert c2.plan is e2.plan


def test_prepare_then_capture_launch_never_needs_a_new_binding(make_site):
    """The gate04 failure, reproduced: prepare binds the eager plan; capture used to get a
    different, unbound plan. Now the capture-mode resolution IS the prepared plan."""
    s = make_site(FakeResources())
    match_prefix, _ = _fresh_plan_matcher()
    sys.modules["aiter.gemm_fusion"].match_prefix = match_prefix
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


def test_a_plan_other_than_the_prepared_one_is_downgraded_at_agree_not_raised_per_call(
    make_site,
):
    s = make_site(FakeResources())
    prepared = _BindingPlan()
    st = SimpleNamespace(
        prepared=True,
        plan=prepared,
        rebound=False,
        ptrs=(0,) * 7,
        xs=None,
        pos=None,
        slot=None,
        outs={},
        m=16,
    )
    s._staging[64] = st
    s._by_tokens[(64, "eager")] = site_mod.Resolution(
        _BindingPlan(), 2, "sp_qkv.small.v1", "profile_hit"
    )
    s._by_tokens[(64, "capture")] = site_mod.Resolution(
        prepared, 2, "sp_qkv.small.v1", "profile_hit"
    )
    s.agree_prepared(64, True)
    assert (
        s._by_tokens[(64, "eager")].plan is None
    )  # other instance: native (uniformly, via agree)
    assert (
        s._by_tokens[(64, "capture")].plan is prepared
    )  # the prepared one stays fused


def test_binding_key_names_every_pointer_aiter_keys_on(make_site):
    class _T:
        def __init__(self, p):
            self.p = p

        def data_ptr(self):
            return self.p

    s = make_site(FakeResources())
    s.linear.weight_scale = _T(1)
    s.build_meta = lambda st, state: {
        "q_norm_w": _T(2),
        "positions": _T(3),
        "k_cache": _T(4),
    }
    st = SimpleNamespace(xs=_T(9), outs={"q": _T(5), "k": _T(6)}, m=16)
    key = TU.binding_key(s, st, {})
    assert key == {
        "x_scale": 9,
        "w_scale": 1,
        "q_norm_w": 2,
        "positions": 3,
        "k_cache": 4,
        "out_q": 5,
        "out_k": 6,
        "rows": 16,
    }
    st.outs["q"] = _T(55)  # a per-call output buffer would show up as a differing key
    assert TU.binding_key(s, st, {}) != key


def test_m3_smoke_register_hf_config_is_idempotent_and_keeps_architectures(monkeypatch):
    """Logic only, against a stand-in transformers (the real 5.16.1 is not installed here):
    registers a PretrainedConfig subclass for the model_type once, tolerates a second call,
    and re-raises unrelated registration errors."""
    import sys
    import types as _t

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
    m = TU  # the smoke-model helpers live in test_utils
    cls = m.register_hf_config()
    assert (
        registry["minimax_m3"] is cls
        and cls.model_type == "minimax_m3"
        and issubclass(cls, _Pre)
    )
    m.register_hf_config()  # second call: already registered -> no error
    _Auto.register = staticmethod(
        lambda mt, c: (_ for _ in ()).throw(ValueError("something else"))
    )
    with pytest.raises(ValueError, match="something else"):
        m.register_hf_config("other_type")


def _load_m3_smoke(name=None):
    return TU  # the smoke-model helpers moved into test_utils


def test_m3_smoke_vl_layout_follows_the_released_checkpoint_and_keeps_geometry():
    m = _load_m3_smoke()
    c = m.m3_vl_config(layers=4)
    # outer / nested model types and the entry class the recipes' VL checkpoint selects
    assert (c["model_type"], c["text_config"]["model_type"]) == (
        "minimax_m3_vl",
        "minimax_m3_text",
    )
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


def test_m3_smoke_vl_dir_ships_a_config_class_and_serve_args_trust_it(
    tmp_path_factory=None,
):
    import os
    import sys
    import types as _t
    import uuid
    from pathlib import Path

    m = _load_m3_smoke("m3_smoke_vl_dir")
    d = (
        Path(__file__).resolve().parents[2] / f"_tmp_m3_{uuid.uuid4().hex[:8]}"
    )  # workspace scratch
    try:
        m.make_model_dir(str(d))
        assert sorted(os.listdir(d)) == ["config.json", "configuration_minimax_m3.py"]
        cfg_py = (d / "configuration_minimax_m3.py").read_text(encoding="utf-8")
        assert (
            "class MiniMaxM3Config" in cfg_py
            and 'model_type = "minimax_m3_vl"' in cfg_py
        )

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
            exec(cfg_py, ns)  # noqa: S102 - runs the smoke config module under test
            import json

            raw = json.loads((d / "config.json").read_text(encoding="utf-8"))
            obj = ns["MiniMaxM3Config"](
                **{k: v for k, v in raw.items() if k != "auto_map"}
            )
        finally:
            if saved is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = saved
        assert (
            type(obj.text_config).__name__ == "MiniMaxM3TextConfig"
            and obj.hidden_size == 6144
        )
        assert obj.architectures == ["MiniMaxM3SparseForConditionalGeneration"]
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)
    argv = m.serve_args("D", level=2, profile="p")
    assert (
        "--trust-remote-code" in argv
        and argv[argv.index("--gemm-fusion-profile") + 1] == "p"
    )
    assert "--trust-remote-code" not in m.serve_args("D", trust_remote_code=False)
    # flat layout is still available for the register_hf_config() route
    assert m.m3_config()["model_type"] == "minimax_m3"


# ----------------------------------------------- shutdown must never hang or mask an earlier failure
def test_shutdown_is_bounded_when_release_hangs():
    import threading
    import time

    res = FakeResources()
    gate = threading.Event()
    res.release = (
        lambda: gate.wait()
    )  # a peer never arrives: the barrier would block forever
    TU.reset_for_tests(res)
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
    TU.reset_for_tests(res)
    hooks.register_plan(FakePlan())
    hooks.shutdown(
        timeout_s=2
    )  # does not raise: it would mask the real start-up failure
    assert hooks.resources() is None and not hooks._PLANS


def test_shutdown_normal_path_still_releases_once():
    res = FakeResources()
    calls = []
    res.release = lambda: calls.append(1)
    TU.reset_for_tests(res)
    hooks.shutdown()
    assert calls == [1]
    hooks.shutdown()  # idempotent
    assert calls == [1]


def test_m3_smoke_serve_args_use_a_sparse_compatible_block_size():
    """E2 OFF arm: the CLI default --block-size 16 fails the sparse backend's
    `model_runner.block_size % self.block_size == 0` (block 128). serve_args must pass 128.
    """
    m = _load_m3_smoke("m3_smoke_blocksize")
    argv = m.serve_args("D")
    assert argv[argv.index("--block-size") + 1] == "128"
    assert (
        m.serve_args("D", block_size=256)[
            m.serve_args("D", block_size=256).index("--block-size") + 1
        ]
        == "256"
    )
    with pytest.raises(ValueError, match="multiple of 128"):
        m.serve_args("D", block_size=16)
    # the sparse constant is pinned to the real one in the source
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2] / "atom/model_ops/minimax_m3/sparse_attn.py"
    ).read_text(encoding="utf-8")
    assert f"SPARSE_BLOCK_SIZE = {m.SPARSE_BLOCK_SIZE}" in src
    backends = (
        Path(__file__).resolve().parents[2] / "atom/model_ops/attentions/backends.py"
    ).read_text(encoding="utf-8")
    assert "model_runner.block_size % self.block_size == 0" in backends


def test_m3_smoke_default_is_all_dense_ffn_but_keeps_all_three_attention_types():
    """E2 OFF arm (job-e2_off03): any MoE layer raises a stock TypeError (x_scale). The smoke
    config therefore uses dense MLPs only, and must still cover dense, sparse-indexed and
    skip-index ATTENTION layers with the real attention geometry."""
    m = _load_m3_smoke("m3_smoke_dense_ffn")
    for cfg in (m.m3_config(layers=4), m.m3_vl_config(layers=4)["text_config"]):
        assert cfg["moe_layer_freq"] == [0, 0, 0, 0]  # no MoE layer anywhere
        assert cfg["sparse_attention_config"]["sparse_attention_freq"] == [0, 1, 1, 1]
        assert (
            cfg["num_attention_heads"] * cfg["head_dim"] == 8192
            and cfg["hidden_size"] == 6144
        )
        assert cfg["dense_intermediate_size"] == 6144
    # skip-index layers: decided by ATOM's REAL function, not a copy of its formula (see the test below)
    # explicit opt-in restores the old MoE layout
    assert m.m3_config(layers=4, moe=True)["moe_layer_freq"] == [0, 1, 1, 1]
    assert m.m3_vl_config(layers=4, moe=True)["text_config"]["moe_layer_freq"] == [
        0,
        1,
        1,
        1,
    ]
    assert m.m3_vl_config(layers=3)["text_config"]["moe_layer_freq"] == [0, 0, 0]


def test_stock_moe_forward_still_lacks_x_scale_so_the_dense_default_is_still_needed():
    """Pins the reason for moe=False to the real source. When ATOM fixes MiniMaxM3MoE.forward to
    take x_scale this test FAILS, which is the signal to revisit the default."""
    import re
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2] / "atom/models/minimax_m3/model.py"
    ).read_text(encoding="utf-8")
    assert "hidden_states = ffn(hidden_states, x_scale=hidden_states_scale)" in src
    moe_cls = src.split("class MiniMaxM3MoE", 1)[1].split(
        "class MiniMaxM3Attention", 1
    )[0]
    assert re.search(
        r"def forward\(self, hidden_states: torch\.Tensor\) -> torch\.Tensor:", moe_cls
    )
    assert "x_scale" not in moe_cls
    mlp_cls = src.split("class MiniMaxM3MLP", 1)[1].split("class MiniMaxM3MoE", 1)[0]
    assert "x_scale: torch.Tensor | None = None" in mlp_cls  # the dense path accepts it


def test_m3_smoke_serve_env_sets_the_flag_the_real_m3_recipes_require():
    """E2 job-e2_new_on03: without ATOM_FORCE_ATTN_TRITON=1 stock decode runs the ASM paged-attention
    kernel and aborts (no heuristic kernel for bf16 q / fp8 kv / gqa 16 / block 128). The smoke
    launch must carry the flag, pinned to ATOM's own recipe text, and the coverage consequence
    must stay visible."""
    from pathlib import Path

    m = _load_m3_smoke("m3_smoke_env")
    assert m.serve_env()["ATOM_FORCE_ATTN_TRITON"] == "1"
    assert (
        m.serve_env({"X": "1"})["X"] == "1"
        and m.serve_env()["ATOM_FORCE_ATTN_TRITON"] == "1"
    )
    root = Path(__file__).resolve().parents[2]
    recipe = (root / "recipes/MiniMax-M3-Agentic-InferenceX.md").read_text(
        encoding="utf-8"
    )
    assert "ATOM_FORCE_ATTN_TRITON=1" in recipe and "is not optional" in recipe
    envs_src = (root / "atom/utils/envs.py").read_text(encoding="utf-8")
    assert '"ATOM_FORCE_ATTN_TRITON"' in envs_src
    doc = m.serve_env.__doc__
    assert "dense layers ARE eligible" in doc and "NOT fused" not in doc


# ------------- dense fusion under ATOM_FORCE_ATTN_TRITON: behaviour of the REAL dense methods
def _real_dense_impl(monkeypatch, flag, **attrs):
    """Instantiate a stand-in carrying the REAL ``PagedAttentionImpl`` methods (their source,
    extracted from attention_mha.py) so the eligibility / dispatch-state logic is exercised as
    written, without importing the model stack. ``flag`` is ATOM_FORCE_ATTN_TRITON."""
    import ast
    import sys
    import types as _t
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "atom/model_ops/attention_mha.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PagedAttentionImpl"
    )
    wanted = (
        "_native_use_triton_attn",
        "gemm_fusion_prologue_supported",
        "gemm_fusion_prologue_state",
    )
    # ast.get_source_segment returns each method with its first line un-indented and the rest at
    # class-body indent, so prefixing 4 spaces re-creates a valid class body.
    methods = [
        "    " + ast.get_source_segment(src, fn)
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name in wanted
    ]
    assert len(methods) == len(wanted), "dense impl no longer defines all of " + repr(
        wanted
    )
    code = "class Real:\n" + "\n\n".join(methods) + "\n"

    class _Gemma:  # the isinstance target the real method imports
        pass

    fake_ln = _t.ModuleType("atom.model_ops.layernorm")
    fake_ln.GemmaRMSNorm = _Gemma
    monkeypatch.setitem(sys.modules, "atom.model_ops.layernorm", fake_ln)
    env = SimpleNamespace(ATOM_FORCE_ATTN_TRITON=flag)
    ns = {"envs": env, "ForwardContext": object}
    exec(  # noqa: S102 - runs the real impl source extracted by ast
        compile(code, "attention_mha_extracted", "exec"), ns
    )
    impl = ns["Real"]()
    impl.__dict__.update(
        {
            "sliding_window": -1,
            "head_dim": 128,
            "rotary_emb": object(),
            "q_norm": _Gemma(),
            "k_norm": object(),
            "layer_num": 0,
            "use_triton_attn": None,
            "_cache_format": None,
        },
        **attrs,
    )
    return impl, _Gemma


class _FakeCache:
    def __init__(self, shape, esize=1):
        self.shape, self._e = shape, esize

    def dim(self):
        return len(self.shape)

    def element_size(self):
        return self._e

    def view(self, *shape):
        return ("view", shape)


def _fwd_ctx():
    layer = SimpleNamespace(
        k_cache=_FakeCache((4, 1, 8, 128, 16)),
        v_cache=_FakeCache((4, 1, 128, 128)),
        k_scale="ks",
        v_scale="vs",
    )
    return SimpleNamespace(
        kv_cache_data={"layer_0": layer},
        attn_metadata=SimpleNamespace(slot_mapping="sm"),
    )


@pytest.mark.parametrize("flag", [False, True])
def test_dense_fusion_eligibility_does_not_depend_on_the_decode_flag(monkeypatch, flag):
    impl, _ = _real_dense_impl(monkeypatch, flag)
    assert (
        impl.gemm_fusion_prologue_supported() is True
    )  # M3 production sets the flag: still eligible
    # the prologue's own prerequisites still gate it
    for bad in ({"rotary_emb": None}, {"k_norm": None}, {"q_norm": object()}):
        i2, _ = _real_dense_impl(monkeypatch, flag, **bad)
        assert i2.gemm_fusion_prologue_supported() is False


@pytest.mark.parametrize(
    "flag,sliding,hd,expect_triton",
    [
        (False, -1, 128, False),  # plain ASM-decode configuration
        (
            True,
            -1,
            128,
            True,
        ),  # production M3: ATOM_FORCE_ATTN_TRITON=1 -> Triton decode
        (False, 4096, 128, True),  # sliding window forces Triton too
        (False, -1, 64, True),  # head_dim != 128 forces Triton too
    ],
)
def test_fused_prologue_sets_use_triton_attn_exactly_as_native_and_writes_shuffle(
    monkeypatch, flag, sliding, hd, expect_triton
):
    """The fused path must leave the decode dispatch where native rope_cache leaves it: the
    Triton decode runs on the fused-written SHUFFLE cache under the flag, ASM otherwise.
    """
    impl, _ = _real_dense_impl(monkeypatch, flag, sliding_window=sliding, head_dim=hd)
    state = impl.gemm_fusion_prologue_state(_fwd_ctx())
    assert state is not None
    assert impl.use_triton_attn is expect_triton is impl._native_use_triton_attn()
    assert (
        impl._cache_format == "SHUFFLE"
    )  # same layout native's Gemma branch writes in BOTH modes
    assert state["v_cache_shuffle"] == (
        "view",
        (4, 1, 128 // 16, 128, 16),
    )  # SHUFFLE V view
    assert (
        state["k_cache"].shape == (4, 1, 8, 128, 16) and state["slot_mapping"] == "sm"
    )


def test_native_rope_cache_writes_shuffle_in_the_gemma_branch_whatever_the_flag():
    """Layout evidence, from the real source: in rope_cache the GemmaRMSNorm branch (the one the
    fused prologue replaces) never reads use_triton_attn, calls triton_fused_norm_rope_cache on
    the SHUFFLE V view, and sets _cache_format = 'SHUFFLE'. The NHD branch is the ELIF taken only
    when rotary/q_norm/k_norm are absent, which M3 dense layers never are."""
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "atom/model_ops/attention_mha.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PagedAttentionImpl"
    )
    rope = next(
        f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "rope_cache"
    )
    top_if = next(n for n in rope.body if isinstance(n, ast.If))
    cond = ast.get_source_segment(src, top_if.test)
    assert (
        "self.rotary_emb is not None" in cond
        and "self.q_norm is not None" in cond
        and "self.k_norm is not None" in cond
        and "use_triton_attn" not in cond
    )  # norms present => this branch
    # main added a LONG-BATCH branch in front of the fused one:
    #   if Gemma and tokens > ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS and asm layout: unfused twin
    #   elif Gemma: <- the fused Triton branch the small-M prologue replaces
    long_if = next(
        n
        for n in top_if.body
        if isinstance(n, ast.If)
        and "ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS"
        in ast.get_source_segment(src, n.test)
    )
    assert "_use_asm_cache_layout" in ast.get_source_segment(src, long_if.test)
    assert "_gemma_norm_rope_then_cache" in "\n".join(
        ast.get_source_segment(src, n) for n in long_if.body
    )
    gemma_if = long_if.orelse[0]
    assert isinstance(gemma_if, ast.If)
    assert "isinstance(self.q_norm, GemmaRMSNorm)" in ast.get_source_segment(
        src, gemma_if.test
    )
    gemma_src = "\n".join(ast.get_source_segment(src, n) for n in gemma_if.body)
    assert (
        "triton_fused_norm_rope_cache" in gemma_src
        and "use_triton_attn" not in gemma_src
    )
    assert "v_cache_shuffle" in gemma_src
    assert any(
        isinstance(n, ast.Assign)
        and ast.get_source_segment(src, n) == 'self._cache_format = "SHUFFLE"'
        for n in top_if.body
    )
    # the long-batch twin writes the same SHUFFLE cache (its docstring says so) and is only reachable
    # with an asm cache layout; it sits above the small-M buckets (default threshold 8192 tokens)
    twin = next(
        f
        for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name == "_gemma_norm_rope_then_cache"
    )
    assert "Same SHUFFLE layout" in ast.get_docstring(twin)
    # the NHD layout belongs to the elif branch: a different condition entirely
    elif_if = top_if.orelse[0]
    assert isinstance(elif_if, ast.If) and "use_triton_attn" in ast.get_source_segment(
        src, elif_if.test
    )
    assert 'self._cache_format = "NHD"' in ast.get_source_segment(src, elif_if)


# ------------- E-3 / M-1: the per-call decision is a pure function of rank-uniform inputs
def _decide_inputs(**over):
    base = {
        "resolution": site_mod.Resolution(FakePlan(), 2, "fake.v1", "profile_hit"),
        "is_dummy_run": False,
        "x_dtype_is_fp8": True,
        "x_scale_ok": True,
    }
    base.update(over)
    return base


@pytest.mark.parametrize(
    "over,expect",
    [
        ({}, ("fused", "profile_hit")),
        ({"is_dummy_run": True}, ("native", "dummy_run")),
        ({"x_dtype_is_fp8": False}, ("native", "unquantized_input")),
        ({"x_scale_ok": False}, ("native", "xscale_layout")),
        (
            {"resolution": site_mod.Resolution(None, 0, "native", "uncovered")},
            ("native", "uncovered"),
        ),
        (
            {
                "resolution": site_mod.Resolution(
                    FakePlan(), 1, "fake.large", "profile_hit"
                )
            },
            ("fused", "profile_hit"),
        ),
    ],
)
def test_decide_path_is_the_documented_function_of_its_inputs(over, expect):
    assert site_mod.GemmFusionSite.decide_path(**_decide_inputs(**over)) == expect


def test_decide_path_reads_no_rank_env_device_or_fault_state():
    """Pin purity to the source: the function may not look at hooks / faults / env / torch.cuda."""
    import inspect
    import re

    body = inspect.getsource(site_mod.GemmFusionSite.decide_path)
    code = re.sub(r'""".*?"""', "", body, flags=re.DOTALL)
    for forbidden in (
        "hooks",
        "faults",
        "os.environ",
        "getenv",
        "torch",
        "sp_rank",
        "self.",
    ):
        assert forbidden not in code, forbidden


def test_prepare_has_no_rank_local_raise(make_site):
    """The only raise sites left in the per-call path are the post-launch stream check and the
    bind-time page check; _prepare itself must not raise (E-3)."""
    import inspect
    import re

    code = re.sub(
        r'""".*?"""',
        "",
        inspect.getsource(site_mod.GemmFusionSite._prepare),
        flags=re.DOTALL,
    )
    assert "raise " not in code
    assert "GemmFusionBindError" not in code


# --- staged-mode KV pointer guard at bind time (collective, not per call)
class _P:
    def __init__(self, p):
        self.p = p

    def data_ptr(self):
        return self.p


def _kv_layer(base):
    return {
        "layer_3": SimpleNamespace(
            k_cache=_P(base),
            v_cache=_P(base + 1),
            k_scale=_P(base + 2),
            v_scale=_P(base + 3),
        )
    }


def _staged_prepared_site(make_site, res, base):
    s = make_site(res)
    s.attn = SimpleNamespace(impl=SimpleNamespace(layer_num=3))
    s._staging[64] = SimpleNamespace(prepared=True, kv_ptrs=s.kv_ptrs(_kv_layer(base)))
    hooks._SITES[s.name] = s
    return s


class _T:
    """Torch-free tensor stand-in: the attributes site.py reads (shape/dtype/dim/data_ptr)."""

    _next = [10_000]  # noqa: RUF012 - one counter shared by all instances, on purpose

    def __init__(self, shape=(1,), dtype=None, ptr=None, contiguous=True):
        import torch as _torch

        self.shape, self.dtype = tuple(shape), (
            dtype if dtype is not None else _torch.float32
        )
        self._contiguous = contiguous
        if ptr is None:
            _T._next[0] += 64
            ptr = _T._next[0]
        self.p = ptr

    def dim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self.p


def _fake_direct(make_site, res=None):
    s = make_site(res or FakeResources(plan_n=(FakePlan(), 2)), direct=True)
    s.attn = SimpleNamespace(
        impl=SimpleNamespace(layer_num=3, gemm_fusion_prologue_supported=lambda: True)
    )
    import torch as _torch

    t = {
        "x": _T((16, 3072), _torch.float8_e4m3fn),
        "xs": _T((16, 1), _torch.float32),
        "pos": _T((64,), _torch.int64),
        "slot": _T((64,), _torch.int64),
    }
    state = {
        "slot_mapping": t["slot"],
        "k_cache": _T(),
        "v_cache": _T(),
        "k_scale": _T(),
        "v_scale": _T(),
    }
    ctx = SimpleNamespace(
        is_dummy_run=False,
        real_tokens=64,
        padded_tokens=64,
        positions=t["pos"],
        prologue_state=lambda: state,
    )
    return s, t, state, ctx


def _seed_prepared(s, t, state):
    st = SimpleNamespace(
        prepared=True,
        rebound=False,
        plan=None,
        ptrs=s._live_ptrs(t["xs"], t["pos"], state),
    )
    s._staging[64] = st
    s._by_tokens[(64, "eager")] = site_mod.Resolution(
        FakePlan(), 2, "fake.v1", "profile_hit"
    )
    s._by_tokens[(64, "capture")] = site_mod.Resolution(
        FakePlan(), 2, "fake.v1", "profile_hit"
    )
    return st


def test_direct_mode_per_call_path_never_raises_and_never_falls_back_per_rank(
    make_site,
):
    """M-1 / E-3: the per-call path has no rank-local raise. A moved x_scale is self-healed by
    copy (still fused, uniform). What used to raise per call is decided at prepare time and
    agreed across ranks (agree_prepared -> native on ALL ranks)."""
    s, t, state, ctx = _fake_direct(make_site)
    st = _seed_prepared(s, t, state)
    r, _, _, got = s._prepare(ctx, t["x"], t["xs"])
    assert r.plan is not None and got is st and st.rebound is False
    r, _, _, got = s._prepare(ctx, t["x"], _T((16, 1)))  # x_scale moved
    assert r.plan is not None and got.rebound is True  # fused + copy, not native
    state["k_cache"] = _T()  # KV cache moved AFTER prepare: no per-call raise any more
    r, _, _, _ = s._prepare(ctx, t["x"], t["xs"])
    assert r.plan is not None  # the guard is at prepare time (next test), not here


def test_direct_prepare_reports_moved_kv_so_agree_downgrades_every_rank(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s, t, state, _ctx = _fake_direct(make_site, res)
    _seed_prepared(s, t, state)
    direct = {
        "x_scale": t["xs"],
        "positions": t["pos"],
        "slot_mapping": t["slot"],
        "state": state,
    }
    assert s.prepare_bucket(64, None, {}, None, direct=direct) is True  # same addresses
    moved = dict(state, k_cache=_T())  # the caller's KV cache moved
    assert (
        s.prepare_bucket(64, None, {}, None, direct={**direct, "state": moved}) is False
    )
    s.agree_prepared(64, False)  # what prepare_buckets does with that verdict
    assert (
        s._by_tokens[(64, "eager")].plan is None
        and s._by_tokens[(64, "eager")].reason == "unprepared"
    )
    assert s._by_tokens[(64, "capture")].plan is None


def test_per_call_decision_is_identical_on_every_simulated_rank(make_site, monkeypatch):
    """E-3: N simulated ranks, same inputs -> the same (path, reason, n_fused). Each 'rank' has
    its own site and sp_rank and NO fault seam: nothing but the uniform tuple may vary the
    outcome. Covers every fallback class."""
    import torch as _torch

    fp8, bf16 = _torch.float8_e4m3fn, _torch.bfloat16
    cases = [
        {"dummy": False, "real": 64, "padded": 64, "dtype": fp8, "scale": (16, 1)},
        {"dummy": True, "real": 64, "padded": 64, "dtype": fp8, "scale": (16, 1)},
        {"dummy": False, "real": 40, "padded": 64, "dtype": fp8, "scale": (16, 1)},
        {"dummy": False, "real": 64, "padded": 64, "dtype": bf16, "scale": (16, 1)},
        {"dummy": False, "real": 64, "padded": 64, "dtype": fp8, "scale": (16, 2)},
    ]
    expected = [
        True,
        False,
        True,
        False,
        False,
    ]  # fused for the first and for the PADDED one (40 -> 64)
    for c, want_fused in zip(cases, expected):
        outcomes = set()
        for rank in range(4):
            s, t, state, ctx = _fake_direct(make_site)
            hooks.resources().rank = rank  # the decision must not depend on the rank
            _seed_prepared(s, t, state)
            ctx.is_dummy_run, ctx.real_tokens, ctx.padded_tokens = (
                c["dummy"],
                c["real"],
                c["padded"],
            )
            r, reason, _, _ = s._prepare(
                ctx, _T((16, 3072), c["dtype"]), _T(c["scale"])
            )
            outcomes.add((r.plan is not None, r.n_fused, reason))
        assert len(outcomes) == 1, (c, outcomes)  # every rank decided identically
        assert next(iter(outcomes))[0] is want_fused, c


def _no_prepare(monkeypatch):
    monkeypatch.setattr(hooks, "prepare_buckets", lambda *a, **k: None)
    monkeypatch.setattr(hooks, "profile_tokens", lambda *a, **k: [])


def test_rebind_with_moved_kv_fails_before_state_changes(make_site, monkeypatch):
    _no_prepare(monkeypatch)
    res = FakeResources()
    s = _staged_prepared_site(make_site, res, 1000)
    hooks._KV = old = _kv_layer(1000)
    hooks.bind_kv_cache(_kv_layer(1000))  # same addresses: fine
    hooks._KV = old
    with pytest.raises(site_mod.GemmFusionBindError, match="different addresses"):
        hooks.bind_kv_cache(_kv_layer(2000))  # a re-allocated pool
    assert hooks._KV is old  # nothing was rebound
    assert s.kv_moved(_kv_layer(2000)) is True and s.kv_moved(_kv_layer(1000)) is False


def test_rebind_guard_is_a_collective_so_one_ranks_move_fails_all_ranks(
    make_site, monkeypatch
):
    """Rank 0's pool did not move, rank 1's did. Both must raise (agreed via the reduce)."""
    _no_prepare(monkeypatch)
    outcomes = []
    for local_moved, peer_moved in ((0, 1), (1, 0)):
        res = FakeResources()
        _staged_prepared_site(make_site, res, 1000)
        hooks._KV = _kv_layer(1000)
        res._reduce_min = lambda v, p=peer_moved: [
            min(v[0], -p),
            v[1],
            v[2],
        ]  # MIN over ranks
        try:
            hooks.bind_kv_cache(_kv_layer(2000 if local_moved else 1000))
            outcomes.append("ok")
        except site_mod.GemmFusionBindError:
            outcomes.append("raised")
    assert outcomes == ["raised", "raised"]


def test_first_bind_runs_no_collective(make_site, monkeypatch):
    _no_prepare(monkeypatch)
    res = FakeResources()
    calls = []
    res._reduce_min = lambda v: calls.append(v) or v
    s = make_site(res)
    s.attn = SimpleNamespace(impl=SimpleNamespace(layer_num=3))
    hooks._SITES[s.name] = s
    hooks._KV = None
    hooks.bind_kv_cache(
        _kv_layer(1000)
    )  # first bind: nothing prepared, no agreement needed
    assert calls == []


# ------------- C-S1: a failed fused step must not publish KV or return output as valid
def _runner_tree():
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2] / "atom/model_engine/model_runner.py"
    ).read_text(encoding="utf-8")
    return src, ast.parse(src)


def _function(tree, cls_name, fn_name):
    import ast

    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == fn_name
    )


def _calls(fn, attr):
    import ast

    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]


def test_every_forward_path_checks_the_step_before_publishing_kv_ready():
    """Reviewer C-S1, for the verdict design. Every ``_record_kv_cache_ready`` is reached only when
    the step's check returned no verdict: in the two ``ModelRunner.forward`` returns it sits under
    ``if aborted is None:`` (``aborted`` is the check's result), and ``prefill_forward`` (plain token
    ids, no output object) raises before it on any verdict. A step with a verdict therefore never
    tells a transfer connector its KV is ready."""
    import ast

    _src, tree = _runner_tree()
    fwd = _function(tree, "ModelRunner", "forward")
    assert len(_calls(fwd, "_gemm_fusion_check_step")) == 2  # one per return path
    for call in _calls(fwd, "_record_kv_cache_ready"):
        guards = [
            n
            for n in ast.walk(fwd)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "aborted is None"
            and any(call in ast.walk(s) for s in n.body)
        ]
        assert guards, f"KV-ready at line {call.lineno} is not guarded by the verdict"
    # every ModelRunner.forward return hands its output through the verdict
    for ret in [n for n in ast.walk(fwd) if isinstance(n, ast.Return)]:
        if ret.value is not None and "ScheduledBatchOutput" in ast.unparse(ret.value):
            assert "_apply_gemm_fusion_verdict" in ast.unparse(ret.value)
    # the PD prefill path: check, raise on a verdict, only then publish
    pre = _function(tree, "RapidServeModelRunner", "prefill_forward")
    (ready,) = _calls(pre, "_record_kv_cache_ready")
    guard = next(
        s
        for s in pre.body
        if isinstance(s, ast.If) and "_gemm_fusion_check_step" in ast.unparse(s.test)
    )
    assert any(isinstance(n, ast.Raise) for n in ast.walk(guard))
    assert guard.lineno < ready.lineno
    # the RapidServe decode forward also passes its output through the verdict
    rs = _function(tree, "RapidServeModelRunner", "forward")
    assert "_apply_gemm_fusion_verdict" in ast.unparse(rs)
    assert len(_calls(rs, "_gemm_fusion_check_step")) == 1


def test_the_runner_wrapper_swallows_nothing_so_fatal_errors_still_end_the_worker():
    """A verdict is a return value now, not an exception. What is still an exception is fatal
    (a failed recovery, a hung reset, a verdict in a PD prefill, a wedged readback): the wrapper
    must not catch it, and the worker loop does not either, so the worker dies and the engine
    shuts down instead of answering from a failed step."""
    from pathlib import Path

    src, tree = _runner_tree()
    import ast

    wrapper = ast.get_source_segment(
        src, _function(tree, "ModelRunner", "_gemm_fusion_check_step")
    )
    assert "except" not in wrapper and "try:" not in wrapper
    busy = (
        Path(__file__).resolve().parents[2] / "atom/model_engine/async_proc.py"
    ).read_text(encoding="utf-8")
    loop = busy[busy.index("def busy_loop") : busy.index("def _run_generic_rpc")]
    # Main's busy_loop has a generic-RPC path (RpcPayload calls only: it turns an outcome into a
    # reply) and a barrier handler. The model step is a PLAIN rpc (payload None): its call is not
    # inside any try, and the only handler in the loop is the barrier's, which re-raises when
    # payload is None.
    assert loop.count("except") == 1 and "except threading.BrokenBarrierError:" in loop
    assert "if payload is None:\n                            raise" in loop
    plain = loop[loop.index("func = getattr(runner, func_name, None)") :]
    plain = plain[: plain.index("if need_barrier")]
    assert "out = func(*call_args)" in plain and "try:" not in plain


# ------------- C-S4: nothing returned or stored by the op may alias a shared window
class _Span:
    def __init__(self, ptr, nbytes):
        self._p, self._n = ptr, nbytes

    def data_ptr(self):
        return self._p


def _inject_aiter_test_utils(monkeypatch, helper):
    """Stand in for ``aiter.ops.gemm_fusion.sp_qkv.test_utils`` (aiter's own alias helper)."""
    import types

    class WindowAliasError(RuntimeError):
        pass

    mod = types.ModuleType("aiter.ops.gemm_fusion.sp_qkv.test_utils")
    mod.assert_no_window_alias = helper
    mod.WindowAliasError = WindowAliasError
    monkeypatch.setitem(sys.modules, "aiter.ops.gemm_fusion.sp_qkv.test_utils", mod)
    return mod


def test_alias_check_calls_aiters_test_utils_helper_with_its_real_signature(
    make_site, monkeypatch
):
    """aiter's helper is ``test_utils.assert_no_window_alias(res, obj, what=<str>, force=<bool>)``
    (moved out of FusionResources). ATOM passes the wrapper's ``.inner`` as ``res``, a dict keyed by
    label, and force=True; aiter's WindowAliasError is re-raised as GemmFusionBindError.
    """
    res = FakeResources()
    make_site(res)
    seen = {}

    def helper(res_, obj, what="gemm_fusion output", force=False):
        seen.update(res=res_, obj=obj, what=what, force=force)

    _inject_aiter_test_utils(monkeypatch, helper)
    t_out, t_scale, t_iq = (
        _Span(0x40000, 0x100),
        _Span(0x41000, 0x10),
        _Span(0x42000, 0x20),
    )
    TU.check_no_window_alias(
        [("out", t_out), ("scale", t_scale), ("impl._index_q", t_iq), ("skipped", None)]
    )
    assert (
        seen["res"] is res.inner
        and seen["force"] is True
        and isinstance(seen["what"], str)
    )
    assert seen["obj"] == {
        "out": t_out,
        "scale": t_scale,
        "impl._index_q": t_iq,
    }  # None entries dropped

    def boom(res_, obj, what="", force=False):
        raise RuntimeError("out points into a CCO window")

    _inject_aiter_test_utils(monkeypatch, boom)
    with pytest.raises(
        site_mod.GemmFusionBindError, match="out points into a CCO window"
    ):
        TU.check_no_window_alias([("out", _Span(0x10800, 0x100))])


def test_install_alias_check_wraps_the_op_and_checks_its_outputs(monkeypatch):
    import types

    fake_ops = types.ModuleType("atom.model_ops.gemm_fusion.ops")
    out, scale = _Span(0x1000, 8), _Span(0x2000, 4)
    fake_ops.run_qkv_attention = lambda attn, x, xs, pos: ((out, scale), "decision")
    monkeypatch.setitem(sys.modules, "atom.model_ops.gemm_fusion.ops", fake_ops)
    checked = []
    monkeypatch.setattr(
        TU, "check_no_window_alias", lambda named: checked.append(dict(named))
    )
    attn = SimpleNamespace(impl=SimpleNamespace(_index_q=_Span(0x3000, 2)))
    TU.install_alias_check()
    TU.install_alias_check()  # idempotent: one wrapper, one check per call
    result = fake_ops.run_qkv_attention(attn, None, None, None)
    assert result == ((out, scale), "decision")
    assert len(checked) == 1 and set(checked[0]) == {"out", "scale", "impl._index_q"}
    TU.uninstall_faults()
    checked.clear()
    fake_ops.run_qkv_attention(attn, None, None, None)
    assert checked == []  # uninstalled: the op is the original again


def test_alias_check_is_opt_in_through_the_environment_only(monkeypatch):
    import types

    fake_ops = types.ModuleType("atom.model_ops.gemm_fusion.ops")

    def orig(attn, x, xs, pos):
        return ((None, None), "d")

    fake_ops.run_qkv_attention = orig
    monkeypatch.setitem(sys.modules, "atom.model_ops.gemm_fusion.ops", fake_ops)
    monkeypatch.delenv(TU.CHECK_ALIAS_ENV, raising=False)
    TU.install_from_env()
    assert fake_ops.run_qkv_attention is orig  # not installed without the env var
    monkeypatch.setenv(TU.CHECK_ALIAS_ENV, "1")
    TU.install_from_env()
    assert fake_ops.run_qkv_attention is not orig


# ------------- skip-index layers in the smoke config, decided by ATOM's REAL functions
def _real_skip_functions():
    """Extract ATOM's own skip-index functions from model.py (ast) and run them in pure Python. They
    only read attributes of a config object, so no torch/transformers is needed."""
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2] / "atom/models/minimax_m3/model.py"
    ).read_text(encoding="utf-8")
    want = {
        "_sparse_attention_layer_ids",
        "_sparse_attention_layer_ordinals",
        "_should_skip_minimax_m3_index_topk",
    }
    code = "\n\n".join(
        ast.get_source_segment(src, n)
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name in want
    )
    assert (
        code.count("def ") == 3
    ), "model.py no longer defines the three skip functions"
    ns = {"PretrainedConfig": object}
    exec(  # noqa: S102 - runs ATOM's real skip functions extracted by ast
        compile(code, "model_extracted", "exec"), ns
    )
    return ns["_should_skip_minimax_m3_index_topk"]


def _skip_flags(cfg_text, outer=None):
    """Per-layer skip flags after the SAME outer -> text copy-down ATOM's
    _normalize_minimax_m3_text_config performs (model.py/config.py)."""
    from types import SimpleNamespace as NS

    should_skip = _real_skip_functions()
    text = NS(**cfg_text)
    for a in (
        "use_index_cache",
        "index_topk_freq",
        "index_topk_pattern",
        "index_skip_topk_offset",
    ):
        v = (outer or {}).get(a)
        if v is not None:
            setattr(text, a, v)
    return [should_skip(text, i)[0] for i in range(cfg_text["num_hidden_layers"])]


def test_smoke_config_has_a_skip_index_sparse_layer_under_atoms_real_function():
    """A-4 level 0 found NO skip-index site: the real function returns 'no skip' for every layer
    unless use_index_cache is true (model.py:88). The smoke config must produce at least one skip
    layer, with dense, indexed and skip layers all present, through the real released layout's
    outer->text copy-down."""
    m = _load_m3_smoke("m3_smoke_skip")
    vl = m.m3_vl_config(layers=4)
    outer = {
        k: vl[k]
        for k in ("use_index_cache", "index_topk_freq", "index_skip_topk_offset")
        if k in vl
    }
    assert (
        vl["use_index_cache"] is True and "use_index_cache" not in vl["text_config"]
    )  # outer, as the recipes set it
    flags = _skip_flags(vl["text_config"], outer)
    sparse = vl["text_config"]["sparse_attention_config"]["sparse_attention_freq"]
    assert flags == [
        False,
        False,
        True,
        False,
    ]  # layer 0 dense; 1 indexed; 2 SKIP; 3 indexed
    kinds = [
        "dense" if not sparse[i] else ("skip" if flags[i] else "indexed")
        for i in range(4)
    ]
    assert set(kinds) == {"dense", "indexed", "skip"}
    # the flat layout carries the flag on the one config object
    flat = m.m3_config(layers=4)
    assert flat["use_index_cache"] is True and _skip_flags(flat) == flags


def test_without_use_index_cache_no_layer_skips_which_is_the_bug_this_fixes():
    """Control: the earlier smoke config (use_index_cache absent) gives all False under the REAL
    function even with index_topk_freq 2, so a test on a formula copy could not see the bug.
    """
    m = _load_m3_smoke("m3_smoke_skip_control")
    vl = m.m3_vl_config(layers=4)
    outer = {
        k: vl[k] for k in ("index_topk_freq", "index_skip_topk_offset")
    }  # use_index_cache missing
    assert _skip_flags(vl["text_config"], outer) == [False, False, False, False]
    # and a freq of 1 means "never share", even with the cache on
    flat = m.m3_config(layers=4, index_topk_freq=1)
    assert _skip_flags(flat) == [False, False, False, False]


def test_skip_layers_scale_with_depth_and_frequency_in_the_smoke_config():
    m = _load_m3_smoke("m3_smoke_skip_scale")
    flat = m.m3_config(layers=8, index_topk_freq=2)
    assert _skip_flags(flat) == [
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]  # ordinals 1,3,5 skip
    # freq 4 (the recipes' value) over sparse ordinals 0,1,2: only ordinal 0 recomputes its indexer,
    # ordinals 1 and 2 share it. I first expected NO skip here; the real function says otherwise
    # (max(ordinal - offset, 0) % freq != 0), which is why these tests run it instead of a formula copy.
    assert _skip_flags(m.m3_config(layers=4, index_topk_freq=4)) == [
        False,
        False,
        True,
        True,
    ]


# ------------- staged (runtime) vs direct (test_utils) input handling
class _Rec:
    """Records copy_ calls; slicing returns a view that records into the same log."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def copy_(self, src):
        self.log.append((self.name, "copy", src))

    def __getitem__(self, idx):
        return _Rec(
            (
                f"{self.name}[{idx.stop}]"
                if idx.stop is not None
                else f"{self.name}[{idx.start}:]"
            ),
            self.log,
        )

    def zero_(self):
        self.log.append((self.name, "zero"))

    def fill_(self, v):
        self.log.append((self.name, "fill", v))


def _inputs_fixture(make_site, direct):
    s = make_site(FakeResources(plan_n=(FakePlan(), 2)), direct=direct)
    log = []
    st = SimpleNamespace(
        xs=_Rec("xs", log), pos=_Rec("pos", log), slot=_Rec("slot", log), rebound=False
    )
    state = {"slot_mapping": _Rec("sm", log)}
    ctx = SimpleNamespace(
        real_tokens=40, padded_tokens=40, positions=_Rec("positions", log)
    )
    return s, st, state, ctx, log


def test_the_runtime_site_always_copies_the_per_call_inputs_into_its_staging_buffers(
    make_site,
):
    s, st, state, ctx, log = _inputs_fixture(make_site, direct=False)
    s._stage_inputs("X_SCALE", ctx, state, st)
    assert [e[0] for e in log] == [
        "xs",
        "pos[40]",
        "slot[40]",
    ]  # x_scale whole, positions/slot [:real]
    assert log[0][2] == "X_SCALE"


def test_the_direct_site_copies_only_when_an_input_moved(make_site):
    s, st, state, ctx, log = _inputs_fixture(make_site, direct=True)
    s._stage_inputs("X_SCALE", ctx, state, st)
    assert log == []  # unmoved: st.xs/pos/slot ARE the caller's tensors
    st.rebound = True
    s._stage_inputs("X_SCALE", ctx, state, st)
    assert [e[0] for e in log] == ["xs", "pos[40]", "slot[40]"]  # self-heal by copy


def test_seams_leave_proof_in_the_log_that_they_were_installed_and_fired(
    monkeypatch, caplog
):
    """Spawned workers must PROVE the seams ran, not assume the env activated them: install_from_env
    always logs one summary line (its absence means it was never called), each installed kind logs
    once, a fault logs once when it first takes effect, and stats() reports the same."""
    import logging

    _ready_hooks(FakeResources())
    monkeypatch.setenv(TU.FAULT_ENV, "error_word")
    monkeypatch.setenv(TU.RECOVER_TIMEOUT_ENV, "5")
    with caplog.at_level(logging.WARNING, logger="atom.gemm_fusion"):
        TU.install_from_env()
        assert _wrapper_error_word() == 1
        assert _wrapper_error_word() == 1  # fires twice, logs once
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert (
        "install_from_env ran" in text
        and "faults=error_word" in text
        and "RECOVER_TIMEOUT_S=5" in text
    )
    assert text.count("fault 'error_word' installed") == 1
    assert text.count("fault 'error_word' FIRED") == 1
    s = TU.stats()
    assert (
        s["fired"] == ["error_word"]
        and "error_word" in s["faults"]
        and s["alias_checks"] == 0
    )
    TU.uninstall_faults()
    assert TU.stats()["fired"] == [] and TU.stats()["faults"] == {}


def test_install_from_env_logs_even_when_the_environment_selects_nothing(
    monkeypatch, caplog
):
    import logging

    for env in (
        TU.FAULT_ENV,
        TU.RECOVER_TIMEOUT_ENV,
        TU.SHUTDOWN_TIMEOUT_ENV,
        TU.CHECK_ALIAS_ENV,
    ):
        monkeypatch.delenv(env, raising=False)
    with caplog.at_level(logging.WARNING, logger="atom.gemm_fusion"):
        TU.install_from_env()
    assert any(
        "nothing selected by the environment" in r.getMessage() for r in caplog.records
    )


# ------------- task-265: the site refuses a weight or activation aiter's plan cannot take as-is
@pytest.mark.parametrize(
    "n_fused", [2, 1]
)  # both plans read w: the check is not 2-stage only
@pytest.mark.parametrize(
    "kw,reason",
    [
        (
            {"shuffled": False},
            "weight_layout",
        ),  # ATOM_USE_TRITON_GEMM=1: stored unshuffled
        ({"shuffled": None}, "weight_layout"),  # no marker at all
        ({"k_cols": 6144}, "weight_shape"),  # weight K != input_size
        ({"rows": 10000}, "weight_shape"),  # fewer rows than N
        ({"rows": 10368}, "weight_shape"),  # more rows, but not a padded weight
        (
            {"rows": 10000, "padded": True},
            "weight_shape",
        ),  # padded weights still need >= N rows
        ({"ws_rows": 10000}, "weight_shape"),  # w_scale with fewer than N rows
    ],
)
def test_static_weight_checks_make_the_bucket_native_and_are_agreed(
    make_site, kw, reason, n_fused
):
    res = FakeResources(plan_n=(FakePlan(), n_fused))
    s = make_site(res)
    s.linear = _fake_linear(10240, 3072, **kw)
    r = s.resolve(64, "eager")
    assert (r.plan, r.n_fused, r.reason) == (None, 0, reason)
    assert (
        res.agree_calls == 1
    )  # the collective still ran on this rank: the verdict is AGREED, not local


def test_a_padded_weight_with_extra_rows_is_still_fused(make_site):
    """is_output_padded: N padded up to the GEMM tile for the preshuffle kernel. aiter reads exactly N
    rows (a 16-row-block shuffle keeps the first N rows contiguous), so more rows than N is fine.
    """
    s = make_site(FakeResources(plan_n=(FakePlan(), 2)))
    s.linear = _fake_linear(10240, 3072, rows=10368, padded=True, ws_rows=10368)
    r = s.resolve(64, "eager")
    assert r.plan is not None and r.reason == "profile_hit"


def test_the_weight_check_names_the_layout_before_the_prologue_reason(make_site):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    s.attn = SimpleNamespace(
        impl=SimpleNamespace(gemm_fusion_prologue_supported=lambda: False)
    )
    s.linear = _fake_linear(10240, 3072, shuffled=False)
    assert s.resolve(64, "eager").reason == "weight_layout"
    s2 = make_site(FakeResources(plan_n=(FakePlan(), 2)))
    s2.attn = SimpleNamespace(
        impl=SimpleNamespace(gemm_fusion_prologue_supported=lambda: False)
    )
    assert (
        s2.resolve(64, "eager").reason == "prologue_unsupported"
    )  # unchanged when the weight is fine


def test_decide_path_rejects_a_k_mismatch_after_the_other_input_facts():
    kw = {
        "resolution": site_mod.Resolution(FakePlan(), 2, "fake.v1", "profile_hit"),
        "is_dummy_run": False,
        "x_dtype_is_fp8": True,
        "x_scale_ok": True,
    }
    decide = site_mod.GemmFusionSite.decide_path
    assert decide(**kw) == (
        "fused",
        "profile_hit",
    )  # x_k_ok defaults to True: existing callers unchanged
    assert decide(**kw, x_k_ok=False) == ("native", "x_k_mismatch")
    assert decide(**{**kw, "x_scale_ok": False}, x_k_ok=False) == (
        "native",
        "xscale_layout",
    )
    assert decide(**{**kw, "is_dummy_run": True}, x_k_ok=False) == (
        "native",
        "dummy_run",
    )


def test_a_call_whose_x_k_differs_from_the_layer_is_native_on_every_simulated_rank(
    make_site,
):
    import torch as _torch

    fp8 = _torch.float8_e4m3fn
    for k, want_fused in ((3072, True), (2048, False), (6144, False)):
        outcomes = set()
        for rank in range(4):
            s, t, state, ctx = _fake_direct(make_site)
            hooks.resources().rank = rank
            _seed_prepared(s, t, state)
            r, reason, _, _ = s._prepare(ctx, _T((16, k), fp8), _T((16, 1)))
            outcomes.add((r.plan is not None, reason))
        assert len(outcomes) == 1, (k, outcomes)  # identical on every rank
        got = next(iter(outcomes))
        assert got[0] is want_fused and (want_fused or got[1] == "x_k_mismatch"), (
            k,
            got,
        )
    assert (
        TU.fallback_reason(s, ctx, _T((16, 2048), fp8), _T((16, 1))) == "x_k_mismatch"
    )  # the test view agrees


# ------------- task-265: padded rows run the fused kernel (small-M plan reads all T rows)
def test_pad_rows_are_rewritten_to_position_0_slot_minus_1_on_every_call(make_site):
    for direct in (False, True):
        s, st, state, ctx, log = _inputs_fixture(make_site, direct=direct)
        st.rebound = True  # (direct site) an input moved: the copy path runs
        ctx.padded_tokens = 64  # 40 real rows of a 64-row bucket
        s._stage_inputs("X_SCALE", ctx, state, st)
        assert [e[0] for e in log] == [
            "xs",
            "pos[40]",
            "slot[40]",
            "pos[40:]",
            "slot[40:]",
        ]
        assert log[3][1:] == ("zero",) and log[4][1:] == (
            "fill",
            -1,
        )  # the decode pad-row values
        log.clear()
        ctx.padded_tokens = 40  # no padding in this call: nothing to rewrite
        s._stage_inputs("X_SCALE", ctx, state, st)
        assert [e[0] for e in log] == ["xs", "pos[40]", "slot[40]"]


class _Arr:
    def __init__(self, shape, dtype, fill):
        self.shape, self.dtype, self.fill = tuple(shape), dtype, fill


def test_staging_buffers_are_t_long_and_start_as_pad_rows(make_site, monkeypatch):
    """The kernel reads positions/slot_mapping for all T = padded tokens: never shorter, and never
    uninitialised (position 0, slot -1 is exactly what a pad row must hold)."""
    fake = SimpleNamespace(
        empty=lambda shape, dtype=None, device=None: _Arr(shape, dtype, None),
        zeros=lambda shape, dtype=None, device=None: _Arr(shape, dtype, 0),
        full=lambda shape, v, dtype=None, device=None: _Arr(shape, dtype, v),
        float32="f32",
        int64="i64",
    )
    monkeypatch.setattr(site_mod, "torch", fake)
    s = make_site(FakeResources())
    s.attn = SimpleNamespace(
        impl=SimpleNamespace(num_heads=16, head_dim=128, num_kv_heads=1)
    )
    st = s._stage(64, "dev", "bf16")
    assert st.pos.shape == (64,) and st.pos.fill == 0 and st.pos.dtype == "i64"
    assert st.slot.shape == (64,) and st.slot.fill == -1 and st.slot.dtype == "i64"
    assert (
        st.xs.shape == (16, 1) and st.outs["q"].shape[0] == 64
    )  # outputs are T rows too


def test_the_post_prologue_attention_drops_pad_rows_like_native(monkeypatch):
    """q/k/v/index_q are sliced to the real rows before attention, and the output is re-padded to the
    SP-divisible row count (finish_attention): what native ``attend_packed`` does for the packed path.
    """
    import types

    calls = []
    ulysses = types.ModuleType("atom.distributed.ulysses_sp")
    ulysses.finish_attention = (
        lambda out, padded, real, tf: calls.append((out, padded, real, tf)) or "RESULT"
    )
    attn8 = types.ModuleType("atom.model_ops.minimax_m3.attention_fp8")
    attn8.gather_heads_fp8 = "GATHER"
    monkeypatch.setitem(sys.modules, "atom.distributed.ulysses_sp", ulysses)
    monkeypatch.setitem(sys.modules, "atom.model_ops.minimax_m3.attention_fp8", attn8)
    from atom.model_ops.gemm_fusion import stages_native

    seen = {}
    impl = SimpleNamespace(
        forward_after_prologue=lambda q, k, v, kc, vc, ks, vs, fc, index_q=None: seen.update(
            q=q, k=k, v=v, iq=index_q
        )
        or "ATTN"
    )
    ctx = SimpleNamespace(
        real_tokens=61, padded_tokens=64, attn=SimpleNamespace(impl=impl), fwd_ctx="F"
    )
    log = []
    outs = {
        "q": _Rec("q", log),
        "k": _Rec("k", log),
        "v": _Rec("v", log),
        "index_q": _Rec("iq", log),
    }
    out = stages_native.attend_after_prologue(
        None, outs, {"k_cache": 1, "v_cache": 2}, ctx
    )
    assert out == "RESULT"
    assert (seen["q"].name, seen["k"].name, seen["v"].name, seen["iq"].name) == (
        "q[61]",
        "k[61]",
        "v[61]",
        "iq[61]",
    )
    assert calls == [("ATTN", 64, 61, "GATHER")]  # re-padded to 64 rows, like native
    # indexer layers have no k/v outputs
    seen.clear()
    stages_native.attend_after_prologue(
        None,
        {"q": _Rec("q", log), "index_q": _Rec("iq", log)},
        {"k_cache": 1, "v_cache": 2},
        ctx,
    )
    assert seen["k"] is None and seen["v"] is None and seen["iq"].name == "iq[61]"


def test_native_packed_path_also_drops_the_pad_rows_it_is_pinned_in_the_source():
    """``attend_packed`` (the 1-stage large-M remainder) slices packed[:real] and ``finish_attention``
    re-pads: pinned to the source so the two entry depths cannot drift apart."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2] / "atom/distributed/ulysses_sp.py"
    ).read_text(encoding="utf-8")
    assert (
        "if real_tokens < padded_tokens:\n        packed = packed[:real_tokens]" in src
    )
    assert "padded_tokens - real_tokens" in src and "out.new_zeros" in src


def test_direct_site_binds_its_own_t_long_arrays_when_the_callers_are_shorter(
    make_site,
):
    import torch as _torch

    for caller_len, own in ((61, True), (64, False)):
        s, t, state, _ctx = _fake_direct(make_site)
        staged = SimpleNamespace(
            xs=None,
            pos=_T((64,), _torch.int64),
            slot=_T((64,), _torch.int64),
            prepared=False,
            plan=None,
            kv_ptrs=None,
            m=16,
            ptrs=None,
            rebound=False,
            own_token_arrays=False,
        )
        s._stage = (
            lambda tokens, device, dtype, _st=staged, _s=s: _s._staging.setdefault(
                tokens, _st
            )
        )
        s._bind_plan = lambda *a, **k: True
        caller_pos, caller_slot = _T((caller_len,), _torch.int64), _T(
            (caller_len,), _torch.int64
        )
        direct = {
            "x_scale": t["xs"],
            "positions": caller_pos,
            "slot_mapping": caller_slot,
            "state": state,
        }
        assert s.prepare_bucket(64, None, {}, None, direct=direct) is True
        assert staged.own_token_arrays is own
        if (
            own
        ):  # the caller's 61-long arrays would be read out of bounds: bind the site's 64-long ones
            assert staged.pos.shape == (64,) and staged.slot.shape == (64,)
            assert (
                staged.ptrs[1] == staged.pos.data_ptr()
                and staged.ptrs[2] == staged.slot.data_ptr()
            )
        else:  # full-length caller arrays are bound as-is
            assert staged.pos is caller_pos and staged.slot is caller_slot


def test_direct_site_always_copies_for_a_padded_bucket(make_site):
    s, t, state, ctx = _fake_direct(make_site)
    st = _seed_prepared(s, t, state)
    st.own_token_arrays = True  # bound to site-owned arrays: the caller's tensors never ARE the bound ones
    fp8 = __import__("torch").float8_e4m3fn
    r, _reason, _, got = s._prepare(ctx, _T((16, 3072), fp8), t["xs"])
    assert r.plan is not None and got.rebound is True
    st.own_token_arrays = False
    _, _, _, got = s._prepare(ctx, _T((16, 3072), fp8), t["xs"])
    assert got.rebound is False  # unpadded and unmoved: no copy


def test_golden_adapter_returns_only_the_real_rows_of_a_padded_case(monkeypatch):
    import types

    log = []
    monkeypatch.setitem(sys.modules, "operators", types.ModuleType("operators"))
    monkeypatch.setattr(TU, "_site_for", lambda c, comm: ("S", "H", "A"))
    monkeypatch.setattr(TU, "_ctx", lambda c, a: "CTX")
    monkeypatch.setattr(
        TU,
        "run_prologue",
        lambda *a, **k: ("outs", {"q": _Rec("q", log), "k": _Rec("k", log)}, "D"),
    )
    for tokens, padded in ((61, 64), (64, 64)):
        case = SimpleNamespace(
            spec=SimpleNamespace(tokens=tokens, padded_tokens=padded, mode="qkv_dense"),
            x="X",
            x_scale="XS",
            caches={"k_cache": "KC"},
        )
        out = TU.qkv_proj_all2all_norm_rope_cache(case, SimpleNamespace(rank=0))
        if tokens < padded:
            assert (out["q"].name, out["k"].name) == (
                "q[61]",
                "k[61]",
            )  # baseline shape: real rows only
        else:
            assert (out["q"].name, out["k"].name) == ("q", "k")  # untouched, no slicing
        assert out["k_cache"] == "KC"


def test_golden_prepare_cases_prepares_padded_cases_instead_of_skipping_them(
    monkeypatch,
):
    calls = []
    fake_site = SimpleNamespace(
        resolve=lambda *a, **k: None,
        prepare_bucket=lambda *a, **k: calls.append((a, k)) or True,
        agree_prepared=lambda *a: None,
    )
    monkeypatch.setattr(TU, "_site_for", lambda c, comm: (fake_site, None, None))
    TU.reset_for_tests(FakeResources())
    case = SimpleNamespace(
        spec=SimpleNamespace(tokens=61, padded_tokens=64),
        x=SimpleNamespace(device="dev"),
        x_scale="XS",
        positions="POS",
        slot_mapping="SLOT",
        caches={"k_cache": 1, "v_cache": 2, "k_scale": 3, "v_scale": 4},
    )
    TU.prepare_cases([case], SimpleNamespace(rank=0))
    assert len(calls) == 1
    args, kw = calls[0]
    assert args[0] == 64 and "pad_rows" not in kw  # no skip, no flag
    assert kw["direct"]["positions"] == "POS" and kw["direct"]["slot_mapping"] == "SLOT"


# ------------- task-265: profile-wide prebuild at init
def _init_with_fake_gf(monkeypatch, res, **gf_attrs):
    calls = []
    gf = SimpleNamespace(
        load_profile=lambda name: calls.append(("load", name)), **gf_attrs
    )
    TU.inject_aiter(monkeypatch, gf)
    TU.reset_for_tests()
    TU.init_with_resources(
        SimpleNamespace(gemm_fusion=GemmFusionConfig(profile="prof_x")), res
    )
    return calls


def test_init_prebuilds_the_profile_once_with_the_detected_target(monkeypatch):
    res = FakeResources()
    built = []
    calls = _init_with_fake_gf(
        monkeypatch,
        res,
        prebuild_profile=lambda name, *, arch=None, cu=None, world=None: built.append(
            (name, arch, cu, world)
        )
        or ["v1"],
    )
    assert built == [("prof_x", "gfx950", 256, 4)]
    assert calls == [("load", "prof_x")]  # fail-fast profile load stays first
    assert hooks.phase() == "pre"  # before any forward / bucket resolution


def test_prebuild_failure_on_this_rank_is_reported_to_peers_then_raised(monkeypatch):
    res = FakeResources()
    reports = []
    res._reduce_min = lambda v: reports.append(list(v)) or list(v)

    def boom(name, **kw):
        raise RuntimeError("hipcc failed")

    with pytest.raises(RuntimeError, match="hipcc failed"):
        _init_with_fake_gf(monkeypatch, res, prebuild_profile=boom)
    assert reports == [[0]]  # the collective still ran and carried this rank's failure


def test_prebuild_failure_on_a_peer_rank_stops_every_rank(monkeypatch):
    res = FakeResources()
    res._reduce_min = lambda v: [0]  # a peer reported failure
    with pytest.raises(RuntimeError, match="failed on a peer rank"):
        _init_with_fake_gf(monkeypatch, res, prebuild_profile=lambda name, **kw: ["ok"])


def test_prebuild_success_runs_the_agreement_collective_exactly_once(monkeypatch):
    res = FakeResources()
    reports = []
    res._reduce_min = lambda v: reports.append(list(v)) or list(v)
    _init_with_fake_gf(monkeypatch, res, prebuild_profile=lambda name, **kw: None)
    assert reports == [[1]]


def test_an_aiter_without_prebuild_is_logged_and_skipped_without_a_collective(
    monkeypatch, caplog
):
    import logging

    res = FakeResources()
    reports = []
    res._reduce_min = lambda v: reports.append(list(v)) or list(v)
    with caplog.at_level(logging.WARNING, logger="atom.gemm_fusion"):
        _init_with_fake_gf(monkeypatch, res)  # no prebuild_profile on the fake aiter
    assert any("no prebuild_profile" in r.getMessage() for r in caplog.records)
    assert (
        reports == [] and hooks.resources() is res
    )  # uniform skip: nothing to agree on


# ------------------------------------------------------- R12: check period at init
def _init_period(monkeypatch, *, check_period=None, kv_publisher=False):
    TU.inject_aiter(monkeypatch, SimpleNamespace(load_profile=lambda name: None))
    TU.reset_for_tests()
    gf_cfg = SimpleNamespace(profile="prof_x")
    if check_period is not None:
        gf_cfg.check_period = check_period
    TU.init_with_resources(
        SimpleNamespace(gemm_fusion=gf_cfg), FakeResources(), kv_publisher=kv_publisher
    )
    return hooks._PERIOD


def test_init_takes_the_period_from_the_config_and_defaults_to_14(monkeypatch):
    assert _init_period(monkeypatch) == 14
    assert _init_period(monkeypatch, check_period=3) == 3


def test_a_kv_publishing_rank_checks_synchronously_whatever_the_config_says(
    monkeypatch,
):
    assert _init_period(monkeypatch, check_period=14, kv_publisher=True) == 1


def test_init_starts_the_check_state_clean(monkeypatch):
    hooks._STEP_IDX, hooks._PENDING = 7, 5
    hooks._WINDOW.append((1, (1,)))
    _init_period(monkeypatch)
    assert (hooks._STEP_IDX, hooks._PENDING, len(hooks._WINDOW)) == (0, None, 0)


# ------------------------------------------- R12: the adapter block of FusionResourcesWrapper
def _adapter(monkeypatch, inner, world=1):
    TU.inject_aiter(monkeypatch, SimpleNamespace())
    return TU.InjectedResourcesWrapper(
        world=world,
        rank=0,
        inner=inner,
        arch="gfx950",
        cu=256,
        reduce_min=lambda v: v,  # not under test: skips the harness's own torch reducer
    )


def test_the_adapter_forwards_every_r12_call_to_one_aiter_name_each(monkeypatch):
    log = []
    inner = SimpleNamespace(
        step_tensor=lambda: log.append("step_tensor") or "STEP",
        readback_enqueue=lambda step, stream: log.append(("enqueue", step, stream)),
        reset=lambda: log.append("reset"),
        quiesce_and_reset=lambda: log.append("quiesce_and_reset"),
    )
    w = _adapter(monkeypatch, inner)
    assert w.step_tensor() == "STEP"
    w.error_readback_async(7, "S")
    w.reset_after_error()  # the response to an observed error
    w.clean_epoch()  # the call the clean points have always made
    assert log == ["step_tensor", ("enqueue", 7, "S"), "reset", "quiesce_and_reset"]


def test_error_word_consumes_without_blocking_first_then_waits(monkeypatch):
    calls = []

    def consume(step, wait):
        calls.append((step, wait))
        return (
            None if (len(calls) == 1) else 3
        )  # the first, non-blocking look finds it incomplete

    w = _adapter(monkeypatch, SimpleNamespace(readback_consume=consume))
    assert w.error_word(5) == 3
    assert calls == [(5, False), (5, True)]
    calls.clear()
    ready = _adapter(monkeypatch, SimpleNamespace(readback_consume=lambda s, wt: 0))
    assert ready.error_word(5) == 0  # complete: one non-blocking call, no wait


def test_error_description_is_best_effort_and_never_raises(monkeypatch):
    err = SimpleNamespace(describe=lambda: "reason=timeout origin_rank=2")
    ok = _adapter(monkeypatch, SimpleNamespace(error_info=lambda: (err, 9, [])))
    assert ok.error_description() == "reason=timeout origin_rank=2 step=9"
    none = _adapter(monkeypatch, SimpleNamespace(error_info=lambda: None))
    assert none.error_description() == "no error page"

    def boom():
        raise RuntimeError("hip error")

    bad = _adapter(monkeypatch, SimpleNamespace(error_info=boom))
    assert "unreadable" in bad.error_description()


def test_check_call_ids_goes_to_aiters_module_and_degrades_without_it(monkeypatch):
    import types

    w = _adapter(monkeypatch, SimpleNamespace())
    monkeypatch.delitem(sys.modules, "aiter.gemm_fusion.callid", raising=False)
    assert (
        w.check_call_ids({("s", 64, "v", 2)}) is None
    )  # older aiter: skipped (logged)
    seen = []
    fake = types.ModuleType("aiter.gemm_fusion.callid")
    fake.check_call_ids = lambda keys: seen.append(set(keys)) or "REPORT"
    monkeypatch.setitem(sys.modules, "aiter.gemm_fusion.callid", fake)
    assert w.check_call_ids({("s", 64, "v", 2)}) == "REPORT"
    assert seen == [{("s", 64, "v", 2)}]


def test_reduce_max_is_trivial_on_one_rank(monkeypatch):
    w = _adapter(monkeypatch, SimpleNamespace(), world=1)
    assert w.reduce_max([0, 1, True]) == [0, 1, 1]


def _fake_torch_distributed(monkeypatch, peer_values):
    """Just enough torch / torch.distributed to see WHICH tensor and group the reduction uses."""
    seen = {}

    class T:
        def __init__(self, values, device):
            self.values, self.device = list(values), device

        def tolist(self):
            return list(self.values)

    def tensor(values, dtype=None, device=None):
        return T(values, device)

    def all_reduce(t, op=None, group=None):
        seen.update(device=t.device, op=op, group=group)
        t.values = [max(a, b) for a, b in zip(t.values, peer_values)]

    fake_torch = SimpleNamespace(
        tensor=tensor,
        int64="i64",
        cuda=SimpleNamespace(current_device=lambda: 0),
        device=lambda kind, idx: f"{kind}:{idx}",
    )
    fake_dist = SimpleNamespace(
        all_reduce=all_reduce, ReduceOp=SimpleNamespace(MAX="MAX")
    )
    fake_torch.distributed = fake_dist
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    return seen


def test_reduce_max_uses_a_cpu_tensor_on_the_gloo_group(monkeypatch):
    seen = _fake_torch_distributed(monkeypatch, peer_values=[1, 0])
    w = _adapter(monkeypatch, SimpleNamespace(), world=4)
    w.cpu_pg, w.pg = "GLOO", "GLOO"
    assert w.reduce_max([0, 0]) == [1, 0]
    assert seen == {"device": None, "op": "MAX", "group": "GLOO"}  # no device tensor


def test_reduce_max_falls_back_to_a_device_tensor_without_a_cpu_group(monkeypatch):
    seen = _fake_torch_distributed(monkeypatch, peer_values=[0])
    w = _adapter(monkeypatch, SimpleNamespace(), world=4)
    w.cpu_pg, w.pg = None, "NCCL"
    assert w.reduce_max([1]) == [1]
    assert seen["group"] == "NCCL" and seen["device"] == "cuda:0"


# -------------------------------------------------- R12: call_id and the static collision check
def _prepared_site(make_site, tokens=(64, 128)):
    res = FakeResources(plan_n=(FakePlan(), 2))
    s = make_site(res)
    for tk in tokens:
        s.resolve(tk, "eager")
        s.resolve(tk, "capture")
    return res, s


def test_the_launch_carries_the_decision_key_as_its_call_id(make_site, monkeypatch):
    res = FakeResources()
    s = make_site(res)
    runs = []
    plan = SimpleNamespace(
        run=lambda **kw: runs.append(kw), recv_view=lambda bound: "VIEW", mode="x"
    )
    monkeypatch.setattr(
        site_mod,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: "S")),
        raising=False,
    )
    r = site_mod.Resolution(
        plan, 1, "fake.large", "profile_hit"
    )  # large plan: raw pointers
    ctx = SimpleNamespace(padded_tokens=64)
    TU.set_phase("ready")
    s._execute(r, "X", "XS", ctx, None, None, None)
    (kw,) = runs
    want = consensus.decision_key("L0.qkv", 64, "fake.large", 1)
    assert kw["call_id"] == want and 0 < want < 2**31
    assert kw["stream"] == "S"  # the existing arguments are untouched


def test_prepare_checks_every_fused_key_once_and_a_clean_report_changes_nothing(
    make_site,
):
    res, s = _prepared_site(make_site)
    res.call_id_report = SimpleNamespace(ok=True, refused_keys=frozenset())
    hooks._refuse_colliding_call_ids()
    assert res.call_id_keys == {
        ("L0.qkv", 64, "fake.v1", 2),
        ("L0.qkv", 128, "fake.v1", 2),
    }
    assert all(r.plan is not None for r in s._by_tokens.values())


def test_a_missing_callid_module_changes_nothing(make_site):
    res, s = _prepared_site(make_site)
    res.call_id_report = None  # older aiter: no check available
    hooks._refuse_colliding_call_ids()
    assert all(r.plan is not None for r in s._by_tokens.values())


def test_colliding_call_ids_make_exactly_those_buckets_native_in_both_modes(make_site):
    res, s = _prepared_site(make_site)
    res.call_id_report = SimpleNamespace(
        ok=False, refused_keys=frozenset({("L0.qkv", 64, "fake.v1", 2)})
    )
    hooks._refuse_colliding_call_ids()
    for mode in ("eager", "capture"):
        r = s._by_tokens[(64, mode)]
        assert (r.plan, r.n_fused, r.reason) == (None, 0, "call_id_refused")
        assert (
            s._by_tokens[(128, mode)].plan is not None
        )  # the other bucket still fuses


def test_native_and_error_entries_never_reach_the_key_set(make_site):
    res = FakeResources(plan_n=(None, 0))
    s = make_site(res)
    s.resolve(64, "eager")  # native: no plan
    s._by_tokens[(96, "eager")] = RuntimeError("an entry can also be an Exception")
    res.call_id_report = SimpleNamespace(ok=True, refused_keys=frozenset())
    hooks._refuse_colliding_call_ids()
    assert res.call_id_keys == set()


# ---------------------------------- R12: r19 plans take x_scale / positions / slot_mapping per launch
class _R19Plan(FakePlan):
    """An aiter small plan of the R12 generation (``abi`` 3): its kernel takes the three arrays per
    launch and pads the rows past ``t_valid``; ``run`` records what the site passes."""

    abi = 3

    def __init__(self):
        self.runs = []

    def run(self, **kw):
        self.runs.append(kw)


def _per_launch_case(make_site, *, abi=3, pos=None, slot=None, xs=None):
    import torch as _torch

    plan = _R19Plan()
    plan.abi = abi
    s = make_site(FakeResources(plan_n=(plan, 2)))
    s.attn = SimpleNamespace(
        impl=SimpleNamespace(layer_num=3, gemm_fusion_prologue_supported=lambda: True)
    )
    s.resolve(64, "eager")
    s.resolve(64, "capture")
    TU.set_phase("ready")
    st = SimpleNamespace(
        outs={"q": "Q"},
        xs=_T((16, 1), _torch.float32),
        pos=_T((64,), _torch.int64),
        slot=_T((64,), _torch.int64),
        prepared=True,
        plan=plan,
        kv_ptrs=None,
        m=16,
    )
    s._staging[64] = st
    state = {
        "slot_mapping": slot or _T((61,), _torch.int64),
        "k_cache": _T(),
        "v_cache": _T(),
        "k_scale": _T(),
        "v_scale": _T(),
    }
    ctx = SimpleNamespace(
        is_dummy_run=False,
        real_tokens=61,
        padded_tokens=64,
        positions=pos or _T((61,), _torch.int64),
        prologue_state=lambda: state,
    )
    x = _T((16, 3072), _torch.float8_e4m3fn)
    x_scale = xs or _T((16, 1), _torch.float32)
    return s, plan, st, state, ctx, x, x_scale


def test_the_site_knows_which_plans_take_their_inputs_per_launch():
    per_launch = site_mod.GemmFusionSite.per_launch
    assert per_launch(FakePlan()) is False  # no abi: the older generation
    assert per_launch(SimpleNamespace(abi=2)) is False  # r17 / r18
    assert per_launch(SimpleNamespace(abi=3)) is True  # r19


@pytest.mark.parametrize(
    "xs_kw,pos_kw,slot_kw,ok",
    [
        ({}, {}, {}, True),
        ({"contiguous": False}, {}, {}, False),
        ({}, {"contiguous": False}, {}, False),
        ({}, {}, {"contiguous": False}, False),
        ({}, {"shape": (0,)}, {}, False),  # an empty array: the plan refuses numel < 1
        ({}, {}, {"shape": (0,)}, False),
        (
            {},
            {"shape": (3,)},
            {"shape": (61,)},
            True,
        ),  # SHORT arrays are legal: the kernel pads
    ],
)
def test_per_launch_inputs_are_checked_against_the_plans_contract(
    xs_kw, pos_kw, slot_kw, ok
):
    import torch as _torch

    xs = _T(**{"shape": (16, 1), "dtype": _torch.float32, **xs_kw})
    pos = _T(**{"shape": (61,), "dtype": _torch.int64, **pos_kw})
    slot = _T(**{"shape": (61,), "dtype": _torch.int64, **slot_kw})
    assert site_mod.GemmFusionSite.per_launch_ok(xs, pos, slot) is ok


def test_a_wrong_dtype_of_positions_or_slot_mapping_fails_the_check():
    import torch as _torch

    xs = _T((16, 1), _torch.float32)
    good = _T((61,), _torch.int64)
    bad = _T((61,), "not-int64")
    assert site_mod.GemmFusionSite.per_launch_ok(xs, bad, good) is False
    assert site_mod.GemmFusionSite.per_launch_ok(xs, good, bad) is False


def test_decide_path_sends_a_violating_per_launch_call_native_with_meta_layout():
    kw = {
        "resolution": site_mod.Resolution(FakePlan(), 2, "fake.v1", "profile_hit"),
        "is_dummy_run": False,
        "x_dtype_is_fp8": True,
        "x_scale_ok": True,
    }
    decide = site_mod.GemmFusionSite.decide_path
    assert decide(**kw, tokens_ok=False) == ("native", "meta_layout")
    assert decide(**kw) == (
        "fused",
        "profile_hit",
    )  # default: nothing to check (r17/r18)
    # an earlier reason still wins, so the reason names the FIRST thing wrong
    assert decide(**{**kw, "is_dummy_run": True}, tokens_ok=False) == (
        "native",
        "dummy_run",
    )
    assert decide(**{**kw, "x_scale_ok": False}, tokens_ok=False) == (
        "native",
        "xscale_layout",
    )
    assert decide(**kw, x_k_ok=False, tokens_ok=False) == ("native", "x_k_mismatch")


def test_an_r19_call_with_conforming_inputs_stays_fused(make_site):
    s, plan, st, state, ctx, x, xs = _per_launch_case(make_site)
    r, _reason, got_state, got_st = s._prepare(ctx, x, xs)
    assert r.plan is plan and got_state is state and got_st is st


@pytest.mark.parametrize("which", ["positions", "slot", "x_scale"])
def test_an_r19_call_with_a_violating_input_is_native_on_the_rank_not_a_raise(
    make_site, which
):
    import torch as _torch

    bad = {
        "positions": {"pos": _T((61,), "not-int64")},  # a dtype that is not int64
        "slot": {"slot": _T((61,), _torch.int64, contiguous=False)},
        "x_scale": {"xs": _T((16, 1), _torch.float32, contiguous=False)},
    }[which]
    s, _plan, _st, _state, ctx, x, xs = _per_launch_case(make_site, **bad)
    r, reason, got_state, got_st = s._prepare(ctx, x, xs)
    assert (r.plan, r.reason, reason) == (None, "meta_layout", "meta_layout")
    assert got_state is None and got_st is None


def test_an_r17_r18_call_never_needs_the_per_launch_contract(make_site):
    """The staging copies convert dtype and layout, so those plans must not be sent native."""
    s, plan, _st, _state, ctx, x, xs = _per_launch_case(
        make_site, abi=2, pos=_T((61,), "not-int64")
    )
    r, _reason, _state, _st = s._prepare(ctx, x, xs)
    assert r.plan is plan


def _recording_torch(monkeypatch):
    monkeypatch.setattr(
        site_mod,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda: "S"),
            int64=site_mod.torch.int64,
        ),
        raising=False,
    )


def _stub_meta(s):
    """``build_meta`` needs the layer's rope tables and norm weights (real tensors); the part under
    test is only which positions / slot_mapping it hands the plan."""
    s._f32 = lambda key, t: t
    impl = s.attn.impl
    impl.rotary_emb = SimpleNamespace(cos_cache=_T(), sin_cache=_T())
    impl.q_norm = SimpleNamespace(weight=_T())
    impl.k_norm = SimpleNamespace(weight=_T())


def test_an_r19_launch_passes_the_callers_tensors_and_stages_nothing(
    make_site, monkeypatch
):
    s, plan, st, state, ctx, x, xs = _per_launch_case(make_site)
    _recording_torch(monkeypatch)
    monkeypatch.setattr(site_mod, "_rows2d", lambda t: t, raising=False)
    _stub_meta(s)

    def staged(*a, **k):
        raise AssertionError("an r19 plan must not use the staging copies")

    s._stage_inputs = staged
    outs = s._execute(
        plan_r := site_mod.Resolution(plan, 2, "fake.v1", "x"),
        x,
        xs,
        ctx,
        state,
        st,
        None,
    )
    assert outs is st.outs and plan_r.plan is plan
    (kw,) = plan.runs
    assert kw["x_scale"] is xs  # the caller's own x_scale, not st.xs
    assert kw["meta"]["positions"] is ctx.positions
    assert kw["meta"]["slot_mapping"] is state["slot_mapping"]
    assert kw["outs"] is st.outs  # outs stay site-owned (they ARE part of the binding)
    assert isinstance(kw["call_id"], int) and 0 < kw["call_id"] < 2**31


def test_an_r17_r18_launch_still_stages_its_inputs(make_site, monkeypatch):
    s, plan, st, state, ctx, x, xs = _per_launch_case(make_site, abi=2)
    _recording_torch(monkeypatch)
    monkeypatch.setattr(site_mod, "_rows2d", lambda t: t, raising=False)
    _stub_meta(s)
    staged = []
    s._stage_inputs = lambda *a: staged.append(a)
    s._execute(
        site_mod.Resolution(plan, 2, "fake.v1", "x"), x, xs, ctx, state, st, None
    )
    assert len(staged) == 1
    (kw,) = plan.runs
    assert kw["x_scale"] is st.xs  # the staging array
    assert kw["meta"]["positions"] is st.pos and kw["meta"]["slot_mapping"] is st.slot


def test_prepare_binds_on_the_sites_conforming_dummies_not_on_a_call(
    make_site, monkeypatch
):
    """prepare() validates its inputs too, so an r19 bind uses the site's T-long arrays."""
    s, _plan, st, state, _ctx, _x, _xs = _per_launch_case(make_site)
    monkeypatch.setattr(site_mod, "_rows2d", lambda t: t, raising=False)
    _stub_meta(s)
    meta = s.build_meta(st, state)  # no ctx: the dummies
    assert meta["positions"] is st.pos and meta["slot_mapping"] is st.slot


def test_the_decision_label_names_the_path_the_inputs_took(make_site):
    s, plan, st, _state, ctx, _x, _xs = _per_launch_case(make_site)
    r = site_mod.Resolution(plan, 2, "fake.v1", "profile_hit")
    assert s._decision(ctx, r, "profile_hit", "fused", st).bind_mode == "per_launch"
    r2 = site_mod.Resolution(SimpleNamespace(abi=2), 2, "fake.v1", "profile_hit")
    assert s._decision(ctx, r2, "profile_hit", "fused", st).bind_mode == "staged"
    r3 = site_mod.Resolution(plan, 1, "fake.large", "profile_hit")
    assert s._decision(ctx, r3, "profile_hit", "fused", st).bind_mode == "raw"
