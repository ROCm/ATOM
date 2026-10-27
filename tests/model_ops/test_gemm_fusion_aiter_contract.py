# SPDX-License-Identifier: MIT
# Contract test: ATOM's gemm_fusion runtime against the REAL aiter.gemm_fusion sources
# (no hand-written aiter stand-in). Only the native layer is faked, and with aiter's OWN test doubles
# (aiter.ops.gemm_fusion.sp_qkv.test_utils: FakeDist, RealisticFirstFit, ArenaBackend): the CCO shared
# library and torch.distributed. aiter's real package __init__, resources.py, match.py, catalog,
# profile and the real sp_qkv variants run unmodified.
#
# Skips when the aiter worktree is not on disk. Needs no GPU and no real torch.
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.model_ops.gemm_fusion import test_utils as TU


def _aiter_root():
    """The aiter worktree with the R12 host API (A1), overridable; falls back to the older one."""
    import os

    wt = Path(__file__).resolve().parents[3]
    candidates = [os.environ.get("ATOM_TEST_AITER_ROOT"), wt / "aiter-gemm-fusion-p1"]
    candidates.append(wt / "aiter-gemm-fusion")
    for c in candidates:
        if c and (Path(c) / "aiter" / "gemm_fusion" / "__init__.py").exists():
            return Path(c)
    return Path(candidates[-1])


AITER_ROOT = _aiter_root()
GF = AITER_ROOT / "aiter" / "gemm_fusion"
PROFILE_NAME = "minimax_m3_sp4_gfx950_v1"

# every aiter.gemm_fusion module-level name ATOM reads (gf.<name>)
ATOM_GF_NAMES = (
    "FusionSpec",
    "GemmSpec",
    "Field",
    "SPExchange",
    "HeadNormRopeCache",
    "load_profile",
    "match_prefix",
    "MatchCtx",
    "ContractError",
    "detect_capabilities",
    "detect_arch",
    "detect_cu",
)


def _load_pkg(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def real_gf(monkeypatch):
    if not (GF / "__init__.py").exists():
        pytest.skip("aiter worktree not available")
    # empty package shells only: aiter/__init__ itself imports torch and the native build
    for n, p in (
        ("aiter", AITER_ROOT / "aiter"),
        ("aiter.ops", AITER_ROOT / "aiter" / "ops"),
        ("aiter.ops.gemm_fusion", AITER_ROOT / "aiter" / "ops" / "gemm_fusion"),
        (
            "aiter.ops.gemm_fusion.sp_qkv",
            AITER_ROOT / "aiter" / "ops" / "gemm_fusion" / "sp_qkv",
        ),
    ):
        shell = types.ModuleType(n)
        shell.__path__ = [str(p)]
        monkeypatch.setitem(sys.modules, n, shell)
    try:
        return _load_pkg(monkeypatch, "aiter.gemm_fusion", GF)
    except ImportError as e:  # a heavy dependency (torch) at import time
        pytest.skip(f"aiter.gemm_fusion not importable without the native stack: {e}")


@pytest.fixture
def atu(real_gf, monkeypatch):
    """aiter's own test doubles and alias helpers (aiter.ops.gemm_fusion.sp_qkv.test_utils)."""
    path = AITER_ROOT / "aiter" / "ops" / "gemm_fusion" / "sp_qkv" / "test_utils.py"
    if not path.exists():
        pytest.skip("aiter predates sp_qkv/test_utils.py")
    name = "aiter.ops.gemm_fusion.sp_qkv.test_utils"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    yield
    TU.reset_for_tests()


@pytest.fixture(autouse=True)
def _no_real_jit_build(request, monkeypatch):
    """hooks.init_gemm_fusion calls aiter's prebuild_profile when it exists; the contract tests run on
    CPU boxes and must never start a real JIT build. The real function stays reachable as
    ``real_gf._orig_prebuild`` for the one test that pins its signature."""
    if "real_gf" not in request.fixturenames:
        return
    gf = request.getfixturevalue("real_gf")
    monkeypatch.setattr(
        gf, "_orig_prebuild", getattr(gf, "prebuild_profile", None), raising=False
    )
    monkeypatch.setattr(gf, "prebuild_profile", lambda *a, **k: [], raising=False)


def test_every_symbol_atom_reads_exists_on_the_real_package_or_its_submodule(real_gf):
    from atom.model_ops.gemm_fusion.resources import fusion_resources_cls

    missing = [n for n in ATOM_GF_NAMES if not hasattr(real_gf, n)]
    assert not missing, f"aiter.gemm_fusion lacks: {missing}"
    cls = fusion_resources_cls(real_gf)  # package export OR aiter.gemm_fusion.resources
    for m in ("ensure", "finalize", "release"):
        assert callable(getattr(cls, m, None)), f"FusionResources.{m} missing"


# The R12 host API (aiter 9b081852). ATOM reaches it ONLY through the adapter block in
# FusionResourcesWrapper (resources.py).
R12_RESOURCE_API = (
    "step_tensor",
    "readback_enqueue",
    "readback_consume",
    "error_info",
    "reset",
    "quiesce_and_reset",
)


def test_aiter_ships_the_r12_host_api_the_adapter_calls(real_gf):
    from atom.model_ops.gemm_fusion.resources import fusion_resources_cls

    cls = fusion_resources_cls(real_gf)
    for m in R12_RESOURCE_API:
        assert callable(getattr(cls, m, None)), f"FusionResources.{m} missing"
    import inspect

    def params(name):
        return list(inspect.signature(getattr(cls, name)).parameters)

    assert params("readback_enqueue")[1:] == ["step_idx", "stream"]
    assert params("readback_consume")[1:] == ["step_idx", "wait"]


def test_aiters_call_id_key_is_atoms_decision_key_and_its_check_is_reachable(
    real_gf, monkeypatch
):
    """The kernels' tags and ATOM's launches must use ONE key function, and the static uniqueness
    check must be what the adapter calls."""
    import importlib

    from atom.model_ops.gemm_fusion import consensus
    from atom.model_ops.gemm_fusion.resources import FusionResourcesWrapper

    callid = importlib.import_module("aiter.gemm_fusion.callid")
    for key in (
        ("L0.qkv", 64, "sp_qkv.small.v1", 2),
        ("model.layers.3.self_attn.qkv", 8192, "sp_qkv.large.r12", 1),
        ("x", 1, "v", 0),
    ):
        assert callid.decision_key(*key) == consensus.decision_key(*key)
    w = TU.InjectedResourcesWrapper(
        world=1, rank=0, inner=SimpleNamespace(), arch="gfx950", cu=256
    )
    rep = FusionResourcesWrapper.check_call_ids(
        w, {("a", 64, "v", 2), ("b", 64, "v", 2)}
    )
    assert rep.ok and rep.keys == 2  # distinct keys, nothing refused


def test_fusion_resources_constructor_accepts_the_kwargs_atom_passes(real_gf):
    import inspect

    from atom.model_ops.gemm_fusion.resources import fusion_resources_cls

    params = inspect.signature(fusion_resources_cls(real_gf).__init__).parameters
    for kw in (
        "rank",
        "world",
        "group",
        "device",
        "caps",
        "profile",
    ):  # exactly what the wrapper passes
        assert kw in params, f"FusionResources.__init__ has no '{kw}'"


def _fake_torch(monkeypatch, atu):
    """A minimal ``torch`` (+ ``torch.distributed``) so the real FusionResources, which does
    ``import torch.distributed as dist`` when no dist is injected, can be built on a CPU box.
    """
    dist = atu.FakeDist()
    tmod = types.ModuleType("torch")
    tmod.cuda = types.SimpleNamespace(current_device=lambda: 0)
    tmod.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", tmod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)


def _profile_budget(real_gf):
    """window_budget as recorded IN the shipped profile (the source of truth for the asserts)."""
    wb = dict(real_gf.load_profile(PROFILE_NAME).meta.get("window_budget") or {})
    assert wb.get("window_bytes") and wb.get(
        "max_windows"
    ), "shipped profile has no window_budget"
    return wb


def _config(profile):
    """What ModelRunner hands init_gemm_fusion: only ``config.gemm_fusion.profile`` matters."""
    return SimpleNamespace(
        gemm_fusion=SimpleNamespace(profile=profile, enabled=bool(profile))
    )


def _real_inner(real_gf, atu, backend=None, **kw):
    """The REAL FusionResources on aiter's fake backend, built with the profile like the wrapper does."""
    from atom.model_ops.gemm_fusion.resources import fusion_resources_cls

    return fusion_resources_cls(real_gf)(
        rank=0,
        world=4,
        group=None,
        dist=atu.FakeDist(),
        backend=backend if backend is not None else atu.ArenaBackend(),
        device=0,
        caps=frozenset({"mori_cco", "gfx950_sc_bypass"}),
        profile=PROFILE_NAME,
        **kw,
    )


def _wrapper(real_gf, atu, backend=None):
    return TU.InjectedResourcesWrapper(
        None,
        4,
        rank=0,
        inner=_real_inner(real_gf, atu, backend),
        reduce_min=lambda v: list(v),
        arch="gfx950",
        cu=256,
    )


def test_window_budget_comes_from_the_config_profile_with_the_env_unset(
    real_gf, atu, monkeypatch
):
    """E2 new_on02: with AITER_GEMM_FUSION_PROFILE unset the resources' budget was unknown, so the
    first (4 MiB, small-M) request sized every window and the first large-M plan (40 MiB) was
    rejected. Through hooks.init_gemm_fusion(config) (the runtime builds its own wrapper) the REAL
    FusionResources must get its budget and VMM heap from config.gemm_fusion.profile."""
    from atom.model_ops.gemm_fusion import hooks, stages
    from atom.model_ops.gemm_fusion import resources as atom_res

    for k in (
        "AITER_GEMM_FUSION_PROFILE",
        "AITER_GEMM_FUSION_WINDOW_POLICY",
        "AITER_GEMM_FUSION_ARENA_MIB",
    ):
        monkeypatch.delenv(k, raising=False)
    wb = _profile_budget(real_gf)

    class _Recording(atu.ArenaBackend):
        def comm_create(self, rank, world, uid, vmm, device=None):
            self.device = (
                device  # the HIP device ATOM passes (the SP rank is not a device index)
            )
            return super().comm_create(rank, world, uid, vmm, device)

    backend = _Recording()
    res_mod = importlib.import_module("aiter.gemm_fusion.resources")
    monkeypatch.setattr(res_mod, "CcoBackend", lambda *a, **k: backend)
    _fake_torch(monkeypatch, atu)
    monkeypatch.setattr(atom_res, "_device", lambda: 0)
    # the wrapper builds a real min-reducer for world > 1; a single process uses identity
    monkeypatch.setattr(
        atom_res.consensus, "torch_min_reducer", lambda *a, **k: (lambda v: list(v))
    )
    TU.reset_for_tests()
    sp_group = types.SimpleNamespace(rank_in_group=0, cpu_group=None, device_group=None)
    monkeypatch.setattr(stages, "sp_world", lambda: 4)
    try:
        hooks.init_gemm_fusion(_config(PROFILE_NAME), sp_group)
        inner = hooks.resources().inner
        # The numbers are aiter's (its profile changes them whenever a variant is added: 39 -> 40 with
        # the global page, 52 with r20/r21), so they are read from the profile, never copied here.
        # What is ATOM's to prove: the REAL resources got exactly the profile's budget.
        assert inner.window_bytes == wb["window_bytes"] > 0
        assert inner.max_windows == wb["max_windows"] > 0
        # arena (the default): ONE registered window of (shared S/R/C + small windows) bytes, and a
        # heap of roundup_4GiB(arena + 256 MiB slack)
        assert inner.policy == "arena" and inner.policy_fallback_reason is None
        arena = inner._arena_size_from_budget()
        assert arena == sum(wb["shared"].values()) + sum(wb["small_window_sizes"]) > 0
        assert inner._vmm >= arena + (256 << 20) and inner._vmm % (4 << 30) == 0
        assert (
            backend.device == 0
        )  # the real constructor ran its comm init on ATOM's device
    except ImportError as e:
        pytest.skip(f"resources need the native stack to import: {e}")


def test_without_the_profile_the_real_resources_reject_the_large_plan(
    real_gf, atu, monkeypatch
):
    """Control: the same real FusionResources built WITHOUT a profile reproduces the E2 failure
    (uniform window sized by the first small request, then the large plan does not fit). Shows the
    test above has power and that the failure is ATOM's omission, not aiter's."""
    for k in (
        "AITER_GEMM_FUSION_PROFILE",
        "AITER_GEMM_FUSION_WINDOW_POLICY",
        "AITER_GEMM_FUSION_ARENA_MIB",
    ):
        monkeypatch.delenv(k, raising=False)
    res_mod = importlib.import_module("aiter.gemm_fusion.resources")
    kw = {
        "rank": 0,
        "world": 4,
        "group": None,
        "device": 0,
        "caps": frozenset({"mori_cco"}),
    }
    bare = real_gf.FusionResources(
        dist=atu.FakeDist(), backend=atu.RealisticFirstFit(), **kw
    )
    assert bare.window_bytes in (None, 0)  # budget unknown without a profile
    small = real_gf.WindowReq("o/rx", 4 << 20, "p2p_window")
    large = real_gf.WindowReq("o2/R", 40 << 20, "p2p_window")
    bare.ensure("small", [small])
    with pytest.raises(res_mod.ResourceError, match="exceeds the uniform window size"):
        bare.ensure("large", [large])
    # with the profile the same sequence is accepted
    ok = real_gf.FusionResources(
        dist=atu.FakeDist(), backend=atu.ArenaBackend(), profile=PROFILE_NAME, **kw
    )
    ok.ensure("small", [small])
    ok.ensure("large", [large])


def test_wrapper_drives_the_real_fusion_resources_end_to_end(real_gf, atu):
    """ATOM's wrapper around the REAL FusionResources, a real small-M plan from the real
    catalog/profile, and the real window ownership rules."""
    from atom.model_ops.gemm_fusion.resources import plan_owner

    backend = atu.ArenaBackend()
    w = TU.InjectedResourcesWrapper(
        None,
        4,
        rank=0,
        inner=_real_inner(real_gf, atu, backend),
        reduce_min=lambda v: list(v),
        arch="gfx950",
        cu=256,  # 1-process identity reduce
    )
    assert w.finalized is False and w.rank == 0 and w.local_m(64) == 16
    assert w.caps == frozenset({"mori_cco", "gfx950_sc_bypass"})

    # a real plan, from the real shipped profile, via the real matcher
    prof = real_gf.load_profile(PROFILE_NAME)
    row = next(r for r in prof.rows() if "small" in r.vid and r.m == 16)
    spec_mod = importlib.import_module("aiter.ops.gemm_fusion.sp_qkv.specs")
    mode = next(
        m for m in spec_mod.MODES if spec_mod.sp_qkv_spec(m).signature() == row.sig
    )
    spec = spec_mod.sp_qkv_spec(mode)
    try:
        plan, n = real_gf.match_prefix(spec, w.match_ctx(16, "eager"), PROFILE_NAME)
    except ImportError as e:
        pytest.skip(f"variants need the native stack to import: {e}")
    assert plan is not None and n == 2 and plan.vid == "sp_qkv.small.v1"
    # aiter 46a73020: every small row of the shipped profile runs src r19 (abi 3). ATOM must then
    # pass x_scale / positions / slot_mapping per launch and stage nothing for it.
    from atom.model_ops.gemm_fusion.site import GemmFusionSite

    assert (plan.src, plan.abi) == ("r19", 3)
    assert GemmFusionSite.per_launch(plan) is True

    # THE ownership contract: ATOM must key windows exactly as the plan does
    assert (
        plan_owner(plan) == plan.owner == plan.build_key != plan.vid
    )  # aiter: build_key == owner
    assert w.can_ensure(plan) is True
    w.ensure(plan)
    assert plan.owner in w.inner.windows
    expected = {plan.owner + "/rx", plan.owner + "/ctl"}
    if (
        getattr(plan, "abi", 2) >= 3
    ):  # an R12 plan also requests the per-comm ERR/STEP page
        expected.add(plan.owner + "/gpage")
    assert set(w.inner.windows[plan.owner]) == expected
    w.finalize()
    assert w.finalized is True and any(
        e[0] == "dev" for e in backend.log
    )  # the devcomm was created
    assert w.can_ensure(plan) is True  # same owner after finalize: allowed (idempotent)
    # a plan with a DIFFERENT owner (different params/mode) cannot ensure after finalize
    other = SimpleNamespace(owner="sp_qkv.small/other", vid=plan.vid)
    assert w.can_ensure(other) is False
    # (the collective reset / error-word API of R12 is covered by
    # test_aiter_ships_the_r12_host_api_the_adapter_calls once aiter A1 lands)
    w.release()


def test_plan_attributes_atom_reads_exist_on_the_real_variants(real_gf):
    import ast

    root = AITER_ROOT / "aiter" / "ops" / "gemm_fusion" / "sp_qkv"
    small = ast.parse((root / "variants.py").read_text(encoding="utf-8"))
    large = ast.parse((root / "variants_large.py").read_text(encoding="utf-8"))

    def methods(tree, cls):
        c = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls
        )
        return {n.name for n in c.body if isinstance(n, ast.FunctionDef)}

    assert {"resources", "prepare", "run"} <= methods(small, "SpQkvSmall")
    assert {
        "resources",
        "prepare",
        "run",
        "recv_view",
        "reuse_view",
        "poison",
    } <= methods(large, "SpQkvLarge")
    # ATOM reads plan.owner and plan.mode: both assigned in __init__
    assert "self.owner =" in (root / "variants.py").read_text(encoding="utf-8")
    assert "self.owner =" in (root / "variants_large.py").read_text(encoding="utf-8")
    assert "self.mode =" in (root / "variants_large.py").read_text(encoding="utf-8")


def test_atom_only_calls_plan_methods_that_exist(real_gf):
    """Every `plan.<x>` / `r.plan.<x>` ATOM RUNTIME source calls must be a real plan method."""
    import re

    src = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (
            Path(__file__).resolve().parents[2] / "atom/model_ops/gemm_fusion"
        ).glob("*.py")
        if p.name != "test_utils.py"
    )
    called = set(re.findall(r"\b(?:r\.)?plan\.(\w+)\(", src))
    allowed = {"resources", "prepare", "run", "recv_view", "reuse_view", "poison"}
    assert (
        called <= allowed
    ), f"ATOM calls plan methods aiter does not define: {sorted(called - allowed)}"


def test_atom_runtime_reads_only_resource_attributes_aiter_keeps(real_gf):
    """aiter's cleanup removed local_m/peers/peer_ptr/va_info/find_window_alias/assert_no_window_alias
    and the zero/registration modes from FusionResources. The ATOM runtime may read only what is kept.
    """
    import re

    runtime = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (
            Path(__file__).resolve().parents[2] / "atom/model_ops/gemm_fusion"
        ).glob("*.py")
        if p.name != "test_utils.py"
    )
    read = set(re.findall(r"\binner\.(\w+)", runtime)) | set(
        re.findall(r"\bbound\.(\w+)", runtime)
    )
    kept = {
        "ensure",
        "finalize",
        "release",
        "windows",
        "finalized",
        "caps",
        "window_bytes",
        "max_windows",
        "policy",
        "policy_fallback_reason",
    }
    r12 = set(
        R12_RESOURCE_API
    )  # the adapter's R12 reads; existence: see the xfail test above
    assert (
        read <= kept | r12
    ), f"ATOM runtime reads FusionResources attributes aiter dropped: {sorted(read - kept - r12)}"
    removed = {
        "local_m",
        "peers",
        "peer_ptr",
        "va_info",
        "window_ranges",
        "find_window_alias",
        "assert_no_window_alias",
        "debug_alias",
        "zero_mode",
        "reg_barrier",
        "reg_sync",
        "allow_late",
    }
    assert not (read & removed)


def _dense_site(spec):
    from atom.model_ops.gemm_fusion import site as site_mod

    impl = SimpleNamespace(gemm_fusion_prologue_supported=lambda: True)
    site = site_mod.GemmFusionSite(
        "t.qkv",
        SimpleNamespace(
            output_size=9216,
            input_size=6144,
            is_output_padded=False,
            weight=SimpleNamespace(shape=(9216, 6144), is_shuffled=True),
            weight_scale=SimpleNamespace(shape=(9216, 1)),
        ),
        list(spec.stages),
        None,
        attn=SimpleNamespace(impl=impl),
        q_size=2048,
        kv_size=128,
    )
    site.spec = spec
    return site


def test_real_matcher_builds_a_new_plan_per_call_but_the_site_keeps_one(
    real_gf, atu, monkeypatch
):
    """gate04 root cause, on the REAL aiter classes: ``match_prefix`` returns a distinct plan
    (with its own pointer-binding dict) on every call, so a plan prepared for 'eager' is not
    the plan a 'capture' lookup would get. The site must hand back ONE instance."""
    spec_mod = importlib.import_module("aiter.ops.gemm_fusion.sp_qkv.specs")
    spec = spec_mod.sp_qkv_spec("dense")
    ctx = real_gf.MatchCtx(
        arch="gfx950",
        cu=256,
        world=4,
        m=16,
        available=frozenset({"mori_cco", "gfx950_sc_bypass"}),
        graph_mode="eager",
    )
    try:
        a, _ = real_gf.match_prefix(spec, ctx, PROFILE_NAME)
        b, _ = real_gf.match_prefix(spec, ctx, PROFILE_NAME)
    except ImportError as e:
        pytest.skip(f"variants need the native stack to import: {e}")
    assert a is not b and a._bindings is not b._bindings  # the hazard is real
    a._bindings[("probe",)] = "bound by prepare"
    assert ("probe",) not in b._bindings

    # through the REAL runtime path: hooks.init_gemm_fusion(config) -> site.resolve, on the real
    # aiter package, with the profile env var unset (so only the config can supply it)
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    TU.reset_for_tests()
    TU.init_with_resources(_config(PROFILE_NAME), _wrapper(real_gf, atu))
    TU.set_phase("preparing")
    site = _dense_site(spec)
    eager = site.resolve(64, "eager")
    capture = site.resolve(64, "capture")
    assert (
        eager.plan is not None and eager.n_fused == 2 and eager.reason == "profile_hit"
    )
    assert (
        eager.plan is capture.plan
    )  # one instance => one set of bindings for eager+capture+replay


def test_runtime_path_takes_the_profile_from_config_not_from_the_environment(
    real_gf, atu, monkeypatch
):
    """E2 gate failure (job-e2_new_on01): bind_kv_cache -> prepare_buckets -> site.resolve ->
    match_prefix raised ProfileError 'no profile given and AITER_GEMM_FUSION_PROFILE is unset'
    although --gemm-fusion-profile had been passed. Reproduces that path on the REAL aiter
    package with the env var unset: init stores the config profile, bind_kv_cache's
    profile_tokens and every resolve use it."""
    from atom.model_ops.gemm_fusion import hooks

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    spec_mod = importlib.import_module("aiter.ops.gemm_fusion.sp_qkv.specs")
    TU.reset_for_tests()
    TU.init_with_resources(_config(PROFILE_NAME), _wrapper(real_gf, atu))
    try:
        assert hooks.profile() == PROFILE_NAME
        toks = hooks.profile_tokens(
            4
        )  # what bind_kv_cache enumerates; used to read the env
        assert toks and all(t % 4 == 0 for t in toks)
        assert 64 in toks and 128 in toks  # the small-M golden cells
        TU.set_phase("preparing")
        r = _dense_site(spec_mod.sp_qkv_spec("dense")).resolve(64, "eager")
        assert (r.reason, r.n_fused) == ("profile_hit", 2)
    except ImportError as e:
        pytest.skip(f"variants need the native stack to import: {e}")


def test_init_fails_fast_when_the_configured_profile_cannot_load(real_gf, monkeypatch):
    """A bad --gemm-fusion-profile must stop start-up inside init_gemm_fusion (aiter's
    ProfileError), before any CCO resource is built and long before bind_kv_cache."""
    from atom.model_ops.gemm_fusion import hooks
    from atom.model_ops.gemm_fusion import resources as R

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    TU.reset_for_tests()
    built = []

    class _Boom:
        def __init__(self, *a, **k):
            built.append(1)

    monkeypatch.setattr(R, "FusionResourcesWrapper", _Boom)
    with pytest.raises(real_gf.ProfileError):
        hooks.init_gemm_fusion(_config("no_such_profile_xyz"), None)
    assert built == [] and hooks.resources() is None  # no resource was created
    with pytest.raises(RuntimeError, match="no profile configured"):
        hooks.profile()  # and nothing was stored


def test_env_var_does_not_substitute_for_the_config_profile(real_gf, atu, monkeypatch):
    """With the env var SET to a valid profile but no config profile, ATOM must not silently
    use it: that would mask a broken CLI -> runtime path (the very bug above)."""
    from atom.model_ops.gemm_fusion import hooks

    monkeypatch.setenv("AITER_GEMM_FUSION_PROFILE", PROFILE_NAME)
    TU.reset_for_tests(_wrapper(real_gf, atu))
    hooks._PROFILE = ""  # resources exist but init never recorded a profile
    TU.set_phase("preparing")
    spec_mod = importlib.import_module("aiter.ops.gemm_fusion.sp_qkv.specs")
    try:
        with pytest.raises(RuntimeError, match="no profile configured"):
            _dense_site(spec_mod.sp_qkv_spec("dense")).resolve(64, "eager")
        with pytest.raises(RuntimeError, match="no profile configured"):
            hooks.profile_tokens(4)
    except ImportError as e:
        pytest.skip(f"variants need the native stack to import: {e}")


class _Tens:
    """Tensor-like with exactly what aiter's alias walker reads (data_ptr/shape/stride/element_size)."""

    def __init__(self, ptr, n):
        self._p, self.shape = ptr, (n,)

    def data_ptr(self):
        return self._p

    def stride(self):
        return (1,)

    def element_size(self):
        return 1


def test_check_no_window_alias_drives_aiters_real_helper_on_real_windows(
    real_gf, atu, monkeypatch
):
    """C-S4 against aiter's REAL ``test_utils.assert_no_window_alias`` and its window registry: a
    tensor inside the registered arena is rejected, outside is accepted, and the verdict comes from
    aiter's walker with force=True (the env flag is deliberately unset)."""
    from atom.model_ops.gemm_fusion import site as site_mod

    monkeypatch.delenv(
        "AITER_GEMM_FUSION_DEBUG_ALIAS", raising=False
    )  # ATOM must force, not rely on env
    w = _wrapper(real_gf, atu)
    w.inner.ensure(
        "o", [real_gf.WindowReq("o/R", 4 << 20, "p2p_window")]
    )  # registers the arena
    TU.reset_for_tests(w)
    reg = w.inner._regs[0]
    base, size = int(reg.local), int(reg.bytes)
    TU.check_no_window_alias(
        [("out", _Tens(base + size + 4096, 64)), ("scale", _Tens(0x1000, 8))]
    )
    with pytest.raises(site_mod.GemmFusionBindError, match="points into a CCO window"):
        TU.check_no_window_alias([("out", _Tens(base + 4096, 64))])
    with pytest.raises(site_mod.GemmFusionBindError, match="impl._index_q"):
        TU.check_no_window_alias(
            [("impl._index_q", _Tens(base - 16, 64))]
        )  # straddles the start


def test_release_frees_every_window_after_its_deregister_and_before_comm_destroy(
    real_gf, atu
):
    """MORI's teardown contract: WindowDeregister -> ccoMemFree(local) for EVERY registered window,
    all before ccoCommDestroy (aiter always does it now; there is no switch). Drives the REAL
    FusionResources.release() on aiter's recording backend and reads the call order back.
    """
    be = atu.ArenaBackend()
    res = _real_inner(real_gf, atu, be)
    res.ensure("small", [real_gf.WindowReq("o/rx", 4 << 20, "p2p_window")])
    res.ensure("large", [real_gf.WindowReq("o2/R", 40 << 20, "p2p_window")])
    registered = list(res._regs)
    assert (
        len(registered) >= 1
    )  # arena (default): ONE window; the views are freed separately
    res.release()
    log = be.log
    dest = log.index(("commdestroy",))
    assert dest == len(log) - 1  # comm_destroy is last
    for w in registered:
        assert log.index(("dereg", w.handle)) < log.index(("free", w.local)) < dest
    assert sum(1 for e in log if e[0] == "free") == len(registered)  # once each
    res.release()  # idempotent: no second free, no second destroy
    assert be.log == log


def test_release_order_holds_for_a_multi_window_policy_too(real_gf, atu, monkeypatch):
    """Same teardown contract with several registered windows (explicit uniform policy), so the
    one-window arena default does not hide a per-window ordering bug."""
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "uniform")
    be = atu.RealisticFirstFit()
    res = _real_inner(real_gf, atu, be)
    assert res.policy == "uniform"
    res.ensure("small", [real_gf.WindowReq("o/rx", 4 << 20, "p2p_window")])
    res.ensure("large", [real_gf.WindowReq("o2/R", 40 << 20, "p2p_window")])
    registered = list(res._regs)
    assert len(registered) >= 2
    res.release()
    log = be.log
    dest = log.index(("commdestroy",))
    for w in registered:
        assert log.index(("dereg", w.handle)) < log.index(("free", w.local)) < dest
    assert sum(1 for e in log if e[0] == "free") == len(registered)


def test_atom_shutdown_releases_the_comm_once(real_gf, atu):
    """ATOM's runner calls gemm_fusion.shutdown() from ModelRunner.exit; shutdown() reaches the real
    release() and therefore the free-before-destroy sequence, and does so once."""
    from atom.model_ops.gemm_fusion import hooks

    be = atu.ArenaBackend()
    inner = _real_inner(real_gf, atu, be)
    inner.ensure("small", [real_gf.WindowReq("o/rx", 4 << 20, "p2p_window")])
    w = TU.InjectedResourcesWrapper(
        None,
        4,
        rank=0,
        inner=inner,
        reduce_min=lambda v: list(v),
        arch="gfx950",
        cu=256,
    )
    TU.reset_for_tests(w)
    hooks.shutdown()
    assert be.log.count(("commdestroy",)) == 1 and any(e[0] == "free" for e in be.log)
    assert hooks.resources() is None
    hooks.shutdown()  # idempotent
    assert be.log.count(("commdestroy",)) == 1


def test_explicit_uniform_policy_keeps_the_phase1_heap_for_the_profile_budget(
    real_gf, atu, monkeypatch
):
    """The pre-arena layout stays selectable (AITER_GEMM_FUSION_WINDOW_POLICY=uniform): the heap is
    sized for window_bytes * max_windows from the profile budget, with no env var carrying the
    profile."""
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "uniform")
    wb = _profile_budget(real_gf)
    res = _real_inner(real_gf, atu, atu.RealisticFirstFit())
    assert res.policy == "uniform"
    assert (
        res.window_bytes == wb["window_bytes"] and res.max_windows == wb["max_windows"]
    )
    assert res._vmm >= res.window_bytes * res.max_windows


def test_aiters_prebuild_profile_has_the_signature_atom_calls(real_gf):
    """ATOM calls ``aiter.gemm_fusion.prebuild_profile(profile, *, arch=None, cu=None, world=None)`` once
    at init on every rank (agreed name and shape). Skipped until the installed aiter defines it.
    """
    import inspect

    fn = real_gf._orig_prebuild
    if fn is None:
        pytest.skip(
            "aiter does not define prebuild_profile yet (ATOM logs a warning and skips the call)"
        )
    params = inspect.signature(fn).parameters
    assert next(iter(params)) in ("profile", "name"), list(params)
    for kw in ("arch", "cu", "world"):
        assert kw in params and params[kw].kind is inspect.Parameter.KEYWORD_ONLY, (
            kw,
            list(params),
        )


def test_init_calls_prebuild_with_the_wrapper_target_on_the_real_aiter(
    real_gf, atu, monkeypatch
):
    from atom.model_ops.gemm_fusion import hooks

    seen = []
    monkeypatch.setattr(
        real_gf,
        "prebuild_profile",
        lambda name, *, arch=None, cu=None, world=None: seen.append(
            (name, arch, cu, world)
        )
        or [],
        raising=False,
    )
    TU.reset_for_tests()
    TU.init_with_resources(_config(PROFILE_NAME), _wrapper(real_gf, atu))
    assert seen == [(PROFILE_NAME, "gfx950", 256, 4)]
    assert hooks.phase() == "pre"


def test_the_adapter_recognises_the_real_aiter_reset_error(real_gf):
    """aiter 44f67753: reset() verifies the error word with an all-gather and raises ResetError
    (exported, a ResourceError, with ``words`` = {rank: ERR word}) when it cannot clear it. ATOM turns
    that into a fatal GemmFusionDeviceError (hooks._fatal_reset), which needs this recognition to
    keep working. The reset sequence itself is aiter's to test (its own suite)."""
    assert issubclass(real_gf.ResetError, real_gf.ResourceError)
    w = TU.InjectedResourcesWrapper(
        world=1, rank=0, inner=SimpleNamespace(), arch="gfx950", cu=256
    )
    words = {0: 0, 1: 0, 2: 0x21, 3: 0}
    err = real_gf.ResetError("reset: ERR still non-zero", words)
    assert w.reset_error_words(err) == words
    assert w.reset_error_words(real_gf.ResourceError("not a reset failure")) is None


def test_the_real_fusion_resources_carry_the_launch_gate_the_adapter_reads(
    real_gf, atu
):
    """aiter f6309062: ``FusionResources.launch_gate`` is True for the whole of reset() and
    ``Plan.run`` raises ``LaunchGateError`` while it is set; a graph replay bypasses Python, so ATOM
    reads the flag itself (hooks.check_launch_gate) through the adapter."""
    assert issubclass(real_gf.LaunchGateError, real_gf.ResourceError)
    w = TU.InjectedResourcesWrapper(
        None,
        4,
        rank=0,
        inner=_real_inner(real_gf, atu, atu.ArenaBackend()),
        reduce_min=lambda v: list(v),
        arch="gfx950",
        cu=256,
    )
    assert w.inner.launch_gate is False and w.launch_gate is False
    w.inner.launch_gate = True
    assert w.launch_gate is True


# ---------------------------------------------------------------------- R16 pad-M (aiter da8e786a)
def _needs_pad_m(real_gf):
    if not hasattr(real_gf, "bucket_for_m"):
        pytest.skip("aiter predates R16 pad-M (no bucket_for_m)")


def _fn_params(path, name, cls=None):
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope = tree
    if cls is not None:
        scope = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls
        )
    fn = next(
        n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return [a.arg for a in fn.args.args + fn.args.kwonlyargs]


def test_the_pad_m_api_atom_calls_exists_with_the_signatures_it_uses(real_gf):
    _needs_pad_m(real_gf)
    root = AITER_ROOT / "aiter"
    assert {"bucket_for_m", "match_padded"} <= set(real_gf.__all__)
    assert _fn_params(GF / "match.py", "bucket_for_m") == ["spec", "ctx", "profile"]
    assert _fn_params(GF / "match.py", "match_padded") == ["spec", "ctx", "profile"]
    ids = _fn_params(GF / "callid.py", "tag_unique_ids")
    assert ids[:1] == ["keys"] and "tag_bits" in ids
    chk = _fn_params(GF / "callid.py", "check_call_ids")
    for kw in ("key_fn", "tag_bits", "tag_vids", "per_tokens"):
        assert kw in chk, f"check_call_ids lost '{kw}', which the adapter passes"
    large = root / "ops" / "gemm_fusion" / "sp_qkv" / "variants_large.py"
    for view in ("recv_view", "reuse_view"):
        assert _fn_params(large, view, "SpQkvLarge") == ["self", "res", "m_real"]
    for f in ("variants.py", "variants_large.py"):  # the plan attributes the site reads
        src = (large.parent / f).read_text(encoding="utf-8")
        assert "self.pad_m =" in src and "self.src =" in src


def test_the_site_resolves_every_m_through_aiters_real_bucket_for_m_on_the_shipped_profile(
    real_gf, atu, monkeypatch
):
    """The pure pad-M decision on the REAL matcher and profile: a token count that was not prepared
    resolves to the prepared bucket aiter's ``bucket_for_m`` names (rows are prepared, nothing else),
    and its call_id key is one aiter's own pad-M key universe contains."""
    _needs_pad_m(real_gf)
    from atom.model_ops.gemm_fusion import hooks

    spec_mod = importlib.import_module("aiter.ops.gemm_fusion.sp_qkv.specs")
    spec = spec_mod.sp_qkv_spec("dense")
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    TU.reset_for_tests()
    TU.init_with_resources(_config(PROFILE_NAME), _wrapper(real_gf, atu))
    TU.set_phase("preparing")
    site = _dense_site(spec)
    prof = real_gf.load_profile(PROFILE_NAME)
    rows = sorted({r.m for r in prof.rows() if r.m <= 256})
    assert rows[:3] == [1, 2, 4] and rows[-1] <= 256
    try:
        for m in rows:
            for mode in ("eager", "capture"):
                assert site.resolve(4 * m, mode).plan is not None, m
    except ImportError as e:
        pytest.skip(f"variants need the native stack to import: {e}")
    TU.set_phase("ready")
    caps = frozenset({"mori_cco", "gfx950_sc_bypass"})
    keys, _ = importlib.import_module("aiter.gemm_fusion.callid").profile_call_id_keys(
        prof, [site.name], world=4, pad_m=True
    )
    universe = set(keys)
    padded = 0
    for m in range(1, rows[-1] + 1):
        ctx = real_gf.MatchCtx(
            arch="gfx950", cu=256, world=4, m=m, available=caps, graph_mode="eager"
        )
        bucket = real_gf.bucket_for_m(spec, ctx, PROFILE_NAME)
        r = site.resolve(4 * m, "eager")
        if m in rows:
            assert (r.bucket_tokens, r.reason) == (0, "profile_hit"), m
            continue
        assert bucket in rows and bucket > m, (m, bucket)  # rows are what is prepared
        assert (r.reason, r.bucket_tokens) == ("pad_hit", 4 * bucket), m
        assert r.plan is site._by_tokens[(4 * bucket, "eager")].plan
        assert (site.name, 4 * m, r.vid, r.n_fused) in universe, m
        padded += 1
    assert padded > 100  # most of m 1..256 is served by padding
    assert hooks.world_or_1() == 4
