# SPDX-License-Identifier: MIT
# Contract test: ATOM's gemm_fusion runtime against the REAL aiter.gemm_fusion sources
# (no hand-written aiter stand-in). Only the native layer is faked: the CCO shared
# library (CcoBackend) and torch.distributed. aiter's real package __init__,
# resources.py, match.py, catalog, profile and the real sp_qkv variants run unmodified.
#
# Skips when the aiter worktree is not on disk. Needs no GPU and no real torch.
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

AITER_ROOT = Path(__file__).resolve().parents[3] / "aiter-gemm-fusion"
GF = AITER_ROOT / "aiter" / "gemm_fusion"

# every aiter.gemm_fusion module-level name ATOM reads (gf.<name>)
ATOM_GF_NAMES = (
    "FusionSpec", "GemmSpec", "Field", "SPExchange", "HeadNormRopeCache",
    "load_profile", "match_prefix", "MatchCtx", "ContractError",
    "detect_capabilities", "detect_arch", "detect_cu",
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
    for n, p in (("aiter", AITER_ROOT / "aiter"), ("aiter.ops", AITER_ROOT / "aiter" / "ops"),
                 ("aiter.ops.gemm_fusion", AITER_ROOT / "aiter" / "ops" / "gemm_fusion"),
                 ("aiter.ops.gemm_fusion.sp_qkv", AITER_ROOT / "aiter" / "ops" / "gemm_fusion" / "sp_qkv")):
        shell = types.ModuleType(n)
        shell.__path__ = [str(p)]
        monkeypatch.setitem(sys.modules, n, shell)
    try:
        return _load_pkg(monkeypatch, "aiter.gemm_fusion", GF)
    except ImportError as e:  # a heavy dependency (torch) at import time
        pytest.skip(f"aiter.gemm_fusion not importable without the native stack: {e}")


def test_every_symbol_atom_reads_exists_on_the_real_package_or_its_submodule(real_gf):
    from atom.model_ops.gemm_fusion.resources import fusion_resources_cls

    missing = [n for n in ATOM_GF_NAMES if not hasattr(real_gf, n)]
    assert not missing, f"aiter.gemm_fusion lacks: {missing}"
    cls = fusion_resources_cls(real_gf)  # package export OR aiter.gemm_fusion.resources
    for m in ("ensure", "finalize", "quiesce_and_reset", "release", "poison_local"):
        assert callable(getattr(cls, m, None)), f"FusionResources.{m} missing"


def test_fusion_resources_constructor_accepts_the_kwargs_atom_passes(real_gf):
    import inspect

    from atom.model_ops.gemm_fusion.resources import fusion_resources_cls

    params = inspect.signature(fusion_resources_cls(real_gf).__init__).parameters
    for kw in ("rank", "world", "group", "device", "caps"):  # exactly what the wrapper passes
        assert kw in params, f"FusionResources.__init__ has no '{kw}'"


class _FakeBackend:
    """The native CCO layer only; everything above it is the real aiter code."""

    def __init__(self):
        self.calls, self._n, self._views = [], 0, {}

    def uid(self, rank):
        return b"u" * 128

    def comm_create(self, rank, world, uid, vmm_bytes, device=None):
        # aiter now passes the HIP device explicitly (the SP rank is not a device index)
        self.calls.append(("comm_create", device))
        return 1

    def devcomm_info(self, comm, dev):
        # read back by finalize(); keys as in aiter's CcoBackend.devcomm_info
        return {"lsaSize": 4, "lsaRank": 0, "sdmaNumQueue": 1, "worldSize": 4, "commRank": 0,
                "myNodeStart": 0, "sizeofDevComm": 0}

    def win_register(self, comm, nbytes):
        self._n += 1
        self.calls.append(("win", nbytes))
        return 100 + self._n, 1000 + self._n

    def peer_ptr(self, comm, local, peer):
        return local + 0x1000 * (peer + 1)  # distinct fake peer address, not dereferenced

    def abi_check(self):
        # aiter b22268c3 (arena review A-1): (ok, reason) of the MORI library/header hash pin and
        # the compiled ccoWindowDevice layout. The fake has neither, so it reports "ok": the arena
        # policies (now the DEFAULT) are then not refused by the ABI pin. FusionResources calls it
        # only through getattr(backend, "abi_check", None).
        self.calls.append("abi_check")
        return True, "ok"

    def win_free(self, comm, local):
        # aiter c05f49b6: MORI's teardown contract, WindowDeregister -> ccoMemFree(local) before
        # ccoCommDestroy. Recorded so a test can assert the order.
        self.calls.append(("win_free", local))

    def win_view(self, comm, arena_win, offset):
        # aiter 7648ac09 (arena policies): a VIEW struct over one registered arena window at
        # `offset`. The fake remembers (arena handle, offset) so win_info can read it back
        # coherently, exactly what aiter's view self-test (A-1) checks.
        self._n += 1
        h = 5000 + self._n
        self._views[h] = (arena_win, offset)
        self.calls.append(("win_view", offset))
        return h

    def win_view_free(self, comm, view):
        self._views.pop(view, None)
        self.calls.append(("win_view_free", view))

    def win_info(self, comm, win):
        # (winBase, stride4G, lsaRank) of a window or view. A view reads back its arena's base +
        # offset with the arena's stride/lsaRank (the property the self-test proves).
        if win in self._views:
            arena, off = self._views[win]
            base, stride, lsa = self.win_info(comm, arena)
            return (base + off, stride, lsa)
        return (0x7000_0000_0000 + 0x1_0000_0000 * (win % 64), 1 << 32, 0)
    def va_info(self, comm):
        # aiter a2dee878: (slot_base_va, per_rank_size, granule) of this rank's slot, where the
        # first-fit allocator starts. 2 MiB-aligned base, 4 GiB slot, 4 MiB granule: values the
        # abs-aligned policies can plan against without any real comm.
        return (0x7000_0000_0000, 4 << 30, 4 << 20)

    def set_zero_mode(self, mode):
        self.zero_mode = mode  # aiter D14: selects how windows are zeroed; nothing native here

    def mem_info(self, device):
        return (1 << 40, 1 << 40)  # (free, total) bytes: plenty, so no accounting path trips

    def zero(self, device, local, nbytes):
        pass

    def device_sync(self, device):
        pass

    def devcomm_create(self, comm, sdma_queues):
        self.calls.append("devcomm")
        return 7

    def devcomm_destroy(self, comm, dev):
        pass

    def win_deregister(self, comm, win):
        self.calls.append(("win_deregister", win))

    def comm_destroy(self, comm):
        self.calls.append("comm_destroy")


def _fake_torch(monkeypatch):
    """A minimal ``torch`` (+ ``torch.distributed``) so the real FusionResources, which does
    ``import torch.distributed as dist`` when no dist is injected, can be built on a CPU box."""
    dist = _FakeDist()
    tmod = types.ModuleType("torch")
    tmod.cuda = types.SimpleNamespace(current_device=lambda: 0)
    tmod.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", tmod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)


PROFILE_NAME = "minimax_m3_sp4_gfx950_v1"


def _profile_budget(real_gf):
    """window_budget as recorded IN the shipped profile (the source of truth for the asserts)."""
    wb = dict(real_gf.load_profile(PROFILE_NAME).meta.get("window_budget") or {})
    assert wb.get("window_bytes") and wb.get("max_windows"), "shipped profile has no window_budget"
    return wb


def _wrapper_via_real_constructor(real_gf, monkeypatch, backend, profile):
    """ATOM's wrapper building the REAL FusionResources itself (no ``inner=`` injected), so the
    constructor arguments ATOM passes are what is under test. Only the native backend and
    torch.distributed are faked."""
    from atom.model_ops.gemm_fusion import stages
    from atom.model_ops.gemm_fusion.resources import FusionResourcesWrapper

    res_mod = importlib.import_module("aiter.gemm_fusion.resources")
    monkeypatch.setattr(res_mod, "CcoBackend", lambda *a, **k: backend)
    _fake_torch(monkeypatch)
    monkeypatch.setattr(stages, "_aiter_override", real_gf)
    # a real FusionResources takes dist=None -> `import torch.distributed`: serve the fake
    kwargs = dict(rank=0, reduce_min=lambda v: list(v), arch="gfx950", cu=256, device=0)
    if profile is not None:
        kwargs["profile"] = profile
    return FusionResourcesWrapper(None, 4, **kwargs)


def test_window_budget_comes_from_the_config_profile_with_the_env_unset(real_gf, monkeypatch):
    """E2 new_on02: with AITER_GEMM_FUSION_PROFILE unset the resources' uniform window size was
    unknown, so the first (4 MiB, small-M) request sized every window and the first large-M
    plan (40 MiB) was rejected: 'window need 41943040 B exceeds the uniform window size 4194304 B'.
    Through hooks.init_gemm_fusion(config) (no resources injected) the REAL FusionResources must
    get its budget and VMM heap from config.gemm_fusion.profile."""
    from atom.model_ops.gemm_fusion import hooks

    for k in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_WINDOW_MIB",
              "AITER_GEMM_FUSION_WINDOW_POLICY"):
        monkeypatch.delenv(k, raising=False)
    wb = _profile_budget(real_gf)
    backend = _FakeBackend()
    res_mod = importlib.import_module("aiter.gemm_fusion.resources")
    monkeypatch.setattr(res_mod, "CcoBackend", lambda *a, **k: backend)
    _fake_torch(monkeypatch)
    from atom.model_ops.gemm_fusion import resources as atom_res
    from atom.model_ops.gemm_fusion import stages

    monkeypatch.setattr(stages, "_aiter_override", real_gf)
    monkeypatch.setattr(atom_res, "_device", lambda: 0)
    # the wrapper builds a real min-reducer for world > 1; a single process uses identity
    monkeypatch.setattr(atom_res.consensus, "torch_min_reducer", lambda *a, **k: (lambda v: list(v)))
    hooks.reset_for_tests()
    sp_group = types.SimpleNamespace(rank_in_group=0, cpu_group=None, device_group=None)
    monkeypatch.setattr(stages, "sp_world", lambda: 4)
    try:
        hooks.init_gemm_fusion(_config(PROFILE_NAME), sp_group)  # NO resources= injected
        inner = hooks.resources().inner
        assert inner.window_bytes == wb["window_bytes"] == 171966464  # 164 MiB, from the profile
        assert inner.max_windows == wb["max_windows"] == 39
        # heap sized for the budget under the CURRENT default policy. aiter b22268c3 made `arena`
        # the default: ONE registered window of (shared S/R/C + small windows) bytes, and a heap of
        # roundup_4GiB(arena + 256 MiB slack), not the old uniform window_bytes * max_windows.
        assert inner.policy == "arena" and inner.policy_fallback_reason is None
        arena = inner._arena_size_from_budget()
        assert arena == wb["shared_large_total_bytes"] == 398458880  # the profile's own total
        assert inner._vmm >= arena + (256 << 20) and inner._vmm % (4 << 30) == 0
        assert ("comm_create", 0) in backend.calls  # the real constructor ran its comm init
    except ImportError as e:
        pytest.skip(f"resources need the native stack to import: {e}")
    finally:
        hooks.reset_for_tests()


def test_without_the_profile_the_real_resources_reject_the_large_plan(real_gf, monkeypatch):
    """Control: the same real FusionResources built WITHOUT a profile reproduces the E2 failure
    (uniform window sized by the first small request, then the large plan does not fit). Shows the
    test above has power and that the failure is ATOM's omission, not aiter's."""
    for k in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_WINDOW_MIB"):
        monkeypatch.delenv(k, raising=False)
    res_mod = importlib.import_module("aiter.gemm_fusion.resources")
    monkeypatch.setattr(res_mod, "CcoBackend", lambda *a, **k: _FakeBackend())
    bare = real_gf.FusionResources(rank=0, world=4, group=None, dist=_FakeDist(), backend=_FakeBackend(),
                                   device=0, caps=frozenset({"mori_cco"}))
    assert bare.window_bytes in (None, 0)  # budget unknown without a profile
    small = real_gf.WindowReq("o/rx", 4 << 20, "p2p_window")
    large = real_gf.WindowReq("o2/R", 40 << 20, "p2p_window")
    bare.ensure("small", [small])
    with pytest.raises(res_mod.ResourceError, match="exceeds the uniform window size"):
        bare.ensure("large", [large])
    # with the profile the same sequence is accepted
    ok = real_gf.FusionResources(rank=0, world=4, group=None, dist=_FakeDist(), backend=_FakeBackend(),
                                 device=0, caps=frozenset({"mori_cco"}), profile=PROFILE_NAME)
    ok.ensure("small", [small])
    ok.ensure("large", [large])


def test_fake_backend_has_exactly_the_real_ccobackend_signatures(real_gf):
    """The fake stands in for aiter's native CcoBackend. It drifted once (comm_create gained a
    `device` argument and finalize started calling devcomm_info), which the real aiter caught
    and the fake did not. Pin every public method name and parameter list to the real class so
    the next drift fails HERE, with the method named, not as an opaque TypeError mid-test."""
    import inspect

    res_mod = importlib.import_module("aiter.gemm_fusion.resources")
    real = res_mod.CcoBackend

    def sig(cls, name):
        return [p for p in inspect.signature(getattr(cls, name)).parameters if p != "self"]

    public = [n for n, f in inspect.getmembers(real, inspect.isfunction)
              if not n.startswith("_")]
    assert public, "found no public CcoBackend methods (module layout changed?)"
    missing = [n for n in public if not hasattr(_FakeBackend, n)]
    assert not missing, f"_FakeBackend lacks CcoBackend methods: {missing}"
    drift = {n: (sig(real, n), sig(_FakeBackend, n)) for n in public if sig(real, n) != sig(_FakeBackend, n)}
    assert not drift, f"_FakeBackend signatures differ from CcoBackend (real, fake): {drift}"


class _FakeDist:
    def barrier(self, group=None):
        pass

    def all_gather_object(self, out, obj, group=None):
        # single process standing in for ranks that all agree: every slot gets the local value
        for i in range(len(out)):
            out[i] = obj

    def broadcast_object_list(self, obj, src=0, group=None):
        pass


def _real_inner(real_gf, backend):
    from atom.model_ops.gemm_fusion.resources import fusion_resources_cls

    return fusion_resources_cls(real_gf)(
        rank=0, world=4, group=None, dist=_FakeDist(), backend=backend, device=0,
        caps=frozenset({"mori_cco", "gfx950_sc_bypass"}),
    )


def test_wrapper_drives_the_real_fusion_resources_end_to_end(real_gf, monkeypatch):
    """configure()-equivalent: ATOM's wrapper around the REAL FusionResources, a real
    small-M plan from the real catalog/profile, and the real window ownership rules."""
    from atom.model_ops.gemm_fusion import stages
    from atom.model_ops.gemm_fusion.resources import FusionResourcesWrapper, plan_owner

    monkeypatch.setattr(stages, "_aiter_override", real_gf)
    backend = _FakeBackend()
    w = FusionResourcesWrapper(
        None, 4, rank=0, inner=_real_inner(real_gf, backend),
        reduce_min=lambda v: list(v), arch="gfx950", cu=256,  # 1-process identity reduce
    )
    assert w.finalized is False and w.rank == 0 and w.local_m(64) == 16
    assert w.caps == frozenset({"mori_cco", "gfx950_sc_bypass"})

    # a real plan, from the real shipped profile, via the real matcher
    prof = real_gf.load_profile("minimax_m3_sp4_gfx950_v1")
    row = next(r for r in prof.rows() if "small" in r.vid and r.m == 16)
    spec_mod = sys.modules["aiter.ops.gemm_fusion.sp_qkv.specs"] if "aiter.ops.gemm_fusion.sp_qkv.specs" in sys.modules \
        else importlib.import_module("aiter.ops.gemm_fusion.sp_qkv.specs")
    mode = next(m for m in spec_mod.MODES if spec_mod.sp_qkv_spec(m).signature() == row.sig)
    spec = spec_mod.sp_qkv_spec(mode)
    try:
        plan, n = real_gf.match_prefix(spec, w.match_ctx(16, "eager"), "minimax_m3_sp4_gfx950_v1")
    except ImportError as e:
        pytest.skip(f"variants need the native stack to import: {e}")
    assert plan is not None and n == 2 and plan.vid == "sp_qkv.small.v1"

    # THE ownership contract: ATOM must key windows exactly as the plan does
    assert plan_owner(plan) == plan.owner == plan.build_key != plan.vid  # aiter f8a8ebf7: build_key == owner
    assert w.can_ensure(plan) is True
    w.ensure(plan)
    assert plan.owner in w.inner.windows
    assert set(w.inner.windows[plan.owner]) == {plan.owner + "/rx", plan.owner + "/ctl"}
    w.finalize()
    assert w.finalized is True and "devcomm" in backend.calls
    assert w.can_ensure(plan) is True  # same owner after finalize: allowed (idempotent)
    # a plan with a DIFFERENT owner (different params/mode) cannot ensure after finalize
    other = SimpleNamespace(owner="sp_qkv.small/other", vid=plan.vid)
    assert w.can_ensure(other) is False
    w.quiesce_and_reset()
    assert w.inner.poison_local() == 0  # no poison hooks registered yet
    w.release()


def test_plan_attributes_atom_reads_exist_on_the_real_variants(real_gf):
    import ast

    root = AITER_ROOT / "aiter" / "ops" / "gemm_fusion" / "sp_qkv"
    small = ast.parse((root / "variants.py").read_text(encoding="utf-8"))
    large = ast.parse((root / "variants_large.py").read_text(encoding="utf-8"))

    def methods(tree, cls):
        c = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
        return {n.name for n in c.body if isinstance(n, ast.FunctionDef)}

    assert {"resources", "prepare", "run"} <= methods(small, "SpQkvSmall")
    assert {"resources", "prepare", "run", "recv_view", "reuse_view", "poison"} <= methods(large, "SpQkvLarge")
    # ATOM reads plan.owner and plan.mode: both assigned in __init__
    assert "self.owner =" in (root / "variants.py").read_text(encoding="utf-8")
    assert "self.owner =" in (root / "variants_large.py").read_text(encoding="utf-8")
    assert "self.mode =" in (root / "variants_large.py").read_text(encoding="utf-8")


def test_atom_only_calls_plan_methods_that_exist(real_gf):
    """Every `plan.<x>` / `r.plan.<x>` ATOM source calls must be a real plan method."""
    import re

    src = "\n".join(p.read_text(encoding="utf-8")
                    for p in (Path(__file__).resolve().parents[2] / "atom/model_ops/gemm_fusion").glob("*.py")
                    if p.name != "golden_candidate.py")
    called = set(re.findall(r"\b(?:r\.)?plan\.(\w+)\(", src))
    allowed = {"resources", "prepare", "run", "recv_view", "reuse_view", "poison"}
    assert called <= allowed, f"ATOM calls plan methods aiter does not define: {sorted(called - allowed)}"


def test_real_matcher_builds_a_new_plan_per_call_but_the_site_keeps_one(real_gf, monkeypatch):
    """gate04 root cause, on the REAL aiter classes: ``match_prefix`` returns a distinct plan
    (with its own pointer-binding dict) on every call, so a plan prepared for 'eager' is not
    the plan a 'capture' lookup would get. The site must hand back ONE instance."""
    from atom.model_ops.gemm_fusion import hooks, stages
    from atom.model_ops.gemm_fusion import site as site_mod
    from atom.model_ops.gemm_fusion.resources import FusionResourcesWrapper

    spec_mod = importlib.import_module("aiter.ops.gemm_fusion.sp_qkv.specs")
    spec = spec_mod.sp_qkv_spec("dense")
    ctx = real_gf.MatchCtx(arch="gfx950", cu=256, world=4, m=16,
                           available=frozenset({"mori_cco", "gfx950_sc_bypass"}), graph_mode="eager")
    real_match = real_gf.match_prefix
    try:
        a, _ = real_match(spec, ctx, "minimax_m3_sp4_gfx950_v1")
        b, _ = real_match(spec, ctx, "minimax_m3_sp4_gfx950_v1")
    except ImportError as e:
        pytest.skip(f"variants need the native stack to import: {e}")
    assert a is not b and a._bindings is not b._bindings  # the hazard is real
    a._bindings[("probe",)] = "bound by prepare"
    assert ("probe",) not in b._bindings

    # through the REAL runtime path: hooks.init_gemm_fusion(config) -> site.resolve, on the real
    # aiter package, with NO shim and the profile env var unset (so only the config can supply it)
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    monkeypatch.setattr(stages, "_aiter_override", real_gf)
    w = FusionResourcesWrapper(None, 4, rank=0, inner=_real_inner(real_gf, _FakeBackend()),
                               reduce_min=lambda v: list(v), arch="gfx950", cu=256)
    hooks.reset_for_tests()
    hooks.init_gemm_fusion(_config("minimax_m3_sp4_gfx950_v1"), None, resources=w)
    hooks.set_phase("preparing")
    site = _dense_site(spec)
    eager = site.resolve(64, "eager")
    capture = site.resolve(64, "capture")
    hooks.reset_for_tests()
    assert eager.plan is not None and eager.n_fused == 2 and eager.reason == "profile_hit"
    assert eager.plan is capture.plan  # one instance => one set of bindings for eager+capture+replay


def _config(profile):
    """What ModelRunner hands init_gemm_fusion: only ``config.gemm_fusion.profile`` matters."""
    return SimpleNamespace(gemm_fusion=SimpleNamespace(profile=profile, enabled=bool(profile)))


def _dense_site(spec):
    from atom.model_ops.gemm_fusion import site as site_mod

    impl = SimpleNamespace(gemm_fusion_prologue_supported=lambda: True)
    site = site_mod.GemmFusionSite("t.qkv", SimpleNamespace(output_size=9216, input_size=6144),
                                   list(spec.stages), None, attn=SimpleNamespace(impl=impl),
                                   q_size=2048, kv_size=128, register=False)
    site.spec = spec
    return site


def _wrapper(real_gf):
    from atom.model_ops.gemm_fusion.resources import FusionResourcesWrapper

    return FusionResourcesWrapper(None, 4, rank=0, inner=_real_inner(real_gf, _FakeBackend()),
                                  reduce_min=lambda v: list(v), arch="gfx950", cu=256)


def test_runtime_path_takes_the_profile_from_config_not_from_the_environment(real_gf, monkeypatch):
    """E2 gate failure (job-e2_new_on01): bind_kv_cache -> prepare_buckets -> site.resolve ->
    match_prefix raised ProfileError 'no profile given and AITER_GEMM_FUSION_PROFILE is unset'
    although --gemm-fusion-profile had been passed. Reproduces that path on the REAL aiter
    package with the env var unset: init stores the config profile, bind_kv_cache's
    profile_tokens and every resolve use it."""
    from atom.model_ops.gemm_fusion import hooks, stages

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    monkeypatch.setattr(stages, "_aiter_override", real_gf)
    spec_mod = importlib.import_module("aiter.ops.gemm_fusion.sp_qkv.specs")
    hooks.reset_for_tests()
    hooks.init_gemm_fusion(_config("minimax_m3_sp4_gfx950_v1"), None, resources=_wrapper(real_gf))
    try:
        assert hooks.profile() == "minimax_m3_sp4_gfx950_v1"
        toks = hooks.profile_tokens(4)  # what bind_kv_cache enumerates; used to read the env
        assert toks and all(t % 4 == 0 for t in toks)
        assert 64 in toks and 128 in toks  # the small-M golden cells
        hooks.set_phase("preparing")
        r = _dense_site(spec_mod.sp_qkv_spec("dense")).resolve(64, "eager")
        assert (r.reason, r.n_fused) == ("profile_hit", 2)
    except ImportError as e:
        pytest.skip(f"variants need the native stack to import: {e}")
    finally:
        hooks.reset_for_tests()


def test_init_fails_fast_when_the_configured_profile_cannot_load(real_gf, monkeypatch):
    """A bad --gemm-fusion-profile must stop start-up inside init_gemm_fusion (aiter's
    ProfileError), before any CCO resource is built and long before bind_kv_cache."""
    from atom.model_ops.gemm_fusion import hooks, stages

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    monkeypatch.setattr(stages, "_aiter_override", real_gf)
    hooks.reset_for_tests()
    built = []

    class _Boom:
        def __init__(self, *a, **k):
            built.append(1)

    import atom.model_ops.gemm_fusion.resources as R

    monkeypatch.setattr(R, "FusionResourcesWrapper", _Boom)
    with pytest.raises(real_gf.ProfileError):
        hooks.init_gemm_fusion(_config("no_such_profile_xyz"), None)
    assert built == [] and hooks.resources() is None  # no resource was created
    with pytest.raises(RuntimeError, match="no profile configured"):
        hooks.profile()  # and nothing was stored


def test_env_var_does_not_substitute_for_the_config_profile(real_gf, monkeypatch):
    """With the env var SET to a valid profile but no config profile, ATOM must not silently
    use it: that would mask a broken CLI -> runtime path (the very bug above)."""
    from atom.model_ops.gemm_fusion import hooks, stages

    monkeypatch.setenv("AITER_GEMM_FUSION_PROFILE", "minimax_m3_sp4_gfx950_v1")
    monkeypatch.setattr(stages, "_aiter_override", real_gf)
    hooks.reset_for_tests(_wrapper(real_gf))
    hooks._PROFILE = ""  # resources exist but init never recorded a profile
    hooks.set_phase("preparing")
    spec_mod = importlib.import_module("aiter.ops.gemm_fusion.sp_qkv.specs")
    try:
        with pytest.raises(RuntimeError, match="no profile configured"):
            _dense_site(spec_mod.sp_qkv_spec("dense")).resolve(64, "eager")
        with pytest.raises(RuntimeError, match="no profile configured"):
            hooks.profile_tokens(4)
    finally:
        hooks.reset_for_tests()

class _Tens:
    """Tensor-like with exactly what aiter's alias walker reads (data_ptr/shape/stride/element_size)."""

    def __init__(self, ptr, n):
        self._p, self.shape, self._n = ptr, (n,), n

    def data_ptr(self):
        return self._p

    def stride(self):
        return (1,)

    def element_size(self):
        return 1


def test_check_no_window_alias_drives_the_real_aiter_helper_on_real_windows(real_gf, monkeypatch):
    """C-S4 against aiter 7c857474's REAL FusionResources.assert_no_window_alias and window
    registry: a tensor inside a registered shared window is rejected, outside is accepted, and the
    verdict comes from aiter's walker (force=True), not from ATOM's fallback."""
    from atom.model_ops.gemm_fusion import hooks, stages
    from atom.model_ops.gemm_fusion import site as site_mod
    from atom.model_ops.gemm_fusion.resources import FusionResourcesWrapper

    monkeypatch.delenv("AITER_GEMM_FUSION_DEBUG_ALIAS", raising=False)  # ATOM must force, not rely on env
    monkeypatch.setattr(stages, "_aiter_override", real_gf)
    inner = _real_inner(real_gf, _FakeBackend())
    if not hasattr(inner, "assert_no_window_alias"):
        pytest.skip("installed aiter predates assert_no_window_alias (7c857474)")
    w = FusionResourcesWrapper(None, 4, rank=0, inner=inner, reduce_min=lambda v: list(v), arch="gfx950", cu=256)
    hooks.reset_for_tests(w)
    try:
        # register one SHARED window through aiter's own bookkeeping
        base, size = 0x7000_0000, 1 << 20
        inner._regs.append(real_gf.resources.Window("<shared>", "R", "p2p_window", size, 1, base, shared=True)) \
            if hasattr(real_gf, "resources") else inner._regs.append(
                importlib.import_module("aiter.gemm_fusion.resources").Window(
                    "<shared>", "R", "p2p_window", size, 1, base, shared=True))
        hooks.check_no_window_alias([("out", _Tens(base + size + 4096, 64)), ("scale", _Tens(0x1000, 8))])
        with pytest.raises(site_mod.GemmFusionBindError, match="points into a CCO window"):
            hooks.check_no_window_alias([("out", _Tens(base + 4096, 64))])
        with pytest.raises(site_mod.GemmFusionBindError, match="impl._index_q"):
            hooks.check_no_window_alias([("impl._index_q", _Tens(base - 16, 64))])  # straddles the start
    finally:
        hooks.reset_for_tests()

def _built_resources(real_gf, backend, **kw):
    return real_gf.FusionResources(rank=0, world=4, group=None, dist=_FakeDist(), backend=backend,
                                   device=0, caps=frozenset({"mori_cco"}), profile=PROFILE_NAME, **kw)


def test_release_frees_every_window_after_its_deregister_and_before_comm_destroy(real_gf, monkeypatch):
    """aiter c05f49b6 (MORI's teardown contract): WindowDeregister -> ccoMemFree(local) for EVERY
    registered window, all before ccoCommDestroy. Drives the REAL FusionResources.release() with
    the strict fake backend and reads the call order back; this is also the behaviour Evidence's
    MemFree-before-CommDestroy A/B exercises on the GPU."""
    monkeypatch.delenv("AITER_GEMM_FUSION_FREE_BEFORE_DESTROY", raising=False)  # default ON
    be = _FakeBackend()
    res = _built_resources(real_gf, be)
    res.ensure("small", [real_gf.WindowReq("o/rx", 4 << 20, "p2p_window")])
    res.ensure("large", [real_gf.WindowReq("o2/R", 40 << 20, "p2p_window")])
    registered = [w for w in res._regs]
    assert len(registered) >= 1  # arena (default): ONE window; per-owner policies: several
    res.release()
    calls = be.calls
    dest = calls.index("comm_destroy")
    assert dest == len(calls) - 1  # comm_destroy is last
    for w in registered:
        d = calls.index(("win_deregister", w.handle))
        f = calls.index(("win_free", w.local))
        assert d < f < dest, (w.name, d, f, dest)  # deregister, then free, then destroy
    assert sum(1 for c in calls if isinstance(c, tuple) and c[0] == "win_free") == len(registered)  # once each
    res.release()  # idempotent: no second free, no second destroy
    assert be.calls == calls


def test_release_skips_win_free_when_the_ab_flag_turns_it_off(real_gf, monkeypatch):
    monkeypatch.setenv("AITER_GEMM_FUSION_FREE_BEFORE_DESTROY", "0")  # the old behaviour (A/B arm)
    be = _FakeBackend()
    res = _built_resources(real_gf, be)
    res.ensure("small", [real_gf.WindowReq("o/rx", 4 << 20, "p2p_window")])
    res.release()
    assert not any(isinstance(c, tuple) and c[0] == "win_free" for c in be.calls)
    assert "comm_destroy" in be.calls


def test_atom_shutdown_releases_the_comm_and_is_what_the_old_harnesses_never_called(real_gf, monkeypatch):
    """Evidence observed that the old uniform-gate / formal harnesses never called
    hooks.shutdown()/release, so no native comm destroy ran at exit. ATOM's runner calls
    gemm_fusion.shutdown() from ModelRunner.exit; this shows shutdown() reaches the real release()
    and therefore the free-before-destroy sequence, and that it does so once."""
    from atom.model_ops.gemm_fusion import hooks, stages
    from atom.model_ops.gemm_fusion.resources import FusionResourcesWrapper

    monkeypatch.delenv("AITER_GEMM_FUSION_FREE_BEFORE_DESTROY", raising=False)
    monkeypatch.setattr(stages, "_aiter_override", real_gf)
    be = _FakeBackend()
    inner = _built_resources(real_gf, be)
    inner.ensure("small", [real_gf.WindowReq("o/rx", 4 << 20, "p2p_window")])
    w = FusionResourcesWrapper(None, 4, rank=0, inner=inner, reduce_min=lambda v: list(v), arch="gfx950", cu=256)
    hooks.reset_for_tests(w)
    hooks.shutdown()
    assert be.calls.count("comm_destroy") == 1 and any(c[0] == "win_free" for c in be.calls if isinstance(c, tuple))
    assert hooks.resources() is None
    hooks.shutdown()  # idempotent
    assert be.calls.count("comm_destroy") == 1

def test_explicit_uniform_policy_keeps_the_phase1_heap_for_the_profile_budget(real_gf, monkeypatch):
    """The pre-arena behaviour stays selectable (AITER_GEMM_FUSION_WINDOW_POLICY=uniform): the heap
    is sized for window_bytes * max_windows from the profile budget, with no env var carrying the
    profile."""
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "uniform")
    wb = _profile_budget(real_gf)
    res = _built_resources(real_gf, _FakeBackend())
    assert res.policy == "uniform" and res.policy_source == "explicit"
    assert res.window_bytes == wb["window_bytes"] and res.max_windows == wb["max_windows"]
    assert res._vmm >= res.window_bytes * res.max_windows

def test_release_order_holds_for_a_multi_window_policy_too(real_gf, monkeypatch):
    """Same teardown contract with several registered windows (explicit per_owner policy), so the
    one-window arena default does not hide a per-window ordering bug."""
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "uniform")
    monkeypatch.delenv("AITER_GEMM_FUSION_FREE_BEFORE_DESTROY", raising=False)
    be = _FakeBackend()
    res = _built_resources(real_gf, be)
    res.ensure("small", [real_gf.WindowReq("o/rx", 4 << 20, "p2p_window")])
    res.ensure("large", [real_gf.WindowReq("o2/R", 40 << 20, "p2p_window")])
    registered = list(res._regs)
    assert len(registered) >= 2
    res.release()
    calls = be.calls
    dest = calls.index("comm_destroy")
    for w in registered:
        assert calls.index(("win_deregister", w.handle)) < calls.index(("win_free", w.local)) < dest
    assert sum(1 for c in calls if isinstance(c, tuple) and c[0] == "win_free") == len(registered)
