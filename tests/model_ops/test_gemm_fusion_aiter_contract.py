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
        self.calls, self._n = [], 0

    def uid(self, rank):
        return b"u" * 128

    def comm_create(self, rank, world, uid, vmm):
        self.calls.append("comm_create")
        return 1

    def win_register(self, comm, nbytes):
        self._n += 1
        self.calls.append(("win", nbytes))
        return 100 + self._n, 1000 + self._n

    def zero(self, device, local, nbytes):
        pass

    def device_sync(self, device):
        pass

    def devcomm_create(self, comm, queues):
        self.calls.append("devcomm")
        return 7

    def devcomm_destroy(self, comm, dev):
        pass

    def win_deregister(self, comm, win):
        pass

    def comm_destroy(self, comm):
        pass


class _FakeDist:
    def barrier(self, group=None):
        pass

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
