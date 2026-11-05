# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests: QSA index scorer call-site route, drift guards, aiter contract.

No GPU, no Triton. The route is tested against a fake aiter module; the
contract tests load the real ``aiter.ops.qsa_index_scorer`` (installed aiter,
or ``$QSA_AITER_ROOT`` = an aiter checkout loaded standalone) and skip when
neither is available.
"""

from __future__ import annotations

import ast
import enum
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.qwen4_exp.qsa_index_scorer import capability as CAP
from atom.model_ops.qwen4_exp.qsa_index_scorer import capture as C
from atom.model_ops.qwen4_exp.qsa_index_scorer import route as R

ROOT = Path(__file__).resolve().parents[2]
QSA_PY = ROOT / "atom" / "model_ops" / "qwen4_exp" / "ops" / "qsa.py"
SELECT_PY = ROOT / "atom" / "model_ops" / "qwen4_exp" / "qsa_index_scorer" / "select.py"
ATTN_PY = ROOT / "atom" / "model_ops" / "qwen4_exp" / "qsa_attention.py"
# The frozen golden scorer source aiter's contract is written against
# (aiter ``ATOM_SCORER_SOURCE_SHA256``; ATOM 8a07708's qsa.py): aaa78f43... is
# its sha256 with CRLF line endings (the golden_baseline copy and a Windows
# autocrlf checkout); c2e46e50... is the same content with LF (the git blob).
FROZEN_QSA_SHA256 = "aaa78f43255cf950bfbdf5d618246f9a7a671f0262cff2398b2ca4ea703810ee"
FROZEN_QSA_SHA256_LF = (
    "c2e46e50cd9b8bdbdc4b6a09b29cafc5c971dfff4246799a3c8dc32f8b4d14b9"
)


# ---------------------------------------------------------------------------
# Drift guards against ATOM's own frozen scorer (AST only: no Triton needed)
# ---------------------------------------------------------------------------
def _func(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def test_qsa_py_is_the_frozen_scorer():
    """ATOM ab4ada40's qsa.py is still the frozen golden scorer aiter's
    contract names (unchanged since 8a07708), line-ending independent."""
    lf = QSA_PY.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(lf).hexdigest() == FROZEN_QSA_SHA256_LF
    crlf = lf.replace(b"\n", b"\r\n")
    assert hashlib.sha256(crlf).hexdigest() == FROZEN_QSA_SHA256


def test_native_params_match_qsa_py():
    fn = _func(QSA_PY, "qsa_paged_mqa_logits")
    names = [a.arg for a in fn.args.args]
    defaults = [ast.literal_eval(d) for d in fn.args.defaults]
    defaults = [R._REQUIRED] * (len(names) - len(defaults)) + defaults
    assert tuple(zip(names, defaults)) == R.NATIVE_SCORER_PARAMS
    assert not fn.args.kwonlyargs


def test_contract_params_are_the_native_params():
    # The aiter op is a drop-in for this tree's scorer, scale plane included.
    assert R.SCORER_PARAMS == R.NATIVE_SCORER_PARAMS
    assert R.SCORER_PARAMS[-1] == ("compressed_k_scale", None)


class _RenameScorer(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "qsa_paged_mqa_logits":
            return ast.copy_location(ast.Name(id="scorer", ctx=node.ctx), node)
        return node


def _body_without_docstring(fn):
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    return body


def test_routed_select_is_stock_select_with_scorer_injected():
    stock = _func(QSA_PY, "qsa_select_paged_tokens")
    routed = _func(SELECT_PY, "qsa_select_paged_tokens_routed")
    want = [ast.dump(_RenameScorer().visit(s)) for s in _body_without_docstring(stock)]
    got = [ast.dump(s) for s in _body_without_docstring(routed)]
    assert (
        got == want
    ), "select.py drifted from qsa.py::qsa_select_paged_tokens; re-sync it"
    assert [a.arg for a in routed.args.args] == [a.arg for a in stock.args.args]
    assert [ast.dump(d) for d in routed.args.defaults] == [
        ast.dump(d) for d in stock.args.defaults
    ]
    assert [a.arg for a in routed.args.kwonlyargs] == ["scorer"]


def test_attention_imports_route_only_when_active():
    """OFF => the route package (and aiter's module) is never imported."""
    tree = ast.parse(ATTN_PY.read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for n in top:
        mod = getattr(n, "module", "") or ""
        assert "qsa_index_scorer" not in mod
    src = ATTN_PY.read_text(encoding="utf-8")
    assert "if scorer_cfg is not None and scorer_cfg.active:" in src
    assert "from atom.model_ops.qwen4_exp.qsa_index_scorer import build_route" in src
    assert "qsa_select_paged_tokens\n            if self.qsa_scorer is None" in src


def test_route_package_imports_nothing_heavy():
    """Importing the route package adds no aiter / Triton module (fresh process)."""
    import subprocess

    code = (
        "import sys, atom; before = set(sys.modules); "
        "import atom.model_ops.qwen4_exp.qsa_index_scorer.route, "
        "atom.model_ops.qwen4_exp.qsa_index_scorer.capture; "
        "new = set(sys.modules) - before; "
        "bad = sorted(m for m in new if m.split('.')[0] in ('aiter', 'triton', 'flydsl')); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, res.stdout + res.stderr


# ---------------------------------------------------------------------------
# Fake aiter module
# ---------------------------------------------------------------------------
class FakeReason(str, enum.Enum):
    OK = "ok"
    DISABLED = "disabled"
    CACHE_SHAPE = "cache_shape_unsupported"
    COLUMNS = "columns_uncovered"
    ROWS_RULE = "rows_rule_uncovered"


@dataclass(frozen=True)
class FakeDecision:
    ok: bool
    reason: FakeReason
    detail: str = ""


class FakeUnsupported(RuntimeError):
    def __init__(self, reason):
        super().__init__(f"qsa_index_scorer: {reason.value}")
        self.reason = reason


def make_fake_api(
    layout=FakeReason.OK,
    decide=FakeReason.OK,
    raise_on_call=None,
    version="1.0",
    prewarm_raises=None,
):
    """Fake ``aiter.ops.qsa_index_scorer``; ``version >= "1.1"`` adds the
    contract v1.1 ``prewarm`` and ``launch``."""
    calls = {
        "layout": 0,
        "decide": 0,
        "fused": 0,
        "launch": 0,
        "prewarm": [],
        "decisions": [],
        "profiles": [],
    }

    def layout_eligibility(cache, scale=None, *, profile=None):
        calls["layout"] += 1
        calls["profiles"].append(profile)
        return FakeDecision(layout is FakeReason.OK, layout, "layout")

    def _decide(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=4,
        score_divisor=None,
        max_columns=None,
        row_starts=None,
        allow_flydsl=False,
        compressed_k_scale=None,
        *,
        logits=None,
        visible_groups=None,
        profile=None,
        allow_unverified=False,
    ):
        calls["decide"] += 1
        r = decide(q, max_columns) if callable(decide) else decide
        d = FakeDecision(r is FakeReason.OK, r, "decide")
        calls["decisions"].append(d)
        return d

    def qsa_paged_mqa_logits(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=4,
        score_divisor=None,
        max_columns=None,
        row_starts=None,
        allow_flydsl=False,
        compressed_k_scale=None,
        *,
        logits=None,
        visible_groups=None,
        profile=None,
    ):
        calls["fused"] += 1
        calls["profiles"].append(profile)
        if raise_on_call is not None:
            raise FakeUnsupported(raise_on_call)
        return ("fused", q.shape[0])

    def launch(
        decision,
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=4,
        score_divisor=None,
        max_columns=None,
        row_starts=None,
        allow_flydsl=False,
        compressed_k_scale=None,
        *,
        logits=None,
        visible_groups=None,
    ):
        calls["launch"] += 1
        calls["launched"] = decision
        assert decision.ok
        if raise_on_call is not None:
            raise FakeUnsupported(raise_on_call)
        return ("launch", q.shape[0])

    def prewarm(
        profile=None,
        *,
        cache_dtypes=("bf16",),
        pos_dtypes=(torch.int32, torch.int64),
        device=None,
    ):
        calls["prewarm"].append(
            {
                "profile": profile,
                "cache_dtypes": cache_dtypes,
                "pos_dtypes": pos_dtypes,
                "device": device,
            }
        )
        if prewarm_raises is not None:
            raise prewarm_raises
        return ["r32_persist", "decg", "draft"]

    api = SimpleNamespace(
        API_VERSION=version,
        Reason=FakeReason,
        QSAIndexScorerUnsupported=FakeUnsupported,
        layout_eligibility=layout_eligibility,
        decide=_decide,
        qsa_paged_mqa_logits=qsa_paged_mqa_logits,
    )
    if R.api_version_tuple(version) >= (1, 1):
        api.launch = launch
        api.prewarm = prewarm
    return api, calls


@pytest.fixture(autouse=True)
def fresh_process_state(monkeypatch):
    """Each test is a fresh process for the once-per-process prewarm."""
    monkeypatch.setattr(R, "_PREWARMED", {})
    monkeypatch.setattr(CAP, "_LAUNCH_DRIFT_LOGGED", set())
    monkeypatch.setattr(CAP, "_CAPABILITIES", {})
    # No exit-summary hook from tests (pytest has closed its streams by then).
    monkeypatch.setattr(R, "_SUMMARY_REGISTERED", True)


class FakeNative:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        assert len(args) == len(R.NATIVE_SCORER_PARAMS) == 12
        self.calls.append(args)
        return ("native", args[0].shape[0])


def call_args(
    rows=8, requests=2, width=16, page=4, max_columns=64, dtype=torch.bfloat16
):
    q = torch.zeros(rows, 4, 128, dtype=dtype)
    cache = torch.zeros(32, page, 1, 128, dtype=torch.bfloat16)
    table = torch.zeros(requests, width, dtype=torch.int32)
    t2r = torch.zeros(rows, dtype=torch.int32)
    pos = torch.arange(rows, dtype=torch.int64)
    ctx = torch.full((requests,), 64, dtype=torch.int32)
    return q, cache, table, t2r, pos, ctx, max_columns


def select_style_call(route, a, flydsl=False, scale=None):
    """Exactly how the stock select loop calls the scorer."""
    q, cache, table, t2r, pos, ctx, cols = a
    rs = torch.empty(q.shape[0], dtype=torch.int32)
    return route(
        q,
        cache,
        table,
        t2r,
        pos,
        ctx,
        4,
        max_columns=cols,
        row_starts=rs,
        allow_flydsl=flydsl,
        compressed_k_scale=scale,
    )


class _BlockAiter:
    """sys.meta_path finder refusing any aiter import (proves 'nothing imported')."""

    def find_spec(self, name, path=None, target=None):
        if name == "aiter" or name.startswith("aiter."):
            raise ImportError(f"blocked: {name}")


@pytest.fixture
def block_aiter(monkeypatch):
    saved = {
        k: v for k, v in sys.modules.items() if k == "aiter" or k.startswith("aiter.")
    }
    for k in saved:
        monkeypatch.delitem(sys.modules, k)
    blocker = _BlockAiter()
    sys.meta_path.insert(0, blocker)
    yield
    sys.meta_path.remove(blocker)


# ---------------------------------------------------------------------------
# Route behaviour
# ---------------------------------------------------------------------------
def test_build_route_off_is_none_and_imports_nothing(block_aiter):
    assert R.build_route(None, site="x") is None
    assert (
        R.build_route(SimpleNamespace(active=False, profile="", capture=""), site="x")
        is None
    )
    assert not any(k == "aiter" or k.startswith("aiter.") for k in sys.modules)


def test_capture_only_route_never_imports_aiter(block_aiter):
    native = FakeNative()
    route = R.QSAScorerRoute("", site="L0", native=native)
    assert route.pinned[0] == R.CAPTURE_ONLY and route.api is None
    out = select_style_call(route, call_args())
    assert out == ("native", 8) and len(native.calls) == 1
    assert route.counts[("stock", R.CAPTURE_ONLY)] == 1
    assert not any(k == "aiter" or k.startswith("aiter.") for k in sys.modules)


def test_aiter_unavailable_pins_native(block_aiter):
    native = FakeNative()
    route = R.QSAScorerRoute("qsa_page4_gfx1250_v1", site="L0", native=native)
    assert route.pinned[0] == R.AITER_UNAVAILABLE
    assert route.prepare(call_args()[1]) == R.AITER_UNAVAILABLE
    assert select_style_call(route, call_args()) == ("native", 8)
    assert route.counts[("stock", R.AITER_UNAVAILABLE)] == 1


def test_api_version_mismatch_pins_native():
    api, calls = make_fake_api(version="2.0")
    route = R.QSAScorerRoute("p", api=api, native=FakeNative())
    assert route.pinned[0] == R.API_VERSION_MISMATCH
    select_style_call(route, call_args())
    assert calls["decide"] == 0 and calls["fused"] == 0


def test_signature_drift_pins_native():
    api, _calls = make_fake_api()

    def drifted(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=4,
        max_columns=None,
        *,
        profile=None,
    ):
        raise AssertionError("must not be called")

    api.qsa_paged_mqa_logits = drifted
    route = R.QSAScorerRoute("p", api=api, native=FakeNative())
    assert route.pinned[0] == R.SIGNATURE_DRIFT
    assert select_style_call(route, call_args()) == ("native", 8)


def test_fused_when_covered_passes_profile_and_all_args():
    api, calls = make_fake_api()
    native = FakeNative()
    route = R.QSAScorerRoute("qsa_page4_gfx1250_v1", site="L3", api=api, native=native)
    assert route.pinned is None
    assert route.prepare(call_args()[1]) == "ok"
    assert select_style_call(route, call_args()) == ("fused", 8)
    assert not native.calls
    assert set(calls["profiles"]) == {"qsa_page4_gfx1250_v1"}
    assert route.counts[("fused", "ok")] == 1


def test_decision_cached_once_per_class():
    api, calls = make_fake_api()
    route = R.QSAScorerRoute("p", api=api, native=FakeNative())
    a = call_args()
    route.prepare(a[1])
    for _ in range(5):
        select_style_call(route, a)
    assert calls["decide"] == 1 and calls["layout"] == 1 and calls["fused"] == 5
    select_style_call(route, call_args(max_columns=32))  # new bucket => one more decide
    assert calls["decide"] == 2
    b = call_args(rows=16)  # new rows bucket, same cache tensor identity? (new tensor)
    select_style_call(route, b)
    assert calls["decide"] == 3


def test_static_layout_reason_pins_cache_without_decide():
    api, calls = make_fake_api(layout=FakeReason.CACHE_SHAPE)
    native = FakeNative()
    route = R.QSAScorerRoute("p", api=api, native=native)
    a = call_args(page=16)
    assert route.prepare(a[1]) == "cache_shape_unsupported"
    for _ in range(3):
        assert select_style_call(route, a) == ("native", 8)
    assert calls["decide"] == 0 and calls["layout"] == 1 and calls["fused"] == 0
    assert route.counts[("stock", "cache_shape_unsupported")] == 3


def test_uncovered_call_runs_native_with_reason_and_native_args_unchanged():
    api, _calls = make_fake_api(decide=FakeReason.COLUMNS)
    native = FakeNative()
    route = R.QSAScorerRoute("p", api=api, native=native)
    a = call_args()
    scale = None
    select_style_call(route, a, flydsl=True, scale=scale)
    (args,) = native.calls
    q, cache, table, t2r, pos, ctx, cols = a
    assert args[:6] == (q, cache, table, t2r, pos, ctx)
    assert args[6:9] == (4, None, cols) and args[10] is True and len(args) == 12
    assert route.counts[("stock", "columns_uncovered")] == 1


def test_unsupported_raised_at_call_falls_back_and_repins():
    api, calls = make_fake_api(raise_on_call=FakeReason.ROWS_RULE)
    native = FakeNative()
    route = R.QSAScorerRoute("p", api=api, native=native)
    a = call_args()
    assert select_style_call(route, a) == ("native", 8)
    assert select_style_call(route, a) == ("native", 8)
    assert calls["fused"] == 1  # second call does not retry the fused op
    assert route.counts[("stock", "rows_rule_uncovered")] == 2


def test_other_fused_exceptions_propagate():
    api, _ = make_fake_api()

    def boom(*a, **k):
        raise RuntimeError("kernel fault")

    boom.__signature__ = inspect.signature(api.qsa_paged_mqa_logits)
    api.qsa_paged_mqa_logits = boom
    route = R.QSAScorerRoute("p", api=api, native=FakeNative())
    with pytest.raises(RuntimeError, match="kernel fault"):
        select_style_call(route, call_args())


def test_one_log_line_per_site_and_reason(caplog):
    def decide(q, cols):
        return FakeReason.OK if cols == 64 else FakeReason.COLUMNS

    api, _ = make_fake_api(decide=decide)
    route = R.QSAScorerRoute(
        "p", site="model.layers.3.self_attn", api=api, native=FakeNative()
    )
    with caplog.at_level(logging.INFO, logger="atom"):
        for cols in (64, 64, 32, 32, 16, 64):
            select_style_call(route, call_args(max_columns=cols))
    lines = [
        r.getMessage() for r in caplog.records if "QSA index scorer" in r.getMessage()
    ]
    assert len(lines) == 2
    assert any("-> fused (ok)" in m for m in lines)
    assert any("-> stock (columns_uncovered)" in m for m in lines)
    assert all("model.layers.3.self_attn" in m for m in lines)


def test_binding_matches_native_signature():
    a = call_args()
    q, cache, table, t2r, pos, ctx, cols = a
    b1 = R.QSAScorerRoute._bind((q, cache, table, t2r, pos, ctx), {"max_columns": cols})
    assert len(b1) == 12 and b1[6] == 4 and b1[8] == cols and b1[10] is False
    with pytest.raises(TypeError):
        R.QSAScorerRoute._bind((q,), {})
    with pytest.raises(TypeError):
        R.QSAScorerRoute._bind((q, cache, table, t2r, pos, ctx), {"bogus": 1})


def test_decision_independent_of_values_and_q_base():
    """Rank-uniformity: same class on another rank (other values / q base) => same key."""
    a = call_args()
    b = call_args()
    b[0].fill_(3.0)
    k = lambda x: R.call_class_key(
        x[0], x[1], x[2], x[3], x[4], x[5], 4, None, x[6], None, None
    )
    ka, kb = k(a), k(b)
    i = R.CLASS_KEY_LAYOUT_INDEX
    assert ka[:i] == kb[:i] and ka[i + 1 :] == kb[i + 1 :]
    assert ka[i][1:] == kb[i][1:]  # caches differ only by address


def test_build_route_with_capture_writes_json():
    import tempfile

    base = os.environ.get("QSA_TEST_TMP")  # sandboxed hosts: a writable dir
    if base:  # mkdtemp's 0o700 is unwritable under some Windows sandboxes
        tmp_path = Path(base) / f"qsa_capture_{os.getpid()}_{id(base)}"
        os.makedirs(tmp_path, exist_ok=True)
    else:
        tmp_path = Path(tempfile.mkdtemp(prefix="qsa_capture_"))
    api, _ = make_fake_api()
    path = str(tmp_path / "cap_{rank}.json")
    cfg = SimpleNamespace(active=True, profile="p", capture=path)
    route = R.build_route(cfg, site="L0", api=api, native=FakeNative())
    # One set of tensors: the signature carries the cache's base alignment,
    # which fresh CPU allocations do not keep constant.
    a4 = call_args(rows=4, requests=1, width=16, max_columns=64)
    for _ in range(3):
        select_style_call(route, a4)
    select_style_call(route, call_args(rows=6, requests=1))
    route.capture.flush()
    data = json.loads((tmp_path / "cap_0.json").read_text())
    assert data["schema"] == C.SCHEMA and len(data["classes"]) == 2
    first = next(c for c in data["classes"] if c["signature"]["rows"] == 4)
    assert (
        first["count"] == 3
        and first["sites"] == ["L0"]
        and first["decisions"] == {"fused:ok": 3}
    )
    sig = first["signature"]
    assert sig["regime"] == "decode" and sig["columns"] == 64 and sig["page_size"] == 4
    assert sig["q"]["base_mod_16"] == 0 or isinstance(sig["q"]["base_mod_16"], int)
    assert C.golden_case_of(sig) == "I5_min_decode_bf16"
    assert {
        "cache_dtype": "bf16",
        "rows": 4,
        "requests": 1,
        "columns": 64,
        "table_width": 16,
        "cache_pages": 32,
    } in data["plan"]
    # merge CLI -> plan file (aiter gen_profile --plan-file) + coverage matrix
    plan, matrix = tmp_path / "plan.json", tmp_path / "matrix.json"
    assert (
        C.main(
            [
                "merge",
                str(tmp_path / "cap_*.json"),
                "--plan",
                str(plan),
                "--matrix",
                str(matrix),
            ]
        )
        == 0
    )
    assert len(json.loads(plan.read_text())) == 2
    m = json.loads(matrix.read_text())["matrix"]
    assert {r["status"] for r in m} == {"covered", "uncovered:extend-golden"}
    assert all(r["decision_uniform"] for r in m)


def test_regime_mirror():
    assert R.regime_of(16, 16) == "draft"
    assert R.regime_of(64, 16) == "decode"
    assert R.regime_of(512, 2) == "prefill"
    assert R.regime_of(6, 1) is None


# ---------------------------------------------------------------------------
# Contract v1.1: prewarm at prepare, launch(cached_decision) per call
# ---------------------------------------------------------------------------
P = "qsa_page4_gfx1250_v1"


def test_api_version_tuple():
    assert R.api_version_tuple("1.1") == (1, 1) >= R.LAUNCH_API_VERSION
    assert R.api_version_tuple("1.0") < R.LAUNCH_API_VERSION
    assert R.api_version_tuple("1.10") > R.LAUNCH_API_VERSION
    assert R.api_version_tuple("1.1rc1") == (1,)
    assert R.api_version_tuple("") == ()


def test_route_call_signature_is_the_scorer_signature():
    """``__call__`` binds natively: same names/defaults as the contract."""
    assert R._params_of(R.QSAScorerRoute.__call__)[1:] == R.SCORER_PARAMS


def test_prewarm_once_per_process_with_atom_dtypes(caplog):
    api, calls = make_fake_api(version="1.1")
    a = call_args()
    routes = [
        R.QSAScorerRoute(P, site=f"model.layers.{i}", api=api, native=FakeNative())
        for i in range(4)
    ]
    with caplog.at_level(logging.INFO, logger="atom"):
        for route in routes:  # bind_caches of every layer (own cache each)
            assert route.prepare(torch.zeros_like(a[1])) == "ok"
        routes[0].prepare(a[1])  # a re-bind does not prewarm again
    assert calls["prewarm"] == [
        {
            "profile": P,
            "cache_dtypes": ("bf16",),
            "pos_dtypes": (torch.int64,),
            "device": a[1].device,
        }
    ]
    assert all(r.prewarmed == ("r32_persist", "decg", "draft") for r in routes)
    lines = [r.getMessage() for r in caplog.records if "prewarm" in r.getMessage()]
    assert len(lines) == 1
    assert "3 variant(s)" in lines[0] and "r32_persist, decg, draft" in lines[0]
    assert " s: " in lines[0] and P in lines[0]
    assert routes[1].summary()["prewarmed"] == ["r32_persist", "decg", "draft"]


def test_prewarm_dtype_matches_real_call_path_positions():
    """ATOM's QSA metadata builds ``logical_positions`` (the scorer's
    ``query_positions``) as int64 -- the dtype prewarm builds for."""
    src = (ROOT / "atom/model_ops/attentions/qwen4_exp_attn.py").read_text("utf-8")
    assert "logical_positions: torch.Tensor  # [tokens] int64" in src
    assert (
        '"qsa_logical_positions"] = CpuGpuBuffer(\n'
        '            max_tokens, publication_group="qsa", **i64'
    ) in src
    assert R.QUERY_POSITION_DTYPES == ("int64",)


def test_no_prewarm_on_api_10_static_failure_or_pinned_route():
    api, calls = make_fake_api(version="1.0")
    route = R.QSAScorerRoute(P, api=api, native=FakeNative())
    assert route.prepare(call_args()[1]) == "ok" and route.prewarmed is None
    api, calls = make_fake_api(version="1.1", layout=FakeReason.CACHE_SHAPE)
    route = R.QSAScorerRoute(P, api=api, native=FakeNative())
    assert route.prepare(call_args()[1]) == "cache_shape_unsupported"
    assert calls["prewarm"] == []
    api, calls = make_fake_api(version="2.1")
    route = R.QSAScorerRoute(P, api=api, native=FakeNative())
    assert route.prepare(call_args()[1]) == R.API_VERSION_MISMATCH
    assert calls["prewarm"] == []


def test_prewarm_failure_is_logged_once_and_fused_still_runs(caplog):
    api, calls = make_fake_api(version="1.1", prewarm_raises=RuntimeError("hipcc"))
    native = FakeNative()
    routes = [R.QSAScorerRoute(P, api=api, native=native) for _ in range(3)]
    a = call_args()
    with caplog.at_level(logging.INFO, logger="atom"):
        for route in routes:
            assert route.prepare(a[1]) == "ok"
    assert len(calls["prewarm"]) == 1  # not retried per layer
    warn = [r for r in caplog.records if "prewarm" in r.getMessage()]
    assert len(warn) == 1 and warn[0].levelno == logging.WARNING
    assert "hipcc" in warn[0].getMessage()
    assert routes[0].prewarmed is None
    assert select_style_call(routes[0], a) == ("launch", 8) and not native.calls


def test_launch_used_on_api_11_with_cached_decision():
    api, calls = make_fake_api(version="1.1")
    native = FakeNative()
    route = R.QSAScorerRoute(P, site="L0", api=api, native=native)
    a = call_args()
    route.prepare(a[1])
    for _ in range(6):
        assert select_style_call(route, a) == ("launch", 8)
    assert calls["decide"] == 1 and calls["launch"] == 6 and calls["fused"] == 0
    assert calls["launched"] is calls["decisions"][0]  # the cached decision
    assert route.counts[("fused", "ok")] == 6 and not native.calls
    assert route.summary()["fused_call"] == "launch"
    # A new class decides once more, then launches with its own decision.
    b = call_args(max_columns=32)
    for _ in range(3):
        select_style_call(route, b)
    assert calls["decide"] == 2 and calls["launch"] == 9
    assert calls["launched"] is calls["decisions"][1]


def test_api_10_uses_the_public_op():
    api, calls = make_fake_api(version="1.0")
    route = R.QSAScorerRoute(P, api=api, native=FakeNative())
    a = call_args()
    for _ in range(3):
        assert select_style_call(route, a) == ("fused", 8)
    assert calls["fused"] == 3 and calls["launch"] == 0 and calls["decide"] == 1
    assert set(calls["profiles"]) == {P}
    assert route.summary()["fused_call"] == "op"


def test_api_11_with_drifted_launch_falls_back_to_the_10_op(caplog):
    api, calls = make_fake_api(version="1.1")

    def bad_launch(decision, q, compressed_k_cache, *, logits=None):
        raise AssertionError("must not be called")

    api.launch = bad_launch
    with caplog.at_level(logging.INFO, logger="atom"):
        routes = [R.QSAScorerRoute(P, api=api, native=FakeNative()) for _ in range(2)]
    assert all(r.pinned is None and r._launch is None for r in routes)
    assert select_style_call(routes[0], call_args()) == ("fused", 8)
    warn = [r for r in caplog.records if "launch unusable" in r.getMessage()]
    assert len(warn) == 1  # once per process, not per layer
    assert routes[0].prepare(call_args()[1]) == "ok"
    assert len(calls["prewarm"]) == 1  # prewarm is independent of launch


def test_launch_refusal_repins_class_native_and_logs_once(caplog):
    api, calls = make_fake_api(version="1.1", raise_on_call=FakeReason.ROWS_RULE)
    native = FakeNative()
    route = R.QSAScorerRoute(P, site="L9", api=api, native=native)
    a = call_args()
    with caplog.at_level(logging.INFO, logger="atom"):
        for _ in range(3):
            assert select_style_call(route, a) == ("native", 8)
    assert calls["launch"] == 1 and calls["decide"] == 1
    assert route.counts[("stock", "rows_rule_uncovered")] == 3
    lines = [
        r.getMessage()
        for r in caplog.records
        if "QSA index scorer [L9]" in r.getMessage()
    ]
    assert lines == [
        (
            "QSA index scorer [L9] -> stock (rows_rule_uncovered): "
            "qsa_index_scorer: rows_rule_uncovered"
        )
    ]


def test_api_11_fallbacks_unchanged():
    # per-call uncovered class: decide once, native every call, launch never
    api, calls = make_fake_api(version="1.1", decide=FakeReason.COLUMNS)
    native = FakeNative()
    route = R.QSAScorerRoute(P, api=api, native=native)
    a = call_args()
    for _ in range(4):
        assert select_style_call(route, a, flydsl=True) == ("native", 8)
    assert calls["decide"] == 1 and calls["launch"] == 0 and len(native.calls) == 4
    q, cache, table, t2r, pos, ctx, cols = a
    args = native.calls[0]
    assert args[:6] == (q, cache, table, t2r, pos, ctx) and len(args) == 12
    assert args[6:9] == (4, None, cols) and args[10] is True
    assert route.counts[("stock", "columns_uncovered")] == 4
    # static layout reason: no decide, no launch
    api, calls = make_fake_api(version="1.1", layout=FakeReason.CACHE_SHAPE)
    route = R.QSAScorerRoute(P, api=api, native=FakeNative())
    for _ in range(2):
        select_style_call(route, a)
    assert calls["decide"] == 0 and calls["launch"] == 0 and calls["layout"] == 1
    # the stock scorer takes the scale plane (FP8 index cache) unchanged
    native = FakeNative()
    pinned = R.QSAScorerRoute("", native=native)
    scale = torch.ones(1)
    assert select_style_call(pinned, a, scale=scale) == ("native", 8)
    assert native.calls[0][11] is scale


def test_other_launch_exceptions_propagate():
    api, _ = make_fake_api(version="1.1")
    sig = inspect.signature(api.launch)

    def boom(*a, **k):
        raise RuntimeError("kernel fault")

    boom.__signature__ = sig
    api.launch = boom
    route = R.QSAScorerRoute(P, api=api, native=FakeNative())
    assert route._launch is boom
    with pytest.raises(RuntimeError, match="kernel fault"):
        select_style_call(route, call_args())


def test_capture_records_launch_and_native_paths():
    class Rec:
        def __init__(self):
            self.rows = []

        def record(self, site, args, reason, path, profile):
            assert len(args) == len(R.SCORER_PARAMS)
            self.rows.append((site, args[8], reason, path, profile))

    api, _ = make_fake_api(
        version="1.1",
        decide=lambda q, cols: FakeReason.OK if cols == 64 else FakeReason.COLUMNS,
    )
    rec = Rec()
    route = R.QSAScorerRoute(P, site="L1", api=api, capture=rec, native=FakeNative())
    select_style_call(route, call_args(max_columns=64))
    select_style_call(route, call_args(max_columns=32))
    assert rec.rows == [
        ("L1", 64, "ok", "fused", P),
        ("L1", 32, "columns_uncovered", "stock", P),
    ]


def test_stable_keys_follow_tensor_objects():
    """The identity memo never serves a key for a different tensor."""
    a = call_args()
    t = a[2]
    assert R._stable_tensor_key(t) == R._tensor_key(t)
    assert R._stable_tensor_key(t) is R._stable_tensor_key(t)  # memo hit
    u = t[:1]  # another object, other shape
    assert R._stable_tensor_key(u) == R._tensor_key(u) != R._tensor_key(t)
    assert R._stable_layout_key(a[1], None) == R.layout_key(a[1], None)
    other = torch.zeros(8, 4, 1, 128, dtype=torch.bfloat16)
    assert R._stable_layout_key(other, None) == R.layout_key(other, None)
    k = R.call_class_key(*a[:6], 4, None, a[6], None, None)
    assert k[6] == R._tensor_key(a[2]) and k[15] == R._tensor_key(a[5])
    assert len(k) == 20


def test_select_wrapper_imports_once(monkeypatch):
    import types

    seen = []
    fake = types.ModuleType(R.__package__ + ".select")
    fake.qsa_select_paged_tokens_routed = lambda *a, scorer, **k: seen.append(scorer)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    route = R.QSAScorerRoute("", native=FakeNative())
    route.select_paged_tokens(1, 2)
    route.select_paged_tokens(1, 2)
    assert seen == [route, route]
    assert route._select is fake.qsa_select_paged_tokens_routed


# ---------------------------------------------------------------------------
# Contract / drift tests against the real aiter module (skip if unavailable)
# ---------------------------------------------------------------------------
def _load_aiter_api():
    try:
        return importlib.import_module("aiter.ops.qsa_index_scorer")
    except Exception:  # noqa: BLE001, S110 -- fall back to QSA_AITER_ROOT
        pass
    root = os.environ.get("QSA_AITER_ROOT")
    if not root:
        return None
    d = os.path.join(root, "aiter", "ops", "qsa_index_scorer")
    if not os.path.isfile(os.path.join(d, "__init__.py")):
        return None
    name = "qsa_isc_atom_contract"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(d, "__init__.py"), submodule_search_locations=[d]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def aiter_api():
    api = _load_aiter_api()
    if api is None:
        pytest.skip("aiter.ops.qsa_index_scorer not importable (set QSA_AITER_ROOT)")
    return api


def test_contract_api_version(aiter_api):
    assert str(aiter_api.API_VERSION).split(".")[0] == R.CONTRACT_API_MAJOR
    assert aiter_api.DEFAULT_PROFILE == "qsa_page4_gfx1250_v1"


def test_contract_reasons_frozen(aiter_api):
    values = [r.value for r in aiter_api.Reason]
    assert set(values) == set(R.CONTRACT_REASONS)
    assert {r.value for r in aiter_api.STATIC_REASONS} == set(R.CONTRACT_STATIC_REASONS)
    assert not set(R.ATOM_REASONS) & set(values)


def test_contract_signature(aiter_api):
    assert R.signature_drift(aiter_api) is None
    for fn in (aiter_api.qsa_paged_mqa_logits, aiter_api.decide, aiter_api.eligibility):
        assert R._params_of(fn) == R.SCORER_PARAMS
        sig = inspect.signature(fn)
        assert sig.parameters["profile"].kind is inspect.Parameter.KEYWORD_ONLY
    lay = inspect.signature(aiter_api.layout_eligibility).parameters
    assert list(lay)[:2] == ["compressed_k_cache", "compressed_k_scale"]
    assert lay["profile"].kind is inspect.Parameter.KEYWORD_ONLY
    api_mod = sys.modules[aiter_api.__name__ + ".api"]
    assert api_mod.ATOM_SCORER_SOURCE_SHA256 == FROZEN_QSA_SHA256
    fields = set(aiter_api.Decision.__dataclass_fields__)
    assert {"ok", "reason", "detail"} <= fields


def test_contract_real_api_cpu_decisions(aiter_api, monkeypatch):
    monkeypatch.delenv("AITER_QSA_INDEX_SCORER_PROFILE", raising=False)
    a = call_args(rows=4, requests=1, width=16, max_columns=64)
    q, cache, table, t2r, pos, ctx, cols = a
    assert (
        aiter_api.decide(q, cache, table, t2r, pos, ctx, 4, None, cols).reason.value
        == "disabled"
    )
    native = FakeNative()
    route = R.QSAScorerRoute(
        aiter_api.DEFAULT_PROFILE, site="cpu", api=aiter_api, native=native
    )
    assert route.pinned is None
    assert route.prepare(cache) == "device_not_gpu"  # CPU tensors: never fused
    out = select_style_call(route, a)
    if route.aiter_native_path:
        # aiter >= 1.2 (Decision.path): aiter's own native path serves it.
        assert not native.calls and out[0].shape == (4, 64)
        assert route.counts[("aiter-native", "device_not_gpu")] == 1
    else:
        # aiter <= 1.1: the interim stock scorer serves it.
        assert out == ("native", 4)
        assert route.counts[("stock", "device_not_gpu")] == 1
    bogus = R.QSAScorerRoute("no_such_profile_zz", api=aiter_api, native=FakeNative())
    assert bogus.prepare(cache) == "profile_not_found"


def test_golden_adapter_noncontig_table_is_really_noncontiguous():
    """A row-padded table view with 1 request is contiguous to torch (size-1
    dim): aiter accepts it. The fallback fixture must use a column stride."""
    from atom.model_ops.qwen4_exp.qsa_index_scorer import golden_adapter as A

    table = torch.arange(16, dtype=torch.int32).view(1, 16)
    padded = torch.full((1, 32), -1, dtype=torch.int32)
    padded[:, :16] = table
    assert padded[:, :16].is_contiguous()  # the old, non-triggering fixture
    cases = {c[0]: c for c in A._fallback_cases("p")}
    for reqs in (1, 3):
        t = torch.arange(16 * reqs, dtype=torch.int32).view(reqs, 16)
        out = cases["metadata(noncontig table)"][2]({"page_table": t})["page_table"]
        assert not out.is_contiguous() and torch.equal(out, t)


def test_golden_adapter_native_defined_equal_ignores_unwritten_tail():
    from atom.model_ops.qwen4_exp.qsa_index_scorer import golden_adapter as A

    vis = torch.tensor([32, 0, 5], dtype=torch.int32)
    a = torch.randn(3, 64)
    b = a.clone()
    b[:, 32:] = float("nan")  # past every row's visible count: undefined
    b[1] = 7.0
    b[2, 5:] = -1.0
    r = A.native_defined_equal((a, vis), (b, vis.clone()))
    assert r == {"visible_equal": True, "defined_bitwise": True, "full_bitwise": False}
    b[2, 4] += 1.0
    assert not A.native_defined_equal((a, vis), (b, vis))["defined_bitwise"]
    other = vis.clone()
    other[0] = 31
    r = A.native_defined_equal((a, vis), (a.clone(), other))
    assert not r["visible_equal"] and not r["defined_bitwise"]


# ---------------------------------------------------------------------------
# Skill compliance: process capability, startup hook, prewarm scope, paths
# ---------------------------------------------------------------------------
def _ok_profile_api(version="1.1", **kw):
    api, calls = make_fake_api(version=version, **kw)

    def load_profile(name):
        calls.setdefault("load_profile", []).append(name)
        if name == "missing":
            raise FakeProfileError(FakeReason.DISABLED, "no such profile")
        return SimpleNamespace(kernel_sha256="58051973" + "0" * 56)

    api.load_profile = load_profile
    return api, calls


class FakeProfileError(ValueError):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def test_capability_absent_aiter_is_fail_open_and_cached(block_aiter, monkeypatch):
    seen = []
    real = importlib.import_module

    def counting(name, *a, **k):
        seen.append(name)
        return real(name, *a, **k)

    monkeypatch.setattr(CAP.importlib, "import_module", counting)
    cap = CAP.check_capability(P)
    assert not cap.ok and cap.reason == R.AITER_UNAVAILABLE
    assert CAP.check_capability(P) is cap  # once per process
    assert seen == [R.AITER_MODULE]
    route = R.QSAScorerRoute(P, native=FakeNative())  # binds the cached result
    assert route.pinned[0] == R.AITER_UNAVAILABLE and seen == [R.AITER_MODULE]
    assert select_style_call(route, call_args()) == ("native", 8)
    assert route.counts[("stock", R.AITER_UNAVAILABLE)] == 1


def test_capability_profile_must_load():
    api, calls = _ok_profile_api()
    cap = CAP.check_capability("missing", api=api)
    assert not cap.ok and cap.reason == "disabled"  # aiter's reason, verbatim
    route = R.QSAScorerRoute("missing", api=api, native=FakeNative())
    assert route.pinned[0] == "disabled"
    select_style_call(route, call_args())
    assert calls["decide"] == 0 and calls["launch"] == 0

    def raising(name):
        raise OSError("unreadable")

    api.load_profile = raising
    cap = CAP.check_capability(P, api=api)
    assert not cap.ok and cap.reason == CAP.PROFILE_UNLOADABLE


def test_capability_ok_fields_and_startup_line():
    api, _ = _ok_profile_api()
    cap = CAP.check_capability(P, api=api)
    assert cap.ok and cap.api_version == "1.1" and cap.fused_call == "launch"
    assert cap.kernel_sha256.startswith("58051973")
    assert cap.prewarm is not None and not cap.prewarm_scoped
    assert not cap.aiter_native_path  # FakeDecision has no `path`
    cfg = SimpleNamespace(
        kv_cache_block_size=32,
        hf_config=SimpleNamespace(indexer_compress_ratio=4),
    )
    line = CAP.startup_line(cap, cfg, "dp0_tp1")
    for part in (
        "[dp0_tp1]",
        "enabled",
        f"profile={P}",
        "aiter_api=1.1",
        "kernel_sha=5805197300000000",
        "fused_call=launch",
        "uncovered=stock",
        "kv_block=32",
        "index_page=8",
    ):
        assert part in line, (part, line)
    bad = CAP.check_capability(P, api=make_fake_api(version="2.0")[0])
    assert "DISABLED (stock scorer)" in CAP.startup_line(bad, cfg)
    assert "atom_api_version_mismatch" in CAP.startup_line(bad, cfg)


def test_capability_check_never_raises(monkeypatch):
    def boom(profile, api):
        raise RuntimeError("bug")

    monkeypatch.setattr(CAP, "_check", boom)
    cap = CAP.check_capability(P)
    assert not cap.ok and cap.reason == CAP.CAPABILITY_ERROR


def test_worker_hook_off_imports_nothing(block_aiter):
    from atom.aiter_fused_ops import check_aiter_fused_ops

    off = SimpleNamespace(qsa_index_scorer=SimpleNamespace(enabled=False, profile=""))
    assert check_aiter_fused_ops(off, rank_name="r0") == {}
    assert check_aiter_fused_ops(SimpleNamespace(), rank_name="r0") == {}
    assert CAP.init_qsa_index_scorer(off) is None
    assert not any(k == "aiter" or k.startswith("aiter.") for k in sys.modules)


def test_worker_hook_on_without_aiter_logs_and_does_not_raise(block_aiter, caplog):
    from atom.aiter_fused_ops import check_aiter_fused_ops

    cfg = SimpleNamespace(
        qsa_index_scorer=SimpleNamespace(enabled=True, profile=P),
        kv_cache_block_size=16,
        hf_config=SimpleNamespace(indexer_compress_ratio=4),
    )
    with caplog.at_level(logging.INFO, logger="atom"):
        out = check_aiter_fused_ops(cfg, rank_name="rank_0")
    cap = out["qsa_index_scorer"]
    assert not cap.ok and cap.reason == R.AITER_UNAVAILABLE
    lines = [r for r in caplog.records if "QSA index scorer [rank_0]" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    assert "index_page=4" in lines[0].getMessage()


def _method_calls(fn: ast.FunctionDef, names) -> dict:
    """First line of each call ``self.<name>(...)`` / ``<name>(...)`` in fn."""
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in names and name not in out:
                out[name] = node.lineno
    return out


def test_model_runner_checks_capability_before_building_the_model():
    """ModelRunner.__init__: _setup_device_and_distributed -> capability check
    -> _build_and_load_model (so before warm-up, KV allocation and capture)."""
    src = (ROOT / "atom" / "model_engine" / "model_runner.py").read_text("utf-8")
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    lines = _method_calls(
        init,
        {
            "_setup_device_and_distributed",
            "check_aiter_fused_ops",
            "_build_and_load_model",
            "_maybe_warmup",
        },
    )
    assert (
        lines["_setup_device_and_distributed"]
        < lines["check_aiter_fused_ops"]
        < lines["_build_and_load_model"]
    ), lines
    # the hook module imports no aiter at module scope
    hook = ast.parse((ROOT / "atom" / "aiter_fused_ops.py").read_text("utf-8"))
    assert not [
        n
        for n in hook.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and "aiter" in (getattr(n, "module", "") or "").replace("aiter_fused", "")
    ]


def _scoped_api():
    api, calls = _ok_profile_api()

    def prewarm(
        profile=None,
        *,
        cache_dtypes=("bf16",),
        pos_dtypes=(torch.int32, torch.int64),
        device=None,
        page_sizes=None,
        compressed_k_cache=None,
        compressed_k_scale=None,
    ):
        calls["prewarm"].append(
            {
                "cache_dtypes": cache_dtypes,
                "pos_dtypes": pos_dtypes,
                "page": int(compressed_k_cache.shape[1]),
                "cache": compressed_k_cache,
            }
        )
        return [f"p{int(compressed_k_cache.shape[1])}"]

    api.prewarm = prewarm
    return api, calls


@pytest.mark.parametrize("block", [16, 32, 64])
def test_prewarm_is_scoped_to_the_bound_cache_page(block):
    """KV block 16/32/64 -> index page 4/8/16: aiter builds only what that
    cache can select (its page size and dtype), with int64 positions, once
    per process however many layers bind a cache of that geometry."""
    api, calls = _scoped_api()
    page = block // 4
    routes = [
        R.QSAScorerRoute(P, site=f"L{i}", api=api, native=FakeNative())
        for i in range(3)
    ]
    assert routes[0].capability.prewarm_scoped
    caches = [torch.zeros(8, page, 1, 128, dtype=torch.bfloat16) for _ in routes]
    for route, cache in zip(routes, caches):
        assert route.prepare(cache) == "ok"
    assert len(calls["prewarm"]) == 1
    (pw,) = calls["prewarm"]
    assert pw["page"] == page and pw["cache"] is caches[0]
    assert pw["cache_dtypes"] == ("bf16",) and pw["pos_dtypes"] == (torch.int64,)
    assert R.index_page_of(caches[0]) == page
    assert all(r.prewarmed == (f"p{page}",) for r in routes)


def test_prewarm_once_per_page_size_and_dtype():
    api, calls = _scoped_api()
    route = R.QSAScorerRoute(P, api=api, native=FakeNative())
    for page in (4, 8, 16, 4):
        route.prepare(torch.zeros(8, page, 1, 128, dtype=torch.bfloat16))
    assert [c["page"] for c in calls["prewarm"]] == [4, 8, 16]


def test_release_kv_pools_rebind_with_none_is_a_noop():
    api, calls = _ok_profile_api()
    route = R.QSAScorerRoute(P, api=api, native=FakeNative())
    assert route.prepare(None) is None
    assert calls["layout"] == 0 and calls["prewarm"] == []


def test_mtp_verify_bucket_is_not_gated_by_atom():
    """MTP3 verify (rows == 3 x requests) is offered to aiter like any other
    class: aiter's default-on buckets decide; ATOM adds no rows rule."""
    api, calls = make_fake_api(version="1.1")
    route = R.QSAScorerRoute(P, api=api, native=FakeNative())
    for requests in (1, 2, 8, 64):
        a = call_args(rows=3 * requests, requests=requests)
        assert select_style_call(route, a) == ("launch", 3 * requests)
    assert calls["decide"] == 4 and route.counts[("fused", "ok")] == 4


@dataclass(frozen=True)
class PathDecision:
    ok: bool
    reason: FakeReason
    detail: str = ""
    path: str = "fused"


def test_aiter_native_path_serves_uncovered_calls_without_atom_fallback():
    """With an aiter whose Decision carries ``path``, an uncovered class is
    launched through aiter (its own native path): ATOM's scorer never runs."""
    api, calls = make_fake_api(version="1.1")
    api.Decision = PathDecision

    def decide(q, cache, table, t2r, pos, ctx, ratio=4, div=None, cols=None, *a, **k):
        calls["decide"] += 1
        if cols == 64:
            return PathDecision(True, FakeReason.OK, "", "fused")
        return PathDecision(False, FakeReason.COLUMNS, "uncovered", "native")

    def launch(decision, q, *a, **k):
        calls["launch"] += 1
        return (decision.path, q.shape[0])

    launch.__signature__ = inspect.signature(api.launch)
    api.decide, api.launch = decide, launch
    native = FakeNative()
    route = R.QSAScorerRoute(P, api=api, native=native)
    assert route.aiter_native_path and route.summary()["uncovered"] == "aiter-native"
    assert select_style_call(route, call_args(max_columns=64)) == ("fused", 8)
    a32 = call_args(max_columns=32)  # one cache object: one class
    for _ in range(2):
        assert select_style_call(route, a32) == ("native", 8)
    assert not native.calls and calls["launch"] == 3 and calls["decide"] == 2
    assert route.counts[("fused", "ok")] == 1
    assert route.counts[("aiter-native", "columns_uncovered")] == 2


def test_process_summary_sums_paths_over_routes(caplog):
    def decide(q, cols):
        return FakeReason.OK if cols == 64 else FakeReason.COLUMNS

    api, _ = make_fake_api(version="1.1", decide=decide)
    routes = [
        R.QSAScorerRoute(P, site=f"S{i}", api=api, native=FakeNative())
        for i in range(2)
    ]
    for route in routes:
        select_style_call(route, call_args(max_columns=64))
        select_style_call(route, call_args(max_columns=32))
    summary = R.process_summary()
    mine = [s for s in summary["sites"] if s["site"] in ("S0", "S1")]
    assert len(mine) == 2
    assert summary["by_path"]["fused"] >= 2 and summary["by_path"]["stock"] >= 2
    assert summary["paths"]["fused:ok"] >= 2
    with caplog.at_level(logging.INFO, logger="atom"):
        R.log_process_summary()
    assert any(
        "QSA index scorer decisions: fused=" in r.getMessage() for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Call site (Qwen4ExpAttention): flag off -> stock select, on -> the route
# ---------------------------------------------------------------------------
def _qsa_attention():
    try:
        return importlib.import_module("atom.model_ops.qwen4_exp.qsa_attention")
    except Exception as exc:  # noqa: BLE001 -- needs aiter's python package
        pytest.skip(f"qsa_attention not importable on this host: {exc!r}")


def _fake_layer(M, monkeypatch, scorer):
    for name in ("qsa_store_rows", "qsa_store_quantized_rows"):
        monkeypatch.setattr(M, name, lambda *a, **k: None)
    monkeypatch.setattr(
        M, "qsa_compress_groups", lambda *a, **k: (torch.zeros(1), torch.zeros(1))
    )
    ctx = SimpleNamespace(context=SimpleNamespace(is_draft=False, is_prefill=False))
    monkeypatch.setattr(M, "get_forward_context", lambda: ctx)
    indexer = SimpleNamespace(
        project_qk=lambda h, p: ("index_q", "raw_key"),
        normalize_compressed_keys=lambda pooled, first: "normalized",
        token_topk=2048,
        compress_ratio=4,
    )
    return SimpleNamespace(
        indexer=indexer,
        raw_key_cache="raw",
        rope_position_cache=None,
        compressed_key_cache="ccache",
        compressed_key_scale=None,
        qsa_scorer=scorer,
    )


def _qsa_meta():
    return SimpleNamespace(
        slot_mapping=None,
        block_tables="bt",
        token_to_req="t2r",
        logical_positions="pos",
        compressed_slot_mapping=None,
        seq_lens="lens",
        max_seq_len=128,
    )


def test_call_site_flag_off_uses_the_stock_select(monkeypatch):
    M = _qsa_attention()
    seen = []
    monkeypatch.setattr(
        M, "qsa_select_paged_tokens", lambda *a, **k: seen.append(("stock", a, k))
    )
    layer = _fake_layer(M, monkeypatch, scorer=None)
    M.Qwen4ExpAttention._select_tokens(
        layer, torch.zeros(1), torch.zeros(1), _qsa_meta()
    )
    assert len(seen) == 1 and seen[0][0] == "stock"
    assert seen[0][1][:2] == ("index_q", "ccache")


def test_call_site_flag_on_uses_the_route(monkeypatch):
    M = _qsa_attention()
    monkeypatch.setattr(
        M, "qsa_select_paged_tokens", lambda *a, **k: pytest.fail("stock called")
    )
    seen = []
    scorer = SimpleNamespace(select_paged_tokens=lambda *a, **k: seen.append((a, k)))
    layer = _fake_layer(M, monkeypatch, scorer=scorer)
    M.Qwen4ExpAttention._select_tokens(
        layer, torch.zeros(1), torch.zeros(1), _qsa_meta()
    )
    ((a, k),) = seen
    assert a[:2] == ("index_q", "ccache") and k["compressed_k_scale"] is None
    assert k["allow_flydsl"] is False and k["max_seq_len"] == 128


def test_bind_caches_prepares_the_route_after_kv_allocation():
    M = _qsa_attention()
    prepared = []
    layer = SimpleNamespace(
        qsa_scorer=SimpleNamespace(prepare=lambda c, s: prepared.append((c, s)))
    )
    M.Qwen4ExpAttention.bind_caches(
        layer, "k", "v", "raw", "ccache", None, compressed_key_scale="scale"
    )
    assert prepared == [("ccache", "scale")]
    off = SimpleNamespace(qsa_scorer=None)
    M.Qwen4ExpAttention.bind_caches(off, "k", "v", "raw", "ccache")
    assert off.compressed_key_cache == "ccache"


@pytest.mark.parametrize("page", [4, 8, 16])
def test_capture_plan_rows_for_every_index_page(page):
    a = call_args(rows=4, requests=1, width=16, page=page, max_columns=None)
    sig = C.describe_call(a[:6] + (4, None, None, None, False, None))
    row = C.plan_row(sig)
    assert row is not None and row["columns"] == 16 * page
    assert row.get("page_size", 4) == page
