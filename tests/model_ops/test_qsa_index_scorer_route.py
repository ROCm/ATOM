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

from atom.model_ops.qwen4_exp.qsa_index_scorer import capture as C
from atom.model_ops.qwen4_exp.qsa_index_scorer import route as R

ROOT = Path(__file__).resolve().parents[2]
QSA_PY = ROOT / "atom" / "model_ops" / "qwen4_exp" / "ops" / "qsa.py"
SELECT_PY = ROOT / "atom" / "model_ops" / "qwen4_exp" / "qsa_index_scorer" / "select.py"
ATTN_PY = ROOT / "atom" / "model_ops" / "qwen4_exp" / "qsa_attention.py"
# The frozen scorer source: aaa78f43... is its sha256 with CRLF line endings
# (the golden_baseline copy and a Windows autocrlf checkout); c2e46e50... is
# the same content with LF (the git blob, a Linux checkout).
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
    layout=FakeReason.OK, decide=FakeReason.OK, raise_on_call=None, version="1.0"
):
    calls = {"layout": 0, "decide": 0, "fused": 0, "profiles": []}

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
        return FakeDecision(r is FakeReason.OK, r, "decide")

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

    api = SimpleNamespace(
        API_VERSION=version,
        Reason=FakeReason,
        QSAIndexScorerUnsupported=FakeUnsupported,
        layout_eligibility=layout_eligibility,
        decide=_decide,
        qsa_paged_mqa_logits=qsa_paged_mqa_logits,
    )
    return api, calls


class FakeNative:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        assert len(args) == 12
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
    assert route.counts[("native", R.CAPTURE_ONLY)] == 1
    assert not any(k == "aiter" or k.startswith("aiter.") for k in sys.modules)


def test_aiter_unavailable_pins_native(block_aiter):
    native = FakeNative()
    route = R.QSAScorerRoute("qsa_page4_gfx1250_v1", site="L0", native=native)
    assert route.pinned[0] == R.AITER_UNAVAILABLE
    assert route.prepare(call_args()[1]) == R.AITER_UNAVAILABLE
    assert select_style_call(route, call_args()) == ("native", 8)
    assert route.counts[("native", R.AITER_UNAVAILABLE)] == 1


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
    assert route.counts[("native", "cache_shape_unsupported")] == 3


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
    assert args[6:9] == (4, None, cols) and args[10] is True and args[11] is None
    assert route.counts[("native", "columns_uncovered")] == 1


def test_unsupported_raised_at_call_falls_back_and_repins():
    api, calls = make_fake_api(raise_on_call=FakeReason.ROWS_RULE)
    native = FakeNative()
    route = R.QSAScorerRoute("p", api=api, native=native)
    a = call_args()
    assert select_style_call(route, a) == ("native", 8)
    assert select_style_call(route, a) == ("native", 8)
    assert calls["fused"] == 1  # second call does not retry the fused op
    assert route.counts[("native", "rows_rule_uncovered")] == 2


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
    assert any("-> native (columns_uncovered)" in m for m in lines)
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
    assert ka[0] == kb[0] and ka[1] == kb[1] and ka[3:] == kb[3:]


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
    for _ in range(3):
        select_style_call(
            route, call_args(rows=4, requests=1, width=16, max_columns=64)
        )
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
        assert R._params_of(fn) == R.NATIVE_SCORER_PARAMS
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
    assert select_style_call(route, a) == ("native", 4)
    assert route.counts[("native", "device_not_gpu")] == 1
    bogus = R.QSAScorerRoute("no_such_profile_zz", api=aiter_api, native=FakeNative())
    assert bogus.prepare(cache) == "profile_not_found"
