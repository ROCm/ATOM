# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests: the QSA indexer scoring through aiter's routed op.

No GPU. ``aiter.plugin`` (``API_VERSION`` / ``configure`` / ``status`` /
``explain``) and ``aiter.ops.qsa`` (the routed ``qsa_paged_mqa_logits``) are fake
modules placed in ``sys.modules``, and the provider entry point is a fake with a
``warmup`` hook, so these run with any or no aiter installed. Covered:
per-process provider enablement + scorer configure (ordering, idempotence,
fail-open), the call site (flag off -> ATOM's stock select; on -> the routed op
itself, all 12 arguments, no ATOM fallback), provider warm-up placement and
payload, the opt-in explain() observer / capture, and the AST drift guards
against the frozen ``qsa.py``.
"""

from __future__ import annotations

import ast
import collections
import functools
import hashlib
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom import aiter_provider as AP
from atom.config import QSAIndexScorerConfig
from atom.model_ops.qwen4_exp.qsa_index_scorer import capture as C
from atom.model_ops.qwen4_exp.qsa_index_scorer import enablement as E

ROOT = Path(__file__).resolve().parents[2]
QSA_PY = ROOT / "atom" / "model_ops" / "qwen4_exp" / "ops" / "qsa.py"
PKG = ROOT / "atom" / "model_ops" / "qwen4_exp" / "qsa_index_scorer"
SELECT_PY = PKG / "select.py"
ATTN_PY = ROOT / "atom" / "model_ops" / "qwen4_exp" / "qsa_attention.py"
RUNNER_PY = ROOT / "atom" / "model_engine" / "model_runner.py"
ENGINE_CORE_PY = ROOT / "atom" / "model_engine" / "engine_core.py"
HOOK_PY = ROOT / "atom" / "aiter_fused_ops.py"
ATTN = "atom.model_ops.qwen4_exp.qsa_attention"
# The frozen golden scorer source aiter's contract is written against: ATOM
# 8a07708's qsa.py (whole file, unchanged through 10b99a08: sha256 c2e46e50...
# LF / aaa78f43... CRLF). Main's #2847 split the selection loop into
# `_selection_widths` / `qsa_select_paged_groups` / `qsa_select_paged_tokens`
# (whole file at 920a200a: f8f04ab3... LF / 683a9528... CRLF) and left the
# rest -- the scorer, its kernels and helpers -- unchanged. This pins that
# rest: qsa.py's lines outside the selection entries (which the AST tests
# below compare instead), LF, blank lines dropped.
SELECTION_ENTRIES = (
    "_selection_widths",
    "qsa_select_paged_groups",
    "qsa_select_paged_tokens",
)
FROZEN_QSA_SCORER_SHA256 = (
    "4f48345db70873741e487d68aaff441997fa90e77438c7af2bf53b126d9d5b7f"
)
P = "qsa-index-scorer-provider"


@pytest.fixture(autouse=True)
def _fresh_state():
    E._reset_for_testing()
    AP._reset_for_testing()
    yield
    E._reset_for_testing()
    AP._reset_for_testing()


# ---------------------------------------------------------------------------
# fake aiter.plugin + routed aiter.ops.qsa + provider entry point
# ---------------------------------------------------------------------------
Decision = collections.namedtuple("Decision", "route reason provider")


def make_fake_aiter(
    *,
    plugin=True,
    api_version=1,
    routed=True,
    providers=((P, ("qsa_paged_mqa_logits",)),),
    warmup_raises=None,
    op_raises=None,
):
    calls = collections.defaultdict(list)
    api = types.ModuleType(E.AITER_QSA_MODULE)

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
        stacked_decode=None,
        *,
        logits=None,
        visible_groups=None,
    ):
        calls["op"].append(
            {
                "rows": q.shape[0],
                "cache": compressed_k_cache,
                "compress_ratio": compress_ratio,
                "max_columns": max_columns,
                "row_starts": row_starts,
                "allow_flydsl": allow_flydsl,
                "compressed_k_scale": compressed_k_scale,
                "stacked_decode": stacked_decode,
            }
        )
        if op_raises is not None:
            raise op_raises
        cols = max_columns or page_table.shape[1] * compressed_k_cache.shape[1]
        return (
            torch.zeros(q.shape[0], cols),
            torch.full((q.shape[0],), cols, dtype=torch.int32),
        )

    if routed:
        qsa_paged_mqa_logits.__aiter_plugin_op__ = ("qsa_paged_mqa_logits", 1)
    api.qsa_paged_mqa_logits = qsa_paged_mqa_logits

    plug = types.ModuleType("aiter.plugin")
    plug.API_VERSION = api_version

    def _configure(*, source, enable=()):
        calls["configure"].append((source, tuple(enable)))

    def _status():
        calls["status"].append(1)
        return {
            "enabled": tuple(x for c in calls["configure"][-1:] for x in c[1]),
            "providers": [
                {"name": n, "op_versions": {op: [1] for op in ops}}
                for n, ops in providers
            ],
            "skipped_providers": [],
            "routed_ops": ["qsa_paged_mqa_logits"] if routed else [],
        }

    def _explain(op, *args):
        assert op == "qsa_paged_mqa_logits" and len(args) == 12
        calls["explain"].append(args)
        if args[8] == 64:
            d = Decision("provider", "selected", P)
        else:
            d = Decision("native", "provider-declined", None)
        return types.SimpleNamespace(decision=d)

    plug.configure, plug.status, plug.explain = _configure, _status, _explain

    class Provider:
        name = P

        def warmup(self, payload, *, device=None):
            calls["warmup"].append({**payload, "device": device})
            if warmup_raises is not None:
                raise warmup_raises
            return True

    return api, plug if plugin else None, Provider(), calls


@pytest.fixture
def fake_aiter(monkeypatch):
    def install(**kw):
        api, plug, provider, calls = make_fake_aiter(**kw)
        monkeypatch.setitem(sys.modules, E.AITER_QSA_MODULE, api)
        monkeypatch.setitem(sys.modules, "aiter.plugin", plug)  # None: ImportError
        monkeypatch.setattr(AP, "load_provider", lambda name: provider)
        return api, calls

    return install


def cfg(providers=(P,), capture=""):
    return SimpleNamespace(
        aiter_provider_enable=tuple(providers),
        qsa_index_scorer=QSAIndexScorerConfig(capture=capture),
    )


def call_args(rows=8, requests=2, width=16, page=4, max_columns=64, fp8=False):
    q = torch.zeros(rows, 4, 128, dtype=torch.bfloat16)
    if fp8:
        cache = torch.zeros(32, page, 1, 128, dtype=torch.float8_e4m3fn)
        scale = torch.ones(32, page, 1, dtype=torch.float32)
    else:
        cache = torch.zeros(32, page, 1, 128, dtype=torch.bfloat16)
        scale = None
    table = torch.zeros(requests, width, dtype=torch.int32)
    t2r = torch.zeros(rows, dtype=torch.int32)
    pos = torch.arange(rows, dtype=torch.int64)
    ctx = torch.full((requests,), 64, dtype=torch.int32)
    return (q, cache, table, t2r, pos, ctx), {
        "max_columns": max_columns,
        "row_starts": torch.zeros(rows, dtype=torch.int32),
        "allow_flydsl": False,
        "compressed_k_scale": scale,
    }


# ---------------------------------------------------------------------------
# drift guards against ATOM's own frozen scorer (AST only)
# ---------------------------------------------------------------------------
def _func(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _scorer_section_sha256(path: Path) -> str:
    src = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    selection = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in SELECTION_ENTRIES:
            first = min([d.lineno for d in node.decorator_list] + [node.lineno])
            selection.update(range(first, node.end_lineno + 1))
    kept = [
        line
        for i, line in enumerate(src.split("\n"), 1)
        if i not in selection and line.strip()
    ]
    return hashlib.sha256("\n".join(kept).encode()).hexdigest()


def test_qsa_py_is_the_frozen_scorer():
    """Everything in qsa.py but the selection entries is 8a07708's."""
    assert _scorer_section_sha256(QSA_PY) == FROZEN_QSA_SCORER_SHA256
    for name in SELECTION_ENTRIES:
        _func(QSA_PY, name)


def test_observer_binding_is_atom_scorer_signature():
    """The 12 leading parameters of the neutral op are ATOM's scorer's."""
    fn = _func(QSA_PY, "qsa_paged_mqa_logits")
    names = tuple(a.arg for a in fn.args.args)
    assert names == E.SCORER_PARAMS
    defaults = [ast.literal_eval(d) for d in fn.args.defaults]
    assert dict(zip(names[-len(defaults) :], defaults)) == E._DEFAULTS


class _RenameScorer(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "qsa_paged_mqa_logits":
            return ast.copy_location(ast.Name(id="scorer", ctx=node.ctx), node)
        return node


def _body(fn):
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    return body


class _InjectScorerIntoGroups(ast.NodeTransformer):
    """``qsa_select_paged_groups(...)`` -> ``..._with_scorer(..., scorer=scorer)``."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if getattr(node.func, "id", None) == "qsa_select_paged_groups":
            node.func = ast.Name(
                id="qsa_select_paged_groups_with_scorer", ctx=ast.Load()
            )
            node.keywords.append(
                ast.keyword(arg="scorer", value=ast.Name(id="scorer", ctx=ast.Load()))
            )
        return node


def _assert_copy(stock_name, transform):
    stock = _func(QSA_PY, stock_name)
    mine = _func(SELECT_PY, f"{stock_name}_with_scorer")
    want = [ast.dump(transform.visit(s)) for s in _body(stock)]
    assert [ast.dump(s) for s in _body(mine)] == want, "re-sync select.py with qsa.py"
    assert [a.arg for a in mine.args.args] == [a.arg for a in stock.args.args]
    assert [ast.dump(d) for d in mine.args.defaults] == [
        ast.dump(d) for d in stock.args.defaults
    ]
    assert [a.arg for a in mine.args.kwonlyargs] == ["scorer"]


def test_groups_copy_is_stock_groups_with_scorer_injected():
    """The score + top-k loop (#2847's `qsa_select_paged_groups`)."""
    _assert_copy("qsa_select_paged_groups", _RenameScorer())


def test_select_copy_is_stock_select_with_scorer_injected():
    """The whole-batch entry: the stock wrapper over the groups copy."""
    _assert_copy("qsa_select_paged_tokens", _InjectScorerIntoGroups())


def test_no_atom_fallback_in_the_scoring_path():
    """select.py never names ATOM's scorer or its stock selection entries;
    the package has no route, decide/launch or retry machinery left."""
    src = SELECT_PY.read_text(encoding="utf-8")
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    names |= {
        a.name
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.ImportFrom)
        for a in n.names
    }
    assert "qsa_paged_mqa_logits" not in names
    assert "qsa_select_paged_groups" not in names
    assert "qsa_select_paged_tokens" not in names
    files = {p.name for p in PKG.glob("*.py")}
    assert files == {"__init__.py", "enablement.py", "select.py", "capture.py"}
    for p in PKG.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert ".decide(" not in text and ".launch(" not in text, p.name


def test_package_imports_nothing_heavy():
    import subprocess

    code = (
        "import sys, atom; before = set(sys.modules); "
        "import atom.model_ops.qwen4_exp.qsa_index_scorer, "
        "atom.model_ops.qwen4_exp.qsa_index_scorer.capture, atom.aiter_fused_ops; "
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


def test_attention_imports_the_package_only_when_active():
    tree = ast.parse(ATTN_PY.read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            assert "qsa_index_scorer" not in (getattr(n, "module", "") or "")
    src = ATTN_PY.read_text(encoding="utf-8")
    assert "if (scorer_cfg is not None and scorer_cfg.active) or getattr(" in src
    assert '"aiter_provider_enable", ()' in src
    assert (
        "qsa_select_paged_tokens if self.qsa_select is None else self.qsa_select" in src
    )
    groups_switch = (
        "qsa_select_paged_groups if self.qsa_select_groups is None "
        "else self.qsa_select_groups"
    )
    assert groups_switch in " ".join(src.split())


# ---------------------------------------------------------------------------
# configure: once per process, before the model is built, fail-open
# ---------------------------------------------------------------------------
def _calls_in(fn: ast.FunctionDef, names) -> dict:
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in names:
                out[name] = min(out.get(name, node.lineno), node.lineno)
    return out


def _method(path: Path, cls_name: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def test_model_runner_configures_before_building_the_model():
    init = _method(RUNNER_PY, "ModelRunner", "__init__")
    at = _calls_in(
        init,
        {
            "_setup_device_and_distributed",
            "configure_aiter_fused_ops",
            "_build_and_load_model",
            "_maybe_warmup",
        },
    )
    assert (
        at["_setup_device_and_distributed"]
        < at["configure_aiter_fused_ops"]
        < at["_build_and_load_model"]
        < at["_maybe_warmup"]
    ), at
    hook = ast.parse(HOOK_PY.read_text("utf-8"))
    assert not [
        n
        for n in hook.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and "aiter" in (getattr(n, "module", "") or "").replace("aiter_fused", "")
    ]


def test_prewarm_runs_after_kv_binding_and_before_capture():
    alloc = _method(RUNNER_PY, "ModelRunner", "allocate_kv_cache")
    at = _calls_in(
        alloc,
        {"build_kv_cache_tensor", "set_kv_cache_data", "_prewarm_aiter_fused_ops"},
    )
    assert (
        at["build_kv_cache_tensor"]
        < at["set_kv_cache_data"]
        < at["_prewarm_aiter_fused_ops"]
    ), at
    rebind = _method(RUNNER_PY, "RapidServeModelRunner", "_bind_kv_cache_to_modules")
    at = _calls_in(rebind, {"set_kv_cache_data", "_prewarm_aiter_fused_ops"})
    assert at["set_kv_cache_data"] < at["_prewarm_aiter_fused_ops"], at
    # EngineCore: allocate_kv_cache (-> prewarm) strictly before capture.
    src = ENGINE_CORE_PY.read_text("utf-8")
    assert src.index('"allocate_kv_cache"') < src.index('"capture_cudagraph"')


def test_off_imports_and_configures_nothing(monkeypatch):
    from atom.aiter_fused_ops import configure_aiter_fused_ops, prewarm_aiter_fused_ops

    def boom(name, *a, **k):
        raise AssertionError(f"imported {name}")

    monkeypatch.setattr(E.importlib, "import_module", boom)
    monkeypatch.setattr(AP.importlib, "import_module", boom)
    assert configure_aiter_fused_ops(cfg(providers=()), rank_name="r0") == {}
    assert configure_aiter_fused_ops(SimpleNamespace(), rank_name="r0") == {}
    assert prewarm_aiter_fused_ops(cfg(providers=()), [], rank_name="r0") == {}
    assert E.configure(cfg(providers=())) is None
    assert E.index_scorer(cfg(providers=())) is None


def test_configure_once_per_process_with_source_and_enable(fake_aiter, caplog):
    from atom.aiter_fused_ops import configure_aiter_fused_ops

    api, calls = fake_aiter()
    with caplog.at_level(logging.INFO, logger="atom"):
        out = configure_aiter_fused_ops(cfg(), rank_name="dp0_tp0")
        configure_aiter_fused_ops(cfg(), rank_name="dp0_tp0")
        E.index_scorer(cfg())
    state = out["qsa_index_scorer"]
    assert state.enabled and state.op is api.qsa_paged_mqa_logits
    assert state.providers == (P,)
    ((source, enable),) = calls["configure"]
    assert source.startswith("atom ") and enable == (P,)
    assert len(calls["status"]) == 1
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("QSA index scorer [dp0_tp0]")
    ]
    assert len(lines) == 1 and f"providers=['{P}']" in lines[0]
    assert any(
        r.getMessage().startswith("aiter providers [dp0_tp0]: enabled=")
        for r in caplog.records
    )


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("no_plugin", "aiter.plugin unavailable"),
        ("api_version", "aiter.plugin API 2"),
        ("not_routed", E.NO_PROVIDER),
        ("other_provider", E.NO_PROVIDER),
        ("op_not_routed", E.OP_NOT_ROUTED),
    ],
)
def test_fail_open_keeps_atom_scorer(fake_aiter, caplog, kind, reason):
    from atom.aiter_fused_ops import configure_aiter_fused_ops

    if kind == "no_plugin":
        _, calls = fake_aiter(plugin=False)
    elif kind == "api_version":
        _, calls = fake_aiter(api_version=2)
    elif kind == "not_routed":
        _, calls = fake_aiter(routed=False)
    elif kind == "other_provider":
        _, calls = fake_aiter(providers=((P, ("sp_qkv_fused",)),))
    else:  # status says routed, but the op object carries no route marker
        _, calls = fake_aiter()
        del sys.modules[E.AITER_QSA_MODULE].qsa_paged_mqa_logits.__aiter_plugin_op__
    with caplog.at_level(logging.INFO, logger="atom"):
        state = configure_aiter_fused_ops(cfg(), rank_name="r0")["qsa_index_scorer"]
        assert E.index_scorer(cfg()) is None
        assert E.index_scorer(cfg()) is None
    assert not state.enabled and state.reason.startswith(reason), state.reason
    warn = [r for r in caplog.records if "DISABLED (ATOM scorer)" in r.getMessage()]
    assert len(warn) == 1 and warn[0].levelno == logging.WARNING
    assert not calls.get("op")


def test_index_scorer_is_the_routed_op_itself(fake_aiter):
    """Default ON path: no ATOM wrapper, nothing per call."""
    api, _ = fake_aiter()
    assert E.index_scorer(cfg()) is api.qsa_paged_mqa_logits


# ---------------------------------------------------------------------------
# call site
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_load():
    """Import ATOM model code against tests/aiter_stub.py's stand-ins for
    aiter / Triton (a no-device container cannot initialise Triton's driver).
    Request it FIRST so the fake ``aiter.ops.qsa`` installed later sits on top
    of the stubs and is removed before them."""
    from aiter_stub import atom_under_stubs

    with atom_under_stubs() as load:

        def _load(name):
            try:
                return load(name)
            except Exception as exc:  # noqa: BLE001 -- import chain needs more
                pytest.skip(f"{name} not importable under the aiter stubs: {exc!r}")

        yield _load


@pytest.fixture
def select_mod(stub_load, monkeypatch):
    S = stub_load("atom.model_ops.qwen4_exp.qsa_index_scorer.select")
    monkeypatch.setattr(S, "top_k_per_row_prefill", lambda *a, **k: None)
    # The stock expansion fills and returns `out` (7th positional).
    monkeypatch.setattr(S, "qsa_expand_block_indices", lambda *a, **k: a[6])
    return S


@pytest.mark.parametrize("fp8", [False, True])
def test_select_scores_through_the_routed_op_with_all_12_args(
    select_mod, fake_aiter, fp8
):
    _api, calls = fake_aiter()
    (q, cache, table, t2r, pos, ctx), kw = call_args(rows=10, fp8=fp8)
    out = select_mod.qsa_select_paged_tokens_with_scorer(
        q,
        cache,
        table,
        t2r,
        pos,
        ctx,
        64,
        4,
        logits_workspace_bytes=4 * 64 * 4,  # 4 rows per chunk -> 3 chunks
        max_seq_len=200,
        allow_flydsl=True,
        compressed_k_scale=kw["compressed_k_scale"],
        scorer=E.index_scorer(cfg()),
    )
    assert out.shape[0] == 10
    assert [c["rows"] for c in calls["op"]] == [4, 4, 2]
    for c in calls["op"]:
        assert c["cache"] is cache and c["compress_ratio"] == 4
        assert c["max_columns"] == 64 and c["allow_flydsl"] is True
        assert c["compressed_k_scale"] is kw["compressed_k_scale"]
        assert c["row_starts"] is not None and c["row_starts"].dtype == torch.int32
        # ATOM passes the contract's 12 leading params only (aiter's 13th,
        # stacked_decode, stays at its default None = aiter decides).
        assert c["stacked_decode"] is None


def test_row_sharded_groups_score_through_the_routed_op(select_mod, fake_aiter):
    """#2847's row-sharded entry (`qsa_select_paged_groups`) on the op too."""
    _api, calls = fake_aiter()
    (q, cache, table, t2r, pos, ctx), _ = call_args(rows=6)
    groups, width = select_mod.qsa_select_paged_groups_with_scorer(
        q,
        cache,
        table,
        t2r,
        pos,
        ctx,
        64,
        4,
        logits_workspace_bytes=4 * 64 * 4,  # 4 rows per chunk -> 2 chunks
        max_seq_len=200,
        scorer=E.index_scorer(cfg()),
    )
    assert groups.shape == (6, 16) and groups.dtype == torch.int32
    assert width == 16 * 4 + 3
    assert [c["rows"] for c in calls["op"]] == [4, 2]
    assert all(c["max_columns"] == 64 for c in calls["op"])


def test_op_errors_propagate_without_an_atom_fallback(
    stub_load, select_mod, fake_aiter, monkeypatch
):
    qsa_ops = stub_load("atom.model_ops.qwen4_exp.ops.qsa")
    monkeypatch.setattr(
        qsa_ops,
        "qsa_paged_mqa_logits",
        lambda *a, **k: pytest.fail("ATOM's scorer ran after the op"),
    )
    _, calls = fake_aiter(op_raises=RuntimeError("kernel failed"))
    (q, cache, table, t2r, pos, ctx), _ = call_args()
    with pytest.raises(RuntimeError, match="kernel failed"):
        select_mod.qsa_select_paged_tokens_with_scorer(
            q, cache, table, t2r, pos, ctx, 64, 4, scorer=E.index_scorer(cfg())
        )
    assert len(calls["op"]) == 1


def _fake_layer(M, monkeypatch, qsa_select, qsa_select_groups=None):
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
        qsa_select=qsa_select,
        qsa_select_groups=qsa_select_groups,
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


def test_call_site_flag_off_uses_the_stock_select(stub_load, monkeypatch):
    M = stub_load(ATTN)
    seen = []
    monkeypatch.setattr(
        M, "qsa_select_paged_tokens", lambda *a, **k: seen.append((a, k))
    )
    layer = _fake_layer(M, monkeypatch, qsa_select=None)
    M.Qwen4ExpAttention._select_tokens(
        layer, torch.zeros(1), torch.zeros(1), _qsa_meta()
    )
    ((a, k),) = seen
    assert a[:2] == ("index_q", "ccache") and "scorer" not in k


def test_call_site_flag_on_injects_the_routed_op(stub_load, fake_aiter, monkeypatch):
    M = stub_load(ATTN)
    api, _ = fake_aiter()
    monkeypatch.setattr(
        M, "qsa_select_paged_tokens", lambda *a, **k: pytest.fail("stock called")
    )
    seen = []

    def select(*a, scorer, **k):
        seen.append((a, k, scorer))

    qsa_select = functools.partial(select, scorer=E.index_scorer(cfg()))
    layer = _fake_layer(M, monkeypatch, qsa_select=qsa_select)
    M.Qwen4ExpAttention._select_tokens(
        layer, torch.zeros(1), torch.zeros(1), _qsa_meta()
    )
    ((a, k, scorer),) = seen
    assert scorer is api.qsa_paged_mqa_logits
    assert a[:2] == ("index_q", "ccache") and k["compressed_k_scale"] is None
    assert k["allow_flydsl"] is False and k["max_seq_len"] == 128


@pytest.mark.parametrize("on", [False, True])
def test_row_sharded_call_site_switches_with_the_flag(
    stub_load, fake_aiter, monkeypatch, on
):
    """#2847's row-sharded selection (`rows` given) takes the stock
    `qsa_select_paged_groups` when off and the routed op when on; the
    whole-batch entry is never called on that path."""
    M = stub_load(ATTN)
    api, _ = fake_aiter()
    seen, expanded = [], []

    def groups(*a, **k):
        seen.append((a, k))
        return torch.zeros(a[0].shape[0], 16, dtype=torch.int32), 67

    for name in ("qsa_select_paged_tokens", "qsa_select_paged_groups"):
        monkeypatch.setattr(
            M, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} called")
        )
    if not on:
        monkeypatch.setattr(M, "qsa_select_paged_groups", groups)
    monkeypatch.setattr(
        M, "qsa_expand_block_indices", lambda *a, **k: expanded.append((a, k))
    )
    layer = _fake_layer(
        M,
        monkeypatch,
        qsa_select=None,
        qsa_select_groups=(
            functools.partial(groups, scorer=E.index_scorer(cfg())) if on else None
        ),
    )
    layer.indexer.project_qk = lambda h, p: (torch.zeros(6, 4, 128), "raw_key")
    meta = _qsa_meta()
    meta.token_to_req = torch.zeros(6, dtype=torch.int32)
    meta.logical_positions = torch.arange(6)
    rows = SimpleNamespace(start=4, chunk=4, num_tokens=6, gather=lambda g: g)
    M.Qwen4ExpAttention._select_tokens(
        layer, torch.zeros(6), torch.zeros(6), meta, rows
    )
    ((a, k),) = seen
    assert a[0].shape[0] == 2 and a[1] == "ccache"  # this rank's rows 4..5
    assert (k.get("scorer") is api.qsa_paged_mqa_logits) is on
    ((ea, ek),) = expanded
    assert ea[0].shape == (4, 16) and ek["output_width"] == 67


def test_layer_exposes_its_caches_for_prewarm_only_when_on(stub_load):
    M = stub_load(ATTN)
    on = SimpleNamespace(
        qsa_select=object(), compressed_key_cache="c", compressed_key_scale="s"
    )
    assert M.Qwen4ExpAttention.qsa_index_scorer_caches(on) == ("c", "s")
    off = SimpleNamespace(
        qsa_select=None, compressed_key_cache="c", compressed_key_scale=None
    )
    assert M.Qwen4ExpAttention.qsa_index_scorer_caches(off) is None
    unbound = SimpleNamespace(qsa_select=object(), compressed_key_cache=None)
    assert M.Qwen4ExpAttention.qsa_index_scorer_caches(unbound) is None


# ---------------------------------------------------------------------------
# prewarm
# ---------------------------------------------------------------------------
class _Layer(torch.nn.Module):
    def __init__(self, caches):
        super().__init__()
        self._caches = caches

    def qsa_index_scorer_caches(self):
        return self._caches


def _model(*caches):
    m = torch.nn.Module()
    for i, c in enumerate(caches):
        m.add_module(f"l{i}", _Layer(c))
    return m


@pytest.mark.parametrize("block", [16, 32, 64])
def test_warmup_once_per_bound_layout_with_atom_dtypes(fake_aiter, block, caplog):
    from atom.aiter_fused_ops import prewarm_aiter_fused_ops

    _, calls = fake_aiter()
    E.configure(cfg())
    page = block // 4
    shared = torch.zeros(32, page, 1, 128, dtype=torch.bfloat16)
    target = _model((shared, None), (shared, None), None)
    draft = _model((torch.zeros(32, page, 1, 128, dtype=torch.bfloat16), None))
    with caplog.at_level(logging.INFO, logger="atom"):
        prewarm_aiter_fused_ops(cfg(), [target, draft], rank_name="r0")
        prewarm_aiter_fused_ops(cfg(), [target, draft], rank_name="r0")
    (pw,) = calls["warmup"]  # same layout everywhere, once per process
    assert pw["page_sizes"] == (page,) and pw["cache_dtypes"] == ("bf16",)
    assert pw["pos_dtypes"] == (torch.int64,) and pw["compressed_k_cache"] is shared
    assert pw["compressed_k_scale"] is None and pw["device"] == shared.device
    assert any(
        f"provider warm-up page={page} cache=bf16" in r.getMessage()
        for r in caplog.records
    )


def test_warmup_fp8_cache_passes_its_scale(fake_aiter):
    _, calls = fake_aiter()
    E.configure(cfg())
    (_, cache, *_), kw = call_args(fp8=True)
    E.prewarm([(cache, kw["compressed_k_scale"])])
    (pw,) = calls["warmup"]
    assert pw["cache_dtypes"] == ("fp8",)
    assert pw["compressed_k_scale"] is kw["compressed_k_scale"]


def test_warmup_failure_is_logged_not_raised(fake_aiter, caplog):
    _, calls = fake_aiter(warmup_raises=RuntimeError("no compiler"))
    E.configure(cfg())
    with caplog.at_level(logging.INFO, logger="atom"):
        out = E.prewarm([(torch.zeros(4, 4, 1, 128, dtype=torch.bfloat16), None)])
    assert list(out.values()) == [{P: False}] and len(calls["warmup"]) == 1
    assert any(
        "provider warm-up" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )


def test_no_warmup_without_an_enabled_op(fake_aiter):
    _, calls = fake_aiter(routed=False)
    E.configure(cfg())
    assert E.prewarm([(torch.zeros(4, 4, 1, 128), None)]) == {}
    assert not calls["warmup"]


# ---------------------------------------------------------------------------
# opt-in observer: aiter.plugin.explain() per class, counters, capture
# ---------------------------------------------------------------------------
def test_observer_explains_once_per_class_and_counts(fake_aiter, tmp_path, caplog):
    api, calls = fake_aiter()
    path = str(tmp_path / "cap_{rank}.json")
    scorer = E.index_scorer(cfg(capture=path))
    assert isinstance(scorer, E.ClassObserver) and scorer.op is api.qsa_paged_mqa_logits
    a64, kw64 = call_args(max_columns=64)
    a32, kw32 = call_args(max_columns=32)
    for _ in range(3):
        logits, visible = scorer(*a64, 4, **kw64)
    scorer(*a32, 4, **kw32)
    assert logits.shape == (8, 64) and visible.dtype == torch.int32
    assert len(calls["explain"]) == 2 and len(calls["op"]) == 4
    served = f"provider/{P}"
    assert E.decisions() == {served: 3, "native": 1}
    with caplog.at_level(logging.INFO, logger="atom"):
        E.log_decisions()
    assert any(
        f"QSA index scorer decisions: native=1, {served}=3" in r.getMessage()
        for r in caplog.records
    )
    scorer.capture.flush()
    data = json.loads(Path(scorer.capture.path).read_text(encoding="utf-8"))
    assert data["schema"] == C.SCHEMA and data["profile"] == P
    got = {
        c["signature"]["columns"]: (c["count"], c["decisions"]) for c in data["classes"]
    }
    assert got == {
        64: (3, {f"{served}:selected": 3}),
        32: (1, {"native:provider-declined": 1}),
    }
    assert len(data["plan"]) == 2


def test_observer_without_aiter_records_the_atom_path(tmp_path):
    seen = []

    def atom_scorer(*a, **k):
        seen.append(a)
        return "ok"

    obs = E.ClassObserver(str(tmp_path / "c.json"), atom_scorer, None, None)
    args, kw = call_args()
    assert obs(*args, 4, **kw) == "ok" and len(seen) == 1
    assert dict(obs.calls) == {f"{E.ATOM_PATH}:{E.CAPTURE_ONLY}": 1}


def test_bind_args_matches_the_scorer_signature():
    args, kw = call_args()
    a = E.bind_args(args + (4,), kw)
    assert len(a) == 12 and a[6] == 4 and a[7] is None and a[8] == 64
    with pytest.raises(TypeError):
        E.bind_args(args, {"logits": None, **kw})


@pytest.mark.parametrize("page", [4, 8, 16])
def test_capture_plan_rows_for_every_index_page(page):
    (q, cache, table, t2r, pos, ctx), _ = call_args(rows=4, requests=1, page=page)
    sig = C.describe_call(
        (q, cache, table, t2r, pos, ctx, 4, None, None, None, False, None)
    )
    row = C.plan_row(sig)
    assert row is not None and row["columns"] == 16 * page
    assert row.get("page_size", 4) == page


def test_capture_merge(tmp_path):
    (q, cache, table, t2r, pos, ctx), _ = call_args()
    for rank in (0, 1):
        cap = C.ScorerCapture(str(tmp_path / f"cap_{rank}.json"))
        cap.rank = rank
        ent = cap.class_entry(
            (q, cache, table, t2r, pos, ctx, 4, None, 64, None, False, None), P
        )
        cap.count(ent, "fused:ok")
        cap.flush()
    out = C.merge(sorted(str(p) for p in tmp_path.glob("cap_*.json")))
    (row,) = out["matrix"]
    assert row["count"] == 2 and row["decisions"] == {"fused:ok": 2}
    assert row["decision_uniform"]
