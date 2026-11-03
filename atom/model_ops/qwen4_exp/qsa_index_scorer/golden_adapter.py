# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Golden adapter: Phase-1 golden cases through ATOM's real scorer call path.

GPU only (gfx1250). Every case of the Phase-1 GPU golden suite (same specs as
aiter ``op_tests/test_qsa_index_scorer.py::CASES``) is replayed through:

``site``
    the exact object ``Qwen4ExpAttention`` installs -- a
    :class:`~.route.QSAScorerRoute` -- called the way the engine calls it
    (``qsa_select_paged_tokens`` staging: ``row_starts`` scratch, bucket
    ``max_columns``, ``allow_flydsl=is_prefill``, positional ratio, keyword
    tail), after ``prepare()`` on the cache as ``bind_caches`` does. Checked
    against the frozen golden (``$QSA_GOLDEN_DIR/qsa_golden.py``,
    ``compare_logits(rtol=1e-4, atol=1e-4)`` over ``valid_page & c < visible``)
    plus: path taken == fused for profile-covered cases, ``visible`` exact,
    every element outside the mask bit-identical to the golden ``-inf``, no NaN,
    ``row_starts`` zeroed, inputs untouched; fused vs ATOM native on identical
    inputs (same comparator); CUDA-graph capture/replay bitwise == eager.

``engine``
    the whole selection path the engine runs, ``qsa_select_paged_tokens_routed``
    (stock loop: power-of-two ``max_seq_len`` bucketing of ``max_columns``,
    row chunking under the 128 MiB logits workspace -- 2880-row prefill becomes
    5 x 512 + 320 -- per-chunk ``row_starts``) over engine-shaped metadata
    (``block_tables`` int32 ``[reqs, width]``, ``token_to_req`` int32,
    ``logical_positions`` int64, ``seq_lens`` int32, ``max_seq_len``). Each
    scorer chunk is checked against the golden of that chunk; the stock
    ``qsa_select_paged_tokens`` and the routed copy with the native scorer
    must agree bitwise; fused-vs-native selection agreement is reported.

``engine_pad``
    decode cases with CUDA-graph pad-M as ``_build_qsa_metadata`` builds it
    (``build_batch_ids(pad_to=...)``: padded rows ``token_to_req = -1``,
    ``logical_positions = -1``; padded requests ``seq_len 0``).

``fallbacks``
    every native-fallback reason reachable from ATOM's call site, each checked
    to have taken the native path, carried its reason, and returned the native
    result bitwise.

Run (root, under gpu-lock)::

    QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline \
      python -m atom.model_ops.qwen4_exp.qsa_index_scorer.golden_adapter \
        --json out.json [-k substr] [--profile qsa_page4_gfx1250_v1] [--modes site,engine,...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import torch

from .route import CAPTURE_ONLY, OK, QSAScorerRoute

RTOL = ATOL = 1e-4  # frozen Phase-1 tolerance
W, PAGES, CTX = 16384, 738964, 262144
TOKEN_TOPK = 2048  # Qwen3.8-Flash-Next indexer_budget
DEFAULT_PROFILE = "qsa_page4_gfx1250_v1"

# Phase-1 GPU golden suite (identical specs to aiter's GPU test).
CASES = [
    (
        "I5_min_decode_bf16",
        {
            "regime": "decode",
            "rows": 4,
            "requests": 1,
            "fmt": "bf16",
            "table_width": 16,
            "cache_pages": 64,
        },
    ),
    (
        "I5_min_decode_fp8",
        {
            "regime": "decode",
            "rows": 4,
            "requests": 1,
            "fmt": "fp8",
            "table_width": 16,
            "cache_pages": 64,
        },
    ),
    (
        "I6_2req_bf16",
        {
            "regime": "prefill",
            "rows": 16,
            "requests": 2,
            "fmt": "bf16",
            "table_width": 32,
            "cache_pages": 128,
        },
    ),
    (
        "I6_2req_fp8",
        {
            "regime": "prefill",
            "rows": 16,
            "requests": 2,
            "fmt": "fp8",
            "table_width": 32,
            "cache_pages": 128,
        },
    ),
    (
        "I7_64rows_bf16",
        {
            "regime": "prefill",
            "rows": 64,
            "requests": 2,
            "fmt": "bf16",
            "table_width": 64,
            "cache_pages": 256,
        },
    ),
    (
        "I8_272rows_bf16",
        {
            "regime": "prefill",
            "rows": 272,
            "requests": 1,
            "fmt": "bf16",
            "table_width": 64,
            "cache_pages": 64,
        },
    ),
    (
        "tail_columns_1000_bf16",
        {
            "regime": "prefill",
            "rows": 8,
            "requests": 2,
            "fmt": "bf16",
            "table_width": 256,
            "cache_pages": 1024,
            "max_columns": 1000,
        },
    ),
    (
        "tail_columns_1000_fp8",
        {
            "regime": "prefill",
            "rows": 8,
            "requests": 2,
            "fmt": "fp8",
            "table_width": 256,
            "cache_pages": 1024,
            "max_columns": 1000,
        },
    ),
    (
        "edge_visible_zero_bf16",
        {
            "regime": "decode",
            "rows": 4,
            "requests": 1,
            "fmt": "bf16",
            "table_width": 16,
            "cache_pages": 64,
            "patch": "visible_zero",
        },
    ),
    (
        "edge_t2r_invalid_bf16",
        {
            "regime": "decode",
            "rows": 4,
            "requests": 1,
            "fmt": "bf16",
            "table_width": 16,
            "cache_pages": 64,
            "patch": "t2r_invalid",
        },
    ),
    (
        "edge_bad_page_bf16",
        {
            "regime": "decode",
            "rows": 4,
            "requests": 1,
            "fmt": "bf16",
            "table_width": 16,
            "cache_pages": 64,
            "patch": "bad_page",
        },
    ),
    (
        "edge_bad_page_fp8",
        {
            "regime": "decode",
            "rows": 4,
            "requests": 1,
            "fmt": "fp8",
            "table_width": 16,
            "cache_pages": 64,
            "patch": "bad_page",
        },
    ),
    (
        "edge_page_oob_bf16",
        {
            "regime": "decode",
            "rows": 4,
            "requests": 1,
            "fmt": "bf16",
            "table_width": 16,
            "cache_pages": 64,
            "patch": "bad_page_oob",
        },
    ),
    (
        "edge_page_oob_fp8",
        {
            "regime": "decode",
            "rows": 4,
            "requests": 1,
            "fmt": "fp8",
            "table_width": 16,
            "cache_pages": 64,
            "patch": "bad_page_oob",
        },
    ),
    (
        "edge_short_ctx_fp8",
        {
            "regime": "decode",
            "rows": 4,
            "requests": 1,
            "fmt": "fp8",
            "table_width": 16,
            "cache_pages": 64,
            "patch": "short_ctx",
        },
    ),
    (
        "I1_e2e_decode_bf16",
        {
            "regime": "decode",
            "rows": 64,
            "requests": 16,
            "fmt": "bf16",
            "table_width": W,
            "cache_pages": PAGES,
            "max_ctx": CTX,
        },
    ),
    (
        "I2_e2e_decode_fp8",
        {
            "regime": "decode",
            "rows": 64,
            "requests": 16,
            "fmt": "fp8",
            "table_width": W,
            "cache_pages": PAGES,
            "max_ctx": CTX,
        },
    ),
    (
        "I3_e2e_draft_bf16",
        {
            "regime": "draft",
            "rows": 16,
            "requests": 16,
            "fmt": "bf16",
            "table_width": W,
            "cache_pages": PAGES,
            "max_ctx": CTX,
        },
    ),
    (
        "I3_e2e_draft_fp8",
        {
            "regime": "draft",
            "rows": 16,
            "requests": 16,
            "fmt": "fp8",
            "table_width": W,
            "cache_pages": PAGES,
            "max_ctx": CTX,
        },
    ),
    (
        "prefill_chunk512_bf16",
        {
            "regime": "prefill",
            "rows": 512,
            "requests": 2,
            "fmt": "bf16",
            "table_width": W,
            "cache_pages": PAGES,
            "max_ctx": CTX,
        },
    ),
    (
        "prefill_chunk512_fp8",
        {
            "regime": "prefill",
            "rows": 512,
            "requests": 2,
            "fmt": "fp8",
            "table_width": W,
            "cache_pages": PAGES,
            "max_ctx": CTX,
        },
    ),
    (
        "I14_decode_c32768_bf16",
        {
            "regime": "decode",
            "rows": 64,
            "requests": 16,
            "fmt": "bf16",
            "table_width": W,
            "cache_pages": PAGES,
            "max_ctx": CTX,
            "max_columns": 32768,
        },
    ),
]
#: Engine-path extras: the e2e prefill (2880 rows / 2 requests) the engine
#: chunks into 5 x 512 + 320 rows.
ENGINE_EXTRA = [
    (
        "e2e_prefill2880_bf16",
        {
            "regime": "prefill",
            "rows": 2880,
            "requests": 2,
            "fmt": "bf16",
            "table_width": W,
            "cache_pages": PAGES,
            "max_ctx": CTX,
        },
    ),
    (
        "e2e_prefill2880_fp8",
        {
            "regime": "prefill",
            "rows": 2880,
            "requests": 2,
            "fmt": "fp8",
            "table_width": W,
            "cache_pages": PAGES,
            "max_ctx": CTX,
        },
    ),
]
#: Cases that must take the fused path under the default profile (profile
#: rows 'pass'); the edge cases share the I5 shape and must too.
EXPECT_FUSED = {n for n, _ in CASES}

G = S = None


def load_golden():
    global G, S
    if G is None:
        gdir = os.environ.get("QSA_GOLDEN_DIR")
        if not gdir:
            raise SystemExit(
                "set QSA_GOLDEN_DIR to the frozen golden_baseline directory"
            )
        sys.path.insert(0, os.path.abspath(gdir))
        import qsa_golden
        import qsa_shapes

        G, S = qsa_golden, qsa_shapes
    return G, S


def fixture(spec, seed=0):
    """Same tensors as aiter's GPU golden test (golden ``build_case`` + patches)."""
    G, S = load_golden()
    max_ctx = spec.get("max_ctx", 16 * spec["table_width"])
    case = S.make_case(
        "t",
        spec["regime"],
        rows=spec["rows"],
        requests=spec["requests"],
        max_ctx=max_ctx,
        index_dtype=spec["fmt"],
        table_width=spec["table_width"],
        cache_pages=spec["cache_pages"],
    )
    d = G.build_case(case, device="cuda", seed=seed)
    d["max_ctx"] = max_ctx
    d["max_columns"] = spec.get("max_columns", case.columns)
    what = spec.get("patch")
    if what == "visible_zero":
        d["query_positions"][0] = 2
        d["query_positions"][1] = 0
    elif what == "t2r_invalid":
        d["token_to_request"][0] = -1
        d["token_to_request"][1] = 999
    elif what == "bad_page":
        d["page_table"][0, 0] = -1
    elif what == "bad_page_oob":
        d["page_table"][0, 0] = 1 << 24
    elif what == "short_ctx":
        d["context_lens"][0] = 8
    return d


def scorer_args(d, rows_slice=None, cols=None, row_starts=None, allow_flydsl=False):
    """Bound native arguments exactly as the stock select loop passes them."""
    sl = slice(None) if rows_slice is None else rows_slice
    return (
        d["q"][sl],
        d["compressed_k_cache"],
        d["page_table"],
        d["token_to_request"][sl],
        d["query_positions"][sl],
        d["context_lens"],
        4,
        None,
        int(d["max_columns"] if cols is None else cols),
        row_starts,
        allow_flydsl,
        d["compressed_k_scale"],
    )


def golden_compare(out, vis, args):
    G, _ = load_golden()
    q, cache, table, t2r, pos, ctx, ratio, _div, cols, _rs, _fly, scale = args
    gold, gvis, valid = G.reference_logits(
        q,
        cache,
        table,
        t2r,
        pos,
        ctx,
        compress_ratio=ratio,
        max_columns=cols,
        compressed_k_scale=scale,
    )
    res = G.compare_logits(out, vis, gold, gvis, valid, rtol=RTOL, atol=ATOL)
    colix = torch.arange(gold.shape[1], device=gold.device)
    inside = valid & (colix[None, :] < gvis.to(torch.int64)[:, None])
    oa, ob = out[~inside], gold[~inside]
    res["outside_diff"] = int(
        (~((oa == ob) | (torch.isnan(oa) & torch.isnan(ob)))).sum()
    )
    res["nan"] = int(torch.isnan(out).sum())
    res["pass"] = bool(res["pass"] and res["outside_diff"] == 0 and res["nan"] == 0)
    return res, (gold, gvis, valid)


def _oob_contract(out, d):
    seg = out[d["token_to_request"] == 0][:, :4]
    return {
        "pass": bool(
            torch.isinf(seg).all() and (seg < 0).all() and not torch.isnan(out).any()
        ),
        "contract": "page >= num_pages -> -inf in its 4 columns, no NaN",
    }


def run_site(name, spec, route):
    d = fixture(spec)
    rows = spec["rows"]
    is_prefill = spec["regime"] == "prefill"
    route.prepare(d["compressed_k_cache"], d["compressed_k_scale"])  # bind_caches
    snap = [t.clone() for t in scorer_args(d)[:6]]
    rs = torch.full((rows,), 7, dtype=torch.int32, device="cuda")
    args = scorer_args(d, row_starts=rs, allow_flydsl=is_prefill)
    reason, detail = route.decision_for(*args)
    if reason != OK and native_aborts(args):
        return {
            "name": name,
            "mode": "site",
            "reason": reason,
            "detail": detail,
            "passed": False,
            "error": f"not fused and {NATIVE_ABORT_NOTE}",
        }
    before = dict(route.counts)
    out, vis = route(*args)
    torch.cuda.synchronize()
    took = {
        k: v - before.get(k, 0)
        for k, v in route.counts.items()
        if v != before.get(k, 0)
    }
    path = "fused" if any(p == "fused" for p, _ in took) else "native"
    rec = {
        "name": name,
        "mode": "site",
        "reason": reason,
        "detail": detail,
        "path": path,
    }
    checks = {
        "expected_path": path == "fused" if name in EXPECT_FUSED else True,
        "row_starts_zero": bool((rs == 0).all()),
        "inputs_untouched": all(
            torch.equal(a, b) for a, b in zip(snap, scorer_args(d)[:6])
        ),
    }
    if spec.get("patch") == "bad_page_oob":
        res = _oob_contract(out, d)
    else:
        res, _ = golden_compare(out, vis, args)
    # fused vs native on identical inputs (same comparator, native as reference)
    nat_args = scorer_args(
        d,
        row_starts=torch.empty(rows, dtype=torch.int32, device="cuda"),
        allow_flydsl=is_prefill,
    )
    G, _ = load_golden()
    if native_aborts(nat_args):
        vs_native = {"pass": True, "skipped": NATIVE_ABORT_NOTE}
    elif spec.get("patch") == "bad_page_oob":
        nout, nvis = route.native(*nat_args)
        vs_native = {
            "pass": bool(torch.equal(vis, nvis)),
            "note": "oob page: visible only",
        }
    else:
        nout, nvis = route.native(*nat_args)
        _, (_gold, gvis, valid) = golden_compare(nout, nvis, nat_args)
        vs_native = G.compare_logits(out, vis, nout, nvis, valid, rtol=RTOL, atol=ATOL)
    # CUDA graph: capture the route call, replay, bitwise == eager
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        route(*args)  # warm-up outside capture (lazy kernel build)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gout, gvis = route(*args)
    gout.fill_(0)
    g.replay()
    torch.cuda.synchronize()
    checks["graph_bitwise"] = bool(
        torch.equal(gout.view(torch.int32), out.view(torch.int32))
        and torch.equal(gvis, vis)
    )
    rec.update(
        golden=res,
        vs_native=vs_native,
        checks=checks,
        passed=bool(res["pass"] and vs_native["pass"] and all(checks.values())),
    )
    return rec


#: Pre-existing ATOM issue (records/g2/final-report.md): the native scorer's
#: fp8 MFMA route (rows > 256) hits an LLVM PHINode assertion on gfx1250 and
#: aborts the process. The adapter never calls native there.
NATIVE_ABORT_NOTE = (
    "ATOM native fp8 scorer, rows > 256: aborts (LLVM PHINode); native leg skipped"
)


def native_aborts(args) -> bool:
    q, cache = args[0], args[1]
    return int(q.shape[0]) > 256 and cache.dtype not in (torch.bfloat16, torch.uint8)


class _NativeAbortRisk(RuntimeError):
    pass


class _Tap:
    """Scorer wrapper recording each chunk call of the select loop.

    Refuses (raises) a chunk that would reach ATOM's aborting fp8 native route.
    """

    def __init__(self, scorer, route=None):
        self.scorer = scorer
        self.route = route
        self.calls = []

    def __call__(self, *args, **kwargs):
        a = QSAScorerRoute._bind(args, kwargs)
        if native_aborts(a) and (
            self.route is None or self.route.decision_for(*a)[0] != OK
        ):
            raise _NativeAbortRisk(f"rows={int(a[0].shape[0])}: {NATIVE_ABORT_NOTE}")
        out = self.scorer(*a)
        self.calls.append((a, out))
        return out


def _engine_metadata(d):
    """Engine dtypes/layouts (Qwen4ExpQSAMetadata): asserted, not converted."""
    assert d["page_table"].dtype == torch.int32 and d["page_table"].is_contiguous()
    assert d["token_to_request"].dtype == torch.int32
    assert d["query_positions"].dtype == torch.int64
    assert d["context_lens"].dtype == torch.int32
    return (
        d["page_table"],
        d["token_to_request"],
        d["query_positions"],
        d["context_lens"],
    )


def run_engine(name, spec, route, pad=0):
    from atom.model_ops.qwen4_exp.ops.qsa import qsa_select_paged_tokens

    from .select import qsa_select_paged_tokens_routed

    _, S = load_golden()
    d = fixture(spec)
    if pad:
        # CUDA-graph pad-M: the last `pad` requests are padding (build_batch_ids
        # pad=-1 rows, logical -1, seq_len 0), rows/requests unchanged.
        reqs = spec["requests"]
        padded_rows = d["token_to_request"] >= reqs - pad
        d["token_to_request"][padded_rows] = -1
        d["query_positions"][padded_rows] = -1
        d["context_lens"][reqs - pad :] = 0
    if "max_columns" in spec and spec["max_columns"] != S.columns_bucket(
        d["max_ctx"], spec["table_width"]
    ):
        return {
            "name": name,
            "mode": "engine",
            "skipped": True,
            "passed": True,
            "note": "explicit max_columns is not an engine bucket; covered by site mode",
        }
    is_prefill = spec["regime"] == "prefill"
    table, t2r, pos, ctx = _engine_metadata(d)
    route.prepare(d["compressed_k_cache"], d["compressed_k_scale"])
    kw = {
        "max_seq_len": d["max_ctx"],
        "allow_flydsl": is_prefill,
        "compressed_k_scale": d["compressed_k_scale"],
    }
    tap = _Tap(route, route)
    before = dict(route.counts)
    tag = {
        "name": name + (f"+pad{pad}" if pad else ""),
        "mode": "engine_pad" if pad else "engine",
    }
    try:
        sel = qsa_select_paged_tokens_routed(
            d["q"],
            d["compressed_k_cache"],
            table,
            t2r,
            pos,
            ctx,
            TOKEN_TOPK,
            4,
            scorer=tap,
            **kw,
        )
    except _NativeAbortRisk as exc:
        return {**tag, "passed": False, "error": f"uncovered chunk: {exc}"}
    torch.cuda.synchronize()
    took = {
        f"{p}:{r}": v - before.get((p, r), 0)
        for (p, r), v in route.counts.items()
        if v != before.get((p, r), 0)
    }
    chunks = []
    ok = True
    for a, (lg, vg) in tap.calls:
        res, _ = golden_compare(lg, vg, a)
        chunks.append(
            {
                "rows": int(a[0].shape[0]),
                "max_columns": a[8],
                "pass": res["pass"],
                "max_abs_err": res.get("max_abs_err"),
                "bad": res.get("bad"),
            }
        )
        ok &= res["pass"]
    if any(native_aborts(a) for a, _ in tap.calls):
        return {
            **tag,
            "chunks": chunks,
            "paths": took,
            "native_skipped": NATIVE_ABORT_NOTE,
            "passed": bool(ok),
        }
    nat_tap = _Tap(route.native)
    sel_native = qsa_select_paged_tokens_routed(
        d["q"],
        d["compressed_k_cache"],
        table,
        t2r,
        pos,
        ctx,
        TOKEN_TOPK,
        4,
        scorer=nat_tap,
        **kw,
    )
    sel_stock = qsa_select_paged_tokens(
        d["q"], d["compressed_k_cache"], table, t2r, pos, ctx, TOKEN_TOPK, 4, **kw
    )
    torch.cuda.synchronize()
    stock_equal = bool(torch.equal(sel_native, sel_stock))
    rows_equal = (sel == sel_native).all(dim=1)
    rec = {
        **tag,
        "chunks": chunks,
        "paths": took,
        "routed_native_equals_stock": stock_equal,
        "selection_rows_equal_fused_vs_native": f"{int(rows_equal.sum())}/{rows_equal.numel()}",
        "passed": bool(ok and stock_equal),
    }
    return rec


# ---------------------------------------------------------------------------
# Fallback reasons reachable from ATOM's call site
# ---------------------------------------------------------------------------
def _decode_small(fmt="bf16", **kw):
    spec = {
        "regime": "decode",
        "rows": 4,
        "requests": 1,
        "fmt": fmt,
        "table_width": 16,
        "cache_pages": 64,
    }
    spec.update(kw)
    return fixture(spec)


def _with(d, **kw):
    e = dict(d)
    e.update(kw)
    return e


def _fallback_cases(profile):
    """name -> (route profile, data builder, scorer-arg overrides, expected reasons)."""

    def page16(d):
        c = d["compressed_k_cache"]
        big = torch.zeros(
            (c.shape[0], 16, 1, c.shape[3]), dtype=c.dtype, device=c.device
        )
        big[:, :4] = c
        return _with(d, compressed_k_cache=big)

    def noncontig_cache(d):
        c = d["compressed_k_cache"]
        big = torch.zeros(
            (c.shape[0], 8, 1, c.shape[3]), dtype=c.dtype, device=c.device
        )
        big[:, :4] = c
        return _with(d, compressed_k_cache=big[:, :4])

    def misaligned_cache(d):
        c = d["compressed_k_cache"]
        buf = torch.empty(c.numel() + 8, dtype=c.dtype, device=c.device)
        v = buf[8:].view(c.shape)
        v.copy_(c)
        return _with(d, compressed_k_cache=v)

    def noncontig_scale(d):
        s = d["compressed_k_scale"]
        big = torch.ones((s.shape[0], 8, 1), dtype=s.dtype, device=s.device)
        big[:, :4] = s
        return _with(d, compressed_k_scale=big[:, :4])

    def q_fp32(d):
        return _with(d, q=d["q"].float())

    def q_heads8(d):
        return _with(d, q=torch.cat([d["q"], d["q"]], dim=1))

    def q_noncontig(d):
        q = d["q"]
        big = torch.zeros(
            (q.shape[0], q.shape[1], 2 * q.shape[2]), dtype=q.dtype, device=q.device
        )
        big[..., : q.shape[2]] = q
        return _with(d, q=big[..., : q.shape[2]])

    def table_noncontig(d):
        t = d["page_table"]
        big = torch.full(
            (t.shape[0], 2 * t.shape[1]), -1, dtype=t.dtype, device=t.device
        )
        big[:, : t.shape[1]] = t
        return _with(d, page_table=big[:, : t.shape[1]])

    def empty_rows(d):
        return _with(
            d,
            q=d["q"][:0],
            token_to_request=d["token_to_request"][:0],
            query_positions=d["query_positions"][:0],
        )

    def rows6(d):  # 6 rows / 1 request: no profiled regime
        return _with(
            d,
            q=torch.cat([d["q"], d["q"][:2]]),
            token_to_request=torch.cat(
                [d["token_to_request"], d["token_to_request"][:2]]
            ),
            query_positions=torch.cat([d["query_positions"], d["query_positions"][:2]]),
        )

    def width32(d):
        t = d["page_table"]
        big = torch.full((t.shape[0], 32), -1, dtype=t.dtype, device=t.device)
        big[:, : t.shape[1]] = t
        return _with(d, page_table=big)

    ident = lambda d: d
    return [
        ("capture_only(disabled)", "", ident, {}, {CAPTURE_ONLY}),
        (
            "profile_not_found",
            "no_such_qsa_profile_zz",
            ident,
            {},
            {"profile_not_found"},
        ),
        (
            "cache_shape(page16=KV block 64)",
            profile,
            page16,
            {},
            {"cache_shape_unsupported"},
        ),
        (
            "cache_layout(noncontig)",
            profile,
            noncontig_cache,
            {},
            {"cache_layout_unsupported"},
        ),
        (
            "cache_alignment(+16B)",
            profile,
            misaligned_cache,
            {},
            {"cache_alignment_unsupported"},
        ),
        (
            "scale_layout(fp8 noncontig)",
            profile,
            noncontig_scale,
            {"fmt": "fp8"},
            {"scale_layout_unsupported"},
        ),
        (
            "q_dtype(fp32 q, fp8 cache)",
            profile,
            q_fp32,
            {"fmt": "fp8"},
            {"q_dtype_unsupported"},
        ),
        ("q_shape(8 heads)", profile, q_heads8, {}, {"q_shape_unsupported"}),
        ("q_layout(noncontig)", profile, q_noncontig, {}, {"q_layout_unsupported"}),
        (
            "metadata(noncontig table)",
            profile,
            table_noncontig,
            {},
            {"metadata_unsupported"},
        ),
        (
            "divisor_uncovered",
            profile,
            ident,
            {"score_divisor": 8.0},
            {"divisor_uncovered"},
        ),
        (
            "max_columns_invalid",
            profile,
            ident,
            {"max_columns": 0},
            {"max_columns_invalid"},
        ),
        (
            "compress_ratio(8)",
            profile,
            ident,
            {"compress_ratio": 8},
            {"compress_ratio_unsupported"},
        ),
        ("empty_call", profile, empty_rows, {}, {"empty_call"}),
        ("rows_rule(6 rows/1 req)", profile, rows6, {}, {"rows_rule_uncovered"}),
        (
            "columns_uncovered(max_columns 32)",
            profile,
            ident,
            {"max_columns": 32},
            {"columns_uncovered"},
        ),
        (
            "table_width(32)",
            profile,
            width32,
            {"max_columns": 64},
            {"table_width_uncovered", "columns_uncovered"},
        ),
    ]


def run_fallbacks(profile):
    recs = []
    for name, prof, build, over, expect in _fallback_cases(profile):
        fmt = over.pop("fmt", "bf16") if "fmt" in over else "bf16"
        try:
            d = build(_decode_small(fmt))
            route = QSAScorerRoute(prof, site=f"fallback:{name}")
            route.prepare(d["compressed_k_cache"], d["compressed_k_scale"])
            rows = int(d["q"].shape[0])
            a = list(
                scorer_args(
                    d, row_starts=torch.empty(rows, dtype=torch.int32, device="cuda")
                )
            )
            for k, v in over.items():
                a[{"score_divisor": 7, "max_columns": 8, "compress_ratio": 6}[k]] = v
            reason, detail = route.decision_for(*a)
            out = route(*a)
            ref = route.native(*a)
            torch.cuda.synchronize()
            same = torch.equal(
                out[0].view(torch.int32), ref[0].view(torch.int32)
            ) and torch.equal(out[1], ref[1])
            native_only = all(p == "native" for p, _ in route.counts)
            recs.append(
                {
                    "name": "fallback:" + name,
                    "mode": "fallbacks",
                    "reason": reason,
                    "detail": detail,
                    "expected": sorted(expect),
                    "native_bitwise": bool(same),
                    "native_path": native_only,
                    "passed": bool(reason in expect and same and native_only),
                }
            )
        except Exception as exc:  # noqa: BLE001
            recs.append(
                {
                    "name": "fallback:" + name,
                    "mode": "fallbacks",
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )
    # profile_kernel_mismatch: same profile with a different kernel sha
    try:
        import importlib

        P = importlib.import_module("aiter.ops.qsa_index_scorer.profiles")
        prof = P.load_profile(profile)
        data = json.loads(json.dumps(prof.data))
        for key in ("kernel", "kernel_sha256"):
            if isinstance(data.get(key), dict) and "sha256" in data[key]:
                data[key]["sha256"] = "0" * 64
            elif key == "kernel_sha256" and key in data:
                data[key] = "0" * 64
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(data, fh)
        d = _decode_small()
        route = QSAScorerRoute(fh.name, site="fallback:kernel_mismatch")
        reason = route.prepare(d["compressed_k_cache"], None)
        recs.append(
            {
                "name": "fallback:profile_kernel_mismatch(temp profile)",
                "mode": "fallbacks",
                "reason": reason,
                "passed": reason in ("profile_kernel_mismatch", "profile_invalid"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        recs.append(
            {
                "name": "fallback:profile_kernel_mismatch(temp profile)",
                "mode": "fallbacks",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        )
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="QSA index scorer golden adapter (ATOM call path)"
    )
    ap.add_argument("-k", default="")
    ap.add_argument("--json")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--modes", default="site,engine,engine_pad,fallbacks")
    a = ap.parse_args(argv)
    modes = set(a.modes.split(","))
    load_golden()
    props = torch.cuda.get_device_properties(0)
    if not str(props.gcnArchName).startswith("gfx1250"):
        print(f"SKIP: gfx1250 only (device {props.gcnArchName})")
        return 0
    route = QSAScorerRoute(a.profile, site="golden_adapter")
    recs = []

    def run(fn, name, *args):
        if a.k and a.k not in name:
            return
        t0 = time.time()
        try:
            rec = fn(name, *args)
        except Exception as exc:  # noqa: BLE001
            rec = {
                "name": name,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
        rec["seconds"] = round(time.time() - t0, 2)
        print(
            f"[{'PASS' if rec['passed'] else 'FAIL'}] {rec.get('mode', '')} {rec['name']} "
            f"{json.dumps(rec, default=str)[:400]}",
            flush=True,
        )
        recs.append(rec)
        torch.cuda.empty_cache()

    if "site" in modes:
        for name, spec in CASES:
            run(run_site, name, spec, route)
    if "engine" in modes:
        for name, spec in CASES + ENGINE_EXTRA:
            run(run_engine, name, spec, route)
    if "engine_pad" in modes:
        for name, spec in CASES:
            if (
                spec["regime"] == "decode"
                and spec["requests"] >= 4
                and "patch" not in spec
            ):
                run(run_engine, name, spec, route, 3)
    if "fallbacks" in modes and not a.k:
        for rec in run_fallbacks(a.profile):
            print(
                f"[{'PASS' if rec['passed'] else 'FAIL'}] {rec['name']} {json.dumps(rec)[:300]}"
            )
            recs.append(rec)
    bad = [r["name"] for r in recs if not r["passed"]]
    summary = {
        "pass": not bad,
        "n": len(recs),
        "failed": bad,
        "profile": a.profile,
        "route": route.summary(),
        "device": f"{props.gcnArchName} cu={props.multi_processor_count}",
        "golden_dir": os.environ.get("QSA_GOLDEN_DIR"),
        "records": recs,
    }
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(summary, fh, indent=1, default=str)
    print(f"[SUMMARY] {'PASS' if not bad else 'FAIL'} n={len(recs)} failed={bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
