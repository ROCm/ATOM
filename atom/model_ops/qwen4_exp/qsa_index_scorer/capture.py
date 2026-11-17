# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Capture hook: every QSA index scorer call class, to JSON.

Enabled by ``--qsa-index-scorer-capture PATH`` (``QSAIndexScorerConfig.capture``).
The opt-in :class:`~.enablement.ClassObserver` folds scorer calls into
*classes* (everything that can change a decision or a kernel route: shapes,
dtypes, strides, base alignments, bucket ``max_columns`` / columns,
``row_starts`` staging, ``allow_flydsl``, cache layout and format) with a
count, the rank and aiter's decision for the class (``aiter.ops.qsa.explain``:
``fused`` / ``native-triton`` / ``native-torch``; ``atom`` when ATOM's own
scorer runs). The file is rewritten whenever a new class appears and at exit,
so a killed server still leaves everything seen up to its last new class.

Output (``schema`` ``atom.qsa_index_scorer.capture/2``)::

    {"schema", "profile", "rank", "pid", "classes": [...],
     "plan": [{"cache_dtype", "rows", "requests", "columns", "table_width",
               "cache_pages"}, ...]}

``plan`` lists the bf16/fp8 page-4 classes in aiter ``gen_profile --plan-file``
form. Merge per-rank files and build the coverage matrix with::

    python -m atom.model_ops.qwen4_exp.qsa_index_scorer.capture merge \
        cap.rank*.json --plan plan.json --matrix matrix.json

Pure Python (no Triton / aiter).
"""

from __future__ import annotations

import argparse
import atexit
import glob
import json
import logging
import os
import sys
import threading

from .enablement import cache_dtype_name

SCHEMA = "atom.qsa_index_scorer.capture/2"
logger = logging.getLogger("atom")
#: Index page sizes aiter's kernel serves: KV block 16 / 32 / 64 at ratio 4.
PAGES = (4, 8, 16)

#: Phase-1 golden cases (aiter ``op_tests/test_qsa_index_scorer.py::CASES``,
#: the same specs as the Phase-1 GPU suite) as coverage keys:
#: name -> (cache_dtype, rows, requests, columns, table_width).
GOLDEN_CLASSES = {
    "I5_min_decode_bf16": ("bf16", 4, 1, 64, 16),
    "I5_min_decode_fp8": ("fp8", 4, 1, 64, 16),
    "I6_2req_bf16": ("bf16", 16, 2, 128, 32),
    "I6_2req_fp8": ("fp8", 16, 2, 128, 32),
    "I7_64rows_bf16": ("bf16", 64, 2, 256, 64),
    "I8_272rows_bf16": ("bf16", 272, 1, 256, 64),
    "tail_columns_1000_bf16": ("bf16", 8, 2, 1000, 256),
    "tail_columns_1000_fp8": ("fp8", 8, 2, 1000, 256),
    "I1_e2e_decode_bf16": ("bf16", 64, 16, 65536, 16384),
    "I2_e2e_decode_fp8": ("fp8", 64, 16, 65536, 16384),
    "I3_e2e_draft_bf16": ("bf16", 16, 16, 65536, 16384),
    "I3_e2e_draft_fp8": ("fp8", 16, 16, 65536, 16384),
    "prefill_chunk512_bf16": ("bf16", 512, 2, 65536, 16384),
    "prefill_chunk512_fp8": ("fp8", 512, 2, 65536, 16384),
    "I14_decode_c32768_bf16": ("bf16", 64, 16, 32768, 16384),
}


def regime_of(rows: int, requests: int) -> str | None:
    """aiter's row regimes (same thresholds), for capture records."""
    rows, requests = int(rows), int(requests)
    if rows < 1 or requests < 1:
        return None
    if rows <= requests:
        return "draft"
    if rows >= 32 and rows >= 32 * requests:
        return "prefill"
    if rows % 4 == 0:
        return "decode"
    return None


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception as exc:  # noqa: BLE001 -- best effort; fall back to env
        logger.debug("qsa capture: no torch.distributed rank: %r", exc)
    for var in ("RANK", "LOCAL_RANK"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var])
    return 0


def _t(t, align: int = 0) -> dict | None:
    if t is None:
        return None
    d = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "stride": list(t.stride()),
        "contiguous": bool(t.is_contiguous()),
        "device": str(t.device.type),
    }
    if align:
        d[f"base_mod_{align}"] = int(t.data_ptr() % align)
    return d


def describe_call(args: tuple) -> dict:
    """JSON class signature of one scorer call (bound native arguments)."""
    (
        q,
        cache,
        table,
        t2r,
        pos,
        ctx,
        ratio,
        divisor,
        max_columns,
        row_starts,
        allow_flydsl,
        scale,
    ) = args
    width = int(table.shape[1]) if table.ndim == 2 else 0
    page = int(cache.shape[1]) if cache.ndim >= 2 else 0
    columns = width * page
    if max_columns is not None:
        columns = max(1, min(columns, int(max_columns)))
    rows = int(q.shape[0]) if q.ndim >= 1 else 0
    requests = int(table.shape[0]) if table.ndim >= 1 else 0
    return {
        "cache_dtype": cache_dtype_name(cache, scale),
        "rows": rows,
        "requests": requests,
        "columns": columns,
        "table_width": width,
        "page_size": page,
        "cache_pages": int(cache.shape[0]) if cache.ndim >= 1 else 0,
        "regime": regime_of(rows, requests),
        "max_columns": None if max_columns is None else int(max_columns),
        "compress_ratio": int(ratio),
        "score_divisor": None if divisor is None else float(divisor),
        "allow_flydsl": bool(allow_flydsl),
        "row_starts": _t(row_starts),
        "q": _t(q, 16),
        "compressed_k_cache": _t(cache, 128),
        "compressed_k_scale": _t(scale, 4),
        "page_table": _t(table),
        "token_to_request": _t(t2r),
        "query_positions": _t(pos),
        "context_lens": _t(ctx),
    }


def plan_row(sig: dict) -> dict | None:
    """aiter ``gen_profile --plan-file`` row, or None outside the kernel domain.

    Index pages 4 / 8 / 16 (KV block 16 / 32 / 64); a page other than aiter's
    default 4 is carried as ``page_size`` (aiter ``profiles.page_of``)."""
    if sig["cache_dtype"] not in ("bf16", "fp8") or sig["page_size"] not in PAGES:
        return None
    row = {
        k: sig[k]
        for k in (
            "cache_dtype",
            "rows",
            "requests",
            "columns",
            "table_width",
            "cache_pages",
        )
    }
    if sig["page_size"] != 4:
        row["page_size"] = sig["page_size"]
    return row


def golden_case_of(sig: dict) -> str | None:
    key = (
        sig["cache_dtype"],
        sig["rows"],
        sig["requests"],
        sig["columns"],
        sig["table_width"],
    )
    for name, cls in GOLDEN_CLASSES.items():
        if cls == key:
            return name
    return None


class ScorerCapture:
    def __init__(self, path: str):
        self.template = path
        self.rank = _rank()
        self.pid = os.getpid()
        self.path = self._resolve(path)
        self.profile = ""
        self.classes: dict[str, dict] = {}
        self._lock = threading.Lock()
        atexit.register(self.flush)

    def _resolve(self, path: str) -> str:
        if "{" in path:
            return path.format(rank=self.rank, pid=self.pid)
        root, ext = os.path.splitext(path)
        return f"{root}.rank{self.rank}.pid{self.pid}{ext or '.json'}"

    def class_entry(self, args: tuple, profile: str) -> dict:
        """The (shared) entry of the class of one call's 12 bound arguments."""
        sig = describe_call(args)
        key = json.dumps(sig, sort_keys=True)
        with self._lock:
            self.profile = profile
            ent = self.classes.get(key)
            if ent is None:
                ent = self.classes[key] = {
                    "signature": sig,
                    "count": 0,
                    "decisions": {},
                }
        return ent

    def count(self, ent: dict, decision: str) -> None:
        """One call of ``ent``'s class served as ``decision`` (``path:reason``)."""
        with self._lock:
            new = ent["count"] == 0
            ent["count"] += 1
            ent["decisions"][decision] = ent["decisions"].get(decision, 0) + 1
        if new:
            self.flush()

    def snapshot(self) -> dict:
        classes = list(self.classes.values())
        plan = []
        for c in classes:
            row = plan_row(c["signature"])
            if row is not None and row not in plan:
                plan.append(row)
        return {
            "schema": SCHEMA,
            "profile": self.profile,
            "rank": self.rank,
            "pid": self.pid,
            "classes": classes,
            "plan": plan,
        }

    def flush(self) -> None:
        with self._lock:
            data = self.snapshot()
        try:
            d = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(d, exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1, default=str)
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("qsa capture: cannot write %s: %r", self.path, exc)


_CAPTURES: dict[str, ScorerCapture] = {}


def get_capture(path: str) -> ScorerCapture:
    """One capture per path and process (all layers share it)."""
    cap = _CAPTURES.get(path)
    if cap is None:
        cap = _CAPTURES[path] = ScorerCapture(path)
    return cap


# ---------------------------------------------------------------------------
# merge / coverage matrix CLI
# ---------------------------------------------------------------------------
def merge(files: list[str]) -> dict:
    merged: dict[str, dict] = {}
    ranks = set()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        ranks.add(data.get("rank"))
        for c in data.get("classes", []):
            key = json.dumps(c["signature"], sort_keys=True)
            ent = merged.setdefault(
                key,
                {
                    "signature": c["signature"],
                    "count": 0,
                    "decisions": {},
                    "ranks": [],
                },
            )
            ent["count"] += c["count"]
            if data.get("rank") not in ent["ranks"]:
                ent["ranks"].append(data.get("rank"))
            for k, v in c["decisions"].items():
                ent["decisions"][k] = ent["decisions"].get(k, 0) + v
    classes = list(merged.values())
    plan, matrix = [], []
    for c in classes:
        sig = c["signature"]
        row = plan_row(sig)
        if row is not None and row not in plan:
            plan.append(row)
        matrix.append(
            {
                "class": {
                    k: sig[k]
                    for k in (
                        "cache_dtype",
                        "rows",
                        "requests",
                        "columns",
                        "table_width",
                        "page_size",
                        "regime",
                    )
                },
                "golden_case": golden_case_of(sig),
                "count": c["count"],
                "ranks": sorted(r for r in c["ranks"] if r is not None),
                "decisions": c["decisions"],
                # One class, one decision -- across ranks.
                "decision_uniform": len(c["decisions"]) == 1,
                "status": (
                    "covered"
                    if golden_case_of(sig)
                    else (
                        "uncovered:extend-golden"
                        if row is not None
                        else "uncovered:outside-kernel-domain(native)"
                    )
                ),
            }
        )
    return {
        "schema": SCHEMA + "+merged",
        "files": files,
        "ranks": sorted(r for r in ranks if r is not None),
        "classes": classes,
        "plan": plan,
        "matrix": matrix,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("files", nargs="+")
    m.add_argument("--plan", help="write the gen_profile --plan-file list here")
    m.add_argument("--matrix", help="write the merged classes + coverage matrix here")
    a = ap.parse_args(argv)
    files = sorted({f for pat in a.files for f in (glob.glob(pat) or [pat])})
    out = merge(files)
    if a.plan:
        with open(a.plan, "w", encoding="utf-8") as fh:
            json.dump(out["plan"], fh, indent=1)
    if a.matrix:
        with open(a.matrix, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=str)
    for row in out["matrix"]:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
