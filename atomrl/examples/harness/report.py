# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Results of a run or suite: summary.json, summary.md, junit.xml, and comparisons."""

from __future__ import annotations

import json
import time
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from .runner import Outcome


def outcome_json(o: Outcome) -> dict:
    ex = o.example
    return {
        "id": ex.id,
        "title": ex.title,
        "tier": ex.tier,
        "family": ex.family,
        "variant": o.variant,
        "steps": o.steps,
        "status": o.status,
        "reason": o.reason,
        "duration_s": round(o.duration_s, 1),
        "exit_code": o.exit_code,
        "hosts": o.hosts,
        "image": o.image,
        "image_id": o.image_id,
        "atom_commit": o.atom_commit,
        "log": o.log,
        "files": o.files,
        "verdict": o.verdict.to_json() if o.verdict else None,
        "step_metrics": {
            str(k): v for k, v in (o.verdict.steps if o.verdict else {}).items()
        },
    }


def write(results_dir: Path, meta: dict, outcomes: list[Outcome]) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {"meta": meta, "results": [outcome_json(o) for o in outcomes]}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (results_dir / "summary.md").write_text(markdown(summary))
    (results_dir / "junit.xml").write_text(junit(summary))
    return summary


def counts(summary: dict) -> dict[str, int]:
    out = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0, "XFAIL": 0, "XPASS": 0}
    for r in summary["results"]:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out


def _checked(r: dict) -> str:
    v = r.get("verdict") or {}
    parts = []
    for m in v.get("metrics", []):
        if m["status"] == "INFO":
            continue
        parts.append(f"{m['name'].split('/')[-1]}={m['value']} {m['status']}")
    for k, n in (v.get("counts") or {}).items():
        if k.startswith("pytest_"):
            parts.append(f"{k[7:]}={n}")
    return ", ".join(parts)


def markdown(summary: dict) -> str:
    meta = summary["meta"]
    c = counts(summary)
    lines = [
        f"# AtomRL examples: {meta.get('label', 'run')}",
        "",
        f"- started {meta.get('started')}, {meta.get('duration_min', 0):.0f} min",
        f"- code {meta.get('atomrl_src')} @ {meta.get('atomrl_rev')}",
        f"- hosts {', '.join(meta.get('hosts', []))}",
        (
            f"- result: {c['PASS']} pass, {c['FAIL']} fail, {c['ERROR']} error, {c['SKIP']} skip, "
            f"{c['XFAIL']} known failure, {c['XPASS']} unexpected pass"
        ),
        "",
        "| example | tier | status | min | checked | reason |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["results"]:
        reason = (r["reason"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{r['id']}` | {r['tier']} | {r['status']} | {r['duration_s'] / 60:.0f} "
            f"| {_checked(r)} | {reason[:300]} |"
        )
    return "\n".join(lines) + "\n"


def junit(summary: dict) -> str:
    c = counts(summary)
    total = sum(c.values())
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<testsuite name="atomrl-examples" tests="{total}" failures="{c["FAIL"]}" '
            f'errors="{c["ERROR"]}" skipped="{c["SKIP"] + c["XFAIL"]}">'
        ),
    ]
    for r in summary["results"]:
        attrs = f'classname={quoteattr(r["tier"])} name={quoteattr(r["id"])} time="{r["duration_s"]}"'
        reason = escape(r["reason"] or "")
        body = {
            "FAIL": f'<failure message={quoteattr(r["reason"][:500])}>{reason}</failure>',
            "ERROR": f'<error message={quoteattr(r["reason"][:500])}>{reason}</error>',
            "SKIP": f'<skipped message={quoteattr(r["reason"][:500])}/>',
            "XFAIL": f'<skipped message={quoteattr(r["reason"][:500])}/>',
        }.get(r["status"], "")
        out.append(f"  <testcase {attrs}>{body}</testcase>")
    out.append("</testsuite>")
    return "\n".join(out) + "\n"


def compare(a: dict, b: dict) -> str:
    """Status and checked-metric changes from summary ``a`` to summary ``b``."""
    ra = {r["id"]: r for r in a["results"]}
    rb = {r["id"]: r for r in b["results"]}
    lines = [
        f"A: {a['meta'].get('label')} ({a['meta'].get('started')}, {a['meta'].get('atomrl_rev')})",
        f"B: {b['meta'].get('label')} ({b['meta'].get('started')}, {b['meta'].get('atomrl_rev')})",
        "",
        "| example | status A -> B | metric | A | B | change | min A -> B |",
        "|---|---|---|---|---|---|---|",
    ]
    for ex_id in [i for i in ra if i in rb] + [i for i in rb if i not in ra]:
        x, y = ra.get(ex_id), rb.get(ex_id)
        status = f"{x['status'] if x else '-'} -> {y['status'] if y else '-'}"
        mins = (
            f"{x['duration_s'] / 60:.0f} -> {y['duration_s'] / 60:.0f}"
            if x and y
            else ""
        )
        mx = {m["name"]: m for m in ((x or {}).get("verdict") or {}).get("metrics", [])}
        my = {m["name"]: m for m in ((y or {}).get("verdict") or {}).get("metrics", [])}
        names = list(dict.fromkeys([*mx, *my]))
        if not names:
            lines.append(f"| `{ex_id}` | {status} | | | | | {mins} |")
        for i, name in enumerate(names):
            va = (mx.get(name) or {}).get("value")
            vb = (my.get(name) or {}).get("value")
            change = ""
            try:
                fa, fb = float(va), float(vb)
                change = f"{(fb - fa) / abs(fa) * 100:+.1f}%" if fa else ""
            except (TypeError, ValueError):
                pass
            head = (f"`{ex_id}`", status) if i == 0 else ("", "")
            lines.append(
                f"| {head[0]} | {head[1]} | {name} | {va} | {vb} | {change} | "
                f"{mins if i == 0 else ''} |"
            )
    return "\n".join(lines) + "\n"


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")
