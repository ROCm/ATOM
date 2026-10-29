#!/usr/bin/env python3
"""Headline numbers of MLPerf Endpoints points, side by side.
usage: bench_summary.py <name>=<point dir ($R/runs/c<N>)> ...

From each point's report/report.txt (the MLCommons client's): turns, duration, output tok/s (the report's
TPS), total tok/s (turns x (mean input + mean output length) / duration: every prompt token, cached or not),
E2E interactivity, TTFT and TPOT median / mean / p90 / p99. From server_metrics.txt, when present: prefill
tok/s of the prompt tokens the prefix cache did not hold, and the cache's hit fraction. MTP tokens a
forward from the server log. The last columns are each point against the first.
"""

import glob
import re
import sys
from pathlib import Path

STATS = (("Median", "median"), ("Avg.", "avg"), ("90.0", "p90"), ("99.0", "p99"))


def read(path):
    return Path(path).read_text()


def stats(text, name):
    # the section is "<name>:" followed directly by "  Min:", then its Percentiles block
    m = re.search(rf"^{re.escape(name)}:\n  Min: ", text, re.MULTILINE)
    block = text[m.start() :]
    if "Early-stopping" in block:
        block = block[: block.index("Early-stopping")]
    out = {}
    for key, label in STATS:
        v = re.search(rf"^\s+{re.escape(key)}: ([\d.]+)", block, re.MULTILINE)
        out[label] = float(v.group(1))
    return out


def point(d):
    t = read(f"{d}/report/report.txt")

    def g(p):
        return float(re.search(p, t, re.MULTILINE).group(1))

    turns, dur = g(r"^Total samples succeeded: (\d+)"), g(r"^Duration: ([\d.]+)")
    isl = stats(t, "Input sequence lengths")["avg"]
    osl = stats(t, "Output sequence lengths")["avg"]
    r = {
        "turns": turns,
        "failed": g(r"^Total samples failed: (\d+)"),
        "duration s": dur,
        "output tok/s": g(r"^TPS: ([\d.]+)"),
        "total tok/s": turns * (isl + osl) / dur,
        "interactivity": g(r"^E2E average interactivity: ([\d.]+)"),
    }
    try:
        mt = read(f"{d}/server_metrics.txt")

        def mv(k):
            return float(re.search(rf"^atom:{k} ([\d.e+]+)", mt, re.MULTILINE).group(1))

        prompt, cached = mv("prompt_tokens_total"), mv(
            "prefix_cache_cached_tokens_total"
        )
        r["uncached prefill tok/s"] = (prompt - cached) / dur
        r["prefix cache hit %"] = 100 * cached / prompt
    except (OSError, AttributeError):
        pass
    for name in ("TTFT", "TPOT"):
        for label, v in stats(t, name).items():
            r[f"{name} {label} ms"] = v
    r["mean ISL"], r["mean OSL"] = isl, osl
    logs = sorted(glob.glob(f"{d}/server.log") + glob.glob(f"{d}/server[0-9]*.log"))
    if logs:
        m = re.findall(r"\[MTP Stats\s+\] Average toks/fwd: ([\d.]+)", read(logs[0]))
        if m:
            r["MTP toks/fwd"] = float(m[-1])
    return r


def fmt(k, v):
    if v is None:
        return "-"
    if abs(v) >= 1000 or k in ("turns", "failed"):
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def main():
    pts = [(a.split("=", 1)[0], point(a.split("=", 1)[1])) for a in sys.argv[1:]]
    keys = list(pts[0][1])
    for _, p in pts[1:]:
        keys += [k for k in p if k not in keys]
    head = "".join(f"{n:>14s}" for n, _ in pts)
    head += "".join(f"{'vs ' + pts[0][0]:>14s}" for _ in pts[1:])
    print(f"{'':24s}{head}")
    for k in keys:
        vals = [p.get(k) for _, p in pts]
        row = f"{k:24s}" + "".join(f"{fmt(k, v):>14s}" for v in vals)
        for v in vals[1:]:
            rel = (
                f"{100 * (v / vals[0] - 1):+.1f}%" if v is not None and vals[0] else "-"
            )
            row += f"{rel:>14s}"
        print(row)


if __name__ == "__main__":
    main()
