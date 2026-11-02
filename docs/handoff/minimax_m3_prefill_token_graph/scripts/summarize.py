#!/usr/bin/env python3
"""Per-chip throughput, interactivity and TTFT/TTLT for agentic aiperf artifacts.

Usage: summarize.py <artifact dir> [...]   (one dir per concurrency point)
Throughput follows the #201 baseline: ATOM interactivity.agentic_throughput / 4 chips.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/app/ATOM/.github/scripts/atomesh")
from interactivity import agentic_interactivity, agentic_throughput  # noqa: E402

CHIPS = 4


def stat(agg, key, field):
    m = agg.get(key) or {}
    return m.get(field)


print(
    "| point | total tok/s/chip | decode tok/s/chip | intvty p90 | e2e-norm intvty p90 "
    "| TTFT p50 / p90 ms | TTLT p50 / p90 s | requests |"
)
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for d in map(Path, sys.argv[1:]):
    jsonl = next(d.rglob("profile_export.jsonl"), None)
    agg_path = next(d.rglob("profile_export_aiperf.json"), None)
    if jsonl is None:
        print(f"| {d.name} | missing profile_export.jsonl |")
        continue
    tput = agentic_throughput(jsonl)
    inter = agentic_interactivity(jsonl)
    agg = json.loads(agg_path.read_text()) if agg_path else {}
    if "records" in agg and isinstance(agg["records"], dict):
        agg = agg["records"]
    ttft = [stat(agg, "time_to_first_token", f) for f in ("p50", "p90")]
    ttlt = [stat(agg, "request_latency", f) for f in ("p50", "p90")]
    fmt_ms = " / ".join(f"{v:,.0f}" if v is not None else "?" for v in ttft)
    fmt_s = " / ".join(f"{v / 1e3:,.1f}" if v is not None else "?" for v in ttlt)
    print(
        f"| {d.name} | {tput['total_tput_tps'] / CHIPS:,.0f} | {tput['output_tput_tps'] / CHIPS:,.1f} "
        f"| {inter['itl_value']:.1f} | {inter['value']:.1f} | {fmt_ms} | {fmt_s} | {inter['n_requests']} |"
    )
