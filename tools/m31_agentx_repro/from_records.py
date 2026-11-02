#!/usr/bin/env python3
"""Recompute the headline AgentX metrics from profile_export.jsonl records.

For runs whose profile_export_aiperf.json was not written. Uses the
profiling-phase records that were not cancelled; duration is first request
start to last request end of those records.

Usage: from_records.py <run_dir> [...]
"""

from __future__ import annotations

import glob
import json
import os
import sys


def pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(run: str) -> dict:
    [path] = glob.glob(os.path.join(run, "slurm_job-*", "benchmark_results", "aiperf-*", "profile_export.jsonl"))
    recs = []
    with open(path) as handle:
        for line in handle:
            d = json.loads(line)
            m = d["metadata"]
            if m.get("benchmark_phase") != "profiling" or m.get("was_cancelled"):
                continue
            recs.append(d)
    start = min(r["metadata"]["request_start_ns"] for r in recs)
    end = max(r["metadata"]["request_end_ns"] for r in recs)
    dur = (end - start) / 1e9
    # AIPerf divides throughput by the profiling phase's elapsed time
    # (duration + grace), which its log reports; use it when present.
    import re
    [log] = glob.glob(os.path.join(os.path.dirname(path), "aiperf.log"))
    with open(log, errors="replace") as handle:
        m = re.findall(r"Phase profiling \(profiling\) complete .*elapsed=([\d.]+)s", handle.read())
    if m:
        dur = float(m[-1])

    def vals(key):
        return [r["metrics"][key]["value"] for r in recs if key in r["metrics"]]

    prompt, out, cached = sum(vals("usage_prompt_tokens")), sum(vals("usage_completion_tokens")), sum(vals("usage_prompt_cache_read_tokens"))
    ttft, itl = vals("time_to_first_token"), vals("inter_token_latency")
    errors = sum(1 for r in recs if r.get("error"))
    return {
        "requests": len(recs), "errors": errors, "duration_s": round(dur, 1),
        "req_s": round(len(recs) / dur, 3),
        "total_tok_s_per_gpu": round((prompt + out) / dur / 8),
        "out_tok_s_per_gpu": round(out / dur / 8),
        "ttft_p50_ms": round(pct(ttft, 50)), "ttft_p90_ms": round(pct(ttft, 90)), "ttft_p99_ms": round(pct(ttft, 99)),
        "itl_p50_ms": round(pct(itl, 50), 2), "itl_p90_ms": round(pct(itl, 90), 2),
        "isl_avg": round(prompt / len(recs)), "osl_avg": round(out / len(recs)),
        "cache_read_pct": round(100 * cached / prompt, 2),
    }


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(os.path.basename(run.rstrip("/")), json.dumps(summarize(run)))
