#!/usr/bin/env python3
"""summarize.py <run dir> ...: one line per finished AgentX run -- a run.sh / up.sh log directory (/logs/<TAG> in the
container) or its agentx/ artifact directory.

Throughput and interactivity use the InferenceX per-request formulas behind the AgentX numbers on
inferencex.semianalysis.com, from ATOM's .github/scripts/atomesh/interactivity.py (found from this file's place in the
ATOM tree; ATOMESH_SCRIPTS overrides):
  tok/s/chip  (sum ISL + sum OSL of the successful profiling records) / (first start .. last end) / TP; the input and
              output parts follow (output alone is ROCm/ATOM#2747's tok/s/GPU)
  intvty      1 / p90 and 1 / p50 of the per-request ITL, in tok/s/user
TTFT and ITL percentiles and the request and error counts are aiperf's (profile_export_aiperf.json). The prefix hit is
per request, from the cached_tokens the server returns, over the same profiling requests and tokens as aiperf's
theoretical hit next to it. The server's own atom:prefix_cache_* counters are not used for it: with LMCache, a request
parked mid-prefill for a CPU load is counted again when it resumes, which inflates their denominator (by 8% at c64).
The LMCache share is the tokens loaded from the CPU tier during profiling (atom:lmcache_loaded_tokens) over those
requests' prompt tokens.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("ATOMESH_SCRIPTS", str(Path(__file__).resolve().parents[2] / ".github/scripts/atomesh")))
import interactivity as infx  # noqa: E402

TP = int(os.environ.get("TP", "4"))


def counter(metrics: dict, phase: str, name: str) -> float:
    m = (metrics.get(phase) or {}).get(name)
    return m["series"][0]["stats"]["total"] if m else 0.0


def prefix_hit(d: Path, agg: dict) -> str:
    stat = lambda name: (agg.get(name) or {}).get("avg")
    hit, cached, theo = stat("overall_usage_prompt_cache_read_pct"), stat("total_usage_prompt_cache_read_tokens"), stat("theoretical_prefix_cache_hit")
    if not hit:
        return "n/a"
    parts = []
    f = d / "server_metrics_export.json"
    if f.exists() and cached:
        s = json.loads(f.read_text())
        loaded = counter(s, "metrics", "atom:lmcache_loaded_tokens") - counter(s, "warmup_metrics", "atom:lmcache_loaded_tokens")
        if loaded > 0:
            parts.append(f"LMCache {100 * loaded / (cached * 100 / hit):.1f}%")
    if theo is not None:
        parts.append(f"theoretical {theo:.1f}%")
    return f"{hit:.1f}%" + (f" ({'; '.join(parts)})" if parts else "")


def concurrency(agg: dict) -> int | None:
    cfg = agg.get("input_config") or {}
    c = (cfg.get("loadgen") or {}).get("concurrency") or cfg.get("concurrency")
    return c or next((p.get("concurrency") for p in cfg.get("phases") or [] if p.get("kind") == "profiling"), None)


for arg in sys.argv[1:]:
    d = Path(arg)
    if (d / "agentx").is_dir():
        d = d / "agentx"
    name = d.parent.name if d.name == "agentx" else d.name
    agg = json.loads((d / "profile_export_aiperf.json").read_text())
    jsonl = d / infx.RECORDS_FILENAME
    tput = infx.agentic_throughput(jsonl)
    p90 = infx.agentic_interactivity(jsonl)
    p50 = infx.agentic_interactivity(jsonl, percentile=50.0)
    ttft, itl = agg["time_to_first_token"], agg["inter_token_latency"]
    errors = sum(e.get("count", 0) for e in agg.get("error_summary") or [])
    print(
        f"{name}: c{concurrency(agg)} | {tput['total_tput_tps'] / TP:,.0f} tok/s/chip (in {tput['input_tput_tps'] / TP:,.0f}, "
        f"out {tput['output_tput_tps'] / TP:,.1f}) | intvty P90 {p90['itl_value']:.1f}, median {p50['itl_value']:.1f} tok/s/user | "
        f"ITL p50/p90 {itl['p50']:.1f}/{itl['p90']:.1f} ms | TTFT p50/p90 {ttft['p50'] / 1e3:.2f}/{ttft['p90'] / 1e3:.2f} s | "
        f"prefix hit {prefix_hit(d, agg)} | "
        f"{agg['request_count']['avg']:,.0f} requests, {errors} failed | {tput['duration_s']:,.0f} s"
    )
