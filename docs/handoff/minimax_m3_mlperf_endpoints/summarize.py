#!/usr/bin/env python3
"""summarize.py [--check] <results dir> ...: one line per finished MLPerf Endpoints run.

tok/s/GPU is system_tps over NUM_GPUS (config.env, 4 for TP4 on one node); tok/s/user is e2e_avg_interactivity.
OSL is the full-run mean; inline is the inline accuracy. Kimi-K3's 425-520 OSL and 58.32% inline rules are that
model's; for MiniMax-M3 they are reported, not checked. --check exits 1 unless every run is complete and has no
failed turns.
"""
import json
import os
import pathlib
import sys

GPUS = int(os.environ.get("NUM_GPUS", "4"))
check = "--check" in sys.argv
ok = True
for d in (a for a in sys.argv[1:] if a != "--check"):
    p = pathlib.Path(d)
    s = json.load(open(p / "performance/result_summary.json"))
    acc = p / "accuracy/accuracy_results.json"
    inline = None
    if acc.exists():
        scores = json.load(open(acc))["accuracy_scores"]
        inline = next((a["score"] for a in scores if a.get("dataset_type") == "performance"), None)
    osl = ((s.get("output_sequence_lengths_full_run") or {}).get("output_sequence_lengths") or {}).get("avg")
    sec = lambda m, q: s[m]["percentiles"][q] / 1e9
    c = s["run_config"]["load_pattern"]["target_concurrency"]
    print(
        f"{p.name}: c{c} | {s['tps'] / GPUS:,.0f} tok/s/GPU ({s['tps']:,.0f} system) | "
        f"{s['e2e_avg_interactivity']:.1f} tok/s/user | TTFT p50/p90 {sec('ttft', '50.0'):.2f}/{sec('ttft', '90.0'):.2f} s | "
        f"TPOT p50 {1000 * sec('tpot', '50.0'):.1f} ms | {s['n_samples_issued']:,} turns, {s['n_samples_failed']} failed | "
        f"window {s['duration_ns'] / 1e9:,.0f} s | OSL {'n/a' if osl is None else f'{osl:.1f}'} | "
        f"inline {'n/a' if inline is None else f'{100 * inline:.2f}%'}"
    )
    ok = ok and bool(s["complete"]) and s["n_samples_failed"] == 0
sys.exit(0 if ok or not check else 1)
