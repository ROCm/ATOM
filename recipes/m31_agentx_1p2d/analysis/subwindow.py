#!/usr/bin/env python3
"""subwindow.py RUN FROM_S TO_S [GPUS]: CI-like numbers for profiling requests that STARTED in [t0+FROM_S, t0+TO_S)
(t0 = first profiling start): tokens (ISL+OSL) over (last end - first start) / GPUs, req/s/GPU, P90 TPOT
((latency - TTFT) / (OSL - 1)), TTFT p50/p90, mean ISL."""
import glob
import json
import statistics
import sys
from pathlib import Path

run, a, b = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
gpus = int(sys.argv[4]) if len(sys.argv) > 4 else 12
info = json.loads((run / "run.json").read_text())
path = glob.glob(info["job_dir"] + "/benchmark_results/aiperf-*/profile_export.jsonl")[0]
recs = []
with open(path) as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        md = r.get("metadata") or {}
        if md.get("benchmark_phase") != "profiling" or r.get("error"):
            continue
        m = r.get("metrics") or {}
        v = lambda k: (m.get(k) or {}).get("value")
        recs.append((md["request_start_ns"] / 1e9, md["request_end_ns"] / 1e9, v("input_sequence_length") or 0,
                     v("output_sequence_length") or 0, (v("time_to_first_token") or 0) / 1e3, (v("request_latency") or 0) / 1e3))
t0 = min(r[0] for r in recs)
sel = [r for r in recs if t0 + a <= r[0] < t0 + b]
s0, s1 = min(r[0] for r in sel), max(r[1] for r in sel)
tok = sum(r[2] + r[3] for r in sel)
tpot = sorted((r[5] - r[4]) / (r[3] - 1) * 1000 for r in sel if r[3] > 1)
ttft = sorted(r[4] for r in sel)
q = lambda xs, p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
print(json.dumps({"from_s": a, "to_s": b, "requests": len(sel), "span_s": round(s1 - s0, 1),
                  "tok_s_per_gpu": round(tok / (s1 - s0) / gpus), "req_s_per_gpu": round(len(sel) / (s1 - s0) / gpus, 4),
                  "p90_tpot_ms": round(q(tpot, 0.9), 2), "median_tpot_ms": round(q(tpot, 0.5), 2),
                  "ttft_p50_s": round(q(ttft, 0.5), 2), "ttft_p90_s": round(q(ttft, 0.9), 2),
                  "mean_isl": round(statistics.fmean(r[2] for r in sel)), "mean_e2e_s": round(statistics.fmean(r[5] for r in sel), 2),
                  "tail_free_tok_s_per_gpu": round(sum(r[2] + r[3] for r in sel if r[1] < t0 + b) / (b - a) / gpus),
                  "tail_free_req_s_per_gpu": round(sum(1 for r in sel if r[1] < t0 + b) / (b - a) / gpus, 4)}))
