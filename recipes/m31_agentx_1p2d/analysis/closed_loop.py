#!/usr/bin/env python3
"""closed_loop.py --run RUN | --jsonl F --gpus N --conc C: the AgentX closed loop of one point.

aiperf agentic_replay keeps C lanes (session trees: root + subagents); a lane sends a turn, waits for it, waits the
trace's recorded delay_ms, sends the next turn. So   req/s = C / cycle,   cycle = C / req/s   and
  tok/s/GPU = (C / GPUs) x (mean ISL + mean OSL) / cycle
Splits (profiling records, errors dropped, CI span first start .. last end):
  N       = sum(latency) / span        mean requests in flight (Little)
  L_busy  = mean number of lanes (root trees) with >= 1 request in flight
  E2E     = N / req/s;  gap_ifx = (C - N) / req/s   (computable for InferenceX rows from their mean E2E too)
  lane_idle_per_req = (C - L_busy) / req/s          (the replayed think time per request, lane view)
  stream_gap_mean   = mean of start(turn n+1) - end(turn n) inside one session stream (x_correlation_id)
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path


def merged_len(iv, lo, hi):
    tot = 0.0
    cur_s = cur_e = None
    for s, e in sorted(iv):
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def analyse(jsonl, gpus, conc):
    recs = []
    with open(jsonl) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            md = r.get("metadata") or {}
            if md.get("benchmark_phase") not in (None, "profiling") or r.get("error"):
                continue
            m = r.get("metrics") or {}

            def v(k, m=m):
                x = m.get(k)
                return x.get("value") if isinstance(x, dict) else None

            s, e = md.get("request_start_ns"), md.get("request_end_ns")
            if not (s and e):
                continue
            recs.append({
                "s": s / 1e9, "e": e / 1e9, "isl": v("input_sequence_length") or 0, "osl": v("output_sequence_length") or 0,
                "ttft": (v("time_to_first_token") or 0) / 1e3, "lat": (v("request_latency") or 0) / 1e3,
                "corr": md.get("x_correlation_id"), "root": md.get("root_correlation_id") or md.get("x_correlation_id"),
                "turn": md.get("turn_index"), "depth": md.get("agent_depth") or 0,
            })
    n = len(recs)
    t0 = min(r["s"] for r in recs)
    t1 = max(r["e"] for r in recs)
    span = t1 - t0
    X = n / span
    isl = statistics.fmean(r["isl"] for r in recs)
    osl = statistics.fmean(r["osl"] for r in recs)
    lat = statistics.fmean(r["lat"] for r in recs)
    ttft = statistics.fmean(r["ttft"] for r in recs)
    N = sum(r["lat"] for r in recs) / span
    trees = defaultdict(list)
    for r in recs:
        trees[r["root"]].append((r["s"], r["e"]))
    L_busy = sum(merged_len(iv, t0, t1) for iv in trees.values()) / span
    streams = defaultdict(list)
    for r in recs:
        streams[r["corr"]].append(r)
    gaps = []
    for lst in streams.values():
        lst.sort(key=lambda r: (r["turn"] if r["turn"] is not None else 0))
        for a, b in pairwise(lst):
            if a["turn"] is not None and b["turn"] == a["turn"] + 1:
                gaps.append(b["s"] - a["e"])
    out = {
        "completed": n, "span_s": round(span, 1), "gpus": gpus, "conc": conc,
        "req_s": round(X, 3), "req_s_per_gpu": round(X / gpus, 4),
        "tok_s_per_gpu": round(n * (isl + osl) / span / gpus, 1),
        "mean_isl": round(isl), "mean_osl": round(osl, 1),
        "mean_e2e_s": round(lat, 2), "mean_ttft_s": round(ttft, 2), "mean_decode_s": round(lat - ttft, 2),
        "inflight_N": round(N, 1), "lanes_busy": round(L_busy, 1), "subreq_per_busy_lane": round(N / L_busy, 2),
        "cycle_s": round(conc / X, 2), "gap_ifx_s": round((conc - N) / X, 2),
        "lane_idle_per_req_s": round((conc - L_busy) / X, 2),
        "stream_gap_mean_s": round(statistics.fmean(gaps), 2) if gaps else None, "stream_gaps": len(gaps),
        "lanes_per_gpu": round(conc / gpus, 2),
        "ceiling_tok_s_per_gpu_e2e0": round(conc / gpus * (isl + osl) / ((conc - N) / X), 0) if conc > N else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--jsonl")
    ap.add_argument("--gpus", type=int)
    ap.add_argument("--conc", type=int)
    a = ap.parse_args()
    if a.run:
        run = Path(a.run)
        info = json.loads((run / "run.json").read_text())
        point = json.loads((run / "point.json").read_text()) if (run / "point.json").exists() else {}
        jsonl = glob.glob(str(Path(info["job_dir"]) / "benchmark_results/aiperf-*/profile_export.jsonl"))[0]
        gpus = a.gpus or (point.get("ci") or {}).get("total_gpu")
        conc = a.conc or info.get("conc")
    else:
        jsonl, gpus, conc = a.jsonl, a.gpus, a.conc
    out = analyse(jsonl, gpus, conc)
    out["source"] = str(a.run or a.jsonl)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
