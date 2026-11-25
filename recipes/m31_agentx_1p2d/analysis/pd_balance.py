#!/usr/bin/env python3
"""pd_balance.py RUN [--bucket S]: P:D balance and TTFT components of one harness point -> RUN/pd_balance.json (+ printed).

Window = the aiperf profiling phase. From scrape.jsonl (15-s samples) and the server logs:
  per decode: finished req/s, gen tok/s, running / waiting / admitted-pending (pd_admission_pending_requests), credit wait
    queue (waiting - pending), batch mean, ITL mean, KV usage, credit occupancy (pending bytes / cap), deferrals by reason,
    decode queue time (receipt -> first forward: credit wait + KV wait), pd_kv_transfer (admitted -> KV landed),
    decode-held prompt share, Little's-law seconds per request for each queue (credit wait, KV wait, running);
  prefill: prompt / computed tok/s, hit HBM / Store, waiting / parked on Store loads / running, prefill queue time
    (receipt -> first forward, incl. Store-load parking), prefill TTFT (receipt -> prefill done), held KV share
    (deferred_free blocks), batch tokens x batches/s, Store loads (count, GB, mean ms, loads_pending queue);
  time series: per --bucket seconds (default 300) the same headline numbers.
"""

import argparse
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "bin"))
from summarize import day_epoch, profiling_window

ADM = re.compile(r"PD admission enabled: pending_kv_bytes=(\d+) per D rank")
TS = re.compile(r"^\[atom (\d\d):(\d\d):(\d\d)\]")
LOAD = re.compile(r"\[OFFLOAD-LOAD-PROF\] rank=(\d+) .*?status=(\w+).*?total_bytes=(\d+) get_ms=([\d.]+) unpack_ms=([\d.]+) "
                  r"retrieve_ms=([\d.]+) total_ms=([\d.]+)")
PERF_PENDING = re.compile(r'"loads_pending": (\d+)')


def tot(m, n):
    return sum(v for k, v in m.items() if k == n or k.startswith(n + "{"))


def q(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))] if s else None


def st(xs, d=2):
    return {"mean": round(statistics.fmean(xs), d), "p50": round(q(xs, 0.5), d), "p90": round(q(xs, 0.9), d),
            "max": round(max(xs), d)} if xs else None


def delta(rs, name):
    return tot(rs[-1][1], name) - tot(rs[0][1], name)


def hist_mean(rs, name):
    c = delta(rs, name + "_count")
    return round(delta(rs, name + "_sum") / c, 6) if c > 0 else None


def rnd(x, d):
    return round(x, d) if x is not None else None


def decode_block(rs, cap):
    dt = rs[-1][0] - rs[0][0]
    fin = delta(rs, "atom:requests_finished_total")
    lam = fin / dt if dt > 0 else 0
    running = [tot(m, "atom:requests_running") for _, m in rs]
    waiting = [tot(m, "atom:requests_waiting") for _, m in rs]
    pending = [tot(m, "atom:pd_admission_pending_requests") for _, m in rs]
    credit_q = [max(0.0, w - p) for w, p in zip(waiting, pending)]
    pend_b = [tot(m, "atom:pd_admission_pending_kv_bytes") for _, m in rs]
    dfr = {}
    for k, v in rs[-1][1].items():
        if k.startswith("atom:pd_admission_deferred_total{"):
            reason = re.search(r'reason="([^"]+)"', k).group(1)
            dfr[reason] = round(v - rs[0][1].get(k, 0))
    full = delta(rs, "atom:prefix_cache_full_tokens_total")
    cached = delta(rs, "atom:prefix_cache_cached_tokens_total")
    bc = delta(rs, "atom:decode_batch_size_count")
    out = {
        "finished_req_s": round(lam, 3), "gen_tok_s": round(delta(rs, "atom:generation_tokens_total") / dt),
        "running": st(running, 1), "waiting": st(waiting, 1), "admitted_pending": st(pending, 1),
        "credit_wait_queue": st(credit_q, 1),
        "batch_mean": round(delta(rs, "atom:decode_batch_size_sum") / bc, 1) if bc else None,
        "itl_ms_mean": round(1000 * hist_mean(rs, "atom:inter_token_latency_seconds"), 2)
        if hist_mean(rs, "atom:inter_token_latency_seconds") else None,
        "kv_usage": st([tot(m, "atom:kv_cache_usage_ratio") for _, m in rs], 3),
        "credit_cap_gb_per_rank": round(cap / 1e9, 2) if cap else None,
        "pending_kv_gb_per_rank": st([b / 1e9 for b in pend_b], 2),
        "credit_occupancy": st([b / cap for b in pend_b], 3) if cap else None,
        "share_samples_credit_ge_0.85": round(sum(b / cap >= 0.85 for b in pend_b) / len(pend_b), 3) if cap else None,
        "deferred": dfr, "preemptions": round(delta(rs, "atom:preemptions_total")),
        "queue_time_s_mean": rnd(hist_mean(rs, "atom:request_queue_time_seconds"), 3),
        "pd_kv_transfer_s_mean": rnd(hist_mean(rs, "atom:pd_kv_transfer_seconds"), 3),
        "ttft_engine_s_mean": rnd(hist_mean(rs, "atom:time_to_first_token_seconds"), 3),
        "decode_held_prompt": round(cached / full, 3) if full else None,
        "prompt_tok_per_req": round(delta(rs, "atom:prompt_tokens_total") / fin) if fin else None,
        "ctx_tokens_mean": round(delta(rs, "atom:decode_context_tokens_sum") / delta(rs, "atom:decode_context_tokens_count"))
        if delta(rs, "atom:decode_context_tokens_count") else None,
        "gpu_forward_busy": round(delta(rs, "atom:gpu_forward_seconds_sum") / 4 / dt, 3),
    }
    if lam > 0:
        out["little_s_per_req"] = {"credit_wait": round(statistics.fmean(credit_q) / lam, 2),
                                   "kv_wait_admitted": round(statistics.fmean(pending) / lam, 2),
                                   "running": round(statistics.fmean(running) / lam, 2)}
    return out


def prefill_block(rs):
    dt = rs[-1][0] - rs[0][0]
    full = delta(rs, "atom:prefix_cache_full_tokens_total")
    cached = delta(rs, "atom:prefix_cache_cached_tokens_total")
    off = delta(rs, "atom:prefix_cache_offload_tokens_total")
    held = [tot(m, "atom:kv_cache_blocks_deferred_free") / max(tot(m, "atom:kv_cache_blocks_total"), 1) for _, m in rs]
    bc = delta(rs, "atom:prefill_batch_tokens_count")
    return {
        "prompt_tok_s": round(full / dt), "computed_tok_s": round((full - cached - off) / dt),
        "hit_hbm": round(cached / full, 4) if full else None, "hit_store": round(off / full, 4) if full else None,
        "finished_req_s": round(delta(rs, "atom:requests_finished_total") / dt, 3),
        "waiting": st([tot(m, "atom:requests_waiting") for _, m in rs], 1),
        "parked_store_loads": st([tot(m, "atom:requests_parked_kv_load") for _, m in rs], 1),
        "running": st([tot(m, "atom:requests_running") for _, m in rs], 1),
        "deferred_free_requests": st([tot(m, "atom:requests_deferred_free") for _, m in rs], 1),
        "held_kv_share": st(held, 3), "kv_usage": st([tot(m, "atom:kv_cache_usage_ratio") for _, m in rs], 3),
        "queue_time_s_mean": rnd(hist_mean(rs, "atom:request_queue_time_seconds"), 3),
        "ttft_engine_s_mean": rnd(hist_mean(rs, "atom:time_to_first_token_seconds"), 3),
        "batch_tokens_mean": round(delta(rs, "atom:prefill_batch_tokens_sum") / bc) if bc else None,
        "batches_s": round(bc / dt, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--bucket", type=int, default=300)
    ap.add_argument("--from-s", type=float, default=0.0, help="window start, s after the profiling start")
    ap.add_argument("--to-s", type=float, default=None, help="window end, s after the profiling start")
    ap.add_argument("--out", default="pd_balance.json")
    a = ap.parse_args()
    run = Path(a.run)
    info = json.loads((run / "run.json").read_text())
    job = Path(info["job_dir"])
    t0, t1, _ = profiling_window(Path(glob.glob(str(job / "benchmark_results/aiperf-*/aiperf.log"))[0]))
    t0, t1 = t0 + a.from_s, min(t1, t0 + a.to_s) if a.to_s is not None else t1
    rows = defaultdict(list)
    with open(run / "scrape.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if t0 <= r["t"] <= t1:
                rows[r["target"]].append((r["t"], r["m"]))
    caps = []
    for p in sorted(glob.glob(str(job / "logs/decode-rank-*.log"))):
        cap = None
        with open(p, errors="replace") as f:
            for line in f:
                m = ADM.search(line)
                if m:
                    cap = int(m.group(1))
                    break
        caps.append(cap)
    decs = sorted(k for k in rows if k.startswith("decode"))
    out = {"run": str(run), "window": [t0, t1, round(t1 - t0)], "decode": {}, "prefill": {}}
    for i, k in enumerate(decs):
        out["decode"][k] = decode_block(rows[k], caps[i] if i < len(caps) else None)
    if rows.get("prefill0"):
        out["prefill"]["prefill0"] = prefill_block(rows["prefill0"])
    # Store loads and the load queue from the prefill log.
    plog = sorted(glob.glob(str(job / "logs/prefill-rank-*.log")))
    if plog:
        ref = Path(plog[0]).stat().st_mtime
        pend, ms, gb, n = [], [], 0.0, 0
        with open(plog[0], errors="replace") as f:
            for line in f:
                m = TS.match(line)
                if not m:
                    continue
                t = day_epoch(ref, m.group(1, 2, 3))
                if not t0 <= t <= t1:
                    continue
                if "[OFFLOAD-PERF]" in line:
                    x = PERF_PENDING.search(line)
                    if x:
                        pend.append(int(x.group(1)))
                elif "[OFFLOAD-LOAD-PROF]" in line:
                    x = LOAD.search(line)
                    if x:
                        n += 1
                        gb += int(x.group(3)) / 1e9
                        ms.append(float(x.group(7)))
        out["prefill"]["store_loads"] = {"loads": n, "loads_s": round(n / (t1 - t0), 2), "gb": round(gb, 1),
                                         "gb_s": round(gb / (t1 - t0), 2), "total_ms": st(ms, 1),
                                         "loads_pending": st(pend, 1)}
    # Time series.
    series = []
    b = t0
    while b < t1:
        e = min(t1, b + a.bucket)
        row = {"from": __import__("time").strftime("%H:%M", __import__("time").gmtime(b))}
        pre = [x for x in rows.get("prefill0", []) if b <= x[0] <= e]
        if len(pre) >= 2:
            pb = prefill_block(pre)
            row.update({"prompt_k": round(pb["prompt_tok_s"] / 1e3), "computed_k": round(pb["computed_tok_s"] / 1e3),
                        "hit": [pb["hit_hbm"], pb["hit_store"]], "p_wait": pb["waiting"]["mean"],
                        "p_parked": pb["parked_store_loads"]["mean"]})
        for i, k in enumerate(decs):
            ds = [x for x in rows[k] if b <= x[0] <= e]
            if len(ds) >= 2:
                db = decode_block(ds, caps[i] if i < len(caps) else None)
                row[k] = {"fin_s": db["finished_req_s"], "run": db["running"]["mean"], "batch": db["batch_mean"],
                          "itl": db["itl_ms_mean"], "credq": db["credit_wait_queue"]["mean"],
                          "pend": db["admitted_pending"]["mean"],
                          "occ": db["credit_occupancy"]["mean"] if db["credit_occupancy"] else None,
                          "kv": db["kv_usage"]["mean"]}
        series.append(row)
        b = e
    out["series"] = series
    (run / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
