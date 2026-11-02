#!/usr/bin/env python3
"""Summarise local M3.1 P/D AgentX runs (run_local_m31.sh) for the C80-C140 sweep.

Usage: summarize.py <run_dir> [<run_dir> ...] [--json out.json]

<run_dir> is a LOG_ROOT (logs/<tag>-<stamp>) or its slurm_job-* directory.
Client numbers come from AIPerf's profile_export_aiperf.json. Engine numbers
come from the role logs, restricted to the AIPerf profiling window (the log's
"[atom HH:MM:SS]" clock and AIPerf's start/end times are both UTC).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import statistics
import subprocess
import sys

NUM_GPUS = 8
TS_RE = re.compile(r"^\[atom (\d\d):(\d\d):(\d\d)\]")
ENGINE_RE = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, "
    r"Prefix cache hit rate: ([\d.]+)%"
)
BACKPRESSURE_RE = re.compile(
    r"PD backpressure: parked=(\d+), waiting=(\d+), running=(\d+), resident=(\d+)/(\d+), kv_usage=([\d.]+)"
)
BUDGET_RE = re.compile(r"block_bytes=(\d+), num_kvcache_blocks=(\d+)")


def grep(path: str, pattern: str) -> list[str]:
    if not os.path.exists(path):
        return []
    out = subprocess.run(
        ["grep", "-aE", pattern, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=dict(os.environ, LC_ALL="C"),
        check=False,
    ).stdout
    return out.decode("utf-8", "replace").splitlines()


def find_job_dir(path: str) -> str:
    if os.path.basename(path.rstrip("/")).startswith("slurm_job-"):
        return path
    jobs = sorted(glob.glob(os.path.join(path, "slurm_job-*")))
    if not jobs:
        raise SystemExit(f"no slurm_job-* under {path}")
    return jobs[-1]


def in_window(line: str, start: dt.datetime | None, end: dt.datetime | None) -> bool:
    if start is None:
        return True
    m = TS_RE.match(line)
    if not m:
        return False
    h, mi, s = (int(x) for x in m.groups())
    # The log carries a clock time only; anchor it to the window's day and
    # allow a window that crosses midnight.
    t = start.replace(hour=h, minute=mi, second=s, microsecond=0)
    if t < start - dt.timedelta(hours=12):
        t += dt.timedelta(days=1)
    return start <= t <= end


def mean(xs):
    return statistics.fmean(xs) if xs else None


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def metric(d: dict, key: str, field: str = "avg"):
    v = d.get(key)
    return v.get(field) if isinstance(v, dict) else None


def summarize(run: str) -> dict:
    job = find_job_dir(run)
    res: dict = {"run": run}
    exports = sorted(glob.glob(os.path.join(job, "benchmark_results", "aiperf-*", "profile_export_aiperf.json")))
    start = end = None
    if exports:
        with open(exports[-1]) as handle:
            d = json.load(handle)
        start = dt.datetime.fromisoformat(d["start_time"])
        end = dt.datetime.fromisoformat(d["end_time"])
        res["window"] = f"{start:%H:%M:%S}-{end:%H:%M:%S}"
        dur = metric(d, "benchmark_duration")
        res["duration_s"] = dur
        res["requests"] = metric(d, "request_count")
        res["errors"] = sum(int(e.get("count", 1)) for e in d.get("error_summary") or [])
        res["req_s"] = metric(d, "request_throughput")
        res["out_tok_s"] = metric(d, "output_token_throughput")
        res["total_tok_s"] = metric(d, "total_token_throughput")
        res["in_tok_s"] = metric(d, "input_token_throughput")
        for k in ("out_tok_s", "total_tok_s"):
            if res.get(k) is not None:
                res[k + "_per_gpu"] = res[k] / NUM_GPUS
        for name, key in (("ttft", "time_to_first_token"), ("itl", "inter_token_latency"),
                          ("lat", "request_latency")):
            for f in ("avg", "p50", "p90", "p99"):
                res[f"{name}_{f}_ms"] = metric(d, key, f)
        res["isl_avg"] = metric(d, "input_sequence_length")
        res["osl_avg"] = metric(d, "output_sequence_length")
        res["cache_read_pct"] = metric(d, "overall_usage_prompt_cache_read_pct")
        res["eff_decode_conc"] = metric(d, "effective_decode_concurrency")
        res["eff_prefill_conc"] = metric(d, "effective_prefill_concurrency")
        res["tokens_in_flight_avg"] = metric(d, "tokens_in_flight")
        res["tokens_in_flight_p99"] = metric(d, "tokens_in_flight", "p99")
        if res.get("itl_p90_ms"):
            res["interactivity_p90"] = 1000.0 / res["itl_p90_ms"]
        if res.get("cache_read_pct") is not None and res.get("in_tok_s"):
            res["uncached_tok_s"] = res["in_tok_s"] * (1 - res["cache_read_pct"] / 100)
    logs = os.path.join(job, "logs")
    pre = os.path.join(logs, "prefill-rank-0.log")
    dec = os.path.join(logs, "decode-rank-0.log")

    # KV capacity: one line per rank (prefill: one per PP stage).
    for role, path in (("prefill", pre), ("decode", dec)):
        caps = [tuple(map(int, BUDGET_RE.search(l).groups())) for l in grep(path, "Memory budget") if BUDGET_RE.search(l)]
        if caps:
            res[f"{role}_block_bytes"] = sorted({c[0] for c in caps})
            res[f"{role}_blocks_min"] = min(c[1] for c in caps)
            res[f"{role}_kv_tokens_M"] = min(c[1] for c in caps) * 128 / 1e6

    for role, path in (("prefill", pre), ("decode", dec)):
        rows = [ENGINE_RE.search(l) for l in grep(path, "Engine 000:") if in_window(l, start, end)]
        rows = [tuple(float(x) for x in m.groups()) for m in rows if m]
        if not rows:
            continue
        res[f"{role}_samples"] = len(rows)
        res[f"{role}_prompt_tok_s"] = mean([r[0] for r in rows])
        res[f"{role}_gen_tok_s"] = mean([r[1] for r in rows])
        res[f"{role}_running_avg"] = mean([r[2] for r in rows])
        res[f"{role}_running_p90"] = pct([r[2] for r in rows], 90)
        res[f"{role}_waiting_avg"] = mean([r[3] for r in rows])
        res[f"{role}_waiting_max"] = max(r[3] for r in rows)
        res[f"{role}_kv_pct_avg"] = mean([r[4] for r in rows])
        res[f"{role}_kv_pct_max"] = max(r[4] for r in rows)
        res[f"{role}_hbm_hit_pct_last"] = rows[-1][5]
        res[f"{role}_busy_frac"] = sum(1 for r in rows if r[2] > 0) / len(rows)

    bp = [BACKPRESSURE_RE.search(l) for l in grep(dec, "PD backpressure") if in_window(l, start, end)]
    bp = [tuple(float(x) for x in m.groups()) for m in bp if m]
    if bp:
        res["d_parked_avg"] = mean([b[0] for b in bp])
        res["d_parked_max"] = max(b[0] for b in bp)
        res["d_resident_avg"] = mean([b[3] for b in bp])
        res["d_bp_kv_avg"] = mean([b[5] for b in bp])

    # Store: per-load profile records inside the window.
    loads = [l for l in grep(pre, r"\[OFFLOAD-LOAD-PROF\]") if in_window(l, start, end)]
    res["store_loads"] = len(loads)
    res["store_load_fail"] = sum(1 for l in loads if "status=ok" not in l)
    gb = [float(m.group(1)) for l in loads if (m := re.search(r"effective_gbps=([\d.]+)", l))]
    res["store_load_gbps_p50"] = pct(gb, 50)
    rms = [float(m.group(1)) for l in loads if (m := re.search(r"retrieve_ms=([\d.]+)", l))]
    res["store_retrieve_ms_p50"] = pct(rms, 50)
    res["store_retrieve_ms_p99"] = pct(rms, 99)
    saves = [l for l in grep(pre, r"\[OFFLOAD-SAVE-PROF\]") if in_window(l, start, end)]
    res["store_saves"] = len(saves)
    res["store_save_fail"] = sum(1 for l in saves if "status=ok" not in l)
    master = os.path.join(logs, "mooncake-master-rank-0.log")
    usage = [l for l in grep(master, "Mem Storage:")]
    if usage:
        m = re.search(r"Mem Storage: ([^|]+)\|", usage[-1])
        res["store_usage_last"] = m.group(1).strip() if m else None
        res["store_usage_pct_max"] = max(
            (float(x) for l in usage for x in re.findall(r"\(([\d.]+)%\)", l)[:1]), default=None
        )
    res["store_evictions"] = len(grep(master, "evict"))
    if start is not None:
        res.update(nic_rates(start, end))
    return res


def nic_rates(start: dt.datetime, end: dt.datetime) -> dict:
    """Mean per-NIC RDMA GB/s over the window, from bin/nic_sampler.sh CSVs."""
    t0 = start.replace(tzinfo=dt.timezone.utc).timestamp()
    t1 = end.replace(tzinfo=dt.timezone.utc).timestamp()
    first: dict = {}
    last: dict = {}
    for path in sorted(glob.glob(os.path.join(os.environ.get("BASE", "."), "nic", "nic-*.csv"))):
        with open(path) as handle:
            next(handle, None)
            for row in handle:
                ep, nic, tx, rx, retx, cnp, _ecn = row.strip().split(",")
                ep = int(ep)
                if not t0 <= ep <= t1:
                    continue
                vals = (ep, int(tx), int(rx), int(retx), int(cnp))
                first.setdefault(nic, vals)
                last[nic] = vals
    out = {}
    for nic in sorted(first):
        a, b = first[nic], last[nic]
        secs = b[0] - a[0]
        if secs <= 0:
            continue
        out[f"nic_{nic}_tx_rx_GBs"] = f"{(b[1] - a[1]) / secs / 1e9:.2f}/{(b[2] - a[2]) / secs / 1e9:.2f}"
        if b[3] - a[3]:
            out[f"nic_{nic}_retx_MBs"] = round((b[3] - a[3]) / secs / 1e6, 2)
    return out


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [summarize(r) for r in a.runs]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    w = max(len(k) for k in keys)
    for k in keys:
        print(f"{k:<{w}}  " + "  ".join(f"{fmt(r.get(k)):>22}" for r in rows))
    if a.json:
        with open(a.json, "w") as handle:
            json.dump(rows, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
