#!/usr/bin/env python3
"""Per-minute timeline of a local M3.1 P/D AgentX run (prefill, Store, decode).

Usage: timeline.py <run_dir> [--every MIN]

Columns per bucket:
  p_new_k/s   prefill tokens computed per second (Scheduled prefill batch, new)
  p_wait      prefill Waiting (Engine 000, mean)        p_kv%  prefill GPU KV usage
  p_hit%      prefill HBM prefix hit (Engine 000, last)
  st_get_GB   Store bytes read per bucket (OFFLOAD-STORE-STATS deltas, all ranks)
  st_put_GB   Store bytes written per bucket
  d_run       decode Running (Engine 000, mean)        d_wait decode Waiting
  d_kv%       decode GPU KV usage (mean)               d_gen  decode tokens/s
  parked      decode PD backpressure parked (mean)     resident (mean)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
from collections import defaultdict

TS = re.compile(r"^\[atom (\d\d):(\d\d):(\d\d)\]")
ENGINE = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, "
    r"Prefix cache hit rate: ([\d.]+)%"
)
SCHED = re.compile(r"Scheduled prefill batch: (\d+) reqs, (\d+) new tokens")
BP = re.compile(r"PD backpressure: parked=(\d+), waiting=(\d+), running=(\d+), resident=(\d+)/\d+, kv_usage=([\d.]+)")
STATS = re.compile(r"\[OFFLOAD-STORE-STATS\] rank=(\d+) puts=\d+ put_keys=\d+ put_bytes=(\d+) gets=\d+ get_keys=\d+ get_bytes=(\d+)")


def lines(path: str, pattern: str):
    if not os.path.exists(path):
        return []
    out = subprocess.run(
        ["grep", "-aE", pattern, path], stdout=subprocess.PIPE, check=False,
        env=dict(os.environ, LC_ALL="C"),
    ).stdout.decode("utf-8", "replace")
    return out.splitlines()


def minute(line: str, every: int):
    m = TS.match(line)
    if not m:
        return None
    h, mi, _ = (int(x) for x in m.groups())
    t = h * 60 + mi
    return t - t % every


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--every", type=int, default=5)
    a = ap.parse_args()
    job = a.run if os.path.basename(a.run.rstrip("/")).startswith("slurm_job-") else max(glob.glob(os.path.join(a.run, "slurm_job-*")))
    logs = os.path.join(job, "logs")
    pre, dec = os.path.join(logs, "prefill-rank-0.log"), os.path.join(logs, "decode-rank-0.log")
    acc: dict = defaultdict(lambda: defaultdict(list))
    for l in lines(pre, "Scheduled prefill batch"):
        b, m = minute(l, a.every), SCHED.search(l)
        if b is not None and m:
            acc[b]["p_new"].append(int(m.group(2)))
    for l in lines(pre, "Engine 000:"):
        b, m = minute(l, a.every), ENGINE.search(l)
        if b is not None and m:
            acc[b]["p_wait"].append(float(m.group(4)))
            acc[b]["p_kv"].append(float(m.group(5)))
            acc[b]["p_hit"].append(float(m.group(6)))
    for l in lines(dec, "Engine 000:"):
        b, m = minute(l, a.every), ENGINE.search(l)
        if b is not None and m:
            acc[b]["d_gen"].append(float(m.group(2)))
            acc[b]["d_run"].append(float(m.group(3)))
            acc[b]["d_wait"].append(float(m.group(4)))
            acc[b]["d_kv"].append(float(m.group(5)))
    for l in lines(dec, "PD backpressure"):
        b, m = minute(l, a.every), BP.search(l)
        if b is not None and m:
            acc[b]["parked"].append(float(m.group(1)))
            acc[b]["resident"].append(float(m.group(4)))
    last: dict = {}
    for l in lines(pre, r"\[OFFLOAD-STORE-STATS\]"):
        b, m = minute(l, a.every), STATS.search(l)
        if b is None or not m:
            continue
        rank, put_b, get_b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        p0, g0 = last.get(rank, (0, 0))
        acc[b]["st_put"].append(put_b - p0)
        acc[b]["st_get"].append(get_b - g0)
        last[rank] = (put_b, get_b)

    # NIC counters (bin/nic_sampler.sh): GB/s per bucket for rdma0-3 (the prefill
    # stages' NICs) and rdma4-7 (Store owners' and decode GPUs' NICs).
    import datetime as dt

    prev: dict = {}
    for path in sorted(glob.glob(os.path.join(os.environ.get("BASE", "."), "nic", "nic-*.csv"))):
        with open(path) as handle:
            next(handle, None)
            for row in handle:
                ep, nic, tx, rx, *_ = row.strip().split(",")
                t = dt.datetime.fromtimestamp(int(ep), dt.timezone.utc)
                b = t.hour * 60 + t.minute
                b -= b % a.every
                if b not in acc:
                    prev[nic] = (int(tx), int(rx))
                    continue
                if nic in prev:
                    group = "p" if nic in ("rdma0", "rdma1", "rdma2", "rdma3") else "d"
                    acc[b][f"{group}_tx"].append(int(tx) - prev[nic][0])
                    acc[b][f"{group}_rx"].append(int(rx) - prev[nic][1])
                prev[nic] = (int(tx), int(rx))

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    hdr = ("time", "p_new_k/s", "p_wait", "p_kv%", "p_hit%", "st_get_GB", "st_put_GB",
           "d_run", "d_wait", "d_kv%", "d_gen", "parked", "resident",
           "nicP_tx", "nicP_rx", "nicD_tx", "nicD_rx")
    print(" ".join(f"{h:>9}" for h in hdr))
    for b in sorted(acc):
        r = acc[b]
        row = (
            f"{b // 60:02d}:{b % 60:02d}",
            f"{sum(r['p_new']) / (a.every * 60) / 1000:.1f}",
            f"{avg(r['p_wait']):.1f}", f"{avg(r['p_kv']):.1f}",
            f"{r['p_hit'][-1] if r['p_hit'] else float('nan'):.1f}",
            f"{sum(r['st_get']) / 1e9:.1f}", f"{sum(r['st_put']) / 1e9:.1f}",
            f"{avg(r['d_run']):.1f}", f"{avg(r['d_wait']):.1f}", f"{avg(r['d_kv']):.1f}",
            f"{avg(r['d_gen']):.0f}", f"{avg(r['parked']):.1f}", f"{avg(r['resident']):.1f}",
            *(f"{sum(r[k]) / (a.every * 60) / 1e9:.1f}" for k in ("p_tx", "p_rx", "d_tx", "d_rx")),
        )
        print(" ".join(f"{c:>9}" for c in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
