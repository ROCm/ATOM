#!/usr/bin/env python3
"""Server-side stats of minutes [a, b) after the benchmark starts, from a run_ep.sh run dir.

From the engines' periodic stats lines: generation and prompt tok/s (all engines), running and
waiting requests per engine, tok/s per running sequence, est. interactivity = generation /
(engines x (running + waiting)), and the MTP acceptance over the window. The est. interactivity
counts queued requests, so it understates the MLPerf tps_per_user (tpot_p90.py) once they queue.

usage: window_stats.py <run dir> [a=2] [b=8] [engines=4]
"""

import glob
import re
import statistics
import sys

ENGINE = re.compile(
    r"\[atom (\d\d):(\d\d):(\d\d)\] Engine (\d+): Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs"
)
MTP = re.compile(
    r"\[atom (\d\d):(\d\d):(\d\d)\] \[MTP Stats\s+\].*Accepted/Total Draft tokens: (\d+)/(\d+)"
)
BENCH = re.compile(r"BENCH c=\d+ \S+T(\d\d):(\d\d):(\d\d)Z")


def seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def main():
    run = sys.argv[1].rstrip("/")
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 2
    b = float(sys.argv[3]) if len(sys.argv) > 3 else 8
    engines = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    with open(f"{run}/status.txt") as f:
        t0 = seconds(*BENCH.search(f.read()).groups())
    rows, mtp = [], []
    with open(glob.glob(f"{run}/c*/server.log")[0], errors="replace") as f:
        for line in f:
            m = ENGINE.search(line) or MTP.search(line)
            if not m:
                continue
            t = seconds(m[1], m[2], m[3]) - t0
            t += 86400 if t < -3600 else 0
            if not a * 60 <= t < b * 60:
                continue
            if len(m.groups()) == 8:
                rows.append((float(m[5]), float(m[6]), int(m[7]), int(m[8])))
            else:
                mtp.append((int(m[4]), int(m[5])))
    reports = len(rows) / engines
    prompt = sum(r[0] for r in rows) / reports
    gen = sum(r[1] for r in rows) / reports
    running = statistics.mean(r[2] for r in rows)
    waiting = statistics.mean(r[3] for r in rows)
    # The MTP counters are cumulative and interleave over the engines: an engine's counter is the
    # one whose total the new line continues most closely.
    first, last = {}, {}
    for accepted, total in mtp:
        candidates = [e for e in last if total >= last[e][1]]
        if len(last) < engines or not candidates:
            e = len(last)
        else:
            e = min(candidates, key=lambda e: total - last[e][1])
        first.setdefault(e, (accepted, total))
        last[e] = (accepted, total)
    acc = sum(last[e][0] - first[e][0] for e in last)
    drafts = sum(last[e][1] - first[e][1] for e in last)
    print(
        f"{run.split('/')[-1]:16s} min {a:g}-{b:g}: gen {gen:8,.0f} tok/s  prompt {prompt:9,.0f}  "
        f"running/eng {running:5.1f}  waiting/eng {waiting:5.1f}  "
        f"tok/s per running seq {gen / (engines * running):5.1f}  "
        f"est. interactivity {gen / (engines * (running + waiting)):5.1f}  "
        f"MTP accept {100 * acc / max(drafts, 1):4.1f}%"
    )


if __name__ == "__main__":
    main()
