#!/usr/bin/env python3
"""watchdog.py --run RUN: one check of a running point; exit 1 (printing why) when it must stop.

Called by agx_run.sh every 30 s; its state lives in RUN/watchdog.state.json. Stops when:
  - a prefill log shows >= cq_max Mooncake "CQ completion timeout" lines (a stalled NIC
    context never recovers and wedges the prefill; the QP-hang pattern of LLM-Booster#606);
  - a rank published a non-zero rc while another still runs (its peers would wait forever);
  - once AgentX traffic started (warmup or profiling), no decode finished a request for
    stall_min minutes, or the prefill scheduled no batch for stall_min minutes;
  - after profiling sent everything, nothing ended within post_max_min (default 45) minutes;
  - the lease deadline (est_end of the shortest lease) is less than 10 minutes away.
"""

import argparse
import calendar
import glob
import json
import re
import sys
import time
from pathlib import Path

HTTP_OK = 'POST /v1/chat/completions HTTP/1.1" 200'
BATCH = re.compile(rb"\[atom (\d\d):(\d\d):(\d\d)\] Scheduled prefill batch")


def count_in(paths, needle, state, key):
    """Occurrences of needle in the files' complete lines, read incrementally.

    state[key] keeps, per file, the offset just past the last complete line counted
    and the count so far, so a multi-GB log is read once over the whole run.
    """
    seen = state.setdefault(key, {})
    raw = needle.encode()
    for p in paths:
        offset, count = seen.get(p, (0, 0))
        try:
            with open(p, "rb") as f:
                f.seek(offset)
                chunk = f.read()
        except OSError:
            continue
        cut = chunk.rfind(b"\n") + 1
        seen[p] = (offset + cut, count + chunk[:cut].count(raw))
    return sum(c for _, c in seen.values())


def decode_tokens(scrape):
    """Sum over decodes of the newest scraped atom:generation_tokens_total, or None."""
    try:
        with open(scrape, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - (1 << 20)))
            lines = f.read().splitlines()[1:]
    except OSError:
        return None
    newest = {}
    for raw in lines:
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if row.get("target", "").startswith("decode") and "atom:generation_tokens_total" in row["m"]:
            newest[row["target"]] = row["m"]["atom:generation_tokens_total"]
    return sum(newest.values()) if newest else None


def last_batch_age(paths):
    """Seconds since the newest 'Scheduled prefill batch' line (log clock is UTC)."""
    newest = None
    for p in paths:
        try:
            with open(p, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - (8 << 20)))
                hits = BATCH.findall(f.read())
        except OSError:
            continue
        if hits:
            h, m, s = (int(x) for x in hits[-1])
            now = time.gmtime()
            t = calendar.timegm((now.tm_year, now.tm_mon, now.tm_mday, h, m, s, 0, 0, 0))
            if t > time.time() + 300:
                t -= 86400
            newest = t if newest is None else max(newest, t)
    return None if newest is None else time.time() - newest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--post-max-min", type=float, default=45)
    args = ap.parse_args()
    run = Path(args.run)
    info = json.loads((run / "run.json").read_text())
    job = Path(info["job_dir"])
    state_path = run / "watchdog.state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    now = time.time()

    def stop(reason):
        state["stopped"] = reason
        state_path.write_text(json.dumps(state))
        print(reason)
        sys.exit(1)

    if info.get("deadline") and now > info["deadline"] - 600:
        stop("lease deadline is less than 10 minutes away")
    rcs = {}
    for r in range(len(info["hosts"])):
        f = job / f"rank-rc-{r}"
        if f.exists():
            rcs[r] = f.read_text().strip()
    if any(v not in ("0", "") for v in rcs.values()) and len(rcs) < len(info["hosts"]):
        stop(f"rank rc {rcs} while other ranks still run")

    logs = job / "logs"
    prefill_logs = sorted(glob.glob(str(logs / "**/prefill-rank-*.log"), recursive=True))
    decode_logs = sorted(glob.glob(str(logs / "**/decode-rank-*.log"), recursive=True))
    cq = count_in(prefill_logs, "CQ completion timeout", state, "cq_offsets")
    if cq >= int(info.get("cq_max", 3)):
        stop(f"{cq} Mooncake CQ completion timeouts in the prefill log")

    aiperf = glob.glob(str(job / "benchmark_results/aiperf-*/aiperf.log"))
    text = ""
    if aiperf:
        try:
            text = Path(aiperf[0]).read_text(errors="replace")
        except OSError:
            text = ""
    stall_s = 60 * float(info.get("stall_min", 25))
    if re.search(r"Phase profiling \(profiling\) (sending )?complete|Phase profiling timed out", text):
        state.setdefault("post_since", now)
        if now - state["post_since"] > 60 * args.post_max_min:
            stop(f"profiling ended {args.post_max_min:.0f} min ago and the run has not finished")
    elif "Phase warmup (warmup) started" in text or "Phase profiling (profiling) started" in text:
        # Progress = tokens the decodes generated (scraped), else requests their logs finished.
        done = decode_tokens(run / "scrape.jsonl")
        if done is None:
            done = count_in(decode_logs, HTTP_OK, state, "ok_offsets")
        if done != state.get("done"):
            state["done"], state["done_since"] = done, now
        elif now - state.get("done_since", now) > stall_s:
            stop(f"no decode completion for {stall_s / 60:.0f} min ({done} so far)")
        age = last_batch_age(prefill_logs)
        if age is not None and age > stall_s:
            stop(f"no prefill batch scheduled for {age / 60:.0f} min")
    state_path.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
