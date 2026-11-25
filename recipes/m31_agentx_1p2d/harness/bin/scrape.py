#!/usr/bin/env python3
"""scrape.py --run RUN [--interval 15]: poll every server's and Store master's /metrics.

Runs ON node A (host python3, stdlib only). Waits for the launcher's
metadata-rank-0.json (prefill / decode endpoints), then every interval writes one
JSONL line per target to RUN/scrape.jsonl:
  {"t": epoch, "target": "prefill0"|"decode<i>"|"store<pool>", "m": {series: value}}
Only the series the summary needs are kept (no histogram buckets). Exits when
RUN/STOP exists or every rank published its rc.
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

ATOM_KEEP = re.compile(
    r"^atom:("
    r"requests_(running|waiting|parked_kv_load|partial_prefill|deferred_free|finished_total)"
    r"|kv_cache_(usage_ratio|blocks_used|blocks_total|blocks_free|blocks_deferred_free)"
    r"|prompt_tokens_total|generation_tokens_total|preemptions_total"
    r"|prefix_cache_(requests|cached_tokens|full_tokens|wanted_tokens|offload_tokens)_total"
    r"|pd_admission_\w+"
    r"|scheduler_(requests|kv_cache_blocks)"
    r"|(gpu_forward_seconds|prefill_batch_tokens|decode_batch_size|pd_kv_transfer_seconds"
    r"|time_to_first_token_seconds|request_queue_time_seconds|decode_context_tokens"
    r"|prefill_request_tokens|inter_token_latency_seconds)_(sum|count)"
    r"|mtp_\w+"
    r")[{ ]"
)
STORE_KEEP = re.compile(
    r"^master_("
    r"total_capacity_bytes|allocated_bytes|key_count|active_clients"
    r"|evicted_key_count|evicted_size_bytes|successful_evictions_total|attempted_evictions_total"
    r"|put_start_(requests|failures|object_already_exists|alloc_failures)_total"
    r"|batch_put_start_(items|failed_items|object_already_exists)_total"
    r"|batch_get_replica_list_(items|failed_items|requests)_total"
    r"|get_replica_list_(requests|failures)_total"
    r"|batch_exist_key_(items|failed_items)_total|exist_key_requests_total"
    r")[{ ]"
)
LINE = re.compile(r"^(\S+?)(\{[^}]*\})?\s+(\S+)$")


def fetch(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def parse(text, keep):
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#" or not keep(line):
            continue
        m = LINE.match(line)
        if not m:
            continue
        try:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--interval", type=float, default=15.0)
    args = ap.parse_args()
    run = Path(args.run)
    info = json.loads((run / "run.json").read_text())
    job = Path(info["job_dir"])
    plan = json.loads((run / "store_plan.json").read_text())
    pools = 4 if info.get("store_pools", "per_nic") == "per_nic" else 1
    if plan.get("pools") != "per_nic":
        pools = 1
    stop = run / "STOP"
    nranks = len(info["hosts"])
    # metadata-rank-0.json, or metadata-rank-0-<phase>.json for a benchmark / eval phase
    while not (metas := sorted(job.glob("metadata-rank-0*.json"), key=lambda p: p.stat().st_mtime)):
        if stop.exists():
            return
        time.sleep(5)
    md = json.loads(metas[-1].read_text())
    # A restarted phase (eval) shifts every service port by the same offset.
    offset = int(md["prefill_ports"].split(",")[0]) - 8010
    targets = []
    for i, (ip, port) in enumerate(zip(md["prefill_ips"].split(","), md["prefill_ports"].split(","))):
        targets.append((f"prefill{i}", f"http://{ip}:{port}/metrics", ATOM_KEEP.match))
    for i, (ip, port) in enumerate(zip(md["decode_ips"].split(","), md["decode_ports"].split(","))):
        targets.append((f"decode{i}", f"http://{ip}:{port}/metrics", ATOM_KEEP.match))
    node_a = info["ips"][0]
    for p in range(pools):
        targets.append((f"store{p}", f"http://{node_a}:{26090 + offset + 100 * p}/metrics", STORE_KEEP.match))
    out = (run / "scrape.jsonl").open("a")
    misses = {}
    while not stop.exists():
        if sum((job / f"rank-rc-{r}").exists() for r in range(nranks)) == nranks:
            break
        t0 = time.time()
        for name, url, keep in targets:
            try:
                m = parse(fetch(url), keep)
            except Exception as exc:  # noqa: BLE001 - a target that is down is just skipped
                misses[name] = misses.get(name, 0) + 1
                if misses[name] in (1, 10, 100):
                    print(f"{time.strftime('%H:%M:%S')} {name}: {exc!r} (miss {misses[name]})", flush=True)
                continue
            out.write(json.dumps({"t": round(t0, 1), "target": name, "m": m}) + "\n")
        out.flush()
        time.sleep(max(1.0, args.interval - (time.time() - t0)))
    out.close()


if __name__ == "__main__":
    main()
