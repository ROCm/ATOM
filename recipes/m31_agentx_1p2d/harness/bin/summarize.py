#!/usr/bin/env python3
"""summarize.py --run RUN: one point's numbers -> RUN/point.json and RUN/point.txt.

- CI format: the snapshot's own process_result.py over the job dir (records-span total
  tok/s per GPU, P90 TPOT, P90 E2E-normalized interactivity, TTFT p50/p90), as CI.
- Window: the aiperf profiling phase (started .. sending complete).
- From RUN/scrape.jsonl over the window: decode running batch, KV usage, preemptions,
  pd_admission deferred / pending; prefill running / waiting / parked, HBM vs Store hit
  split, computed tok/s, per-stage GPU-forward busy; Store usage, evictions, puts / gets.
- NIC rates per node and port (RUN/nic-<node>.csv), engine-log lines, failure counters.
- GSM8K: the eval score when the point ran --mode gsm8k.
A verdict line names the likely limit (heuristic; read the numbers).
"""

from __future__ import annotations

import argparse
import calendar
import csv
import glob
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[4]
FALLBACK_SCRIPTS = Path(os.environ.get("AGX_CI_SCRIPTS", REPO / ".github/scripts/atomesh"))
PHASE = re.compile(r"^(\d\d):(\d\d):(\d\d)\.\d+ +\w+ +Phase (\w+) \((\w+)\) (started|sending complete|complete)")
ENGINE = re.compile(
    r"\[atom (\d\d:\d\d:\d\d)\] .*Engine 000: (?:step \d+: )?Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+|n/a)%?, Prefix cache hit rate: ([\d.]+|n/a)"
)
MTP = re.compile(r"\[atom (\d\d:\d\d:\d\d)\] .*\[MTP Stats Interval\] Average toks/fwd: ([\d.]+)")
LABEL = re.compile(r'(\w+)="([^"]*)"')


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def stats(xs, digits=2):
    if not xs:
        return None
    return {"mean": round(statistics.fmean(xs), digits), "p50": round(pct(xs, 0.5), digits),
            "p90": round(pct(xs, 0.9), digits), "max": round(max(xs), digits), "n": len(xs)}


def day_epoch(ref_epoch, hms):
    """HH:MM:SS (UTC) on the day of ref_epoch, wrapped to within 12 h of it."""
    d = time.gmtime(ref_epoch)
    h, m, s = (int(x) for x in hms)
    t = calendar.timegm((d.tm_year, d.tm_mon, d.tm_mday, h, m, s, 0, 0, 0))
    if t - ref_epoch > 43200:
        t -= 86400
    elif ref_epoch - t > 43200:
        t += 86400
    return t


def profiling_window(aiperf_log: Path):
    if not aiperf_log.exists():
        return None, None, {}
    ref = aiperf_log.stat().st_mtime
    marks = {}
    for line in aiperf_log.read_text(errors="replace").splitlines():
        m = PHASE.match(line)
        if m:
            key = f"{m.group(5)}_{m.group(6).replace(' ', '_')}"
            marks.setdefault(key, day_epoch(ref, m.group(1, 2, 3)))
    t0 = marks.get("profiling_started")
    t1 = marks.get("profiling_sending_complete") or (t0 + 3600 if t0 else None)
    return t0, t1, marks


def run_ci(run: Path, info: dict, job: Path) -> dict:
    scripts = Path(info["snapshot"]) / ".github/scripts/atomesh"
    if not (scripts / "process_result.py").exists():
        scripts = FALLBACK_SCRIPTS
    out_json, out_md = run / "ci-benchmark-action.json", run / "ci-summary.md"
    proc = subprocess.run(
        [sys.executable, str(scripts / "process_result.py"), str(job), "--output", str(out_json),
         "--summary", str(out_md), "--hardware", "MI355X"],
        capture_output=True, text=True, cwd=str(scripts), check=False,
    )
    (run / "ci-process.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not out_json.exists():
        return {"error": f"process_result rc={proc.returncode}"}
    point = {}
    for entry in json.loads(out_json.read_text()):
        extra = entry.get("extra", "")
        if "perf_point=" in extra:
            point = json.loads(unquote(extra.split("perf_point=", 1)[1].split(" | ")[0]))
    keys = ("tput_per_gpu", "total_tput", "output_tput", "input_tput", "total_gpu", "tpot_p90", "median_tpot_ms",
            "interactivity", "interactivity_p90_itl", "median_ttft_ms", "ttft_p90", "ttft_ms", "e2el_p90",
            "completed", "duration", "cache_hit_rate", "concurrency", "gsm8k", "accuracy_score")
    return {k: point[k] for k in keys if k in point}


def load_scrape(path: Path, t0, t1):
    """{target: [(t, metrics)]} inside [t0 - 60, t1 + 60]."""
    series = defaultdict(list)
    if not path.exists():
        return series
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t0 and not (t0 - 60 <= row["t"] <= t1 + 60):
                continue
            series[row["target"]].append((row["t"], row["m"]))
    return series


def gauge(rows, name, t0, t1, reduce=sum):
    """Per-sample value of a gauge (summed over its label sets), inside the window."""
    out = []
    for t, m in rows:
        if t0 <= t <= t1:
            vals = [v for k, v in m.items() if k == name or k.startswith(name + "{")]
            if vals:
                out.append(reduce(vals))
    return out


def delta(rows, name, t0, t1, match=None):
    """Increase of a counter (summed over label sets matching `match`) across the window, and seconds."""
    inside = [(t, m) for t, m in rows if t0 <= t <= t1]
    if len(inside) < 2:
        return None, None

    def total(m):
        return sum(v for k, v in m.items() if (k == name or k.startswith(name + "{")) and (match is None or match(k)))

    (ta, ma), (tb, mb) = inside[0], inside[-1]
    return total(mb) - total(ma), tb - ta


def by_label(rows, name, label, t0, t1):
    """Counter increase per value of one label."""
    inside = [(t, m) for t, m in rows if t0 <= t <= t1]
    if len(inside) < 2:
        return {}
    out = defaultdict(float)
    for sign, (_, m) in ((-1, inside[0]), (1, inside[-1])):
        for k, v in m.items():
            if k.startswith(name + "{"):
                labels = dict(LABEL.findall(k))
                out[labels.get(label, "?")] += sign * v
    return {k: round(v, 3) for k, v in out.items()}


def nic_rates(path: Path, t0, t1):
    rows = defaultdict(list)
    try:
        with path.open() as f:
            for r in csv.DictReader(f):
                rows[r["nic"]].append((float(r["epoch"]), int(r["tx_bytes"]), int(r["rx_bytes"]), int(r["tx_retx_bytes"])))
    except (OSError, ValueError, KeyError):
        return {}
    out = {}
    for nic, s in sorted(rows.items()):
        s.sort()
        tx, rx, retx = [], [], 0
        for (ta, a0, b0, c0), (tb, a1, b1, c1) in itertools.pairwise(s):
            if ta < t0 or tb > t1 or tb <= ta:
                continue
            tx.append((a1 - a0) / (tb - ta) / 1e9)
            rx.append((b1 - b0) / (tb - ta) / 1e9)
            retx += c1 - c0
        if tx:
            out[nic] = {"tx_mean": round(statistics.fmean(tx), 2), "tx_p95": round(pct(tx, 0.95), 2),
                        "rx_mean": round(statistics.fmean(rx), 2), "rx_p95": round(pct(rx, 0.95), 2),
                        "retx_mb": round(retx / 1e6, 1)}
    return out


def engine_stats(path: str, t0, t1):
    ref = Path(path).stat().st_mtime
    prompt, gen, running, waiting, kv, hit, mtp = [], [], [], [], [], [], []
    with open(path, errors="replace") as f:
        for line in f:
            if "Engine 000" in line:
                m = ENGINE.search(line)
                if m and t0 <= day_epoch(ref, m.group(1).split(":")) <= t1:
                    prompt.append(float(m.group(2)))
                    gen.append(float(m.group(3)))
                    running.append(int(m.group(4)))
                    waiting.append(int(m.group(5)))
                    if m.group(6) != "n/a":
                        kv.append(float(m.group(6)))
                    if m.group(7) != "n/a":
                        hit.append(float(m.group(7)))
            elif "MTP Stats Interval" in line:
                m = MTP.search(line)
                if m and t0 <= day_epoch(ref, m.group(1).split(":")) <= t1:
                    mtp.append(float(m.group(2)))
    return {"prompt_tok_s": stats(prompt, 0), "gen_tok_s": stats(gen, 0), "running": stats(running, 1),
            "waiting": stats(waiting, 1), "kv_usage_pct": stats(kv, 1), "hbm_hit_pct": stats(hit, 1),
            "toks_per_fwd": stats(mtp, 2)}


def grep_count(paths, needle):
    n = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(64 << 20), b""):
                    n += chunk.count(needle.encode())
        except OSError:
            pass
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    info = json.loads((run / "run.json").read_text())
    job = Path(info["job_dir"])
    logs = job / "logs"
    res = {"run": str(run), "tag": info["tag"], "commit": info["commit"], "image": info["image"],
           "topo": info["topo"], "hosts": info["hosts"], "conc": info["conc"], "mode": info["mode"],
           "al": info["al"], "cell": info["cell_id"]}
    rcs = {}
    for r in range(len(info["hosts"])):
        f = job / f"rank-rc-{r}"
        rcs[r] = f.read_text().strip() if f.exists() else None
    res["rank_rc"] = rcs
    stopped = run / "stopped_by_watchdog"
    res["stopped_by_watchdog"] = stopped.read_text().strip() if stopped.exists() else None
    try:
        plan = json.loads((run / "store_plan.json").read_text())
        res["store_plan_gib"] = plan["total_gib"]
    except (OSError, KeyError, json.JSONDecodeError):
        pass

    if info["mode"] == "gsm8k":
        scores = []
        for p in glob.glob(str(job / "eval_results/**/*.json"), recursive=True):
            try:
                d = json.loads(Path(p).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for task, metrics in (d.get("results") or {}).items():
                for k, v in metrics.items():
                    if k.startswith("exact_match,") and isinstance(v, (int, float)):
                        scores.append({"file": Path(p).name, "task": task, "metric": k, "value": v})
        res["gsm8k"] = scores
        res["ci"] = run_ci(run, info, job)
    aiperf_dirs = glob.glob(str(job / "benchmark_results/aiperf-*"))
    t0 = t1 = None
    if aiperf_dirs:
        a = Path(aiperf_dirs[0])
        t0, t1, marks = profiling_window(a / "aiperf.log")
        res["phases"] = {k: time.strftime("%H:%M:%S", time.gmtime(v)) for k, v in marks.items()}
        res["ci"] = run_ci(run, info, job)
    if t0 and t1:
        res["window"] = {"start": time.strftime("%H:%M:%S", time.gmtime(t0)),
                         "end": time.strftime("%H:%M:%S", time.gmtime(t1)), "seconds": round(t1 - t0)}
        sc = load_scrape(run / "scrape.jsonl", t0, t1)
        dec = {}
        for target in sorted(k for k in sc if k.startswith("decode")):
            rows = sc[target]
            gen, secs = delta(rows, "atom:generation_tokens_total", t0, t1)
            pre, _ = delta(rows, "atom:preemptions_total", t0, t1)
            fwd, fsecs = delta(rows, "atom:gpu_forward_seconds_sum", t0, t1)
            nfwd = len({dict(LABEL.findall(k)).get("tp_rank") for _, m in rows for k in m
                        if k.startswith("atom:gpu_forward_seconds_sum{")}) or 1
            bs_sum, _ = delta(rows, "atom:decode_batch_size_sum", t0, t1)
            bs_cnt, _ = delta(rows, "atom:decode_batch_size_count", t0, t1)
            xfer_sum, _ = delta(rows, "atom:pd_kv_transfer_seconds_sum", t0, t1)
            xfer_cnt, _ = delta(rows, "atom:pd_kv_transfer_seconds_count", t0, t1)
            dec[target] = {
                "running": stats(gauge(rows, "atom:requests_running", t0, t1), 1),
                "waiting": stats(gauge(rows, "atom:requests_waiting", t0, t1), 1),
                "kv_usage": stats(gauge(rows, "atom:kv_cache_usage_ratio", t0, t1, max), 3),
                "gen_tok_s": round(gen / secs, 1) if secs else None,
                "preemptions": pre,
                "admission_deferred": by_label(rows, "atom:pd_admission_deferred_total", "reason", t0, t1),
                "admission_pending_kv_gb": stats([v / 1e9 for v in gauge(rows, "atom:pd_admission_pending_kv_bytes", t0, t1)], 2),
                "admission_pending_requests": stats(gauge(rows, "atom:pd_admission_pending_requests", t0, t1), 1),
                "gpu_forward_busy": round(fwd / fsecs / nfwd, 3) if fsecs else None,
                "decode_batch_mean": round(bs_sum / bs_cnt, 1) if bs_cnt else None,
                "pd_kv_transfer_mean_s": round(xfer_sum / xfer_cnt, 3) if xfer_cnt else None,
            }
        res["decode"] = dec
        pre = {}
        for target in sorted(k for k in sc if k.startswith("prefill")):
            rows = sc[target]
            full, secs = delta(rows, "atom:prefix_cache_full_tokens_total", t0, t1)
            cached, _ = delta(rows, "atom:prefix_cache_cached_tokens_total", t0, t1)
            offload, _ = delta(rows, "atom:prefix_cache_offload_tokens_total", t0, t1)
            stages = {}
            for stage in sorted({dict(LABEL.findall(k)).get("pp_rank") for _, m in rows for k in m
                                 if k.startswith("atom:gpu_forward_seconds_sum{")} - {None}):
                d, s = delta(rows, "atom:gpu_forward_seconds_sum", t0, t1,
                             match=lambda k, st=stage: f'pp_rank="{st}"' in k)
                stages[stage] = round(d / s, 3) if s else None
            bt_sum, _ = delta(rows, "atom:prefill_batch_tokens_sum", t0, t1)
            bt_cnt, _ = delta(rows, "atom:prefill_batch_tokens_count", t0, t1)
            pre[target] = {
                "running": stats(gauge(rows, "atom:requests_running", t0, t1), 1),
                "waiting": stats(gauge(rows, "atom:requests_waiting", t0, t1), 1),
                "parked_kv_load": stats(gauge(rows, "atom:requests_parked_kv_load", t0, t1), 1),
                "kv_usage": stats(gauge(rows, "atom:kv_cache_usage_ratio", t0, t1, max), 3),
                "prompt_tok_s": round(full / secs, 0) if secs and full is not None else None,
                "computed_tok_s": round((full - cached - (offload or 0)) / secs, 0) if secs and full else None,
                "hit_hbm": round(cached / full, 4) if full else None,
                "hit_store": round((offload or 0) / full, 4) if full else None,
                "stage_forward_busy": stages,
                "batch_tokens_mean": round(bt_sum / bt_cnt, 0) if bt_cnt else None,
                "batches_per_s": round(bt_cnt / secs, 2) if bt_cnt and secs else None,
            }
        res["prefill"] = pre
        store = {}
        for target in sorted(k for k in sc if k.startswith("store")):
            rows = sc[target]
            cap = gauge(rows, "master_total_capacity_bytes", t0, t1, max)
            used = gauge(rows, "master_allocated_bytes", t0, t1, max)
            ev, _ = delta(rows, "master_evicted_size_bytes", t0, t1)
            gets, gsecs = delta(rows, "master_batch_get_replica_list_items_total", t0, t1)
            puts, _ = delta(rows, "master_batch_put_start_items_total", t0, t1)
            dup, _ = delta(rows, "master_batch_put_start_object_already_exists_total", t0, t1)
            store[target] = {
                "capacity_gib": round(max(cap) / 2**30, 1) if cap else None,
                "used_frac": stats([u / c for u, c in zip(used, cap) if c], 3),
                "evicted_gib": round(ev / 2**30, 1) if ev is not None else None,
                "get_items_s": round(gets / gsecs, 1) if gsecs else None,
                "put_items_s": round(puts / gsecs, 1) if gsecs else None,
                "put_already_exists": dup,
            }
        # Every owner of every node mounted: each pool's capacity equals its owners' segments in the plan.
        try:
            plan = json.loads((run / "store_plan.json").read_text())
            want = defaultdict(int)
            for spec in (plan.get("owners", ""), plan.get("decode_owners", "")):
                for entry in filter(None, spec.split(";")):
                    _, gib, nic = entry.split(":")
                    want[nic if plan.get("pools") == "per_nic" else "shared"] += int(gib)
            pools = sorted(want) if plan.get("pools") == "per_nic" else ["shared"]
            for idx, nic in enumerate(pools):
                if f"store{idx}" in store:
                    store[f"store{idx}"]["pool"] = nic
                    store[f"store{idx}"]["planned_gib"] = want[nic]
                    cap = store[f"store{idx}"]["capacity_gib"]
                    store[f"store{idx}"]["all_owners_mounted"] = cap is not None and cap >= want[nic] - 0.5
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        res["store"] = store
        res["nic"] = {Path(p).stem.removeprefix("nic-"): nic_rates(Path(p), t0, t1)
                      for p in sorted(glob.glob(str(run / "nic-*.csv")))}
        # Per node and half: rdma0-3 carry the P stages (node A) / Store owners (+ D1 in 1p3d node B),
        # rdma4-7 the decodes' P->D landing (and nothing else under per-NIC pools).
        halves = {}
        for node, nics in res["nic"].items():
            for half, names in (("rdma0-3", ("rdma0", "rdma1", "rdma2", "rdma3")),
                                ("rdma4-7", ("rdma4", "rdma5", "rdma6", "rdma7"))):
                rows = [nics[n] for n in names if n in nics]
                if rows:
                    halves[f"{node}:{half}"] = {"tx_gbs": round(sum(r["tx_mean"] for r in rows), 1),
                                                "rx_gbs": round(sum(r["rx_mean"] for r in rows), 1),
                                                "retx_mb": round(sum(r["retx_mb"] for r in rows), 1)}
        res["nic_halves"] = halves
        res["engine"] = {Path(p).stem: engine_stats(p, t0, t1)
                         for p in sorted(glob.glob(str(logs / "**/*-rank-*.log"), recursive=True))
                         if Path(p).stem.startswith(("prefill", "decode"))}
    # logs/ for a combined run, logs/<phase>/ for a benchmark or eval phase
    prefill_logs = glob.glob(str(logs / "**/prefill-rank-*.log"), recursive=True)
    decode_logs = glob.glob(str(logs / "**/decode-rank-*.log"), recursive=True)
    master_logs = glob.glob(str(logs / "**/mooncake-master-*.log"), recursive=True)
    res["events"] = {
        "cq_completion_timeouts": grep_count(prefill_logs, "CQ completion timeout"),
        "store_taint": grep_count(prefill_logs, "holds the taint marker"),
        "admission_timeouts": grep_count(prefill_logs, "Chunked P/D admission timed out"),
        "store_eviction_failure_rounds": grep_count(master_logs, "need_mem_eviction=1"),
        "decode_kv_receive_failed": grep_count(decode_logs, "KV receive failed"),
        "tracebacks": grep_count(prefill_logs + decode_logs, "Traceback (most recent call last)"),
    }
    res["verdict"] = verdict(res)
    (run / "point.json").write_text(json.dumps(res, indent=1))
    (run / "point.txt").write_text(render(res))
    print(render(res))
    return 0


def verdict(res):
    notes = []
    for name, d in (res.get("decode") or {}).items():
        kv = (d.get("kv_usage") or {}).get("p90")
        if kv and kv >= 0.95:
            notes.append(f"{name} KV full (p90 {kv:.2f})")
        deferred = sum((d.get("admission_deferred") or {}).values())
        if deferred and res.get("window"):
            notes.append(f"{name} admission deferred {deferred:.0f} in window")
        if d.get("preemptions"):
            notes.append(f"{name} preemptions {d['preemptions']:.0f}")
    for name, d in (res.get("prefill") or {}).items():
        waiting = (d.get("waiting") or {}).get("mean")
        busy = d.get("stage_forward_busy") or {}
        if waiting and waiting > 20:
            notes.append(f"{name} queue {waiting:.0f} waiting (stage busy {busy})")
        parked = (d.get("parked_kv_load") or {}).get("mean")
        if parked and parked > 5:
            notes.append(f"{name} {parked:.0f} parked on Store loads")
    for name, d in (res.get("store") or {}).items():
        used = (d.get("used_frac") or {}).get("p90")
        if used and used >= 0.87:
            notes.append(f"{name} at its eviction watermark (p90 {used:.2f})")
    ev = res.get("events") or {}
    for k in ("cq_completion_timeouts", "store_taint", "admission_timeouts", "decode_kv_receive_failed"):
        if ev.get(k):
            notes.append(f"{k}={ev[k]}")
    return "; ".join(notes) or "no limit indicator fired (check prefill stage busy and decode batch)"


def render(res):
    ci = res.get("ci") or {}
    lines = [f"{res['tag']} {res['topo']} C{res['conc']} {res['mode']} AL={res['al']} commit={res['commit'][:10]} image={res['image']}",
             f"  ranks rc={res['rank_rc']} watchdog={res['stopped_by_watchdog']} window={res.get('window')}"]
    if ci:
        lines.append(
            "  CI: tok/s/GPU {tput_per_gpu} over {total_gpu} GPUs | P90 TPOT {tpot_p90} ms | P90 E2E-norm intvty {interactivity}"
            " | TTFT p50 {median_ttft_ms} / p90 {ttft_p90} ms | completed {completed} | cache hit {cache_hit_rate}".format(
                **defaultdict(lambda: None, ci)))
    if res.get("gsm8k"):
        lines.append(f"  GSM8K: {res['gsm8k']}")
    for name, d in (res.get("decode") or {}).items():
        lines.append(f"  {name}: running {d['running']} kv {d['kv_usage']} gen {d['gen_tok_s']} tok/s batch {d['decode_batch_mean']}"
                     f" fwd-busy {d['gpu_forward_busy']} preempt {d['preemptions']} adm-deferred {d['admission_deferred']}"
                     f" pending-kv-GB {d['admission_pending_kv_gb']} pending-req {d['admission_pending_requests']}"
                     f" xfer {d['pd_kv_transfer_mean_s']} s")
    for name, d in (res.get("prefill") or {}).items():
        lines.append(f"  {name}: computed {d['computed_tok_s']} tok/s (prompt {d['prompt_tok_s']}) hit HBM {d['hit_hbm']} Store {d['hit_store']}"
                     f" | running {d['running']} waiting {d['waiting']} parked {d['parked_kv_load']}"
                     f" | stage fwd busy {d['stage_forward_busy']} | batch {d['batch_tokens_mean']} tok x {d['batches_per_s']}/s")
    for name, d in (res.get("store") or {}).items():
        lines.append(f"  {name} ({d.get('pool')}): cap {d['capacity_gib']}/{d.get('planned_gib')} GiB mounted={d.get('all_owners_mounted')}"
                     f" used {d['used_frac']} evicted {d['evicted_gib']} GiB"
                     f" gets {d['get_items_s']}/s puts {d['put_items_s']}/s dup {d['put_already_exists']}")
    if res.get("nic_halves"):
        lines.append("  NIC GB/s (mean, sum over 4 ports): " + " | ".join(
            f"{k} tx {v['tx_gbs']} rx {v['rx_gbs']} retx {v['retx_mb']} MB" for k, v in res["nic_halves"].items()))
    for name, e in (res.get("engine") or {}).items():
        def m(key, e=e):
            return (e.get(key) or {}).get("mean")
        lines.append(f"  log {name}: prompt {m('prompt_tok_s')} gen {m('gen_tok_s')} tok/s running {m('running')}"
                     f" waiting {m('waiting')} kv% {m('kv_usage_pct')} hit% {m('hbm_hit_pct')} toks/fwd {m('toks_per_fwd')}")
    lines.append(f"  events: {res.get('events')}")
    lines.append(f"  limit: {res.get('verdict')}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
