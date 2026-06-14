#!/usr/bin/env python3
"""
CLI tool to query experiment status — run locally or remotely.

Usage:
    # Local (if state_dir is accessible):
    python status.py [--dir /path/to/experiment_status]

    # Remote (pull from Docker container over SSH):
    python status.py --remote smci355-ccs-aus-m13-05.cs-aus.dcgpu --container chuali_perf_opt

    # Watch mode (auto-refresh):
    python status.py --watch 30

    # JSON output (for piping):
    python status.py --json

    # Show specific section:
    python status.py --section pareto
    python status.py --section events
    python status.py --section optimizations
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_STATE_DIR = "/app/experiment_status"
LOCAL_CACHE_DIR = Path("experiment_status_cache")


def fetch_remote(host: str, container: str, remote_dir: str) -> dict:
    """Pull progress.json from a remote Docker container via SSH."""
    cmd = (
        f'wsl -- ssh {host} "docker exec {container} '
        f'cat {remote_dir}/progress.json"'
    )
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout)
            LOCAL_CACHE_DIR.mkdir(exist_ok=True)
            (LOCAL_CACHE_DIR / "progress.json").write_text(json.dumps(data, indent=2))
            return data
    except Exception as e:
        print(f"[warn] Remote fetch failed: {e}", file=sys.stderr)

    cached = LOCAL_CACHE_DIR / "progress.json"
    if cached.exists():
        print("[info] Using cached data", file=sys.stderr)
        return json.loads(cached.read_text())
    return {}


def load_local(state_dir: str) -> dict:
    p = Path(state_dir) / "progress.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.0f}m"
    return f"{seconds/3600:.1f}h"


def print_summary(data: dict):
    if not data:
        print("No experiment data found.")
        return

    phase = data.get("phase", "unknown")
    total = data.get("total_planned_benchmarks", 0)
    done = data.get("completed_benchmarks", 0)
    pct = done / total * 100 if total > 0 else 0
    elapsed = time.time() - data.get("started_at", time.time())
    gpu_h = data.get("gpu_hours", 0)

    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "#" * filled + "-" * (bar_width - filled)

    print("=" * 60)
    print("  ATOM GPT-OSS-120B MI355X Experiment Status")
    print("=" * 60)
    print(f"  Phase:    {phase}")
    print(f"  Progress: [{bar}] {pct:.0f}%")
    print(f"  Benchmarks: {done}/{total}")
    print(f"  Elapsed:  {format_elapsed(elapsed)}")
    print(f"  GPU time: {gpu_h:.2f}h")
    print(f"  Machine:  {data.get('machine', '?')}")
    print(f"  Branch:   {data.get('branch', '?')}")

    if data.get("suggest_stop"):
        print(f"\n  !! SUGGEST STOP: {data.get('stop_reason', '?')}")

    current = data.get("current_optimization") or data.get("current_config")
    if current:
        print(f"\n  Current: {current}")


def print_best_results(data: dict):
    best = data.get("best_results", {})
    if not best:
        return
    print("\n--- Best Results ---")
    print(f"  {'Scenario':<20} {'Tput':>10} {'TTFT':>10} {'TPOT':>10} {'Label':>12}")
    print(f"  {'-'*62}")
    for key in sorted(best.keys()):
        r = best[key]
        print(
            f"  {key:<20} {r['throughput']:>10.0f} "
            f"{r['ttft_mean']:>10.1f} {r['tpot_mean']:>10.1f} "
            f"{r.get('label', ''):>12}"
        )


def print_pareto(data: dict):
    frontier = data.get("pareto_frontier", [])
    if not frontier:
        return
    print("\n--- Pareto Frontier ---")
    print(
        f"  {'Scenario':<15} {'Conc':>5} {'Tput':>10} "
        f"{'TPOT':>8} {'TTFT':>8} {'Label':>12}"
    )
    print(f"  {'-'*60}")
    for pt in frontier:
        print(
            f"  {pt['scenario']:<15} {pt['concurrency']:>5} "
            f"{pt['throughput']:>10.0f} {pt['tpot_mean']:>8.1f} "
            f"{pt['ttft_mean']:>8.1f} {pt.get('label', ''):>12}"
        )

    # Shift vs baseline
    baseline = data.get("baseline_results", [])
    if baseline and frontier:
        bl_max = max(r["throughput"] for r in baseline)
        cur_max = max(pt["throughput"] for pt in frontier)
        bl_min_tpot = min(r["tpot_mean"] for r in baseline)
        cur_min_tpot = min(pt["tpot_mean"] for pt in frontier)
        print(
            f"\n  Throughput shift: {bl_max:.0f} -> {cur_max:.0f} "
            f"({(cur_max-bl_max)/bl_max*100:+.1f}%)"
        )
        print(
            f"  TPOT shift:      {bl_min_tpot:.1f} -> {cur_min_tpot:.1f} "
            f"({(bl_min_tpot-cur_min_tpot)/bl_min_tpot*100:+.1f}%)"
        )


def print_optimizations(data: dict):
    opts = data.get("optimizations", [])
    if not opts:
        return
    print("\n--- Optimization History ---")
    for i, o in enumerate(opts, 1):
        dur = ""
        if o.get("finished_at") and o.get("started_at"):
            dur = format_elapsed(o["finished_at"] - o["started_at"])
        status_icon = {
            "success": "[OK]",
            "failed": "[FAIL]",
            "abandoned": "[SKIP]",
            "running": "[..]",
        }.get(o["status"], "[?]")
        print(f"  {i}. {status_icon} {o['name']} ({dur})")
        if o.get("error"):
            print(f"       Error: {o['error']}")


def print_events(data: dict, limit: int = 15):
    events = data.get("events", [])
    if not events:
        return
    print(f"\n--- Recent Events (last {min(limit, len(events))}) ---")
    for evt in events[-limit:]:
        ts = evt.get("time_str", "?")
        print(f"  [{ts}] {evt['type']}: {evt['message']}")


def print_full(data: dict):
    print_summary(data)
    print_best_results(data)
    print_pareto(data)
    print_optimizations(data)
    print_events(data)
    print()


def main():
    parser = argparse.ArgumentParser(description="Query ATOM experiment status")
    parser.add_argument(
        "--dir",
        default=DEFAULT_STATE_DIR,
        help="Local state directory",
    )
    parser.add_argument(
        "--remote",
        default="",
        help="SSH host for remote fetch",
    )
    parser.add_argument(
        "--container",
        default="chuali_perf_opt",
        help="Docker container name",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Auto-refresh interval",
    )
    parser.add_argument(
        "--section",
        choices=["summary", "best", "pareto", "optimizations", "events", "all"],
        default="all",
        help="Show specific section",
    )

    args = parser.parse_args()

    def fetch():
        if args.remote:
            return fetch_remote(args.remote, args.container, args.dir)
        return load_local(args.dir)

    def display(data):
        if args.json:
            print(json.dumps(data, indent=2, default=str))
            return
        section_map = {
            "summary": print_summary,
            "best": print_best_results,
            "pareto": print_pareto,
            "optimizations": print_optimizations,
            "events": print_events,
            "all": print_full,
        }
        section_map[args.section](data)

    if args.watch > 0:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                data = fetch()
                display(data)
                print(f"\n  [Refreshing every {args.watch}s, Ctrl+C to stop]")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        data = fetch()
        display(data)


if __name__ == "__main__":
    main()
