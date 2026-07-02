#!/usr/bin/env python3
"""
GPT-OSS-120B MI355X Performance Benchmark Suite
with integrated experiment tracking and notification.
"""

from __future__ import annotations

import subprocess
import json
import os
import sys
import time
import glob
import re
from pathlib import Path

# Allow importing from same directory when run as script
sys.path.insert(0, str(Path(__file__).parent))

from experiment_tracker import (
    ExperimentTracker,
    BenchResult,
    Phase,
    EventType,
)
from notifier import Notifier

MODEL = "/data/openai/gpt-oss-120b"
PORT = 8080
BASE_URL = f"http://localhost:{PORT}"
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SCENARIOS = {"1k_1k": (1024, 1024), "8k_1k": (8192, 1024)}

STATE_DIR = os.environ.get("EXPERIMENT_STATE_DIR", "/app/experiment_status")


def setup_tracking(label: str) -> tuple[ExperimentTracker, Notifier]:
    notifier = Notifier(config_dir=STATE_DIR)
    tracker = ExperimentTracker(
        state_dir=STATE_DIR,
        notify_callback=lambda evt: notifier.send(tracker.build_notification(evt)),
    )
    total_benchmarks = len(SCENARIOS) * len(CONCURRENCY_LEVELS)
    tracker.plan(
        total_benchmarks=total_benchmarks,
        total_optimizations=7,
        model="GPT-OSS-120B (MXFP4)",
        hardware="8x MI355X",
        machine="smci355-ccs-aus-m13-05",
        branch="perf/gpt-oss-120b-mi355x-opt",
    )
    return tracker, notifier


def run_benchmark(
    isl: int,
    osl: int,
    conc: int,
    scenario: str,
    results_dir: str,
    tracker: ExperimentTracker,
    label: str,
    is_baseline: bool = False,
) -> BenchResult | None:
    num_prompts = max(conc * 10, 32)
    result_file = f"{scenario}_c{conc}.json"
    tracker.state.current_config = f"{scenario} c={conc}"
    tracker.save()

    print(
        f"[{time.strftime('%H:%M:%S')}] Running {scenario} c={conc} "
        f"prompts={num_prompts}"
    )

    cmd = [
        sys.executable,
        "-m",
        "atom.benchmarks.benchmark_serving",
        f"--model={MODEL}",
        "--backend=vllm",
        f"--base-url={BASE_URL}",
        "--dataset-name=random",
        f"--random-input-len={isl}",
        f"--random-output-len={osl}",
        "--random-range-ratio=0.8",
        f"--num-prompts={num_prompts}",
        f"--max-concurrency={conc}",
        "--request-rate=inf",
        "--ignore-eos",
        "--percentile-metrics=ttft,tpot,itl,e2el",
        f"--result-dir={results_dir}",
        f"--result-filename={result_file}",
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        with open(f"{results_dir}/{scenario}_c{conc}.stdout", "w") as f:
            f.write(r.stdout)
        if r.returncode != 0:
            print(f"  WARN: exit code {r.returncode}")
            with open(f"{results_dir}/{scenario}_c{conc}.stderr", "w") as f:
                f.write(r.stderr)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {scenario} c={conc}")
        return None

    result = _parse_result(results_dir, scenario, conc, label)
    if result:
        tracker.record_benchmark(result, is_baseline=is_baseline)
    return result


def _parse_result(
    results_dir: str, scenario: str, conc: int, label: str
) -> BenchResult | None:
    json_file = f"{results_dir}/{scenario}_c{conc}.json"
    stdout_file = f"{results_dir}/{scenario}_c{conc}.stdout"

    # Try JSON first
    if os.path.exists(json_file):
        try:
            d = json.load(open(json_file))
            return BenchResult(
                scenario=scenario,
                concurrency=conc,
                throughput=d.get("output_throughput", d.get("request_throughput", 0)),
                ttft_mean=d.get("mean_ttft_ms", 0),
                ttft_p99=d.get("p99_ttft_ms", 0),
                tpot_mean=d.get("mean_tpot_ms", 0),
                tpot_p99=d.get("p99_tpot_ms", 0),
                timestamp=time.time(),
                label=label,
            )
        except Exception:
            pass

    # Fall back to stdout parsing
    if os.path.exists(stdout_file):
        try:
            text = open(stdout_file).read()
            tput = re.search(r"Output token throughput.*?(\d+\.?\d*)", text)
            ttft_mean = re.search(r"Mean TTFT.*?(\d+\.?\d*)", text)
            ttft_p99 = re.search(r"P99 TTFT.*?(\d+\.?\d*)", text)
            tpot_mean = re.search(r"Mean TPOT.*?(\d+\.?\d*)", text)
            tpot_p99 = re.search(r"P99 TPOT.*?(\d+\.?\d*)", text)
            if all(
                v is not None for v in [tput, ttft_mean, ttft_p99, tpot_mean, tpot_p99]
            ):
                return BenchResult(
                    scenario=scenario,
                    concurrency=conc,
                    throughput=float(tput.group(1)),
                    ttft_mean=float(ttft_mean.group(1)),
                    ttft_p99=float(ttft_p99.group(1)),
                    tpot_mean=float(tpot_mean.group(1)),
                    tpot_p99=float(tpot_p99.group(1)),
                    timestamp=time.time(),
                    label=label,
                )
        except Exception:
            pass

    return None


def summarize(results_dir: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(f"{results_dir}/*.json")):
        if "summary" in f or "progress" in f:
            continue
        try:
            d = json.load(open(f))
            name = Path(f).stem
            rows.append(
                {
                    "scenario": name,
                    "throughput": d.get(
                        "output_throughput", d.get("request_throughput", 0)
                    ),
                    "ttft_mean": d.get("mean_ttft_ms", 0),
                    "ttft_p99": d.get("p99_ttft_ms", 0),
                    "tpot_mean": d.get("mean_tpot_ms", 0),
                    "tpot_p99": d.get("p99_tpot_ms", 0),
                }
            )
        except Exception as e:
            print(f"Error parsing {f}: {e}")
    if rows:
        print(
            f"\n{'Scenario':<20} {'Tput(tok/s)':>12} {'TTFT mean':>10} "
            f"{'TTFT p99':>10} {'TPOT mean':>10} {'TPOT p99':>10}"
        )
        print("-" * 82)
        for r in rows:
            print(
                f"{r['scenario']:<20} {r['throughput']:>12.1f} "
                f"{r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f} "
                f"{r['tpot_mean']:>10.1f} {r['tpot_p99']:>10.1f}"
            )
        with open(f"{results_dir}/summary.json", "w") as out:
            json.dump(rows, out, indent=2)
        print(f"\nSaved summary to {results_dir}/summary.json")
    return rows


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    tag = sys.argv[2] if len(sys.argv) > 2 else time.strftime("%Y%m%d_%H%M%S")
    is_baseline = label == "baseline"

    results_dir = f"/app/benchmark_results/{label}_{tag}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results dir: {results_dir}")

    tracker, notifier = setup_tracking(label)
    tracker.gpu_start()

    if is_baseline:
        tracker.set_phase(Phase.BASELINE, f"Running baseline: {label}")
    else:
        tracker.set_phase(Phase.BENCHMARKING, f"Benchmarking: {label}")

    tracker.emit_custom(
        EventType.EXPERIMENT_STARTED,
        f"Starting benchmark suite '{label}' "
        f"({len(SCENARIOS) * len(CONCURRENCY_LEVELS)} runs)",
    )

    for scenario, (isl, osl) in SCENARIOS.items():
        for conc in CONCURRENCY_LEVELS:
            run_benchmark(
                isl,
                osl,
                conc,
                scenario,
                results_dir,
                tracker,
                label,
                is_baseline=is_baseline,
            )

        tracker.record_batch_done(
            f"{scenario}",
            len(CONCURRENCY_LEVELS),
        )

    tracker.gpu_stop()
    summarize(results_dir)
    tracker.emit_custom(
        EventType.ALL_DONE,
        f"All benchmarks for '{label}' complete. "
        f"GPU time: {tracker.state.gpu_hours:.2f}h",
    )
    tracker.set_phase(Phase.DONE if is_baseline else Phase.OPTIMIZING)

    print("\nAll benchmarks complete")
    print(f"Status files at: {STATE_DIR}/")
    print("  - STATUS.md")
    print("  - progress.json")
    print("  - latest_summary.txt")


if __name__ == "__main__":
    main()
