#!/usr/bin/env python3
"""
Master experiment orchestrator for GPT-OSS-120B MI355X Pareto optimization.

Strategy: targeted experiments, not full scan.
- Only test concurrency points most likely to move the Pareto frontier
- Each batch tests a single optimization variable
- Compare to baseline at key points, skip full sweep
- Early stop if improvement < threshold
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiment_tracker import (
    ExperimentTracker,
    BenchResult,
    OptimizationAttempt,
    Phase,
    EventType,
)
from notifier import Notifier

# ── constants ────────────────────────────────────────────────────

MODEL = "/data/openai/gpt-oss-120b"
PORT = 8080
BASE_URL = f"http://localhost:{PORT}"
STATE_DIR = os.environ.get("EXPERIMENT_STATE_DIR", "/app/experiment_status")
RESULTS_BASE = "/app/benchmark_results"

BASELINE_1K = {
    1: {
        "throughput": 272.8,
        "ttft_mean": 40.1,
        "ttft_p99": 54.2,
        "tpot_mean": 3.6,
        "tpot_p99": 3.6,
    },
    2: {
        "throughput": 522.4,
        "ttft_mean": 32.7,
        "ttft_p99": 69.1,
        "tpot_mean": 3.7,
        "tpot_p99": 3.8,
    },
    4: {
        "throughput": 937.3,
        "ttft_mean": 35.8,
        "ttft_p99": 80.0,
        "tpot_mean": 4.1,
        "tpot_p99": 4.2,
    },
    8: {
        "throughput": 1566.6,
        "ttft_mean": 41.5,
        "ttft_p99": 126.3,
        "tpot_mean": 5.0,
        "tpot_p99": 5.2,
    },
    16: {
        "throughput": 2484.2,
        "ttft_mean": 53.4,
        "ttft_p99": 213.4,
        "tpot_mean": 6.3,
        "tpot_p99": 6.7,
    },
    32: {
        "throughput": 3868.4,
        "ttft_mean": 104.4,
        "ttft_p99": 785.2,
        "tpot_mean": 8.0,
        "tpot_p99": 8.4,
    },
    64: {
        "throughput": 6059.7,
        "ttft_mean": 99.2,
        "ttft_p99": 794.4,
        "tpot_mean": 10.2,
        "tpot_p99": 11.1,
    },
    128: {
        "throughput": 8979.9,
        "ttft_mean": 136.2,
        "ttft_p99": 1361.3,
        "tpot_mean": 13.8,
        "tpot_p99": 14.5,
    },
    256: {
        "throughput": 12022.6,
        "ttft_mean": 1042.4,
        "ttft_p99": 9194.4,
        "tpot_mean": 19.9,
        "tpot_p99": 29.1,
    },
}
BASELINE_8K = {
    1: {
        "throughput": 263.1,
        "ttft_mean": 119.7,
        "ttft_p99": 130.5,
        "tpot_mean": 3.7,
        "tpot_p99": 3.7,
    },
    2: {
        "throughput": 494.3,
        "ttft_mean": 119.4,
        "ttft_p99": 205.2,
        "tpot_mean": 3.9,
        "tpot_p99": 3.9,
    },
    4: {
        "throughput": 856.1,
        "ttft_mean": 130.6,
        "ttft_p99": 357.7,
        "tpot_mean": 4.4,
        "tpot_p99": 4.5,
    },
    8: {
        "throughput": 1384.4,
        "ttft_mean": 159.8,
        "ttft_p99": 679.5,
        "tpot_mean": 5.5,
        "tpot_p99": 5.9,
    },
    16: {
        "throughput": 1989.0,
        "ttft_mean": 275.9,
        "ttft_p99": 1410.3,
        "tpot_mean": 7.6,
        "tpot_p99": 9.9,
    },
    32: {
        "throughput": 2858.7,
        "ttft_mean": 286.0,
        "ttft_p99": 2587.3,
        "tpot_mean": 10.6,
        "tpot_p99": 11.9,
    },
    64: {
        "throughput": 3873.6,
        "ttft_mean": 451.6,
        "ttft_p99": 5169.6,
        "tpot_mean": 15.8,
        "tpot_p99": 18.9,
    },
    128: {
        "throughput": 4723.5,
        "ttft_mean": 805.5,
        "ttft_p99": 10332.9,
        "tpot_mean": 25.8,
        "tpot_p99": 34.0,
    },
    256: {
        "throughput": 5484.8,
        "ttft_mean": 2599.9,
        "ttft_p99": 21740.8,
        "tpot_mean": 43.3,
        "tpot_p99": 56.8,
    },
}

IMPROVEMENT_THRESHOLD = 0.02  # 2% minimum to count as improvement
HEARTBEAT_INTERVAL = 600  # 10 minutes


# ── experiment definitions ───────────────────────────────────────


@dataclass
class ExperimentConfig:
    name: str
    description: str
    server_args: list[str]
    env_vars: dict[str, str]
    test_points: list[
        tuple[str, int, int, int]
    ]  # (scenario_name, isl, osl, concurrency)
    reason: str
    expected_impact: str
    priority: int  # 1=highest

    @property
    def label(self):
        return self.name.replace(" ", "_").lower()


def build_experiment_plan() -> list[ExperimentConfig]:
    """
    Build targeted experiment plan based on baseline analysis.

    Key observations from baseline:
    - TPOT at c1 is 3.6ms (excellent, memory-bandwidth bound)
    - TTFT at c256 is 1042ms/2600ms (BAD — prefill scheduling bottleneck)
    - Throughput scales well to c128, then TTFT kills c256 usability
    - CUDAGraph padding waste is small (existing sizes match most batch sizes)

    Strategy: focus on high-value concurrency points (32/64/128/256)
    """

    base_server = [
        f"--model={MODEL}",
        "--kv_cache_dtype=fp8",
        "--server-port=8080",
    ]

    [("1k_1k", 1024, 1024, c) for c in [1, 32, 64, 128, 256]]
    [("8k_1k", 8192, 1024, c) for c in [1, 64, 128, 256]]
    high_conc_1k = [("1k_1k", 1024, 1024, c) for c in [32, 64, 128, 256]]
    high_conc_8k = [("8k_1k", 8192, 1024, c) for c in [64, 128, 256]]
    ttft_critical = [("1k_1k", 1024, 1024, c) for c in [128, 256]] + [
        ("8k_1k", 8192, 1024, c) for c in [64, 128, 256]
    ]

    return [
        ExperimentConfig(
            name="gpu_util_095",
            description="Increase GPU memory utilization 0.9->0.95 for more KV blocks",
            server_args=base_server + ["--gpu-memory-utilization=0.95"],
            env_vars={"AITER_LOG_LEVEL": "WARNING"},
            test_points=high_conc_1k + high_conc_8k,
            reason="More KV blocks = more concurrent sequences = higher throughput at high concurrency. "
            "TTFT at c256 is our worst metric; more KV capacity helps.",
            expected_impact="Throughput +3-8% at c128/c256, TTFT improvement at high conc",
            priority=1,
        ),
        ExperimentConfig(
            name="cudagraph_dense",
            description="Denser CUDAGraph capture via CLI: add sizes 3,6,12,24",
            server_args=base_server
            + [
                "--gpu-memory-utilization=0.9",
                "--cudagraph-capture-sizes",
                "1",
                "2",
                "3",
                "4",
                "6",
                "8",
                "12",
                "16",
                "24",
                "32",
                "48",
                "64",
                "128",
                "256",
                "512",
            ],
            env_vars={"AITER_LOG_LEVEL": "WARNING"},
            test_points=[("1k_1k", 1024, 1024, c) for c in [1, 4, 8, 32]]
            + [("8k_1k", 8192, 1024, c) for c in [1, 8]],
            reason="At low batch sizes (3,5,6,7,...), current sizes cause padding to next power-of-2. "
            "Dense sizes reduce decode padding waste.",
            expected_impact="TPOT -2-5% at low concurrency, negligible at high conc",
            priority=2,
        ),
        ExperimentConfig(
            name="max_batch_tokens_8k",
            description="Reduce max_num_batched_tokens 16384->8192 for faster prefill/decode switching",
            server_args=base_server
            + [
                "--gpu-memory-utilization=0.9",
                "--max-num-batched-tokens=8192",
            ],
            env_vars={"AITER_LOG_LEVEL": "WARNING"},
            test_points=ttft_critical,
            reason="Smaller prefill batches = decode steps happen sooner = lower TTFT at high concurrency. "
            "Trade: slightly lower peak throughput for much better TTFT.",
            expected_impact="TTFT -15-30% at c128/c256, throughput -3-5%",
            priority=2,
        ),
        ExperimentConfig(
            name="moe_threshold_tune",
            description="Tune dual-stream MoE threshold 1024->512 for GPT-OSS-120B",
            server_args=base_server + ["--gpu-memory-utilization=0.9"],
            env_vars={
                "AITER_LOG_LEVEL": "WARNING",
                "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD": "512",
            },
            test_points=high_conc_1k[:2]
            + high_conc_8k[:1],  # Quick probe: c32,c64 for 1k; c64 for 8k
            reason="GPT-OSS-120B is MoE. Dual-stream dispatch threshold affects MoE kernel efficiency. "
            "512 vs 1024 may better match typical decode batch sizes.",
            expected_impact="Throughput +1-5% if threshold matches workload better",
            priority=3,
        ),
        ExperimentConfig(
            name="block_size_32",
            description="Double KV cache block size 16->32 to reduce metadata overhead",
            server_args=base_server
            + [
                "--gpu-memory-utilization=0.9",
                "--block-size=32",
            ],
            env_vars={"AITER_LOG_LEVEL": "WARNING"},
            test_points=high_conc_1k[:2] + high_conc_8k[:1],  # Quick probe
            reason="Larger blocks = fewer block table entries = less metadata overhead per token. "
            "May slightly improve memory access patterns.",
            expected_impact="TPOT -1-3%, possible TTFT improvement from faster allocation",
            priority=3,
        ),
    ]


# ── server management ────────────────────────────────────────────


def stop_server():
    print("[server] Stopping all Python processes...")
    subprocess.run(
        [
            "bash",
            "-c",
            "pkill -f 'atom.entrypoints' 2>/dev/null; sleep 2; pkill -9 -f 'atom.entrypoints' 2>/dev/null",
        ],
        timeout=15,
    )
    time.sleep(3)


def start_server(args: list[str], env_vars: dict[str, str], log_file: str) -> bool:
    stop_server()

    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
    args_str = " ".join(args)
    cmd = f"{env_str} python -m atom.entrypoints.openai_server {args_str}"

    print(f"[server] Starting: {cmd}")
    subprocess.Popen(
        ["bash", "-c", f"cd /app/ATOM && {cmd} > {log_file} 2>&1"],
    )

    # Wait for server to be ready (health check)
    print("[server] Waiting for server to be ready...")
    for attempt in range(120):  # 10 minutes max
        time.sleep(5)
        try:
            import urllib.request

            req = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[server] Ready after {(attempt+1)*5}s")
                    return True
        except Exception:
            if attempt % 12 == 11:
                print(f"[server] Still waiting... ({(attempt+1)*5}s)")

    print("[server] FAILED to start within 10 minutes")
    return False


def check_server_health() -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── benchmark execution ──────────────────────────────────────────


def run_single_benchmark(
    isl: int,
    osl: int,
    conc: int,
    scenario: str,
    results_dir: str,
    label: str,
) -> BenchResult | None:
    num_prompts = max(conc * 10, 32)
    result_file = f"{scenario}_c{conc}.json"

    print(f"  [{time.strftime('%H:%M:%S')}] {scenario} c={conc} prompts={num_prompts}")

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
        stdout_path = f"{results_dir}/{scenario}_c{conc}.stdout"
        with open(stdout_path, "w") as f:
            f.write(r.stdout)
        if r.returncode != 0:
            with open(f"{results_dir}/{scenario}_c{conc}.stderr", "w") as f:
                f.write(r.stderr)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {scenario} c={conc}")
        return None

    return _parse_result(results_dir, scenario, conc, label)


def _parse_result(
    results_dir: str, scenario: str, conc: int, label: str
) -> BenchResult | None:
    json_file = f"{results_dir}/{scenario}_c{conc}.json"
    stdout_file = f"{results_dir}/{scenario}_c{conc}.stdout"

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


# ── comparison logic ─────────────────────────────────────────────


def get_baseline(scenario: str, conc: int) -> dict | None:
    tbl = BASELINE_1K if "1k_1k" in scenario else BASELINE_8K
    return tbl.get(conc)


def compute_improvement(result: BenchResult) -> dict:
    bl = get_baseline(result.scenario, result.concurrency)
    if not bl:
        return {"has_baseline": False}
    tput_delta = (result.throughput - bl["throughput"]) / bl["throughput"]
    tpot_delta = (bl["tpot_mean"] - result.tpot_mean) / bl["tpot_mean"]
    ttft_delta = (bl["ttft_mean"] - result.ttft_mean) / bl["ttft_mean"]
    return {
        "has_baseline": True,
        "throughput_pct": tput_delta * 100,
        "tpot_pct": tpot_delta * 100,
        "ttft_pct": ttft_delta * 100,
        "is_pareto_improving": tput_delta > IMPROVEMENT_THRESHOLD
        or tpot_delta > IMPROVEMENT_THRESHOLD,
    }


# ── heartbeat ────────────────────────────────────────────────────


class HeartbeatThread(threading.Thread):
    def __init__(self, tracker: ExperimentTracker, notifier: Notifier):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.notifier = notifier
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            evt = {
                "type": "heartbeat",
                "message": f"Alive — phase: {self.tracker.state.phase}, "
                f"progress: {self.tracker.progress_pct:.0f}%",
                "timestamp": time.time(),
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "progress_pct": self.tracker.progress_pct,
            }
            payload = self.tracker.build_notification(evt)
            payload["event_type"] = "heartbeat"
            self.notifier.send(payload)

    def stop(self):
        self._stop.set()


# ── main orchestration ───────────────────────────────────────────


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(RESULTS_BASE, exist_ok=True)

    # Copy notify config if available
    local_cfg = Path(__file__).parent / "notify_config.json"
    target_cfg = Path(STATE_DIR) / "notify_config.json"
    if local_cfg.exists() and not target_cfg.exists():
        target_cfg.write_text(local_cfg.read_text())

    notifier = Notifier(config_dir=STATE_DIR)
    tracker = ExperimentTracker(
        state_dir=STATE_DIR,
        notify_callback=lambda evt: notifier.send(tracker.build_notification(evt)),
    )

    experiments = build_experiment_plan()
    total_benchmarks = sum(len(e.test_points) for e in experiments)

    tracker.plan(
        total_benchmarks=total_benchmarks,
        total_optimizations=len(experiments),
        model="GPT-OSS-120B (MXFP4)",
        hardware="MI355X",
        machine="smci355-ccs-aus-m13-05",
        branch="perf/gpt-oss-120b-mi355x-opt",
    )

    # Seed baseline into tracker
    for conc, data in BASELINE_1K.items():
        tracker.record_benchmark(
            BenchResult(
                scenario="1k_1k",
                concurrency=conc,
                label="baseline",
                **data,
            ),
            is_baseline=True,
        )
    for conc, data in BASELINE_8K.items():
        tracker.record_benchmark(
            BenchResult(
                scenario="8k_1k",
                concurrency=conc,
                label="baseline",
                **data,
            ),
            is_baseline=True,
        )

    tracker.gpu_start()
    tracker.emit_custom(
        EventType.EXPERIMENT_STARTED,
        f"Starting targeted Pareto optimization: {len(experiments)} experiments, "
        f"~{total_benchmarks} benchmarks",
    )

    heartbeat = HeartbeatThread(tracker, notifier)
    heartbeat.start()

    # Track which optimizations showed improvement
    winners = []
    combined_server_args = [
        f"--model={MODEL}",
        "--kv_cache_dtype=fp8",
        "--server-port=8080",
    ]
    combined_env = {"AITER_LOG_LEVEL": "WARNING"}

    # Sort by priority
    experiments.sort(key=lambda e: e.priority)

    for exp_idx, exp in enumerate(experiments):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {exp_idx+1}/{len(experiments)}: {exp.name}")
        print(f"  Description: {exp.description}")
        print(f"  Reason: {exp.reason}")
        print(f"  Expected: {exp.expected_impact}")
        print(f"  Test points: {len(exp.test_points)}")
        print(f"{'='*70}\n")

        opt = OptimizationAttempt(
            name=exp.name,
            description=exp.description,
            server_args=exp.server_args,
            env_vars=exp.env_vars,
        )
        tracker.start_optimization(opt)
        tracker.set_phase(Phase.OPTIMIZING, exp.name)

        # Start server with this config
        log_file = f"/app/server_{exp.label}.log"
        server_ok = start_server(exp.server_args, exp.env_vars, log_file)

        if not server_ok:
            tracker.finish_optimization(exp.name, "failed", "Server failed to start")
            tracker.emit_custom(
                EventType.SERVER_FAILED, f"Server failed for {exp.name}"
            )
            continue

        tracker.emit_custom(EventType.SERVER_STARTED, f"Server ready for {exp.name}")
        tracker.set_phase(Phase.BENCHMARKING, exp.name)

        results_dir = f"{RESULTS_BASE}/{exp.label}_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(results_dir, exist_ok=True)

        improvements = []
        any_pareto_gain = False

        for scenario, isl, osl, conc in exp.test_points:
            result = run_single_benchmark(
                isl, osl, conc, scenario, results_dir, exp.label
            )
            if result:
                tracker.record_benchmark(result)
                imp = compute_improvement(result)
                improvements.append((scenario, conc, imp, result))

                bl = get_baseline(scenario, conc)
                if imp["has_baseline"]:
                    tp = imp["throughput_pct"]
                    tpot = imp["tpot_pct"]
                    ttft = imp["ttft_pct"]
                    marker = " ***" if imp["is_pareto_improving"] else ""
                    print(
                        f"    -> throughput: {tp:+.1f}%, TPOT: {tpot:+.1f}%, "
                        f"TTFT: {ttft:+.1f}%{marker}"
                    )
                    if imp["is_pareto_improving"]:
                        any_pareto_gain = True

        # Batch done — evaluate
        n_improved = sum(
            1 for _, _, imp, _ in improvements if imp.get("is_pareto_improving")
        )
        total_pts = len(improvements)

        tracker.record_batch_done(exp.name, total_pts)

        if any_pareto_gain:
            tracker.finish_optimization(exp.name, "success")
            winners.append(exp)
            # Merge winning config into combined
            for arg in exp.server_args:
                if (
                    arg not in combined_server_args
                    and "--server-port" not in arg
                    and "--model" not in arg
                    and "--kv_cache_dtype" not in arg
                ):
                    combined_server_args.append(arg)
            combined_env.update(exp.env_vars)
            print(
                f"\n  >> WINNER: {exp.name} — {n_improved}/{total_pts} points improved"
            )
        else:
            tracker.finish_optimization(
                exp.name, "failed", f"No Pareto improvement ({n_improved}/{total_pts})"
            )
            print(f"\n  >> NO IMPROVEMENT: {exp.name} — skipping")

        # Early stop check
        if tracker.state.suggest_stop:
            print(f"\n!! EARLY STOP SUGGESTED: {tracker.state.stop_reason}")
            tracker.emit_custom(EventType.EARLY_STOP, tracker.state.stop_reason)
            break

    # ── Final combined experiment ────────────────────────────────
    if len(winners) > 1:
        print(f"\n{'='*70}")
        print(f"FINAL: Combined best configuration ({len(winners)} winners)")
        print(f"  Args: {combined_server_args}")
        print(f"  Env: {combined_env}")
        print(f"{'='*70}\n")

        tracker.set_phase(Phase.FINAL_BENCH, "Combined best config")

        all_key_points = [("1k_1k", 1024, 1024, c) for c in [1, 32, 64, 128, 256]] + [
            ("8k_1k", 8192, 1024, c) for c in [1, 64, 128, 256]
        ]

        log_file = "/app/server_combined.log"
        server_ok = start_server(combined_server_args, combined_env, log_file)

        if server_ok:
            results_dir = f"{RESULTS_BASE}/combined_{time.strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(results_dir, exist_ok=True)

            for scenario, isl, osl, conc in all_key_points:
                result = run_single_benchmark(
                    isl, osl, conc, scenario, results_dir, "combined"
                )
                if result:
                    tracker.record_benchmark(result)
                    imp = compute_improvement(result)
                    if imp["has_baseline"]:
                        print(
                            f"    -> throughput: {imp['throughput_pct']:+.1f}%, "
                            f"TPOT: {imp['tpot_pct']:+.1f}%, "
                            f"TTFT: {imp['ttft_pct']:+.1f}%"
                        )

            tracker.record_batch_done("combined", len(all_key_points))

    elif len(winners) == 1:
        print(f"\n  Single winner: {winners[0].name} — no need for combined run")

    # ── Final report ─────────────────────────────────────────────

    stop_server()
    tracker.gpu_stop()
    tracker.set_phase(Phase.REPORTING)

    # Print Pareto comparison
    shift = tracker.get_pareto_shift()
    print(f"\n{'='*70}")
    print("FINAL PARETO FRONTIER REPORT")
    print(f"{'='*70}")

    print(
        f"\nBaseline max throughput: {shift.get('baseline_max_throughput', 0):.0f} tok/s"
    )
    print(
        f"Current max throughput:  {shift.get('current_max_throughput', 0):.0f} tok/s"
    )
    print(
        f"Throughput improvement:  {shift.get('throughput_improvement_pct', 0):+.1f}%"
    )
    print(f"\nBaseline min TPOT: {shift.get('baseline_min_tpot', 0):.1f} ms")
    print(f"Current min TPOT:  {shift.get('current_min_tpot', 0):.1f} ms")
    print(f"TPOT improvement:  {shift.get('tpot_improvement_pct', 0):+.1f}%")
    print(f"\nFrontier points: {shift.get('frontier_points', 0)}")
    print(f"GPU hours used:  {tracker.state.gpu_hours:.2f}h")

    print(f"\nWinning optimizations: {[w.name for w in winners]}")
    if not winners:
        print("No optimizations improved the Pareto frontier.")

    # Print best results per scenario
    print("\n--- Best Results by Scenario ---")
    for key, res in sorted(tracker.state.best_results.items()):
        bl = get_baseline(res["scenario"], res["concurrency"])
        bl_tput = bl["throughput"] if bl else 0
        delta = ((res["throughput"] - bl_tput) / bl_tput * 100) if bl_tput > 0 else 0
        print(
            f"  {key}: {res['throughput']:.0f} tok/s ({delta:+.1f}% vs baseline), "
            f"TPOT {res['tpot_mean']:.1f}ms, label={res.get('label','')}"
        )

    tracker.emit_custom(
        EventType.ALL_DONE,
        f"Experiment complete. GPU: {tracker.state.gpu_hours:.2f}h. "
        f"Winners: {[w.name for w in winners]}. "
        f"Throughput shift: {shift.get('throughput_improvement_pct', 0):+.1f}%",
    )
    tracker.set_phase(Phase.DONE)

    heartbeat.stop()
    print(f"\nStatus files: {STATE_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
