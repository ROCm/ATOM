#!/usr/bin/env python3
"""
Experiment progress tracker with Pareto frontier analysis.

Maintains structured state across optimization iterations,
detects Pareto improvements, and generates status files.
"""

from __future__ import annotations

import json
import time
import os
import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class Phase(str, Enum):
    INIT = "initializing"
    BASELINE = "baseline_benchmarking"
    OPTIMIZING = "optimizing"
    BENCHMARKING = "benchmarking_optimization"
    PROFILING = "profiling"
    FINAL_BENCH = "final_benchmarking"
    REPORTING = "generating_report"
    SUBMITTING_PR = "submitting_pr"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


class EventType(str, Enum):
    EXPERIMENT_STARTED = "experiment_started"
    BATCH_COMPLETED = "batch_completed"
    NEW_PARETO_POINT = "new_pareto_point"
    BEST_REFRESHED = "best_refreshed"
    NO_PROGRESS = "no_progress"
    EARLY_STOP = "early_stop_suggested"
    ALL_DONE = "all_experiments_done"
    PR_CREATED = "pr_created"
    CODE_COMMITTED = "code_committed"
    SERVER_STARTED = "server_started"
    SERVER_FAILED = "server_failed"
    OPT_APPLIED = "optimization_applied"
    PHASE_CHANGED = "phase_changed"


@dataclass
class BenchResult:
    scenario: str
    concurrency: int
    throughput: float
    ttft_mean: float
    ttft_p99: float
    tpot_mean: float
    tpot_p99: float
    timestamp: float = 0.0
    label: str = ""

    @property
    def tok_per_s_per_user(self) -> float:
        return 1000.0 / self.tpot_mean if self.tpot_mean > 0 else 0.0


@dataclass
class OptimizationAttempt:
    name: str
    description: str
    code_changes: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    server_args: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, success, failed, abandoned
    results: list[dict] = field(default_factory=list)
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class ExperimentState:
    phase: str = Phase.INIT.value
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    total_planned_benchmarks: int = 0
    completed_benchmarks: int = 0
    total_planned_optimizations: int = 0
    completed_optimizations: int = 0

    current_config: str = ""
    current_optimization: str = ""

    baseline_results: list[dict] = field(default_factory=list)
    best_results: dict = field(default_factory=dict)  # scenario -> best result
    pareto_frontier: list[dict] = field(default_factory=list)
    pareto_changed: bool = False

    optimizations: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    gpu_hours: float = 0.0
    gpu_start_time: float = 0.0

    stagnant_rounds: int = 0
    suggest_stop: bool = False
    stop_reason: str = ""

    model: str = "GPT-OSS-120B"
    hardware: str = "MI355X"
    machine: str = ""

    pr_url: str = ""
    branch: str = ""


class ExperimentTracker:
    """
    Central tracker that maintains experiment state, computes Pareto frontier,
    and generates status files on every update.
    """

    STATE_DIR = Path("/app/experiment_status")
    FALLBACK_DIR = Path(".")  # for local dev

    def __init__(
        self,
        state_dir: Optional[str] = None,
        notify_callback=None,
    ):
        if state_dir:
            self.state_dir = Path(state_dir)
        elif os.path.isdir("/app"):
            self.state_dir = self.STATE_DIR
        else:
            self.state_dir = self.FALLBACK_DIR / "experiment_status"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state = ExperimentState()
        self._notify = notify_callback
        self._load_if_exists()

    # ── persistence ────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self.state_dir / "progress.json"

    def _load_if_exists(self):
        p = self._state_path()
        if p.exists():
            try:
                raw = json.loads(p.read_text())
                for k, v in raw.items():
                    if hasattr(self.state, k):
                        setattr(self.state, k, v)
            except Exception:
                pass

    def save(self):
        self.state.updated_at = time.time()
        self._state_path().write_text(
            json.dumps(asdict(self.state), indent=2, default=str)
        )
        self._write_status_md()
        self._write_summary_txt()

    # ── phase transitions ──────────────────────────────────────

    def set_phase(self, phase: Phase, detail: str = ""):
        old = self.state.phase
        self.state.phase = phase.value
        if old != phase.value:
            self._emit(EventType.PHASE_CHANGED, f"{old} -> {phase.value}: {detail}")
        self.save()

    # ── GPU time tracking ──────────────────────────────────────

    def gpu_start(self):
        self.state.gpu_start_time = time.time()

    def gpu_stop(self):
        if self.state.gpu_start_time > 0:
            elapsed_h = (time.time() - self.state.gpu_start_time) / 3600
            self.state.gpu_hours += elapsed_h
            self.state.gpu_start_time = 0

    # ── plan ───────────────────────────────────────────────────

    def plan(
        self,
        total_benchmarks: int,
        total_optimizations: int,
        model: str = "",
        hardware: str = "",
        machine: str = "",
        branch: str = "",
    ):
        self.state.total_planned_benchmarks = total_benchmarks
        self.state.total_planned_optimizations = total_optimizations
        if model:
            self.state.model = model
        if hardware:
            self.state.hardware = hardware
        if machine:
            self.state.machine = machine
        if branch:
            self.state.branch = branch
        self.save()

    # ── recording results ──────────────────────────────────────

    def record_benchmark(self, result: BenchResult, is_baseline: bool = False):
        rd = asdict(result)
        rd["timestamp"] = time.time()
        self.state.completed_benchmarks += 1
        self.state.current_config = result.scenario

        if is_baseline:
            self.state.baseline_results.append(rd)

        key = f"{result.scenario}"
        old_best = self.state.best_results.get(key)
        if old_best is None or result.throughput > old_best.get("throughput", 0):
            improved = old_best is not None
            self.state.best_results[key] = rd
            if improved:
                self._emit(
                    EventType.BEST_REFRESHED,
                    f"{key}: {old_best['throughput']:.1f} -> {result.throughput:.1f} tok/s",
                )

        pareto_changed = self._update_pareto(result)
        if pareto_changed:
            self.state.pareto_changed = True
            self._emit(
                EventType.NEW_PARETO_POINT,
                f"{result.scenario} c{result.concurrency}: "
                f"{result.throughput:.0f} tok/s, TPOT {result.tpot_mean:.1f}ms",
            )
        self.save()

    def record_batch_done(self, label: str, count: int):
        self._emit(
            EventType.BATCH_COMPLETED,
            f"Batch '{label}' done ({count} benchmarks, "
            f"{self.state.completed_benchmarks}/{self.state.total_planned_benchmarks} total)",
        )
        self.save()

    # ── optimizations ──────────────────────────────────────────

    def start_optimization(self, opt: OptimizationAttempt):
        opt.started_at = time.time()
        opt.status = "running"
        self.state.current_optimization = opt.name
        self.state.optimizations.append(asdict(opt))
        self._emit(EventType.OPT_APPLIED, f"Starting: {opt.name} — {opt.description}")
        self.save()

    def finish_optimization(self, name: str, status: str, error: str = ""):
        for o in self.state.optimizations:
            if o["name"] == name:
                o["status"] = status
                o["error"] = error
                o["finished_at"] = time.time()
                break
        self.state.completed_optimizations += 1
        if status == "success":
            self.state.stagnant_rounds = 0
        else:
            self.state.stagnant_rounds += 1
        self._check_early_stop()
        self.save()

    # ── Pareto frontier ────────────────────────────────────────

    def _update_pareto(self, result: BenchResult) -> bool:
        """
        Maintain a Pareto frontier on (throughput ↑, TPOT_mean ↓).
        Returns True if the frontier changed.
        """
        point = {
            "scenario": result.scenario,
            "concurrency": result.concurrency,
            "throughput": result.throughput,
            "tpot_mean": result.tpot_mean,
            "ttft_mean": result.ttft_mean,
            "label": result.label,
            "timestamp": time.time(),
        }
        old_frontier = copy.deepcopy(self.state.pareto_frontier)

        candidates = self.state.pareto_frontier + [point]
        # Filter by same scenario family for comparable frontier
        new_frontier = []
        for p in candidates:
            dominated = False
            for q in candidates:
                if p is q:
                    continue
                # q dominates p if q has higher throughput AND lower TPOT
                if (
                    q["throughput"] >= p["throughput"]
                    and q["tpot_mean"] <= p["tpot_mean"]
                    and (
                        q["throughput"] > p["throughput"]
                        or q["tpot_mean"] < p["tpot_mean"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                new_frontier.append(p)

        self.state.pareto_frontier = sorted(new_frontier, key=lambda x: x["throughput"])
        return len(new_frontier) != len(old_frontier) or any(
            p not in old_frontier for p in new_frontier
        )

    def get_pareto_shift(self) -> dict:
        """Compare current frontier to baseline, return shift metrics."""
        baseline_pts = [r for r in self.state.baseline_results]
        current_pts = self.state.pareto_frontier
        if not baseline_pts or not current_pts:
            return {"shift": "no_data"}

        bl_max_tput = max((r["throughput"] for r in baseline_pts), default=0)
        cur_max_tput = max((r["throughput"] for r in current_pts), default=0)
        bl_min_tpot = min((r["tpot_mean"] for r in baseline_pts), default=999)
        cur_min_tpot = min((r["tpot_mean"] for r in current_pts), default=999)

        return {
            "throughput_improvement_pct": (
                (cur_max_tput - bl_max_tput) / bl_max_tput * 100
                if bl_max_tput > 0
                else 0
            ),
            "tpot_improvement_pct": (
                (bl_min_tpot - cur_min_tpot) / bl_min_tpot * 100
                if bl_min_tpot > 0
                else 0
            ),
            "baseline_max_throughput": bl_max_tput,
            "current_max_throughput": cur_max_tput,
            "baseline_min_tpot": bl_min_tpot,
            "current_min_tpot": cur_min_tpot,
            "frontier_points": len(current_pts),
        }

    # ── early stop logic ───────────────────────────────────────

    def _check_early_stop(self):
        if self.state.stagnant_rounds >= 3:
            self.state.suggest_stop = True
            self.state.stop_reason = (
                f"{self.state.stagnant_rounds} consecutive optimizations "
                "showed no improvement"
            )
            self._emit(EventType.EARLY_STOP, self.state.stop_reason)

    # ── event emission ─────────────────────────────────────────

    def _emit(self, event_type: EventType, message: str):
        evt = {
            "type": event_type.value,
            "message": message,
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "progress_pct": self.progress_pct,
        }
        self.state.events.append(evt)
        # Keep only last 100 events in state
        if len(self.state.events) > 100:
            self.state.events = self.state.events[-100:]

        if self._notify:
            self._notify(evt)

    def emit_custom(self, event_type: EventType, message: str):
        self._emit(event_type, message)
        self.save()

    # ── computed properties ────────────────────────────────────

    @property
    def progress_pct(self) -> float:
        total = self.state.total_planned_benchmarks
        if total <= 0:
            return 0.0
        return min(100.0, self.state.completed_benchmarks / total * 100)

    @property
    def remaining_benchmarks(self) -> int:
        return max(
            0,
            self.state.total_planned_benchmarks - self.state.completed_benchmarks,
        )

    # ── status file generators ─────────────────────────────────

    def _write_status_md(self):
        s = self.state
        shift = self.get_pareto_shift()
        elapsed = time.time() - s.started_at
        elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.0f}m"

        lines = [
            "# Experiment Status",
            "",
            f"**Phase**: `{s.phase}`  ",
            f"**Progress**: {self.progress_pct:.0f}% "
            f"({s.completed_benchmarks}/{s.total_planned_benchmarks} benchmarks)  ",
            f"**Elapsed**: {elapsed_str}  ",
            f"**GPU Hours**: {s.gpu_hours:.2f}h  ",
            f"**Model**: {s.model} on {s.hardware}  ",
            f"**Machine**: `{s.machine}`  ",
            f"**Branch**: `{s.branch}`  ",
            f"**Last Updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
            "",
        ]

        if s.suggest_stop:
            lines += [f"> **SUGGEST STOP**: {s.stop_reason}", ""]

        if s.current_optimization:
            lines += ["## Current Optimization", f"`{s.current_optimization}`", ""]

        if s.best_results:
            lines += ["## Best Results", ""]
            lines.append("| Scenario | Throughput | TTFT mean | TPOT mean | Label |")
            lines.append("|---|---|---|---|---|")
            for k, r in sorted(s.best_results.items()):
                lines.append(
                    f"| {k} | {r['throughput']:.0f} tok/s "
                    f"| {r['ttft_mean']:.1f}ms "
                    f"| {r['tpot_mean']:.1f}ms "
                    f"| {r.get('label', '')} |"
                )
            lines.append("")

        if isinstance(shift, dict) and shift.get("shift") != "no_data":
            lines += [
                "## Pareto Frontier Shift",
                f"- Max throughput: {shift['baseline_max_throughput']:.0f} -> "
                f"{shift['current_max_throughput']:.0f} tok/s "
                f"(**{shift['throughput_improvement_pct']:+.1f}%**)",
                f"- Min TPOT: {shift['baseline_min_tpot']:.1f} -> "
                f"{shift['current_min_tpot']:.1f} ms "
                f"(**{shift['tpot_improvement_pct']:+.1f}%**)",
                f"- Frontier points: {shift['frontier_points']}",
                "",
            ]

        if s.optimizations:
            lines += ["## Optimization History", ""]
            lines.append("| # | Name | Status | Duration |")
            lines.append("|---|---|---|---|")
            for i, o in enumerate(s.optimizations, 1):
                dur = ""
                if o.get("finished_at") and o.get("started_at"):
                    dur = f"{(o['finished_at'] - o['started_at'])/60:.0f}m"
                lines.append(f"| {i} | {o['name']} | {o['status']} | {dur} |")
            lines.append("")

        if s.events:
            lines += ["## Recent Events", ""]
            for evt in s.events[-10:]:
                icon = {
                    "new_pareto_point": "***",
                    "best_refreshed": "++",
                    "early_stop_suggested": "!!",
                    "all_experiments_done": "==",
                    "no_progress": "--",
                }.get(evt["type"], ">")
                lines.append(
                    f"- `{evt['time_str']}` {icon} **{evt['type']}**: {evt['message']}"
                )
            lines.append("")

        (self.state_dir / "STATUS.md").write_text("\n".join(lines))

    def _write_summary_txt(self):
        s = self.state
        shift = self.get_pareto_shift()
        elapsed = time.time() - s.started_at

        text = [
            f"=== EXPERIMENT STATUS ({time.strftime('%H:%M:%S')}) ===",
            f"Phase:    {s.phase}",
            f"Progress: {self.progress_pct:.0f}% ({s.completed_benchmarks}/{s.total_planned_benchmarks})",
            f"Elapsed:  {elapsed/60:.0f}min | GPU: {s.gpu_hours:.2f}h",
            f"Current:  {s.current_optimization or s.current_config or 'idle'}",
            "",
        ]

        if s.best_results:
            text.append("--- Best Results ---")
            for k, r in sorted(s.best_results.items()):
                text.append(
                    f"  {k}: {r['throughput']:.0f} tok/s, "
                    f"TPOT {r['tpot_mean']:.1f}ms"
                )
            text.append("")

        if isinstance(shift, dict) and shift.get("shift") != "no_data":
            tp = shift["throughput_improvement_pct"]
            text.append(
                f"Pareto shift: throughput {tp:+.1f}%, "
                f"TPOT {shift['tpot_improvement_pct']:+.1f}%"
            )
            text.append("")

        if s.suggest_stop:
            text.append(f"!! SUGGEST STOP: {s.stop_reason}")
        else:
            remaining = self.remaining_benchmarks
            text.append(f"Remaining: ~{remaining} benchmarks")
            text.append("Recommend: continue")

        text.append("")
        if s.events:
            text.append(
                f"Latest: [{s.events[-1]['time_str']}] {s.events[-1]['message']}"
            )

        (self.state_dir / "latest_summary.txt").write_text("\n".join(text))

    # ── notification payload builder ───────────────────────────

    def build_notification(self, event: dict) -> dict:
        """Build a structured notification payload for external dispatch."""
        s = self.state
        shift = self.get_pareto_shift()
        best_tput = max((r["throughput"] for r in s.best_results.values()), default=0)
        best_tpot = min((r["tpot_mean"] for r in s.best_results.values()), default=0)

        return {
            "event_type": event["type"],
            "message": event["message"],
            "timestamp": event["timestamp"],
            "progress_pct": self.progress_pct,
            "phase": s.phase,
            "best_throughput": best_tput,
            "best_tpot": best_tpot,
            "pareto_changed": s.pareto_changed,
            "suggest_stop": s.suggest_stop,
            "gpu_hours": s.gpu_hours,
            "model": s.model,
            "hardware": s.hardware,
            "shift": shift if isinstance(shift, dict) else {},
            "next_step": self._next_step_hint(),
        }

    def _next_step_hint(self) -> str:
        s = self.state
        if s.suggest_stop:
            return "Consider stopping — diminishing returns"
        if s.phase == Phase.BASELINE.value:
            return "Running baseline benchmarks"
        if s.phase == Phase.OPTIMIZING.value:
            return f"Applying optimization: {s.current_optimization}"
        if s.phase == Phase.BENCHMARKING.value:
            return (
                f"Benchmarking ({s.completed_benchmarks}/"
                f"{s.total_planned_benchmarks})"
            )
        if s.phase == Phase.DONE.value:
            return "All done — review results and submit PR"
        return f"Phase: {s.phase}"
