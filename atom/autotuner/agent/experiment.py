"""
Experiment tracking and history management.

Each experiment is one iteration of the autoresearch loop.
The tracker maintains a persistent log of all experiments, enabling:
- Crash recovery (resume from last checkpoint)
- Result analysis (what mutations helped / hurt)
- Learning rate of the search process
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from atom.autotuner.types import (
    BenchmarkResult,
    Experiment,
    ExperimentStatus,
    InferenceConfig,
)

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Tracks all experiments in an autoresearch session.

    Experiments are written to a JSON-lines log in real time for crash recovery.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / "experiments.jsonl"
        self._experiments: list[Experiment] = []
        self._best: Optional[Experiment] = None

    @property
    def experiments(self) -> list[Experiment]:
        return list(self._experiments)

    @property
    def best(self) -> Optional[Experiment]:
        return self._best

    @property
    def completed_count(self) -> int:
        return sum(1 for e in self._experiments if e.status == ExperimentStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for e in self._experiments if e.status == ExperimentStatus.FAILED)

    def create(
        self,
        config: InferenceConfig,
        parent_id: Optional[str] = None,
        mutation: str = "",
    ) -> Experiment:
        """Create and register a new experiment."""
        exp = Experiment(
            config=config,
            parent_id=parent_id,
            mutation=mutation,
            status=ExperimentStatus.PENDING,
        )
        self._experiments.append(exp)
        self._write_log(exp)
        return exp

    def start(self, exp: Experiment) -> None:
        exp.status = ExperimentStatus.RUNNING
        self._write_log(exp)

    def complete(self, exp: Experiment, result: BenchmarkResult) -> None:
        exp.result = result
        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = time.time()
        self._write_log(exp)

        if exp.is_better_than(self._best):
            self._best = exp
            logger.info(
                "NEW BEST: exp %s → %.2f tok/s/gpu (mutation: %s)",
                exp.id, result.throughput_per_gpu, exp.mutation,
            )

    def fail(self, exp: Experiment, error: str) -> None:
        exp.status = ExperimentStatus.FAILED
        exp.error_message = error
        exp.completed_at = time.time()
        self._write_log(exp)

    def discard(self, exp: Experiment) -> None:
        exp.status = ExperimentStatus.DISCARDED
        exp.completed_at = time.time()
        self._write_log(exp)

    def get_improvement_rate(self, window: int = 10) -> float:
        """Fraction of recent experiments that improved over their parent."""
        recent = [
            e for e in self._experiments[-window:]
            if e.status == ExperimentStatus.COMPLETED and e.parent_id
        ]
        if not recent:
            return 0.0
        improved = sum(1 for e in recent if self._improved_over_parent(e))
        return improved / len(recent)

    def get_timeline(self) -> list[dict]:
        """Return experiment timeline for visualization."""
        timeline = []
        for e in self._experiments:
            if e.status != ExperimentStatus.COMPLETED or e.result is None:
                continue
            timeline.append({
                "id": e.id,
                "elapsed_sec": e.duration_sec(),
                "throughput_per_gpu": e.result.throughput_per_gpu,
                "ttft_ms": e.result.ttft_ms,
                "tpot_ms": e.result.tpot_ms,
                "mutation": e.mutation,
                "is_best": e.id == (self._best.id if self._best else ""),
            })
        return timeline

    def format_summary(self) -> str:
        lines = [
            "=" * 60,
            "Experiment Summary",
            "=" * 60,
            f"  Total experiments: {len(self._experiments)}",
            f"  Completed: {self.completed_count}",
            f"  Failed: {self.failed_count}",
            f"  Improvement rate (last 10): {self.get_improvement_rate():.1%}",
        ]
        if self._best and self._best.result:
            r = self._best.result
            lines.extend([
                "",
                "  Best Configuration:",
                f"    Throughput/GPU:  {r.throughput_per_gpu:.2f} tok/s/gpu",
                f"    Throughput/User: {r.throughput_per_user:.2f} tok/s/user",
                f"    TTFT:           {r.ttft_ms:.2f} ms",
                f"    TPOT:           {r.tpot_ms:.2f} ms",
                f"    Config:         tp{r.config.tp} pp{r.config.pp} bs{r.config.batch_size}",
                f"                    quant={r.config.quant_format} kv={r.config.kv_cache_dtype}",
                f"                    disagg={r.config.disagg}",
            ])
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """Save full tracker state for crash recovery."""
        path = path or self.log_dir / "checkpoint.json"
        data = {
            "experiments": [self._exp_to_dict(e) for e in self._experiments],
            "best_id": self._best.id if self._best else None,
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, path: Optional[Path] = None) -> int:
        """Load tracker state from checkpoint. Returns number of experiments loaded."""
        path = path or self.log_dir / "checkpoint.json"
        if not path.exists():
            return 0

        data = json.loads(path.read_text())
        self._experiments = []
        best_id = data.get("best_id")

        for ed in data.get("experiments", []):
            exp = Experiment(
                id=ed["id"],
                config=InferenceConfig(**ed.get("config", {"model": ""})),
                status=ExperimentStatus(ed.get("status", "pending")),
                parent_id=ed.get("parent_id"),
                mutation=ed.get("mutation", ""),
                created_at=ed.get("created_at", 0),
                completed_at=ed.get("completed_at"),
            )
            if ed.get("result"):
                exp.result = BenchmarkResult(
                    config=exp.config,
                    ttft_ms=ed["result"].get("ttft_ms", 0),
                    tpot_ms=ed["result"].get("tpot_ms", 0),
                    throughput_tokens_per_sec=ed["result"].get("throughput_tokens_per_sec", 0),
                    throughput_per_gpu=ed["result"].get("throughput_per_gpu", 0),
                    throughput_per_user=ed["result"].get("throughput_per_user", 0),
                    request_latency_ms=ed["result"].get("request_latency_ms", 0),
                )
            self._experiments.append(exp)
            if best_id and exp.id == best_id:
                self._best = exp

        logger.info("Loaded %d experiments from checkpoint", len(self._experiments))
        return len(self._experiments)

    def _improved_over_parent(self, exp: Experiment) -> bool:
        if not exp.parent_id or not exp.result:
            return False
        parent = next((e for e in self._experiments if e.id == exp.parent_id), None)
        if parent is None or parent.result is None:
            return False
        return exp.result.throughput_per_gpu > parent.result.throughput_per_gpu

    def _write_log(self, exp: Experiment) -> None:
        with open(self._log_path, "a") as f:
            f.write(json.dumps(self._exp_to_dict(exp)) + "\n")

    def _exp_to_dict(self, exp: Experiment) -> dict:
        from dataclasses import asdict
        d = {
            "id": exp.id,
            "config": asdict(exp.config) if exp.config else {},
            "status": exp.status.value,
            "parent_id": exp.parent_id,
            "mutation": exp.mutation,
            "created_at": exp.created_at,
            "completed_at": exp.completed_at,
            "error_message": exp.error_message,
        }
        if exp.result:
            d["result"] = {
                "ttft_ms": exp.result.ttft_ms,
                "tpot_ms": exp.result.tpot_ms,
                "throughput_tokens_per_sec": exp.result.throughput_tokens_per_sec,
                "throughput_per_gpu": exp.result.throughput_per_gpu,
                "throughput_per_user": exp.result.throughput_per_user,
                "request_latency_ms": exp.result.request_latency_ms,
                "memory_used_gb": exp.result.memory_used_gb,
            }
        return d
