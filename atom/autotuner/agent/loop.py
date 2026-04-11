"""
Autoresearch-style agent loop for kernel autotuning.

Inspired by Karpathy's autoresearch: the agent runs an autonomous loop of
propose → benchmark → evaluate → keep/discard → repeat.

Key differences from autoresearch:
- Instead of modifying training code, we modify *inference configuration*
- Instead of val_bpb, our metric is throughput_per_gpu (and TTFT/TPOT under SLA)
- We maintain a Pareto frontier, not just a single best
- The search is guided by a performance model + optional LLM agent reasoning

The loop supports three evaluation modes:
1. MODEL_ONLY:   use the E2E estimator (fast, ~ms per eval, no GPU needed)
2. REAL_BENCH:   actually deploy + benchmark (slow, ~minutes per eval)
3. HYBRID_EVAL:  model-guided pre-screening → top-K go to real benchmark
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from atom.autotuner.types import (
    BenchmarkResult,
    ExperimentStatus,
    GPUInfo,
    InferenceConfig,
    TunerState,
)
from atom.autotuner.agent.experiment import ExperimentTracker
from atom.autotuner.database.estimator import E2EEstimator, ModelArch
from atom.autotuner.database.perf_model import PerformanceModel
from atom.autotuner.search.pareto import ParetoAnalyzer
from atom.autotuner.search.space import ConfigSpace, SearchBounds
from atom.autotuner.search.strategies import AgentGuidedSearch, BayesianSearch, GridSearch

logger = logging.getLogger(__name__)


class EvalMode(Enum):
    MODEL_ONLY = "model_only"
    REAL_BENCH = "real_bench"
    HYBRID_EVAL = "hybrid_eval"


@dataclass
class LoopConfig:
    """Configuration for the agent loop."""
    budget_sec: int = 3600
    max_experiments: int = 500
    eval_mode: EvalMode = EvalMode.MODEL_ONLY
    checkpoint_interval_sec: int = 300
    strategy: str = "agent_guided"
    ttft_limit_ms: Optional[float] = None
    tpot_limit_ms: Optional[float] = None
    hybrid_topk: int = 10
    log_dir: Path = Path("autotuner_results")


class AgentLoop:
    """
    Main orchestrator for the autonomous tuning loop.

    Usage::

        loop = AgentLoop(
            model_arch=ModelArch.from_hf_config("gpt-oss-120b"),
            gpu_info=GPUInfo.mi355x(num_gpus=8),
            total_gpus=8,
            loop_config=LoopConfig(budget_sec=1800),
            perf_model=perf_model,
        )
        results = loop.run()
        print(results.format_summary())
    """

    def __init__(
        self,
        model_arch: ModelArch,
        gpu_info: GPUInfo,
        total_gpus: int,
        loop_config: LoopConfig,
        perf_model: PerformanceModel,
        real_bench_fn: Optional[Callable[[InferenceConfig], BenchmarkResult]] = None,
    ):
        self.arch = model_arch
        self.gpu = gpu_info
        self.total_gpus = total_gpus
        self.config = loop_config
        self.perf_model = perf_model
        self.real_bench_fn = real_bench_fn

        self.estimator = E2EEstimator(perf_model, gpu_info)
        self.tracker = ExperimentTracker(loop_config.log_dir)
        self.pareto = ParetoAnalyzer(
            ttft_limit_ms=loop_config.ttft_limit_ms,
            tpot_limit_ms=loop_config.tpot_limit_ms,
        )
        self.space = ConfigSpace(
            model_arch=model_arch,
            gpu_info=gpu_info,
            total_gpus=total_gpus,
        )

        self._stop_requested = False
        self._state: Optional[TunerState] = None

    def run(self) -> ExperimentTracker:
        """
        Run the full autoresearch loop.

        Returns the experiment tracker with all results.
        """
        self._setup_signal_handlers()
        start_time = time.time()
        self._state = TunerState(model=self.arch.name, system=self.gpu.name)

        resumed = self.tracker.load_checkpoint()
        if resumed:
            logger.info("Resumed from checkpoint with %d experiments", resumed)

        logger.info(
            "Starting autoresearch loop: model=%s, gpus=%d×%s, budget=%ds, strategy=%s",
            self.arch.name, self.total_gpus, self.gpu.name,
            self.config.budget_sec, self.config.strategy,
        )

        strategy = self._build_strategy()
        evaluate_fn = self._build_evaluate_fn()

        last_checkpoint = time.time()

        try:
            results = strategy.search(
                space=self.space,
                evaluate_fn=evaluate_fn,
                budget=self.config.max_experiments,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted by user — saving checkpoint")
            self._save_state()
            return self.tracker
        except Exception:
            logger.exception("Agent loop failed — saving checkpoint")
            self._save_state()
            raise

        for r in results:
            self.pareto.add_result(r)

        if (self.config.eval_mode == EvalMode.HYBRID_EVAL
                and self.real_bench_fn is not None):
            self._run_hybrid_verification(results)

        self._save_state()
        self._print_final_report()
        return self.tracker

    def _build_strategy(self):
        if self.config.strategy == "grid":
            return GridSearch()
        if self.config.strategy == "bayesian":
            return BayesianSearch()
        return AgentGuidedSearch()

    def _build_evaluate_fn(self) -> Callable[[InferenceConfig], BenchmarkResult]:
        """Build the evaluation function based on eval mode."""
        if self.config.eval_mode == EvalMode.REAL_BENCH and self.real_bench_fn:
            return self._eval_real

        return self._eval_model

    def _eval_model(self, config: InferenceConfig) -> BenchmarkResult:
        """Evaluate via the performance model (fast, no GPU needed)."""
        exp = self.tracker.create(config, mutation="model_eval")
        self.tracker.start(exp)

        try:
            result = self.estimator.estimate(config, self.arch)
            self.tracker.complete(exp, result)
            return result
        except Exception as e:
            self.tracker.fail(exp, str(e))
            raise

    def _eval_real(self, config: InferenceConfig) -> BenchmarkResult:
        """Evaluate via real GPU benchmark (slow but accurate)."""
        exp = self.tracker.create(config, mutation="real_bench")
        self.tracker.start(exp)

        try:
            result = self.real_bench_fn(config)
            self.tracker.complete(exp, result)
            return result
        except Exception as e:
            self.tracker.fail(exp, str(e))
            raise

    def _run_hybrid_verification(self, model_results: list[BenchmarkResult]) -> None:
        """
        Hybrid mode: verify top-K model predictions with real benchmarks.

        This addresses the accuracy concern (Q15): the model might predict
        incorrectly for some configurations.  By verifying the top candidates,
        we get real-world confirmation of the best configs.
        """
        if not self.real_bench_fn:
            return

        model_results.sort(key=lambda r: r.throughput_per_gpu, reverse=True)
        top_k = model_results[:self.config.hybrid_topk]

        logger.info("Hybrid verification: benchmarking top-%d configs on real GPU", len(top_k))

        for i, model_result in enumerate(top_k):
            try:
                real_result = self.real_bench_fn(model_result.config)
                self.pareto.add_result(real_result)

                model_pred = model_result.throughput_per_gpu
                real_val = real_result.throughput_per_gpu
                error_pct = abs(model_pred - real_val) / max(real_val, 0.01) * 100

                logger.info(
                    "  Config %d: model=%.1f, real=%.1f tok/s/gpu (error=%.1f%%)",
                    i + 1, model_pred, real_val, error_pct,
                )
            except Exception:
                logger.exception("Real benchmark failed for config %d", i + 1)

    def _save_state(self) -> None:
        """Save checkpoint for crash recovery."""
        self.tracker.save_checkpoint()
        if self._state:
            self._state.last_checkpoint = time.time()
            self._state.all_experiments = self.tracker.experiments
            self._state.best_experiment = self.tracker.best
            self._state.pareto_frontier = self.pareto.compute_frontier()
            self._state.save(self.config.log_dir / "tuner_state.json")
        logger.info("State saved to %s", self.config.log_dir)

    def _print_final_report(self) -> None:
        """Print the final summary report."""
        print("\n" + "=" * 80)
        print("  ROCm Autotuner — Final Results")
        print("=" * 80)
        print(self.tracker.format_summary())
        print()
        print(self.pareto.format_frontier())
        print()
        print(self.pareto.format_ascii_chart())
        print("=" * 80)

    def _setup_signal_handlers(self) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        def _handler(signum, frame):
            logger.info("Signal %d received — stopping after current experiment", signum)
            self._stop_requested = True

        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError):
            pass
