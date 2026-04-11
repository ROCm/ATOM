"""
Search strategies for configuration optimization.

Three strategies:
1. GridSearch     — exhaustive enumeration + evaluation (baseline)
2. BayesianSearch — Gaussian-process-guided search for expensive evaluations
3. AgentGuidedSearch — autoresearch-style: LLM agent proposes next config
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

from atom.autotuner.types import BenchmarkResult, InferenceConfig
from atom.autotuner.search.space import ConfigSpace

logger = logging.getLogger(__name__)


class SearchBase(ABC):
    """Abstract search strategy."""

    @abstractmethod
    def search(
        self,
        space: ConfigSpace,
        evaluate_fn: Callable[[InferenceConfig], BenchmarkResult],
        budget: int = 100,
    ) -> list[BenchmarkResult]:
        """Run the search and return all evaluated results."""


class GridSearch(SearchBase):
    """
    Exhaustive grid search over the configuration space.

    Fast for small spaces (< 1000 configs); for larger spaces, randomly
    samples up to ``budget`` configurations.
    """

    def search(
        self,
        space: ConfigSpace,
        evaluate_fn: Callable[[InferenceConfig], BenchmarkResult],
        budget: int = 100,
    ) -> list[BenchmarkResult]:
        configs = list(space.enumerate())
        logger.info("GridSearch: %d total configs, budget=%d", len(configs), budget)

        if len(configs) > budget:
            configs = random.sample(configs, budget)
            logger.info("Randomly sampled %d configs", budget)

        results = []
        for i, cfg in enumerate(configs):
            try:
                result = evaluate_fn(cfg)
                results.append(result)
            except Exception:
                logger.exception("Evaluation failed for config %d", i)

            if (i + 1) % 100 == 0:
                logger.info("GridSearch progress: %d / %d", i + 1, len(configs))

        logger.info("GridSearch complete: %d results", len(results))
        return results


class BayesianSearch(SearchBase):
    """
    Bayesian optimization for configuration search.

    Uses a surrogate model (Gaussian Process) to predict the objective
    (throughput_per_gpu) and an acquisition function (Expected Improvement)
    to select the next configuration to evaluate.

    Particularly effective when each evaluation is expensive (real GPU benchmark).
    """

    def __init__(self, exploration_weight: float = 1.0, seed: int = 42):
        self.exploration_weight = exploration_weight
        self.seed = seed

    def search(
        self,
        space: ConfigSpace,
        evaluate_fn: Callable[[InferenceConfig], BenchmarkResult],
        budget: int = 50,
    ) -> list[BenchmarkResult]:
        random.seed(self.seed)
        all_configs = list(space.enumerate())
        if not all_configs:
            return []

        logger.info("BayesianSearch: %d candidate configs, budget=%d", len(all_configs), budget)

        n_initial = min(max(budget // 5, 5), len(all_configs))
        initial_configs = random.sample(all_configs, n_initial)

        results = []
        for cfg in initial_configs:
            try:
                result = evaluate_fn(cfg)
                results.append(result)
            except Exception:
                pass

        remaining_budget = budget - len(results)
        remaining_configs = [c for c in all_configs if c.fingerprint() not in
                            {r.config.fingerprint() for r in results}]

        for step in range(remaining_budget):
            if not remaining_configs:
                break

            next_cfg = self._select_next(results, remaining_configs)
            try:
                result = evaluate_fn(next_cfg)
                results.append(result)
            except Exception:
                pass

            remaining_configs = [c for c in remaining_configs if
                                c.fingerprint() != next_cfg.fingerprint()]

            if (step + 1) % 10 == 0:
                best = max(results, key=lambda r: r.throughput_per_gpu)
                logger.info(
                    "BayesianSearch step %d/%d, best=%.2f tok/s/gpu",
                    step + 1, remaining_budget, best.throughput_per_gpu,
                )

        logger.info("BayesianSearch complete: %d results", len(results))
        return results

    def _select_next(
        self,
        results: list[BenchmarkResult],
        candidates: list[InferenceConfig],
    ) -> InferenceConfig:
        """
        Select next config using a simplified acquisition function.

        For a full GP-based approach, we'd use scikit-learn's GaussianProcessRegressor.
        Here we use a simpler heuristic: score based on similarity to best configs
        with diversity bonus.
        """
        if not results:
            return random.choice(candidates)

        best = max(results, key=lambda r: r.throughput_per_gpu)
        best_cfg = best.config

        def _score(cfg: InferenceConfig) -> float:
            similarity = 0.0
            if cfg.tp == best_cfg.tp:
                similarity += 0.3
            if cfg.pp == best_cfg.pp:
                similarity += 0.2
            if cfg.quant_format == best_cfg.quant_format:
                similarity += 0.15
            if cfg.kv_cache_dtype == best_cfg.kv_cache_dtype:
                similarity += 0.1

            bs_dist = abs(cfg.batch_size - best_cfg.batch_size) / max(best_cfg.batch_size, 1)
            exploration = min(bs_dist, 2.0) * self.exploration_weight * 0.25

            return similarity + exploration + random.gauss(0, 0.1)

        scored = [(c, _score(c)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0]


class AgentGuidedSearch(SearchBase):
    """
    LLM-agent-guided search inspired by Karpathy's autoresearch.

    The agent:
    1. Reviews the history of experiments and their results
    2. Proposes a mutation to the best-known config
    3. The mutation is evaluated
    4. If better, it becomes the new best; if worse, it's logged and we continue

    Mutations include: change TP, change batch size, toggle disagg mode,
    switch quant format, adjust PP, etc.

    This strategy is most powerful when combined with real GPU benchmarks,
    as the agent can reason about *why* certain configurations work better.
    """

    MUTATION_TYPES = [
        "increase_tp",
        "decrease_tp",
        "increase_pp",
        "decrease_pp",
        "increase_batch",
        "decrease_batch",
        "toggle_disagg",
        "change_quant",
        "change_kv_dtype",
        "increase_prefill_workers",
        "increase_decode_workers",
        "change_ep",
    ]

    def __init__(self, mutation_rate: float = 0.3, seed: int = 42):
        self.mutation_rate = mutation_rate
        self.seed = seed

    def search(
        self,
        space: ConfigSpace,
        evaluate_fn: Callable[[InferenceConfig], BenchmarkResult],
        budget: int = 50,
    ) -> list[BenchmarkResult]:
        random.seed(self.seed)
        logger.info("AgentGuidedSearch: budget=%d iterations", budget)

        configs = list(space.enumerate())
        if not configs:
            return []

        current = random.choice(configs)
        try:
            result = evaluate_fn(current)
        except Exception:
            return []

        results = [result]
        best_result = result
        stagnation = 0

        for step in range(budget - 1):
            n_mutations = max(1, int(random.expovariate(1 / 2)))
            candidate = self._mutate(best_result.config, space, n_mutations)

            try:
                result = evaluate_fn(candidate)
                results.append(result)
            except Exception:
                continue

            if result.throughput_per_gpu > best_result.throughput_per_gpu:
                improvement = (
                    (result.throughput_per_gpu - best_result.throughput_per_gpu)
                    / max(best_result.throughput_per_gpu, 0.01) * 100
                )
                logger.info(
                    "Step %d: NEW BEST %.2f tok/s/gpu (+%.1f%%) via %s",
                    step + 1, result.throughput_per_gpu, improvement,
                    self._describe_diff(best_result.config, candidate),
                )
                best_result = result
                stagnation = 0
            else:
                stagnation += 1

            if stagnation > budget // 4:
                logger.info("Stagnation detected, increasing exploration")
                candidate = random.choice(configs)
                try:
                    result = evaluate_fn(candidate)
                    results.append(result)
                    if result.throughput_per_gpu > best_result.throughput_per_gpu:
                        best_result = result
                except Exception:
                    pass
                stagnation = 0

        logger.info(
            "AgentGuidedSearch complete: %d results, best=%.2f tok/s/gpu",
            len(results), best_result.throughput_per_gpu,
        )
        return results

    def _mutate(
        self, config: InferenceConfig, space: ConfigSpace, n_mutations: int = 1
    ) -> InferenceConfig:
        """Apply random mutations to a configuration."""
        import copy
        cfg = copy.deepcopy(config)

        mutations = random.sample(
            self.MUTATION_TYPES, min(n_mutations, len(self.MUTATION_TYPES))
        )

        for mut in mutations:
            if mut == "increase_tp" and cfg.tp * 2 in space.bounds.tp_values:
                cfg.tp *= 2
            elif mut == "decrease_tp" and cfg.tp // 2 in space.bounds.tp_values:
                cfg.tp //= 2
            elif mut == "increase_pp" and cfg.pp * 2 in space.bounds.pp_values:
                cfg.pp *= 2
            elif mut == "decrease_pp" and cfg.pp // 2 in space.bounds.pp_values:
                cfg.pp //= 2
            elif mut == "increase_batch":
                idx = space.bounds.batch_sizes.index(cfg.batch_size) if cfg.batch_size in space.bounds.batch_sizes else 0
                if idx + 1 < len(space.bounds.batch_sizes):
                    cfg.batch_size = space.bounds.batch_sizes[idx + 1]
            elif mut == "decrease_batch":
                idx = space.bounds.batch_sizes.index(cfg.batch_size) if cfg.batch_size in space.bounds.batch_sizes else 0
                if idx > 0:
                    cfg.batch_size = space.bounds.batch_sizes[idx - 1]
            elif mut == "toggle_disagg":
                cfg.disagg = not cfg.disagg
                if cfg.disagg:
                    cfg.prefill_workers = random.choice(space.bounds.prefill_worker_counts)
                    cfg.decode_workers = random.choice(space.bounds.decode_worker_counts)
            elif mut == "change_quant":
                cfg.quant_format = random.choice(space.bounds.quant_formats)
            elif mut == "change_kv_dtype":
                cfg.kv_cache_dtype = random.choice(space.bounds.kv_cache_dtypes)
            elif mut == "change_ep" and space.arch.is_moe:
                cfg.ep = random.choice(space.bounds.ep_values)

        return cfg

    def _describe_diff(self, old: InferenceConfig, new: InferenceConfig) -> str:
        """Human-readable description of what changed."""
        diffs = []
        if old.tp != new.tp:
            diffs.append(f"tp:{old.tp}→{new.tp}")
        if old.pp != new.pp:
            diffs.append(f"pp:{old.pp}→{new.pp}")
        if old.batch_size != new.batch_size:
            diffs.append(f"bs:{old.batch_size}→{new.batch_size}")
        if old.disagg != new.disagg:
            diffs.append(f"disagg:{old.disagg}→{new.disagg}")
        if old.quant_format != new.quant_format:
            diffs.append(f"quant:{old.quant_format}→{new.quant_format}")
        if old.kv_cache_dtype != new.kv_cache_dtype:
            diffs.append(f"kv:{old.kv_cache_dtype}→{new.kv_cache_dtype}")
        return ", ".join(diffs) if diffs else "no change"
