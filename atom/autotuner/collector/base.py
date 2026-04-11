"""Abstract base for kernel micro-benchmark collectors."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from atom.autotuner.types import (
    GPUInfo,
    KernelBenchResult,
    KernelConfig,
    KernelType,
)

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """
    Template for collecting kernel-level performance data on AMD GPUs.

    Each subclass targets one kernel family (GEMM, Attention, …).
    The collector manages warm-up, repetition, outlier filtering, and
    GPU state control (clock locking, power mode) via *GPUStateManager*.

    Design note (addresses Q1 / Q4 from the AIConfigurator review):
    - Parameter space sampling is LLM-workload-informed, not uniform grid.
      Each subclass defines ``_build_sweep_configs`` which picks (m, n, k) etc.
      from shapes that actually arise during inference for common model families.
    - GPU state is pinned via ``rocm-smi --setperflevel high`` before collection
      and restored afterwards.
    """

    kernel_type: KernelType

    def __init__(
        self,
        gpu_info: GPUInfo,
        warmup_iters: int = 10,
        bench_iters: int = 100,
        cooldown_sec: float = 0.5,
    ):
        self.gpu_info = gpu_info
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.cooldown_sec = cooldown_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_all(self, configs: Sequence[KernelConfig] | None = None) -> list[KernelBenchResult]:
        """Run the full sweep and return results."""
        if configs is None:
            configs = self._build_sweep_configs()

        logger.info(
            "Collecting %d %s benchmarks (warmup=%d, iters=%d)",
            len(configs),
            self.kernel_type.value,
            self.warmup_iters,
            self.bench_iters,
        )

        results: list[KernelBenchResult] = []
        for i, cfg in enumerate(configs):
            try:
                res = self._bench_one(cfg)
                results.append(res)
                if (i + 1) % 50 == 0:
                    logger.info("  … %d / %d done", i + 1, len(configs))
            except Exception:
                logger.exception("Benchmark failed for %s", cfg.params)
            finally:
                if self.cooldown_sec > 0:
                    time.sleep(self.cooldown_sec)

        logger.info(
            "Collected %d / %d %s results",
            len(results),
            len(configs),
            self.kernel_type.value,
        )
        return results

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_sweep_configs(self) -> list[KernelConfig]:
        """Generate the parameter-space sweep for this kernel family."""

    @abstractmethod
    def _bench_one(self, config: KernelConfig) -> KernelBenchResult:
        """Run a single micro-benchmark and return the result."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _llm_workload_m_values() -> list[int]:
        """
        Typical M dimensions that arise during LLM inference.

        Prefill: M = seq_len (128 … 32768)
        Decode:  M = batch_size (1 … 512)
        """
        prefill = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        decode = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        return sorted(set(prefill + decode))

    def save_results(self, results: list[KernelBenchResult], path: Path) -> None:
        """Persist results as JSON lines."""
        import json
        from dataclasses import asdict

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in results:
                row = {
                    "kernel_type": r.config.kernel_type.value,
                    "params": r.config.params,
                    "latency_us": r.latency_us,
                    "throughput_tflops": r.throughput_tflops,
                    "memory_bw_gbps": r.memory_bw_gbps,
                    "power_watts": r.power_watts,
                    "gpu_util_pct": r.gpu_util_pct,
                    "timestamp": r.timestamp,
                }
                f.write(json.dumps(row) + "\n")
        logger.info("Saved %d results to %s", len(results), path)
