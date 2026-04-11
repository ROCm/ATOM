"""
Abstract inference adapter interface.

Any LLM inference framework (ATOM, vLLM, SGLang, TensorRT-LLM) can be plugged
into the autotuner by implementing this interface.  The adapter handles:
1. Deploying a model with a given configuration
2. Running a benchmark and collecting metrics
3. Cleaning up after the benchmark
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

from atom.autotuner.types import BenchmarkResult, GPUInfo, InferenceConfig

logger = logging.getLogger(__name__)


class InferenceAdapter(ABC):
    """
    Abstract interface for inference framework integration.

    Implementors must provide deploy(), benchmark(), get_gpu_info().
    Common server lifecycle helpers are provided as static/class methods.
    """

    @abstractmethod
    def deploy(self, config: InferenceConfig) -> None:
        """Deploy the model with the specified configuration."""

    @abstractmethod
    def benchmark(
        self,
        config: InferenceConfig,
        duration_sec: int = 60,
        concurrency: int = 32,
        isl: int = 4000,
        osl: int = 1000,
    ) -> BenchmarkResult:
        """Run a benchmark and return results."""

    @abstractmethod
    def teardown(self) -> None:
        """Stop the serving instance and free resources."""

    @abstractmethod
    def get_gpu_info(self) -> GPUInfo:
        """Query the GPU hardware info."""

    def run_full(
        self,
        config: InferenceConfig,
        duration_sec: int = 60,
        concurrency: int = 32,
    ) -> BenchmarkResult:
        """Deploy -> benchmark -> teardown in one call."""
        try:
            self.deploy(config)
            return self.benchmark(config, duration_sec, concurrency)
        finally:
            self.teardown()

    def health_check(self) -> bool:
        """Return True if the serving instance is healthy and GPU is loaded."""
        return False

    # ------------------------------------------------------------------
    # Shared helpers for server-based adapters
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_benchmark_output(
        output: str, config: InferenceConfig
    ) -> BenchmarkResult:
        """Parse common benchmark tool output (ATOM / vLLM / SGLang) into metrics."""
        result = BenchmarkResult(config=config)
        for line in output.splitlines():
            ll = line.lower()
            if "ttft" in ll:
                m = re.search(r"([\d.]+)\s*ms", line)
                if m:
                    result.ttft_ms = float(m.group(1))
            if "tpot" in ll or "itl" in ll:
                m = re.search(r"([\d.]+)\s*ms", line)
                if m:
                    result.tpot_ms = float(m.group(1))
            if "throughput" in ll and "tok" in ll:
                m = re.search(r"([\d.]+)\s*tok", line)
                if m:
                    result.throughput_tokens_per_sec = float(m.group(1))

        total_gpus = config.total_gpus_used()
        result.throughput_per_gpu = (
            result.throughput_tokens_per_sec / max(total_gpus, 1)
        )
        if result.tpot_ms > 0:
            result.throughput_per_user = 1000.0 / result.tpot_ms
        return result

    @staticmethod
    def _http_health_check(host: str, port: int) -> bool:
        """HTTP GET /health probe."""
        try:
            resp = urllib.request.urlopen(
                f"http://{host}:{port}/health", timeout=5
            )
            return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _wait_for_server(
        proc: subprocess.Popen,
        check_fn,
        timeout: int = 300,
        interval: int = 5,
    ) -> bool:
        """Block until *check_fn()* returns True or *proc* exits."""
        start = time.time()
        while time.time() - start < timeout:
            if proc.poll() is not None:
                logger.error("Server process exited prematurely")
                return False
            if check_fn():
                return True
            time.sleep(interval)
        return False

    @staticmethod
    def _terminate_proc(
        proc: Optional[subprocess.Popen], timeout: int = 30
    ) -> None:
        """Gracefully terminate a subprocess, falling back to kill."""
        if proc is None:
            return
        logger.info("Shutting down server (pid=%d)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
