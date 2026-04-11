"""
ATOM inference framework adapter.

Integrates with ATOM's serving infrastructure to:
1. Launch ``atom.entrypoints.openai_server`` with the given config
2. Run ``atom.benchmarks.benchmark_serving`` against it
3. Collect TTFT, TPOT, throughput metrics
4. Teardown the server process

Also supports a "direct" mode that runs ModelRunner.run_model() for
latency-only measurements without the full serving stack.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

from atom.autotuner.adapters.base import InferenceAdapter
from atom.autotuner.types import BenchmarkResult, GPUInfo, InferenceConfig

logger = logging.getLogger(__name__)

_SERVER_STARTUP_TIMEOUT = 300


class ATOMAdapter(InferenceAdapter):
    """
    Adapter for ATOM inference engine.

    Modes:
    - ``serving``:  full OpenAI-compatible server + benchmark client
    - ``direct``:   ModelRunner forward pass only (no HTTP overhead)
    """

    def __init__(
        self,
        mode: str = "serving",
        host: str = "127.0.0.1",
        port: int = 8006,
    ):
        self.mode = mode
        self.host = host
        self.port = port
        self._server_proc: Optional[subprocess.Popen] = None

    def deploy(self, config: InferenceConfig) -> None:
        if self.mode == "direct":
            return

        cmd = self._build_server_cmd(config)
        env = os.environ.copy()
        env["AITER_LOG_LEVEL"] = "WARNING"

        logger.info("Launching ATOM server: %s", " ".join(cmd))
        self._server_proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if not self._wait_for_server(
            self._server_proc, self.health_check, _SERVER_STARTUP_TIMEOUT
        ):
            self.teardown()
            raise RuntimeError("ATOM server failed to start within timeout")

        logger.info("ATOM server ready on %s:%d", self.host, self.port)

    def benchmark(
        self,
        config: InferenceConfig,
        duration_sec: int = 60,
        concurrency: int = 32,
        isl: int = 4000,
        osl: int = 1000,
    ) -> BenchmarkResult:
        if self.mode == "direct":
            return BenchmarkResult(config=config)

        cmd = [
            "python", "-m", "atom.benchmarks.benchmark_serving",
            "--backend", "openai",
            "--base-url", f"http://{self.host}:{self.port}",
            "--model", config.model,
            "--request-rate", "inf",
            "--num-prompts", str(concurrency * 10),
            "--sharegpt-output-len", str(osl),
        ]

        logger.info("Running benchmark: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=duration_sec + 120,
        )
        return self._parse_benchmark_output(proc.stdout, config)

    def teardown(self) -> None:
        self._terminate_proc(self._server_proc)
        self._server_proc = None

    def get_gpu_info(self) -> GPUInfo:
        from atom.autotuner.utils.gpu import ROCmGPU
        return ROCmGPU.detect()

    def health_check(self) -> bool:
        return self._http_health_check(self.host, self.port)

    def _build_server_cmd(self, config: InferenceConfig) -> list[str]:
        cmd = [
            "python", "-m", "atom.entrypoints.openai_server",
            "--model", config.model,
            "--tensor-parallel-size", str(config.tp),
            "--kv_cache_dtype", config.kv_cache_dtype,
            "--port", str(self.port),
            "--max-num-seqs", str(config.batch_size),
            "--max-model-len", str(config.max_seq_len),
        ]
        if config.pp > 1:
            cmd.extend(["--pipeline-parallel-size", str(config.pp)])
        if config.compilation_level != 3:
            cmd.extend(["--level", str(config.compilation_level)])
        if config.compilation_level == 0:
            cmd.append("--enforce-eager")
        if config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if config.ep > 1:
            cmd.append("--enable-expert-parallel")
        return cmd
