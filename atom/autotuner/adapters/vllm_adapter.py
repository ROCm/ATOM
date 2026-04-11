"""
vLLM inference framework adapter.

Enables the autotuner to optimize vLLM deployments on AMD GPUs.
Uses vLLM's OpenAI-compatible server and benchmark_serving script.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

from atom.autotuner.adapters.base import InferenceAdapter
from atom.autotuner.types import BenchmarkResult, GPUInfo, InferenceConfig

logger = logging.getLogger(__name__)


class VLLMAdapter(InferenceAdapter):
    """Adapter for vLLM inference engine."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self._server_proc: Optional[subprocess.Popen] = None

    def deploy(self, config: InferenceConfig) -> None:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model,
            "--tensor-parallel-size", str(config.tp),
            "--port", str(self.port),
            "--max-num-seqs", str(config.batch_size),
            "--max-model-len", str(config.max_seq_len),
            "--kv-cache-dtype", config.kv_cache_dtype,
        ]
        if config.pp > 1:
            cmd.extend(["--pipeline-parallel-size", str(config.pp)])
        if config.compilation_level == 0:
            cmd.append("--enforce-eager")
        if config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        logger.info("Launching vLLM server: %s", " ".join(cmd))
        self._server_proc = subprocess.Popen(
            cmd, env=os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        if not self._wait_for_server(self._server_proc, self.health_check):
            self.teardown()
            raise RuntimeError("vLLM server failed to start")

    def benchmark(
        self,
        config: InferenceConfig,
        duration_sec: int = 60,
        concurrency: int = 32,
        isl: int = 4000,
        osl: int = 1000,
    ) -> BenchmarkResult:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.run_batch",
            "--backend", "openai",
            "--base-url", f"http://{self.host}:{self.port}/v1",
            "--model", config.model,
            "--num-prompts", str(concurrency * 5),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=duration_sec + 60,
            )
            return self._parse_benchmark_output(proc.stdout, config)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("vLLM benchmark failed: %s", e)
            return BenchmarkResult(config=config)

    def teardown(self) -> None:
        self._terminate_proc(self._server_proc)
        self._server_proc = None

    def get_gpu_info(self) -> GPUInfo:
        from atom.autotuner.utils.gpu import ROCmGPU
        return ROCmGPU.detect()

    def health_check(self) -> bool:
        return self._http_health_check(self.host, self.port)
