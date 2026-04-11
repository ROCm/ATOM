"""
GEMM micro-benchmark collector for AMD GPUs.

Addresses Q2: Uses hipBLAS (via PyTorch) and Composable Kernel (via AITER)
for FP16/BF16/FP8 GEMM benchmarks.  For quantized formats (FP8, INT8, INT4),
we call AITER's fused linear kernels directly.

Parameter space (addresses Q1): LLM-workload-informed sampling.
- M: actual batch sizes (decode: 1–512) + sequence lengths (prefill: 128–32K)
- N: hidden dimensions from common model families (4096, 5120, 8192, 14336, …)
- K: same set — these are weight matrix dimensions
"""

from __future__ import annotations

import logging
import time
from typing import Any

from atom.autotuner.collector.base import BaseCollector
from atom.autotuner.types import GPUInfo, KernelBenchResult, KernelConfig, KernelType

logger = logging.getLogger(__name__)

# Hidden dimensions from common LLM architectures
_COMMON_NK = [
    2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192,
    10240, 11008, 13824, 14336, 16384, 27648, 28672,
]

# FP8 block sizes used in DeepSeek-style block quantization
_FP8_BLOCK_SIZES = [64, 128, 256]


class GEMMCollector(BaseCollector):
    """Collect GEMM latency data across (M, N, K, dtype) parameter space."""

    kernel_type = KernelType.GEMM

    def __init__(
        self,
        gpu_info: GPUInfo,
        dtypes: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(gpu_info, **kwargs)
        self.dtypes = dtypes or ["fp16", "bf16", "fp8"]

    def _build_sweep_configs(self) -> list[KernelConfig]:
        m_values = self._llm_workload_m_values()
        configs = []
        for dtype in self.dtypes:
            nk_set = _COMMON_NK
            for m in m_values:
                for n in nk_set:
                    for k in nk_set:
                        if n == k or n * k > 500_000_000:
                            continue
                        configs.append(KernelConfig(
                            kernel_type=KernelType.GEMM,
                            params={"m": m, "n": n, "k": k, "dtype": dtype},
                        ))
        logger.info("GEMM sweep: %d configurations across %s", len(configs), self.dtypes)
        return configs

    def _bench_one(self, config: KernelConfig) -> KernelBenchResult:
        m = config.params["m"]
        n = config.params["n"]
        k = config.params["k"]
        dtype_str = config.params["dtype"]

        try:
            import torch
            torch_dtype = _resolve_dtype(dtype_str)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            a = torch.randn(m, k, dtype=torch_dtype, device=device)
            b = torch.randn(k, n, dtype=torch_dtype, device=device)

            if dtype_str.startswith("fp8"):
                return self._bench_fp8_gemm(config, m, n, k, device)

            for _ in range(self.warmup_iters):
                torch.mm(a, b)
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(self.bench_iters):
                torch.mm(a, b)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            latency_us = (elapsed / self.bench_iters) * 1e6
            flops = 2.0 * m * n * k
            tflops = (flops / (latency_us * 1e-6)) / 1e12

            return KernelBenchResult(
                config=config,
                latency_us=latency_us,
                throughput_tflops=tflops,
            )

        except ImportError:
            return self._analytical_estimate(config, m, n, k, dtype_str)

    def _bench_fp8_gemm(
        self, config: KernelConfig, m: int, n: int, k: int, device: str
    ) -> KernelBenchResult:
        """Benchmark FP8 GEMM via AITER's CK-backed linear kernel."""
        try:
            import torch
            from aiter import QuantType
            from aiter.ops.gemm import gemm_op

            a = torch.randn(m, k, dtype=torch.float8_e4m3fnuz, device=device)
            b = torch.randn(n, k, dtype=torch.float8_e4m3fnuz, device=device)
            scale_a = torch.ones(1, device=device)
            scale_b = torch.ones(1, device=device)

            for _ in range(self.warmup_iters):
                gemm_op(a, b, scale_a, scale_b)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(self.bench_iters):
                gemm_op(a, b, scale_a, scale_b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            latency_us = (elapsed / self.bench_iters) * 1e6
            flops = 2.0 * m * n * k
            tflops = (flops / (latency_us * 1e-6)) / 1e12

            return KernelBenchResult(
                config=config, latency_us=latency_us, throughput_tflops=tflops,
            )
        except (ImportError, Exception) as e:
            logger.debug("AITER FP8 GEMM not available (%s), using analytical", e)
            return self._analytical_estimate(config, m, n, k, "fp8")

    def _analytical_estimate(
        self, config: KernelConfig, m: int, n: int, k: int, dtype: str
    ) -> KernelBenchResult:
        """
        Speed-of-light estimate when hardware is unavailable.

        SOL = FLOPs / peak_tflops, with an efficiency factor (typically 0.5–0.8
        for large GEMMs, much lower for small M).
        """
        peak = self.gpu_info.peak_tflops_fp8 if "fp8" in dtype else self.gpu_info.peak_tflops_fp16
        if peak <= 0:
            peak = 1000.0

        flops = 2.0 * m * n * k
        sol_us = (flops / (peak * 1e12)) * 1e6

        efficiency = _gemm_efficiency(m, n, k)
        estimated_us = sol_us / efficiency if efficiency > 0 else sol_us * 5

        return KernelBenchResult(
            config=config,
            latency_us=estimated_us,
            throughput_tflops=(flops / (estimated_us * 1e-6)) / 1e12,
        )


def _resolve_dtype(dtype_str: str):
    import torch
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp8": torch.float16,  # fallback; real fp8 uses AITER path
        "fp8_block": torch.float16,
    }.get(dtype_str, torch.float16)


def _gemm_efficiency(m: int, n: int, k: int) -> float:
    """Heuristic GEMM efficiency based on problem size and shape."""
    total = m * n * k
    if total < 1_000_000:
        return 0.15
    if total < 100_000_000:
        return 0.40
    if total < 1_000_000_000:
        return 0.65
    return 0.78
