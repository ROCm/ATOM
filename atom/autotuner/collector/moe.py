"""
MoE (Mixture of Experts) kernel benchmark collector for AMD GPUs.

Benchmarks fused MoE kernels (AITER/Triton) across parameter spaces relevant
to DeepSeek V3, Qwen3-MoE, Mixtral, GLM-MoE, etc.

Key parameters: num_tokens, num_experts, top_k, hidden_dim, intermediate_dim,
expert_parallel mode, and quantization format.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from atom.autotuner.collector.base import BaseCollector
from atom.autotuner.types import GPUInfo, KernelBenchResult, KernelConfig, KernelType

logger = logging.getLogger(__name__)

_MOE_ARCHITECTURES = [
    # (num_experts, top_k, hidden, intermediate, name)
    (8, 2, 4096, 14336, "mixtral-8x7b"),
    (64, 6, 7168, 2048, "deepseek-v3"),
    (64, 6, 5120, 1536, "deepseek-v2-lite"),
    (128, 8, 4096, 2048, "qwen3-moe"),
    (36, 4, 4096, 10240, "glm-moe"),
]


class MoECollector(BaseCollector):
    """Collect fused MoE kernel latency."""

    kernel_type = KernelType.MOE

    def __init__(
        self,
        gpu_info: GPUInfo,
        dtypes: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(gpu_info, **kwargs)
        self.dtypes = dtypes or ["fp16", "fp8"]

    def _build_sweep_configs(self) -> list[KernelConfig]:
        token_counts = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        configs = []
        for ne, topk, hidden, inter, arch_name in _MOE_ARCHITECTURES:
            for nt in token_counts:
                for dtype in self.dtypes:
                    for ep_size in [1, 2, 4, 8]:
                        if ep_size > ne:
                            continue
                        configs.append(KernelConfig(
                            kernel_type=KernelType.MOE,
                            params={
                                "num_tokens": nt,
                                "num_experts": ne,
                                "top_k": topk,
                                "hidden_dim": hidden,
                                "intermediate_dim": inter,
                                "dtype": dtype,
                                "ep_size": ep_size,
                                "arch": arch_name,
                            },
                        ))
        logger.info("MoE sweep: %d configurations", len(configs))
        return configs

    def _bench_one(self, config: KernelConfig) -> KernelBenchResult:
        p = config.params
        try:
            return self._bench_fused_moe(config)
        except (ImportError, Exception) as e:
            logger.debug("Fused MoE not available (%s), using SOL", e)
            return self._analytical_estimate(config)

    def _bench_fused_moe(self, config: KernelConfig) -> KernelBenchResult:
        """Benchmark AITER/Triton fused MoE kernel."""
        import torch

        p = config.params
        nt = p["num_tokens"]
        ne = p["num_experts"]
        topk = p["top_k"]
        hidden = p["hidden_dim"]
        inter = p["intermediate_dim"]
        device = "cuda"

        hidden_states = torch.randn(nt, hidden, device=device, dtype=torch.float16)
        router_logits = torch.randn(nt, ne, device=device, dtype=torch.float32)

        try:
            from atom.model_ops.fused_moe_triton import fused_moe

            w1 = torch.randn(ne, 2 * inter, hidden, device=device, dtype=torch.float16)
            w2 = torch.randn(ne, hidden, inter, device=device, dtype=torch.float16)

            for _ in range(self.warmup_iters):
                fused_moe(hidden_states, w1, w2, router_logits, topk, renormalize=True)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(self.bench_iters):
                fused_moe(hidden_states, w1, w2, router_logits, topk, renormalize=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            latency_us = (elapsed / self.bench_iters) * 1e6
            flops = 2.0 * nt * topk * (2 * hidden * inter + hidden * inter)
            tflops = (flops / (latency_us * 1e-6)) / 1e12

            return KernelBenchResult(
                config=config, latency_us=latency_us, throughput_tflops=tflops,
            )

        except (ImportError, Exception):
            return self._analytical_estimate(config)

    def _analytical_estimate(self, config: KernelConfig) -> KernelBenchResult:
        """SOL estimate for fused MoE based on roofline model."""
        p = config.params
        nt = p["num_tokens"]
        topk = p["top_k"]
        hidden = p["hidden_dim"]
        inter = p["intermediate_dim"]

        flops = 2.0 * nt * topk * (2 * hidden * inter + hidden * inter)
        peak = self.gpu_info.peak_tflops_fp16
        if peak <= 0:
            peak = 1000.0

        sol_us = (flops / (peak * 1e12)) * 1e6

        bytes_weights = p["num_experts"] * (2 * inter * hidden + hidden * inter) * 2
        bytes_activations = nt * hidden * 2 * 3
        total_bytes = bytes_weights + bytes_activations
        bw = self.gpu_info.memory_bw_gbps * 1e9
        if bw <= 0:
            bw = 5e12
        mem_bound_us = (total_bytes / bw) * 1e6

        estimated_us = max(sol_us, mem_bound_us) / 0.55
        tflops = (flops / (estimated_us * 1e-6)) / 1e12 if estimated_us > 0 else 0

        return KernelBenchResult(
            config=config, latency_us=estimated_us, throughput_tflops=tflops,
        )
