"""
Configuration space definition and enumeration.

Addresses Q9: defines the full search space for LLM inference configurations,
with intelligent pruning to avoid combinatorial explosion.

Pruning rules:
- TP must divide num_attention_heads
- TP * PP must divide total GPUs
- Memory constraint: model_params * bytes_per_param / TP / PP < GPU memory
- Communication constraint: TP ≤ GPUs per node (XGMI), PP may span nodes
- MoE: EP must divide num_experts, EP * MoE_TP ≤ total GPUs per worker
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterator

from atom.autotuner.types import GPUInfo, InferenceConfig
from atom.autotuner.database.estimator import ModelArch

logger = logging.getLogger(__name__)


@dataclass
class SearchBounds:
    """Defines the ranges for each searchable parameter."""
    tp_values: list[int] = None
    pp_values: list[int] = None
    dp_values: list[int] = None
    ep_values: list[int] = None
    batch_sizes: list[int] = None
    kv_cache_dtypes: list[str] = None
    quant_formats: list[str] = None
    compilation_levels: list[int] = None
    cudagraph_modes: list[str] = None
    attention_backends: list[str] = None
    disagg_modes: list[bool] = None
    prefill_worker_counts: list[int] = None
    decode_worker_counts: list[int] = None

    def __post_init__(self):
        self.tp_values = self.tp_values or [1, 2, 4, 8]
        self.pp_values = self.pp_values or [1, 2, 4]
        self.dp_values = self.dp_values or [1]
        self.ep_values = self.ep_values or [1]
        self.batch_sizes = self.batch_sizes or [1, 4, 8, 16, 32, 64, 128, 256]
        self.kv_cache_dtypes = self.kv_cache_dtypes or ["fp8", "bf16"]
        self.quant_formats = self.quant_formats or ["fp8", "bf16"]
        self.compilation_levels = self.compilation_levels or [3]
        self.cudagraph_modes = self.cudagraph_modes or ["piecewise"]
        self.attention_backends = self.attention_backends or ["aiter"]
        self.disagg_modes = self.disagg_modes or [False, True]
        self.prefill_worker_counts = self.prefill_worker_counts or [1, 2, 4]
        self.decode_worker_counts = self.decode_worker_counts or [1, 2, 4]


class ConfigSpace:
    """
    Generates valid inference configurations within the search bounds,
    applying architectural and hardware constraints to prune infeasible
    combinations.
    """

    def __init__(
        self,
        model_arch: ModelArch,
        gpu_info: GPUInfo,
        total_gpus: int,
        bounds: SearchBounds | None = None,
        isl: int = 4000,
        osl: int = 1000,
    ):
        self.arch = model_arch
        self.gpu = gpu_info
        self.total_gpus = total_gpus
        self.bounds = bounds or SearchBounds()
        self.isl = isl
        self.osl = osl

        if model_arch.is_moe:
            self.bounds.ep_values = [
                e for e in [1, 2, 4, 8, 16, 32]
                if e <= model_arch.num_experts and e <= total_gpus
            ]

    def enumerate(self) -> Iterator[InferenceConfig]:
        """Yield all valid configurations after pruning."""
        count = 0
        pruned = 0

        for disagg in self.bounds.disagg_modes:
            if disagg:
                yield from self._enumerate_disagg()
                continue

            for tp in self.bounds.tp_values:
                for pp in self.bounds.pp_values:
                    for dp in self.bounds.dp_values:
                        gpus_needed = tp * pp * dp
                        if gpus_needed > self.total_gpus:
                            pruned += 1
                            continue
                        if not self._valid_parallelism(tp, pp, dp):
                            pruned += 1
                            continue

                        for bs in self.bounds.batch_sizes:
                            if not self._valid_memory(tp, pp, bs):
                                pruned += 1
                                continue

                            for kv_dt in self.bounds.kv_cache_dtypes:
                                for qf in self.bounds.quant_formats:
                                    for cl in self.bounds.compilation_levels:
                                        for cg in self.bounds.cudagraph_modes:
                                            for ab in self.bounds.attention_backends:
                                                ep = self._best_ep(tp) if self.arch.is_moe else 1
                                                cfg = InferenceConfig(
                                                    model=self.arch.name,
                                                    tp=tp, pp=pp, dp=dp, ep=ep,
                                                    batch_size=bs,
                                                    max_seq_len=self.isl + self.osl,
                                                    kv_cache_dtype=kv_dt,
                                                    quant_format=qf,
                                                    compilation_level=cl,
                                                    cudagraph_mode=cg,
                                                    attention_backend=ab,
                                                    isl=self.isl,
                                                    osl=self.osl,
                                                )
                                                count += 1
                                                yield cfg

        logger.info(
            "ConfigSpace: enumerated %d configs, pruned %d infeasible", count, pruned
        )

    def _enumerate_disagg(self) -> Iterator[InferenceConfig]:
        """Enumerate disaggregated (prefill/decode split) configurations."""
        for tp in self.bounds.tp_values:
            for pp in self.bounds.pp_values:
                gpus_per_worker = tp * pp
                for pw in self.bounds.prefill_worker_counts:
                    for dw in self.bounds.decode_worker_counts:
                        total_needed = gpus_per_worker * (pw + dw)
                        if total_needed > self.total_gpus:
                            continue
                        if not self._valid_parallelism(tp, pp, 1):
                            continue

                        for bs in self.bounds.batch_sizes:
                            if not self._valid_memory(tp, pp, bs):
                                continue
                            for kv_dt in self.bounds.kv_cache_dtypes:
                                for qf in self.bounds.quant_formats:
                                    ep = self._best_ep(tp) if self.arch.is_moe else 1
                                    yield InferenceConfig(
                                        model=self.arch.name,
                                        tp=tp, pp=pp, dp=1, ep=ep,
                                        batch_size=bs,
                                        max_seq_len=self.isl + self.osl,
                                        kv_cache_dtype=kv_dt,
                                        quant_format=qf,
                                        disagg=True,
                                        prefill_workers=pw,
                                        decode_workers=dw,
                                        isl=self.isl,
                                        osl=self.osl,
                                    )

    def _valid_parallelism(self, tp: int, pp: int, dp: int) -> bool:
        if self.arch.num_q_heads % tp != 0:
            return False
        if self.arch.num_layers % pp != 0:
            return False
        if tp > 8:
            return False
        return True

    def _valid_memory(self, tp: int, pp: int, batch_size: int) -> bool:
        """Conservative memory check: model weights + KV cache < GPU memory."""
        param_bytes = 2  # fp16/bf16 baseline
        layers_per_stage = self.arch.num_layers // max(pp, 1)
        weight_bytes_per_gpu = (
            self.arch.hidden_dim * self.arch.intermediate_dim * 3 * layers_per_stage * param_bytes
        ) / tp

        if self.arch.is_moe:
            weight_bytes_per_gpu += (
                self.arch.num_experts * self.arch.intermediate_dim * self.arch.hidden_dim * 3 * param_bytes
                * layers_per_stage
            ) / tp

        kv_bytes_per_token = (
            2 * self.arch.num_kv_heads * self.arch.head_dim * 2  # K + V, fp16
        ) / tp
        kv_total = kv_bytes_per_token * batch_size * (self.isl + self.osl) * layers_per_stage

        total_gb = (weight_bytes_per_gpu + kv_total) / 1e9
        available_gb = self.gpu.memory_gb * 0.85

        return total_gb < available_gb

    def _best_ep(self, tp: int) -> int:
        """Pick the largest valid EP for MoE models given TP."""
        for ep in sorted(self.bounds.ep_values, reverse=True):
            if self.arch.num_experts % ep == 0 and ep * tp <= self.total_gpus:
                return ep
        return 1

    def count(self) -> int:
        """Count total valid configurations (without materializing all)."""
        return sum(1 for _ in self.enumerate())
