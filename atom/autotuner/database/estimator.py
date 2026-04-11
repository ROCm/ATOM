"""
End-to-end latency estimator: kernel-level predictions → iteration time.

Addresses Q6: the composition from individual kernel latencies to E2E time
must account for:
1. Kernel launch overhead (~3-5 μs per launch on MI300X/MI355X)
2. Memory allocation / sync overhead
3. Pipeline parallel bubble ratio
4. Scheduler + sampling overhead
5. KV cache management overhead
6. Overlap between compute and communication (when applicable)

For disaggregated serving (Q8): prefill and decode are modeled separately,
with KV cache transfer cost computed from the P2P / network bandwidth
between prefill and decode workers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from atom.autotuner.types import (
    BenchmarkResult,
    GPUInfo,
    InferenceConfig,
    KernelConfig,
    KernelType,
)
from atom.autotuner.database.perf_model import PerformanceModel

logger = logging.getLogger(__name__)

KERNEL_LAUNCH_OVERHEAD_US = 3.5
SCHEDULER_OVERHEAD_US = 50.0
SAMPLING_OVERHEAD_US = 20.0
KV_CACHE_MGMT_OVERHEAD_US = 10.0


@dataclass
class LayerBreakdown:
    """Latency breakdown for a single transformer layer."""
    qkv_proj_us: float = 0.0
    attn_kernel_us: float = 0.0
    attn_out_proj_us: float = 0.0
    mlp_gate_up_us: float = 0.0
    mlp_down_us: float = 0.0
    moe_us: float = 0.0
    layernorm_us: float = 0.0
    allreduce_us: float = 0.0
    alltoall_us: float = 0.0
    residual_us: float = 0.0
    launch_overhead_us: float = 0.0

    @property
    def total_us(self) -> float:
        return (
            self.qkv_proj_us
            + self.attn_kernel_us
            + self.attn_out_proj_us
            + self.mlp_gate_up_us
            + self.mlp_down_us
            + self.moe_us
            + self.layernorm_us
            + self.allreduce_us
            + self.alltoall_us
            + self.residual_us
            + self.launch_overhead_us
        )


@dataclass
class IterationBreakdown:
    """Full iteration latency breakdown."""
    embedding_us: float = 0.0
    layers: list[LayerBreakdown] = None
    lm_head_us: float = 0.0
    scheduler_us: float = SCHEDULER_OVERHEAD_US
    sampling_us: float = SAMPLING_OVERHEAD_US
    kv_mgmt_us: float = KV_CACHE_MGMT_OVERHEAD_US
    pp_bubble_us: float = 0.0
    kv_transfer_us: float = 0.0

    def __post_init__(self):
        if self.layers is None:
            self.layers = []

    @property
    def compute_us(self) -> float:
        return self.embedding_us + sum(l.total_us for l in self.layers) + self.lm_head_us

    @property
    def overhead_us(self) -> float:
        return self.scheduler_us + self.sampling_us + self.kv_mgmt_us

    @property
    def total_us(self) -> float:
        return self.compute_us + self.overhead_us + self.pp_bubble_us + self.kv_transfer_us


class E2EEstimator:
    """
    Estimates end-to-end inference latency from kernel-level performance model.

    Given a model architecture description and an InferenceConfig, composes
    per-kernel latencies into prefill and decode iteration times, then
    derives TTFT, TPOT, and throughput metrics.
    """

    def __init__(self, perf_model: PerformanceModel, gpu_info: GPUInfo):
        self.perf_model = perf_model
        self.gpu_info = gpu_info

    def estimate(self, config: InferenceConfig, model_arch: ModelArch) -> BenchmarkResult:
        """Estimate full inference metrics for a deployment configuration."""
        prefill_iter = self._estimate_iteration(config, model_arch, phase="prefill")
        decode_iter = self._estimate_iteration(config, model_arch, phase="decode")

        prefill_time_ms = prefill_iter.total_us / 1000.0
        decode_time_ms = decode_iter.total_us / 1000.0

        if config.disagg:
            kv_transfer_ms = self._estimate_kv_transfer(config, model_arch)
            ttft_ms = prefill_time_ms + kv_transfer_ms
        else:
            ttft_ms = prefill_time_ms

        tpot_ms = decode_time_ms

        tokens_per_sec_per_user = 1000.0 / tpot_ms if tpot_ms > 0 else 0
        request_latency_ms = ttft_ms + config.osl * tpot_ms
        total_gpus = config.total_gpus_used()
        concurrency = config.batch_size * (config.dp if not config.disagg else 1)
        throughput = concurrency * tokens_per_sec_per_user
        throughput_per_gpu = throughput / max(total_gpus, 1)

        return BenchmarkResult(
            config=config,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            throughput_tokens_per_sec=throughput,
            throughput_per_gpu=throughput_per_gpu,
            throughput_per_user=tokens_per_sec_per_user,
            request_latency_ms=request_latency_ms,
        )

    def _estimate_iteration(
        self,
        config: InferenceConfig,
        arch: ModelArch,
        phase: str,
    ) -> IterationBreakdown:
        """Build full iteration breakdown for prefill or decode."""
        breakdown = IterationBreakdown()

        if phase == "prefill":
            seq_len = config.isl
            batch = 1
        else:
            seq_len = 1
            batch = config.batch_size

        tp = config.tp
        hidden = arch.hidden_dim
        num_heads = arch.num_q_heads
        num_kv_heads = arch.num_kv_heads
        head_dim = arch.head_dim
        intermediate = arch.intermediate_dim

        breakdown.embedding_us = self._predict_gemm(
            batch * seq_len, hidden, arch.vocab_size // tp, config.quant_format
        ) + KERNEL_LAUNCH_OVERHEAD_US

        layers_per_stage = arch.num_layers // max(config.pp, 1)
        num_kernels_per_layer = 8  # approximate

        for _ in range(layers_per_stage):
            layer = LayerBreakdown()

            heads_per_tp = num_heads // tp
            kv_heads_per_tp = max(num_kv_heads // tp, 1)

            layer.qkv_proj_us = self._predict_gemm(
                batch * seq_len,
                hidden,
                (heads_per_tp + 2 * kv_heads_per_tp) * head_dim,
                config.quant_format,
            )

            if phase == "prefill":
                layer.attn_kernel_us = self._predict_attention(
                    phase, batch, seq_len, seq_len,
                    heads_per_tp, kv_heads_per_tp, head_dim,
                    config.kv_cache_dtype,
                )
            else:
                ctx_len = config.isl + config.osl // 2
                layer.attn_kernel_us = self._predict_attention(
                    phase, batch, 1, ctx_len,
                    heads_per_tp, kv_heads_per_tp, head_dim,
                    config.kv_cache_dtype,
                )

            layer.attn_out_proj_us = self._predict_gemm(
                batch * seq_len, heads_per_tp * head_dim, hidden, config.quant_format
            )

            if arch.is_moe:
                layer.moe_us = self._predict_moe(
                    batch * seq_len, arch.num_experts, arch.top_k,
                    hidden, intermediate, config.quant_format, config.ep,
                )
                if config.ep > 1:
                    msg_bytes = batch * seq_len * hidden * 2 * arch.top_k
                    layer.alltoall_us = self._predict_comm(
                        "all_to_all", tp, msg_bytes
                    )
            else:
                layer.mlp_gate_up_us = self._predict_gemm(
                    batch * seq_len, hidden, 2 * intermediate // tp, config.quant_format
                )
                layer.mlp_down_us = self._predict_gemm(
                    batch * seq_len, intermediate // tp, hidden, config.quant_format
                )

            layer.layernorm_us = 2.0
            layer.residual_us = 1.0

            if tp > 1:
                ar_bytes = batch * seq_len * hidden * 2
                layer.allreduce_us = self._predict_comm("all_reduce", tp, ar_bytes)
                if not arch.is_moe:
                    layer.allreduce_us *= 2  # after attn + after MLP

            layer.launch_overhead_us = num_kernels_per_layer * KERNEL_LAUNCH_OVERHEAD_US

            breakdown.layers.append(layer)

        breakdown.lm_head_us = self._predict_gemm(
            batch * seq_len, hidden, arch.vocab_size // tp, config.quant_format
        ) + KERNEL_LAUNCH_OVERHEAD_US

        if config.pp > 1:
            pp_stages = config.pp
            micro_batches = max(batch, 1)
            if micro_batches >= pp_stages:
                bubble_ratio = (pp_stages - 1) / micro_batches
            else:
                bubble_ratio = (pp_stages - 1) / pp_stages
            breakdown.pp_bubble_us = breakdown.compute_us * bubble_ratio

        return breakdown

    def _estimate_kv_transfer(
        self, config: InferenceConfig, arch: ModelArch
    ) -> float:
        """
        Estimate KV cache transfer time for disaggregated serving (Q8).

        Transfer size = num_layers * 2 * num_kv_heads * seq_len * head_dim * dtype_size
        Transfer bandwidth depends on interconnect (XGMI intra-node, network inter-node).
        """
        dtype_bytes = 1 if "fp8" in config.kv_cache_dtype else 2
        kv_size = (
            arch.num_layers * 2 * arch.num_kv_heads * config.isl * arch.head_dim * dtype_bytes
        )
        bw = self.gpu_info.interconnect_bw_gbps * 1e9
        if bw <= 0:
            bw = 100e9
        transfer_us = (kv_size / bw) * 1e6
        return transfer_us / 1000.0  # return ms

    # ------------------------------------------------------------------
    # Kernel-level prediction wrappers
    # ------------------------------------------------------------------

    def _predict_gemm(self, m: int, n: int, k: int, dtype: str) -> float:
        config = KernelConfig(KernelType.GEMM, {"m": m, "n": n, "k": k, "dtype": dtype})
        return self.perf_model.predict(config)

    def _predict_attention(
        self, phase: str, batch: int, seq_len: int, ctx_len: int,
        nqh: int, nkvh: int, hd: int, kv_dtype: str,
    ) -> float:
        config = KernelConfig(KernelType.ATTENTION, {
            "phase": phase, "batch_size": batch, "seq_len": seq_len,
            "context_len": ctx_len, "num_q_heads": nqh, "num_kv_heads": nkvh,
            "head_dim": hd, "kv_dtype": kv_dtype,
        })
        return self.perf_model.predict(config)

    def _predict_moe(
        self, nt: int, ne: int, topk: int, hidden: int, inter: int,
        dtype: str, ep: int,
    ) -> float:
        config = KernelConfig(KernelType.MOE, {
            "num_tokens": nt, "num_experts": ne, "top_k": topk,
            "hidden_dim": hidden, "intermediate_dim": inter,
            "dtype": dtype, "ep_size": ep, "arch": "generic",
        })
        return self.perf_model.predict(config)

    def _predict_comm(self, op: str, tp: int, msg_bytes: int) -> float:
        config = KernelConfig(KernelType.COMMUNICATION, {
            "op": op, "tp_size": tp, "message_bytes": msg_bytes,
        })
        return self.perf_model.predict(config)


# ---------------------------------------------------------------------------
# Model architecture descriptor
# ---------------------------------------------------------------------------

@dataclass
class ModelArch:
    """Simplified model architecture for E2E estimation."""
    name: str
    num_layers: int
    hidden_dim: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_dim: int
    vocab_size: int
    is_moe: bool = False
    num_experts: int = 1
    top_k: int = 1

    @classmethod
    def from_hf_config(cls, model_path: str) -> ModelArch:
        """Load architecture from HuggingFace config.json."""
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            num_experts = getattr(cfg, "num_local_experts", getattr(cfg, "n_routed_experts", 1))
            top_k = getattr(cfg, "num_experts_per_tok", getattr(cfg, "topk_group", 1))

            return cls(
                name=model_path.split("/")[-1],
                num_layers=getattr(cfg, "num_hidden_layers", 32),
                hidden_dim=getattr(cfg, "hidden_size", 4096),
                num_q_heads=getattr(cfg, "num_attention_heads", 32),
                num_kv_heads=getattr(cfg, "num_key_value_heads",
                                      getattr(cfg, "num_attention_heads", 32)),
                head_dim=getattr(cfg, "head_dim",
                                  getattr(cfg, "hidden_size", 4096) //
                                  getattr(cfg, "num_attention_heads", 32)),
                intermediate_dim=getattr(cfg, "intermediate_size", 11008),
                vocab_size=getattr(cfg, "vocab_size", 32000),
                is_moe=num_experts > 1,
                num_experts=num_experts,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning("Cannot load HF config for %s: %s", model_path, e)
            return cls.llama_70b()

    @classmethod
    def llama_70b(cls) -> ModelArch:
        return cls("llama-70b", 80, 8192, 64, 8, 128, 28672, 128256)

    @classmethod
    def deepseek_v3(cls) -> ModelArch:
        return cls("deepseek-v3", 61, 7168, 128, 1, 128, 2048, 129280,
                    is_moe=True, num_experts=256, top_k=8)

    @classmethod
    def gpt_oss_120b(cls) -> ModelArch:
        return cls("gpt-oss-120b", 96, 12288, 96, 8, 128, 40960, 128256)

    @classmethod
    def qwen3_32b(cls) -> ModelArch:
        return cls("qwen3-32b", 64, 5120, 40, 8, 128, 25600, 152064)

    @classmethod
    def kimi_k2(cls) -> ModelArch:
        return cls("kimi-k2", 61, 7168, 128, 1, 128, 2048, 129280,
                    is_moe=True, num_experts=256, top_k=8)
