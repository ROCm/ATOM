"""Gemma 4 + Qwen 3.5 operator optimization integration for MI355X.

This module documents the optimization status of each operator used by
Gemma 4 and Qwen 3.5, provides configuration helpers, and contains
integration code for AITER-accelerated paths.

Optimization hierarchy (fastest to slowest):
  ASM (HSACO) > CK (Composable Kernel) > FlyDSL (MLIR) > Gluon > Triton > CUDA/HIP > PyTorch

Status codes:
  [DONE]   - Implemented and integrated
  [READY]  - Available in AITER, needs model-level wiring
  [EXTEND] - Kernel exists but needs extension for Gemma 4/Qwen 3.5 specifics
  [TODO]   - Requires new kernel development
"""

import os
import logging
from dataclasses import dataclass

logger = logging.getLogger("atom.gemma4_ops")


@dataclass
class OptimizerStatus:
    operator: str
    current_level: str
    target_level: str
    status: str
    model: str
    notes: str


GEMMA4_OPS = [
    # --- P0: DONE ---
    OptimizerStatus(
        operator="GeluAndMul",
        current_level="AITER CUDA JIT",
        target_level="AITER CUDA JIT",
        status="DONE",
        model="Gemma 4",
        notes="Replaced PyTorch F.gelu with aiter.gelu_tanh_and_mul. "
              "Full quant paths (FP8, MXFP4) wired through GeluAndMul class.",
    ),
    OptimizerStatus(
        operator="Logit Softcapping",
        current_level="Fused Triton",
        target_level="Fused Triton",
        status="DONE",
        model="Gemma 4",
        notes="fused_logit_softcap() Triton kernel replaces 3 PyTorch ops (div+tanh+mul) "
              "with single in-place kernel. cap=30.0 for Gemma 4.",
    ),
    OptimizerStatus(
        operator="CUDA Graph (decode)",
        current_level="Piecewise",
        target_level="Piecewise",
        status="DONE",
        model="Gemma 4",
        notes="Enabled via --compilation-config '{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}'. "
              "PiecewiseBackend handles sliding/global attention layer transitions.",
    ),
    OptimizerStatus(
        operator="torch.compile",
        current_level="Inductor",
        target_level="Inductor",
        status="DONE",
        model="Gemma 4",
        notes="@support_torch_compile on Gemma4Model with dynamic_arg_dims for "
              "input_ids, positions, intermediate_tensors, inputs_embeds.",
    ),

    # --- P1: READY / EXTEND ---
    OptimizerStatus(
        operator="PA Decode (ASM v1)",
        current_level="Gluon (Triton CDNA4)",
        target_level="ASM PA v1",
        status="EXTEND",
        model="Gemma 4",
        notes="ASM PA v1 (aiter/aot/pa_v1.py) only supports head_size={64,128}. "
              "Gemma 4 uses head_dim=256 (sliding) and global_head_dim=512 (full). "
              "REQUIRES: extending pa_v1 HSACO configs for head_size={256,512}. "
              "GQA ratios: sliding=32/16=2, global=32/4=8 — both supported. "
              "logits_soft_cap_enabled=True is supported by PA v1.",
    ),
    OptimizerStatus(
        operator="Fused QK-norm-RoPE-cache-quant",
        current_level="Disabled",
        target_level="CUDA fused",
        status="READY",
        model="Gemma 4",
        notes="ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1 enables the fused kernel "
              "(aiter/ops/fused_qk_norm_rope_cache_quant). Works for Qwen3-MoE. "
              "Gemma 4 extension needed: per-layer-type RoPE (sliding uses theta=10000, "
              "global uses theta=1M + partial_rotary_factor=0.25). "
              "REQUIRES: separate fused kernel configs per layer_type, or dynamic "
              "RoPE parameter dispatch inside the fused kernel.",
    ),
    OptimizerStatus(
        operator="FlyDSL Fused MoE",
        current_level="ASM + CK",
        target_level="FlyDSL",
        status="TODO",
        model="Gemma 4 26B-A4B",
        notes="ROCm/FlyDSL demonstrated 65% TTFT, 69% TPOT, 162% throughput gains on "
              "Kimi-K2.5 MoE. Gemma 4 26B-A4B: 128 experts, top-k=8, intermediate=704. "
              "REQUIRES: clone ROCm/FlyDSL, adapt W4A16+BF16 fused MoE kernel for "
              "Gemma 4 expert dimensions. Current ASM+CK path is already good but "
              "FlyDSL can fuse gate+up projection and optimize per-expert-count layout.",
    ),
    OptimizerStatus(
        operator="AllReduce + RMSNorm fusion",
        current_level="CUDA (enabled by default)",
        target_level="CUDA",
        status="DONE",
        model="Both",
        notes="ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=1 (default). Already active via "
              "tensor_model_parallel_fused_allreduce_rmsnorm() in atom.model_ops.layernorm.",
    ),
    OptimizerStatus(
        operator="Quick AllReduce INT4",
        current_level="Available",
        target_level="Enabled",
        status="READY",
        model="Both (TP > 1)",
        notes="AITER_QUICK_REDUCE_QUANTIZATION=INT4. Added to Gemma 4 recipe. "
              "Reduces inter-GPU communication bandwidth by 4x.",
    ),
]

QWEN35_OPS = [
    OptimizerStatus(
        operator="GatedDeltaNet",
        current_level="Triton",
        target_level="CK / FlyDSL",
        status="TODO",
        model="Qwen 3.5",
        notes="chunk_gated_delta_rule and fused_recurrent_gated_delta_rule in "
              "aiter/ops/triton/gated_delta_net/ are pure Triton. This is the critical "
              "path: ~75% of layers (3 of every 4) use linear_attention. "
              "REQUIRES: CK-tile or FlyDSL implementation with explicit layout "
              "optimization for the chunk-wise state update pattern. "
              "FlyDSL's layout algebra maps well to the blocked state recurrence.",
    ),
    OptimizerStatus(
        operator="causal_conv1d",
        current_level="Triton",
        target_level="CK",
        status="TODO",
        model="Qwen 3.5",
        notes="aiter/ops/triton/causal_conv1d.py is Triton. Used in GatedDeltaNet layers. "
              "REQUIRES: CK kernel with optimal memory layout for conv_kernel_dim=4. "
              "Lower priority since conv1d is a small part of the compute.",
    ),
    OptimizerStatus(
        operator="512-expert MoE",
        current_level="ASM + CK",
        target_level="FlyDSL tuned",
        status="TODO",
        model="Qwen 3.5 MoE",
        notes="512 experts with top-k=10+1 shared. Current ASM+CK handles generic MoE. "
              "REQUIRES: FlyDSL tuning for 512-expert scale with shared expert fusion. "
              "MORI EP communication for distributing 512 experts across 8 GPUs.",
    ),
    OptimizerStatus(
        operator="Flash Attention Prefill",
        current_level="Triton",
        target_level="CK Tile",
        status="TODO",
        model="Both",
        notes="Triton flash attention for prefill. CK-tile flash attention would give "
              "better CDNA4 utilization on MI355X. "
              "REQUIRES: evaluate ROCm CK-tile flash attention availability, "
              "integrate via AITER attention backend selection.",
    ),
]


def get_optimization_summary() -> str:
    """Return a human-readable summary of all operator optimizations."""
    lines = ["=" * 80]
    lines.append("Operator Optimization Status: Gemma 4 + Qwen 3.5 on MI355X")
    lines.append("=" * 80)

    for title, ops in [("Gemma 4", GEMMA4_OPS), ("Qwen 3.5", QWEN35_OPS)]:
        lines.append(f"\n--- {title} ---")
        for op in ops:
            lines.append(
                f"  [{op.status:6s}] {op.operator:35s} "
                f"{op.current_level:20s} -> {op.target_level}"
            )

    done = sum(1 for o in GEMMA4_OPS + QWEN35_OPS if o.status == "DONE")
    ready = sum(1 for o in GEMMA4_OPS + QWEN35_OPS if o.status == "READY")
    extend = sum(1 for o in GEMMA4_OPS + QWEN35_OPS if o.status == "EXTEND")
    todo = sum(1 for o in GEMMA4_OPS + QWEN35_OPS if o.status == "TODO")

    lines.append(f"\nTotal: DONE={done}, READY={ready}, EXTEND={extend}, TODO={todo}")
    return "\n".join(lines)


# Environment variable configuration for optimal performance
GEMMA4_ENV_VARS = {
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION": "1",
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION": "1",
    "AITER_QUICK_REDUCE_QUANTIZATION": "INT4",
}

QWEN35_ENV_VARS = {
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION": "1",
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION": "1",
    "AITER_QUICK_REDUCE_QUANTIZATION": "INT4",
    "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD": "1024",
}


def configure_env_for_model(model_type: str) -> None:
    """Set environment variables for optimal ATOM performance."""
    env_vars = GEMMA4_ENV_VARS if "gemma4" in model_type.lower() else QWEN35_ENV_VARS
    for k, v in env_vars.items():
        os.environ.setdefault(k, v)
        logger.info("Set %s=%s", k, v)


if __name__ == "__main__":
    print(get_optimization_summary())
