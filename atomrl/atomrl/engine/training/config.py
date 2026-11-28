# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Unified YAML + OmegaConf configuration system for AtomRL."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from atomrl.architecture.config.assembly_config import RuntimeAssemblyConfig
from atomrl.core.types import (
    AlgorithmName,
    GenerationBackend,
    TrainingBackend,
)

logger = logging.getLogger(__name__)


@dataclass
class FSDPEngineConfig:
    """Configuration for the FSDP2 training engine."""

    strategy: str = "fsdp2"
    fsdp_size: int = -1
    param_offload: bool = False
    optimizer_offload: bool = False
    grad_offload: bool = False
    reshard_after_forward: bool = True
    forward_only: bool = False
    seed: int = 42
    model_dtype: str = "bf16"
    mixed_precision: dict | None = None
    use_remove_padding: bool = True
    ulysses_sequence_parallel_size: int = 1
    forward_prefetch: bool = False
    use_orig_params: bool = True
    use_torch_compile: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer and LR scheduler configuration."""

    optimizer_type: str = "adamw"
    lr: float = 1e-6
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    lr_scheduler_type: str = "cosine"
    lr_warmup_steps: int = 10
    lr_warmup_steps_ratio: float = 0.0
    lr_warmup_init: float = 0.0
    total_training_steps: int = 1000
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5


@dataclass
class LoRAConfig:
    """LoRA / PEFT configuration."""

    enabled: bool = False
    rank: int = 0
    alpha: int = 16
    target_modules: list | None = None
    exclude_modules: list | None = None
    merge: bool = False
    adapter_path: str | None = None


@dataclass
class HFModelConfig:
    """HuggingFace model loading configuration."""

    local_path: str = ""
    model_type: str = "language_model"
    trust_remote_code: bool = True
    use_remove_padding: bool = True
    enable_gradient_checkpointing: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    use_liger: bool = False
    use_fused_kernels: bool = False
    calculate_entropy: bool = False
    calculate_sum_pi_squared: bool = False


@dataclass
class ClusterConfig:
    num_nodes: int = 1
    gpus_per_node: int = 1
    ray_address: str | None = None


@dataclass
class RayWorkerRoleConfig:
    """Per-role worker-group orchestration knobs for Ray controller path."""

    # 0 means auto-infer from pool world size.
    num_workers: int = 0
    # Supported dispatch modes:
    # - dp_compute_proto (default)
    # - dp_compute
    # - dp_compute_proto_with_func
    # - dp_compute_metric
    # - one_to_all
    # - all_to_all
    # - rank_zero
    # - direct_rollout_method (forbidden in controller dispatch path)
    # Legacy alias accepted at runtime: broadcast -> one_to_all
    dispatch_mode: str = "dp_compute_proto"
    mesh_mapping: list[int] | None = None
    lazy_dispatch_key: str | None = None
    detached: bool = False
    process_on_nodes: list[int] | None = None
    max_colocate_count: int = 1
    topology_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class RayControllerConfig:
    """Ray-controller runtime options for the trainer main path."""

    enabled: bool = False
    fuse_actor_ref: bool = False
    # See atomrl.core.config.RayControllerConfig.ref_in_actor.
    ref_in_actor: bool = False
    actor: RayWorkerRoleConfig = field(default_factory=RayWorkerRoleConfig)
    ref: RayWorkerRoleConfig = field(default_factory=RayWorkerRoleConfig)
    rollout: RayWorkerRoleConfig = field(default_factory=RayWorkerRoleConfig)
    # Optional role->pool name mapping for complex topology routing.
    topology_map: dict[str, str] = field(default_factory=dict)


@dataclass
class ControllerConfig:
    ray: RayControllerConfig = field(default_factory=RayControllerConfig)


@dataclass
class MegatronConfig:
    # Parallelism sizes. NOTE: field names match what ``actor_worker`` and the
    # Megatron engines read (``*_model_parallel_size`` + ``context_parallel_size``).
    # (Previously named ``tensor_parallel_size`` / ``pipeline_parallel_size`` /
    # ``expert_parallel_size``, which silently never reached the engine -> TP/PP
    # config was ignored. See megatron-native-refactor handoff §2.)
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    # Sequence parallelism (TP-only): shards activations along the sequence dim.
    # Requires seq length divisible by TP, so it is OFF by default for RL's
    # variable-length forwards (per-sequence / packed thd).
    sequence_parallel: bool = False
    # DSV4 TileLang indexer launch tuning. MI300X requires block_n=64 and
    # num_stages=1 to stay within its 64 KiB LDS limit.
    v4_indexer_block_n: int | None = None
    v4_indexer_num_stages: int | None = None
    # ---- MoE / Expert Parallel ----
    # ``num_experts`` (a.k.a. num_moe_experts) is auto-detected from the HF config
    # when None; set it explicitly only to override. All the ``moe_*`` knobs below
    # are likewise auto-derived from the HF config when left None, and forwarded to
    # Megatron's TransformerConfig only when the model is MoE (dense path ignores
    # them). ``expert_model_parallel_size`` shards experts across EP ranks;
    # ``expert_tensor_parallel_size`` (ETP) additionally TP-shards each expert.
    num_experts: int | None = None
    moe_grouped_gemm: bool = True
    expert_tensor_parallel_size: int | None = None
    moe_router_topk: int | None = None
    moe_router_load_balancing_type: str = (
        "aux_loss"  # "aux_loss" | "none" | "seq_aux_loss"
    )
    moe_router_pre_softmax: bool | None = None
    moe_router_score_function: str | None = None  # "softmax" | "sigmoid"
    # fp32 router improves expert-selection stability (bf16 top-k flips experts ->
    # large train/rollout log-prob mismatch). Defaults to fp32 for MoE.
    moe_router_dtype: str | None = "fp32"  # None | "fp32" | "fp64"
    moe_router_topk_scaling_factor: float | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_aux_loss_coeff: float = 0.0  # 0 keeps RL loss unchanged
    moe_router_bias_update_rate: float | None = None
    freeze_moe_router: bool = False
    moe_token_dispatcher_type: str = "alltoall"
    # Flex dispatcher backend when ``moe_token_dispatcher_type`` is ``flex``.
    # ``mori`` matches Megatron-LM ``ENABLE_MORI=true``.
    moe_flex_dispatcher_backend: str | None = None  # "deepep" | "hybridep" | "mori"
    moe_mori_max_tokens_per_rank: int | None = None
    moe_mori_kernel_type: str | None = None
    moe_permute_fusion: bool = False
    # Distributed optimizer: shard FP32 master + Adam state across DP ranks.
    use_distributed_optimizer: bool = True
    # MILES numerical-alignment knobs for BF16 training.
    grad_reduce_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    # Activation recomputation (gradient checkpointing) for long sequences.
    recompute_granularity: str | None = None  # None | "full" | "selective"
    recompute_method: str | None = None  # "uniform" | "block"
    recompute_num_layers: int | None = None
    # Long-sequence memory: flash attention (O(L) vs local O(L^2)) + memory-efficient
    # chunked/fused token log-prob. Both default off (unchanged smoke behavior).
    attention_backend: str = "unfused"  # "flash" | "unfused"
    # Megatron native: TransformerEngine's attention backend, as Megatron's
    # attention_backend (auto | flash | fused | unfused). None: fused on ROCm
    # when flash_attn is installed, else auto.
    te_attention_backend: str | None = None
    use_packed_sequences: bool = (
        True  # varlen flash packing; disable for incompatible ROCm kernels
    )
    log_probs_chunk_size: int = 0  # >0 enables fused/chunked log-prob
    # Dynamic-batch packing: concat multiple sequences into one packed TE forward.
    enable_dynamic_batch: bool = False
    max_tokens_per_gpu: int = 0  # per-forward token budget (0 -> 21504)
    # Optimizer CPU offload: move Adam states (exp_avg/exp_avg_sq) to CPU memory.
    # Frees ~2x model-size GPU memory at the cost of slower optimizer steps.
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: float = 1.0  # fraction of states to offload (0.0-1.0)
    # Streaming mode: off | sgd | adam.
    streamed_optimizer_mode: str = "off"
    # Positive MiB; validated at DSV4 startup.
    streamed_optimizer_chunk_size_mib: int = 256
    # Persistent Adam moment storage; bf16 halves host-memory residency.
    streamed_optimizer_moment_dtype: str = "fp32"
    # Let Megatron's optimizer own the FP32 master directly. With full CPU
    # offload this avoids first materializing another FP32 master on GPU.
    use_precision_aware_optimizer: bool = False
    # PP layer distribution: override uniform layer-per-stage split.
    num_layers_in_first_pipeline_stage: int | None = None
    num_layers_in_last_pipeline_stage: int | None = None
    # R3 routing replay (MoE only)
    moe_enable_routing_replay: bool = False


@dataclass
class AtomConfig:
    tensor_parallel_size: int = 1
    kv_cache_dtype: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.6
    gpu_id: int | None = None
    # Ray-controller online rollout path. The common rollout knobs (sampling,
    # sequence limits, IPC bucket size, sleep) are still read from vllm_cfg so the
    # DAPO vLLM/ATOM configs stay comparable.
    transport: str = "fifo"
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    enable_prefix_caching: bool = False
    online_quant_config: dict | None = None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLLMConfig:
    """Native vLLM rollout engine configuration (vanilla upstream vLLM).

    Mirrors the runbook's ``actor_rollout_ref.rollout`` vLLM knobs so a DAPO run
    can use vLLM for inference + Lumen FSDP for training without verl.
    """

    tensor_parallel_size: int = 1
    kv_cache_dtype: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.6
    gpu_id: int | None = None
    dtype: str = "bfloat16"
    enforce_eager: bool = True
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 64
    swap_space: int = 4
    trust_remote_code: bool = True
    # Rollout quantization: "" / "fp8" / "fp8_per_block" (vLLM `quantization=`)
    quantization: str = ""
    # When True, vLLM returns per-token rollout log-probs needed for TIS / MIS
    # rollout correction (verl: actor_rollout_ref.rollout.calculate_log_probs).
    calculate_log_probs: bool = False
    # verl alignment: verl seeds each vLLM engine with ``replica_rank + data.seed``
    # so rollout sampling is reproducible and matched across ranks. ``None`` keeps
    # vLLM's internal default. Set by the trainer from the top-level ``config.seed``;
    # the engine adds the per-rank offset (local_rank) at worker launch.
    seed: int | None = None
    # Keep the vLLM engine resident across steps and update weights in place via
    # `collective_rpc("reload_weights")` instead of killing + rebuilding the
    # subprocess each step. Removes the per-step rebuild (~45s) AND the ROCm
    # rebuild-leak OOM. Memory: one resident engine (gpu_memory_utilization)
    # coexists with FSDP training.
    persistent: bool = True
    # Data-parallel rollout: every rank runs its own vLLM on its local GPU and
    # generates a shard of the batch (then all-gather), instead of rank-0
    # generating everything. ~Nx generation throughput on N GPUs.
    data_parallel_rollout: bool = True
    # verl-style ONLINE rollout: drive the engine with vLLM v1 ``AsyncLLM`` and
    # submit every prompt as its own concurrent ``generate(request_id=...)`` call
    # (continuous batching), matching verl's async server, instead of the offline
    # ``LLM.generate(list)`` batch call. Weight updates are applied in place via
    # ``collective_rpc("reload_weights", weights_path=...)`` + ``reset_prefix_cache``
    # (no engine rebuild). Overridable with env ``LUMEN_VLLM_ASYNC=0/1``.
    online: bool = True
    # Rollout transport / orchestration:
    #   "fifo"    -> per-rank subprocess + named-FIFO JSON (default, VLLMEngine)
    #   "ray_http"-> verl-style: each rank hosts vLLM AsyncLLM inside a Ray actor
    #                that also runs a uvicorn HTTP server; the trainer drives it
    #                via Ray RPC (VLLMHttpEngine). Same 8×TP=1 DP layout as verl's
    #                rollout replicas (tensor_model_parallel_size=1).
    transport: str = "fifo"
    # ray_http-only knobs:
    ray_http_base_port: int = 8700  # actor i listens on base_port + local_rank
    ray_http_start_server: bool = True  # also launch uvicorn OpenAI server in actor
    # cumem sleep/wake (verl alignment). On ROCm (cumem_available=False) the
    # gpu_worker patch falls back to KV-cache-only sleep with weights resident;
    # on CUDA with cumem, level 2 offloads weights too.
    enable_sleep_mode: bool = True
    sleep_level: int = 2
    # ZMQ CUDA-IPC bucketed weight transfer (verl BucketedWeightSender/Receiver).
    update_weights_bucket_megabytes: int = 512
    use_shm: bool = False  # use shared-memory buffer instead of CUDA IPC (NPU)
    # Sampling defaults (overridable per-algorithm in the trainer).
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1


@dataclass
class TrainingConfig:
    megatron_cfg: MegatronConfig = field(default_factory=MegatronConfig)
    fsdp_cfg: dict | None = None
    # Compute (forward/backward) dtype used by FSDP2 MixedPrecisionPolicy
    # param_dtype. The optimizer ALWAYS keeps FP32 master weights + Adam
    # moments regardless of this value, so small updates (lr ~1e-6) are not
    # lost to bf16 rounding. Use "bf16" for mixed precision (recommended) or
    # "fp32" for full FP32. Reduce dtype is always FP32 for stability.
    optimizer_dtype: str = "bf16"


@dataclass
class GenerationConfig:
    atom_cfg: AtomConfig = field(default_factory=AtomConfig)
    vllm_cfg: VLLMConfig = field(default_factory=VLLMConfig)


@dataclass
class PolicyConfig:
    model_name: str = ""
    training_backend: str = TrainingBackend.FSDP2.value
    generation_backend: str = GenerationBackend.ATOM.value
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    max_total_sequence_length: int = 4096
    max_response_length: int = 20480
    # See atomrl.core.config.PolicyConfig.max_prompt_length.
    max_prompt_length: int = 0
    train_global_batch_size: int = 64
    # Number of *prompt* sequences sampled per generation round for DAPO dynamic
    # sampling (verl: data.gen_batch_size). 0 = same as the number of prompts
    # implied by train_global_batch_size (no over-sampling). When DAPO
    # filter_groups is enabled this should be a multiple of the train prompt
    # count (e.g. 3x) so degenerate groups can be filtered out.
    gen_batch_size: int = 0
    train_micro_batch_size: int = 8
    max_token_len_per_gpu: int = 0
    ppo_mini_batch_size: int = 0
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-6
    lr_warmup_steps: int = 10
    # See atomrl.core.config.PolicyConfig.lr_warmup_init.
    lr_warmup_init: float = 0.0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    sgd_momentum: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.0
    min_lr: float = 0.0
    lr_decay_style: str = "cosine"  # constant, linear, cosine, WSD
    wsd_decay_ratio: float = 0.2
    wsd_decay_style: str = "cosine"
    balance_batch: bool = False  # seqlen-balanced partitioning across DP ranks


@dataclass
class GRPOConfig:
    num_generations: int = 8
    kl_coeff: float = 0.0
    clip_ratio: float = 0.2
    num_ppo_epochs: int = 1
    num_mini_batches: int = 1
    discount: float = 1.0


@dataclass
class FilterGroupsConfig:
    """DAPO dynamic-sampling filter (verl recipe/dapo filter_groups).

    When enabled the trainer over-samples prompts (``policy.gen_batch_size``),
    drops prompt groups whose per-prompt ``metric`` has zero std (all-correct or
    all-wrong → zero GRPO advantage), and keeps generating up to
    ``max_num_gen_batches`` rounds until ``train_batch_size`` valid prompt groups
    are collected.
    """

    enable: bool = False
    metric: str = "acc"  # acc / score / seq_reward / seq_final_reward
    max_num_gen_batches: int = 0  # <=0 means unlimited rounds


@dataclass
class OverlongBufferConfig:
    """DAPO soft overlong-buffer reward shaping (verl reward_manager/dapo).

    penalty = min(-(resp_len - (max_resp_len - len)) / len * penalty_factor, 0)
    so the penalty ramps linearly from 0 to ``-penalty_factor`` across the last
    ``len`` tokens before ``max_resp_len``.
    """

    enable: bool = False
    len: int = 0
    penalty_factor: float = 0.0
    log: bool = False


@dataclass
class DAPOConfig:
    num_generations: int = 8
    kl_coeff: float = 0.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    clip_ratio_c: float = 3.0
    dynamic_sampling: bool = True
    token_level_pg: bool = True
    overlong_reward_shaping: bool = True
    loss_mode: str = (
        "token_level"  # "token_level" (standard DAPO) or "gmpo" (geometric mean PO)
    )
    discount: float = 1.0
    # verl-faithful dynamic sampling + soft overlong shaping.
    filter_groups: FilterGroupsConfig = field(default_factory=FilterGroupsConfig)
    overlong_buffer: OverlongBufferConfig = field(default_factory=OverlongBufferConfig)
    # Max response length used by the soft overlong buffer (tokens). 0 = use
    # policy.max_response_length.
    max_resp_len: int = 0


@dataclass
class PPOConfig:
    kl_coeff: float = 0.02
    clip_ratio: float = 0.2
    num_ppo_epochs: int = 4
    num_mini_batches: int = 4
    gae_lambda: float = 0.95
    discount: float = 1.0


@dataclass
class OPDConfig:
    """On-Policy Distillation (DeepSeek-V4 style)."""

    kl_direction: str = "reverse"
    temperature: float = 1.0
    position_weighting: bool = False
    position_decay: float = 0.8
    opd_coeff: float = 1.0
    lazy_logits: bool = True
    teacher_micro_batch_size: int = 4


@dataclass
class SpecDistillConfig:
    """Speculative Decoding draft model distillation."""

    draft_type: str = "eagle3"
    loss_type: str = "forward_kl"
    position_decay: float = 0.8
    loss_decay_gamma: float = 7.0
    num_target_layers: int = 1
    aux_hidden_state_layer_ids: list[int] | None = None
    anchor_num: int = 512
    spec_length: int = 5


@dataclass
class TeacherConfig:
    """Teacher / target model configuration."""

    model_name: str = ""
    key: str = ""  # routing key for multi-teacher
    lm_head_key: str = "lm_head.weight"
    norm_key: str = "model.norm.weight"
    load_norm: bool = False
    inference_backend: str = "hf"  # "hf" | "atom" | "vllm"
    quantization: str = ""  # "" | "fp8" | "fp4" | "mxfp4"
    tensor_parallel_size: int = 1  # ATOM tensor parallelism
    gpu_ids: list[int] | None = None  # GPUs for ATOM inference
    transport: str = "mooncake"  # "mooncake" | "mori"
    atom: Any = None  # ATOM config extra args
    # MORI-IO P2P RDMA for GPU-direct hidden state transfer
    mori_io_host: str = "127.0.0.1"  # OOB communication address
    mori_io_port: int = 0  # 0 = auto-assign
    mori_io_qp_per_transfer: int = 2  # RDMA queue pairs per transfer

    def __post_init__(self) -> None:
        if self.inference_backend not in {"hf", "atom", "vllm"}:
            raise ValueError(
                "Unsupported teacher inference backend "
                f"{self.inference_backend!r}; choose 'hf', 'atom', or 'vllm'."
            )


@dataclass
class DraftModelConfig:
    """Draft model (student) configuration for speculative distillation."""

    model_name: str = ""
    from_scratch: bool = False
    head_dim: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    ffn_dim: int | None = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling_type: str | None = None
    rope_scaling_factor: float = 64.0
    rope_original_max_pos: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    # Llama3-specific RoPE (nvidia/gpt-oss-120b-Eagle3 layout)
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    # HF eagle_config toggles
    use_aux_hidden_state: bool = True
    use_input_layernorm_in_first_layer: bool = True
    use_last_layernorm: bool = True
    use_mtp_layernorm: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    max_window_layers: int | None = None
    dtype: str = "float16"
    resume_from: str | None = None


@dataclass
class DistillationConfig:
    """Multi-teacher distillation configuration."""

    enabled: bool = False
    teacher_key: str = "data_source"  # field name in dataset used to route samples
    teachers: dict = field(default_factory=dict)  # key -> teacher config overrides
    loss_mode: str = "reverse_kl"  # k1, k3, forward_kl, reverse_kl
    topk: int | None = None  # for top-k distillation losses
    use_task_rewards: bool = False  # combine with task rewards
    distillation_loss_coef: float = 1.0  # coefficient for distillation loss


@dataclass
class AlgorithmConfig:
    name: str = AlgorithmName.GRPO.value
    adv_estimator: str = ""  # empty = auto-infer from algorithm.name
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    opd: OPDConfig = field(default_factory=OPDConfig)
    spec_distill: SpecDistillConfig = field(default_factory=SpecDistillConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    draft: DraftModelConfig = field(default_factory=DraftModelConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    # KL controller settings (verl/trainer/ppo/core_algos.py L193-212)
    kl_ctrl_type: str = "fixed"  # "fixed" | "adaptive"
    kl_penalty: str = "kl"  # kl, abs, mse, k3, k3+, etc.
    kl_target: float = 0.01  # target KL for adaptive controller
    kl_horizon: int = 10000  # horizon for adaptive controller
    use_kl_in_reward: bool = (
        False  # apply KL penalty to token rewards before advantages
    )
    # Policy loss registry (verl/trainer/ppo/core_algos.py L50-85)
    loss_mode: str = ""  # "" = use algorithm default; or "vanilla", "gspo", etc.
    loss_agg_mode: str = (
        "token-mean"  # token-mean, seq-mean-token-sum, seq-mean-token-sum-norm, seq-mean-token-mean
    )
    # Unified clip ratios for policy loss registry
    clip_ratio: float = 0.2
    clip_ratio_low: float | None = None
    clip_ratio_high: float | None = None
    clip_ratio_c: float = 3.0
    # Rollout correction (also available via quantization.rollout_correction)
    rollout_correction: RolloutCorrectionConfig | None = None


@dataclass
class RolloutQuantConfig:
    precision: str = "bf16"
    use_deep_gemm: bool = True
    num_first_layers_in_bf16: int = 0
    num_last_layers_in_bf16: int = 0


@dataclass
class TrainingQuantConfig:
    fp8: str | None = None
    fp8_recipe: str = "blockwise"
    fp8_weight_cache: bool = False
    lumen_norm: bool = False
    fused_mlp: bool = False
    fused_rope: bool = False
    lumen_linear: bool = False
    hf_attn_patch: bool = False


@dataclass
class RolloutCorrectionConfig:
    enabled: bool = False
    method: str = "tis"
    clip: float = 1.5
    # IS weights (verl rollout_corr_helper.py)
    rollout_is: str | None = ""  # "token" | "sequence" | "" | None
    rollout_is_threshold: str = "2.0"  # float or "lower_upper" (IcePop)
    rollout_is_batch_normalize: bool = False
    # Rejection sampling (11 criteria: token_k1/k2/k3, seq_sum/mean/max_k1/k2/k3)
    rollout_rs: str = ""  # comma-separated: "seq_mean_k1", "seq_mean_k3", etc.
    rollout_rs_threshold: str = ""  # comma-separated thresholds; K1 uses "lower_upper"
    # Bypass mode: set pi_old = pi_rollout, skip old_log_prob computation
    bypass_mode: bool = False
    loss_type: str = "ppo_clip"  # "ppo_clip" | "reinforce" (bypass mode only)


@dataclass
class QuantizationConfig:
    rollout: RolloutQuantConfig = field(default_factory=RolloutQuantConfig)
    training: TrainingQuantConfig = field(default_factory=TrainingQuantConfig)
    rollout_correction: RolloutCorrectionConfig = field(
        default_factory=RolloutCorrectionConfig
    )


@dataclass
class R3Config:
    # Production R3 is vLLM + hard_assignment only. ``assert_supported_r3``
    # fail-closes ATOM backends and replay_mode=distribution.
    enabled: bool = False
    record_router_logits: bool = False
    replay_mode: str = "hard_assignment"
    # Rollout Routing Replay (arXiv 2510.11370): take the top-k expert ids the
    # ROLLOUT engine actually selected and replay them in the training forward
    # and backward, instead of re-deriving routing from the trainer's own
    # hidden states. ``enabled`` above is the older trainer-internal replay
    # (old-logprob forward -> update), which is a different mechanism.
    #
    # This is the only thing that removes train/rollout expert-selection flips
    # rather than making them less likely: on Qwen3-30B-A3B, 6.4% of
    # (token, layer) decisions change under a mere bf16->fp32 router recompute,
    # so only ~4% of tokens route identically through all 48 layers, and forcing
    # the gate to fp32 on both sides measured as no improvement at all.
    rollout_replay: bool = False
    # Print ``r3_verify/*`` (same-weight train vs rollout log-prob mismatch).
    # Off by default; independent of ``enabled``.
    print_train_rollout_mismatch: bool = False


@dataclass
class MoEConfig:
    r3: R3Config = field(default_factory=R3Config)


@dataclass
class RewardConfig:
    type: str = "function"
    function: str = "math_reward"
    dataset: str = ""
    dataset_split: str = "train"
    model_name: str | None = None
    # verl alignment: verl's dataloader shuffles the training prompts with a
    # torch.Generator seeded by ``data.seed`` (RandomSampler). Lumen historically
    # read prompts sequentially, which fed a *different* prompt set per step and
    # diverged from verl on step-1 metrics. Enable a verl-equivalent shuffle so
    # both frameworks consume the same prompt order. ``shuffle_seed=None`` falls
    # back to the top-level ``config.seed``.
    shuffle: bool = True
    shuffle_seed: int | None = None
    # See atomrl.core.config.RewardConfig.scorer.
    scorer: str = "dapo"
    num_workers: int = 8


@dataclass
class DatasetConfig:
    """Dataset preprocessing configuration for speculative distillation."""

    chat_template: str = ""
    last_turn_loss_only: str = "false"  # "true", "false", or "auto"
    min_loss_tokens: int = 0
    num_preprocess_workers: int = 16
    cache_dir: str = "/dev/shm/atomrl_cache"


@dataclass
class EvalConfig:
    """Validation / evaluation configuration."""

    enabled: bool = False
    interval: int = 1000
    num_samples: int = 256
    micro_batch_size: int = 8
    num_generations: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1


@dataclass
class CriticConfig:
    """Configuration for the critic (value) network used by PPO/GAE."""

    enabled: bool = False
    model_name: str = ""
    training_backend: str = "fsdp2"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    value_clip_ratio: float = 0.2
    num_critic_epochs: int = 1


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "results/default"
    save_steps: int = 50
    save_total_limit: int = 3
    resume: bool = True


@dataclass
class WandbConfig:
    project: str = "atomrl"
    name: str = ""
    entity: str | None = None


@dataclass
class LoggerConfig:
    wandb_enabled: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_interval: int = 1
    num_val_samples_to_print: int = 5


@dataclass
class MooncakeTransferConfig:
    """Mooncake distributed KV store for hidden state transfer."""

    master_server_address: str | None = None
    metadata_server: str | None = None
    local_hostname: str = ""
    protocol: str = "rdma"
    device_name: str = ""
    global_segment_size: str = "16GB"
    local_buffer_size: str = "4GB"
    host_buffer_size: int = 536870912  # 512 MB
    gpu_buffer_size: int = 536870912
    async_put_pool_size: int = 4
    enable_gpu_direct: bool = False
    enable_hard_pin: bool = False
    kv_lease_ttl_s: float = 120.0
    get_retry_wait_seconds: float = 1.0
    get_retry_max_wait_seconds: float = 90.0


@dataclass
class RDMAWeightSyncConfig:
    """RCCL/RoCE transport settings for cross-node policy weight updates."""

    backend: str = "rccl"
    require_rdma: bool = True
    hca: str = "mlx5_0"
    interface: str = "ens11np0"
    gid_index: int = 3
    gdr_mode: str = "auto"  # off | auto | required


@dataclass
class WeightSyncConfig:
    """Policy weight transport between separated training and rollout nodes."""

    # auto preserves the legacy selection; production choices are
    # shared_folder and rdma.
    backend: str = "auto"  # auto | shared_folder | rdma
    shared_folder: str = "/volumes/oss1/atomrl_weight_sync"
    bucket_size_mb: int = 1024
    timeout_s: int = 600
    verify_full_load: bool = True
    fp8_quantize: bool = (
        False  # Quantize BF16 weights to FP8 per-block before sync (halves transfer size)
    )
    fp8_quantization_location: str | None = (
        None  # trainer | inference; None uses legacy fp8_quantize
    )
    rdma: RDMAWeightSyncConfig = field(default_factory=RDMAWeightSyncConfig)

    def resolve_fp8_quantize(self) -> bool:
        """Return whether weight-sync copies are quantized on the trainer."""
        location = self.fp8_quantization_location
        if location is None:
            return bool(self.fp8_quantize)
        if location not in {"trainer", "inference"}:
            raise ValueError(
                "weight_sync.fp8_quantization_location must be "
                f"'trainer' or 'inference', got {location!r}"
            )
        if self.fp8_quantize and location != "trainer":
            raise ValueError(
                "weight_sync.fp8_quantization_location='inference' conflicts "
                "with legacy weight_sync.fp8_quantize=true"
            )
        return location == "trainer"

    def validate_fp8_quantization_location(
        self,
        rollout_quantization: str | None,
    ) -> None:
        """Validate cross-component requirements for an explicit FP8 location."""
        location = self.fp8_quantization_location
        if location is None and not self.fp8_quantize:
            return
        self.resolve_fp8_quantize()
        if self.backend != "rdma":
            raise ValueError(
                "weight_sync.fp8_quantization_location requires "
                "weight_sync.backend='rdma'"
            )
        if rollout_quantization != "fp8_per_block":
            raise ValueError(
                "weight_sync.fp8_quantization_location requires "
                "policy.generation.vllm_cfg.quantization='fp8_per_block'"
            )


@dataclass
class AsyncTrainingConfig:
    """Configuration for fully-async separated rollout + training."""

    enabled: bool = False
    require_batches: int = 4
    trigger_parameter_sync_step: int = 4
    staleness_threshold: float = 0.0
    partial_rollout: bool = False
    use_rollout_log_probs: bool = True
    rollout_n_gpus: int = 0
    trainer_n_gpus: int = 0
    queue_maxsize: int = 64
    weight_sync_dir: str = "/tmp/atomrl_weight_sync"


@dataclass
class TorchProfilerToolConfig:
    """Configuration for torch.profiler backend."""

    # Supported values: "cpu", "cuda", "memory", "shapes", "stack"
    contents: list[str] = field(default_factory=lambda: ["cpu", "cuda"])


@dataclass
class RocprofToolConfig:
    """Configuration for ROCm `rocprof` command-line profiling."""

    # Trace toggles.
    hip_trace: bool = True
    hsa_trace: bool = True
    kernel_trace: bool = False
    memory_copy_trace: bool = False
    sys_trace: bool = False
    timestamp_on: bool = True

    # Optional summary/statistics dump.
    stats: bool = False

    # Output control.
    output_file: str = "rocprof_trace"
    output_format: str = "csv"  # csv | json

    # Optional kernel filter (regex), only when kernel tracing is enabled.
    kernel_regex: str | None = None

    # Extra raw CLI arguments appended at the end.
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ProfilerConfig:
    """Global profiler configuration for trainer/controller process.

    Example (rocprof):

    ```yaml
    profiler:
      tool: rocprof
      enable: true
      all_ranks: false
      ranks: [0]
      save_path: outputs/profile
      steps: [10, 20, 30]
      profile_continuous_steps: false
      tool_config:
        hip_trace: true
        hsa_trace: true
        kernel_trace: true
        memory_copy_trace: true
        sys_trace: false
        timestamp_on: true
        stats: true
        output_file: rocprof_trace
        output_format: csv
        kernel_regex: null
        extra_args: []
    ```
    """

    tool: str = "torch"
    enable: bool = False
    all_ranks: bool = False
    ranks: list[int] = field(default_factory=list)
    save_path: str = "outputs/profile"
    steps: list[int] | None = None
    profile_continuous_steps: bool = False
    # NOTE: typed ``Any`` (not a Union of dataclasses) because OmegaConf's
    # structured-config schema does not support unions of containers. The
    # profiler consumers (utils/profiler.py) re-validate via isinstance and fall
    # back to the appropriate default tool config, so YAML still works.
    tool_config: Any = field(default_factory=TorchProfilerToolConfig)


@dataclass
class AtomRLConfig:
    """Top-level configuration for AtomRL."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    mooncake: MooncakeTransferConfig = field(default_factory=MooncakeTransferConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    async_training: AsyncTrainingConfig = field(default_factory=AsyncTrainingConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    assembly: RuntimeAssemblyConfig = field(default_factory=RuntimeAssemblyConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    num_training_steps: int = 1000
    seed: int = 42
    val_dataset: str = ""
    val_steps: int = 0  # validate every N steps; 0 = no validation
    val_batch_size: int = 16

    def __post_init__(self) -> None:
        # Fail-close unsupported R3 combinations at construction. Reuses the
        # single source of truth in ``r3_scope`` (imported directly, not via
        # ``atomrl.moe``, to avoid cycling through ``R3Manager``). Runs on
        # ``OmegaConf.to_object`` too, so ``from_yaml`` needs no extra check.
        if not self.moe.r3.enabled:
            return
        from atomrl.moe.r3_scope import assert_supported_r3

        assert_supported_r3(
            replay_mode=self.moe.r3.replay_mode,
            generation_backend=self.policy.generation_backend,
        )

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: list[str] | None = None
    ) -> AtomRLConfig:
        """Load config from YAML file with optional CLI overrides."""
        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, file_cfg)
        if overrides:
            cli_cfg = OmegaConf.from_dotlist(overrides)
            merged = OmegaConf.merge(merged, cli_cfg)
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    @classmethod
    def from_cli(cls) -> AtomRLConfig:
        """Parse config from command-line arguments."""
        parser = argparse.ArgumentParser(description="AtomRL")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to YAML config"
        )
        args, unknown = parser.parse_known_args()
        return cls.from_yaml(args.config, overrides=unknown)


def load_config(
    config_path: str | Path, overrides: list[str] | None = None
) -> AtomRLConfig:
    """Convenience function to load a AtomRLConfig."""
    return AtomRLConfig.from_yaml(config_path, overrides=overrides)
