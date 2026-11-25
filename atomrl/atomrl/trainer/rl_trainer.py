# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""High-level RL training orchestration on the controller process.

Supports three modes:
- **Sync colocate** (default): rollout and training share the same GPUs,
  swapping memory between vLLM and FSDP2 via optimizer offload.
- **Async separated**: Rollouter and Trainer run on separate GPU groups with
  a message queue and periodic parameter sync (see ``AsyncRLTrainer``).
- **Local mode** (single-GPU / testing): all workers run in the controller process.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

import atomrl.algorithms  # noqa: F401  — populate ALGORITHM_REGISTRY
from atomrl.controller import (
    DispatchMode,
    RayCluster,
    RayWorkerGroup,
    create_fused_worker_cls,
)
from atomrl.controller.dispatch import dispatch_proto
from atomrl.core.config import AtomRLConfig
from atomrl.core.protocol import DataProto
from atomrl.core.registry import ALGORITHM_REGISTRY
from atomrl.engine.training.base_engine import BaseEngine, EngineRegistry
from atomrl.quantization.rollout_correction import apply_rollout_correction
from atomrl.trainer.callbacks import Callback, LoggingCallback
from atomrl.utils.metrics import MetricsTracker
from atomrl.utils.profiler import DistProfiler
from atomrl.workers import AtomActorWorker, RefPolicyWorker

logger = logging.getLogger(__name__)
ATOMRL_DEBUG = os.environ.get("ATOMRL_DEBUG", "0") in ("1", "true", "True")


def _ray_boundary_errors() -> tuple[type[BaseException], ...]:
    """Expected local and remote failures from Ray worker-group RPCs."""
    errors: list[type[BaseException]] = [
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
    ]
    try:
        from ray.exceptions import RayError
    except (AttributeError, ImportError):
        return tuple(errors)
    errors.append(RayError)
    return tuple(errors)


def _algo_num_generations(config: AtomRLConfig) -> int:
    name = config.algorithm.name.lower()
    if name == "ppo":
        return 1
    if name == "dapo":
        return int(config.algorithm.dapo.num_generations)
    return int(config.algorithm.grpo.num_generations)


def _response_token_ids(
    sequence: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """Return response IDs from a left- or right-padded rollout row."""
    real_positions = attention_mask.nonzero(as_tuple=True)[0]
    if real_positions.numel() == 0:
        return sequence[:0]
    response_start = int(real_positions[0].item()) + int(prompt_length)
    real_end = int(real_positions[-1].item()) + 1
    return sequence[min(response_start, real_end) : real_end]


def _merge_left_padded(parts: list[torch.Tensor], value: float) -> torch.Tensor:
    """Stack left-padded row blocks, widening each to the widest on the left.

    Real tokens sit at the right end of every row (what pack_sequences expects),
    so the padding a narrower block needs goes in front.
    """
    width = max(t.shape[1] for t in parts)
    widened = []
    for t in parts:
        if t.shape[1] < width:
            pad = torch.full(
                (t.shape[0], width - t.shape[1]),
                value,
                dtype=t.dtype,
                device=t.device,
            )
            t = torch.cat([pad, t], dim=1)
        widened.append(t)
    return torch.cat(widened, dim=0)


def _format_chat_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """Format chat messages without silently dropping model turn markers."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    convert_token = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_token):
        user_id = convert_token("<｜User｜>")
        assistant_id = convert_token("<｜Assistant｜>")
        if user_id is not None and assistant_id is not None:
            # DeepSeek-V4 intentionally ships a Python encoder rather than a
            # Jinja chat template. vLLM carries the model's reference encoder.
            from vllm.tokenizers.deepseek_v4_encoding import encode_messages

            return encode_messages(messages, thinking_mode="thinking")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


class _RayRolloutGroups:
    """Prompt groups on the Ray vLLM replicas, the backend of ``StreamingRollout``.

    All methods run on the rollout thread. A group's completions all go to one
    replica, the one with the fewest groups in flight.
    """

    def __init__(
        self, trainer: RLTrainer, num_generations: int, use_filter: bool
    ) -> None:
        self._trainer = trainer
        self._num_generations = num_generations
        self._use_filter = use_filter
        self._mgr = trainer._ray_rollout_mgr
        self._servers = self._mgr.servers
        want_lp = bool(trainer.config.policy.generation.vllm_cfg.calculate_log_probs)
        self._sampling_params = trainer._ray_sampling_params(want_lp)
        self._load = [0] * len(self._servers)
        self._groups: dict[Any, tuple[int, str]] = {}

    def submit(self, index: int) -> Any:
        prompts, gts = self._trainer._get_prompts_range(index, 1)
        [prompt_ids] = self._trainer._ray_rollout_prompt_ids(prompts)
        replica = min(range(len(self._servers)), key=self._load.__getitem__)
        handle = self._servers[replica].generate_group.remote(
            prompt_ids,
            self._num_generations,
            self._sampling_params,
        )
        self._load[replica] += 1
        self._groups[handle] = (replica, gts[0])
        return handle

    def wait(self, handles: list[Any], timeout: float) -> list[Any]:
        import ray

        ready, _ = ray.wait(
            handles,
            num_returns=len(handles),
            timeout=timeout,
            fetch_local=False,
        )
        return ready

    def finish(self, handle: Any) -> tuple | None:
        import ray

        from atomrl.algorithms.dapo_sampling import filter_groups_keep_mask

        replica, gt = self._groups.pop(handle)
        self._load[replica] -= 1
        sequences, seq_mask, prompt_lengths, rollout_lp, ragged = (
            self._trainer._assemble_ray_rollout(ray.get(handle))
        )
        gts = [gt] * self._num_generations
        rewards, responses, accs = self._trainer._compute_rewards_full(
            sequences,
            seq_mask,
            prompt_lengths,
            gts,
            verbose=False,
        )
        if self._use_filter:
            _, kept = filter_groups_keep_mask(accs, [0] * self._num_generations)
            if not kept:
                return None
        return (
            sequences,
            seq_mask,
            prompt_lengths,
            rewards,
            responses,
            gts,
            rollout_lp,
            ragged,
        )

    def pause(self) -> None:
        self._mgr.pause_all(mode="keep", clear_cache=True)

    def resume(self) -> None:
        self._mgr.resume_all()


class RLTrainer:
    """Coordinates rollout, reference, reward, and actor for RL training.

    In local mode, all computation happens in-process without Ray, using
    ``HFEngine`` for generation and ``FSDP2Backend`` for training.
    """

    def __init__(self, config: AtomRLConfig) -> None:
        self.config = config
        self.global_step: int = 0
        self.last_metrics: dict[str, float] = {}
        self.callbacks: list[Callback] = []
        self._algorithm: Any = None
        self._metrics = MetricsTracker()

        self._engine: BaseEngine | None = None
        self._actor_model: torch.nn.Module | None = None
        self._ref_model: torch.nn.Module | None = None
        self._ref_on_cpu: bool = True
        self._optimizer: torch.optim.Optimizer | None = None
        self._tokenizer: Any = None
        self._dataset: Any = None
        self._atom_engine: Any = None
        self._gen_backend: str = config.policy.generation_backend.lower()
        self._use_vllm: bool = self._gen_backend == "vllm"
        _vcfg = config.policy.generation.vllm_cfg
        # Keep one vLLM resident per GPU and update weights in place (no rebuild).
        self._vllm_persistent: bool = self._use_vllm and bool(
            getattr(_vcfg, "persistent", True)
        )
        # Each rank generates its own shard of the batch on its local GPU.
        self._vllm_dp: bool = self._use_vllm and bool(
            getattr(_vcfg, "data_parallel_rollout", True)
        )
        # ``_use_atom`` is kept as the "external colocated inference engine" flag
        # (offload optimizer during rollout, sleep engine during training,
        # safetensors weight sync). Both ATOM and vanilla vLLM use this path.
        self._use_atom: bool = self._gen_backend in ("atom", "vllm")
        # Ray controller orchestration path is opt-in via config/env/ray address.
        self._use_ray_controller: bool = (
            bool(getattr(config.controller.ray, "enabled", False))
            or os.environ.get("ATOMRL_USE_RAY_CONTROLLER", "0") == "1"
            or bool(getattr(config.cluster, "ray_address", None))
        )
        self._critic_worker: Any = None
        self._kl_ctrl: Any = None
        self._ray_cluster: RayCluster | None = None
        self._actor_wg: RayWorkerGroup | None = None
        self._ref_wg: RayWorkerGroup | None = None
        self._actor_mp: int = 1  # Megatron model-parallel size (TP*PP*CP)
        self._actor_dp_size: int = (
            0  # data-parallel size (0 => not queried; fallback below)
        )
        self._ray_dispatch_state: dict[str, Any] = {}
        self._profiler: DistProfiler | None = None
        self._prev_step_profile: bool = False
        self._curr_step_profile: bool = False
        self._val_dataset: Any = None
        # Running prompt cursor for DAPO dynamic sampling (advances across
        # generation rounds, not just steps).
        self._prompt_cursor: int = 0
        self._prompt_perm: Any = None
        # Background generation of the next step (controller.ray.one_step_off_policy).
        self._rollout_pipeline: Any = None
        # verl-aligned Ray rollout (ray_http transport); populated in setup.
        self._ray_use_vllm: bool = False
        self._ray_use_atom: bool = False
        self._ray_vllm_engine: Any = None
        self._ray_rollout_mgr: Any = None
        self._last_weight_sync_metrics: dict[str, float] = {}
        self._is_distributed: bool = torch.distributed.is_initialized()
        self._rank: int = torch.distributed.get_rank() if self._is_distributed else 0
        self._world_size: int = (
            torch.distributed.get_world_size() if self._is_distributed else 1
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

    def setup(self) -> None:
        """Initialize models, optimizer, dataset, and algorithm."""
        model_name = self.config.policy.model_name
        if not model_name:
            raise ValueError("config.policy.model_name is required.")

        algo_cls: type[Any] = ALGORITHM_REGISTRY.get(self.config.algorithm.name)
        self._algorithm = algo_cls(self.config)

        if self._use_ray_controller:
            self._setup_ray_controller()
            return

        quant = {}
        tq = self.config.quantization.training
        if tq.fp8:
            quant["fp8"] = tq.fp8
        quant["fp8_weight_cache"] = tq.fp8_weight_cache
        quant["lumen_norm"] = tq.lumen_norm
        quant["fused_mlp"] = tq.fused_mlp
        quant["fused_rope"] = tq.fused_rope

        optimizer_dtype_str = getattr(
            self.config.policy.training, "optimizer_dtype", "bf16"
        )
        lr = getattr(self.config.policy, "learning_rate", 1e-6)
        self._base_lr = lr
        self._lr_warmup_steps = getattr(self.config.policy, "lr_warmup_steps", 10)
        self._param_offload = False

        fsdp_cfg_dict: dict = {}
        if hasattr(self.config.policy, "training") and hasattr(
            self.config.policy.training, "fsdp_cfg"
        ):
            _fc = self.config.policy.training.fsdp_cfg
            if isinstance(_fc, dict):
                fsdp_cfg_dict = _fc
                self._param_offload = _fc.get("param_offload", False)

        backend_str = getattr(self.config.policy, "training_backend", "fsdp2").lower()
        if backend_str in ("fsdp", "fsdp2"):
            backend_key = "fsdp2"
        else:
            raise ValueError(
                "The non-Ray trainer supports only policy.training_backend=fsdp2; "
                f"got {backend_str!r}."
            )

        logger.info(
            "[rank %d] Building actor model via Engine layer: %s (backend=%s, optimizer_dtype=%s)",
            self._rank,
            model_name,
            backend_key,
            optimizer_dtype_str,
        )

        # Mixed-precision training (verl-aligned): the optimizer keeps FP32
        # master weights so that small Adam updates (lr ~1e-6) accumulate
        # correctly, while forward/backward compute uses ``optimizer_dtype``
        # (typically bf16) via FSDP2 MixedPrecisionPolicy. Storing the master
        # in bf16 silently drops ~1e-6 updates to rounding (bf16 eps ~8e-3),
        # which stalls learning entirely. Reduce dtype stays FP32.
        compute_dtype_str = optimizer_dtype_str
        master_dtype_str = "fp32"
        engine_config = {
            "param_offload": fsdp_cfg_dict.get("param_offload", False),
            "optimizer_offload": fsdp_cfg_dict.get("optimizer_offload", False),
            "grad_offload": fsdp_cfg_dict.get("grad_offload", False),
            "reshard_after_forward": fsdp_cfg_dict.get("reshard_after_forward", True),
            "model_dtype": master_dtype_str,
            "mixed_precision": {
                "param_dtype": compute_dtype_str,
                "reduce_dtype": "fp32",
            },
            "seed": getattr(self.config, "seed", 42),
        }
        optimizer_config = {
            "lr": lr,
            "weight_decay": getattr(self.config.policy, "weight_decay", 0.01),
            "clip_grad": getattr(self.config.policy, "max_grad_norm", 1.0),
            "lr_scheduler_type": getattr(
                self.config.policy, "lr_decay_style", "cosine"
            ),
            "lr_warmup_steps": self._lr_warmup_steps,
            "lr_warmup_steps_ratio": getattr(self.config.policy, "warmup_ratio", 0.0),
            "total_training_steps": int(self.config.num_training_steps),
        }
        model_config = {
            "local_path": model_name,
            "trust_remote_code": True,
        }

        engine_cls = EngineRegistry.get_engine_cls(
            model_type="language_model",
            backend=backend_key,
        )
        self._engine = engine_cls(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            model_name=model_name,
            quant_config=quant,
        )
        self._engine.initialize()
        self._actor_model = self._engine.module
        self._optimizer = self._engine.optimizer

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        kl_coeff = 0.0
        algo_name = self.config.algorithm.name.lower()
        if algo_name == "dapo":
            kl_coeff = self.config.algorithm.dapo.kl_coeff
        elif algo_name == "grpo":
            kl_coeff = self.config.algorithm.grpo.kl_coeff
        elif algo_name == "ppo":
            kl_coeff = self.config.algorithm.ppo.kl_coeff

        if kl_coeff > 0.0:
            logger.info(
                "[rank %d] Loading reference model (kl_coeff=%.4f): %s",
                self._rank,
                kl_coeff,
                model_name,
            )
            self._ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            self._ref_model.eval()
            for p in self._ref_model.parameters():
                p.requires_grad_(False)
            self._ref_on_cpu = True
        else:
            self._ref_model = None
            self._ref_on_cpu = True
            logger.info("[rank %d] Skipping reference model (kl_coeff=0).", self._rank)

        if self._use_vllm:
            from atomrl.engine.inference.vllm_engine import VLLMEngine

            vllm_cfg = self.config.policy.generation.vllm_cfg
            # verl alignment: seed the rollout engine with the top-level seed when
            # the config doesn't override it, so vLLM sampling is reproducible and
            # matched to verl (which uses replica_rank + data.seed per engine).
            if getattr(vllm_cfg, "seed", None) is None:
                vllm_cfg.seed = int(getattr(self.config, "seed", 42))
            self._atom_engine = VLLMEngine(config=vllm_cfg, model_name=model_name)
            logger.info(
                "[rank %d] vLLM engine configured (lazy init on first rollout, "
                "calculate_log_probs=%s).",
                self._rank,
                vllm_cfg.calculate_log_probs,
            )
        elif self._use_atom:
            from atomrl.engine.inference.atom_engine import AtomEngine

            atom_cfg = self.config.policy.generation.atom_cfg
            self._atom_engine = AtomEngine(config=atom_cfg, model_name=model_name)
            logger.info(
                "[rank %d] ATOM engine configured (lazy init on first rollout).",
                self._rank,
            )

        self._load_dataset()

        # ---- Validation dataset ----
        val_path = getattr(self.config, "val_dataset", "")
        if val_path:
            self._load_val_dataset(val_path)

        if not self.callbacks:
            self.callbacks.append(
                LoggingCallback(interval=max(1, self.config.logger.log_interval))
            )

        # ---- Critic worker (value model for PPO/GAE) ----
        if getattr(self.config, "critic", None) and self.config.critic.enabled:
            from atomrl.workers import CriticWorker

            critic_config_dict = {
                "critic": {
                    "model_name": self.config.critic.model_name
                    or self.config.policy.model_name,
                    "training_backend": self.config.critic.training_backend,
                    "learning_rate": self.config.critic.learning_rate,
                    "weight_decay": self.config.critic.weight_decay,
                    "max_grad_norm": self.config.critic.max_grad_norm,
                    "value_clip_ratio": self.config.critic.value_clip_ratio,
                },
                "policy": {
                    "model_name": self.config.critic.model_name
                    or self.config.policy.model_name,
                    "training": (
                        vars(self.config.policy.training)
                        if hasattr(self.config.policy.training, "__dict__")
                        else {}
                    ),
                    "seed": getattr(self.config, "seed", 42),
                },
            }
            self._critic_worker = CriticWorker(
                self._rank, self._world_size, critic_config_dict
            )
            self._critic_worker.init_model()
            logger.info("[rank %d] CriticWorker initialized.", self._rank)

        # --- KL controller (verl/trainer/ppo/ray_trainer.py L316) ---
        algo_cfg = self.config.algorithm
        _kl_coef = 0.0
        algo_name_lc = algo_cfg.name.lower()
        if algo_name_lc == "dapo":
            _kl_coef = algo_cfg.dapo.kl_coeff
        elif algo_name_lc == "grpo":
            _kl_coef = algo_cfg.grpo.kl_coeff
        elif algo_name_lc == "ppo":
            _kl_coef = algo_cfg.ppo.kl_coeff
        if _kl_coef > 0.0 and algo_cfg.use_kl_in_reward:
            from atomrl.algorithms.kl_controller import get_kl_controller

            self._kl_ctrl = get_kl_controller(
                kl_ctrl_type=algo_cfg.kl_ctrl_type,
                kl_coef=_kl_coef,
                target_kl=algo_cfg.kl_target,
                horizon=algo_cfg.kl_horizon,
            )
            logger.info(
                "[rank %d] KL controller: type=%s, coef=%.4f, use_kl_in_reward=True",
                self._rank,
                algo_cfg.kl_ctrl_type,
                _kl_coef,
            )

        self._resume_step = 0
        if getattr(self.config.checkpointing, "resume", True):
            self._try_resume_checkpoint()

        if self._is_distributed:
            torch.distributed.barrier()

        logger.info(
            "[rank %d] RLTrainer.setup complete: algo=%s, model=%s, world_size=%d, atom=%s, resume_step=%d",
            self._rank,
            self.config.algorithm.name,
            model_name,
            self._world_size,
            self._use_atom,
            self._resume_step,
        )
        self._init_profiler()

    def _rendezvous_ray_group(self, wg: RayWorkerGroup, timeout_s: int = 7200) -> None:
        """Form a cross-actor torch.distributed group so FSDP2 shards + syncs grads.

        verl-aligned: pick the master address from actor rank 0's node plus a
        free port, then have every actor join a single ``world_size``-rank NCCL
        group (each pinned to its own GPU as ``local_rank=0``). Must run BEFORE
        ``init_model`` so the FSDP backend sees an initialized process group and
        applies ``fully_shard`` instead of falling back to an unsharded replica.
        """
        import ray

        n = wg.num_workers
        if n <= 0:
            return
        master_addr = wg.call_single(0, "get_node_ip")
        master_port = wg.call_single(0, "find_free_port")
        # Default Ray pinning: one GPU per actor, remapped to cuda:0 → local_rank=0.
        # MORI (and other intra-node peer-access paths) need every actor to see
        # all node GPUs (CUDA_VISIBLE_DEVICES=0..N-1) and bind cuda:rank instead.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        n_visible = len([x for x in visible.split(",") if x.strip()]) if visible else 0
        bind_by_rank = n_visible >= n
        refs = [
            wg.call_single_async(
                i,
                "setup_distributed",
                rank=i,
                world_size=n,
                master_addr=master_addr,
                master_port=master_port,
                local_rank=i if bind_by_rank else 0,
                timeout_s=timeout_s,
            )
            for i in range(n)
        ]
        ray.get(refs)
        logger.info(
            "Ray rendezvous complete: %d distributed actors on %s:%s.",
            n,
            master_addr,
            master_port,
        )

    def _compute_actor_mp(self) -> int:
        """Total model-parallel size (TP x PP x CP) of a Megatron actor.

        FSDP is pure DP (mp=1). Megatron actor engines consume
        ``megatron_cfg.tensor_model_parallel_size`` /
        ``pipeline_model_parallel_size`` / ``context_parallel_size``.
        """
        backend = str(getattr(self.config.policy, "training_backend", "")).lower()
        if backend not in (
            "megatron_native",
            "megatron-native",
            "megatron",
            "megatron_lumen_dsv4",
        ):
            return 1
        try:
            meg = self.config.policy.training.megatron_cfg
            tp = int(
                getattr(meg, "tensor_model_parallel_size", None)
                or getattr(meg, "tensor_parallel_size", 1)
                or 1
            )
            pp = int(getattr(meg, "pipeline_model_parallel_size", 1) or 1)
            cp = int(getattr(meg, "context_parallel_size", 1) or 1)
        except (AttributeError, TypeError, ValueError):
            tp = pp = cp = 1
        return max(1, tp * pp * cp)

    def _setup_ray_vllm_rollout(self, model_name: str, vcfg: Any) -> None:
        """Build verl-style colocated vLLM rollout replicas + client (ray_http)."""
        from atomrl.engine.inference.vllm_http_engine import VLLMHttpEngine
        from atomrl.engine.inference.vllm_ray_server import VLLMReplicaManager

        self.config.weight_sync.validate_fp8_quantization_location(
            getattr(vcfg, "quantization", None),
        )
        seed = self.config.seed if getattr(vcfg, "seed", None) is None else vcfg.seed
        tp = int(getattr(vcfg, "tensor_parallel_size", 1) or 1)
        engine_kwargs: dict[str, Any] = {
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": float(vcfg.gpu_memory_utilization),
            "dtype": str(vcfg.dtype),
            "enforce_eager": bool(vcfg.enforce_eager),
            "disable_custom_all_reduce": bool(vcfg.disable_custom_all_reduce),
            "enable_chunked_prefill": bool(vcfg.enable_chunked_prefill),
            "enable_prefix_caching": bool(vcfg.enable_prefix_caching),
            "max_num_batched_tokens": int(vcfg.max_num_batched_tokens),
            "max_num_seqs": int(vcfg.max_num_seqs),
            "trust_remote_code": bool(vcfg.trust_remote_code),
            "enable_sleep_mode": bool(vcfg.enable_sleep_mode),
            "disable_log_stats": True,
        }
        # "" is the same sentinel as "auto": both mean "let vLLM select". Passing
        # "" through reaches vLLM as an explicit request for a backend named "",
        # which an unquantized MoE model rejects outright.
        if str(vcfg.moe_backend) not in ("auto", ""):
            engine_kwargs["moe_backend"] = str(vcfg.moe_backend)
        if str(vcfg.linear_backend) not in ("auto", ""):
            engine_kwargs["linear_backend"] = str(vcfg.linear_backend)
        if bool(getattr(self.config.moe.r3, "enabled", False)):
            # Patched vLLM/MILES captures [seq_len-1, num_layers, top_k]
            # expert ids on the rollout engine. Fail-fast validation happens
            # when assembling each response below.
            engine_kwargs["enable_return_routed_experts"] = True
        if vcfg.max_model_len:
            engine_kwargs["max_model_len"] = int(vcfg.max_model_len)
        if vcfg.kv_cache_dtype and vcfg.kv_cache_dtype != "auto":
            engine_kwargs["kv_cache_dtype"] = str(vcfg.kv_cache_dtype)
        if vcfg.quantization:
            engine_kwargs["quantization"] = str(vcfg.quantization)
            # Online vLLM quant (fp8_per_block, ...) quantizes every Linear; pass
            # quantization_config.ignore so MoE gates stay BF16 for enable_fp32_moe_router.
            from atomrl.moe.router_precision import vllm_quantization_config

            quant_cfg = vllm_quantization_config(engine_kwargs["quantization"])
            if quant_cfg is not None:
                engine_kwargs["quantization_config"] = quant_cfg
        if self._r3_rollout_replay:
            # makes the engine attach the selected expert ids to every completion
            engine_kwargs["enable_return_routed_experts"] = True

        colocation_wg = getattr(self, "_rollout_wg", None) or self._actor_wg
        if colocation_wg is not self._actor_wg:
            logger.info(
                "vLLM replicas will be placed on rollout workers (separation mode)."
            )
        mgr = VLLMReplicaManager(
            colocation_wg,
            model_name,
            engine_kwargs,
            base_port=int(vcfg.ray_http_base_port),
            start_http=bool(vcfg.ray_http_start_server),
            max_concurrency=max(8, int(vcfg.max_num_seqs)),
            base_seed=(int(seed) if seed is not None else None),
        )
        mgr.create()
        self._ray_rollout_mgr = mgr
        self._ray_vllm_engine = VLLMHttpEngine(
            mgr,
            sleep_level=int(vcfg.sleep_level),
            enable_sleep=bool(vcfg.enable_sleep_mode),
        )
        weight_backend = str(getattr(self.config.weight_sync, "backend", "auto"))
        if weight_backend == "rdma":
            rdma_cfg = self.config.weight_sync.rdma
            group_name = f"lumen-rdma-weights-{os.getpid()}"
            mgr.init_rdma_weight_group(
                self._actor_wg,
                interface=str(rdma_cfg.interface),
                hca=str(rdma_cfg.hca),
                require_rdma=bool(rdma_cfg.require_rdma),
                timeout_s=int(self.config.weight_sync.timeout_s),
                group_name=group_name,
            )
        logger.info(
            "Ray vLLM rollout ready: %d replicas (TP=%d, separation=%s, weight_sync=%s).",
            mgr.num_replicas,
            tp,
            colocation_wg is not self._actor_wg,
            weight_backend,
        )

    def _release_actor_cached_memory(self) -> None:
        """Drop the actors' cached-but-unused device memory before an ATOM replica
        sizes its KV cache.

        ATOM subtracts everything the driver reports as occupied on the device
        (``non_torch``) from ``gpu_memory_utilization x total``, so the post-shard
        allocator cache of a colocated FSDP2 actor -- 113.6 GiB for
        Qwen3-30B-A3B -- would otherwise be charged against the KV budget and
        drive it negative. vLLM only charges its own increments, which is why
        this is needed on the ATOM path alone.
        """
        if self._actor_wg is None:
            return
        stats = self._actor_wg.execute_all_sync("free_cached_memory")
        freed_gb = sum(
            (s.get("reserved_before_bytes", 0.0) - s.get("reserved_after_bytes", 0.0))
            for s in stats
        ) / (1 << 30)
        logger.info(
            "Released actor allocator cache before ATOM rollout: %.2f GiB over %d ranks.",
            freed_gb,
            len(stats),
        )

    def _setup_ray_atom_rollout(
        self, model_name: str, vcfg: Any, atom_cfg: Any
    ) -> None:
        """Build colocated ATOM rollout replicas + client on the Ray controller path."""
        from atomrl.engine.inference.atom_ray_server import ATOMReplicaManager
        from atomrl.engine.inference.vllm_http_engine import VLLMHttpEngine

        seed = self.config.seed if getattr(vcfg, "seed", None) is None else vcfg.seed
        max_model_len = getattr(atom_cfg, "max_model_len", None) or getattr(
            vcfg, "max_model_len", None
        )
        quant_cfg = getattr(atom_cfg, "online_quant_config", None)
        vllm_quant = str(getattr(vcfg, "quantization", "") or "")
        if quant_cfg is None and vllm_quant in {"fp8_per_block", "per_block_fp8"}:
            quant_cfg = {"global_quant_config": "per_block_fp8"}

        engine_kwargs: dict[str, Any] = {
            "model": model_name,
            "tensor_parallel_size": int(
                getattr(atom_cfg, "tensor_parallel_size", 1) or 1
            ),
            "data_parallel_size": int(getattr(atom_cfg, "data_parallel_size", 1) or 1),
            "enable_expert_parallel": int(
                getattr(atom_cfg, "expert_parallel_size", 1) or 1
            )
            > 1,
            "gpu_memory_utilization": float(
                getattr(atom_cfg, "gpu_memory_utilization", None)
                or vcfg.gpu_memory_utilization
            ),
            "max_num_batched_tokens": int(vcfg.max_num_batched_tokens),
            "max_num_seqs": int(vcfg.max_num_seqs),
            "enforce_eager": bool(vcfg.enforce_eager),
            "trust_remote_code": bool(vcfg.trust_remote_code),
            "enable_chunked_prefill": bool(vcfg.enable_chunked_prefill),
            "enable_prefix_caching": bool(
                getattr(atom_cfg, "enable_prefix_caching", False)
            ),
        }
        kv_cache_dtype = str(getattr(atom_cfg, "kv_cache_dtype", "auto") or "auto")
        if kv_cache_dtype != "auto":
            engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if max_model_len:
            engine_kwargs["max_model_len"] = int(max_model_len)
        if quant_cfg:
            engine_kwargs["online_quant_config"] = quant_cfg

        extra = getattr(atom_cfg, "engine_kwargs", {}) or {}
        if extra:
            engine_kwargs.update(dict(extra))

        colocation_wg = getattr(self, "_rollout_wg", None) or self._actor_wg
        if colocation_wg is not self._actor_wg:
            logger.info(
                "ATOM replicas will be placed on rollout workers (separation mode)."
            )
        mgr = ATOMReplicaManager(
            colocation_wg,
            model_name,
            engine_kwargs,
            max_concurrency=max(8, int(vcfg.max_num_seqs)),
            base_seed=(int(seed) if seed is not None else None),
        )
        mgr.create()
        self._ray_rollout_mgr = mgr
        self._ray_vllm_engine = VLLMHttpEngine(
            mgr,
            sleep_level=int(vcfg.sleep_level),
            enable_sleep=bool(vcfg.enable_sleep_mode),
        )
        weight_backend = str(getattr(self.config.weight_sync, "backend", "auto"))
        if weight_backend == "rdma":
            # Built once at startup and held for the run: the RCCL group is
            # persistent, so per-step sync is a broadcast rather than a
            # rendezvous. Mirrors _setup_ray_vllm_rollout.
            rdma_cfg = self.config.weight_sync.rdma
            group_name = f"lumen-atom-rdma-weights-{os.getpid()}"
            mgr.init_rdma_weight_group(
                self._actor_wg,
                interface=str(rdma_cfg.interface),
                hca=str(rdma_cfg.hca),
                require_rdma=bool(rdma_cfg.require_rdma),
                timeout_s=int(self.config.weight_sync.timeout_s),
                group_name=group_name,
            )
        logger.info(
            "Ray ATOM rollout ready: %d colocated replicas (TP=%d, online_quant=%s, weight_sync=%s).",
            mgr.num_replicas,
            int(engine_kwargs["tensor_parallel_size"]),
            engine_kwargs.get("online_quant_config"),
            weight_backend,
        )

    @property
    def _r3_rollout_replay(self) -> bool:
        """Whether to carry rollout expert selections into the training forward."""
        moe = getattr(self.config, "moe", None)
        r3 = getattr(moe, "r3", None) if moe is not None else None
        return bool(getattr(r3, "rollout_replay", False))

    @property
    def _r3_print_train_rollout_mismatch(self) -> bool:
        """Whether to print same-weight train vs rollout log-prob mismatch."""
        moe = getattr(self.config, "moe", None)
        r3 = getattr(moe, "r3", None) if moe is not None else None
        return bool(getattr(r3, "print_train_rollout_mismatch", False))

    def _rollout_with_ray_vllm(
        self,
        prompts: list[str],
        num_generations: int,
        sampling_params: dict[str, Any] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Token-in/token-out DP rollout across colocated vLLM replicas.

        Returns ``(sequences, seq_mask, prompt_lengths, rollout_log_probs,
        rollout_routed_experts)``. R3 routes use the MILES contract
        ``[batch, seq_len-1, num_layers, top_k]`` and align with the left-padded
        token/log-prob positions.
        """
        want_lp = bool(self.config.policy.generation.vllm_cfg.calculate_log_probs)
        expanded = [
            ids
            for ids in self._ray_rollout_prompt_ids(prompts)
            for _ in range(num_generations)
        ]
        sp = (
            dict(sampling_params)
            if sampling_params is not None
            else self._ray_sampling_params(want_lp)
        )
        if self._ray_vllm_engine.is_sleeping:
            logger.info(
                "Ray rollout engine sleeping before generation; refreshing weights before wake."
            )
            self._sync_weights_ipc()
        results = self._ray_vllm_engine.generate_tokens(expanded, sp)
        return self._assemble_ray_rollout(results)

    def _ray_rollout_prompt_ids(self, prompts: list[str]) -> list[list[int]]:
        """Token ids the Ray rollout sends for each prompt.

        Send pre-tokenized prompt_token_ids (verl's token-in path), NOT the
        prompt string. A controlled A/B (identical prompts/params/seed/base
        weights) showed vLLM's INTERNAL string tokenization diverges from the
        token-in path even when the offline HF ids are identical: the string
        path generated ~15% longer responses with 2.4x more cap-hits. Feeding
        HF-tokenized ids (add_special_tokens=False, matching verl's
        apply_chat_template(tokenize=True)) makes response length match verl.
        """
        max_prompt_len = min(
            int(self.config.policy.max_total_sequence_length) // 2,
            1024,
        )
        return [
            list(
                self._tokenizer(
                    p,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_prompt_len,
                )["input_ids"]
            )
            for p in prompts
        ]

    def _assemble_ray_rollout(
        self,
        results: list[dict[str, Any]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.Tensor | None,
        dict[str, list[Any]],
    ]:
        """Left-pad the replicas' token-out results into one block of rows.

        Returns ``(sequences, seq_mask, prompt_lengths, rollout_log_probs,
        ragged)`` in the order of ``results``.
        """
        vcfg = self.config.policy.generation.vllm_cfg
        pad_id = self._tokenizer.pad_token_id or 0
        want_lp = bool(vcfg.calculate_log_probs)
        # Megatron MILES R3 uses ``enabled`` while the FSDP replay path uses
        # ``rollout_replay``. Both consume the same per-completion expert ids,
        # so retain the ragged payload for either mode.
        want_r3 = bool(
            getattr(self.config.moe.r3, "enabled", False) or self._r3_rollout_replay
        )

        # Assemble left-padded prompt + response into a single tensor block.
        seqs_tok: list[list[int]] = []
        plens: list[int] = []
        resp_lps: list[list[float]] = []
        response_routes: list[torch.Tensor] = []
        for res in results:
            p_ids = res["prompt_token_ids"]
            g_ids = res["token_ids"]
            seqs_tok.append(list(p_ids) + list(g_ids))
            plens.append(len(p_ids))
            if want_lp:
                resp_lps.append(res.get("logprobs") or [0.0] * len(g_ids))
            if want_r3:
                raw_routes = res.get("routed_experts")
                if raw_routes is None:
                    raise RuntimeError(
                        "moe.r3.enabled=true but vLLM returned no routed_experts. "
                        "The rollout image must include the MILES vLLM patch and "
                        "enable_return_routed_experts support."
                    )
                routes = torch.as_tensor(raw_routes)
                expected = len(p_ids) + len(g_ids) - 1
                # Some vLLM revisions expose the final, loss-unused token too.
                if routes.shape[0] == expected + 1:
                    routes = routes[:expected]
                if routes.ndim != 3 or routes.shape[0] != expected:
                    raise ValueError(
                        "vLLM routed_experts shape mismatch: expected "
                        f"[{expected}, num_layers, top_k], got {tuple(routes.shape)}"
                    )
                response_routes.append(routes.to(dtype=torch.int16, device="cpu"))

        max_len = max(len(s) for s in seqs_tok)
        b = len(seqs_tok)
        sequences = torch.full((b, max_len), pad_id, dtype=torch.long)
        seq_mask = torch.zeros((b, max_len), dtype=torch.long)
        prompt_lengths: list[int] = []
        for i, s in enumerate(seqs_tok):
            # left-pad so real tokens sit at the right end (matches pack_sequences)
            start = max_len - len(s)
            sequences[i, start:] = torch.tensor(s, dtype=torch.long)
            seq_mask[i, start:] = 1
            # _build_response_mask expects the REAL prompt token count (it locates
            # the first non-pad position itself), NOT the absolute start offset.
            prompt_lengths.append(plens[i])

        rollout_lp = None
        if want_lp:
            rollout_lp = torch.zeros((b, max_len - 1), dtype=torch.float32)
            for i, s in enumerate(seqs_tok):
                start = max_len - len(s)
                # response log-probs align to target positions after the prompt
                resp_start = start + plens[i] - 1
                lps = resp_lps[i]
                for j, lp in enumerate(lps):
                    col = resp_start + j
                    if 0 <= col < max_len - 1:
                        rollout_lp[i, col] = lp

        sequences = sequences.to(self._device)
        seq_mask = seq_mask.to(self._device)
        if rollout_lp is not None:
            rollout_lp = rollout_lp.to(self._device)

        ragged: dict[str, list[Any]] = {}
        if want_r3:
            # Use the validated/copied tensors assembled above. In particular,
            # revisions that expose the final loss-unused route were trimmed to
            # ``sequence_length - 1`` there; re-reading the raw results here
            # would silently discard that normalization.
            ragged["rollout_routing"] = response_routes
        return sequences, seq_mask, prompt_lengths, rollout_lp, ragged

    def _ray_sampling_params(self, want_logprobs: bool) -> dict[str, Any]:
        """Sampling params matching _rollout_with_vllm for cross-transport parity."""
        vcfg = self.config.policy.generation.vllm_cfg
        algo_name = self.config.algorithm.name.lower()
        max_total = int(
            getattr(self.config.policy, "max_total_sequence_length", 0) or 0
        )
        max_resp = int(getattr(self.config.policy, "max_response_length", 0) or 0)
        max_tok = max_resp if max_resp > 0 else max(128, max_total // 2)
        sp: dict[str, Any] = {
            "max_tokens": max_tok,
            "temperature": float(vcfg.temperature),
            "top_p": float(vcfg.top_p),
            "top_k": int(vcfg.top_k),
        }
        if algo_name in ("dapo", "grpo") and vcfg.temperature == 0.0:
            sp["temperature"] = 1.0
        if want_logprobs:
            sp["logprobs"] = 0
        return sp

    def _ray_eval_sampling_params(self) -> dict[str, Any]:
        """Validation sampling params, independently configurable from rollout."""
        ecfg = self.config.eval
        max_resp = int(getattr(self.config.policy, "max_response_length", 0) or 0)
        max_total = int(
            getattr(self.config.policy, "max_total_sequence_length", 0) or 0
        )
        return {
            "max_tokens": max_resp if max_resp > 0 else max(128, max_total // 2),
            "temperature": float(getattr(ecfg, "temperature", 0.0)),
            "top_p": float(getattr(ecfg, "top_p", 1.0)),
            "top_k": int(getattr(ecfg, "top_k", -1)),
        }

    def _ipc_endpoints_match_actors(self, mgr) -> bool:
        """Can every actor find an IPC peer sitting on its own GPU?

        The sender addresses ``replica rank//TP, tp-rank rank%TP``
        (``actor_worker.update_weights_ipc_send``), so what matters is whether
        the rollout side publishes a socket per TP worker:

        * vLLM's ``vLLMColocateWorkerExtension._get_zmq_handle`` keys on
          ``self.local_rank``, so a TP=N replica exposes N endpoints and any
          ``replicas * TP == actors`` layout pairs up one-to-one.
        * ATOM's ``ATOMRayServer._get_zmq_handle`` is hardcoded to ``rank-0``,
          i.e. one endpoint per replica whatever TP is, so only TP=1 works
          there. That single-backend limitation used to gate both engines,
          which silently sent every TP>1 vLLM run down the safetensors path --
          54.8 GB through /dev/shm per step on the 4-layer DSv4 slice.
        """
        workers = int(self._actor_wg.num_workers)
        replicas = int(getattr(mgr, "num_replicas", 0) or 0)
        if replicas == workers:
            return True
        tp = int(getattr(mgr, "tensor_parallel_size", 1) or 1)
        return bool(getattr(self, "_ray_use_vllm", False)) and replicas * tp == workers

    def _sync_weights_ipc(self) -> None:
        """verl-aligned weight sync: wake weights -> ZMQ IPC -> wake KV cache.

        The training actors (senders) and colocated vLLM workers (receivers) run
        concurrently: each actor all-gathers its full BF16 weights and streams
        them over CUDA IPC to its own replica's socket.

        When the rollout engine cannot expose one IPC endpoint per actor, falls
        back to a safetensors-on-disk path: actors export HF weights, the engine
        reloads. See ``_ipc_endpoints_match_actors`` for when that applies.
        """
        import ray

        mgr = self._ray_rollout_mgr
        if mgr is None or self._actor_wg is None:
            return

        backend = str(getattr(self.config.weight_sync, "backend", "auto"))
        if backend == "rdma":
            self._sync_weights_rdma(mgr)
            return
        if backend == "shared_folder":
            self._sync_weights_safetensors(mgr)
            return
        if backend != "auto":
            raise ValueError(
                f"Unsupported weight_sync.backend={backend!r}; "
                "expected auto, shared_folder, or rdma"
            )

        fp8_sync = self.config.weight_sync.resolve_fp8_quantize()
        separated = getattr(self, "_rollout_wg", None) is not None
        if separated or not self._ipc_endpoints_match_actors(mgr):
            if fp8_sync:
                # The safetensors fallback exports BF16 through disk and has no
                # quantization hook, so honouring fp8_quantize here would silently
                # mean "not quantized". Say so instead of degrading in silence.
                raise RuntimeError(
                    "weight_sync.fp8_quantize=true is not supported by the "
                    "safetensors weight-sync fallback (separated="
                    f"{separated}, replicas={mgr.num_replicas}, "
                    f"actors={self._actor_wg.num_workers}). Use "
                    "weight_sync.backend='rdma' for a separated topology, or a "
                    "colocated topology with one replica per actor for CUDA IPC."
                )
            if ATOMRL_DEBUG:
                logger.info(
                    "[DBG] _sync_weights_ipc: separated=%s replicas=%d workers=%d, using safetensors",
                    separated,
                    mgr.num_replicas,
                    self._actor_wg.num_workers,
                )
            self._sync_weights_safetensors(mgr)
            return

        vcfg = self.config.policy.generation.vllm_cfg
        bmb = int(vcfg.update_weights_bucket_megabytes)
        use_shm = bool(vcfg.use_shm)
        sleeping = bool(getattr(self._ray_vllm_engine, "enable_sleep", False))

        rollout_engine = self._ray_vllm_engine

        # Same numbering as the RDMA path, so a mismatch between the two ends is
        # an error here instead of a slow train/rollout divergence later.
        version = int(self.global_step) + 1

        # 1) wake weight memory before loading (only when sleep is in use).
        if sleeping:
            # Both engines wake into whatever the driver reports free -- ATOM its
            # weights, vLLM on ROCm its KV cache (its sleep keeps the weights
            # resident) -- and a Megatron actor keeps its training peak in the
            # allocator cache (198 GiB reserved, 53 GiB live for Qwen3-30B-A3B at
            # EP=8), which left too little for ATOM's weights.
            self._release_actor_cached_memory()
            rollout_engine.wake(tags=["weights"])
        # 2) start receivers + senders concurrently, then join both.
        # Keyword, and only when it is on: `prequantized_fp8` exists on the vLLM
        # server, not on ATOMRayServer.update_weights_from_ipc(use_shm, version).
        # Passing it positionally broke every ATOM rollout (release examples 4, 5
        # and 9) with "TypeError: too many positional arguments" from inside Ray's
        # arg validation -- far from the cause. Off is the default, so this keeps
        # ATOM on its existing two-arg call; on, it is a vLLM-only feature and
        # weight_sync.validate_fp8_quantization_location already requires a vLLM
        # fp8_per_block model.
        recv_kw = {"prequantized_fp8": True} if fp8_sync else {}
        recv = [
            s.update_weights_from_ipc.remote(use_shm, version, **recv_kw)
            for s in mgr.servers
        ]
        send = self._actor_wg.execute_all_async(
            "update_weights_ipc_send",
            bucket_size_mb=bmb,
            use_shm=use_shm,
            version=version,
            fp8_quantize=fp8_sync,
        )
        # Join as they finish rather than senders-then-receivers. The two sides
        # are a ZMQ REQ/REP pair, so a receiver that raises leaves its socket
        # closed with the sender parked in recv() forever: waiting on the
        # senders first turns any receiver-side error into a silent hang that
        # only a stack dump explains. Taking whichever finishes first re-raises
        # the real exception within a bucket's time.
        send_set = set(send)
        send_results: list[dict] = []
        recv_results: list[dict] = []
        pending = list(send) + list(recv)
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            val = ray.get(done[0])
            if done[0] in send_set:
                if isinstance(val, dict):
                    send_results.append(val)
            elif isinstance(val, list):
                recv_results.extend(
                    r for r in val if isinstance(r, dict) and "total_s" in r
                )
        if send_results:

            def _max_metric(rows: list[dict], key: str) -> float:
                return max(float(r.get(key, 0.0) or 0.0) for r in rows)

            metrics = {
                "timing/weight_sync_gather_s": _max_metric(send_results, "gather_s"),
                "timing/weight_sync_send_s": _max_metric(send_results, "send_s"),
                "mem/weight_sync_peak_alloc_gb": _max_metric(
                    send_results, "peak_alloc_gb"
                ),
                "mem/weight_sync_peak_extra_gb": _max_metric(
                    send_results, "peak_extra_gb"
                ),
            }
            from atomrl.engine.inference.bucketed_weight_transfer import _debug_enabled

            debug = _debug_enabled()
            cleanup_keys = [
                "cleanup_gc_s",
                "cleanup_ipc_collect_s",
                "cleanup_empty_cache_s",
            ]
            sender_keys = ["setup_s", "sync_s", "ack_s", "cleanup_s"]
            if debug:
                sender_keys += cleanup_keys
            for key in sender_keys:
                metrics[f"timing/weight_sync_sender_{key}"] = _max_metric(
                    send_results,
                    f"sender_{key}",
                )
            recv_keys = ["wait_s", "load_weights_s"]
            if debug:
                recv_keys += [
                    "setup_s",
                    "load_s",
                    "route_s",
                    "sync_s",
                    "cleanup_s",
                    "post_s",
                    "reset_prefix_cache_s",
                    "total_s",
                    *cleanup_keys,
                ]
            if recv_results:
                for key in recv_keys:
                    metrics[f"timing/weight_sync_recv_{key}"] = _max_metric(
                        recv_results, key
                    )
            self._last_weight_sync_metrics = metrics

            if debug:

                def _fmt(r: dict) -> str:
                    return " ".join(
                        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in sorted(r.items())
                        if isinstance(v, (int, float))
                    )

                logger.info(
                    "weight_sync breakdown v%d\n  send: %s\n  recv: %s",
                    version,
                    "\n        ".join(_fmt(r) for r in send_results),
                    "\n        ".join(_fmt(r) for r in recv_results),
                )
        # 3) wake KV cache so the next rollout can run.
        if sleeping:
            rollout_engine.wake(tags=["kv_cache"])

    def _sync_weights_rdma(self, mgr) -> None:
        """Broadcast HF-named BF16 buckets directly over a persistent RCCL group."""
        import ray

        if self._actor_wg is None:
            return
        if not getattr(mgr, "rdma_group_name", None):
            raise RuntimeError("RDMA weight group is not initialized")
        rollout_engine = self._ray_vllm_engine
        sleeping = bool(getattr(rollout_engine, "enable_sleep", False))
        if sleeping:
            rollout_engine.wake(tags=["weights"])

        version = int(self.global_step) + 1
        cfg = self.config.weight_sync
        fp8_sync = cfg.resolve_fp8_quantize()
        started = time.perf_counter()
        recv_refs = mgr.start_receive_weights_rdma(
            version=version,
            verify_full_load=bool(cfg.verify_full_load),
            prequantized_fp8=fp8_sync,
            online_staging=str(cfg.rdma.online_staging),
        )
        send_refs = self._actor_wg.execute_all_async(
            "send_weights_rdma",
            version=version,
            bucket_size_mb=int(cfg.bucket_size_mb),
            fp8_quantize=fp8_sync,
            integrity_check=(
                os.environ.get("ATOMRL_WEIGHT_SYNC_INTEGRITY", "0") == "1"
            ),
        )
        results = ray.get(
            send_refs + recv_refs,
            timeout=float(cfg.timeout_s),
        )
        self._last_weight_sync_integrity = [
            result.get("integrity")
            for result in results[len(send_refs) :]
            if isinstance(result, dict)
        ]
        sender = next(
            (x for x in results[: len(send_refs)] if x.get("writer")),
            None,
        )
        if sender is None:
            raise RuntimeError("RDMA weight sync returned no source statistics")
        total_s = time.perf_counter() - started
        recv_rows = [
            row
            for result in results[len(send_refs) :]
            for row in (result if isinstance(result, list) else [result])
            if isinstance(row, dict)
        ]
        self._last_weight_sync_metrics = {
            "timing/weight_sync_rdma_s": total_s,
            "weight_sync/bytes": float(sender.get("bytes", 0.0)),
            "weight_sync/gbps": float(sender.get("gbps", 0.0)),
            "weight_sync/buckets": float(sender.get("buckets", 0.0)),
            "weight_sync/version": float(version),
            "weight_sync/backend_rdma": 1.0,
            "weight_sync/fp8_trainer_quantized": float(fp8_sync),
        }
        for key in ("gather_s", "pack_s", "bcast_s", "final_sync_s"):
            self._last_weight_sync_metrics[f"timing/weight_sync_rdma_sender_{key}"] = (
                float(sender.get(key, 0.0) or 0.0)
            )
        for key in (
            "wait_s",
            "recv_s",
            "host_copy_s",
            "observe_s",
            "route_s",
            "load_weights_s",
            "verify_s",
            "finalize_s",
        ):
            if recv_rows:
                self._last_weight_sync_metrics[
                    f"timing/weight_sync_rdma_recv_{key}"
                ] = max(float(r.get(key, 0.0) or 0.0) for r in recv_rows)
        logger.info(
            "RDMA weight sync committed: version=%d buckets=%d bytes=%.1fGB "
            "broadcast=%.2fs effective=%.2fGb/s total=%.2fs fp8_location=%s",
            version,
            int(sender.get("buckets", 0)),
            float(sender.get("bytes", 0)) / 1e9,
            float(sender.get("seconds", 0)),
            float(sender.get("gbps", 0)),
            total_s,
            "trainer" if fp8_sync else "inference",
        )
        logger.info(
            "RDMA weight sync breakdown v%d: %s",
            version,
            " ".join(
                f"{k.removeprefix('timing/weight_sync_rdma_')}={v:.3f}"
                for k, v in self._last_weight_sync_metrics.items()
                if k.startswith("timing/weight_sync_rdma_")
            ),
        )
        if sleeping:
            rollout_engine.wake(tags=["kv_cache"])

    def _sync_weights_safetensors(self, mgr) -> None:
        """Weight sync for TP>1 ATOM: export HF weights to /dev/shm, ATOM reloads."""
        t0 = time.time()
        rollout_engine = self._ray_vllm_engine
        sleeping = bool(getattr(rollout_engine, "enable_sleep", False))

        if self._actor_wg is None:
            return

        backend = str(getattr(self.config.weight_sync, "backend", "auto"))
        configured_dir = (
            str(self.config.weight_sync.shared_folder)
            if backend == "shared_folder"
            else os.environ.get("ATOMRL_WEIGHT_SYNC_DIR", "/dev/shm/atomrl_weight_sync")
        )
        sync_dir = Path(configured_dir)
        sync_dir.mkdir(parents=True, exist_ok=True)
        export_results = self._actor_wg.execute_all_sync(
            "export_state_dict_safetensors",
            sync_dir=str(sync_dir),
        )
        export_meta = export_results[0]

        orig = Path(self.config.policy.model_name)
        for fname in [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "generation_config.json",
        ]:
            src = orig / fname
            if src.exists():
                shutil.copy2(str(src), str(sync_dir / fname))

        save_time = time.time() - t0
        logger.info(
            "Weight sync (safetensors): saved %d params to %s in %.1fs (%.1f GB)",
            int(export_meta["num_params"]),
            sync_dir,
            save_time,
            int(export_meta["total_bytes"]) / 1e9,
        )

        if sleeping:
            rollout_engine.wake(tags=["weights"])
        separated = getattr(self, "_rollout_wg", None) is not None
        if separated and backend != "shared_folder":
            source_url = str(export_meta.get("weight_url") or "")
            if not source_url:
                raise RuntimeError(
                    "Separated rollout requires a training-node weight URL; "
                    "the safetensors writer did not publish one."
                )
            rollout_sync_dir = os.environ.get(
                "ATOMRL_ROLLOUT_WEIGHT_SYNC_DIR",
                "/dev/shm/atomrl_weight_sync_received",
            )
            logger.info(
                "Weight sync (cross-node): %s -> %s",
                source_url,
                rollout_sync_dir,
            )
            mgr.reload_weights_from_http(source_url, rollout_sync_dir)
        else:
            mgr.reload_weights_from_path(str(sync_dir))
        self._last_weight_sync_metrics = {
            "timing/weight_sync_export_s": save_time,
            "weight_sync/bytes": float(export_meta["total_bytes"]),
            "weight_sync/backend_shared_folder": (
                1.0 if backend == "shared_folder" else 0.0
            ),
        }
        if sleeping:
            rollout_engine.wake(tags=["kv_cache"])

    def _setup_ray_controller(self) -> None:
        """Initialize Ray cluster + worker groups for actor/ref orchestration."""
        if str(getattr(self.config.weight_sync, "backend", "auto")) == "rdma":
            rdma_cfg = self.config.weight_sync.rdma
            os.environ["NCCL_IB_DISABLE"] = "0"
            os.environ["NCCL_SOCKET_IFNAME"] = str(rdma_cfg.interface)
            os.environ["NCCL_IB_HCA"] = str(rdma_cfg.hca)
            os.environ["NCCL_IB_GID_INDEX"] = str(rdma_cfg.gid_index)
            gdr_mode = str(rdma_cfg.gdr_mode).lower()
            if gdr_mode == "required":
                os.environ["NCCL_NET_GDR_LEVEL"] = "PHB"
            elif gdr_mode == "off":
                os.environ["NCCL_NET_GDR_LEVEL"] = "LOC"
            elif gdr_mode != "auto":
                raise ValueError(
                    f"weight_sync.rdma.gdr_mode must be off, auto, or required; "
                    f"got {rdma_cfg.gdr_mode!r}"
                )
            if str(rdma_cfg.online_staging) not in ("gpu", "cpu"):
                raise ValueError(
                    f"weight_sync.rdma.online_staging must be gpu or cpu; "
                    f"got {rdma_cfg.online_staging!r}"
                )
        if RayCluster is None or RayWorkerGroup is None:
            raise RuntimeError(
                "Ray controller modules are unavailable in this environment."
            )
        if not self._use_atom:
            raise NotImplementedError(
                "Ray controller path currently requires policy.generation_backend in {atom, vllm}."
            )

        # Main path should not depend on torch.distributed collectives.
        self._is_distributed = False
        self._rank = 0
        self._world_size = 1

        model_name = self.config.policy.model_name
        cfg_dict = self._to_plain_dict(self.config)
        default_workers = max(
            1, self.config.cluster.num_nodes * self.config.cluster.gpus_per_node
        )
        ray_cfg = self.config.controller.ray

        self._ray_cluster = RayCluster(self.config.cluster)
        self._ray_cluster.init()

        kl_coeff = 0.0
        algo_name = self.config.algorithm.name.lower()
        if algo_name == "dapo":
            kl_coeff = self.config.algorithm.dapo.kl_coeff
        elif algo_name == "grpo":
            kl_coeff = self.config.algorithm.grpo.kl_coeff
        elif algo_name == "ppo":
            kl_coeff = self.config.algorithm.ppo.kl_coeff
        actor_role = ray_cfg.actor
        ref_role = ray_cfg.ref
        actor_workers = (
            actor_role.num_workers if actor_role.num_workers > 0 else default_workers
        )
        ref_workers = (
            ref_role.num_workers if ref_role.num_workers > 0 else default_workers
        )
        actor_pool_name = ray_cfg.topology_map.get("actor", "actor")
        ref_pool_name = ray_cfg.topology_map.get("ref", "ref")

        actor_pool = self._ray_cluster.create_pool(
            actor_pool_name,
            num_gpus=max(1, actor_workers),
            process_on_nodes=actor_role.process_on_nodes,
            max_colocate_count=max(1, actor_role.max_colocate_count),
            detached=actor_role.detached,
            topology_tags=actor_role.topology_tags,
        )

        rollout_role = ray_cfg.rollout
        self._rollout_wg = None
        if rollout_role.num_workers > 0:
            from atomrl.workers.base_worker import RolloutPlacementWorker

            rollout_pool = self._ray_cluster.create_pool(
                "rollout",
                num_gpus=rollout_role.num_workers,
                process_on_nodes=rollout_role.process_on_nodes,
                max_colocate_count=max(1, rollout_role.max_colocate_count),
                detached=rollout_role.detached,
                topology_tags=rollout_role.topology_tags,
            )
            self._rollout_wg = RayWorkerGroup(
                worker_cls=RolloutPlacementWorker,
                pool=rollout_pool,
                num_workers=rollout_role.num_workers,
                worker_kwargs={},
            )
            self._rollout_wg.start()
            logger.info(
                "Rollout placement workers started: %d workers (separation mode)",
                rollout_role.num_workers,
            )

        use_ref = kl_coeff > 0.0
        if ray_cfg.fuse_actor_ref and use_ref:
            if ref_workers != actor_workers:
                raise ValueError(
                    "controller.ray.fuse_actor_ref requires actor/ref num_workers to match."
                )
            fused_cls = create_fused_worker_cls(
                {"actor": AtomActorWorker, "ref": RefPolicyWorker},
                name="ActorRefFusedWorker",
            )
            fused_wg = RayWorkerGroup(
                worker_cls=fused_cls,
                pool=actor_pool,
                num_workers=actor_workers,
                worker_kwargs={"config": cfg_dict},
                dispatch_mode=DispatchMode(actor_role.dispatch_mode),
                detached=actor_role.detached,
            )
            fused_wg.start()
            self._rendezvous_ray_group(fused_wg)
            spawned = fused_wg.spawn(["actor", "ref"])
            self._actor_wg = spawned["actor"]
            self._ref_wg = spawned["ref"]
            self._actor_wg.call_all(
                "init_model",
                forward_only=(os.environ.get("ATOMRL_FORWARD_ONLY_INIT", "0") == "1"),
            )
            self._ref_wg.call_all("init_model")
            self._actor_wg.setup_dispatch_collect_info()
            self._ref_wg.setup_dispatch_collect_info()
        else:
            self._actor_wg = RayWorkerGroup(
                worker_cls=AtomActorWorker,
                pool=actor_pool,
                num_workers=actor_workers,
                worker_kwargs={"config": cfg_dict},
                dispatch_mode=DispatchMode(actor_role.dispatch_mode),
                detached=actor_role.detached,
            )
            self._actor_wg.start()
            self._rendezvous_ray_group(self._actor_wg)
            self._actor_wg.call_all(
                "init_model",
                forward_only=(os.environ.get("ATOMRL_FORWARD_ONLY_INIT", "0") == "1"),
            )
            self._actor_wg.setup_dispatch_collect_info()

        # Megatron model-parallel (TP/PP/CP): the actor world is a
        # DP x (TP,PP,CP) mesh. All model-parallel members of one DP shard must
        # receive the SAME data shard, so we build ``mesh_mapping`` from each
        # actor's real DP rank (robust to Megatron's rank ordering) and normalize
        # the loss by pure DP size (num_workers // (tp*pp*cp)), not num_workers.
        self._actor_mp = self._compute_actor_mp()
        if self._actor_mp > 1 and self._actor_wg is not None:
            infos = self._actor_wg.execute_all_sync("get_mp_info")
            mesh = [int(info["dp_rank"]) for info in infos]
            self._actor_dp_size = int(infos[0]["dp_size"])
            actor_role.mesh_mapping = mesh
            logger.info(
                "Megatron model-parallel=%d: actor DP=%d, mesh_mapping=%s",
                self._actor_mp,
                self._actor_dp_size,
                mesh,
            )

        self._try_resume_ray_checkpoint()

        if use_ref and self._ref_wg is None:
            ref_pool = self._ray_cluster.create_pool(
                ref_pool_name,
                num_gpus=max(1, ref_workers),
                process_on_nodes=ref_role.process_on_nodes,
                max_colocate_count=max(1, ref_role.max_colocate_count),
                detached=ref_role.detached,
                topology_tags=ref_role.topology_tags,
            )
            self._ref_wg = RayWorkerGroup(
                worker_cls=RefPolicyWorker,
                pool=ref_pool,
                num_workers=ref_workers,
                worker_kwargs={"config": cfg_dict},
                dispatch_mode=DispatchMode(ref_role.dispatch_mode),
                detached=ref_role.detached,
            )
            self._ref_wg.start()
            self._rendezvous_ray_group(self._ref_wg)
            self._ref_wg.call_all("init_model")
            self._ref_wg.setup_dispatch_collect_info()

        if os.environ.get("ATOMRL_SKIP_ROLLOUT_INIT", "0") == "1":
            self._ray_use_vllm = False
            self._ray_use_atom = False
            self._ray_rollout_mgr = None
            self._ray_vllm_engine = None
            self._atom_engine = None
            logger.info(
                "Actor-only diagnostic setup: skipped tokenizer, rollout "
                "engine, and datasets"
            )
            return

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        vcfg = self.config.policy.generation.vllm_cfg
        atom_cfg = self.config.policy.generation.atom_cfg
        self._ray_use_vllm = (
            self._use_vllm and str(getattr(vcfg, "transport", "fifo")) == "ray_http"
        )
        self._ray_use_atom = (
            self._gen_backend == "atom"
            and str(getattr(atom_cfg, "transport", "fifo")) == "ray_http"
        )
        self._ray_rollout_mgr = None
        self._ray_vllm_engine = None
        if self._ray_use_vllm:
            self._setup_ray_vllm_rollout(model_name, vcfg)
            self._atom_engine = None
        elif self._ray_use_atom:
            self._release_actor_cached_memory()
            self._setup_ray_atom_rollout(model_name, vcfg, atom_cfg)
            self._atom_engine = None
        else:
            from atomrl.engine.inference.atom_engine import AtomEngine

            self._atom_engine = AtomEngine(config=atom_cfg, model_name=model_name)
        if os.environ.get("ATOMRL_SKIP_DATASET_INIT", "0") == "1":
            logger.info("Diagnostic setup: skipped train and validation datasets")
            return
        self._load_dataset()

        # ---- Validation dataset ----
        val_path = getattr(self.config, "val_dataset", "")
        if val_path:
            self._load_val_dataset(val_path)

        if not self.callbacks:
            self.callbacks.append(
                LoggingCallback(interval=max(1, self.config.logger.log_interval))
            )
        logger.info(
            "RLTrainer.setup (ray-controller) complete: algo=%s, model=%s, actor_workers=%d, ref=%s, resume_step=%d",
            self.config.algorithm.name,
            model_name,
            actor_workers,
            self._ref_wg is not None,
            self._resume_step,
        )
        self._init_profiler()

    @staticmethod
    def _to_plain_dict(config: Any) -> dict[str, Any]:
        if is_dataclass(config):
            return asdict(config)
        if isinstance(config, dict):
            return dict(config)
        return dict(vars(config))

    def _init_profiler(self) -> None:
        """Initialize trainer-side profiler dispatcher from config."""
        self._profiler = DistProfiler(rank=self._rank, config=self.config.profiler)
        self._prev_step_profile = False
        self._curr_step_profile = False

    def _is_profile_step(self, step: int) -> bool:
        if self._profiler is None or not self._profiler.check_enable():
            return False
        steps = self.config.profiler.steps
        return True if steps is None else (step in steps)

    def _maybe_start_profile(self, step: int) -> None:
        curr = self._is_profile_step(step)
        self._curr_step_profile = curr
        if not curr or self._profiler is None:
            return
        if self.config.profiler.profile_continuous_steps:
            if not self._prev_step_profile:
                self._profiler.start(profile_step=step)
        else:
            self._profiler.start(profile_step=step)

    def _maybe_stop_profile(self, step: int) -> None:
        if self._profiler is None:
            return
        next_step_profile = self._is_profile_step(step + 1)
        if self._curr_step_profile:
            if self.config.profiler.profile_continuous_steps:
                if not next_step_profile:
                    self._profiler.stop()
            else:
                self._profiler.stop()
        self._prev_step_profile = self._curr_step_profile
        self._curr_step_profile = next_step_profile

    def _load_dataset(self) -> None:
        """Load the training dataset from config."""
        dataset_path = self.config.reward.dataset
        if not dataset_path:
            logger.warning("No dataset configured; using synthetic prompts.")
            self._dataset = None
            return

        from datasets import load_dataset

        if os.path.isfile(dataset_path) or os.path.isdir(dataset_path):
            if dataset_path.endswith(".parquet"):
                self._dataset = load_dataset(
                    "parquet", data_files=dataset_path, split="train"
                )
            elif dataset_path.endswith((".jsonl", ".json")):
                self._dataset = load_dataset(
                    "json", data_files=dataset_path, split="train"
                )
            else:
                self._dataset = load_dataset(dataset_path, split="train")
        else:
            self._dataset = load_dataset(dataset_path, split="train")

        logger.info(
            "Loaded dataset: %d samples from %s", len(self._dataset), dataset_path
        )
        self._build_prompt_permutation()

    def _build_prompt_permutation(self) -> None:
        """Build a verl-equivalent shuffle of the training prompts.

        verl's StatefulDataLoader uses ``RandomSampler`` with a
        ``torch.Generator`` seeded by ``data.seed``. That sampler yields exactly
        ``torch.randperm(N, generator=Generator(seed))`` for the first epoch
        (verified against torchdata). We reproduce that permutation here so Lumen
        consumes the same prompt order as verl. When disabled, the identity
        permutation reproduces the previous sequential behavior.
        """
        self._prompt_perm = None
        if self._dataset is None:
            return
        reward_cfg = getattr(self.config, "reward", None)
        do_shuffle = (
            bool(getattr(reward_cfg, "shuffle", True))
            if reward_cfg is not None
            else True
        )
        if not do_shuffle:
            logger.info("Prompt shuffle disabled; reading dataset sequentially.")
            return
        seed = (
            getattr(reward_cfg, "shuffle_seed", None)
            if reward_cfg is not None
            else None
        )
        if seed is None:
            seed = int(getattr(self.config, "seed", 42))
        n = len(self._dataset)
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        self._prompt_perm = torch.randperm(n, generator=gen).tolist()
        logger.info(
            "Built verl-equivalent prompt shuffle: N=%d seed=%d first8=%s",
            n,
            int(seed),
            self._prompt_perm[:8],
        )

    def _load_val_dataset(self, path: str) -> None:
        """Load validation dataset."""
        from datasets import load_dataset

        if os.path.isfile(path) or os.path.isdir(path):
            if path.endswith(".parquet"):
                self._val_dataset = load_dataset(
                    "parquet", data_files=path, split="train"
                )
            elif path.endswith((".jsonl", ".json")):
                self._val_dataset = load_dataset("json", data_files=path, split="train")
            else:
                self._val_dataset = load_dataset(path, split="train")
        else:
            self._val_dataset = load_dataset(path, split="train")

        logger.info(
            "Loaded validation dataset: %d samples from %s",
            len(self._val_dataset),
            path,
        )

    def _try_resume_checkpoint(self) -> None:
        """Load model + optimizer state from the latest checkpoint if available."""
        from atomrl.utils.checkpoint import CheckpointManager

        ckpt_dir = self.config.checkpointing.checkpoint_dir
        latest = CheckpointManager.get_latest(ckpt_dir)
        if latest is None:
            logger.info(
                "[rank %d] No checkpoint found in %s; training from scratch.",
                self._rank,
                ckpt_dir,
            )
            return

        logger.info("[rank %d] Resuming from checkpoint: %s", self._rank, latest)
        payload = CheckpointManager.load(latest)
        # Checkpoints store verl-style 1-based global steps. The internal
        # training loop is 0-based, so global_step_N resumes at internal step N.
        self._resume_step = int(payload.get("step", 0))

        # Unwrap nested structure: CheckpointManager.save wraps state in
        # {"step": N, "state_dict": {actual_data}}, so model_state_dict and
        # optimizer_state_dict live one level deeper than expected.
        inner = payload.get("state_dict", {})
        if isinstance(inner, dict) and "model_state_dict" in inner:
            logger.info("[rank %d] Unwrapping nested checkpoint structure.", self._rank)
            payload = inner

        model_sd = payload.get("model_state_dict")
        if model_sd and self._actor_model is not None:
            if self._is_distributed:
                try:
                    from torch.distributed.checkpoint.state_dict import (
                        StateDictOptions,
                        set_model_state_dict,
                        set_optimizer_state_dict,
                    )

                    opts = StateDictOptions(full_state_dict=True)
                    set_model_state_dict(self._actor_model, model_sd, options=opts)
                    logger.info("[rank %d] Restored FSDP2 model state.", self._rank)
                except (ImportError, RuntimeError, ValueError) as exc:
                    logger.warning(
                        "[rank %d] FSDP2 set_model_state_dict failed (%s); "
                        "trying load_state_dict.",
                        self._rank,
                        exc,
                    )
                    self._actor_model.load_state_dict(model_sd, strict=False)
            else:
                self._actor_model.load_state_dict(model_sd, strict=False)
                logger.info("[rank %d] Restored model state.", self._rank)

        opt_sd = payload.get("optimizer_state_dict")
        if opt_sd and self._optimizer is not None:
            if self._is_distributed:
                try:
                    from torch.distributed.checkpoint.state_dict import (
                        StateDictOptions,
                        set_optimizer_state_dict,
                    )

                    opts = StateDictOptions(full_state_dict=True)
                    set_optimizer_state_dict(
                        self._actor_model,
                        self._optimizer,
                        opt_sd,
                        options=opts,
                    )
                    logger.info("[rank %d] Restored FSDP2 optimizer state.", self._rank)
                except (ImportError, RuntimeError, ValueError) as exc:
                    logger.warning(
                        "[rank %d] FSDP2 set_optimizer_state_dict failed (%s); "
                        "trying load_state_dict.",
                        self._rank,
                        exc,
                    )
                    try:
                        self._optimizer.load_state_dict(opt_sd)
                    except (RuntimeError, ValueError):
                        logger.warning(
                            "[rank %d] Optimizer state restore failed; using fresh optimizer.",
                            self._rank,
                        )
            else:
                try:
                    self._optimizer.load_state_dict(opt_sd)
                    logger.info("[rank %d] Restored optimizer state.", self._rank)
                except (RuntimeError, ValueError):
                    logger.warning(
                        "[rank %d] Optimizer state restore failed; using fresh optimizer.",
                        self._rank,
                    )

        # Restore FP32 master weights (bf16 optimizer) and LR scheduler position
        # if the checkpoint carried them; without this a bf16-optimizer resume
        # would reinitialise the master copy and drift from the saved model.
        fp32_params = payload.get("fp32_params")
        if (
            fp32_params
            and self._optimizer is not None
            and hasattr(self._optimizer, "fp32_params")
        ):
            try:
                for dst, src in zip(self._optimizer.fp32_params, fp32_params):
                    dst.data.copy_(src.to(dst.data.device, dtype=dst.data.dtype))
                logger.info(
                    "[rank %d] Restored %d FP32 master params.",
                    self._rank,
                    len(fp32_params),
                )
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "[rank %d] FP32 master restore failed (%s).", self._rank, exc
                )
        sched_epoch = payload.get("scheduler_last_epoch")
        if (
            sched_epoch is not None
            and self._optimizer is not None
            and hasattr(self._optimizer, "scheduler")
        ):
            try:
                self._optimizer.scheduler.last_epoch = int(sched_epoch)
            except (TypeError, ValueError):
                pass

        del payload
        gc.collect()
        logger.info(
            "[rank %d] Resume complete. Will start from step %d.",
            self._rank,
            self._resume_step,
        )

    def _find_latest_ray_checkpoint(self) -> tuple[int, str] | None:
        ckpt_dir = Path(self.config.checkpointing.checkpoint_dir)
        if not ckpt_dir.is_dir():
            return None
        tracker = ckpt_dir / "latest_checkpointed_iteration.txt"
        if tracker.exists():
            try:
                step = int(tracker.read_text(encoding="utf-8").strip())
                actor_dir = ckpt_dir / f"global_step_{step}" / "actor"
                if actor_dir.is_dir():
                    return step, str(actor_dir)
            except (OSError, ValueError):
                pass
        best: tuple[int, Path] | None = None
        for child in ckpt_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("global_step_"):
                continue
            try:
                step = int(child.name[len("global_step_") :])
            except ValueError:
                continue
            actor_dir = child / "actor"
            if actor_dir.is_dir() and (best is None or step > best[0]):
                best = (step, actor_dir)
        return (best[0], str(best[1])) if best else None

    def _try_resume_ray_checkpoint(self) -> None:
        if self._actor_wg is None:
            return
        if not getattr(self.config.checkpointing, "resume", True):
            self._resume_step = 0
            return
        latest = self._find_latest_ray_checkpoint()
        if latest is None:
            self._resume_step = 0
            logger.info(
                "No Ray checkpoint found in %s; training from scratch.",
                self.config.checkpointing.checkpoint_dir,
            )
            return
        step, actor_dir = latest
        logger.info("Resuming Ray actor checkpoint from %s (step=%d).", actor_dir, step)
        loaded = self._actor_wg.execute_all_sync("load_checkpoint", actor_dir)
        # Checkpoint directories use verl-style 1-based global steps. The
        # internal loop is 0-based, so global_step_N resumes at internal step N
        # and the next emitted callback line is step=N+1.
        self._resume_step = int(max([step] + [int(x) for x in loaded]))
        logger.info(
            "Ray resume complete. Next training log will be global_step=%d.",
            self._resume_step + 1,
        )

    def _get_batch_prompts(self, step: int) -> tuple[list[str], list[str]]:
        """Get a batch of (prompts, ground_truths) for the current step."""
        g = _algo_num_generations(self.config)
        num_prompts = max(1, self.config.policy.train_global_batch_size // g)
        start = step * num_prompts
        return self._get_prompts_range(start, num_prompts)

    def _get_prompts_range(self, start: int, count: int) -> tuple[list[str], list[str]]:
        """Get ``count`` (prompt, ground_truth) pairs starting at global index ``start``.

        Wraps around the dataset. Used by both the fixed-batch path and the DAPO
        dynamic-sampling regeneration loop (which advances a running cursor).
        """
        if self._dataset is None:
            prompts = [f"What is {start + i} + {start + i + 1}?" for i in range(count)]
            gts = [str(2 * (start + i) + 1) for i in range(count)]
            return prompts, gts

        dataset_len = len(self._dataset)
        indices = [(start + i) % dataset_len for i in range(count)]
        perm = getattr(self, "_prompt_perm", None)
        if perm is not None:
            indices = [perm[idx] for idx in indices]
        samples = [self._dataset[idx] for idx in indices]

        prompts = []
        gts = []
        for s in samples:
            p, gt = self._extract_prompt_gt(s)
            prompts.append(p)
            gts.append(gt)
        return prompts, gts

    def _extract_prompt_gt(
        self,
        s: dict,
        *,
        ensure_answer_format: bool = False,
    ) -> tuple[str, str]:
        """Extract (prompt_text, ground_truth) from a dataset row.

        Shared by training (`_get_prompts_range`) and validation so the chat
        template and ground-truth field (``reward_model.ground_truth``) are
        parsed identically.
        """
        import json as _json

        # "problem"/"expected_answer" are nvidia/OpenMathInstruct-2's fields, the
        # dataset of the configs under configs/.
        prompt_raw = (
            s.get("prompt")
            or s.get("question")
            or s.get("problem")
            or s.get("input")
            or ""
        )
        prompt_messages = None
        if isinstance(prompt_raw, list):
            prompt_messages = prompt_raw
            prompt_text = "\n".join(
                m.get("content", "") for m in prompt_raw if isinstance(m, dict)
            )
        elif isinstance(prompt_raw, str) and prompt_raw.startswith("["):
            try:
                msgs = _json.loads(prompt_raw)
                prompt_messages = msgs if isinstance(msgs, list) else None
                prompt_text = "\n".join(
                    m.get("content", "") for m in msgs if isinstance(m, dict)
                )
            except (_json.JSONDecodeError, TypeError):
                prompt_text = prompt_raw
        else:
            prompt_text = str(prompt_raw)

        rm_raw = s.get("reward_model", {})
        if isinstance(rm_raw, str):
            try:
                rm_raw = _json.loads(rm_raw)
            except (_json.JSONDecodeError, TypeError):
                rm_raw = {}
        if isinstance(rm_raw, dict) and rm_raw.get("ground_truth", "") != "":
            gt = rm_raw.get("ground_truth", "")
        else:
            gt = (
                s.get("answer")
                or s.get("solution")
                or s.get("expected_answer")
                or s.get("target")
                or s.get("label")
                or ""
            )

        if ensure_answer_format and prompt_messages is not None:
            prompt_messages = [
                dict(message) if isinstance(message, dict) else message
                for message in prompt_messages
            ]
            for message in reversed(prompt_messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = str(message.get("content", ""))
                    if "Answer:" not in content:
                        message["content"] = (
                            content
                            + "\n\nPut the final answer on its own line as: "
                            + r"Answer: \boxed{answer}."
                        )
                    break

        if self._tokenizer is not None and prompt_messages is not None:
            prompt_text = _format_chat_prompt(self._tokenizer, prompt_messages)

        return prompt_text, str(gt)

    def _tokenize_prompts(
        self, prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompts to input_ids and attention_mask."""
        max_prompt_len = min(self.config.policy.max_total_sequence_length // 2, 1024)
        encoding = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        return encoding["input_ids"], encoding["attention_mask"]

    def _log_gpu_mem(self, phase: str, step: int) -> None:
        """Log GPU 0 memory at a critical point (rank 0 only)."""
        if self._rank != 0:
            return
        free, total = torch.cuda.mem_get_info(0)
        alloc = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        logger.info(
            "GPU-MEM [step=%d phase=%s] alloc=%.1fGB reserved=%.1fGB free=%.1fGB/%.1fGB",
            step,
            phase,
            alloc,
            reserved,
            free / 1e9,
            total / 1e9,
        )

    def _offload_optimizer_to_cpu(self) -> None:
        """Move optimizer state tensors to CPU to free GPU for ATOM rollout.

        Delegates to Engine.to() when available.
        """
        if self._engine is not None:
            self._engine.to(device="cpu", model=False, optimizer=True, grad=False)
            torch.cuda.empty_cache()
            return
        if self._optimizer is None:
            return
        if self._param_offload:
            torch.cuda.empty_cache()
            return
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                    state[k] = v.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _reload_optimizer_to_gpu(self) -> None:
        """Move optimizer state tensors back to GPU for the next training step.

        Delegates to Engine.to() when available.
        """
        if self._engine is not None:
            self._engine.to(device="cuda", model=False, optimizer=True, grad=False)
            return
        if self._optimizer is None or self._param_offload:
            return
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                    state[k] = v.to(self._device, non_blocking=True)

    def _sync_weights_to_atom(self) -> None:
        """Push updated FSDP2 weights to ATOM engine for next rollout.

        Follows verl's approach: extract per-tensor from FSDP2's DTensor
        state_dict, save to /dev/shm as safetensors (HF format), then
        tell the ATOM subprocess to reload from the new path.

        Sequence (GPU 0 memory-safe):
        1. Rank 0: ensure ATOM is already sleeping (done after generation)
        2. All ranks: FSDP2 ``full_tensor()`` all-gather (needs GPU headroom)
        3. Rank 0: save gathered weights, update ``_weight_dir``
        """
        if self._atom_engine is None:
            return

        t0 = time.time()

        # Non-persistent path frees the engine before gathering weights; the
        # persistent path keeps it resident and reloads in place at the end.
        if (
            not self._vllm_persistent
            and self._rank == 0
            and not self._atom_engine.is_sleeping
        ):
            self._atom_engine.sleep_inprocess()
            logger.info("Weight sync: inference engine released (in-process sleep).")

        if self._is_distributed:
            torch.distributed.barrier()
        torch.cuda.empty_cache()

        cpu_state = self._fetch_actor_cpu_state()
        if cpu_state is None:
            return
        torch.cuda.empty_cache()

        sync_dir = Path(
            os.environ.get(
                "ATOMRL_WEIGHT_SYNC_DIR",
                "/dev/shm/atomrl_weight_sync",
            )
        )

        total_bytes = 0
        if self._rank == 0:
            sync_dir.mkdir(parents=True, exist_ok=True)

            from safetensors.torch import save_file

            max_shard_bytes = 4 * 1024 * 1024 * 1024
            shards: list[dict[str, torch.Tensor]] = [{}]
            current_bytes = 0
            for name, tensor in cpu_state.items():
                t_bytes = tensor.numel() * tensor.element_size()
                if current_bytes + t_bytes > max_shard_bytes and shards[-1]:
                    shards.append({})
                    current_bytes = 0
                shards[-1][name] = tensor
                current_bytes += t_bytes

            weight_map: dict[str, str] = {}
            for i, shard in enumerate(shards, 1):
                fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
                save_file(shard, str(sync_dir / fname))
                for k, v in shard.items():
                    weight_map[k] = fname
                    total_bytes += v.numel() * v.element_size()

            index = {
                "metadata": {"total_size": total_bytes},
                "weight_map": weight_map,
            }
            (sync_dir / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2)
            )

            orig = Path(self.config.policy.model_name)
            for fname in [
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "generation_config.json",
                "vocab.json",
                "merges.txt",
            ]:
                src = orig / fname
                if src.exists():
                    shutil.copy2(str(src), str(sync_dir / fname))

            save_time = time.time() - t0
            logger.info(
                "Weight sync: saved %d params to %s in %.1fs (%.1f GB)",
                len(cpu_state),
                sync_dir,
                save_time,
                total_bytes / 1e9,
            )

            logger.info("Weight sync: weights written to %s.", sync_dir)

        del cpu_state
        gc.collect()

        if self._is_distributed:
            torch.distributed.barrier()

        # All ranks point their engine at the new weights.
        self._atom_engine.set_weight_path(sync_dir)

        if self._vllm_persistent:
            # In-place reload on every rank that runs a resident vLLM (no rebuild).
            if self._vllm_dp or self._rank == 0:
                try:
                    self._atom_engine.reload_weights(str(sync_dir))
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.warning(
                        "Weight sync: in-place reload failed (%s); forcing rebuild next gen.",
                        exc,
                    )
                    self._atom_engine.sleep()
            if self._is_distributed:
                torch.distributed.barrier()

    def _fetch_actor_cpu_state(self) -> dict[str, torch.Tensor] | None:
        """Fetch actor weights as CPU tensors for rollout sync."""
        if self._use_ray_controller:
            if self._actor_wg is None:
                return None
            # full_tensor() inside get_state_dict is an FSDP all-gather collective:
            # invoke on ALL actors concurrently, then keep rank 0's gathered copy.
            states = self._actor_wg.execute_all_sync("get_state_dict")
            state = states[0]
            return {k: v.detach().cpu().contiguous() for k, v in state.items()}

        if self._actor_model is None:
            return None

        from torch.distributed._tensor import DTensor

        sd = self._actor_model.state_dict()
        cpu_state: dict[str, torch.Tensor] = {}
        for name, param in sd.items():
            full = param.full_tensor() if isinstance(param, DTensor) else param
            full = full.detach()
            # Master weights are FP32; the rollout engine (vLLM) runs bf16, so
            # downcast here to halve the sync payload and match its dtype.
            # (Set ATOMRL_SYNC_FP32=1 to keep FP32, e.g. for weight-delta debug.)
            if (
                full.dtype == torch.float32
                and os.environ.get("ATOMRL_SYNC_FP32") != "1"
            ):
                full = full.to(torch.bfloat16)
            cpu_state[name] = full.cpu().contiguous()
        return cpu_state

    def _rollout_with_atom(
        self,
        prompts: list[str],
        num_generations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Generate using vLLM engine (PagedAttention, continuous batching).

        Only rank 0 runs generation; results are broadcast to all ranks.
        vLLM ``generate()`` returns response-only text, so we concatenate
        prompt + response and tokenize the full sequence.
        """
        algo_name = self.config.algorithm.name.lower()
        max_resp = self.config.policy.max_response_length
        if max_resp > 0:
            max_tok = max_resp
        else:
            max_tok = max(128, self.config.policy.max_total_sequence_length // 2)
        sp: dict[str, Any] = {"max_tokens": max_tok}
        # Read sampling params from config (vllm_cfg) so the training rollout
        # matches the configured/verl values instead of hardcoded ones. verl
        # uses temperature=1.0, top_p=1.0, top_k=-1; hardcoding top_p=0.95 here
        # truncated the nucleus and systematically shortened the response tail
        # (response_length/max stuck ~2700 vs verl ~8192 on identical seed/data).
        _vcfg = getattr(self.config.policy.generation, "vllm_cfg", None)
        if _vcfg is not None:
            sp.update(
                {
                    "temperature": float(getattr(_vcfg, "temperature", 1.0) or 1.0),
                    "top_p": float(getattr(_vcfg, "top_p", 1.0) or 1.0),
                    "top_k": int(getattr(_vcfg, "top_k", -1)),
                }
            )
        elif algo_name == "dapo":
            sp.update({"temperature": 1.0, "top_p": 1.0})
        elif algo_name == "grpo":
            sp.update({"temperature": 0.7})
        else:
            sp.update({"temperature": 0.0})

        expanded_prompts = []
        for p in prompts:
            for _ in range(num_generations):
                expanded_prompts.append(p)

        if os.environ.get("ATOMRL_DRY_RUN") == "1":
            target_len = int(os.environ.get("ATOMRL_DRY_RUN_RESP_LEN", "100"))
            mock_resp = (
                "Step: think carefully about this problem. " * (target_len // 7 + 1)
            )[: target_len * 4] + " The answer is \\boxed{42}."
            response_texts = [mock_resp] * len(expanded_prompts)
            full_texts = [p + mock_resp for p in expanded_prompts]
            encoding = self._tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=self.config.policy.max_total_sequence_length,
                return_tensors="pt",
            )
            sequences = encoding["input_ids"].to(self._device)
            seq_mask = encoding["attention_mask"].to(self._device)
            if self._is_distributed:
                torch.distributed.barrier()
                shape_tensor = torch.tensor(
                    list(sequences.shape), device=self._device, dtype=torch.long
                )
                torch.distributed.broadcast(shape_tensor, src=0)
                torch.distributed.broadcast(sequences, src=0)
                torch.distributed.broadcast(seq_mask, src=0)
            prompt_lengths = []
            for p in expanded_prompts:
                p_enc = self._tokenizer(p, return_tensors="pt")
                prompt_lengths.append(p_enc["input_ids"].shape[1])
            return sequences, seq_mask, prompt_lengths

        if self._rank == 0:
            self._atom_engine.wake_up()
            response_texts = self._atom_engine.generate(
                expanded_prompts, sampling_params=sp
            )

            full_texts = [p + r for p, r in zip(expanded_prompts, response_texts)]
            encoding = self._tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=self.config.policy.max_total_sequence_length,
                return_tensors="pt",
            )
            sequences = encoding["input_ids"].to(self._device)
            seq_mask = encoding["attention_mask"].to(self._device)
        else:
            sequences = torch.zeros(1, 1, dtype=torch.long, device=self._device)
            seq_mask = torch.zeros(1, 1, dtype=torch.long, device=self._device)

        if self._is_distributed:
            torch.distributed.barrier()
            shape_tensor = torch.tensor(
                list(sequences.shape), device=self._device, dtype=torch.long
            )
            torch.distributed.broadcast(shape_tensor, src=0)
            if self._rank != 0:
                sequences = torch.zeros(
                    int(shape_tensor[0]),
                    int(shape_tensor[1]),
                    dtype=torch.long,
                    device=self._device,
                )
                seq_mask = torch.zeros_like(sequences)
            torch.distributed.broadcast(sequences, src=0)
            torch.distributed.broadcast(seq_mask, src=0)

        prompt_encoding = self._tokenizer(
            expanded_prompts,
            padding=True,
            truncation=True,
            max_length=min(1024, self.config.policy.max_total_sequence_length // 2),
            return_tensors="pt",
        )
        prompt_lengths = prompt_encoding["attention_mask"].sum(dim=1).tolist()

        return sequences, seq_mask, prompt_lengths

    def _rollout_with_vllm(
        self,
        prompts: list[str],
        num_generations: int,
        eval_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor | None]:
        """Generate with vanilla vLLM, returning rollout log-probs for TIS.

        Builds sequences directly from vLLM token ids (prompt + response, right
        padded) and a shifted ``rollout_log_probs`` tensor ``[B, S-1]`` aligned
        with ``old_log_probs`` so token-level importance sampling correction can
        be applied. Only rank 0 generates; everything is broadcast to all ranks.

        Returns ``(sequences, seq_mask, prompt_lengths, rollout_log_probs)`` where
        ``rollout_log_probs`` is ``None`` when log-probs were not requested.
        """
        algo_name = self.config.algorithm.name.lower()
        vcfg = self.config.policy.generation.vllm_cfg
        max_total = int(self.config.policy.max_total_sequence_length)
        max_resp = int(self.config.policy.max_response_length)
        max_tok = max_resp if max_resp > 0 else max(128, max_total // 2)

        if eval_mode:
            sp = self._ray_eval_sampling_params()
        else:
            sp = {
                "max_tokens": max_tok,
                "temperature": float(vcfg.temperature),
                "top_p": float(vcfg.top_p),
                "top_k": int(vcfg.top_k),
            }
            if algo_name in ("dapo", "grpo") and vcfg.temperature == 0.0:
                sp["temperature"] = 1.0

        expanded_prompts = [p for p in prompts for _ in range(num_generations)]
        pad_id = self._tokenizer.pad_token_id or 0
        want_lp = bool(vcfg.calculate_log_probs) and not eval_mode

        def _wake() -> None:
            self._atom_engine.wake_up()

        def _build(
            results: list[dict],
        ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor | None]:
            """Left-pad vLLM token outputs into (sequences, mask, plens, rollout_lp)."""
            seqs: list[list[int]] = []
            lps: list[list[float]] = []
            plens: list[int] = []
            for r in results:
                p_ids = list(r["prompt_token_ids"])
                r_ids = list(r["token_ids"])
                r_lp = r.get("logprobs")
                budget = max(1, max_total - len(p_ids))
                if len(r_ids) > budget:
                    r_ids = r_ids[:budget]
                    if r_lp is not None:
                        r_lp = r_lp[:budget]
                seqs.append(p_ids + r_ids)
                plens.append(len(p_ids))
                lps.append(r_lp if r_lp is not None else [0.0] * len(r_ids))

            S = max((len(s) for s in seqs), default=1)
            B = len(seqs)
            sequences = torch.full((B, S), pad_id, dtype=torch.long)
            seq_mask = torch.zeros((B, S), dtype=torch.long)
            rlp = (
                torch.zeros((B, max(1, S - 1)), dtype=torch.float32)
                if want_lp
                else None
            )
            # LEFT-pad (real tokens at the right end) to match pack_sequences /
            # unpack_log_probs / _build_response_mask.
            for i, s in enumerate(seqs):
                L = len(s)
                off = S - L
                sequences[i, off : off + L] = torch.tensor(s, dtype=torch.long)
                seq_mask[i, off : off + L] = 1
                if want_lp:
                    plen = plens[i]
                    for j, lp in enumerate(lps[i]):
                        idx = off + plen + j - 1
                        if 0 <= idx < rlp.shape[1]:
                            rlp[i, idx] = lp
            return (
                sequences.to(self._device),
                seq_mask.to(self._device),
                plens,
                (rlp.to(self._device) if rlp is not None else None),
            )

        # ---- Data-parallel rollout: each rank generates its own shard. ----
        if self._vllm_dp and self._is_distributed and self._world_size > 1:
            N = len(expanded_prompts)
            ws = self._world_size
            chunk = (N + ws - 1) // ws
            s_idx = min(N, self._rank * chunk)
            e_idx = min(N, s_idx + chunk)
            local_prompts = expanded_prompts[s_idx:e_idx]
            _wake()
            if local_prompts:
                results = self._atom_engine.generate_with_logprobs(
                    local_prompts,
                    sampling_params=sp,
                    want_logprobs=want_lp,
                )
                lseq, lmask, lplens, llp = _build(results)
            else:
                lseq = torch.zeros(0, 1, dtype=torch.long, device=self._device)
                lmask = torch.zeros(0, 1, dtype=torch.long, device=self._device)
                lplens = []
                llp = (
                    torch.zeros(0, 1, dtype=torch.float32, device=self._device)
                    if want_lp
                    else None
                )
            return self._allgather_vllm(lseq, lmask, llp, lplens, want_lp, pad_id)

        # ---- Single-GPU rollout: rank 0 generates, broadcast to all ranks. ----
        if self._rank == 0:
            _wake()
            results = self._atom_engine.generate_with_logprobs(
                expanded_prompts,
                sampling_params=sp,
                want_logprobs=want_lp,
            )
            sequences, seq_mask, plens, rollout_lp = _build(results)
        else:
            sequences = torch.zeros(1, 1, dtype=torch.long, device=self._device)
            seq_mask = torch.zeros(1, 1, dtype=torch.long, device=self._device)
            rollout_lp = None
            plens = []

        if self._is_distributed:
            torch.distributed.barrier()
            meta_t = torch.tensor(
                [sequences.shape[0], sequences.shape[1], 1 if want_lp else 0],
                device=self._device,
                dtype=torch.long,
            )
            torch.distributed.broadcast(meta_t, src=0)
            B, S, lp_flag = int(meta_t[0]), int(meta_t[1]), int(meta_t[2])
            if self._rank != 0:
                sequences = torch.zeros(B, S, dtype=torch.long, device=self._device)
                seq_mask = torch.zeros(B, S, dtype=torch.long, device=self._device)
                rollout_lp = (
                    torch.zeros(
                        B, max(1, S - 1), dtype=torch.float32, device=self._device
                    )
                    if lp_flag
                    else None
                )
            torch.distributed.broadcast(sequences, src=0)
            torch.distributed.broadcast(seq_mask, src=0)
            if lp_flag:
                torch.distributed.broadcast(rollout_lp, src=0)
            plen_t = (
                torch.tensor(plens, device=self._device, dtype=torch.long)
                if self._rank == 0
                else torch.zeros(B, dtype=torch.long, device=self._device)
            )
            if self._rank == 0 and plen_t.shape[0] != B:
                plen_t = torch.zeros(B, dtype=torch.long, device=self._device)
            torch.distributed.broadcast(plen_t, src=0)
            prompt_lengths = plen_t.tolist()
        else:
            prompt_lengths = plens

        return sequences, seq_mask, prompt_lengths, rollout_lp

    def _allgather_vllm(
        self,
        lseq: torch.Tensor,
        lmask: torch.Tensor,
        llp: torch.Tensor | None,
        lplens: list[int],
        want_lp: bool,
        pad_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor | None]:
        """All-gather per-rank LEFT-padded rollout shards into the full ordered batch.

        Each rank contributes ``lseq[n_r, S_r]`` (+ mask, lp[n_r, S_r-1], plens);
        we pad to global max width (LEFT, prepend) and max row count, all-gather,
        then trim per-rank counts and concat in rank order.
        """
        import torch.distributed as dist

        dev = self._device
        ws = self._world_size

        n_r = lseq.shape[0]
        S_r = lseq.shape[1]
        # Global max width and per-rank counts.
        wmax = torch.tensor([S_r], device=dev, dtype=torch.long)
        dist.all_reduce(wmax, op=dist.ReduceOp.MAX)
        S = int(wmax.item())
        cnt = torch.tensor([n_r], device=dev, dtype=torch.long)
        counts = [torch.zeros_like(cnt) for _ in range(ws)]
        dist.all_gather(counts, cnt)
        counts = [int(c.item()) for c in counts]
        max_n = max(counts) if counts else 0
        if max_n == 0:
            empty = torch.zeros(0, S, dtype=torch.long, device=dev)
            return (
                empty,
                empty.clone(),
                [],
                (torch.zeros(0, max(1, S - 1), device=dev) if want_lp else None),
            )

        def _lpad_cols(t: torch.Tensor, width: int, value) -> torch.Tensor:
            if t.shape[1] >= width:
                return t
            pad = torch.full(
                (t.shape[0], width - t.shape[1]), value, dtype=t.dtype, device=t.device
            )
            return torch.cat([pad, t], dim=1)

        def _pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
            if t.shape[0] >= rows:
                return t
            pad = torch.zeros(
                (rows - t.shape[0], t.shape[1]), dtype=t.dtype, device=t.device
            )
            return torch.cat([t, pad], dim=0)

        seq_pad = _pad_rows(_lpad_cols(lseq, S, pad_id), max_n)
        mask_pad = _pad_rows(_lpad_cols(lmask, S, 0), max_n)
        seq_g = [torch.zeros(max_n, S, dtype=torch.long, device=dev) for _ in range(ws)]
        mask_g = [
            torch.zeros(max_n, S, dtype=torch.long, device=dev) for _ in range(ws)
        ]
        dist.all_gather(seq_g, seq_pad)
        dist.all_gather(mask_g, mask_pad)

        if want_lp:
            lp_pad = _pad_rows(
                _lpad_cols(
                    (
                        llp
                        if llp is not None
                        else torch.zeros(n_r, max(1, S - 1), device=dev)
                    ),
                    S - 1,
                    0.0,
                ),
                max_n,
            )
            lp_g = [
                torch.zeros(max_n, max(1, S - 1), dtype=torch.float32, device=dev)
                for _ in range(ws)
            ]
            dist.all_gather(lp_g, lp_pad)

        plen_local = torch.zeros(max_n, dtype=torch.long, device=dev)
        if n_r > 0:
            plen_local[:n_r] = torch.tensor(lplens, dtype=torch.long, device=dev)
        plen_g = [torch.zeros(max_n, dtype=torch.long, device=dev) for _ in range(ws)]
        dist.all_gather(plen_g, plen_local)

        seqs, masks, lps, plens = [], [], [], []
        for r in range(ws):
            c = counts[r]
            if c == 0:
                continue
            seqs.append(seq_g[r][:c])
            masks.append(mask_g[r][:c])
            if want_lp:
                lps.append(lp_g[r][:c])
            plens.extend(plen_g[r][:c].tolist())
        sequences = torch.cat(seqs, dim=0)
        seq_mask = torch.cat(masks, dim=0)
        rollout_lp = torch.cat(lps, dim=0) if want_lp else None
        return sequences, seq_mask, plens, rollout_lp

    def _compute_rewards_full(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
        gts_expanded: list[str],
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[str], list[float]]:
        """Compute rewards + accuracy on the FULL (unsharded) rollout set.

        Rank 0 decodes and scores; ``rewards`` and ``acc`` are broadcast so every
        rank can run identical dynamic-sampling filtering. Returns
        ``(rewards [N] on device, responses, acc [N] floats)``. ``verbose=False``
        drops the per-call reward and sample log lines, for callers that score
        one prompt group at a time.
        """
        from atomrl.rewards.math_reward import compute_math_reward

        N = int(sequences.shape[0])
        responses: list[str] = []
        response_lengths: list[int] = []
        if self._rank == 0:
            seq_cpu = sequences.cpu()
            mask_cpu = attention_mask.cpu()
            for i in range(N):
                plen = int(prompt_lengths[i]) if i < len(prompt_lengths) else 0
                response_ids = _response_token_ids(seq_cpu[i], mask_cpu[i], plen)
                response_lengths.append(int(response_ids.numel()))
                text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(text)
            rewards_t, details = compute_math_reward(responses, gts_expanded)
            rewards = rewards_t.to(self._device)
            accs = torch.tensor(
                [1.0 if d["acc"] else 0.0 for d in details],
                dtype=torch.float32,
                device=self._device,
            )
            if verbose:
                self._log_rollout_diagnostics(
                    rewards,
                    accs,
                    responses,
                    response_lengths,
                    details,
                    gts_expanded,
                )
        else:
            rewards = torch.zeros(N, dtype=torch.float32, device=self._device)
            accs = torch.zeros(N, dtype=torch.float32, device=self._device)

        if self._is_distributed:
            torch.distributed.broadcast(rewards, src=0)
            torch.distributed.broadcast(accs, src=0)
        return rewards, responses, accs.tolist()

    def _log_rollout_diagnostics(
        self,
        rewards: torch.Tensor,
        accs: torch.Tensor,
        responses: list[str],
        response_lengths: list[int],
        details: list[dict[str, Any]],
        gts_expanded: list[str],
    ) -> None:
        N = len(responses)
        acc_frac = float(accs.mean().item()) if N else 0.0
        logger.info(
            "Rollout reward: N=%d accuracy=%.4f mean=%.4f",
            N,
            acc_frac,
            float(rewards.mean().item()),
        )
        max_response_length = int(
            getattr(self.config.policy, "max_response_length", 0) or 0
        )
        cap_hits = (
            sum(length >= max_response_length for length in response_lengths)
            if max_response_length > 0
            else 0
        )
        invalid = sum(detail.get("pred") == "[INVALID]" for detail in details)
        mean_length = (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
        )
        logger.info(
            "Rollout response diagnostics: tokens_mean=%.1f tokens_max=%d "
            "cap_hits=%d/%d invalid_format=%d/%d",
            mean_length,
            max(response_lengths, default=0),
            cap_hits,
            N,
            invalid,
            N,
        )
        for idx in range(min(2, len(responses), len(details))):
            logger.info(
                "Rollout sample[%d]: tokens=%d pred=%r gt=%r tail=%r",
                idx,
                response_lengths[idx],
                details[idx].get("pred", "N/A"),
                gts_expanded[idx] if idx < len(gts_expanded) else "?",
                responses[idx][-400:],
            )

    def _collect_rollout_batch(
        self,
        step: int,
        num_generations: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.Tensor,
        list[str],
        list[str],
        torch.Tensor | None,
        dict[str, list[Any]],
    ]:
        """Generate the full (unsharded) rollout batch for one training step.

        Handles DAPO dynamic sampling (verl ``filter_groups``): when enabled,
        over-samples ``policy.gen_batch_size`` prompts per round, drops prompt
        groups whose per-prompt ``acc`` has zero variance, and keeps generating
        up to ``filter_groups.max_num_gen_batches`` rounds until
        ``train_global_batch_size // num_generations`` valid prompt groups are
        collected. Returns full-set tensors identical on every rank:
        ``(sequences, seq_mask, prompt_lengths, rewards, responses, gts_expanded,
        rollout_log_probs, rollout_routed_experts)``.
        """
        from atomrl.algorithms.dapo_sampling import filter_groups_keep_mask

        g = num_generations
        pol = self.config.policy
        train_prompts = max(1, pol.train_global_batch_size // g)
        algo_lc = self.config.algorithm.name.lower()
        fg = (
            getattr(self.config.algorithm.dapo, "filter_groups", None)
            if algo_lc == "dapo"
            else None
        )
        use_filter = bool(fg is not None and fg.enable)

        gen_prompts = (
            int(pol.gen_batch_size) if pol.gen_batch_size > 0 else train_prompts
        )
        if not use_filter:
            gen_prompts = train_prompts

        def _one_round(prompts: list[str]):
            rag: dict[str, list[Any]] = {}
            if (
                getattr(self, "_ray_use_vllm", False)
                or getattr(self, "_ray_use_atom", False)
            ) and self._ray_vllm_engine is not None:
                seqs, mask, plen, lp, rag = self._rollout_with_ray_vllm(prompts, g)
            elif self._use_vllm:
                seqs, mask, plen, lp = self._rollout_with_vllm(prompts, g)
            elif self._use_atom and self._atom_engine is not None:
                seqs, mask, plen = self._rollout_with_atom(prompts, g)
                lp = None
            else:
                ids, am = self._tokenize_prompts(prompts)
                seqs, mask, plen = self._rollout_phase(ids, am, g)
                lp = None
            return seqs, mask, plen, lp, rag

        # Simple (no dynamic sampling) path: one round, return everything.
        if not use_filter:
            prompts, gts = self._get_prompts_range(self._prompt_cursor, gen_prompts)
            self._prompt_cursor += gen_prompts
            if ATOMRL_DEBUG:
                logger.info(
                    "[DBG] _collect_rollout: no-filter path, %d prompts × %d gen",
                    gen_prompts,
                    g,
                )
            seqs, mask, plen, lp, rag = _one_round(prompts)
            if ATOMRL_DEBUG:
                logger.info(
                    "[DBG] _collect_rollout: seqs=%s mask=%s plen=%s lp=%s",
                    list(seqs.shape),
                    list(mask.shape),
                    list(plen.shape) if hasattr(plen, "shape") else len(plen),
                    list(lp.shape) if lp is not None else None,
                )
            gts_exp = [gt for gt in gts for _ in range(g)]
            rewards, responses, _accs = self._compute_rewards_full(
                seqs, mask, plen, gts_exp
            )
            if ATOMRL_DEBUG:
                _acc_mean = (
                    float(torch.tensor(_accs).float().mean())
                    if _accs is not None and len(_accs) > 0
                    else -1.0
                )
                logger.info(
                    "[DBG] _collect_rollout: rewards=%s  acc_mean=%.4f  nonzero_reward=%d/%d",
                    list(rewards.shape),
                    _acc_mean,
                    int((rewards.abs() > 1e-8).sum()),
                    rewards.numel(),
                )
            return seqs, mask, plen, rewards, responses, gts_exp, lp, rag

        # Dynamic-sampling regeneration loop (verl filter_groups).
        acc_rows: list[torch.Tensor] = []
        acc_mask: list[torch.Tensor] = []
        acc_lp: list[torch.Tensor] = []
        acc_plen: list[int] = []
        acc_rewards: list[torch.Tensor] = []
        acc_gts: list[str] = []
        acc_ragged: dict[str, list[Any]] = {}
        kept_prompts = 0
        rounds = 0
        max_rounds = fg.max_num_gen_batches if fg.max_num_gen_batches > 0 else 10_000
        want_lp = (
            self._use_vllm or getattr(self, "_ray_use_atom", False)
        ) and self.config.policy.generation.vllm_cfg.calculate_log_probs

        while kept_prompts < train_prompts and rounds < max_rounds:
            rounds += 1
            prompts, gts = self._get_prompts_range(self._prompt_cursor, gen_prompts)
            self._prompt_cursor += gen_prompts
            seqs, mask, plen, lp, rag = _one_round(prompts)
            gts_exp = [gt for gt in gts for _ in range(g)]
            rewards, _responses, accs = self._compute_rewards_full(
                seqs, mask, plen, gts_exp
            )

            uids = [i // g for i in range(seqs.shape[0])]
            keep_mask, kept_uids = filter_groups_keep_mask(accs, uids)
            kept_idx = [i for i, k in enumerate(keep_mask.tolist()) if k]
            if kept_idx:
                acc_rows.append(seqs[kept_idx])
                acc_mask.append(mask[kept_idx])
                acc_rewards.append(rewards[kept_idx])
                if want_lp and lp is not None:
                    acc_lp.append(lp[kept_idx])
                acc_plen.extend([int(plen[i]) for i in kept_idx])
                acc_gts.extend([gts_exp[i] for i in kept_idx])
                for key, rows in rag.items():
                    acc_ragged.setdefault(key, []).extend(rows[i] for i in kept_idx)
                kept_prompts += len(kept_uids)
            if self._rank == 0:
                logger.info(
                    "[step %d] filter_groups round %d: kept %d/%d prompt groups (total %d/%d)",
                    step,
                    rounds,
                    len(kept_uids),
                    gen_prompts,
                    kept_prompts,
                    train_prompts,
                )

        if not acc_rows:
            raise RuntimeError(
                "filter_groups collected no valid groups; check data difficulty / max_num_gen_batches."
            )

        pad_id = self._tokenizer.pad_token_id or 0
        sequences = _merge_left_padded(acc_rows, pad_id)
        seq_mask = _merge_left_padded(acc_mask, 0)
        rewards = torch.cat(acc_rewards, dim=0)
        rollout_lp = _merge_left_padded(acc_lp, 0.0) if acc_lp else None

        # Truncate to exactly train_prompts groups (whole groups of g rows each).
        keep_rows = train_prompts * g
        sequences = sequences[:keep_rows]
        seq_mask = seq_mask[:keep_rows]
        rewards = rewards[:keep_rows]
        prompt_lengths = acc_plen[:keep_rows]
        gts_exp = acc_gts[:keep_rows]
        ragged = {k: v[:keep_rows] for k, v in acc_ragged.items()}
        if rollout_lp is not None:
            rollout_lp = rollout_lp[:keep_rows]

        responses: list[str] = []
        if self._rank == 0:
            seq_cpu = sequences.cpu()
            mask_cpu = seq_mask.cpu()
            for i in range(sequences.shape[0]):
                plen = prompt_lengths[i] if i < len(prompt_lengths) else 0
                response_ids = _response_token_ids(seq_cpu[i], mask_cpu[i], plen)
                responses.append(
                    self._tokenizer.decode(response_ids, skip_special_tokens=True)
                )

        return (
            sequences,
            seq_mask,
            prompt_lengths,
            rewards,
            responses,
            gts_exp,
            rollout_lp,
            ragged,
        )

    def _set_reshard(self, reshard: bool) -> None:
        """Toggle FSDP2 reshard_after_forward on the actor model."""
        if not self._is_distributed:
            return
        try:
            from atomrl.engine.training.fsdp_backend import set_reshard_after_forward

            set_reshard_after_forward(self._actor_model, reshard)
        except RuntimeError:
            pass

    def _rollout_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Generate completions using the actor model in eval mode.

        Optimizations vs naive approach:
        - Phase 1a: disables reshard_after_forward during generate so FSDP2
          keeps parameters all-gathered across decode steps (eliminates
          O(L * tokens) redundant all-gathers).
        - Phase 1b: in distributed mode, shards prompts across ranks so each
          rank generates B/world_size sequences, then all-gathers results.

        Returns (sequences, seq_mask, prompt_lengths) on the current device.
        """
        prompt_lens = attention_mask.sum(dim=1).tolist()

        if num_generations > 1:
            input_ids = input_ids.repeat_interleave(num_generations, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)
            prompt_lens = [l for l in prompt_lens for _ in range(num_generations)]

        if self._is_distributed and self._world_size > 1:
            total = input_ids.shape[0]
            chunk = max(1, total // self._world_size)
            start_idx = self._rank * chunk
            end_idx = start_idx + chunk if self._rank < self._world_size - 1 else total
            local_ids = input_ids[start_idx:end_idx]
            local_mask = attention_mask[start_idx:end_idx]
            local_plens = prompt_lens[start_idx:end_idx]
        else:
            local_ids = input_ids
            local_mask = attention_mask
            local_plens = prompt_lens

        max_resp = self.config.policy.max_response_length
        if max_resp > 0:
            max_gen = min(
                max_resp,
                self.config.policy.max_total_sequence_length - local_ids.shape[1],
            )
        else:
            max_gen = max(
                128, self.config.policy.max_total_sequence_length - local_ids.shape[1]
            )
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_gen,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        algo_name = self.config.algorithm.name.lower()
        if algo_name == "dapo":
            gen_kwargs.update({"temperature": 1.0, "top_p": 0.95, "do_sample": True})
        elif algo_name == "grpo":
            gen_kwargs.update({"temperature": 0.7, "do_sample": True})
        else:
            gen_kwargs.update({"do_sample": False})

        self._actor_model.eval()
        had_grad_ckpt = hasattr(self._actor_model, "gradient_checkpointing_disable")
        if had_grad_ckpt:
            try:
                self._actor_model.gradient_checkpointing_disable()
            except RuntimeError:
                pass

        self._set_reshard(False)

        ids_gpu = local_ids.to(self._device)
        mask_gpu = local_mask.to(self._device)

        with torch.no_grad():
            local_seqs = self._actor_model.generate(
                input_ids=ids_gpu,
                attention_mask=mask_gpu,
                **gen_kwargs,
            )

        self._set_reshard(True)

        if had_grad_ckpt:
            try:
                self._actor_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except RuntimeError:
                pass

        if self._is_distributed and self._world_size > 1:
            sequences, seq_mask, prompt_lens = self._allgather_sequences(
                local_seqs,
                mask_gpu,
                local_plens,
            )
        else:
            sequences = local_seqs
            seq_mask = torch.ones(
                sequences.shape, dtype=torch.long, device=sequences.device
            )
            for i, plen in enumerate(local_plens):
                seq_mask[i, :plen] = local_mask[i, :plen].to(sequences.device)
                pad_id = self._tokenizer.pad_token_id
                if pad_id is not None:
                    seq_mask[i, plen:] = (sequences[i, plen:] != pad_id).long()
            prompt_lens = local_plens

        return sequences, seq_mask, prompt_lens

    def _allgather_sequences(
        self,
        local_seqs: torch.Tensor,
        local_mask: torch.Tensor,
        local_plens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """All-gather variable-length generated sequences across ranks.

        Pads to the global max sequence length, gathers, then trims.
        """
        local_max = torch.tensor([local_seqs.shape[1]], device=self._device)
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX)
        global_max_len = int(local_max.item())

        if local_seqs.shape[1] < global_max_len:
            pad = torch.full(
                (local_seqs.shape[0], global_max_len - local_seqs.shape[1]),
                self._tokenizer.pad_token_id or 0,
                dtype=local_seqs.dtype,
                device=local_seqs.device,
            )
            local_seqs = torch.cat([local_seqs, pad], dim=1)

        local_count = torch.tensor(
            [local_seqs.shape[0]], device=self._device, dtype=torch.long
        )
        counts_list = [torch.zeros_like(local_count) for _ in range(self._world_size)]
        torch.distributed.all_gather(counts_list, local_count)
        max_count = max(c.item() for c in counts_list)

        if local_seqs.shape[0] < max_count:
            pad_rows = torch.zeros(
                (max_count - local_seqs.shape[0], global_max_len),
                dtype=local_seqs.dtype,
                device=local_seqs.device,
            )
            local_seqs = torch.cat([local_seqs, pad_rows], dim=0)

        gathered = [torch.zeros_like(local_seqs) for _ in range(self._world_size)]
        torch.distributed.all_gather(gathered, local_seqs)

        all_seqs_list = []
        all_plens = []
        for r, (seqs_r, cnt) in enumerate(zip(gathered, counts_list)):
            n = int(cnt.item())
            all_seqs_list.append(seqs_r[:n])

        plens_tensor = torch.tensor(local_plens, device=self._device, dtype=torch.long)
        if plens_tensor.shape[0] < max_count:
            plens_tensor = torch.nn.functional.pad(
                plens_tensor, (0, max_count - plens_tensor.shape[0])
            )
        gathered_plens = [
            torch.zeros_like(plens_tensor) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(gathered_plens, plens_tensor)
        for r, cnt in enumerate(counts_list):
            n = int(cnt.item())
            all_plens.extend(gathered_plens[r][:n].tolist())

        sequences = torch.cat(all_seqs_list, dim=0)

        pad_id = self._tokenizer.pad_token_id
        seq_mask = torch.ones(
            sequences.shape, dtype=torch.long, device=sequences.device
        )
        for i, plen in enumerate(all_plens):
            plen = int(plen)
            if pad_id is not None:
                seq_mask[i, :plen] = 1
                seq_mask[i, plen:] = (sequences[i, plen:] != pad_id).long()

        return sequences, seq_mask, [int(p) for p in all_plens]

    @staticmethod
    def _fused_token_log_probs(
        logits: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-token log-probs via row-chunked log_softmax to bound peak memory.

        Uses F.log_softmax per sequence row, which avoids promoting the full
        [S, V] logit tensor to float32 and is numerically stable for bf16.
        Matches VERL's ``logprobs_from_logits_v2`` bf16 path.
        """
        logits_shifted = logits[:, :-1]  # [B, S-1, V]  bf16 view
        targets = target_ids[:, 1:].unsqueeze(-1)  # [B, S-1, 1]
        lp_parts = []
        for i in range(logits_shifted.shape[0]):
            row_lp = torch.nn.functional.log_softmax(logits_shifted[i], dim=-1)
            lp_parts.append(row_lp.gather(-1, targets[i]).squeeze(-1))
        return torch.stack(lp_parts, dim=0).float()  # [B, S-1]

    def _compute_log_probs_for_model(
        self,
        model: torch.nn.Module,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        move_to_gpu: bool = False,
    ) -> torch.Tensor:
        """Compute per-token log-probs from model for given sequences.

        Uses the same packed forward path as ``_train_step`` to ensure
        numerical consistency between ``old_log_probs`` and ``log_probs``.
        This is critical: if the two paths differ (e.g. padded SDPA vs
        packed varlen attention), the importance ratio deviates from 1.0
        even with identical weights, causing all ratios to be clipped
        and gradients to vanish.

        When ``move_to_gpu`` is True, moves the model to GPU before forward
        and back to CPU afterward (for CPU-offloaded reference model).
        """
        from atomrl.engine.training.packing import (
            PackingContext,
            pack_sequences,
            packed_token_log_probs,
            unpack_log_probs,
        )

        if move_to_gpu:
            model.to(self._device)

        sequences = sequences.to(self._device)
        attention_mask = attention_mask.to(self._device)

        model.eval()
        S = sequences.shape[1]

        # Same sampling-temperature scaling as _train_step's log_probs, so that
        # old/ref log-probs and train log-probs share verl's div_(temperature)
        # convention and the importance ratio is unbiased.
        _etemp = float(
            getattr(self.config.policy.generation.vllm_cfg, "temperature", 1.0) or 1.0
        )

        # Use _dynamic_mini_batches-style chunking by actual token count
        max_tok = int(self.config.policy.max_token_len_per_gpu)
        seq_lens = attention_mask.sum(dim=1).long()
        all_log_probs = []

        # Build chunk boundaries by actual token budget
        chunks: list[tuple[int, int]] = []
        start = 0
        n = sequences.shape[0]
        while start < n:
            tok_count = 0
            end = start
            while end < n:
                sl = int(seq_lens[end].item())
                if tok_count + sl > max_tok and end > start:
                    break
                tok_count += sl
                end += 1
            chunks.append((start, end))
            start = end

        # FSDP2: equalize chunk count across ranks (same fix as _train_step)
        real_chunk_count = len(chunks)
        if self._is_distributed and self._world_size > 1:
            import torch.distributed as dist

            count_t = torch.tensor([real_chunk_count], device=self._device)
            dist.all_reduce(count_t, op=dist.ReduceOp.MAX)
            global_max = int(count_t.item())
            while len(chunks) < global_max:
                chunks.append(
                    chunks[-1]
                )  # dummy: reuse last chunk for FSDP2 collectives

        with torch.no_grad():
            for ci, (cs, ce) in enumerate(chunks):
                ids_chunk = sequences[cs:ce]
                mask_chunk = attention_mask[cs:ce]

                # The local training engine is FSDP2; tensor packing is not
                # sharded by a tensor-parallel engine.
                _tp = 1
                packed = pack_sequences(ids_chunk, mask_chunk, tp_align=_tp)
                with PackingContext(packed.cu_seqlens, packed.max_seqlen):
                    outputs = model(
                        input_ids=packed.input_ids,
                        position_ids=packed.position_ids,
                        attention_mask=None,
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    logits = logits.squeeze(0)
                    flat_lp = packed_token_log_probs(
                        logits,
                        packed.input_ids.squeeze(0),
                        packed.cu_seqlens,
                        temperature=_etemp,
                    )
                    token_lp = unpack_log_probs(
                        flat_lp,
                        packed.cu_seqlens,
                        packed.seq_lens,
                        S,
                    )
                    if ci < real_chunk_count:
                        all_log_probs.append(token_lp)
                    del outputs, logits

        if move_to_gpu:
            model.cpu()
            torch.cuda.empty_cache()

        return torch.cat(all_log_probs, dim=0)

    def _compute_rewards(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
        ground_truths: list[str],
        num_generations: int,
    ) -> tuple[torch.Tensor, list[str]]:
        """Decode responses and compute math rewards.

        Returns rewards on ``self._device`` to avoid a CPU round-trip.
        """
        seq_cpu = sequences.cpu() if sequences.device.type != "cpu" else sequences
        mask_cpu = (
            attention_mask.cpu()
            if attention_mask.device.type != "cpu"
            else attention_mask
        )
        responses = []
        for i in range(seq_cpu.shape[0]):
            plen = prompt_lengths[i]
            response_ids = _response_token_ids(seq_cpu[i], mask_cpu[i], plen)
            text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(text)

        expanded_gts = []
        for gt in ground_truths:
            expanded_gts.extend([gt] * num_generations)

        from atomrl.rewards.math_reward import compute_math_reward

        rewards, details = compute_math_reward(responses, expanded_gts)

        accuracy = sum(1 for d in details if d["acc"]) / max(1, len(details))
        logger.info("Reward: accuracy=%.4f, mean=%.4f", accuracy, rewards.mean().item())

        n_pos = sum(1 for r in rewards if r > 0)
        n_neg = sum(1 for r in rewards if r < 0)
        n_invalid = sum(1 for d in details if d.get("pred") == "[INVALID]")
        logger.info(
            "Reward breakdown: +1=%d, -1=%d, invalid_format=%d / %d total",
            n_pos,
            n_neg,
            n_invalid,
            len(details),
        )
        for idx in range(min(2, len(responses), len(details))):
            tail = responses[idx][-400:]
            pred = details[idx].get("pred", "N/A")
            gt = expanded_gts[idx] if idx < len(expanded_gts) else "?"
            logger.info(
                "Sample[%d] reward=%.1f pred=%s gt=%s tail=...%s",
                idx,
                rewards[idx].item(),
                pred,
                gt,
                repr(tail[-200:]),
            )

        return rewards.to(self._device), responses

    def _build_response_mask(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> torch.Tensor:
        """Create a mask that is 1 only for response tokens (excluding prompt).

        For left-padded sequences, prompt tokens start at the first non-pad
        position.  We must zero out prompt positions correctly, not just the
        first ``plen`` columns (which are padding for left-padded inputs).
        The returned mask is ``[:, 1:]`` to align with shifted log-probs.
        """
        _B, _S = attention_mask.shape
        mask = attention_mask.clone()
        for i, plen in enumerate(prompt_lengths):
            # Find where actual tokens start (first 1 in attention_mask)
            actual_start = int(
                (attention_mask[i] == 1).nonzero(as_tuple=True)[0][0].item()
            )
            # Prompt spans [actual_start, actual_start + plen)
            mask[i, actual_start : actual_start + plen] = 0
        return mask[:, 1:]

    @torch.no_grad()
    def _packed_entropy_chunked(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        S: int,
    ) -> torch.Tensor:
        """Per-token entropy [B, S-1] via the same chunked packed forward as
        :meth:`_compute_log_probs_for_model` (avoids OOM on the full vocab)."""
        from atomrl.engine.training.packing import (
            PackingContext,
            pack_sequences,
            packed_token_entropy,
            unpack_log_probs,
        )

        self._actor_model.eval()
        max_tok = int(self.config.policy.max_token_len_per_gpu)
        seq_lens = attention_mask.sum(dim=1).long()
        n = sequences.shape[0]
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < n:
            tok = 0
            end = start
            while end < n:
                sl = int(seq_lens[end].item())
                if tok + sl > max_tok and end > start:
                    break
                tok += sl
                end += 1
            chunks.append((start, end))
            start = end
        real = len(chunks)
        if self._is_distributed and self._world_size > 1:
            import torch.distributed as dist

            ct = torch.tensor([real], device=self._device)
            dist.all_reduce(ct, op=dist.ReduceOp.MAX)
            while len(chunks) < int(ct.item()):
                chunks.append(chunks[-1])
        outs = []
        for ci, (cs, ce) in enumerate(chunks):
            _tp = 1
            packed = pack_sequences(
                sequences[cs:ce], attention_mask[cs:ce], tp_align=_tp
            )
            with PackingContext(packed.cu_seqlens, packed.max_seqlen):
                o = self._actor_model(
                    input_ids=packed.input_ids,
                    position_ids=packed.position_ids,
                    attention_mask=None,
                )
                logits = (o.logits if hasattr(o, "logits") else o).squeeze(0)
                ef = packed_token_entropy(
                    logits,
                    packed.cu_seqlens,
                    temperature=temperature,
                    upcast=True,
                )
                eu = unpack_log_probs(ef, packed.cu_seqlens, packed.seq_lens, S)
                if ci < real:
                    outs.append(eu)
                del o, logits
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def replay_compare(self, dump_path: str) -> None:
        """Replay verl's dumped rollout sequences through Lumen's training forward.

        Loads an engine-agnostic verl dump (per-sequence token ids + verl's
        recomputed old_log_probs / rollout_log_probs / per-token advantages /
        sequence reward / uid), rebuilds the SAME sequences, runs Lumen's packed
        forward to obtain log_probs + entropy, recomputes the GRPO advantage and
        the DAPO policy loss, and prints a side-by-side comparison. This isolates
        forward / advantage / loss consistency from rollout sampling noise.
        """
        from collections import OrderedDict

        from atomrl.algorithms.loss_functions import asymmetric_clip_loss

        data = torch.load(dump_path, weights_only=False)
        samples = data["samples"]
        meta = data["meta"]
        g = int(meta.get("n", 8))

        # Reorder so each consecutive block of g rows is one GRPO group (by uid).
        groups: OrderedDict[str, list] = OrderedDict()
        for s in samples:
            groups.setdefault(s["uid"], []).append(s)
        samples = [s for lst in groups.values() for s in lst]
        B = len(samples)

        pad_id = self._tokenizer.pad_token_id or 0
        seqs = [list(s["prompt_ids"]) + list(s["response_ids"]) for s in samples]
        S = max(len(x) for x in seqs)
        input_ids = torch.full((B, S), pad_id, dtype=torch.long)
        attn = torch.zeros((B, S), dtype=torch.long)
        for i, seq in enumerate(seqs):
            L = len(seq)
            input_ids[i, S - L :] = torch.tensor(seq, dtype=torch.long)
            attn[i, S - L :] = 1
        prompt_lengths = [len(s["prompt_ids"]) for s in samples]
        input_ids = input_ids.to(self._device)
        attn = attn.to(self._device)

        response_mask = self._build_response_mask(
            input_ids, attn, prompt_lengths
        )  # [B,S-1]

        verl_olp = torch.zeros((B, S - 1), dtype=torch.float32, device=self._device)
        verl_rolp = torch.zeros((B, S - 1), dtype=torch.float32, device=self._device)
        verl_adv = torch.zeros((B, S - 1), dtype=torch.float32, device=self._device)
        verl_ris = torch.ones((B, S - 1), dtype=torch.float32, device=self._device)
        seq_reward = torch.zeros(B, dtype=torch.float32, device=self._device)
        verl_adv_seq = torch.zeros(B, dtype=torch.float32, device=self._device)
        has_ris = False
        for i, s in enumerate(samples):
            L = len(seqs[i])
            actual_start = S - L
            base = actual_start + prompt_lengths[i]  # abs pos of first response token
            olp = s["old_log_probs"]
            rolp = s["rollout_log_probs"]
            advt = s["adv_tokens"]
            ris = s.get("rollout_is_weights")
            for m in range(len(olp)):
                col = base + m - 1
                if 0 <= col < S - 1:
                    verl_olp[i, col] = olp[m]
                    if rolp is not None:
                        verl_rolp[i, col] = rolp[m]
                    if advt:
                        verl_adv[i, col] = advt[m]
                    if ris is not None:
                        verl_ris[i, col] = ris[m]
                        has_ris = True
            seq_reward[i] = float(s["token_level_reward"])
            verl_adv_seq[i] = float(advt[0]) if advt else 0.0

        _etemp = float(
            getattr(self.config.policy.generation.vllm_cfg, "temperature", 1.0) or 1.0
        )
        # Lumen forward on the EXACT verl sequences (all ranks participate).
        lumen_olp = self._compute_log_probs_for_model(
            self._actor_model, input_ids, attn
        )
        lumen_ent = self._packed_entropy_chunked(input_ids, attn, _etemp, S)

        if self._rank != 0:
            return

        rm = response_mask.bool()
        nz = rm.sum().clamp(min=1).item()

        # (A) forward log-prob consistency
        olp_diff = (lumen_olp - verl_olp)[rm]
        lp_mae = olp_diff.abs().mean().item()
        lp_max = olp_diff.abs().max().item()
        lumen_olp_mean = lumen_olp[rm].mean().item()
        verl_olp_mean = verl_olp[rm].mean().item()
        ratio = torch.exp(olp_diff.clamp(-20, 20))
        ratio_mean = ratio.mean().item()
        ratio_std = ratio.std().item()

        # (B) entropy (token-mean over response mask)
        lumen_entropy = (lumen_ent * rm.float()).sum().item() / nz

        # (C) GRPO advantage: Lumen eps vs verl eps, both vs verl dump
        R = seq_reward.view(-1, g)
        mean = R.mean(dim=1, keepdim=True)
        std_unb = R.std(dim=1, unbiased=True, keepdim=True)
        adv_lumen = ((R - mean) / std_unb.clamp_min(1e-8)).reshape(-1)
        adv_verleps = ((R - mean) / (std_unb + 1e-6)).reshape(-1)
        adv_mae_lumen = (adv_lumen - verl_adv_seq).abs().mean().item()
        adv_mae_verleps = (adv_verleps - verl_adv_seq).abs().mean().item()

        # (D) DAPO policy loss (token-mean, global token denom)
        Ntok = int(rm.sum().item())
        ris_arg = verl_ris if has_ris else None
        loss_lumenfwd = asymmetric_clip_loss(
            lumen_olp,
            verl_olp,
            verl_adv,
            meta["clip_ratio_low"],
            meta["clip_ratio_high"],
            mask=response_mask,
            clip_ratio_c=meta["clip_ratio_c"],
            batch_num_tokens=Ntok,
            dp_size=1,
            rollout_is_weights=ris_arg,
        ).item()
        loss_ratio1 = asymmetric_clip_loss(
            verl_olp,
            verl_olp,
            verl_adv,
            meta["clip_ratio_low"],
            meta["clip_ratio_high"],
            mask=response_mask,
            clip_ratio_c=meta["clip_ratio_c"],
            batch_num_tokens=Ntok,
            dp_size=1,
            rollout_is_weights=ris_arg,
        ).item()

        log = logger.info
        log(
            "================ REPLAY COMPARE (verl seqs -> Lumen forward) ================"
        )
        log(
            "samples=%d groups=%d g=%d S=%d resp_tokens=%d temp=%.3f",
            B,
            B // g,
            g,
            S,
            Ntok,
            _etemp,
        )
        log(
            "[A] log_prob | Lumen mean=%.5f verl mean=%.5f MAE=%.5f max|d|=%.4f",
            lumen_olp_mean,
            verl_olp_mean,
            lp_mae,
            lp_max,
        )
        log(
            "[A] ratio exp(lumen-verl): mean=%.5f std=%.5f (==1.0 => forward identical)",
            ratio_mean,
            ratio_std,
        )
        log(
            "[B] entropy  | Lumen=%.5f  (compare to verl actor/entropy in verl log)",
            lumen_entropy,
        )
        log(
            "[C] adv MAE vs verl dump: lumen-eps(1e-8)=%.6f  verl-eps(1e-6)=%.6f",
            adv_mae_lumen,
            adv_mae_verleps,
        )
        log("[C] adv lumen[:6]=%s", [round(x, 4) for x in adv_lumen[:6].tolist()])
        log("[C] adv verl [:6]=%s", [round(x, 4) for x in verl_adv_seq[:6].tolist()])
        log(
            "[D] pg_loss  | Lumen-fwd=%.6f  ratio==1(adv+TIS+agg)=%.6f  (compare verl actor/pg_loss)",
            loss_lumenfwd,
            loss_ratio1,
        )
        log(
            "[*] rollout_is present=%s mean=%.5f | seq_reward mean=%.4f",
            has_ris,
            (verl_ris[rm].mean().item() if has_ris else float("nan")),
            seq_reward.mean().item(),
        )
        log(
            "============================================================================"
        )

    def _update_lr(self, step: int) -> None:
        """Advance LR scheduler via Engine, falling back to manual warmup."""
        if self._engine is not None and self._engine.lr_scheduler is not None:
            return
        if step < self._lr_warmup_steps:
            warmup_lr = self._base_lr * (step + 1) / self._lr_warmup_steps
        else:
            warmup_lr = self._base_lr
        for pg in self._optimizer.param_groups:
            pg["lr"] = warmup_lr

    def _balance_batch(self, batch: DataProto) -> dict[str, float]:
        """Seqlen-balanced partitioning across DP ranks.
        (verl/utils/seqlen_balancing.py, verl/trainer/ppo/ray_trainer.py L1098-1165)
        """
        from atomrl.utils.seqlen_balancing import (
            calculate_workload,
            get_seqlen_balanced_partitions,
            log_seqlen_unbalance,
        )

        mask = batch.tensors.get("response_mask", batch.tensors.get("attention_mask"))
        if mask is None:
            return {}
        seq_lens = mask.float().sum(dim=-1).long()
        workloads = calculate_workload(seq_lens).cpu().tolist()
        dp_size = self._world_size
        if batch.batch_size < dp_size:
            return {}
        partitions = get_seqlen_balanced_partitions(
            workloads, dp_size, equal_size=False
        )
        stats = log_seqlen_unbalance(workloads, partitions, prefix="balance")
        my_indices = partitions[self._rank]
        perm = torch.tensor(my_indices, device=self._device, dtype=torch.long)
        batch.reorder(perm)
        return stats

    def _dynamic_mini_batches(
        self,
        batch: DataProto,
        max_token_len: int,
        fallback_bs: int,
    ) -> list[DataProto]:
        """Split batch into mini-batches capped by total *actual* token count.

        Uses actual sequence lengths (from ``attention_mask``) and sorts by
        length for efficient packing.  With sequence packing, each
        mini-batch contains multiple sequences whose total actual tokens
        fit within ``max_token_len``.
        """
        if max_token_len <= 0:
            return list(batch.mini_batches(fallback_bs))

        # Use actual token lengths for packing (not padded length).
        seq_lens = batch.tensors["attention_mask"].sum(dim=1).long()

        # Sort by length for better packing (similar lengths together).
        sorted_idx = torch.argsort(seq_lens)
        sorted_lens = seq_lens[sorted_idx]
        sorted_tensors = {k: v[sorted_idx] for k, v in batch.tensors.items()}

        batches: list[DataProto] = []
        start = 0
        n = batch.batch_size
        while start < n:
            tok_count = 0
            end = start
            while end < n:
                sl = int(sorted_lens[end].item())
                if tok_count + sl > max_token_len and end > start:
                    break
                tok_count += sl
                end += 1
            chunk = {k: v[start:end] for k, v in sorted_tensors.items()}
            batches.append(DataProto(tensors=chunk, meta=batch.meta.copy()))
            start = end
        return batches

    def _train_step(
        self,
        batch: DataProto,
        loss_scale: float = 1.0,
        dp_size: int = 1,
    ) -> dict[str, float]:
        """One gradient step on the actor model (with sequence packing)."""
        if self._actor_model is None or self._optimizer is None:
            raise RuntimeError("setup() must be called first.")

        from atomrl.engine.training.packing import (
            PackingContext,
            pack_sequences,
            packed_token_entropy,
            packed_token_log_probs,
            unpack_log_probs,
        )

        self._actor_model.train()
        sequences = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)

        # Compute per-mini-batch batch_num_tokens via all-reduce (Verl pattern).
        # Each mini-batch normalizes by its own global token count, not the
        # total across all mini-batches.
        _resp = batch.tensors.get("response_mask")
        if _resp is not None:
            _local_mb_tokens = int(_resp.sum())
        else:
            _local_mb_tokens = int(attention_mask.sum())
        # Prefer the full-batch global token count (set by the trainer when
        # accumulating gradients across mini-batches into one optimizer step) so
        # token-mean normalization matches verl. Fall back to the per-mini-batch
        # global count when training a standalone batch.
        _full_bt = batch.meta.get("full_batch_num_tokens")
        if _full_bt is not None:
            mb_batch_num_tokens = int(_full_bt)
        elif self._is_distributed:
            _tok_t = torch.tensor(_local_mb_tokens, device=self._device)
            torch.distributed.all_reduce(_tok_t, op=torch.distributed.ReduceOp.SUM)
            mb_batch_num_tokens = int(_tok_t.item())
        else:
            mb_batch_num_tokens = _local_mb_tokens

        # Pack multiple sequences into a single flat tensor for the forward pass
        _tp = 1
        packed = pack_sequences(sequences, attention_mask, tp_align=_tp)

        # PackingContext stays alive through backward (gradient checkpointing)
        with PackingContext(packed.cu_seqlens, packed.max_seqlen):
            outputs = self._actor_model(
                input_ids=packed.input_ids,
                position_ids=packed.position_ids,
                attention_mask=None,  # varlen attention handles masking
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            logits = logits.squeeze(0)  # (total_tokens, V)

            # Sampling temperature: scale logits before BOTH log_prob and entropy,
            # matching verl's logits.div_(temperature) in _forward_micro_batch. It
            # must be applied identically to old/train/ref log-probs (see
            # _compute_log_probs_for_model) so the importance ratio stays correct.
            _etemp = float(
                getattr(self.config.policy.generation.vllm_cfg, "temperature", 1.0)
                or 1.0
            )

            flat_lp = packed_token_log_probs(
                logits,
                packed.input_ids.squeeze(0),
                packed.cu_seqlens,
                temperature=_etemp,
            )
            # Predictive entropy (metric only, detached) over response tokens.
            # Aligned with verl actor/entropy: logits/temperature, token-mean over
            # the response mask. verl computes entropy via torch.compile (default
            # use_torch_compile=True), whose inductor backend accumulates the
            # softmax / sum(pd*logits) reductions over the ~150k vocab in fp32.
            # We therefore compute in fp32 (upcast=True): eager bf16 accumulation
            # over such a large vocab biases the entropy by ~0.03+ vs fp32, which
            # is the residual we previously saw — it is a reduction-precision
            # (not operator) difference.
            _entropy_mean = None
            _entropy_sum = (
                None  # token-weighted sum (for verl-aligned GLOBAL token-mean)
            )
            _entropy_tok = (
                None  # response token count (excludes dummy/padding mini-batches)
            )
            try:
                _ent_flat = packed_token_entropy(
                    logits.detach(),
                    packed.cu_seqlens,
                    temperature=_etemp,
                    upcast=True,
                )
                _ent_unpacked = unpack_log_probs(
                    _ent_flat,
                    packed.cu_seqlens,
                    packed.seq_lens,
                    sequences.shape[1],
                )
                if _resp is not None:
                    _rm_e = _resp.to(dtype=_ent_unpacked.dtype)
                    _entropy_sum = float((_ent_unpacked * _rm_e).sum())
                    _entropy_tok = float(_rm_e.sum())
                    _entropy_mean = _entropy_sum / max(_entropy_tok, 1.0)
                else:
                    _entropy_sum = float(_ent_unpacked.sum())
                    _entropy_tok = float(_ent_unpacked.numel())
                    _entropy_mean = _entropy_sum / max(_entropy_tok, 1.0)
            except (RuntimeError, ValueError):
                _entropy_mean = None
            del outputs, logits

            # Unpack to [B, S-1] padded format (matches old_log_probs shape)
            token_log_probs = unpack_log_probs(
                flat_lp,
                packed.cu_seqlens,
                packed.seq_lens,
                sequences.shape[1],
            )
            batch.tensors["log_probs"] = token_log_probs

            # Mismatch KL: divergence between the *rollout* policy (vLLM
            # rollout_log_probs) and the current training log_probs — i.e. the
            # train/inference policy gap that TIS corrects (verl rollout_corr/kl
            # sign: rollout - train). When rollout_log_probs are unavailable
            # (e.g. ATOM text rollout) this falls back to old_log_probs, which is
            # the train-side self-consistency check (~0 at the first micro-batch).
            _ref_lp = batch.tensors.get("rollout_log_probs")
            if _ref_lp is None:
                _ref_lp = batch.tensors["old_log_probs"]
            if _resp is not None:
                _diff = (_ref_lp - token_log_probs).detach()
                _rm = _resp.to(dtype=_diff.dtype)
                _denom = _rm.sum().clamp(min=1.0)
                _mismatch_kl = float((_diff * _rm).sum() / _denom)
            else:
                _mismatch_kl = float((_ref_lp - token_log_probs).detach().mean())

            # Attach per-mini-batch global normalization info for Verl-aligned loss
            batch.meta["batch_num_tokens"] = mb_batch_num_tokens
            batch.meta["dp_size"] = dp_size

            loss, metrics = self._algorithm.compute_loss(batch)
            loss = loss.to(self._device)

            if loss.isnan():
                metrics["loss"] = float("nan")
                return metrics

            # dp_size compensation is now inside the loss (via batch_num_tokens),
            # so we only apply loss_scale for gradient accumulation here.
            (loss * loss_scale).backward()

        metrics["loss"] = float(loss.detach())
        metrics["mismatch_kl"] = _mismatch_kl
        if _entropy_mean is not None:
            metrics["entropy"] = _entropy_mean
            # verl-aligned GLOBAL token-mean: report per-mb sum + token count so
            # the trainer can aggregate sum/tok across mini-batches and ranks
            # (NOT a simple average of per-mb means, which is biased by uneven
            # mini-batch token counts and by FSDP dummy/padding mini-batches).
            metrics["entropy_sum"] = _entropy_sum
            metrics["entropy_tok"] = _entropy_tok
        return metrics

    def train(self) -> None:
        """Main training loop: rollout → ref → reward → advantages → train → sync."""
        if self._use_ray_controller:
            self._train_with_ray_controller()
            return

        if self._algorithm is None or self._actor_model is None:
            raise RuntimeError("Call setup() before train().")

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_begin(self)

        num_generations = _algo_num_generations(self.config)
        total_steps = int(self.config.num_training_steps)
        start_step = self._resume_step
        if start_step > 0:
            logger.info(
                "[rank %d] Skipping steps 0..%d (resuming from checkpoint).",
                self._rank,
                start_step - 1,
            )

        self._prompt_cursor = self._resume_prompt_cursor(start_step, num_generations)

        # On resume, push the restored actor weights to the rollout engine BEFORE
        # the first rollout. Otherwise the engine would start from the base model
        # (its lazy wake loads model_name), making the first resumed step wildly
        # off-policy (mismatch_kl huge, is_weight≈0). Sets the engine's weight dir
        # so its first wake loads the resumed weights.
        if start_step > 0 and self._use_atom and self._atom_engine is not None:
            logger.info(
                "[rank %d] Resume: syncing restored weights to rollout engine before first rollout.",
                self._rank,
            )
            self._sync_weights_to_atom()

        for step in range(start_step, total_steps):
            step_start = time.time()
            self.global_step = step
            self._maybe_start_profile(step)
            if ATOMRL_DEBUG and self._rank == 0:
                logger.info("[DBG] ===== step %d/%d START =====", step, total_steps)
            if self._rank == 0:
                for cb in self.callbacks:
                    cb.on_step_begin(self, step)

            if self._use_atom and self._atom_engine is not None:
                self._offload_optimizer_to_cpu()

            self._log_gpu_mem("pre_gen", step)
            t0 = time.time()
            # Full (unsharded) rollout batch. Handles DAPO dynamic sampling
            # (filter_groups) + rollout log-probs internally; rewards are
            # computed once on the full set so degenerate groups can be filtered.
            (
                sequences,
                seq_mask,
                prompt_lengths,
                rewards_full,
                responses_full,
                gts_exp_full,
                rollout_lp_full,
                rollout_ragged,
            ) = self._collect_rollout_batch(
                step,
                num_generations,
            )
            gen_time = time.time() - t0
            if ATOMRL_DEBUG and self._rank == 0:
                logger.info(
                    "[DBG] step %d rollout done: seqs=%s gen_time=%.1fs",
                    step,
                    list(sequences.shape),
                    gen_time,
                )
            self._log_gpu_mem("post_gen", step)

            # Persistent vLLM stays resident across steps (memory coexists with
            # FSDP training); only the kill/sleep-based path frees after gen.
            if (
                self._use_atom
                and self._atom_engine is not None
                and not self._vllm_persistent
            ):
                _engine_ranks = self._vllm_dp or self._rank == 0
                if _engine_ranks and not self._atom_engine.is_sleeping:
                    self._atom_engine.sleep_inprocess()
                    logger.info(
                        "Inference engine slept after generation — freeing GPU for training."
                    )
            if self._use_atom and self._is_distributed:
                torch.distributed.barrier()
            torch.cuda.empty_cache()
            self._log_gpu_mem("post_atom_sleep", step)

            prompt_tok = (
                sum(int(p) for p in prompt_lengths) if num_generations > 0 else 0
            )
            gen_tokens = (
                int(seq_mask.sum().item()) - prompt_tok if num_generations > 0 else 0
            )

            # Shard the (already full) sequence-level batch per rank for log-prob
            # computation and training. Every tensor/list below is sequence-level
            # (length N), so they share one slice. FSDP2 all-reduces gradients.
            if self._is_distributed and self._world_size > 1:
                _total = sequences.shape[0]
                _chunk = max(1, _total // self._world_size)
                _s = self._rank * _chunk
                _e = _s + _chunk if self._rank < self._world_size - 1 else _total
                sequences = sequences[_s:_e]
                seq_mask = seq_mask[_s:_e]
                rewards_full = rewards_full[_s:_e]
                if rollout_lp_full is not None:
                    rollout_lp_full = rollout_lp_full[_s:_e]
                if rollout_ragged:
                    rollout_ragged = {k: v[_s:_e] for k, v in rollout_ragged.items()}
                if isinstance(prompt_lengths, list):
                    prompt_lengths = prompt_lengths[_s:_e]
                gts_exp_full = gts_exp_full[_s:_e]
                if responses_full:
                    responses_full = responses_full[_s:_e]

            _rc_cfg = self.config.quantization.rollout_correction
            _bypass_mode = getattr(_rc_cfg, "bypass_mode", False)

            if _bypass_mode and rollout_lp_full is not None:
                old_log_probs = (
                    rollout_lp_full.to(self._device)
                    if rollout_lp_full is not None
                    else None
                )
                if old_log_probs is None:
                    raise ValueError(
                        "bypass_mode=True requires rollout_log_probs. "
                        "Set calculate_log_probs=true in vllm_cfg."
                    )
                if self._rank == 0:
                    logger.info(
                        "[step=%d] bypass_mode: using rollout_log_probs as old_log_probs",
                        step,
                    )
            else:
                self._actor_model.eval()
                old_log_probs = self._compute_log_probs_for_model(
                    self._actor_model,
                    sequences,
                    seq_mask,
                )

            if step < 3 and self._rank == 0:
                logger.info(
                    "NaN-DEBUG [step=%d post-rollout] old_log_probs: shape=%s nan=%d inf=%d "
                    "min=%.4f max=%.4f mean=%.4f",
                    step,
                    list(old_log_probs.shape),
                    old_log_probs.isnan().sum().item(),
                    old_log_probs.isinf().sum().item(),
                    (
                        old_log_probs[~old_log_probs.isnan()].min().item()
                        if not old_log_probs.isnan().all()
                        else float("nan")
                    ),
                    (
                        old_log_probs[~old_log_probs.isnan()].max().item()
                        if not old_log_probs.isnan().all()
                        else float("nan")
                    ),
                    (
                        old_log_probs[~old_log_probs.isnan()].mean().item()
                        if not old_log_probs.isnan().all()
                        else float("nan")
                    ),
                )
                logger.info(
                    "NaN-DEBUG [step=%d post-rollout] sequences: shape=%s, seq_mask: shape=%s, "
                    "seq_mask sum=%d, prompt_lengths=%s",
                    step,
                    list(sequences.shape),
                    list(seq_mask.shape),
                    seq_mask.sum().item(),
                    prompt_lengths[:4],
                )

            t1 = time.time()
            if self._ref_model is not None:
                ref_log_probs = self._compute_log_probs_for_model(
                    self._ref_model,
                    sequences,
                    seq_mask,
                    move_to_gpu=self._ref_on_cpu,
                )
            else:
                ref_log_probs = torch.zeros_like(old_log_probs)
            ref_time = time.time() - t1

            # Rewards were computed on the full set inside _collect_rollout_batch.
            rewards = rewards_full.to(self._device)
            responses = responses_full

            response_mask = self._build_response_mask(
                sequences, seq_mask, prompt_lengths
            )
            response_lengths = [
                int(response_mask[i].sum().item())
                for i in range(response_mask.shape[0])
            ]

            _batch_tensors = {
                "input_ids": sequences,
                "attention_mask": seq_mask,
                "old_log_probs": old_log_probs,
                "ref_log_probs": ref_log_probs,
                "rewards": rewards,
                "response_mask": response_mask,
            }
            # Rollout log-probs (vLLM) enable TIS/MIS rollout correction.
            if rollout_lp_full is not None:
                _rlp = rollout_lp_full.to(self._device)
                if _rlp.shape == old_log_probs.shape:
                    _batch_tensors["rollout_log_probs"] = _rlp

            batch = DataProto(
                tensors=_batch_tensors,
                meta={
                    "algorithm": self.config.algorithm.name,
                    "response_lengths": response_lengths,
                    "responses": responses,
                    "ground_truths": gts_exp_full,
                },
                ragged=rollout_ragged,
            )

            # --- KL penalty in reward (verl/trainer/ppo/ray_trainer.py L1546-1553) ---
            kl_metrics = {}
            if self._kl_ctrl is not None:
                from atomrl.algorithms.kl_controller import (
                    apply_kl_penalty as _apply_kl,
                )

                batch, kl_metrics = _apply_kl(
                    batch,
                    kl_ctrl=self._kl_ctrl,
                    kl_penalty_type=self.config.algorithm.kl_penalty,
                )

            # --- Critic: compute values (needed for GAE) ---
            if self._critic_worker is not None:
                values_out = self._critic_worker.compute_values(batch)
                batch.tensors["values"] = values_out.tensors["values"]

            batch = self._algorithm.compute_advantages(batch)
            batch = apply_rollout_correction(batch, self.config)

            # --- Same-group dump (env-gated, first step only) ---
            # On the SAME sequences, capture: rollout_log_probs (sampling H(q)),
            # old_log_probs (cross-entropy q->p on sampled tokens), per-token entropy
            # (training policy H(p)). Lets us check E[-rollout_logp] vs E[-old_logp]
            # vs mean(entropy) on identical tokens (verl baseline: all ≈ equal).
            if step == start_step and os.environ.get("LUMEN_DUMP_ROLLOUT"):
                _etemp = float(
                    getattr(self.config.policy.generation.vllm_cfg, "temperature", 1.0)
                    or 1.0
                )
                _seqd = batch.tensors["input_ids"]
                _amd = batch.tensors["attention_mask"]
                # all ranks run the forward (FSDP collective)
                _entd = self._packed_entropy_chunked(
                    _seqd, _amd, _etemp, _seqd.shape[1]
                )
                try:
                    _rmd = batch.tensors["response_mask"].float()
                    _perseq = (_entd * _rmd).sum(dim=1) / _rmd.sum(dim=1).clamp(
                        min=1.0
                    )  # [n]
                    _rlp = batch.tensors.get("rollout_log_probs")
                    _perseq_rlp = None
                    if _rlp is not None:
                        _perseq_rlp = (_rlp * _rmd).sum(dim=1) / _rmd.sum(dim=1).clamp(
                            min=1.0
                        )
                    _dp = os.environ["LUMEN_DUMP_ROLLOUT"]
                    _base = _dp.rsplit(".", 1)[0]
                    torch.save(
                        {
                            "rank": self._rank,
                            "perseq_entropy": _perseq.detach().float().cpu(),
                            "perseq_rollout_logp": (
                                _perseq_rlp.detach().float().cpu()
                                if _perseq_rlp is not None
                                else None
                            ),
                            "resp_len": _rmd.sum(dim=1).detach().cpu(),
                            "sequences": _seqd.detach().cpu(),
                            "attention_mask": _amd.detach().cpu(),
                            "prompt_lengths": (
                                torch.tensor(prompt_lengths, dtype=torch.long)
                                if isinstance(prompt_lengths, list)
                                else prompt_lengths
                            ),
                        },
                        f"{_base}_rank{self._rank}.pt",
                    )
                    logger.info(
                        "[LUMEN_DUMP] rank %d saved per-seq entropy dump", self._rank
                    )
                except (OSError, RuntimeError, ValueError) as _e:
                    logger.warning("[LUMEN_DUMP] failed: %s", _e)

            # --- Extended rollout correction: IS weights + rejection sampling ---
            _rc_metrics = {}
            _rollout_lp = batch.tensors.get(
                "rollout_log_probs", batch.tensors.get("fp8_logprobs")
            )
            _rc_want = (
                (_rc_cfg.rollout_is or _rc_cfg.rollout_rs or _bypass_mode)
                and "old_log_probs" in batch.tensors
                and _rollout_lp is not None
            )
            if _rc_want:
                from atomrl.algorithms.rollout_correction import (
                    compute_rollout_correction_and_add_to_batch,
                )

                batch, _rc_metrics = compute_rollout_correction_and_add_to_batch(
                    batch, _rc_cfg
                )

            # --- Seqlen balanced partitioning ---
            # (verl/utils/seqlen_balancing.py, verl/trainer/ppo/ray_trainer.py L1098-1165)
            _balance_metrics = {}
            if (
                self.config.policy.balance_batch
                and self._is_distributed
                and self._world_size > 1
            ):
                _balance_metrics = self._balance_batch(batch)

            if self._use_atom and self._atom_engine is not None:
                self._reload_optimizer_to_gpu()

            self._log_gpu_mem("pre_train", step)
            t2 = time.time()
            micro_bs = max(1, int(self.config.policy.train_micro_batch_size))
            max_tok = int(self.config.policy.max_token_len_per_gpu)

            # PPO multi-epoch: iterate over same batch multiple times
            # (verl/trainer/ppo/ray_trainer.py L1262-1271)
            _algo_lc = self.config.algorithm.name.lower()
            if _algo_lc == "ppo":
                _ppo_epochs = self.config.algorithm.ppo.num_ppo_epochs
            elif _algo_lc == "grpo":
                _ppo_epochs = self.config.algorithm.grpo.num_ppo_epochs
            else:
                _ppo_epochs = 1
            _ppo_epochs = max(1, _ppo_epochs)

            metrics_accum: dict[str, float] = {}
            step_count = 0
            nan_mb_count = 0
            grad_norm = 0.0  # running SUM of per-optimizer-step grad norms
            mismatch_kl_initial: float | None = None
            nan_param_count = 0
            total_param_count = 0
            optimizer_steps = 0

            accum_steps = 1  # set per-epoch below to len(mini_batches)
            _dp_size = self._world_size if self._is_distributed else 1
            self._update_lr(step)
            _cur_lr = self._optimizer.param_groups[0]["lr"]

            # verl-aligned update procedure: accumulate gradients over ALL
            # token-budget mini-batches and take ONE optimizer step per training
            # step, normalized by the full-batch token count. Doing a separate
            # optimizer step per (length-sorted) mini-batch instead biases the
            # update toward short sequences (later chunks get PPO-clipped after
            # the first step), which stalls response-length growth and learning.
            _resp_full = batch.tensors.get("response_mask")
            _local_full = (
                int(_resp_full.sum())
                if _resp_full is not None
                else int(batch.tensors["attention_mask"].sum())
            )
            if self._is_distributed:
                _ft = torch.tensor(_local_full, device=self._device)
                torch.distributed.all_reduce(_ft, op=torch.distributed.ReduceOp.SUM)
                batch.meta["full_batch_num_tokens"] = int(_ft.item())
            else:
                batch.meta["full_batch_num_tokens"] = _local_full

            _fsdp_grad_sync = self._is_distributed and self._world_size > 1
            if _fsdp_grad_sync:
                from atomrl.engine.training.fsdp_backend import (
                    set_requires_gradient_sync,
                )

            _do_engine_step = self._engine is not None

            # PPO multi-epoch loop (verl/trainer/ppo/ray_trainer.py L1262-1271)
            for _epoch in range(_ppo_epochs):
                if _epoch > 0:
                    perm = torch.randperm(batch.batch_size)
                    batch.reorder(perm)

                mini_batches = self._dynamic_mini_batches(batch, max_tok, micro_bs)

                # FSDP2: all ranks MUST run the same number of forward/backward passes.
                if self._is_distributed and self._world_size > 1:
                    import torch.distributed as dist

                    my_count = len(mini_batches)
                    count_t = torch.tensor([my_count], device=self._device)
                    dist.all_reduce(count_t, op=dist.ReduceOp.MAX)
                    global_max = int(count_t.item())
                    if my_count < global_max:
                        pad_batch = mini_batches[-1]
                        dummy_tensors = {
                            k: v.clone() for k, v in pad_batch.tensors.items()
                        }
                        dummy_tensors["response_mask"] = torch.zeros_like(
                            dummy_tensors["response_mask"]
                        )
                        dummy_mb = DataProto(
                            tensors=dummy_tensors, meta=pad_batch.meta.copy()
                        )
                        while len(mini_batches) < global_max:
                            mini_batches.append(dummy_mb)
                        logger.info(
                            "[rank %d] Padded mini-batches: %d -> %d for FSDP2 sync",
                            self._rank,
                            my_count,
                            global_max,
                        )

                # ONE optimizer step per training step: accumulate over all
                # mini-batches. Token-mean normalization uses the full-batch
                # token count, so no 1/N loss scaling is applied.
                accum_steps = max(1, len(mini_batches))

                if self._rank == 0:
                    logger.info(
                        "[step %d epoch %d/%d] Training: %d mini-batches, accum_steps=%d, dp_size=%d, lr=%.2e",
                        step,
                        _epoch + 1,
                        _ppo_epochs,
                        len(mini_batches),
                        accum_steps,
                        _dp_size,
                        _cur_lr,
                    )

                for i, mini in enumerate(mini_batches):
                    if i % accum_steps == 0:
                        if _do_engine_step:
                            self._engine.optimizer_zero_grad()
                        else:
                            self._optimizer.zero_grad(set_to_none=True)
                    cur_loss_scale = 1.0  # full-batch token-mean handles averaging
                    is_last_in_group = (i + 1) % accum_steps == 0 or i == len(
                        mini_batches
                    ) - 1
                    if _fsdp_grad_sync:
                        set_requires_gradient_sync(self._actor_model, is_last_in_group)
                    m = self._train_step(
                        mini,
                        loss_scale=cur_loss_scale,
                        dp_size=_dp_size,
                    )
                    if m.get("loss") is not None and (m["loss"] != m["loss"]):
                        nan_mb_count += 1
                        for k, v in m.items():
                            if not math.isnan(v):
                                metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                        step_count += 1
                        if (i + 1) % accum_steps == 0 or i == len(mini_batches) - 1:
                            if _do_engine_step:
                                _gn = self._engine.optimizer_step()
                            else:
                                _gn = float(
                                    torch.nn.utils.clip_grad_norm_(
                                        self._actor_model.parameters(),
                                        max_norm=1.0,
                                    )
                                )
                                if not torch.isfinite(torch.tensor(_gn)):
                                    self._optimizer.zero_grad(set_to_none=True)
                                else:
                                    self._optimizer.step()
                            grad_norm += _gn  # mean over optimizer steps (verl-aligned)
                            optimizer_steps += 1
                        continue
                    _nan_cnt = 0
                    _total_cnt = 0
                    for p in self._actor_model.parameters():
                        if p.grad is not None:
                            _total_cnt += 1
                            if p.grad.isnan().any():
                                _nan_cnt += 1
                                p.grad = torch.where(
                                    p.grad.isnan(),
                                    torch.zeros_like(p.grad),
                                    p.grad,
                                )
                    nan_param_count = max(nan_param_count, _nan_cnt)
                    total_param_count = _total_cnt
                    if (i + 1) % accum_steps == 0 or i == len(mini_batches) - 1:
                        if _do_engine_step:
                            _gn = self._engine.optimizer_step()
                        else:
                            _gn = float(
                                torch.nn.utils.clip_grad_norm_(
                                    self._actor_model.parameters(),
                                    max_norm=1.0,
                                )
                            )
                            if not torch.isfinite(torch.tensor(_gn)):
                                self._optimizer.zero_grad(set_to_none=True)
                            else:
                                self._optimizer.step()
                        grad_norm += _gn  # mean over optimizer steps (verl-aligned)
                        optimizer_steps += 1
                    if mismatch_kl_initial is None and "mismatch_kl" in m:
                        mismatch_kl_initial = m["mismatch_kl"]
                    for k, v in m.items():
                        if not math.isnan(v):
                            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                    step_count += 1

            if _do_engine_step:
                self._engine.optimizer_zero_grad()
                _cur_lr = self._engine.lr_scheduler_step()
            else:
                self._optimizer.zero_grad(set_to_none=True)
            if _fsdp_grad_sync:
                set_requires_gradient_sync(self._actor_model, True)
            if self._rank == 0:
                logger.info(
                    "[step %d] Completed %d optimizer steps (%d epochs x mini-batches).",
                    step,
                    optimizer_steps,
                    _ppo_epochs,
                )
            train_time = time.time() - t2
            self._log_gpu_mem("post_train", step)

            # --- Critic: update value network ---
            if self._critic_worker is not None:
                for _critic_epoch in range(
                    getattr(self.config.critic, "num_critic_epochs", 1)
                ):
                    critic_metrics = self._critic_worker.train_step(batch)
                metrics_accum.update(critic_metrics)

            if nan_param_count > 0 and self._rank == 0:
                logger.warning(
                    "[step %d] Zeroed NaN grads in %d/%d params, grad_norm=%.4f",
                    step,
                    nan_param_count,
                    total_param_count,
                    float(grad_norm),
                )

            denom = max(1, step_count)
            metrics = {k: v / denom for k, v in metrics_accum.items()}
            # verl-aligned GLOBAL token-mean entropy: aggregate the raw token-weighted
            # sum / token count across mini-batches (sum/tok), NOT a simple average of
            # per-mini-batch means. The simple average is biased because mini-batches
            # have uneven token counts AND FSDP dummy/padding mini-batches contribute
            # entropy=0 (tok=0), which previously dragged the reported entropy well
            # below the true token-mean (e.g. 0.37 vs the true 0.52). Cross-rank
            # token-weighting is handled in the all_reduce block below.
            _ent_sum_local = float(metrics_accum.get("entropy_sum", 0.0))
            _ent_tok_local = float(metrics_accum.get("entropy_tok", 0.0))
            metrics.pop("entropy_sum", None)
            metrics.pop("entropy_tok", None)
            if _ent_tok_local > 0:
                metrics["entropy"] = _ent_sum_local / _ent_tok_local
            # Mean grad norm over optimizer steps (verl reports the mean across
            # mini-batches, not the max).
            metrics["grad_norm"] = float(grad_norm / max(1, optimizer_steps))
            metrics["nan_params"] = nan_param_count
            if mismatch_kl_initial is not None:
                metrics["mismatch_kl"] = mismatch_kl_initial

            step_time = time.time() - step_start
            metrics["timing/step_s"] = step_time
            metrics["timing/gen_s"] = gen_time
            metrics["timing/ref_s"] = ref_time
            metrics["timing/train_s"] = train_time
            if gen_tokens > 0 and gen_time > 0:
                metrics["throughput/gen_tok_per_s"] = gen_tokens / gen_time
            metrics["reward/mean"] = float(rewards.mean().item())
            metrics["reward/accuracy"] = float(
                sum(1 for r in rewards if r > 0) / max(1, len(rewards))
            )
            if kl_metrics:
                metrics.update(kl_metrics)
            if _rc_metrics:
                metrics.update(_rc_metrics)
            if _balance_metrics:
                metrics.update(_balance_metrics)

            # Comprehensive data metrics (verl/trainer/ppo/metric_utils.py L89-268)
            try:
                from atomrl.trainer.metric_utils import compute_data_metrics

                batch.meta["max_response_length"] = int(
                    self.config.policy.max_response_length
                )
                _data_m = compute_data_metrics(
                    batch, use_critic=self._critic_worker is not None
                )
                metrics.update(_data_m)
            except (RuntimeError, ValueError):
                pass

            metrics["seq/max_len"] = int(sequences.shape[1])
            metrics["seq/mean_response_len"] = float(
                sum(response_lengths) / max(1, len(response_lengths))
            )

            if self._is_distributed:
                # Cross-rank reduction must respect the metric's semantics:
                # ``*/max`` is a maximum (use MAX), ``*/min`` is a minimum (use MIN),
                # everything else is a mean (AVG). Previously AVG was applied to all,
                # which turned ``response_length/max`` etc. into the *mean of per-rank
                # maxes* (a fractional value far below the true global max) and made
                # length/entropy max comparisons against verl misleading.
                # entropy: token-weighted GLOBAL token-mean across ranks (verl-aligned).
                # all_reduce SUM of (sum, tok) then divide — NOT AVG of per-rank means,
                # which is biased when ranks have different response-token counts.
                _et = torch.tensor(
                    [_ent_sum_local, _ent_tok_local],
                    dtype=torch.float64,
                    device=self._device,
                )
                torch.distributed.all_reduce(_et, op=torch.distributed.ReduceOp.SUM)
                if float(_et[1].item()) > 0:
                    metrics["entropy"] = float(_et[0].item() / _et[1].item())
                for k in list(metrics.keys()):
                    if k == "entropy":
                        continue  # already reduced (token-weighted) above
                    if k.endswith("/max"):
                        op = torch.distributed.ReduceOp.MAX
                    elif k.endswith("/min"):
                        op = torch.distributed.ReduceOp.MIN
                    else:
                        op = torch.distributed.ReduceOp.AVG
                    t = torch.tensor(
                        metrics[k], dtype=torch.float64, device=self._device
                    )
                    torch.distributed.all_reduce(t, op=op)
                    metrics[k] = float(t.item())

            # Validation
            val_steps = getattr(self.config, "val_steps", 0)
            if val_steps > 0 and (step + 1) % val_steps == 0:
                val_metrics = self.run_validation()
                metrics.update(val_metrics)

            self.last_metrics = metrics
            for k, v in metrics.items():
                self._metrics.update(k, v)

            t_sync = time.time()
            self._sync_rollout_weights()
            sync_time = time.time() - t_sync
            if sync_time > 1.0:
                metrics["timing/weight_sync_s"] = sync_time

            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)

            self._maybe_stop_profile(step)

            del sequences, seq_mask, old_log_probs, ref_log_probs
            del rewards, responses, response_mask, batch, mini_batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_end(self)

        logger.info(
            "[rank %d] RLTrainer.train finished after %d steps.",
            self._rank,
            total_steps,
        )

    def _compute_log_probs_with_worker_group(
        self,
        wg: RayWorkerGroup,
        sequences: torch.Tensor,
        role: str,
        want_entropy: bool = False,
        attention_mask: torch.Tensor | None = None,
        rollout_routed_experts: torch.Tensor | None = None,
        rollout_routing: list[Any] | None = None,
    ):
        meta = {"calculate_entropy": True} if want_entropy else {}
        # verl-aligned packed forward on the worker needs the temperature (for
        # logits.div_) and a per-GPU token budget for chunking the flat forward.
        meta["temperature"] = float(
            getattr(self.config.policy.generation.vllm_cfg, "temperature", 1.0) or 1.0
        )
        meta["max_token_len_per_gpu"] = int(self.config.policy.max_token_len_per_gpu)
        req_tensors = {"input_ids": sequences.detach().cpu()}
        # CRITICAL: the worker must receive the padding mask, otherwise it runs the
        # forward over left-pad tokens with arange positions -> flat logits and
        # ~7x inflated entropy. This is the packed-varlen forward's `attention_mask`.
        if attention_mask is not None:
            req_tensors["attention_mask"] = attention_mask.detach().cpu()
        if rollout_routed_experts is not None:
            req_tensors["rollout_routed_experts"] = (
                rollout_routed_experts.detach().cpu()
            )
        req_ragged = {}
        if rollout_routing is not None:
            if len(rollout_routing) != int(sequences.shape[0]):
                raise ValueError(
                    "rollout_routing row count mismatch: "
                    f"got {len(rollout_routing)}, expected {sequences.shape[0]}"
                )
            req_ragged["rollout_routing"] = list(rollout_routing)
        req = DataProto(tensors=req_tensors, meta=meta, ragged=req_ragged)
        role_cfg = (
            self.config.controller.ray.actor
            if role == "actor"
            else self.config.controller.ray.ref
        )
        out = wg.dispatch_and_call(
            "compute_log_probs",
            req,
            mode=role_cfg.dispatch_mode,
            lazy_key=role_cfg.lazy_dispatch_key,
        )
        if "log_probs" in out.tensors:
            logp = out["log_probs"].to(self._device)
        elif "ref_log_probs" in out.tensors:
            logp = out["ref_log_probs"].to(self._device)
        else:
            raise KeyError(
                f"Expected log_probs/ref_log_probs in worker output, got keys={list(out.tensors.keys())}"
            )
        if want_entropy:
            ent = out.tensors.get("entropy")
            return logp, (ent.to(self._device) if ent is not None else None)
        return logp

    @staticmethod
    def _balance_rows_across_workers(batch: DataProto, num_workers: int) -> None:
        """Reorder rows so each worker's contiguous shard holds similar tokens.

        Dispatch splits by row count, so an unlucky split gives one rank more
        tokens than the rest. That rank forms an extra micro-batch, and because
        FSDP2 collectives must run in lockstep every other rank then runs a
        padding micro-batch too -- paying a full round of all-gather and
        reduce-scatter for zero gradient. Longest-first assignment into
        equal-cardinality bins removes the straggler. Loss normalization uses the
        global token count, so it is invariant to this permutation.
        """
        am = batch.tensors.get("attention_mask")
        n = batch.batch_size
        if am is None or num_workers <= 1 or n < num_workers or n % num_workers:
            return
        per_worker = n // num_workers
        lens = am.sum(dim=1).long().tolist()
        order = sorted(range(n), key=lambda i: -lens[i])
        bins: list[list[int]] = [[] for _ in range(num_workers)]
        loads = [0] * num_workers
        for idx in order:
            b = min(
                (j for j in range(num_workers) if len(bins[j]) < per_worker),
                key=lambda j: loads[j],
            )
            bins[b].append(idx)
            loads[b] += lens[idx]
        perm = [i for b in bins for i in b]
        if perm == list(range(n)):
            return
        batch.reorder(perm)
        for key, val in list(batch.meta.items()):
            if isinstance(val, list) and len(val) == n:
                batch.meta[key] = [val[i] for i in perm]

    @staticmethod
    def _finalize_mismatch_metrics(metrics: dict[str, float]) -> None:
        """Turn the workers' mismatch sums into means, in place.

        Companion to ``AtomActorWorker._mismatch_metrics``; see there for why the
        signed ``rollout_corr/kl`` alone is not enough to see a train/rollout gap.
        """
        tok = metrics.pop("mismatch_tok", None)
        n_seq = metrics.pop("mismatch_seq_n", None)
        per_token = {
            "mismatch_abs_sum": "mismatch/abs_diff",
            "mismatch_k3_sum": "mismatch/k3_kl",
            "mismatch_chi2_tok_sum": "mismatch/chi2_token",
            "mismatch_tail_sum": "mismatch/frac_abs_gt_0.1",
        }
        for src, dst in per_token.items():
            val = metrics.pop(src, None)
            if val is not None and tok:
                metrics[dst] = val / max(tok, 1e-6)
        seq_sum = metrics.pop("mismatch_chi2_seq_sum", None)
        if seq_sum is not None and n_seq:
            metrics["mismatch/chi2_seq"] = seq_sum / max(n_seq, 1e-6)

    @staticmethod
    def _mismatch_by_response_position(
        old_log_probs: torch.Tensor,
        rollout_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        *,
        bucket_edges: tuple[int, ...] = (128, 512, 1024, 2048, 4096),
    ) -> dict[str, float]:
        """Measure rollout/train drift in response-relative position buckets."""
        # These three share a frame -- entry i scores token i+1 -- but not
        # necessarily a width: the mask and the rollout log-probs are built
        # pre-shifted at S-1, while a Megatron actor pads old_log_probs out to the
        # full S. `rollout_correction._clean_batch_logprobs` states the contract
        # and takes the common width; so does this, being a second reader of the
        # same three tensors. Positions count from the left, so trimming the tail
        # only drops columns no bucket could have compared.
        width = min(
            old_log_probs.shape[-1],
            rollout_log_probs.shape[-1],
            response_mask.shape[-1],
        )
        old_log_probs = old_log_probs[..., :width]
        rollout_log_probs = rollout_log_probs[..., :width]
        response_mask = response_mask[..., :width]

        mask = response_mask.bool()
        positions = response_mask.long().cumsum(dim=-1) - 1
        delta = rollout_log_probs.float() - old_log_probs.float()
        metrics: dict[str, float] = {}
        start = 0
        for end in bucket_edges:
            selected = mask & (positions >= start) & (positions < end)
            tokens = int(selected.sum().item())
            if tokens:
                values = delta[selected]
                log_ratio = -values
                k3 = torch.exp(log_ratio.clamp(-10.0, 10.0)) - log_ratio - 1.0
                prefix = f"rollout_corr/pos_{start}_{end - 1}"
                metrics[f"{prefix}/tokens"] = float(tokens)
                metrics[f"{prefix}/kl"] = float(values.mean())
                metrics[f"{prefix}/abs_diff"] = float(values.abs().mean())
                metrics[f"{prefix}/k3_kl"] = float(k3.mean())
            start = end
        return metrics

    @staticmethod
    def _r3_verify_old_vs_rollout(
        old_log_probs: torch.Tensor,
        rollout_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Same-weight train vs rollout gap (before the optimizer step).

        ``old_log_probs`` is the actor forward on the rollout sequences;
        ``rollout_log_probs`` is the sampler. Signed KL can cancel expert flips;
        ``abs_diff`` and ``frac_abs_gt_0.1`` are the R3 telltales.
        """
        width = min(
            old_log_probs.shape[-1],
            rollout_log_probs.shape[-1],
            response_mask.shape[-1],
        )
        mask = response_mask[..., :width].bool()
        if int(mask.sum().item()) == 0:
            return {}
        delta = (
            rollout_log_probs[..., :width].float() - old_log_probs[..., :width].float()
        )[mask]
        abs_d = delta.abs()
        return {
            "r3_verify/tokens": float(delta.numel()),
            "r3_verify/kl": float(delta.mean()),
            "r3_verify/abs_diff": float(abs_d.mean()),
            "r3_verify/frac_abs_gt_0.1": float((abs_d > 0.1).float().mean()),
        }

    def _update_actor_with_ray(self, batch: DataProto) -> dict[str, float]:
        if self._actor_wg is None:
            raise RuntimeError("Ray actor worker group is not initialized.")
        actor_role_cfg = self.config.controller.ray.actor
        n = self._actor_wg.num_workers
        batch.meta.setdefault("global_batch_size", batch.batch_size)
        self._balance_rows_across_workers(batch, n)

        import ray

        dp_rank_mapping = self._actor_wg.dp_rank_mapping
        if dp_rank_mapping is not None:
            dp_size = len(set(dp_rank_mapping))
            unique_chunks = batch.split(dp_size)
            chunk_refs = [ray.put(c) for c in unique_chunks]
            obj_refs = [chunk_refs[dp_rank_mapping[i]] for i in range(n)]
            refs = [
                self._actor_wg.call_single_async(i, "update_policy", obj_refs[i])
                for i in range(n)
            ]
        else:
            chunks = dispatch_proto(
                batch,
                n,
                mode=actor_role_cfg.dispatch_mode,
                lazy_state=self._ray_dispatch_state,
                lazy_key=actor_role_cfg.lazy_dispatch_key,
            )
            if not chunks:
                return {"loss": 0.0}
            if len(chunks) == 1 and n == 1:
                refs = [self._actor_wg.call_single_async(0, "update_policy", chunks[0])]
            elif len(chunks) == n:
                refs = [
                    self._actor_wg.call_single_async(i, "update_policy", chunks[i])
                    for i in range(n)
                ]
            else:
                raise ValueError(
                    f"actor dispatch produced {len(chunks)} chunks for "
                    f"{n} workers; expected 1 (single-actor) or num_workers "
                    "(FSDP requires every actor to participate)."
                )

        outputs: list[dict[str, float]] = ray.get(refs)
        if not outputs:
            return {"loss": 0.0}

        collect_mask = self._actor_wg._collect_mask
        if collect_mask is not None:
            outputs = [o for o, keep in zip(outputs, collect_mask) if keep]
        if not outputs:
            return {"loss": 0.0}
        merged: dict[str, float] = {}
        keys = set().union(*(o.keys() for o in outputs))
        for key in keys:
            vals = [float(o[key]) for o in outputs if key in o]
            merged[key] = float(sum(vals) / max(1, len(vals)))
        return merged

    def _reset_actor_memory_stats(self) -> None:
        if self._actor_wg is None:
            return
        try:
            self._actor_wg.execute_all_sync("reset_memory_stats")
        except _ray_boundary_errors() as exc:
            logger.warning("reset actor memory stats failed: %s", exc)

    def _collect_actor_memory_metrics(self) -> dict[str, float]:
        if self._actor_wg is None:
            return {}
        try:
            stats = self._actor_wg.execute_all_sync("get_memory_stats")
        except _ray_boundary_errors() as exc:
            logger.warning("collect actor memory stats failed: %s", exc)
            return {}
        if not stats:
            return {}
        max_reserved = max(float(s.get("max_reserved_bytes", 0.0)) for s in stats)
        max_allocated = max(float(s.get("max_allocated_bytes", 0.0)) for s in stats)
        cur_reserved = max(float(s.get("reserved_bytes", 0.0)) for s in stats)
        cur_allocated = max(float(s.get("allocated_bytes", 0.0)) for s in stats)
        cpu_used = max(float(s.get("cpu_memory_used_bytes", 0.0)) for s in stats)
        gb = 1024.0**3
        return {
            "mem/actor_max_reserved_gb": max_reserved / gb,
            "mem/actor_max_allocated_gb": max_allocated / gb,
            "mem/actor_reserved_gb": cur_reserved / gb,
            "mem/actor_allocated_gb": cur_allocated / gb,
            "actor/perf/cpu_memory_used_gb": cpu_used / gb,
            "actor/perf/max_memory_allocated_gb": max_allocated / gb,
            "actor/perf/max_memory_reserved_gb": max_reserved / gb,
            "actor/perf/micro_batch_max_reserved_gb": max_reserved / gb,
        }

    def _add_verl_metrics(
        self,
        metrics: dict[str, float],
        batch: DataProto,
        timing_raw: dict[str, float],
        prompt_lengths: list[int],
        response_lengths: list[int],
    ) -> None:
        """Populate the metric names used by the AMD-BF16-VERL dashboard."""
        from atomrl.trainer.metric_utils import (
            compute_data_metrics,
            compute_throughput_metrics,
            compute_timing_metrics,
        )

        max_prompt_length = max(
            0,
            int(self.config.policy.max_total_sequence_length)
            - int(self.config.policy.max_response_length),
        )
        batch.meta["max_prompt_length"] = max_prompt_length
        batch.meta["max_response_length"] = int(self.config.policy.max_response_length)
        metrics.update(
            compute_data_metrics(batch, use_critic=self._critic_worker is not None)
        )
        metrics.update(compute_timing_metrics(batch, timing_raw))

        cluster_gpus = max(
            1,
            int(self.config.cluster.num_nodes) * int(self.config.cluster.gpus_per_node),
        )
        metrics.update(compute_throughput_metrics(batch, timing_raw, cluster_gpus))

        # Actor aliases expected by verl's default W&B workspace.
        aliases = {
            "loss": ("actor/loss", "actor/pg_loss"),
            "lr": ("actor/lr",),
            "grad_norm": ("actor/grad_norm",),
            "entropy": ("actor/entropy",),
            "ppo_kl": ("actor/ppo_kl",),
        }
        for source, targets in aliases.items():
            if source in metrics:
                for target in targets:
                    metrics.setdefault(target, float(metrics[source]))

        metrics.setdefault("perf/mfu/actor", 0.0)
        metrics.setdefault("perf/mfu/actor_infer", 0.0)
        metrics["train/num_gen_batches"] = 1.0

        seq_lengths = [float(p + r) for p, r in zip(prompt_lengths, response_lengths)]
        if seq_lengths:
            metrics["global_seqlen/max"] = max(seq_lengths)
            metrics["global_seqlen/min"] = min(seq_lengths)
            metrics["global_seqlen/mean"] = sum(seq_lengths) / len(seq_lengths)
            metrics["global_seqlen/minmax_diff"] = (
                metrics["global_seqlen/max"] - metrics["global_seqlen/min"]
            )
            # The Ray dispatcher partitions the global batch across actor DP
            # shards. Greedy balancing provides the same workload range used by
            # the verl workspace without mutating the actual batch order here.
            dp_size = max(1, int(self._actor_dp_size or 1))
            loads = [0.0] * dp_size
            for length in sorted(seq_lengths, reverse=True):
                idx = min(range(dp_size), key=loads.__getitem__)
                loads[idx] += length
            metrics["global_seqlen/balanced_max"] = max(loads)
            metrics["global_seqlen/balanced_min"] = min(loads)

        # This workload has one model turn and no tools per sample.
        metrics["num_turns/max"] = 1.0
        metrics["num_turns/mean"] = 1.0
        metrics["num_turns/min"] = 1.0

        gen_time = float(timing_raw.get("gen", 0.0))
        reward_time = float(timing_raw.get("reward", 0.0))
        for suffix in ("max", "mean", "min"):
            metrics[f"timing_s/agent_loop/generate_sequences/{suffix}"] = gen_time
            metrics[f"timing_s/agent_loop/compute_score/{suffix}"] = reward_time
            metrics[f"timing_s/agent_loop/num_preempted/{suffix}"] = 0.0
            metrics[f"timing_s/agent_loop/tool_calls/{suffix}"] = 0.0
        metrics["timing_s/agent_loop/slowest/generate_sequences"] = gen_time
        metrics["timing_s/agent_loop/slowest/compute_score"] = reward_time
        metrics["timing_s/agent_loop/slowest/num_preempted"] = 0.0
        metrics["timing_s/agent_loop/slowest/tool_calls"] = 0.0
        metrics["timing_s/agent_loop/slowest/prompt_length"] = float(
            max(prompt_lengths, default=0)
        )
        metrics["timing_s/agent_loop/slowest/response_length"] = float(
            max(response_lengths, default=0)
        )

        for name in ("save_checkpoint", "start_profile", "stop_profile"):
            metrics.setdefault(f"timing_s/{name}", 0.0)

    @staticmethod
    def _display_step(step: int) -> int:
        """verl-style 1-based global step for logs/checkpoints/W&B."""
        return int(step) + 1

    def _train_with_ray_controller(self) -> None:
        """Ray worker orchestration path (no torch.distributed collectives)."""
        if self._algorithm is None or self._actor_wg is None:
            raise RuntimeError("Call setup() before train().")
        if self._atom_engine is None and self._ray_vllm_engine is None:
            raise RuntimeError(
                "Ray controller path requires an ATOM or ray_http rollout engine."
            )

        for cb in self.callbacks:
            cb.on_train_begin(self)

        num_generations = _algo_num_generations(self.config)
        total_steps = int(self.config.num_training_steps)
        start_step = self._resume_step
        self._prompt_cursor = self._resume_prompt_cursor(start_step, num_generations)
        use_ray_rollout = bool(
            (
                getattr(self, "_ray_use_vllm", False)
                or getattr(self, "_ray_use_atom", False)
            )
            and self._ray_vllm_engine is not None
        )
        if start_step > 0:
            logger.info(
                "Skipping global steps 1..%d (resuming from checkpoint).", start_step
            )
            if use_ray_rollout:
                logger.info(
                    "Resume: syncing restored actor weights to Ray rollout before first rollout."
                )
                self._sync_weights_ipc()

        # Policy version loaded in the rollout engine, counted in optimizer steps.
        rollout_version = start_step
        pipeline_mode = self._resolve_rollout_pipeline(use_ray_rollout)
        if pipeline_mode == "one_step" and start_step < total_steps:
            from atomrl.trainer.one_step_off_policy import OneStepOffPolicyRollout

            self._rollout_pipeline = OneStepOffPolicyRollout(
                lambda s: self._collect_rollout_batch(s, num_generations),
                total_steps,
            )
            self._rollout_pipeline.submit(start_step, rollout_version)
            logger.info(
                "One-step off-policy: each step's batch generates on the rollout pool "
                "while the previous step trains."
            )
        elif pipeline_mode == "fully_async" and start_step < total_steps:
            self._rollout_pipeline = self._start_streaming_rollout(
                num_generations,
                rollout_version,
            )
        background_val = None
        val_blocks_sync = False
        if (
            pipeline_mode == "fully_async"
            and self._rollout_pipeline is not None
            and self.config.controller.ray.fully_async.overlap_validation
        ):
            from atomrl.trainer.fully_async_rollout import BackgroundValidation

            streaming = self._rollout_pipeline
            background_val = BackgroundValidation(
                lambda s: self.run_validation(step=s),
                lambda: streaming.version,
            )
            # Without partial rollout the engine loads weights only when idle.
            val_blocks_sync = not self.config.controller.ray.fully_async.partial_rollout

        for step in range(start_step, total_steps):
            step_start = time.time()
            self.global_step = step
            self._maybe_start_profile(step)
            for cb in self.callbacks:
                cb.on_step_begin(self, step)
            self._reset_actor_memory_stats()

            # ---- rollout ----
            gen_t0 = time.time()
            rollout_lp = None
            if self._rollout_pipeline is not None:
                rollout, gen_time, async_metrics = self._rollout_pipeline.take(
                    step,
                    rollout_version,
                )
                (
                    sequences,
                    seq_mask,
                    prompt_lengths,
                    rewards,
                    responses,
                    ground_truths_expanded,
                    rollout_lp,
                    rollout_ragged,
                ) = rollout
            elif use_ray_rollout:
                # verl-aligned online rollout across colocated replicas, with
                # DAPO filter_groups dynamic sampling handled in the collector.
                (
                    sequences,
                    seq_mask,
                    prompt_lengths,
                    rewards,
                    responses,
                    ground_truths_expanded,
                    rollout_lp,
                    rollout_ragged,
                ) = self._collect_rollout_batch(step, num_generations)
                # free rollout KV so the FSDP training pass has GPU headroom.
                self._ray_vllm_engine.sleep()
            else:
                prompts, ground_truths = self._get_batch_prompts(step)
                sequences, seq_mask, prompt_lengths = self._rollout_with_atom(
                    prompts, num_generations
                )
                rewards, responses = None, None
                ground_truths_expanded = ground_truths * num_generations
                rollout_ragged = {}
            if self._rollout_pipeline is None:
                gen_time = time.time() - gen_t0

            _rc_cfg_ray = self.config.quantization.rollout_correction
            _bypass_ray = getattr(_rc_cfg_ray, "bypass_mode", False)

            old_log_prob_t0 = time.time()
            if _bypass_ray and rollout_lp is not None:
                old_log_probs = rollout_lp
                entropy_full = None
                if self._rank == 0:
                    logger.info(
                        "[step=%d] bypass_mode (ray): using rollout_log_probs as old_log_probs",
                        step,
                    )
            else:
                old_log_probs, entropy_full = self._compute_log_probs_with_worker_group(
                    self._actor_wg,
                    sequences,
                    role="actor",
                    want_entropy=True,
                    attention_mask=seq_mask,
                    rollout_routing=rollout_ragged.get("rollout_routing"),
                )
            old_log_prob_time = time.time() - old_log_prob_t0
            ref_t0 = time.time()
            if self._ref_wg is not None:
                ref_log_probs = self._compute_log_probs_with_worker_group(
                    self._ref_wg,
                    sequences,
                    role="ref",
                    attention_mask=seq_mask,
                )
            else:
                ref_log_probs = torch.zeros_like(old_log_probs)

            ref_time = time.time() - ref_t0
            reward_t0 = time.time()
            if rewards is None:
                rewards, responses = self._compute_rewards(
                    sequences,
                    seq_mask,
                    prompt_lengths,
                    ground_truths,
                    num_generations,
                )
            reward_time = time.time() - reward_t0
            response_mask = self._build_response_mask(
                sequences, seq_mask, prompt_lengths
            )
            response_lengths = [
                int(response_mask[i].sum().item())
                for i in range(response_mask.shape[0])
            ]

            tensors = {
                "input_ids": sequences,
                "attention_mask": seq_mask,
                "old_log_probs": old_log_probs,
                "ref_log_probs": ref_log_probs,
                "rewards": rewards,
                "response_mask": response_mask,
            }
            if rollout_lp is not None:
                tensors["rollout_log_probs"] = rollout_lp
            batch = DataProto(
                tensors=tensors,
                meta={
                    "algorithm": self.config.algorithm.name,
                    "response_lengths": response_lengths,
                    "responses": responses,
                    "ground_truths": ground_truths_expanded,
                    "algo_config": self._to_plain_dict(self.config.algorithm),
                    # verl-aligned packed training forward divides logits by the
                    # sampling temperature (logits.div_(temperature)); pass it so
                    # the worker applies the same convention as old/ref log-probs.
                    "temperature": float(
                        getattr(
                            self.config.policy.generation.vllm_cfg, "temperature", 1.0
                        )
                        or 1.0
                    ),
                    # Keep packed training on the same MILES token budget as
                    # old-logprob inference. Without this, the engine falls
                    # back to 8192 and creates roughly 4x as many backwards.
                    "max_token_len_per_gpu": int(
                        self.config.policy.max_token_len_per_gpu
                    ),
                },
                # per-row, variable-length: R3 rollout routing rides here so that
                # filter_groups selection and the DP split reindex it with the rows
                ragged=rollout_ragged,
            )
            batch.check_consistency()

            adv_t0 = time.time()
            batch = self._algorithm.compute_advantages(batch)
            batch = apply_rollout_correction(batch, self.config)

            # --- Extended rollout correction: IS weights + RS (Ray path) ---
            _rc_metrics_ray = {}
            _position_mismatch_metrics = {}
            _rlp_ray = batch.tensors.get(
                "rollout_log_probs", batch.tensors.get("fp8_logprobs")
            )
            _rc_want_ray = (
                (_rc_cfg_ray.rollout_is or _rc_cfg_ray.rollout_rs or _bypass_ray)
                and "old_log_probs" in batch.tensors
                and _rlp_ray is not None
            )
            _r3_verify_metrics: dict[str, float] = {}
            if (
                self._r3_print_train_rollout_mismatch
                and "old_log_probs" in batch.tensors
                and _rlp_ray is not None
            ):
                _r3_verify_metrics = self._r3_verify_old_vs_rollout(
                    batch.tensors["old_log_probs"],
                    _rlp_ray,
                    response_mask,
                )
                if self._rank == 0 and _r3_verify_metrics:
                    logger.info(
                        "[step=%d] r3_verify tokens=%d kl=%.6g abs_diff=%.6g frac_abs_gt_0.1=%.6g",
                        step,
                        int(_r3_verify_metrics["r3_verify/tokens"]),
                        _r3_verify_metrics["r3_verify/kl"],
                        _r3_verify_metrics["r3_verify/abs_diff"],
                        _r3_verify_metrics["r3_verify/frac_abs_gt_0.1"],
                    )
            if _rc_want_ray:
                from atomrl.algorithms.rollout_correction import (
                    compute_rollout_correction_and_add_to_batch,
                )

                batch, _rc_metrics_ray = compute_rollout_correction_and_add_to_batch(
                    batch, _rc_cfg_ray
                )
                _position_mismatch_metrics = self._mismatch_by_response_position(
                    old_log_probs,
                    _rlp_ray,
                    response_mask,
                )
            adv_time = time.time() - adv_t0

            if use_ray_rollout:
                # verl-aligned loss normalization: GLOBAL response-token count
                # (full batch) + dp_size so each actor's shard normalizes by the
                # global denominator and FSDP grad averaging yields token-mean.
                _rmask = batch.tensors.get("response_mask")
                if _rmask is not None:
                    batch.meta["batch_num_tokens"] = int(_rmask.sum().item())
                # Loss normalization divides by the number of DP shards (grad is
                # averaged across DP×CP, while the differentiable CP gather's
                # backward SUM cancels the CP average) = pure DP width.
                batch.meta["dp_size"] = int(
                    self._actor_dp_size
                    or (self._actor_wg.num_workers // max(1, self._actor_mp))
                )

            # ---- train (worker-side PPO mini-batch loop; FSDP grad sync) ----
            train_t0 = time.time()
            metrics = self._update_actor_with_ray(batch)
            train_time = time.time() - train_t0

            # verl-aligned GLOBAL token-weighted KL (Σkl / Σtok), not a mean of
            # per-sequence means (which over-weights short/outlier seqs and makes
            # core/kl look far noisier than verl). Workers return sum+tok.
            _ks = metrics.pop("rollout_corr_kl_sum", None)
            _kt = metrics.pop("rollout_corr_kl_tok", None)
            if _ks is not None and _kt:
                metrics["rollout_corr/kl"] = _ks / max(_kt, 1e-6)
            _ps = metrics.pop("ppo_kl_sum", None)
            _pt = metrics.pop("ppo_kl_tok", None)
            if _ps is not None and _pt:
                metrics["ppo_kl"] = _ps / max(_pt, 1e-6)
            self._finalize_mismatch_metrics(metrics)

            if _rc_metrics_ray:
                metrics.update(_rc_metrics_ray)
            if _position_mismatch_metrics:
                metrics.update(_position_mismatch_metrics)
            if _r3_verify_metrics:
                metrics.update(_r3_verify_metrics)

            total_tok = int(seq_mask.sum().item())
            prompt_tok = int(sum(prompt_lengths))
            gen_tokens = max(0, total_tok - prompt_tok)
            metrics["timing/step_s"] = time.time() - step_start
            metrics["timing/gen_s"] = gen_time
            metrics["timing/ref_s"] = ref_time
            metrics["timing/train_s"] = train_time
            if gen_tokens > 0 and gen_time > 0:
                metrics["throughput/gen_tok_per_s"] = gen_tokens / gen_time
            metrics["reward/mean"] = float(rewards.mean().item())
            metrics["reward/accuracy"] = float(
                sum(1 for r in rewards if r > 0) / max(1, len(rewards))
            )
            metrics["seq/max_len"] = int(sequences.shape[1])
            metrics["seq/mean_response_len"] = float(
                sum(response_lengths) / max(1, len(response_lengths))
            )
            # verl-aligned actor/entropy: token-weighted mean over response tokens.
            if entropy_full is not None:
                _em = response_mask.to(entropy_full.device).to(entropy_full.dtype)
                _ec = (
                    entropy_full[..., : _em.shape[-1]]
                    if entropy_full.shape[-1] != _em.shape[-1]
                    else entropy_full
                )
                _denom = float(_em.sum().item())
                if _denom > 0:
                    metrics["entropy"] = float((_ec * _em).sum().item() / _denom)
                    if step < 2:
                        _olp = old_log_probs.to(_em.device)
                        _olp = _olp[..., : _em.shape[-1]]
                        _neglp = float((-_olp * _em).sum().item() / _denom)
                        logger.info(
                            "ENT_DIAG step=%d resp_ent=%.3f all_ent=%.3f resp_neglogp=%.3f mask_tok=%d ent_max=%.2f",
                            step,
                            metrics["entropy"],
                            float(entropy_full.mean().item()),
                            _neglp,
                            int(_denom),
                            float(entropy_full.max().item()),
                        )

            # ---- weight sync to rollout engine ----
            pipeline = self._rollout_pipeline
            sync_now = pipeline is None or pipeline.sync_due()
            val_blocked_s = 0.0
            if pipeline is not None:
                if sync_now:
                    if val_blocks_sync:
                        val_metrics, val_blocked_s = background_val.collect(wait=True)
                        metrics.update(val_metrics)
                    async_metrics["async/trainer_wait_s"] += pipeline.before_sync()
                metrics.update(async_metrics)
            t_sync = time.time()
            if sync_now and getattr(self.config.weight_sync, "enabled", True):
                if use_ray_rollout:
                    self._sync_weights_ipc()  # wake weights -> ZMQ IPC -> wake KV
                    rollout_version = step + 1
                else:
                    self._sync_rollout_weights()
            if pipeline is not None and sync_now:
                pipeline.after_sync(rollout_version)
            sync_time = time.time() - t_sync
            metrics["timing/weight_sync_s"] = sync_time
            if sync_now:
                metrics.update(self._last_weight_sync_metrics)
            metrics.update(self._collect_actor_memory_metrics())

            # Validation uses the rollout engine too, so run it after the fresh
            # actor weights have been synced and sleep/wake has restored KV cache.
            val_t0 = time.time()
            metrics.update(
                self._validate_at_step(step, total_steps, pipeline, background_val)
            )
            val_time = time.time() - val_t0 + val_blocked_s

            timing_raw = {
                "gen": gen_time,
                "old_log_prob": old_log_prob_time,
                "ref": ref_time,
                "reward": reward_time,
                "adv": adv_time,
                "update_actor": train_time,
                "update_weights": sync_time,
                "testing": val_time,
                "step": time.time() - step_start,
            }
            self._add_verl_metrics(
                metrics,
                batch,
                timing_raw,
                prompt_lengths,
                response_lengths,
            )

            self.last_metrics = metrics
            for k, v in metrics.items():
                self._metrics.update(k, v)

            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)

            self._maybe_stop_profile(step)

            del sequences, seq_mask, old_log_probs, ref_log_probs
            del rewards, responses, response_mask, batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self._rollout_pipeline is not None:
            self._rollout_pipeline.close()
            self._rollout_pipeline = None
        for cb in self.callbacks:
            cb.on_train_end(self)
        logger.info(
            "RLTrainer.train (ray-controller) finished after %d steps.", total_steps
        )

    def _validate_at_step(
        self,
        step: int,
        total_steps: int,
        pipeline: Any,
        background_val: Any,
    ) -> dict[str, float]:
        """Validation metrics to log at ``step``, validating if it is due.

        With a ``BackgroundValidation`` a due step starts one and a later step
        logs it. The last step validates in the foreground, since nothing is
        left to overlap with; when it does, the step before waits for the one
        running and starts none, so that two validations never land on one step.
        """
        val_steps = getattr(self.config, "val_steps", 0)
        due = val_steps > 0 and (step + 1) % val_steps == 0
        last = step == total_steps - 1
        clear_for_last = (
            background_val is not None
            and step == total_steps - 2
            and val_steps > 0
            and total_steps % val_steps == 0
        )
        done: dict[str, float] = {}
        if background_val is not None:
            # One validation at a time, and none left running past the end.
            done, _ = background_val.collect(wait=due or last or clear_for_last)
        if not due or clear_for_last:
            return done
        if background_val is not None and not last:
            background_val.start(step)
            return done
        with pipeline.submissions_paused() if pipeline is not None else nullcontext():
            return self.run_validation()

    def _resume_prompt_cursor(self, start_step: int, num_generations: int) -> int:
        """Roughly where a run resumed at ``start_step`` left the prompt cursor.

        One generation round of ``gen_batch_size`` prompts per step; exact
        alignment isn't required, only not replaying the prompts already seen.
        """
        pol = self.config.policy
        train_prompts = max(1, pol.train_global_batch_size // max(1, num_generations))
        gen_prompts = (
            int(pol.gen_batch_size) if pol.gen_batch_size > 0 else train_prompts
        )
        return start_step * gen_prompts

    def _resolve_rollout_pipeline(self, use_ray_rollout: bool) -> str:
        """``sync``, ``one_step`` or ``fully_async``; reject layouts that cannot overlap."""
        ray_cfg = self.config.controller.ray
        one_step = bool(getattr(ray_cfg, "one_step_off_policy", False))
        fully_async = bool(
            getattr(getattr(ray_cfg, "fully_async", None), "enabled", False)
        )
        if not one_step and not fully_async:
            return "sync"
        if one_step and fully_async:
            raise ValueError(
                "controller.ray.one_step_off_policy and controller.ray.fully_async.enabled "
                "are alternatives; set one of them."
            )
        name = (
            "controller.ray.fully_async"
            if fully_async
            else "controller.ray.one_step_off_policy"
        )
        if not use_ray_rollout or getattr(self, "_rollout_wg", None) is None:
            raise ValueError(
                f"{name} generates while the trainer trains, so the rollout needs "
                "GPUs of its own: set controller.ray.rollout.num_workers > 0 with a "
                "Ray vLLM or ATOM rollout."
            )
        if getattr(self._ray_vllm_engine, "enable_sleep", False):
            raise ValueError(
                f"{name} keeps the rollout generating while the trainer runs, so it "
                "cannot sleep between steps: turn off enable_sleep_mode for the "
                "rollout engine."
            )
        if not fully_async:
            return "one_step"
        if getattr(self, "_ray_use_atom", False):
            if bool(getattr(ray_cfg.fully_async, "partial_rollout", False)) and (
                "keep" not in self._ray_rollout_mgr.generation_pause_modes()
            ):
                raise ValueError(
                    "controller.ray.fully_async.partial_rollout on the ATOM rollout needs "
                    "an ATOM that can pause requests in flight across a weight sync "
                    "(engine utility pause_generation, mode 'keep'); this one cannot. "
                    "Set partial_rollout=false to let each sync wait for them instead."
                )
        elif not getattr(self, "_ray_use_vllm", False):
            raise ValueError(
                "controller.ray.fully_async needs the Ray vLLM or ATOM rollout."
            )
        if self._r3_rollout_replay or getattr(self.config.moe.r3, "enabled", False):
            raise ValueError(
                "controller.ray.fully_async does not carry R3 rollout routing across "
                "a weight sync yet; turn off moe.r3."
            )
        return "fully_async"

    def _start_streaming_rollout(
        self, num_generations: int, rollout_version: int
    ) -> Any:
        from atomrl.trainer.fully_async_rollout import StreamingRollout

        fa_cfg = self.config.controller.ray.fully_async
        pol = self.config.policy
        groups_per_step = max(1, pol.train_global_batch_size // num_generations)
        mgr = self._ray_rollout_mgr
        max_inflight = int(fa_cfg.max_inflight_groups) or mgr.num_replicas * max(
            1,
            int(pol.generation.vllm_cfg.max_num_seqs) // num_generations,
        )
        fg = (
            getattr(self.config.algorithm.dapo, "filter_groups", None)
            if self.config.algorithm.name.lower() == "dapo"
            else None
        )
        use_filter = bool(fg is not None and fg.enable)
        # The synchronous loop gives up after max_num_gen_batches rounds of
        # gen_batch_size prompts without one valid group; stop at the same count.
        gen_prompts = (
            int(pol.gen_batch_size) if pol.gen_batch_size > 0 else groups_per_step
        )
        max_drops = (
            int(fg.max_num_gen_batches) * gen_prompts
            if use_filter and fg.max_num_gen_batches > 0
            else 0
        )
        pipeline = StreamingRollout(
            _RayRolloutGroups(self, num_generations, use_filter),
            self._merge_rollout_groups,
            groups_per_step=groups_per_step,
            sync_every=int(fa_cfg.trigger_parameter_sync_step),
            staleness_threshold=float(fa_cfg.staleness_threshold),
            max_inflight=max_inflight,
            partial_rollout=bool(fa_cfg.partial_rollout),
            max_consecutive_drops=max_drops,
            start_version=rollout_version,
            start_index=self._prompt_cursor,
        )
        pipeline.start()
        logger.info(
            "Fully async rollout: %d prompt groups per step, weight sync every %d "
            "steps, staleness_threshold=%.2f, partial_rollout=%s, up to %d groups "
            "in flight, filter_groups=%s.",
            groups_per_step,
            int(fa_cfg.trigger_parameter_sync_step),
            float(fa_cfg.staleness_threshold),
            bool(fa_cfg.partial_rollout),
            max_inflight,
            use_filter,
        )
        return pipeline

    def _merge_rollout_groups(self, groups: list[tuple]) -> tuple:
        """Stack prompt groups into the tuple ``_collect_rollout_batch`` returns."""
        pad_id = self._tokenizer.pad_token_id or 0
        log_probs = [g[6] for g in groups]
        ragged: dict[str, list[Any]] = {}
        for g in groups:
            for key, rows in g[7].items():
                ragged.setdefault(key, []).extend(rows)
        return (
            _merge_left_padded([g[0] for g in groups], pad_id),
            _merge_left_padded([g[1] for g in groups], 0),
            [length for g in groups for length in g[2]],
            torch.cat([g[3] for g in groups], dim=0),
            [response for g in groups for response in g[4]],
            [gt for g in groups for gt in g[5]],
            (
                _merge_left_padded(log_probs, 0.0)
                if all(lp is not None for lp in log_probs)
                else None
            ),
            ragged,
        )

    def _sync_rollout_weights(self) -> None:
        """Sync actor weights to ATOM rollout engine if configured."""
        if self._use_atom and self._atom_engine is not None:
            self._sync_weights_to_atom()

    def run_validation(self, step: int | None = None) -> dict[str, float]:
        """Evaluate on the validation set with configured sampling.

        Reports ``val-core/acc/mean@N`` (fraction correct) plus mean
        reward and response length. Generation is colocated (rank 0 generates via
        the inference engine, broadcast to all ranks) so every rank computes
        identical metrics. ``eval.num_samples <= 0`` evaluates the full val set,
        matching verl's val dataloader behavior. ``step`` labels the log lines
        and defaults to the current step; a background validation passes the
        step it validates, since training moves on while it runs.
        """
        if self._val_dataset is None or len(self._val_dataset) == 0:
            return {}
        label_step = (self.global_step if step is None else step) + 1

        from atomrl.rewards.math_reward import compute_math_reward

        cap = int(getattr(self.config.eval, "num_samples", 0) or 0)
        num_samples = len(self._val_dataset)
        if cap > 0:
            num_samples = min(num_samples, cap)
        val_bs_cfg = int(getattr(self.config, "val_batch_size", 16) or 0)
        val_bs = num_samples if val_bs_cfg <= 0 else max(1, val_bs_cfg)
        eval_n = max(1, int(getattr(self.config.eval, "num_generations", 1) or 1))
        eval_sp = self._ray_eval_sampling_params()

        all_scores: list[float] = []
        all_acc: list[float] = []
        all_response_lengths: list[int] = []
        all_responses: list[str] = []

        if self._rank == 0:
            logger.info(
                "[eval] step=%d: evaluating %d prompts x %d samples "
                "(temperature=%.3g, top_p=%.3g, batch_size=%d)",
                label_step,
                num_samples,
                eval_n,
                eval_sp["temperature"],
                eval_sp["top_p"],
                val_bs,
            )

        size_divisor = 1
        if (
            getattr(self, "_ray_vllm_engine", None) is not None
            and self._actor_wg is not None
        ):
            size_divisor = max(1, int(self._actor_wg.num_workers))

        for start in range(0, num_samples, val_bs):
            end = min(start + val_bs, num_samples)
            samples = [self._val_dataset[idx] for idx in range(start, end)]
            prompts, ground_truths = [], []
            for s in samples:
                p, gt = self._extract_prompt_gt(s, ensure_answer_format=True)
                prompts.append(p)
                ground_truths.append(gt)
            real_batch = len(prompts)

            # verl pads validation batches to rollout DP size and unpads after
            # generation. Mirror that behavior for Ray vLLM routing.
            pad_size = 0
            if size_divisor > 1 and real_batch > 0:
                remainder = real_batch % size_divisor
                if remainder:
                    pad_size = size_divisor - remainder
                    for i in range(pad_size):
                        src = i % real_batch
                        prompts.append(prompts[src])
                        ground_truths.append(ground_truths[src])

            # Eval generation (colocated). Ray-controller path generates via the
            # colocated vLLM replicas (self._actor_model is None on the driver, so
            # the torchrun _rollout_phase fallback would crash).
            if getattr(self, "_ray_vllm_engine", None) is not None:
                sequences, seq_mask, prompt_lengths, _, _ = self._rollout_with_ray_vllm(
                    prompts,
                    num_generations=eval_n,
                    sampling_params=eval_sp,
                )
            elif self._use_vllm and self._atom_engine is not None:
                sequences, seq_mask, prompt_lengths, _ = self._rollout_with_vllm(
                    prompts,
                    num_generations=eval_n,
                    eval_mode=True,
                )
            elif self._use_atom and self._atom_engine is not None:
                sequences, seq_mask, prompt_lengths = self._rollout_with_atom(
                    prompts, num_generations=eval_n
                )
            else:
                input_ids, attention_mask = self._tokenize_prompts(prompts)
                sequences, seq_mask, prompt_lengths = self._rollout_phase(
                    input_ids, attention_mask, num_generations=eval_n
                )

            # Decode + score (identical on every rank — sequences are broadcast).
            seq_cpu = sequences.cpu()
            mask_cpu = seq_mask.cpu()
            responses = []
            keep = max(0, seq_cpu.shape[0] - pad_size * eval_n)
            seq_cpu = seq_cpu[:keep]
            prompt_lengths = prompt_lengths[:keep]
            seq_mask = seq_mask[:keep]
            ground_truths = [
                ground_truth for ground_truth in ground_truths for _ in range(eval_n)
            ][:keep]
            for i in range(seq_cpu.shape[0]):
                plen = int(prompt_lengths[i]) if i < len(prompt_lengths) else 0
                response_ids = _response_token_ids(seq_cpu[i], mask_cpu[i], plen)
                responses.append(
                    self._tokenizer.decode(response_ids, skip_special_tokens=True)
                )
            rewards_t, details = compute_math_reward(responses, ground_truths)

            all_scores.extend(rewards_t.tolist())
            all_acc.extend([1.0 if d["acc"] else 0.0 for d in details])
            all_responses.extend(responses)
            response_mask = self._build_response_mask(
                sequences, seq_mask, prompt_lengths
            )
            all_response_lengths.extend(
                [int(x) for x in response_mask.sum(dim=-1).tolist()]
            )

        # Keep the Ray rollout lifecycle consistent across training and
        # validation: after any generation phase, release rollout-side KV/graphs
        # before the next training phase. The next weight sync will wake weights
        # and KV cache in the same order used for normal training rollout.
        if getattr(self, "_ray_vllm_engine", None) is not None:
            try:
                self._ray_vllm_engine.sleep()
            except _ray_boundary_errors():
                logger.exception("Ray rollout sleep after validation failed")

        # Non-persistent path frees the eval vLLM (kill subprocess: reliable GPU
        # release on ROCm). Persistent path keeps the resident engine(s) alive.
        if (
            self._use_vllm
            and not self._vllm_persistent
            and self._atom_engine is not None
        ) and (self._vllm_dp or self._rank == 0):
            try:
                self._atom_engine.sleep()
            except (OSError, RuntimeError):
                pass
        if self._is_distributed:
            torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not all_scores:
            return {}

        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        acc_t = torch.tensor(all_acc, dtype=torch.float32)
        lengths_t = torch.tensor(all_response_lengths, dtype=torch.float32)

        metrics: dict[str, float] = {
            f"val-core/acc/mean@{eval_n}": float(acc_t.mean()),
            f"val-core/math_dapo/acc/mean@{eval_n}": float(acc_t.mean()),
            f"val-aux/math_dapo/reward/mean@{eval_n}": float(scores_t.mean()),
            f"val-aux/math_dapo/score/mean@{eval_n}": float(scores_t.mean()),
            "val-aux/num_turns/max": 1.0,
            "val-aux/num_turns/mean": 1.0,
            "val-aux/num_turns/min": 1.0,
            "val/score_mean": float(scores_t.mean()),
            "val/response_length_mean": float(lengths_t.mean()),
            "val/num_samples": float(len(all_scores)),
        }
        self._last_val_generations = [
            {
                "response": response,
                "score": float(score),
                "accuracy": float(acc),
                "response_length": int(length),
            }
            for response, score, acc, length in zip(
                all_responses, all_scores, all_acc, all_response_lengths
            )
        ]
        if self._rank == 0:
            logger.info(
                "[eval] step=%d acc=%.4f score_mean=%.4f resp_len=%.1f n=%d",
                label_step,
                metrics[f"val-core/acc/mean@{eval_n}"],
                metrics["val/score_mean"],
                metrics["val/response_length_mean"],
                len(all_scores),
            )
            num_print = min(
                getattr(self.config.logger, "num_val_samples_to_print", 3),
                len(all_responses),
            )
            for i in range(num_print):
                logger.info(
                    "[eval] sample %d acc=%.0f resp=...%s",
                    i,
                    all_acc[i],
                    repr(all_responses[i][-160:]),
                )

        return metrics

    def cleanup(self) -> None:
        """Release all resources."""

        def best_effort(label: str, callback: Callable[[], Any]) -> None:
            try:
                callback()
            # Cleanup crosses profiler, Ray, and backend boundaries whose
            # implementations can raise backend-specific exception types.
            except Exception:
                logger.exception("Failed to clean up %s", label)

        if self._profiler is not None:
            best_effort("profiler", self._profiler.stop)
            self._profiler = None
        if self._rollout_pipeline is not None:
            best_effort("rollout pipeline", self._rollout_pipeline.close)
            self._rollout_pipeline = None
        if self._ray_rollout_mgr is not None and getattr(
            self._ray_rollout_mgr, "rdma_group_name", None
        ):
            best_effort(
                "RDMA weight group",
                lambda: self._ray_rollout_mgr.destroy_rdma_weight_group(self._actor_wg),
            )
        # Rollout actors must be stopped while Ray is still alive. In
        # particular, vLLM's manager needs Ray RPCs to terminate its workers.
        if self._ray_vllm_engine is not None:
            best_effort("Ray rollout engine", self._ray_vllm_engine.shutdown)
            self._ray_vllm_engine = None
            self._ray_rollout_mgr = None
        if self._atom_engine is not None:
            best_effort("local rollout engine", self._atom_engine.shutdown)
            self._atom_engine = None
        if self._critic_worker is not None:
            best_effort("critic worker", self._critic_worker.cleanup)
            self._critic_worker = None
        if self._actor_wg is not None:
            best_effort("actor worker group", self._actor_wg.stop)
            self._actor_wg = None
        if self._ref_wg is not None:
            best_effort("reference worker group", self._ref_wg.stop)
            self._ref_wg = None
        if self._ray_cluster is not None:
            best_effort("Ray cluster", self._ray_cluster.shutdown)
            self._ray_cluster = None
        if self._engine is not None:
            del self._engine
            self._engine = None
        if self._actor_model is not None:
            del self._actor_model
        if self._ref_model is not None:
            del self._ref_model
        self._actor_model = None
        self._ref_model = None
        self._optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("[rank %d] RLTrainer.cleanup complete.", self._rank)


__all__ = ["RLTrainer"]
