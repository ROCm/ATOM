# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Policy (actor) worker: training backends and log-prob computation."""

from __future__ import annotations

import gc
import logging
import os
import re
import resource
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from atomrl.algorithms.loss_functions import (
    asymmetric_clip_loss,
    kl_penalty,
    policy_gradient_loss,
    reduce_reported_loss,
    sft_loss,
)
from atomrl.core.protocol import DataProto
from atomrl.core.types import AlgorithmName, TrainingBackend
from atomrl.engine.training.base_engine import BaseEngine, EngineRegistry
from atomrl.moe.r3_scope import assert_supported_r3, r3_fields_from_config
from atomrl.utils.checkpoint import (
    checkpoint_rank_phases,
    create_checkpoint_control_group,
    run_checkpoint_phase,
)

# base_worker must be imported before torch so that HIP_VISIBLE_DEVICES
# is set before HIP initialization (see base_worker.py module-level code).
from atomrl.workers.base_worker import BaseWorker, get_nested_config
from atomrl.workers.numa_affinity import (
    bind_current_process_to_gpu_numa,
    current_physical_gpu_id,
)

logger = logging.getLogger(__name__)


def _normalize_streamed_optimizer_chunk_size_mib(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(
            "streamed_optimizer_chunk_size_mib must be an integer or "
            "canonical positive integer string"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[1-9]\d*", value):
        return int(value)
    raise ValueError(
        "streamed_optimizer_chunk_size_mib must be an integer or "
        "canonical positive integer string"
    )


def _layers_in_bf16(config: dict[str, Any], key: str) -> int:
    """Resolve an FP8 BF16 keep-out: training wins, rollout is the fallback.

    Rollout stays the fallback because that is where every config sets them.
    """
    for section in ("training", "rollout"):
        value = get_nested_config(config, "quantization", section, key, default=None)
        if value is not None:
            return int(value)
    return 0


def _whole_tensors(params):
    """Each parameter whole, gathered on every rank in the same order.

    FSDP2 yields DTensor shards, and ``full_tensor()`` is a collective: a rank
    that skips one leaves its peers in that all-gather. So every rank consumes
    this, and only the writer keeps what it yields. Megatron's tensors are
    already whole and pass through.
    """
    from torch.distributed.tensor import DTensor

    for name, param in params:
        yield name, param.full_tensor() if isinstance(param, DTensor) else param


class AtomActorWorker(BaseWorker):
    """Trainable policy worker using the Engine abstraction layer.

    Delegates model construction, optimizer, LR scheduling, and offload
    management to the Engine layer (FSDP2 or Megatron).
    """

    def __init__(
        self, rank: int, world_size: int, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(rank, world_size, config)
        self._engine: BaseEngine | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._peak_carry_alloc = 0.0
        self._peak_carry_reserved = 0.0

    def init_model(self, forward_only: bool = False) -> None:
        """Build policy network via EngineRegistry."""
        if forward_only:
            os.environ["ATOMRL_FORWARD_ONLY_INIT"] = "1"
            os.environ["MHC_BACKEND"] = "triton"
            os.environ["TILEKERNELS_DIR"] = "/workspace/TileKernels"
            for source_root in (
                "/workspace/Megatron-LM",
                "/workspace/aiter",
                "/workspace/TileKernels",
            ):
                if Path(source_root).is_dir() and source_root not in sys.path:
                    sys.path.insert(0, source_root)
        policy = get_nested_config(self.config, "policy", default={}) or {}
        backend_raw = str(
            policy.get("training_backend", TrainingBackend.FSDP2.value)
        ).lower()

        if backend_raw in ("fsdp", "fsdp2"):
            backend_key = "fsdp2"
        elif backend_raw in ("megatron_native", "megatron-native"):
            backend_key = "megatron_native"
        elif backend_raw in ("megatron",):
            backend_key = "megatron"
        elif backend_raw in ("megatron_lumen_dsv4",):
            backend_key = "megatron_lumen_dsv4"
        else:
            raise ValueError(f"Unknown policy.training_backend: {backend_raw}")

        model_name = str(policy.get("model_name", ""))
        training_cfg = policy.get("training", {}) or {}
        quant = (
            get_nested_config(self.config, "quantization", "training", default={}) or {}
        )

        megatron_cfg = training_cfg.get("megatron_cfg", {}) or {}
        if bool(megatron_cfg.get("numa_affinity", False)):
            try:
                physical_gpu_id = current_physical_gpu_id()
            except (OSError, RuntimeError, ValueError) as exc:
                self._log.warning(
                    "NUMA affinity GPU detection failed; continuing unbound: %s",
                    exc,
                )
            else:
                bind_current_process_to_gpu_numa(
                    physical_gpu_id,
                    logger=self._log,
                )

        engine_config = self._build_engine_config(backend_key, training_cfg, policy)
        optimizer_config = self._build_optimizer_config(policy)
        model_config = self._build_model_config(policy)

        engine_cls = EngineRegistry.get_engine_cls(
            model_type="language_model",
            backend=backend_key,
        )

        if backend_key in ("fsdp", "fsdp2"):
            self._engine = engine_cls(
                model_config=model_config,
                engine_config=engine_config,
                optimizer_config=optimizer_config,
                model_name=model_name,
                quant_config=quant,
            )
        else:
            self._engine = engine_cls(
                model_config=model_config,
                engine_config=engine_config,
                optimizer_config=optimizer_config,
                model_name=model_name,
            )

        self._engine.initialize()

        self._dispatch_dp_rank = self._engine.get_data_parallel_rank()
        self._is_collect = self._engine.is_mp_src_rank_with_outputs()
        self._log.info(
            "AtomActorWorker: initialized %s engine (dp_rank=%d, is_collect=%s).",
            backend_key,
            self._dispatch_dp_rank,
            self._is_collect,
        )

    def get_dp_rank(self) -> int:
        return getattr(self, "_dispatch_dp_rank", self.rank)

    def get_is_collect(self) -> bool:
        return getattr(self, "_is_collect", True)

    def _probe_forward_entropy(self, model_name: str) -> None:
        """One-time sanity probe: entropy of the FSDP forward on a fixed clean
        sequence. A correct base-model forward gives ~1-1.5; ~4+ indicates the
        sharded forward is degenerate."""
        try:
            from transformers import AutoTokenizer

            tk = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            base = tk(
                "Question: What is 12*13? Answer: 12*13 = 156.",
                add_special_tokens=False,
            )["input_ids"]

            def _probe(ids, npad, tag):
                real = list(ids)
                seq = torch.tensor([[0] * npad + real], device=self._device)
                am = torch.tensor([[0] * npad + [1] * len(real)], device=self._device)
                data = {
                    "input_ids": seq,
                    "attention_mask": am,
                    "position_ids": (am.long().cumsum(-1) - 1).clamp(min=0),
                    "meta": {"calculate_entropy": True},
                }
                with self._engine.eval_mode():
                    out = self._engine.infer_batch(data)
                em = am[:, 1:].float()
                e = out["model_output"]["entropy"].float()
                ent = float((e * em).sum() / em.sum().clamp(min=1))
                self._log.info(
                    "PROBE_FWD %s entropy=%.3f (len=%d pad=%d)",
                    tag,
                    ent,
                    len(real),
                    npad,
                )

            _probe(base, 0, "short_nopad")
            _probe(base, 512, "short_pad512")
            _probe(base * 40, 512, "long_pad512")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self._log.warning("PROBE_FWD failed: %s", exc)

    def get_mp_info(self) -> dict[str, Any]:
        """Report this actor's Megatron model-parallel coordinates.

        Used by the controller to build the DP x (TP,PP,CP) data ``mesh_mapping`` and
        the loss-normalizing DP size. Falls back to pure-DP (rank/world) when
        Megatron model-parallel state is not initialized (for example FSDP2).
        """
        try:
            from megatron.core import parallel_state as mpu

            if mpu.is_initialized():
                return {
                    # Exclude CP: all CP ranks holding different token chunks of
                    # one sequence must receive the same controller data shard.
                    "dp_rank": int(
                        mpu.get_data_parallel_rank(with_context_parallel=False)
                    ),
                    "dp_size": int(
                        mpu.get_data_parallel_world_size(with_context_parallel=False)
                    ),
                    "tp_rank": int(mpu.get_tensor_model_parallel_rank()),
                    "pp_rank": int(mpu.get_pipeline_model_parallel_rank()),
                    "cp_rank": int(mpu.get_context_parallel_rank()),
                    "cp_size": int(mpu.get_context_parallel_world_size()),
                    "is_last_stage": bool(mpu.is_pipeline_last_stage()),
                }
        except (AssertionError, RuntimeError):
            pass
        return {
            "dp_rank": int(self.rank),
            "dp_size": int(self.world_size),
            "tp_rank": 0,
            "pp_rank": 0,
            "cp_rank": 0,
            "cp_size": 1,
            "is_last_stage": True,
        }

    def _build_engine_config(
        self,
        backend: str,
        training_cfg: dict,
        policy: dict,
    ) -> dict[str, Any]:
        if backend in ("fsdp", "fsdp2"):
            fsdp_cfg = training_cfg.get("fsdp_cfg") or {}
            if not isinstance(fsdp_cfg, dict):
                from dataclasses import asdict, is_dataclass

                fsdp_cfg = (
                    asdict(fsdp_cfg) if is_dataclass(fsdp_cfg) else dict(vars(fsdp_cfg))
                )
            # Mixed-precision (verl-aligned): keep FP32 master weights so small
            # Adam updates (lr ~1e-6 ~= 5e-7/step) accumulate. Storing the master
            # in bf16 (eps ~8e-3) silently rounds these updates away -> the policy
            # never changes (grad_norm looks normal but entropy/reward stay flat).
            # Compute/all-gather in bf16 via FSDP2 MixedPrecisionPolicy.
            compute_dtype = training_cfg.get("optimizer_dtype", "bf16")
            return {
                "param_offload": fsdp_cfg.get("param_offload", False),
                "optimizer_offload": fsdp_cfg.get("optimizer_offload", False),
                "grad_offload": fsdp_cfg.get("grad_offload", False),
                "reshard_after_forward": fsdp_cfg.get("reshard_after_forward", True),
                "model_dtype": "fp32",
                "mixed_precision": {
                    "param_dtype": compute_dtype,
                    "reduce_dtype": "fp32",
                },
                "seed": int(policy.get("seed", 42)),
                "tiny_lm": training_cfg.get("tiny_lm") or None,
            }
        elif backend in (
            "megatron_native",
            "megatron-native",
            "megatron",
            "megatron_lumen_dsv4",
        ):
            meg_cfg = (
                training_cfg.get("megatron_cfg") or policy.get("megatron_cfg") or {}
            )
            if not isinstance(meg_cfg, dict):
                from dataclasses import asdict, is_dataclass

                if isinstance(meg_cfg, Mapping):
                    meg_cfg = dict(meg_cfg)
                else:
                    meg_cfg = (
                        asdict(meg_cfg)
                        if is_dataclass(meg_cfg)
                        else dict(vars(meg_cfg))
                    )
            r3_cfg = get_nested_config(self.config, "moe", "r3", default={}) or {}
            if not isinstance(r3_cfg, dict):
                from dataclasses import asdict, is_dataclass

                r3_cfg = asdict(r3_cfg) if is_dataclass(r3_cfg) else dict(vars(r3_cfg))
            r3_enabled = bool(
                r3_cfg.get("enabled", False) or meg_cfg.get("r3_enabled", False)
            )
            r3_en, r3_mode, r3_gen = r3_fields_from_config(self.config)
            if r3_enabled or r3_en:
                assert_supported_r3(
                    replay_mode=r3_mode,
                    generation_backend=r3_gen
                    or str(
                        get_nested_config(
                            self.config, "policy", "generation_backend", default=""
                        )
                        or ""
                    ),
                )
            return {
                "tensor_model_parallel_size": meg_cfg.get("tensor_model_parallel_size")
                or meg_cfg.get("tensor_parallel_size", 1),
                "pipeline_model_parallel_size": meg_cfg.get(
                    "pipeline_model_parallel_size"
                )
                or meg_cfg.get("pipeline_parallel_size", 1),
                "context_parallel_size": meg_cfg.get("context_parallel_size", 1),
                "expert_model_parallel_size": meg_cfg.get("expert_model_parallel_size")
                or meg_cfg.get("expert_parallel_size", 1),
                "sequence_parallel": meg_cfg.get("sequence_parallel", False),
                "num_experts": meg_cfg.get("num_experts", None),
                "moe_grouped_gemm": meg_cfg.get("moe_grouped_gemm", True),
                "expert_tensor_parallel_size": meg_cfg.get(
                    "expert_tensor_parallel_size", None
                ),
                "moe_router_topk": meg_cfg.get("moe_router_topk", None),
                "moe_router_load_balancing_type": meg_cfg.get(
                    "moe_router_load_balancing_type", "aux_loss"
                ),
                "moe_router_pre_softmax": meg_cfg.get("moe_router_pre_softmax", None),
                "moe_router_score_function": meg_cfg.get(
                    "moe_router_score_function", None
                ),
                "moe_router_dtype": meg_cfg.get("moe_router_dtype", "fp32"),
                "moe_router_topk_scaling_factor": meg_cfg.get(
                    "moe_router_topk_scaling_factor", None
                ),
                "moe_shared_expert_intermediate_size": meg_cfg.get(
                    "moe_shared_expert_intermediate_size", None
                ),
                "moe_aux_loss_coeff": meg_cfg.get("moe_aux_loss_coeff", 0.0),
                "moe_router_bias_update_rate": meg_cfg.get(
                    "moe_router_bias_update_rate", None
                ),
                "moe_token_dispatcher_type": meg_cfg.get(
                    "moe_token_dispatcher_type", "alltoall"
                ),
                "moe_flex_dispatcher_backend": meg_cfg.get(
                    "moe_flex_dispatcher_backend"
                ),
                "moe_mori_max_tokens_per_rank": meg_cfg.get(
                    "moe_mori_max_tokens_per_rank"
                ),
                "moe_mori_kernel_type": meg_cfg.get("moe_mori_kernel_type"),
                "moe_permute_fusion": meg_cfg.get("moe_permute_fusion", False),
                # One key: the Megatron TransformerConfig field name, which is
                # what all three engines read. ``r3_enabled`` was an alias.
                "moe_enable_routing_replay": r3_enabled,
                # FP8 training. Carried as the resolved FP8Config fields rather
                # than raw strings so the engine never parses a recipe name, and
                # so one place decides what an "AMD FP8 recipe" means -- see
                # quantization/fp8_config.megatron_fp8_kwargs. Absent/bf16 leaves
                # every fp8 field unset and the BF16 path byte-identical.
                "fp8_precision": str(
                    get_nested_config(
                        self.config, "quantization", "training", "fp8", default=None
                    )
                    or ""
                ),
                "fp8_recipe": str(
                    get_nested_config(
                        self.config,
                        "quantization",
                        "training",
                        "fp8_recipe",
                        default="blockwise",
                    )
                ),
                "fp8_num_first_layers_in_bf16": _layers_in_bf16(
                    self.config, "num_first_layers_in_bf16"
                ),
                "fp8_num_last_layers_in_bf16": _layers_in_bf16(
                    self.config, "num_last_layers_in_bf16"
                ),
                "param_offload": meg_cfg.get("param_offload", False),
                "optimizer_cpu_offload": meg_cfg.get(
                    "optimizer_cpu_offload", meg_cfg.get("optimizer_offload", False)
                ),
                "optimizer_offload_fraction": meg_cfg.get(
                    "optimizer_offload_fraction", 1.0
                ),
                "use_precision_aware_optimizer": meg_cfg.get(
                    "use_precision_aware_optimizer", False
                ),
                "streamed_optimizer_mode": str(
                    meg_cfg.get("streamed_optimizer_mode", "off")
                ).lower(),
                "streamed_optimizer_chunk_size_mib": (
                    _normalize_streamed_optimizer_chunk_size_mib(
                        meg_cfg.get("streamed_optimizer_chunk_size_mib", 256)
                    )
                ),
                "streamed_optimizer_moment_dtype": meg_cfg.get(
                    "streamed_optimizer_moment_dtype",
                    "fp32",
                ),
                "grad_offload": meg_cfg.get("grad_offload", False),
                "seed": int(policy.get("seed", 42)),
                "dtype": meg_cfg.get("dtype", "bf16"),
                "use_distributed_optimizer": meg_cfg.get(
                    "use_distributed_optimizer", False
                ),
                "grad_reduce_in_fp32": meg_cfg.get("grad_reduce_in_fp32", False),
                "attention_softmax_in_fp32": meg_cfg.get(
                    "attention_softmax_in_fp32", False
                ),
                "recompute_granularity": meg_cfg.get("recompute_granularity", None),
                "recompute_method": meg_cfg.get("recompute_method", None),
                "recompute_num_layers": meg_cfg.get("recompute_num_layers", None),
                "attention_backend": meg_cfg.get("attention_backend", "unfused"),
                "use_packed_sequences": meg_cfg.get("use_packed_sequences", True),
                "log_probs_chunk_size": meg_cfg.get("log_probs_chunk_size", 0),
                "enable_dynamic_batch": meg_cfg.get("enable_dynamic_batch", False),
                "max_tokens_per_gpu": meg_cfg.get("max_tokens_per_gpu", 0),
                "v4_indexer_block_n": meg_cfg.get("v4_indexer_block_n"),
                "v4_indexer_num_stages": meg_cfg.get("v4_indexer_num_stages"),
                "num_layers_in_first_pipeline_stage": meg_cfg.get(
                    "num_layers_in_first_pipeline_stage"
                ),
                "num_layers_in_last_pipeline_stage": meg_cfg.get(
                    "num_layers_in_last_pipeline_stage"
                ),
                # Initial weights from an offline-converted Megatron
                # dist-checkpoint rather than HF safetensors.
                "dist_checkpoint_path": meg_cfg.get("dist_checkpoint_path", None),
                "deterministic_mode": meg_cfg.get("deterministic_mode", None),
                "build_optimizer": meg_cfg.get("build_optimizer", True),
                # Adam step for the offloaded fraction runs on the CPU; see
                # optimizer_cpu_offload above.
                "overlap_cpu_optimizer_d2h_h2d": meg_cfg.get(
                    "overlap_cpu_optimizer_d2h_h2d", True
                ),
            }
        return {}

    def _build_optimizer_config(self, policy: dict) -> dict[str, Any]:
        lr = float(policy.get("learning_rate", policy.get("lr", 1e-6)))
        cfg = {
            # `optimizer_type` is the name this dict is read under everywhere:
            # PolicyConfig sets it, OptimizerConfig declares it, and the FSDP2
            # engine selects on `cfg.optimizer_type`. Megatron's own
            # OptimizerConfig spells it `optimizer`, which is a translation the
            # Megatron adapters do at their own boundary.
            "optimizer_type": str(policy.get("optimizer_type", "adamw")).lower(),
            "lr": lr,
            "weight_decay": float(policy.get("weight_decay", 0.01)),
            "clip_grad": float(policy.get("max_grad_norm", 1.0)),
            "lr_warmup_steps": int(policy.get("lr_warmup_steps", 10)),
            "lr_warmup_steps_ratio": float(policy.get("warmup_ratio", 0.0)),
        }
        for key in ("adam_beta1", "adam_beta2", "adam_eps"):
            if key in policy:
                cfg[key] = float(policy[key])
        if "sgd_momentum" in policy:
            cfg["sgd_momentum"] = float(policy["sgd_momentum"])
        return cfg

    def _build_model_config(self, policy: dict) -> dict[str, Any]:
        return {
            "local_path": str(policy.get("model_name", "")),
            "trust_remote_code": True,
        }

    def _pack_routing_chunk(self, rows, packed):
        """Lay one chunk's R3 rollout routing out in packed token order."""
        if not rows:
            return None, None
        from atomrl.moe.rollout_routing import pack_rollout_routing

        sample = next((r for r in rows if r is not None), None)
        if sample is None:
            return None, None
        return pack_rollout_routing(
            rows,
            packed.seq_lens,
            int(sample.shape[1]),
            int(sample.shape[2]),
            self._device,
        )

    def compute_log_probs(self, batch: DataProto) -> DataProto:
        """Return per-token log-probs (and optionally entropy) for the policy.

        Uses the SAME packed-varlen forward as the training step and as
        ``RLTrainer._compute_log_probs_for_model`` (remove-padding + AITER
        ``flash_attn_varlen_func``), which is the path validated to match verl
        (``use_remove_padding=True``) to 1e-4 on entropy / log-prob / loss.

        The earlier padded left-pad + SDPA forward here produced ~7x inflated
        entropy: left-pad tokens were fed to attention with plain arange
        positions, flattening the logit distribution. Packing removes all pad
        tokens and gives every real token its exact varlen position.
        """
        if self._engine is None:
            raise RuntimeError(
                "init_model() must be called before compute_log_probs()."
            )
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        # Megatron backend owns its own (GPTModel) forward + logprob path.
        if hasattr(self._engine, "engine_compute_log_probs"):
            out = self._engine.engine_compute_log_probs(batch)
            src = getattr(self._engine, "is_mp_src_rank_with_outputs", None)
            if src is not None and not src():
                return DataProto(meta=dict(batch.meta))
            return out

        from atomrl.engine.training.packing import (
            PackingContext,
            pack_sequences,
            packed_token_entropy,
            packed_token_log_probs,
            unpack_log_probs,
        )
        from atomrl.moe.rollout_routing import RoutingReplayContext

        sequences = batch["input_ids"].to(self._device)
        B, S = sequences.shape
        am = batch.tensors.get("attention_mask")
        if am is not None:
            am = am.to(self._device)
        else:
            # No mask supplied -> treat every token as real. Only safe for
            # already-unpadded input; the Ray controller always sends the mask.
            am = torch.ones_like(sequences)

        want_entropy = bool(batch.meta.get("calculate_entropy", False))
        # verl divides logits by the sampling temperature before log_prob/entropy
        # (logits.div_(temperature)); apply the same so old/train/ref stay unbiased.
        temperature = float(batch.meta.get("temperature", 1.0) or 1.0)

        # Pack as many sequences per forward as the token budget allows. This
        # relies on the varlen packing attention wrapper for cross-sequence
        # isolation, which is now installed unconditionally (see
        # fsdp_backend._install_packing_attention); one row per forward was the
        # fallback for when that wrapper was missing.
        import torch.distributed as dist

        max_tok = self._max_token_len_per_gpu()
        row_lens = am.sum(dim=1).long().tolist()
        chunks: list[tuple[int, int]] = []
        if max_tok > 0:
            start = 0
            while start < B:
                tok, end = 0, start
                while end < B:
                    sl = int(row_lens[end])
                    if tok + sl > max_tok and end > start:
                        break
                    tok += sl
                    end += 1
                chunks.append((start, end))
                start = end
        else:
            chunks = [(r, r + 1) for r in range(B)]

        n_rows = len(chunks)
        if dist.is_initialized() and dist.get_world_size() > 1:
            # FSDP2 all-gathers per forward must run in lockstep across ranks;
            # equalize the number of forwards (pad by repeating the last chunk).
            cnt = torch.tensor([n_rows], device=self._device)
            dist.all_reduce(cnt, op=dist.ReduceOp.MAX)
            n_iters = int(cnt.item())
        else:
            n_iters = n_rows

        lp_parts: list[torch.Tensor] = []
        ent_parts: list[torch.Tensor] = []
        module = self._engine.module
        # R3: old_log_probs must be produced under the SAME routing as the
        # training forward, or the PPO ratio compares two different policies and
        # ppo_kl picks up routing noise instead of the optimizer step.
        routing_rows = batch.ragged.get("rollout_routing")

        with self._engine.eval_mode(), torch.no_grad():
            for i in range(n_iters):
                lo, hi = chunks[min(i, n_rows - 1)]
                ids_row = sequences[lo:hi]
                mask_row = am[lo:hi]
                # FSDP2 shards parameters, not the token-packing dimension.
                _tp = 1
                packed = pack_sequences(ids_row, mask_row, tp_align=_tp)
                r3_idx, r3_valid = self._pack_routing_chunk(
                    routing_rows[lo:hi] if routing_rows else None,
                    packed,
                )
                with (
                    PackingContext(packed.cu_seqlens, packed.max_seqlen),
                    RoutingReplayContext(r3_idx, r3_valid),
                ):
                    outputs = module(
                        input_ids=packed.input_ids,
                        position_ids=packed.position_ids,
                        attention_mask=None,
                        use_cache=False,
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    logits = logits.squeeze(0)
                    flat_lp = packed_token_log_probs(
                        logits,
                        packed.input_ids.squeeze(0),
                        packed.cu_seqlens,
                        temperature=temperature,
                    )
                    token_lp = unpack_log_probs(
                        flat_lp,
                        packed.cu_seqlens,
                        packed.seq_lens,
                        S,
                    )
                    if want_entropy:
                        flat_ent = packed_token_entropy(
                            logits,
                            packed.cu_seqlens,
                            temperature=temperature,
                            upcast=True,
                        )
                        token_ent = unpack_log_probs(
                            flat_ent,
                            packed.cu_seqlens,
                            packed.seq_lens,
                            S,
                        )
                    if i < n_rows:
                        lp_parts.append(token_lp)
                        if want_entropy:
                            ent_parts.append(token_ent)
                    del outputs, logits

        log_probs = torch.cat(lp_parts, dim=0)
        tensors = {"log_probs": log_probs.cpu(), "input_ids": batch["input_ids"]}
        if want_entropy and ent_parts:
            tensors["entropy"] = torch.cat(ent_parts, dim=0).cpu()
        out = DataProto(tensors=tensors, meta=dict(batch.meta))
        return out

    def _max_token_len_per_gpu(self) -> int:
        """Per-micro-batch token budget for the packed forward (0 if unset)."""
        policy = get_nested_config(self.config, "policy", default={}) or {}
        try:
            return int(policy.get("max_token_len_per_gpu", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def update_policy(self, batch: DataProto) -> dict[str, float]:
        """verl-aligned worker-side PPO update over this actor's DP shard.

        The controller dispatches the full advantage-augmented batch across all
        actors (each gets ``global / num_workers`` rows) with ``batch_num_tokens``
        (GLOBAL response-token count) and ``dp_size`` in meta. This runs ONE
        optimizer step; the engine micro-batches internally (grad accumulation),
        deferring the FSDP2 reduce-scatter to the last micro. The per-token PG
        loss is normalized by the global token count and dp_size-compensated, so
        summing shard gradients (FSDP averages across ranks) yields the correct
        global token-mean -- matching the validated torchrun path.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before update_policy().")
        if batch.batch_size == 0:
            return {"loss": 0.0}

        # Megatron backend owns its own (GPTModel) forward-backward + DAPO loss.
        if hasattr(self._engine, "engine_update_policy"):
            return self._engine.engine_update_policy(batch)

        data: dict[str, Any] = {k: v for k, v in batch.tensors.items()}
        # per-row ragged payloads (R3 rollout routing); the engine slices these
        # with the rows when it forms micro-batches.
        data.update(batch.ragged)
        # verl-aligned packed (remove-padding + varlen) training forward: the
        # engine packs each micro, dropping all pad tokens. This is the forward
        # that matches verl (use_remove_padding=True) and keeps old_log_probs
        # (also packed) consistent with the train forward (ratio ~= 1). The old
        # padded SDPA path flattened logits (AITER attn does not mask left-pad),
        # inflating entropy ~7x.
        data["use_packed_forward"] = True
        data["temperature"] = float(batch.meta.get("temperature", 1.0) or 1.0)
        # Pack micros to the same token budget verl uses for
        # ppo_max_token_len_per_gpu. Falls back to one row per micro only if the
        # budget is unset.
        max_tok = self._max_token_len_per_gpu()
        if max_tok > 0:
            data["max_token_len_per_gpu"] = max_tok
        else:
            data["micro_batch_size"] = 1
        self._pol_meta = dict(batch.meta)
        if not getattr(self, "_logged_norm", False):
            self._logged_norm = True
            self._log.info(
                "update_policy norm: batch_num_tokens=%s dp_size=%s rows=%d has_ris=%s",
                self._pol_meta.get("batch_num_tokens"),
                self._pol_meta.get("dp_size"),
                batch.batch_size,
                "rollout_is_weights" in batch.tensors,
            )

        with self._engine.train_mode():
            output = self._engine.train_batch(data, self._policy_loss_fn)

        metrics: dict[str, float] = {}
        # Same normalization either way, applied at a different point: the
        # diagnostics are normalized by the controller, so reducing them here
        # cancels; ``loss`` is already normalized per unit because backward()
        # needs it there, which is why it is summed rather than averaged.
        for k, v in output.get("metrics", {}).items():
            metrics[k] = (sum(v) / len(v)) if isinstance(v, list) and v else float(v)
        if "loss" in output:
            lv = output["loss"]
            if self._pol_meta.get("task_type") == "sft":
                # sft_loss normalizes by each micro's own token count, so a unit
                # is already a mean and summing them would scale by the count.
                metrics["loss"] = (
                    (sum(lv) / len(lv)) if isinstance(lv, list) and lv else float(lv)
                )
            else:
                metrics["loss"] = reduce_reported_loss(lv)
        metrics["lr"] = self._engine.lr_scheduler_step()
        return metrics

    def reset_memory_stats(self) -> bool:
        """Reset per-step CUDA/HIP peak memory counters for this actor rank."""
        self._peak_carry_alloc = 0.0
        self._peak_carry_reserved = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return True

    def free_cached_memory(self) -> dict[str, float]:
        """Give the allocator's unused cached segments back to the driver.

        FSDP2 sharding frees the pre-shard full-precision weights, but they stay
        in the caching allocator until the first micro-batch calls
        ``empty_cache()``. A colocated rollout engine that budgets its KV cache
        from ``mem_get_info`` sees that cache as occupied, so it has to be
        returned before the engine is created.
        """
        if not torch.cuda.is_available():
            return {"reserved_before_bytes": 0.0, "reserved_after_bytes": 0.0}
        before = float(torch.cuda.memory_reserved())
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "reserved_before_bytes": before,
            "reserved_after_bytes": float(torch.cuda.memory_reserved()),
        }

    def get_memory_stats(self) -> dict[str, float]:
        """Return current-step peak memory counters for this actor rank."""
        # Linux reports ru_maxrss in KiB. Keep this alongside HIP memory so the
        # controller can emit the same actor/perf memory family as verl.
        cpu_memory_used_bytes = (
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0
        )
        if not torch.cuda.is_available():
            return {
                "max_reserved_bytes": 0.0,
                "max_allocated_bytes": 0.0,
                "reserved_bytes": 0.0,
                "allocated_bytes": 0.0,
                "cpu_memory_used_bytes": cpu_memory_used_bytes,
            }
        return {
            "max_reserved_bytes": max(
                self._peak_carry_reserved,
                float(torch.cuda.max_memory_reserved()),
            ),
            "max_allocated_bytes": max(
                self._peak_carry_alloc,
                float(torch.cuda.max_memory_allocated()),
            ),
            "reserved_bytes": float(torch.cuda.memory_reserved()),
            "allocated_bytes": float(torch.cuda.memory_allocated()),
            "cpu_memory_used_bytes": cpu_memory_used_bytes,
        }

    def _policy_loss_fn(self, model_output, data):
        """DAPO/GRPO surrogate on ONE micro, using the micro's own tensors.

        Reads old_log_probs/advantages/response_mask/rollout_is_weights from the
        micro (``data``) -- NOT a closed-over full batch -- so shapes always
        match ``model_output``. Applies dual-clip + TIS + global-token
        normalization (batch_num_tokens / dp_size from meta), matching verl.
        """
        token_log_probs = model_output["log_probs"]
        L = token_log_probs.shape[-1]
        dev = token_log_probs.device
        meta = getattr(self, "_pol_meta", {}) or {}

        # SFT dispatch: when task_type is "sft", use supervised loss instead of RL.
        if meta.get("task_type") == "sft":
            lm = data.get("loss_mask")
            if lm is None:
                lm = data.get("attention_mask")
            lm = lm.to(dev).float()
            # Shift loss_mask to align with log_probs (which predict token i+1).
            if lm.shape[-1] == L + 1:
                lm = lm[:, 1:]
            lm = lm[..., :L]
            dp = int(meta.get("dp_size", 1) or 1)
            dp_group = None
            if self._engine is not None:
                dp_group = self._engine.get_data_parallel_group()
            return sft_loss(
                token_log_probs,
                lm,
                loss_agg_mode=meta.get("loss_agg_mode", "token-mean"),
                dp_size=dp,
                dp_group=dp_group,
            )

        algo_name = str(meta.get("algorithm", "dapo")).lower()

        old_logp = data.get("old_log_probs")
        adv = data.get("advantages")
        if old_logp is None or adv is None:
            shift_labels = data["input_ids"][:, 1:].contiguous().to(dev)
            loss = F.cross_entropy(
                (
                    token_log_probs.reshape(-1, token_log_probs.shape[-1])
                    if token_log_probs.dim() == 3
                    else token_log_probs.reshape(-1)
                ),
                shift_labels.reshape(-1),
            )
            return loss, {"loss": float(loss.detach())}

        # response mask aligned to [., L]
        resp_mask = data.get("response_mask")
        if resp_mask is not None:
            mask = resp_mask.to(dev).float()
            if mask.shape[-1] == L + 1:
                mask = mask[:, 1:]
            mask = mask[..., :L]
        else:
            am = data.get("attention_mask")
            mask = am.to(dev)[..., :L].float() if am is not None else None

        old_logp = old_logp.to(dev)[..., :L]
        adv = adv.to(dev)
        if adv.dim() == 1:
            adv = adv.unsqueeze(-1).expand_as(token_log_probs)
        elif adv.shape[-1] != L:
            adv = adv[..., :L]

        algo_cfg_full = meta.get("algo_config", {}) or {}
        _sub = algo_cfg_full.get(algo_name)
        _sub = _sub if isinstance(_sub, dict) else {}

        def _cfg(key, default):
            v = _sub.get(key, algo_cfg_full.get(key, default))
            return default if v is None else v

        bnt = meta.get("batch_num_tokens")
        dp = int(meta.get("dp_size", 1) or 1)
        loss_agg_mode = str(_cfg("loss_agg_mode", "token-mean"))
        global_batch_size = int(
            meta.get("global_batch_size") or token_log_probs.shape[0] * dp
        )
        ris = data.get("rollout_is_weights")
        if ris is not None:
            ris = ris.to(dev)
            if ris.dim() > 1 and ris.shape[-1] != L:
                ris = ris[..., :L]

        if algo_name == AlgorithmName.DAPO.value:
            loss = asymmetric_clip_loss(
                token_log_probs,
                old_logp,
                adv,
                float(_cfg("clip_ratio_low", 0.2)),
                float(_cfg("clip_ratio_high", 0.28)),
                mask=mask,
                clip_ratio_c=float(_cfg("clip_ratio_c", 0.0)),
                batch_num_tokens=bnt,
                dp_size=dp,
                rollout_is_weights=ris,
            )
        elif algo_name == AlgorithmName.GRPO.value:
            loss = asymmetric_clip_loss(
                token_log_probs,
                old_logp,
                adv,
                float(_cfg("clip_ratio", 0.2)),
                float(_cfg("clip_ratio_high", 0.28)),
                mask=mask,
                batch_num_tokens=bnt,
                dp_size=dp,
                loss_agg_mode=loss_agg_mode,
                global_batch_size=global_batch_size,
                rollout_is_weights=ris,
            )
        else:
            loss = policy_gradient_loss(
                token_log_probs,
                old_logp,
                adv,
                float(_cfg("clip_ratio", 0.2)),
                mask=mask,
            )

        kl_c = float(_cfg("kl_coeff", 0.0))
        ref_logp = data.get("ref_log_probs")
        if kl_c > 0.0 and ref_logp is not None:
            ref_logp = ref_logp.to(dev)[..., :L]
            loss = loss + kl_c * kl_penalty(token_log_probs, ref_logp, mask=mask)

        # KL diagnostics (verl-aligned, detached, over response tokens). Return
        # per-micro SUM + token COUNT (not a per-sequence mean): the controller
        # aggregates as a GLOBAL token-weighted mean (verl masked_mean over the
        # whole batch). A mean-of-per-sequence-means would over-weight short/
        # outlier sequences and make core/kl look much noisier than verl.
        #   ppo_kl          = Σ(old_logp - new_logp) / Σtok  -> verl actor/ppo_kl
        #   rollout_corr/kl = Σ(rollout_logp - new_logp) / Σtok -> verl rollout_corr/kl
        out_metrics = {"loss": float(loss.detach())}
        if mask is not None:
            with torch.no_grad():
                tok = float(mask.sum())
                out_metrics["ppo_kl_sum"] = float(
                    ((old_logp - token_log_probs) * mask).sum()
                )
                out_metrics["ppo_kl_tok"] = tok
                rlp = data.get("rollout_log_probs")
                if rlp is not None:
                    rlp = rlp.to(dev)[..., :L]
                    delta = (rlp - token_log_probs) * mask
                    out_metrics["rollout_corr_kl_sum"] = float(delta.sum())
                    out_metrics["rollout_corr_kl_tok"] = tok
                    out_metrics.update(self._mismatch_metrics(delta, mask, tok))
        return loss, out_metrics

    @staticmethod
    def _mismatch_metrics(delta, mask, tok: float) -> dict[str, float]:
        """Train/rollout mismatch diagnostics beyond the signed mean.

        ``rollout_corr/kl`` is a SIGNED mean, so symmetric per-token disagreement
        cancels inside it. On a routed-expert model it hid an order of magnitude:
        measured mean |delta| 0.0226 behind a signed mean of 0.0019, because
        train/rollout expert-selection flips are large but unbiased. These are the
        quantities that do not cancel.

        Everything is returned as a SUM plus its denominator so the controller can
        form a global token-weighted (or sequence-weighted) mean; per-micro and
        per-worker averaging of both parts leaves the ratio exact.
        """
        # exp() of a log-ratio: clamp so a pathological sequence cannot turn the
        # whole metric into inf and hide the rest.
        w = torch.exp(delta.clamp(-10.0, 10.0))
        seq_delta = delta.sum(dim=-1).clamp(-10.0, 10.0)
        w_seq = torch.exp(seq_delta)
        n_seq = float(delta.shape[0])
        return {
            # mean |delta|: the magnitude that the signed mean cancels away
            "mismatch_abs_sum": float(delta.abs().sum()),
            # k3 estimator E[e^r - r - 1] >= 0, unbiased and low-variance for small KL
            "mismatch_k3_sum": float((((w - 1.0) - delta) * mask).sum()),
            # second moment of the token-level IS weight: predicts gradient variance
            "mismatch_chi2_tok_sum": float((((w - 1.0) ** 2) * mask).sum()),
            # same at sequence level, where TIS actually clips
            "mismatch_chi2_seq_sum": float(((w_seq - 1.0) ** 2).sum()),
            "mismatch_seq_n": n_seq,
            # tail mass: the population driving the divergence (expert flips)
            "mismatch_tail_sum": float(((delta.abs() > 0.1).float() * mask).sum()),
            "mismatch_tok": tok,
        }

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU full state dict for weight sync to rollout engines.

        FSDP2 shards parameters as ``DTensor`` across the actor process group,
        so ``full_tensor()`` all-gathers the unsharded tensor. This is a
        COLLECTIVE op: every actor must call ``get_state_dict`` concurrently or
        the all-gather deadlocks (see ``_fetch_actor_cpu_state``).
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before get_state_dict().")
        from torch.distributed.tensor import DTensor

        params, _ = self._engine.get_per_tensor_param()
        out: dict[str, torch.Tensor] = {}
        for name, param in params:
            # Megatron collectives inside get_per_tensor_param() must run on
            # every rank, but returning a complete CPU state from every Ray
            # actor multiplies host/object-store usage by world size.  The
            # controller only consumes states[0], so nonzero ranks participate
            # in the generator collectives and discard the resulting tensors.
            if self.rank != 0:
                continue
            full = param.full_tensor() if isinstance(param, DTensor) else param
            out[name] = full.detach().cpu()
        return out

    def get_rdma_rendezvous(self, interface: str) -> dict[str, Any]:
        """Return an interface address and free TCPStore port on this actor."""
        import fcntl
        import socket
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packed = struct.pack("256s", interface[:15].encode("utf-8"))
            address = socket.inet_ntoa(
                fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24]
            )
        finally:
            sock.close()
        return {"address": address, "port": self.find_free_port()}

    def rdma_preflight(self, interface: str, hca: str) -> dict[str, Any]:
        from pathlib import Path

        uverbs = sorted(Path("/dev/infiniband").glob("uverbs*"))
        hca_path = Path("/sys/class/infiniband") / hca
        if not uverbs or not hca_path.exists():
            raise RuntimeError(
                f"RDMA unavailable in actor container: uverbs={uverbs}, hca={hca}"
            )
        rendezvous = self.get_rdma_rendezvous(interface)
        return {
            "interface": interface,
            "hca": hca,
            "uverbs": len(uverbs),
            "address": rendezvous["address"],
        }

    def init_rdma_weight_group(
        self,
        master_addr: str,
        master_port: int,
        world_size: int,
        group_name: str,
        timeout_s: int = 600,
    ) -> bool:
        """Join the independent RCCL weight group as source rank zero."""
        from datetime import timedelta

        from atomrl.utils.independent_process_group import (
            init_independent_process_group,
        )

        if self.rank != 0:
            raise RuntimeError("only actor rank zero may source RDMA weights")
        existing = getattr(self, "_rdma_weight_group", None)
        if existing is not None:
            return True
        self._rdma_weight_group = init_independent_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=timedelta(seconds=int(timeout_s)),
            world_size=int(world_size),
            rank=0,
            group_name=group_name,
        )
        self._rdma_weight_group_name = group_name
        logger.info(
            "Actor rank 0 joined RDMA weight group %s at %s:%d (world=%d)",
            group_name,
            master_addr,
            master_port,
            world_size,
        )
        return True

    def send_weights_rdma(
        self,
        version: int,
        bucket_size_mb: int,
        fp8_quantize: bool = False,
        integrity_check: bool = False,
    ) -> dict[str, Any]:
        """Collect Megatron HF tensors on all ranks; rank zero broadcasts them.

        When ``fp8_quantize=True``, BF16 weights are quantized to FP8 e4m3 with
        per-128x128-block scaling on the actor side before RDMA broadcast. This
        halves transfer size and is compatible with vLLM's ``fp8_per_block`` mode.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before RDMA weight sync")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        params, _ = self._engine.get_per_tensor_param()
        params = _whole_tensors(params)
        if self.rank != 0:
            # Consume the conversion iterator on every rank: Megatron's TP/EP
            # collectives and FSDP2's all-gathers run as it advances.
            for _ in params:
                pass
            return {"writer": False}

        group = getattr(self, "_rdma_weight_group", None)
        if group is None:
            raise RuntimeError("RDMA weight group is not initialized")

        if os.environ.get("ATOMRL_SYNC_FP32") != "1":
            # The same FP32 -> BF16 cast as update_weights_ipc_send.
            params = (
                (name, t.to(torch.bfloat16) if t.dtype == torch.float32 else t)
                for name, t in params
            )

        integrity_enabled = (
            bool(integrity_check)
            or os.environ.get("ATOMRL_WEIGHT_SYNC_INTEGRITY", "0") == "1"
        )
        if integrity_enabled:
            from atomrl.engine.inference.weight_integrity import (
                require_finite_stream,
            )

            params = require_finite_stream(params, stage="export")

        if fp8_quantize:
            from atomrl.engine.inference.fp8_weight_quantizer import (
                quantize_weights_fp8_per_block,
            )

            params = quantize_weights_fp8_per_block(
                params,
                pow2_scales=bool(
                    get_nested_config(
                        self.config,
                        "quantization",
                        "rollout",
                        "use_weight_pow2_scale",
                        default=False,
                    )
                ),
            )
            if integrity_enabled:
                params = require_finite_stream(params, stage="fp8_quantize")

        from atomrl.engine.inference.rdma_weight_transfer import send_weight_stream

        stats = send_weight_stream(
            group,
            params,
            bucket_size_bytes=int(bucket_size_mb) * 1024 * 1024,
            version=int(version),
        )
        stats["writer"] = True
        stats["fp8_quantized"] = fp8_quantize
        logger.info("RDMA weight broadcast complete: %s", stats)
        return stats

    def destroy_rdma_weight_group(self) -> bool:
        group = getattr(self, "_rdma_weight_group", None)
        if group is not None:
            import torch.distributed as dist

            dist.destroy_process_group(group)
            self._rdma_weight_group = None
        return True

    def export_state_dict_safetensors(
        self,
        sync_dir: str,
        max_shard_bytes: int = 4 * 1024 * 1024 * 1024,
        include_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Collectively export HF weights without returning them through Ray.

        Every rank consumes the parameter generator so Megatron TP/EP
        collectives make progress.  Rank 0 streams bounded CPU shards directly
        to the shared filesystem; other ranks discard each gathered tensor.
        ``include_names`` limits files written by rank 0 while preserving all
        collective generator calls on every rank.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before exporting weights.")

        import json

        from safetensors.torch import save_file

        output_dir = Path(sync_dir)
        is_writer = self.rank == 0
        if is_writer:
            output_dir.mkdir(parents=True, exist_ok=True)
            for old in output_dir.glob("model-*.safetensors"):
                old.unlink()

        params, _ = self._engine.get_per_tensor_param()
        params = _whole_tensors(params)
        shard: dict[str, torch.Tensor] = {}
        shard_keys: list[list[str]] = []
        temp_paths: list[Path] = []
        current_bytes = 0
        total_bytes = 0
        num_params = 0
        selected_names = set(include_names) if include_names is not None else None

        def flush_shard() -> None:
            nonlocal shard, current_bytes
            if not shard:
                return
            idx = len(temp_paths) + 1
            path = output_dir / f"model-{idx:05d}.safetensors.tmp"
            save_file(shard, str(path))
            temp_paths.append(path)
            shard_keys.append(list(shard))
            shard = {}
            current_bytes = 0

        for name, param in params:
            if not is_writer:
                continue
            if selected_names is not None and name not in selected_names:
                continue
            tensor = param.detach().to("cpu").contiguous()
            tensor_bytes = tensor.numel() * tensor.element_size()
            if shard and current_bytes + tensor_bytes > max_shard_bytes:
                flush_shard()
            shard[name] = tensor
            current_bytes += tensor_bytes
            total_bytes += tensor_bytes
            num_params += 1
        if is_writer:
            flush_shard()

            weight_map: dict[str, str] = {}
            shard_count = len(temp_paths)
            for idx, (temp_path, keys) in enumerate(zip(temp_paths, shard_keys), 1):
                filename = f"model-{idx:05d}-of-{shard_count:05d}.safetensors"
                temp_path.replace(output_dir / filename)
                for key in keys:
                    weight_map[key] = filename

            index = {
                "metadata": {"total_size": total_bytes},
                "weight_map": weight_map,
            }
            (output_dir / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2)
            )

        weight_url = self._ensure_weight_http_server(output_dir) if is_writer else ""
        return {
            "writer": is_writer,
            "num_params": num_params,
            "total_bytes": total_bytes,
            "num_shards": len(temp_paths),
            "weight_url": weight_url,
        }

    def _ensure_weight_http_server(self, output_dir: Path) -> str:
        """Expose exported shards to rollout nodes without assuming shared storage."""
        server = getattr(self, "_weight_http_server", None)
        server_root = getattr(self, "_weight_http_root", None)
        if server is None or server_root != output_dir:
            import functools
            import http.server
            import threading

            class _QuietHandler(http.server.SimpleHTTPRequestHandler):
                def log_message(self, _format, *_args) -> None:
                    return

            handler = functools.partial(_QuietHandler, directory=str(output_dir))
            server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), handler)
            thread = threading.Thread(
                target=server.serve_forever,
                name="lumen-weight-http",
                daemon=True,
            )
            thread.start()
            self._weight_http_server = server
            self._weight_http_root = output_dir
            self._weight_http_thread = thread

        import ray

        host = ray.util.get_node_ip_address()
        return f"http://{host}:{server.server_port}"

    def save_checkpoint(self, local_path: str, global_step: int = 0) -> bool:
        """Save this actor rank's sharded training state.

        This mirrors verl's Ray worker checkpoint contract: every actor writes
        its own model/optimizer/scheduler shard under the same step directory.
        """
        if self._engine is None:
            raise RuntimeError("init_model() must be called before save_checkpoint().")
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)
        # Native Megatron engine: use sharded dist-checkpoint (no DP duplication,
        # resharding-capable) instead of per-rank torch.save of full weights.
        use_rank_local = (
            os.environ.get("ATOMRL_CHECKPOINT_FORMAT", "").lower() == "rank_local"
        )
        if not use_rank_local and hasattr(self._engine, "save_dist_checkpoint"):
            self._engine.save_dist_checkpoint(str(path), global_step=global_step)
            return True
        rank = int(self.rank)
        world = int(self.world_size)
        module = getattr(self._engine, "module", None)
        optimizer = getattr(self._engine, "optimizer", None)
        scheduler = getattr(self._engine, "lr_scheduler", None)
        if module is None:
            raise RuntimeError("Engine has no module to checkpoint.")

        distributed = torch.distributed.is_initialized()
        for active_ranks in checkpoint_rank_phases(rank, world):
            if rank in active_ranks:
                torch.save(
                    module.state_dict(),
                    path / f"model_world_size_{world}_rank_{rank}.pt",
                )
                if optimizer is not None:
                    torch.save(
                        optimizer.state_dict(),
                        path / f"optim_world_size_{world}_rank_{rank}.pt",
                    )
                    if hasattr(optimizer, "save_parameter_state"):
                        optimizer.save_parameter_state(
                            str(
                                path
                                / f"optim_parameter_state_world_size_{world}_rank_{rank}.pt"
                            )
                        )
                extra = {
                    "global_step": int(global_step),
                    "lr_scheduler": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    "rng": {
                        "cpu": torch.get_rng_state(),
                        "cuda": (
                            torch.cuda.get_rng_state()
                            if torch.cuda.is_available()
                            else None
                        ),
                    },
                }
                torch.save(
                    extra,
                    path / f"extra_state_world_size_{world}_rank_{rank}.pt",
                )
                del extra
                import gc

                gc.collect()
                try:
                    import ctypes

                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except (OSError, AttributeError):
                    pass
            if distributed:
                torch.distributed.barrier()
        return True

    def prune_checkpoints(self, checkpoint_root: str, max_to_keep: int) -> list[str]:
        """Prune actor shards on the node-local filesystem that owns them."""
        if int(self.rank) != 0:
            return []
        root = Path(checkpoint_root)
        if not root.is_dir():
            return []
        pattern = re.compile(r"global_step_(\d+)$")
        checkpoints: list[tuple[int, Path]] = []
        for child in root.iterdir():
            match = pattern.match(child.name)
            if child.is_dir() and match:
                checkpoints.append((int(match.group(1)), child))
        checkpoints.sort(key=lambda item: item[0])
        removed: list[str] = []
        while len(checkpoints) > max(0, int(max_to_keep)):
            _, old = checkpoints.pop(0)
            shutil.rmtree(old)
            removed.append(str(old))
            logger.info("Pruned actor-node checkpoint: %s", old)
        return removed

    def load_checkpoint(self, local_path: str) -> int:
        """Load this actor rank's sharded training state and return global_step."""
        if self._engine is None:
            raise RuntimeError("init_model() must be called before load_checkpoint().")
        path = Path(local_path)
        use_rank_local = (
            os.environ.get("ATOMRL_CHECKPOINT_FORMAT", "").lower() == "rank_local"
        )
        if not use_rank_local and hasattr(self._engine, "load_dist_checkpoint"):
            return self._engine.load_dist_checkpoint(str(path))
        rank = int(self.rank)
        world = int(self.world_size)
        module = getattr(self._engine, "module", None)
        optimizer = getattr(self._engine, "optimizer", None)
        scheduler = getattr(self._engine, "lr_scheduler", None)
        if module is None:
            raise RuntimeError("Engine has no module to restore.")

        model_path = path / f"model_world_size_{world}_rank_{rank}.pt"
        optim_path = path / f"optim_world_size_{world}_rank_{rank}.pt"
        optim_parameter_path = (
            path / f"optim_parameter_state_world_size_{world}_rank_{rank}.pt"
        )
        extra_path = path / f"extra_state_world_size_{world}_rank_{rank}.pt"
        global_step = 0

        def _load_rank_state() -> None:
            nonlocal global_step
            if not model_path.is_file():
                raise FileNotFoundError(f"Missing actor checkpoint shard: {model_path}")
            module_state = torch.load(
                model_path, map_location="cpu", weights_only=False
            )
            module.load_state_dict(module_state)
            del module_state
            has_parameter_state = optim_parameter_path.is_file()
            if optimizer is not None and optim_path.is_file():
                optim_state = torch.load(
                    optim_path, map_location="cpu", weights_only=False
                )
                patch_hybrid_load = getattr(
                    self._engine,
                    "_patch_hybrid_optimizer_checkpoint_load",
                    None,
                )
                if callable(patch_hybrid_load):
                    patch_hybrid_load()
                optimizer.load_state_dict(optim_state)
                del optim_state
            if optimizer is not None and hasattr(optimizer, "load_parameter_state"):
                if not has_parameter_state:
                    raise FileNotFoundError(
                        "Missing distributed optimizer parameter state: "
                        f"{optim_parameter_path}"
                    )
                optimizer.load_parameter_state(str(optim_parameter_path))
            elif optimizer is not None and hasattr(optimizer, "reload_model_params"):
                optimizer.reload_model_params()
            if extra_path.is_file():
                extra = torch.load(extra_path, map_location="cpu", weights_only=False)
                global_step = int(extra.get("global_step", 0))
                sched_state = extra.get("lr_scheduler")
                if scheduler is not None and sched_state is not None:
                    scheduler.load_state_dict(sched_state)
                rng = extra.get("rng") or {}
                if rng.get("cpu") is not None:
                    torch.set_rng_state(rng["cpu"])
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state(rng["cuda"])
                del extra
            import gc

            gc.collect()
            try:
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass

        checkpoint_group = create_checkpoint_control_group(world)
        try:
            for active_ranks in checkpoint_rank_phases(
                rank,
                world,
                group=checkpoint_group,
            ):
                run_checkpoint_phase(
                    rank,
                    world,
                    _load_rank_state if rank in active_ranks else None,
                    group=checkpoint_group,
                )
        finally:
            if checkpoint_group is not None:
                torch.distributed.destroy_process_group(checkpoint_group)
        return global_step

    @staticmethod
    def _sent_nbytes(param: torch.Tensor, keep_fp32: bool) -> int:
        """Bytes ``update_weights_ipc_send`` puts on the wire for one parameter.

        Mirrors the FP32 -> BF16 cast below; ``numel()`` on a DTensor is already
        the global count, which is what the receiver sees.
        """
        itemsize = param.element_size()
        if param.dtype == torch.float32 and not keep_fp32:
            itemsize = torch.bfloat16.itemsize
        return itemsize * param.numel()

    def update_weights_ipc_send(
        self,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
        version: int | None = None,
        fp8_quantize: bool = False,
    ) -> dict[str, float]:
        """Stream full (all-gathered) weights to the colocated vLLM replica.

        With ``fp8_quantize=True`` the weights are cast to FP8 e4m3 with
        per-128x128-block scales before they go on the wire, the same as
        ``send_weights_rdma`` does for the separated topology: roughly 42% fewer
        bytes, and the rollout loads them directly instead of re-quantizing BF16.
        Requires a vLLM ``fp8_per_block`` model on the receiving end, which
        ``WeightSyncConfig.validate_fp8_quantization_location`` enforces.


        verl-aligned ZMQ CUDA-IPC weight sync. ``full_tensor()`` is an FSDP
        all-gather COLLECTIVE, so every actor must call this concurrently (the
        trainer dispatches it to all actors at once).

        Socket keys must match the receiver's ``_get_zmq_handle``, i.e.
        ``{job_id, replica_rank, local_rank}``. With a TP=N rollout engine per N
        actors, actor ``rank`` owns TP rank ``rank % N`` of replica ``rank // N``
        -- the same order VLLMReplicaManager used to build the engine's
        ``CUDA_VISIBLE_DEVICES``, so each actor talks to the worker sitting on
        its own GPU. Every worker receives the full tensors; vLLM's weight
        loaders take their own slice.

        Returns a dict of sync-only timing and peak-memory metrics. The trainer
        aggregates the max across ranks.
        """
        if self._engine is None:
            raise RuntimeError(
                "init_model() must be called before update_weights_ipc_send()."
            )

        import asyncio

        import ray
        from torch.distributed.tensor import DTensor

        from atomrl.engine.inference.bucketed_weight_transfer import (
            BucketedWeightSender,
        )

        job_id = ray.get_runtime_context().get_job_id()
        rollout_tp = max(
            1,
            int(
                get_nested_config(
                    self.config,
                    "policy",
                    "generation",
                    "vllm_cfg",
                    "tensor_parallel_size",
                    default=1,
                )
                or 1
            ),
        )
        replica_rank = self.rank // rollout_tp
        local_rank = self.rank % rollout_tp
        handle = (
            f"ipc:///tmp/lumen-colocate-zmq-{job_id}"
            f"-replica-{replica_rank}-rank-{local_rank}.sock"
        )
        keep_fp32 = os.environ.get("ATOMRL_SYNC_FP32") == "1"

        gb = 1024.0**3
        alloc0 = 0.0
        if torch.cuda.is_available():
            self._peak_carry_alloc = max(
                self._peak_carry_alloc,
                float(torch.cuda.max_memory_allocated()),
            )
            self._peak_carry_reserved = max(
                self._peak_carry_reserved,
                float(torch.cuda.max_memory_reserved()),
            )
            alloc0 = float(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()

        is_atom = (
            str(
                get_nested_config(
                    self.config,
                    "policy",
                    "generation_backend",
                    default="",
                )
                or ""
            )
            == "atom"
        )
        # Expert stacks load as one copy per stack instead of vLLM's per-expert
        # loader; only vLLM's receiver understands the stacked names.
        stack_experts = (
            not is_atom
            and os.environ.get("ATOMRL_WEIGHT_SYNC_STACK_EXPERTS", "1") != "0"
        )
        params, _ = self._engine.get_per_tensor_param(stack_experts=stack_experts)
        gather_s = 0.0

        def _gen():
            nonlocal gather_s
            it = iter(params)
            while True:
                t0 = time.perf_counter()
                try:
                    name, param = next(it)
                except StopIteration:
                    break
                full = param.full_tensor() if isinstance(param, DTensor) else param
                full = full.detach()
                if full.dtype == torch.float32 and not keep_fp32:
                    full = full.to(torch.bfloat16)
                if not full.is_cuda:
                    full = full.to("cuda", non_blocking=True)
                gather_s += time.perf_counter() - t0
                yield name, full

        # ATOM's ModelRunner opens the staging buffer once per update cycle and
        # holds that mapping until the last bucket, so the bucket must be able to
        # hold the largest single tensor -- growing it mid-cycle would leave the
        # runner reading through a stale, shorter mapping. Qwen3-30B-A3B's fused
        # expert weight is 768 MiB, past the 512 MiB default. vLLM re-opens per
        # bucket and needs no such floor.
        min_bucket_bytes = 0
        if is_atom:
            min_bucket_bytes = max(
                (self._sent_nbytes(p, keep_fp32) for _, p in params),
                default=0,
            )

        # ATOM's receive loop needs the whole bucket for its largest tensor
        # and one mapping per cycle, so it keeps the verl protocol.
        legacy = {"gc_collect": True, "double_buffer": False} if is_atom else {}

        stream = _gen()
        if fp8_quantize:
            # After _gen(), not inside it: the quantizer needs the full (all-gathered),
            # device-resident tensor that _gen() produces.
            from atomrl.engine.inference.fp8_weight_quantizer import (
                quantize_weights_fp8_per_block,
            )

            stream = quantize_weights_fp8_per_block(
                stream,
                pow2_scales=bool(
                    get_nested_config(
                        self.config,
                        "quantization",
                        "rollout",
                        "use_weight_pow2_scale",
                        default=False,
                    )
                ),
            )

        sender = BucketedWeightSender(
            zmq_handle=handle,
            bucket_size_mb=int(bucket_size_mb),
            use_shm=bool(use_shm),
            version=version,
            min_bucket_bytes=min_bucket_bytes,
            **legacy,
        )
        t_send = time.perf_counter()
        asyncio.run(sender.async_send_weights(stream))
        total_s = time.perf_counter() - t_send
        peak_alloc = 0.0
        if torch.cuda.is_available():
            peak_alloc = float(torch.cuda.max_memory_allocated())
        return {
            "gather_s": gather_s,
            "send_s": max(0.0, total_s - gather_s),
            "total_s": total_s,
            "peak_alloc_gb": peak_alloc / gb,
            "peak_extra_gb": max(0.0, peak_alloc - alloc0) / gb,
            **{f"sender_{k}": float(v) for k, v in sender.stats.items()},
        }

    def cleanup(self) -> None:
        self._engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().cleanup()
