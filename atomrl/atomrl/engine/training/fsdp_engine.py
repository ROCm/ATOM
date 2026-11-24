# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 AtomRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code adapted from the verl project
# (https://github.com/verl-project/verl).
# Original: verl/workers/engine/fsdp/transformer_impl.py

"""FSDP2 training engine implementation."""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from atomrl.core.config import (
    FSDPEngineConfig,
    HFModelConfig,
    OptimizerConfig,
)
from atomrl.engine.training.base_engine import BaseEngine, BaseEngineCtx, EngineRegistry
from atomrl.engine.training.fsdp_backend import FSDP2Backend
from atomrl.utils.fsdp_utils import (
    fsdp2_clip_grad_norm_,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from atomrl.utils.lr_scheduler import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from atomrl.utils.torch_functional import (
    calculate_sum_pi_squared_from_logits,
    entropy_from_logits,
    logprobs_from_logits,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ATOMRL_LOGGING_LEVEL", "WARN"))


class FSDP2Engine(BaseEngine):
    """Concrete engine implementation backed by PyTorch FSDP2 (``fully_shard``).

    Integrates the existing ``FSDP2Backend`` for model construction and sharding,
    and adds optimizer, LR scheduler, micro-batching, gradient accumulation,
    offload, and checkpoint management on top.
    """

    def __init__(
        self,
        model_config: HFModelConfig | dict[str, Any],
        engine_config: FSDPEngineConfig | dict[str, Any],
        optimizer_config: OptimizerConfig | dict[str, Any],
        model_name: str = "",
        quant_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.model_config = (
            model_config
            if isinstance(model_config, HFModelConfig)
            else HFModelConfig(**model_config)
        )
        self.engine_config = (
            engine_config
            if isinstance(engine_config, FSDPEngineConfig)
            else FSDPEngineConfig(**engine_config)
        )
        self.optimizer_config = (
            optimizer_config
            if isinstance(optimizer_config, OptimizerConfig)
            else OptimizerConfig(**optimizer_config)
        )
        self.model_name = model_name or self.model_config.local_path
        self.quant_config = quant_config or {}

        self.mode: str | None = None
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = (
            self.model_config.lora.enabled and self.model_config.lora.rank > 0
        )

        self.module: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler = None

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def initialize(self) -> None:
        """Build model, apply FSDP2 sharding, create optimizer and LR scheduler."""
        self._build_model_optimizer()
        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

    def _build_model_optimizer(self) -> None:
        training_cfg = self._make_backend_config()

        module = FSDP2Backend.build_model(self.model_name, training_cfg)
        module = FSDP2Backend.apply_lumen_optimizations(module, self.quant_config)

        if self._is_lora:
            module = self._build_lora_module(module)

        fsdp_cfg = self._make_fsdp_config()
        module = FSDP2Backend.apply_fsdp2(module, fsdp_cfg)

        self.module = module

        if not self.engine_config.forward_only:
            self.optimizer = self._build_optimizer(module)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)

    def _make_backend_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        # An explicit tiny_lm arch selects the stub body even when model_name is
        # set, so a test can name the config it stands in for. Without it only an
        # empty local_path reached _TinyLM -- and then with the default arch,
        # because the declared one never got this far.
        if self.engine_config.tiny_lm:
            cfg["use_tiny_lm"] = True
            cfg["tiny_lm"] = dict(self.engine_config.tiny_lm)
        elif self.model_config.local_path:
            cfg["model_dtype"] = self.engine_config.model_dtype
        else:
            cfg["use_tiny_lm"] = True
        # Propagate Liger kernel flag so build_model can apply it before loading
        if self.model_config.use_liger:
            cfg["use_liger"] = True
        return cfg

    def _make_fsdp_config(self) -> dict[str, Any]:
        mp = self.engine_config.mixed_precision or {}
        return {
            "enabled": True,
            "param_dtype": mp.get("param_dtype", self.engine_config.model_dtype),
            "reduce_dtype": mp.get("reduce_dtype", "fp32"),
            "reshard_after_forward": self.engine_config.reshard_after_forward,
            "param_offload": self.engine_config.param_offload,
        }

    def _build_lora_module(self, module: nn.Module) -> nn.Module:
        from peft import LoraConfig as PeftLoraConfig
        from peft import PeftModel, TaskType, get_peft_model

        module.enable_input_require_grads()
        lora = self.model_config.lora

        if lora.adapter_path:
            module = PeftModel.from_pretrained(
                module, lora.adapter_path, is_trainable=True
            )
        else:
            lora_cfg = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora.rank,
                lora_alpha=lora.alpha,
                target_modules=lora.target_modules,
                exclude_modules=lora.exclude_modules,
                bias="none",
            )
            module = get_peft_model(module, lora_cfg)
        return module

    def _build_optimizer(self, module: nn.Module) -> torch.optim.Optimizer:
        cfg = self.optimizer_config
        params = [p for p in module.parameters() if p.requires_grad]
        if cfg.optimizer_type == "adamw":
            return torch.optim.AdamW(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                eps=cfg.adam_eps,
            )
        raise NotImplementedError(f"Unsupported optimizer: {cfg.optimizer_type}")

    def _build_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        cfg = self.optimizer_config
        num_warmup = cfg.lr_warmup_steps
        if num_warmup <= 0:
            num_warmup = int(cfg.lr_warmup_steps_ratio * cfg.total_training_steps)

        if cfg.lr_scheduler_type == "constant":
            return get_constant_schedule_with_warmup(optimizer, num_warmup)
        elif cfg.lr_scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup,
                cfg.total_training_steps,
                min_lr_ratio=cfg.min_lr_ratio,
                num_cycles=cfg.num_cycles,
            )
        raise NotImplementedError(
            f"LR scheduler type {cfg.lr_scheduler_type} not supported"
        )

    # ------------------------------------------------------------------
    # Mode context managers
    # ------------------------------------------------------------------

    def train_mode(self, **kwargs) -> AbstractContextManager:
        return _EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs) -> AbstractContextManager:
        return _EngineEvalModeCtx(self, **kwargs)

    # ------------------------------------------------------------------
    # Data-parallel helpers
    # ------------------------------------------------------------------

    def get_data_parallel_rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def get_data_parallel_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def get_data_parallel_group(self):
        return dist.group.WORLD if dist.is_initialized() else None

    def is_mp_src_rank_with_outputs(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward_backward_batch(
        self,
        data: dict[str, Any],
        loss_function: Callable,
        forward_only: bool = False,
    ) -> dict[str, Any]:
        micro_batches = self._prepare_micro_batches(data)

        # verl-aligned packed (remove-padding + varlen) forward. Removes all pad
        # tokens so the distribution is not flattened by attending to left-pad
        # (which the AITER attention bias path does not reliably mask). This is
        # the path validated to match verl to 1e-4; the padded SDPA path is not.
        use_packing = bool(data.get("use_packed_forward"))

        output_lst: list[dict] = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        # Gradient sync stays on for every micro-batch. Disabling it (the classic
        # no_sync accumulation pattern) makes FSDP2 skip the reduce-scatter and
        # hold the FULL UNSHARDED gradient until the last micro-batch: ~30 GB for
        # an 8B model, measured as the dominant term in this actor's peak. Syncing
        # per micro-batch keeps gradients sharded (~3.7 GB at DP=8) and is
        # mathematically identical, since averaging across ranks is linear.
        for mb in micro_batches:

            if use_packing:
                from atomrl.engine.training.packing import (
                    PackingContext,
                    pack_sequences,
                )
                from atomrl.moe.rollout_routing import RoutingReplayContext

                ids = mb["input_ids"]
                am = mb.get("attention_mask")
                if am is None:
                    am = torch.ones_like(ids)
                packed = pack_sequences(
                    ids.to(self._device_or_cuda()), am.to(self._device_or_cuda())
                )
                mb["_packed"] = packed
                mb["_padded_seq_len"] = int(ids.shape[1])
                r3_idx, r3_valid = self._packed_rollout_routing(mb, packed)
                # Both contexts must stay alive through backward so gradient
                # checkpointing recompute sees the same varlen metadata and the
                # same routing the forward used.
                with (
                    ctx,
                    PackingContext(packed.cu_seqlens, packed.max_seqlen),
                    RoutingReplayContext(r3_idx, r3_valid),
                ):
                    loss, meta = self.forward_step(mb, loss_function, forward_only)
                    if not forward_only:
                        loss.backward()
            else:
                with ctx:
                    loss, meta = self.forward_step(mb, loss_function, forward_only)
                    if not forward_only:
                        loss.backward()

            output_lst.append(meta)

            # verl-aligned (VERL_EMPTY_CACHE_PER_MICRO_BATCH): free the allocator's
            # cached blocks after every micro-batch so variable-length sequences do
            # not accumulate fragmentation across the ~batch/micro-bsz iterations.
            # Default-on (matches verl); the small sync cost is worth the lower peak.
            del loss, meta
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self._postprocess_batch(output_lst)

    def _device_or_cuda(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_step(
        self,
        micro_batch: dict[str, Any],
        loss_function: Callable,
        forward_only: bool,
    ) -> tuple[torch.Tensor, dict]:
        """Single micro-batch forward (and optional backward).

        Returns ``(loss_tensor, meta_dict)`` where *meta_dict* contains
        ``model_output``, scalar ``loss``, and ``metrics``.
        """
        assert self.module is not None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = micro_batch["input_ids"].to(device)

        packed = micro_batch.get("_packed")
        if packed is not None:
            # Packed (varlen) forward: no pad tokens, exact per-token positions.
            raw_output = self.module(
                input_ids=packed.input_ids,
                position_ids=packed.position_ids,
                attention_mask=None,
                use_cache=False,
            )
            logits = raw_output.logits if hasattr(raw_output, "logits") else raw_output
            model_output = self._prepare_packed_outputs(logits, packed, micro_batch)
        else:
            attention_mask = micro_batch.get("attention_mask")
            position_ids = micro_batch.get("position_ids")

            model_kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask.to(device)
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids.to(device)

            raw_output = self.module(**model_kwargs)
            logits = raw_output.logits if hasattr(raw_output, "logits") else raw_output

            model_output = self._prepare_model_outputs(logits, input_ids, micro_batch)

        if loss_function is not None:
            loss, metrics = loss_function(model_output=model_output, data=micro_batch)
        else:
            assert forward_only
            loss = torch.tensor(0.0, device=device)
            metrics = {}

        meta = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }
        return loss, meta

    def _prepare_model_outputs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        micro_batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compute log_probs (and optionally entropy / sum-pi-squared) from logits."""
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = logprobs_from_logits(shift_logits, shift_labels)
        result: dict[str, torch.Tensor] = {"log_probs": log_probs}

        meta = micro_batch.get("meta", {}) if isinstance(micro_batch, dict) else {}
        if meta.get("calculate_entropy", False):
            # fp32 upcast: eager bf16 reduction over the ~152k vocab biases entropy
            # high; verl-aligned actor/entropy is computed in fp32 (see SMOKE_COMPARE).
            result["entropy"] = entropy_from_logits(shift_logits.float())
        if meta.get("calculate_sum_pi_squared", False):
            result["sum_pi_squared"] = calculate_sum_pi_squared_from_logits(
                shift_logits
            )

        return result

    def _prepare_packed_outputs(
        self,
        logits: torch.Tensor,
        packed: Any,
        micro_batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compute log_probs (and optionally entropy) from packed (flat) logits.

        Mirrors :meth:`_prepare_model_outputs` but on the varlen-packed forward,
        unpacking back to the ``[B, S-1]`` left-padded layout the loss / metric
        code expects. Applies verl's ``logits.div_(temperature)`` convention.
        """
        from atomrl.engine.training.packing import (
            packed_token_entropy,
            packed_token_log_probs,
            unpack_log_probs,
        )

        logits = logits.squeeze(0)  # (total_tokens, V)
        S = int(micro_batch.get("_padded_seq_len", micro_batch["input_ids"].shape[1]))
        temperature = float(micro_batch.get("temperature", 1.0) or 1.0)

        flat_lp = packed_token_log_probs(
            logits,
            packed.input_ids.squeeze(0),
            packed.cu_seqlens,
            temperature=temperature,
        )
        log_probs = unpack_log_probs(flat_lp, packed.cu_seqlens, packed.seq_lens, S)
        result: dict[str, torch.Tensor] = {"log_probs": log_probs}

        meta = micro_batch.get("meta", {}) if isinstance(micro_batch, dict) else {}
        if meta.get("calculate_entropy", False):
            flat_ent = packed_token_entropy(
                logits.detach(),
                packed.cu_seqlens,
                temperature=temperature,
                upcast=True,
            )
            result["entropy"] = unpack_log_probs(
                flat_ent,
                packed.cu_seqlens,
                packed.seq_lens,
                S,
            )
        return result

    def _packed_rollout_routing(self, mb: dict[str, Any], packed: Any):
        """Lay this micro-batch's R3 routing out in packed token order."""
        rows = mb.get("rollout_routing")
        if not rows:
            return None, None
        from atomrl.moe.rollout_routing import pack_rollout_routing

        sample = next((r for r in rows if r is not None), None)
        if sample is None:
            return None, None
        n_layers, top_k = int(sample.shape[1]), int(sample.shape[2])
        return pack_rollout_routing(
            rows,
            packed.seq_lens,
            n_layers,
            top_k,
            self._device_or_cuda(),
        )

    def _prepare_micro_batches(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Split *data* into micro-batches by token budget, else by row count.

        ``max_token_len_per_gpu`` packs as many sequences as fit the budget,
        matching verl's ``use_dynamic_bsz`` + ``ppo_max_token_len_per_gpu``. Row
        splitting (``micro_batch_size``) is the fallback; one row per micro-batch
        leaves the GPU far below saturation and multiplies the FSDP2 all-gather
        cycles by the row count.
        """
        max_token_len = data.get("max_token_len_per_gpu")
        micro_batch_size = data.get("micro_batch_size")
        input_ids = data.get("input_ids")
        if input_ids is None:
            return [data]

        total_rows = int(input_ids.shape[0])

        def slice_rows(start: int, end: int) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v[start:end]
                elif isinstance(v, list) and len(v) == total_rows:
                    # per-row payloads (e.g. R3 rollout routing) must follow rows
                    out[k] = v[start:end]
                else:
                    out[k] = v
            return out

        bounds: list[tuple[int, int]] = []
        total = input_ids.shape[0]
        if max_token_len is not None and max_token_len > 0:
            am = data.get("attention_mask")
            seq_lens = (
                am.sum(dim=1).long().tolist()
                if am is not None
                else [int(input_ids.shape[1])] * total
            )
            start = 0
            while start < total:
                tok, end = 0, start
                while end < total:
                    sl = int(seq_lens[end])
                    if tok + sl > max_token_len and end > start:
                        break
                    tok += sl
                    end += 1
                bounds.append((start, end))
                start = end
        elif micro_batch_size is not None and micro_batch_size > 0:
            bounds = [
                (s, min(s + micro_batch_size, total))
                for s in range(0, total, micro_batch_size)
            ]
        else:
            return [data]

        batches = [slice_rows(s, e) for s, e in bounds]

        # FSDP2 all-gathers are collective, so every rank must run the same
        # number of forward/backward passes. Token-budget splitting can yield
        # different counts per rank; pad with a zero-response-mask copy of the
        # last micro-batch, which contributes no loss and no gradient.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            if world > 1:
                count = torch.tensor([len(batches)], device=self._device_or_cuda())
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX)
                target = int(count.item())
                while len(batches) < target:
                    dummy = {
                        k: (v.clone() if isinstance(v, torch.Tensor) else v)
                        for k, v in batches[-1].items()
                    }
                    if isinstance(dummy.get("response_mask"), torch.Tensor):
                        dummy["response_mask"] = torch.zeros_like(
                            dummy["response_mask"]
                        )
                    batches.append(dummy)
        return batches

    def _postprocess_batch(self, output_lst: list[dict]) -> dict[str, Any]:
        model_output: dict[str, list] = {}
        losses: list[float] = []
        aggregated_metrics: dict[str, list] = {}

        for o in output_lst:
            if "model_output" in o:
                for key, val in o["model_output"].items():
                    model_output.setdefault(key, []).append(val)
            if "loss" in o:
                losses.append(o["loss"])
            if "metrics" in o:
                for key, val in o["metrics"].items():
                    aggregated_metrics.setdefault(key, []).append(val)

        concat_output: dict[str, torch.Tensor] = {}
        for key, vals in model_output.items():
            if all(isinstance(v, torch.Tensor) for v in vals):
                concat_output[key] = torch.cat(vals, dim=0)
            else:
                concat_output[key] = vals

        return {
            "model_output": concat_output,
            "loss": losses,
            "metrics": aggregated_metrics,
        }

    # ------------------------------------------------------------------
    # Optimizer / LR
    # ------------------------------------------------------------------

    def optimizer_zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self) -> float:
        assert self.optimizer is not None
        assert self.module is not None
        clip = self.optimizer_config.clip_grad

        try:
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=clip)
        except RuntimeError:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), max_norm=clip
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        if not torch.isfinite(grad_norm):
            logger.warning("grad_norm is not finite: %s — skipping update", grad_norm)
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.step()

        return grad_norm.item()

    def lr_scheduler_step(self) -> float:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            return self.lr_scheduler.get_last_lr()[0]
        return 0.0

    # ------------------------------------------------------------------
    # Offload / device movement
    # ------------------------------------------------------------------

    def to(
        self,
        device: str,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
    ) -> None:
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        if self.module is None:
            return
        if device in ("cuda", "gpu"):
            if model:
                load_fsdp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                offload_fsdp_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        local_path: str,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs: Any,
    ) -> None:
        assert self.module is not None
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        state = {
            "model": self.module.state_dict(),
            "global_step": global_step,
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()

        save_path = os.path.join(local_path, f"step_{global_step}")
        os.makedirs(save_path, exist_ok=True)
        rank = self.rank
        torch.save(state, os.path.join(save_path, f"rank_{rank}.pt"))

        if dist.is_initialized():
            dist.barrier()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(self, local_path: str, **kwargs: Any) -> None:
        assert self.module is not None
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        rank = self.rank
        ckpt_path = os.path.join(local_path, f"rank_{rank}.pt")
        if not os.path.exists(ckpt_path):
            logger.warning("Checkpoint not found: %s", ckpt_path)
            return

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.module.load_state_dict(state["model"])
        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in state:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])

        if dist.is_initialized():
            dist.barrier()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        if self._is_offload_optimizer and self.optimizer is not None:
            offload_fsdp_optimizer(self.optimizer)

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def get_per_tensor_param(self, **kwargs):
        assert self.module is not None
        load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        return params.items(), None

    def disable_adapter(self) -> AbstractContextManager:
        if hasattr(self.module, "disable_adapter"):
            return self.module.disable_adapter()
        return nullcontext()


# ------------------------------------------------------------------
# Registered engine variants
# ------------------------------------------------------------------


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"])
class FSDP2EngineWithLMHead(FSDP2Engine):
    """Language-model variant — the default for policy/actor training."""


# ------------------------------------------------------------------
# Mode context managers
# ------------------------------------------------------------------


class _EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDP2Engine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        super().__enter__()
        if self.engine.module is not None:
            self.engine.module.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.engine, FSDP2Engine):
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_val, exc_tb)


class _EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDP2Engine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        super().__enter__()
        if self.engine.module is not None:
            self.engine.module.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
