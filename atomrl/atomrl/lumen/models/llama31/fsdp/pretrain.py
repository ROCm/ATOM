# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""LLaMA 3.1 Pretraining with PyTorch FSDP.

This module provides a complete pretraining trainer using:

- **HuggingFace Transformers** for the LLaMA 3.1 model
- **PyTorch FSDP** for distributed sharding (no Megatron dependency)
- **Lumen** for FP8 quantised training

Features (parity with the Megatron backend):
    - Full pretraining or LoRA (via HuggingFace PEFT)
    - FP8 quantised training (weight/activation via ``quant.enable``)
    - Cosine annealing LR with warmup
    - Synthetic warmup with FP8 state reset
    - Early stopping based on validation loss target
    - Gradient accumulation + gradient clipping

Example::

    from atomrl.lumen.models.llama31.fsdp import FSDPTrainer, get_args

    args = get_args()
    trainer = FSDPTrainer(args)
    trainer.train()
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

from atomrl.lumen.models.experiment_ops import (
    ExperimentTracker,
    get_effective_stop_step,
)

# Re-export shared FSDP helpers so existing callers are not broken.
from atomrl.lumen.models.fsdp import (
    add_common_fsdp_args,
    apply_fp8_training,
    apply_lora,
    build_cosine_warmup_scheduler,
    register_quant_optimizer_hooks,
    reset_fp8_state,
    save_fsdp_checkpoint,
    should_run_eval_step,
    sync_scheduler_to_ckpt_step,
)
from atomrl.lumen.models.llama31.dataset import PretrainTextDataset

__all__ = [
    "FSDPTrainer",
    "apply_fp8_training",
    "apply_lora",
    "build_model",
    "get_args",
    "reset_fp8_state",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(args) -> nn.Module:
    """Build LLaMA 3.1 model from HuggingFace."""
    from transformers import LlamaConfig, LlamaForCausalLM

    if args.model_name_or_path:
        logger.info("Loading model from %s", args.model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        logger.info("Building LLaMA 3.1 from scratch (%d layers)", args.num_layers)
        config = LlamaConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads or args.num_attention_heads,
            vocab_size=args.vocab_size,
            max_position_embeddings=args.max_position_embeddings,
            rope_theta=args.rope_theta,
            rms_norm_eps=1e-5,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = LlamaForCausalLM(config)

    return model


# ---------------------------------------------------------------------------
# FSDP Trainer
# ---------------------------------------------------------------------------

_SHARDING_MAP = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
}


class FSDPTrainer:
    """Pretraining trainer using PyTorch FSDP + HuggingFace LLaMA 3.1."""

    def __init__(self, args):
        self.args = args
        from atomrl.lumen.models.fsdp import validate_fsdp_quant_args

        validate_fsdp_quant_args(args)
        self._setup_distributed()

        if args.use_ckpt:
            resume_path = args.initial_ckpt_path or args.continual_ckpt_path
            if resume_path:
                args.model_name_or_path = resume_path
                if self.rank == 0:
                    source_kind = (
                        "HF weights" if args.resume_from_hf else "checkpoint weights"
                    )
                    logger.info("Resuming %s from %s", source_kind, resume_path)

        self.model = build_model(args)

        if getattr(args, "gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            if self.rank == 0:
                logger.info("Gradient checkpointing enabled")

        from atomrl.lumen.config import LumenConfig

        cfg = LumenConfig.from_args(args)
        _manager, self.model = cfg.enable(self.model)

        self.model = self._wrap_fsdp(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=args.weight_decay,
        )
        register_quant_optimizer_hooks(self.model, self.optimizer, args)
        self.scheduler = self._build_scheduler()
        sync_scheduler_to_ckpt_step(self.scheduler, args)

        self.train_loader = self._build_dataloader(
            args.train_data_path, args.train_samples
        )
        self.val_loader = (
            self._build_dataloader(args.val_data_path, args.val_samples)
            if args.val_data_path
            else None
        )
        self.tracker = ExperimentTracker(
            args=args,
            global_batch_size=args.micro_batch_size
            * args.gradient_accumulation_steps
            * self.world_size,
            seq_length=args.seq_length,
            backend="fsdp",
            task_name="llama31-pretrain",
            rank=self.rank,
        )

    # ------------------------------------------------------------------

    def _setup_distributed(self):
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(self.local_rank)

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        if getattr(self.args, "fsdp_version", 1) == 2:
            from atomrl.lumen.models.fsdp import apply_fsdp2

            apply_fsdp2(model, self.args)
            return model

        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={LlamaDecoderLayer},
        )
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=(
                torch.float32
                if (self.args.linear_fp8 or getattr(self.args, "linear_fp4", False))
                else torch.bfloat16
            ),
            buffer_dtype=torch.bfloat16,
        )
        sharding = _SHARDING_MAP.get(
            self.args.sharding_strategy, ShardingStrategy.FULL_SHARD
        )
        return FSDP(
            model,
            sharding_strategy=sharding,
            mixed_precision=mp,
            auto_wrap_policy=wrap_policy,
            device_id=self.local_rank,
        )

    def _build_scheduler(self):
        return build_cosine_warmup_scheduler(self.optimizer, self.args)

    def _build_dataloader(self, data_path, max_samples):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_name_or_path or self.args.model_name_or_path
        )

        ds = PretrainTextDataset(
            data_path=data_path,
            seq_length=self.args.seq_length,
            tokenizer=tokenizer,
            is_hf_tokenizer=True,
            max_samples=max_samples,
        )

        sampler = DistributedSampler(
            ds, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        return DataLoader(
            ds,
            batch_size=self.args.micro_batch_size,
            sampler=sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def _synthetic_warmup_step(self):
        args = self.args
        fake_ids = (
            torch.ones(
                args.micro_batch_size,
                args.seq_length,
                dtype=torch.long,
                device=self.local_rank,
            )
            * 3545
        )
        fake_ids[:, -1] = 2
        labels = fake_ids.clone()

        self.optimizer.zero_grad()
        outputs = self.model(input_ids=fake_ids, labels=labels)
        outputs.loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------

    def train(self):
        args = self.args
        if args.warmup_steps > 0:
            if self.rank == 0:
                logger.info("Running %d synthetic warmup steps ...", args.warmup_steps)
            for _ in range(args.warmup_steps):
                self._synthetic_warmup_step()
            # MXFP4 counts here too, matching the Megatron and TRL warmups.
            if args.linear_fp8 or getattr(args, "linear_fp4", False):
                reset_fp8_state(self.model)
            if dist.is_initialized():
                dist.barrier()
            if self.rank == 0:
                logger.info("Synthetic warmup complete. Real training begins.")

        self.model.train()
        global_step = args.ckpt_start_step
        effective_stop_step = get_effective_stop_step(args)
        self.tracker.on_train_start(
            configs={
                "global_batch_size": self.tracker.global_batch_size,
                "max_sequence_length": args.seq_length,
                "max_steps": args.max_steps,
                "effective_stop_step": effective_stop_step,
            }
        )

        data_iter = iter(self.train_loader)
        try:
            while global_step < effective_stop_step:
                self.tracker.record_train_step_start(global_step)
                self.optimizer.zero_grad()
                accum_loss = 0.0

                for _micro in range(args.gradient_accumulation_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_loader)
                        batch = next(data_iter)

                    input_ids = batch["input_ids"].cuda(non_blocking=True)
                    labels = batch["labels"].cuda(non_blocking=True)

                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation_steps
                    loss.backward()
                    accum_loss += loss.item()

                if args.max_grad_norm > 0:
                    if getattr(args, "fsdp_version", 1) == 2:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), args.max_grad_norm
                        )
                    else:
                        self.model.clip_grad_norm_(args.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                global_step += 1
                lr = self.scheduler.get_last_lr()[0]
                step_time_ms = self.tracker.record_train_step_end(
                    global_step=global_step,
                    train_loss=accum_loss,
                    lr=lr,
                )

                if global_step % args.log_interval == 0 and self.rank == 0:
                    logger.info(
                        "step %d/%d | loss %.4f | lr %.2e | step_time_ms %.1f",
                        global_step,
                        effective_stop_step,
                        accum_loss,
                        lr,
                        step_time_ms,
                    )

                if self.tracker.should_preempt(global_step=global_step):
                    if self.rank == 0:
                        logger.info(
                            "Preemptive stop triggered at step %d.", global_step
                        )
                    break

                if (
                    args.save_interval > 0
                    and global_step % args.save_interval == 0
                    and self.rank == 0
                ):
                    save_path = os.path.join(args.save_dir, f"step_{global_step}")
                    save_fsdp_checkpoint(self.model, save_path)
                    logger.info("Checkpoint saved to %s", save_path)

                if self.val_loader and should_run_eval_step(global_step, args):
                    self.tracker.log_validation_start(global_step)
                    val_loss = self._validate()
                    if self.rank == 0:
                        logger.info("step %d | val_loss %.4f", global_step, val_loss)
                    if self.tracker.should_stop_on_validation(
                        global_step=global_step, val_loss=val_loss
                    ):
                        if self.rank == 0 and self.tracker.target is not None:
                            logger.info(
                                "Val loss %.4f <= target %.4f. Stopping.",
                                val_loss,
                                self.tracker.target,
                            )
                        break

            if args.save_ckpt and self.rank == 0:
                final_path = args.continual_ckpt_path or os.path.join(
                    args.save_dir, "final"
                )
                save_fsdp_checkpoint(self.model, final_path)
                logger.info("Final checkpoint saved to %s", final_path)
        finally:
            self.tracker.finish_run(global_step=global_step)

        if self.rank == 0:
            logger.info("Training complete at step %d.", global_step)

    def _validate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].cuda(non_blocking=True)
                labels = batch["labels"].cuda(non_blocking=True)
                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                n_batches += 1
                if n_batches >= 10:
                    break
        self.model.train()

        avg = total_loss / max(n_batches, 1)
        if dist.is_initialized():
            t = torch.tensor([avg], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()
        return avg


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for FSDP-based LLaMA 3.1 pretraining."""
    parser = argparse.ArgumentParser(
        description="LLaMA 3.1 Pretraining with FSDP + Lumen"
    )

    # -- Model --
    m = parser.add_argument_group("model")
    m.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="HuggingFace model name or local path.",
    )
    m.add_argument("--hidden-size", type=int, default=4096)
    m.add_argument("--intermediate-size", type=int, default=14336)
    m.add_argument("--num-layers", type=int, default=32)
    m.add_argument("--num-attention-heads", type=int, default=32)
    m.add_argument("--num-key-value-heads", type=int, default=8)
    m.add_argument("--vocab-size", type=int, default=128256)
    m.add_argument("--max-position-embeddings", type=int, default=131072)
    m.add_argument("--rope-theta", type=float, default=500000.0)
    m.add_argument("--seq-length", type=int, default=8192)
    m.add_argument("--tokenizer-name-or-path", type=str, default=None)
    add_common_fsdp_args(parser)
    parser.set_defaults(
        micro_batch_size=2,
        max_steps=1200000,
        lr=8e-4,
        min_lr=8e-5,
        lr_warmup_steps=128,
        weight_decay=0.1,
        train_samples=None,
        val_samples=None,
        linear_fp8_amax_algo="most_recent",
        linear_fp8_amax_history=4,
    )

    mlp = parser.add_argument_group("mlperf")
    mlp.add_argument(
        "--size",
        type=str,
        default="8b",
        choices=["8b"],
        help="Model size (for Docker compatibility).",
    )

    return parser.parse_args()
