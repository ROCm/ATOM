# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Standalone SFT trainer for AtomRL.

Launched via ``torchrun``, uses the same FSDP2Engine / MegatronEngine as RL
training but with a supervised NLL loss over loss-masked tokens.  Does not
require Ray, rollout engines, or reward models.

Usage::

    torchrun --nproc_per_node=8 -m atomrl.trainer.sft_trainer \
        --model_name /path/to/model \
        --data_files /path/to/data.parquet \
        --max_length 4096 \
        --num_epochs 3 \
        --lr 2e-5
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from atomrl.algorithms.loss_functions import sft_loss
from atomrl.core.config import (
    FSDPEngineConfig,
    HFModelConfig,
    OptimizerConfig,
    ProfilerConfig,
    TorchProfilerScheduleConfig,
    TorchProfilerToolConfig,
)
from atomrl.data.sft_dataset import SFTDataset
from atomrl.engine.training.base_engine import EngineRegistry
from atomrl.utils.profiler import DistProfiler

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Minimal SFT trainer wrapping AtomRL's FSDP2Engine."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(self.local_rank)

        self._build_dataset()
        self._build_engine()
        self._init_profiler()
        self.global_step = 0

    def _build_dataset(self) -> None:
        self.dataset = SFTDataset(
            data_files=self.args.data_files,
            tokenizer=self.args.model_name,
            max_length=self.args.max_length,
            messages_key=self.args.messages_key,
            pad_mode="right",
            truncation="right",
        )
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def _build_engine(self) -> None:
        model_config = HFModelConfig(local_path=self.args.model_name)
        engine_config = FSDPEngineConfig(
            model_dtype="fp32",
            mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32"},
            reshard_after_forward=True,
            seed=self.args.seed,
        )
        optimizer_config = OptimizerConfig(
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            clip_grad=self.args.max_grad_norm,
            lr_warmup_steps=self.args.lr_warmup_steps,
            total_training_steps=len(self.dataloader) * self.args.num_epochs,
            lr_scheduler_type=self.args.lr_scheduler_type,
        )

        engine_cls = EngineRegistry.get_engine_cls(
            model_type="language_model",
            backend="fsdp2",
        )
        self.engine = engine_cls(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            model_name=self.args.model_name,
            quant_config={},
        )
        self.engine.initialize()

    def _init_profiler(self) -> None:
        sched_cfg = None
        if self.args.profiler_schedule_active > 0:
            sched_cfg = TorchProfilerScheduleConfig(
                skip_first=self.args.profiler_schedule_skip_first,
                wait=self.args.profiler_schedule_wait,
                warmup=self.args.profiler_schedule_warmup,
                active=self.args.profiler_schedule_active,
                repeat=self.args.profiler_schedule_repeat,
            )
        tool_cfg = TorchProfilerToolConfig(schedule=sched_cfg)
        profiler_cfg = ProfilerConfig(
            tool=self.args.profiler_tool,
            enable=self.args.profiler_enable,
            all_ranks=self.args.profiler_all_ranks,
            save_path=self.args.profiler_save_path,
            tool_config=tool_cfg,
        )
        self._profiler = DistProfiler(rank=self.rank, config=profiler_cfg)
        self._profile_start = self.args.profiler_start_step
        self._profile_end = self.args.profiler_end_step

    def _sft_loss_fn(self, model_output, data):
        log_probs = model_output["log_probs"]
        L = log_probs.shape[-1]
        lm = data["loss_mask"].to(log_probs.device).float()
        if lm.shape[-1] == L + 1:
            lm = lm[:, 1:]
        lm = lm[..., :L]
        dp_group = self.engine.get_data_parallel_group()
        return sft_loss(
            log_probs,
            lm,
            loss_agg_mode=self.args.loss_agg_mode,
            dp_size=self.world_size,
            dp_group=dp_group,
        )

    def train(self) -> None:
        for epoch in range(self.args.num_epochs):
            self.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_tokens = 0.0
            step_count = 0
            t0 = time.time()

            for batch in self.dataloader:
                self.global_step += 1
                if self.global_step == self._profile_start:
                    self._profiler.start(profile_step=self.global_step)

                data = {k: v.cuda() for k, v in batch.items()}
                data["use_packed_forward"] = True

                with self.engine.train_mode():
                    output = self.engine.train_batch(data, self._sft_loss_fn)

                self.engine.lr_scheduler_step()

                self._profiler.step()

                if self.global_step == self._profile_end:
                    self._profiler.stop()
                step_count += 1

                metrics = output.get("metrics", {})
                loss_val = output.get("loss", 0.0)
                if isinstance(loss_val, list):
                    loss_val = sum(loss_val) / max(len(loss_val), 1)
                epoch_loss += float(loss_val)

                num_tok = metrics.get("num_tokens", [0.0])
                if isinstance(num_tok, list):
                    num_tok = sum(num_tok)
                epoch_tokens += float(num_tok)

                if self.rank == 0 and self.global_step % self.args.log_interval == 0:
                    avg_loss = epoch_loss / step_count
                    grad_norm_vals = metrics.get("grad_norm", [0.0])
                    if isinstance(grad_norm_vals, list):
                        grad_norm = sum(grad_norm_vals) / max(len(grad_norm_vals), 1)
                    else:
                        grad_norm = float(grad_norm_vals)
                    elapsed = time.time() - t0
                    tps = epoch_tokens / max(elapsed, 1e-6)
                    logger.info(
                        "epoch=%d step=%d loss=%.4f grad_norm=%.4f tokens/s=%.0f lr=%.2e",
                        epoch,
                        self.global_step,
                        avg_loss,
                        grad_norm,
                        tps,
                        (
                            self.engine.lr_scheduler.get_last_lr()[0]
                            if self.engine.lr_scheduler
                            else 0.0
                        ),
                    )

                if (
                    self.args.save_steps > 0
                    and self.global_step % self.args.save_steps == 0
                ):
                    self._save_checkpoint()

            if self.rank == 0:
                logger.info(
                    "Epoch %d done: avg_loss=%.4f total_tokens=%.0f",
                    epoch,
                    epoch_loss / max(step_count, 1),
                    epoch_tokens,
                )

        if self._profiler is not None:
            self._profiler.stop()

        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        save_dir = os.path.join(self.args.output_dir, f"global_step_{self.global_step}")
        self.engine.save_checkpoint(save_dir, global_step=self.global_step)
        if self.rank == 0:
            logger.info("Checkpoint saved to %s", save_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AtomRL SFT Trainer")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--data_files", type=str, nargs="+", required=True)
    p.add_argument("--output_dir", type=str, default="./sft_checkpoints")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--messages_key", type=str, default="messages")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_warmup_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--loss_agg_mode", type=str, default="token-mean")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # Profiler arguments
    p.add_argument("--profiler_enable", action="store_true", help="Enable profiling")
    p.add_argument(
        "--profiler_tool", type=str, default="torch", choices=["torch", "rocprof"]
    )
    p.add_argument(
        "--profiler_start_step",
        type=int,
        default=-1,
        help="Step to start profiling (negative = disabled)",
    )
    p.add_argument(
        "--profiler_end_step",
        type=int,
        default=-1,
        help="Step to stop profiling (negative = disabled)",
    )
    p.add_argument("--profiler_save_path", type=str, default="./profiler_output")
    p.add_argument(
        "--profiler_all_ranks", action="store_true", help="Profile all ranks"
    )
    # Schedule arguments (torch.profiler.schedule)
    p.add_argument("--profiler_schedule_skip_first", type=int, default=0)
    p.add_argument("--profiler_schedule_wait", type=int, default=0)
    p.add_argument("--profiler_schedule_warmup", type=int, default=1)
    p.add_argument("--profiler_schedule_active", type=int, default=3)
    p.add_argument("--profiler_schedule_repeat", type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    trainer = SFTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
