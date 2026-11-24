# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Training loop callbacks."""

from __future__ import annotations

import logging
import re
import shutil
import typing
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from atomrl.utils.checkpoint import CheckpointManager

if TYPE_CHECKING:
    from atomrl.trainer.rl_trainer import RLTrainer

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Hook points for the RL training loop."""

    def on_train_begin(self, trainer: RLTrainer) -> None:
        """Invoked once before the first training step."""

    def on_train_end(self, trainer: RLTrainer) -> None:
        """Invoked once after training completes."""

    def on_step_begin(self, trainer: RLTrainer, step: int) -> None:
        """Invoked at the beginning of each optimizer/rollout step."""

    def on_step_end(
        self, trainer: RLTrainer, step: int, metrics: dict[str, float]
    ) -> None:
        """Invoked after ``weight_sync`` with metrics from the finished step."""


class LoggingCallback(Callback):
    """Emit structured logs for rolling metrics every ``interval`` steps."""

    def __init__(self, interval: int = 1) -> None:
        self.interval = max(1, int(interval))

    def on_step_end(
        self, trainer: RLTrainer, step: int, metrics: dict[str, float]
    ) -> None:
        if (step + 1) % self.interval != 0:
            return
        if trainer._rank != 0:
            return
        parts = [f"{k}={v:.6g}" for k, v in sorted(metrics.items())]
        # Display 1-based step (aligns with verl's 1-indexed global_steps).
        logger.info("step=%d %s", step + 1, " ".join(parts))


class CheckpointCallback(Callback):
    """Save full training state (model, optimizer, step) for crash recovery.

    Uses FSDP2-aware ``get_model_state_dict`` / ``get_optimizer_state_dict``
    when distributed, falling back to plain ``state_dict()`` otherwise.
    Only rank 0 writes to disk.  Old checkpoints beyond ``save_total_limit``
    are pruned automatically.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int,
        save_total_limit: int = 3,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = max(1, int(save_interval))
        self.save_total_limit = max(1, int(save_total_limit))
        self._manager = CheckpointManager()

    def on_step_end(
        self, trainer: RLTrainer, step: int, metrics: dict[str, float]
    ) -> None:
        # Checkpoint names follow verl's 1-based global_step convention. Never
        # expose or persist the trainer's internal 0-based step as checkpoint_0.
        global_step = int(step) + 1
        if global_step <= 0:
            return
        if global_step % self.save_interval != 0:
            return

        if (
            getattr(trainer, "_use_ray_controller", False)
            and getattr(trainer, "_actor_wg", None) is not None
        ):
            self._save_ray_controller_checkpoint(trainer, global_step, metrics)
            return

        rank = trainer._rank
        ckpt_dir = Path(self.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"checkpoint_{global_step}.pt"

        state: dict[str, Any] = {
            "step": global_step,
            "metrics": metrics,
            "algo": trainer.config.algorithm.name,
        }

        model = getattr(trainer, "_actor_model", None) or getattr(
            trainer, "_draft_model", None
        )
        opt = trainer._optimizer

        # FSDP2 (fully_shard) parameters are sharded DTensors: a plain
        # ``model.state_dict()`` returns only THIS rank's 1/N shard, so saving it
        # from rank 0 alone persists just 1/N of every weight and the checkpoint
        # is unrecoverable. Use the FSDP2-aware ``get_model_state_dict`` with
        # ``full_state_dict=True`` (a collective all-gather that MUST run on every
        # rank) so the file holds the complete, unsharded weights.
        if model is not None:
            if getattr(trainer, "_is_distributed", False):
                from torch.distributed.checkpoint.state_dict import (
                    StateDictOptions,
                    get_model_state_dict,
                    get_optimizer_state_dict,
                )

                sd_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
                state["model_state_dict"] = get_model_state_dict(model, options=sd_opts)
                if opt is not None:
                    if hasattr(opt, "distributed_state_dict"):
                        state["optimizer_state_dict"] = opt.distributed_state_dict(
                            model, sd_opts
                        )
                        state["optimizer_state_format"] = "fp32_master_dcp"
                    else:
                        state["optimizer_state_dict"] = get_optimizer_state_dict(
                            model,
                            opt,
                            options=sd_opts,
                        )
            else:
                state["model_state_dict"] = {
                    k: v.cpu() for k, v in model.state_dict().items()
                }
                if opt is not None:
                    state["optimizer_state_dict"] = opt.state_dict()

        if opt is not None:
            if hasattr(opt, "fp32_params"):
                state["fp32_params"] = [p.data.cpu().clone() for p in opt.fp32_params]
            if hasattr(opt, "scheduler"):
                state["scheduler_last_epoch"] = opt.scheduler.last_epoch
                state["scheduler_state_dict"] = opt.scheduler.state_dict()

        if rank == 0:
            n_model = len(state.get("model_state_dict", {}))
            n_fp32 = len(state.get("fp32_params", []))
            logger.info(
                "Saving checkpoint step=%d: %d model keys, %d fp32 params, opt=%s, sched_epoch=%s",
                global_step,
                n_model,
                n_fp32,
                "yes" if "optimizer_state_dict" in state else "no",
                state.get("scheduler_last_epoch", "N/A"),
            )
            # Prune before writing: peak disk is then one checkpoint, not two.
            # A 30B-A3B FSDP2 checkpoint (fp32 weights + optimizer) is ~342GB,
            # so the transient second copy is the difference between fitting and
            # filling the disk mid-write.
            self._prune_old_checkpoints(ckpt_dir, keep=self.save_total_limit - 1)
            self._manager.save(state, path, global_step)

        if trainer._is_distributed:
            torch.distributed.barrier()

    def _save_ray_controller_checkpoint(
        self, trainer: RLTrainer, global_step: int, metrics: dict[str, float]
    ) -> None:
        ckpt_root = Path(self.checkpoint_dir)
        step_dir = ckpt_root / f"global_step_{global_step}"
        actor_dir = step_dir / "actor"
        # See _prune_old_checkpoints above: free the old checkpoint before the
        # new one is written so peak usage is one checkpoint, not two.
        ckpt_root.mkdir(parents=True, exist_ok=True)
        self._prune_old_ray_checkpoints(ckpt_root, keep=self.save_total_limit - 1)
        # Actor shards live on the training node's local filesystem while this
        # callback runs on the controller node. Prune on the owning node before
        # writing so checkpoint peak usage never exceeds save_total_limit.
        trainer._actor_wg.execute_all_sync(
            "prune_checkpoints",
            str(ckpt_root),
            max(0, self.save_total_limit - 1),
        )
        actor_dir.mkdir(parents=True, exist_ok=True)
        trainer._actor_wg.execute_all_sync(
            "save_checkpoint", str(actor_dir), global_step=global_step
        )
        meta = {
            "step": global_step,
            "metrics": metrics,
            "algo": trainer.config.algorithm.name,
            "format": "verl_ray_sharded",
        }
        self._manager.save(meta, step_dir / f"checkpoint_{global_step}.pt", global_step)
        (ckpt_root / "latest_checkpointed_iteration.txt").write_text(
            str(global_step), encoding="utf-8"
        )
        logger.info(
            "Saved Ray checkpoint to %s (global_step=%d)", step_dir, global_step
        )
        self._prune_old_ray_checkpoints(ckpt_root)
        trainer._actor_wg.execute_all_sync(
            "prune_checkpoints",
            str(ckpt_root),
            self.save_total_limit,
        )

    @staticmethod
    def _verify_checkpoint(path: Path, model, opt, step: int) -> None:
        """Load saved checkpoint back and compare against live model weights."""
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            saved = payload.get("state_dict", payload)
            saved_sd = saved.get("model_state_dict", {})

            live_sd = {k: v.cpu() for k, v in model.state_dict().items()}

            if len(saved_sd) != len(live_sd):
                logger.error(
                    "CKPT VERIFY FAIL step=%d: saved %d keys vs live %d keys",
                    step,
                    len(saved_sd),
                    len(live_sd),
                )
                return

            max_diff = 0.0
            for k, live_value in live_sd.items():
                if k not in saved_sd:
                    logger.error(
                        "CKPT VERIFY FAIL step=%d: key %s missing from checkpoint",
                        step,
                        k,
                    )
                    return
                diff = (saved_sd[k].float() - live_value.float()).abs().max().item()
                max_diff = max(max_diff, diff)

            if max_diff > 1e-6:
                logger.error(
                    "CKPT VERIFY FAIL step=%d: max diff=%.6g (should be 0)",
                    step,
                    max_diff,
                )
            else:
                logger.info(
                    "CKPT VERIFY OK step=%d: %d keys, max_diff=%.2e",
                    step,
                    len(saved_sd),
                    max_diff,
                )

            saved_fp32 = saved.get("fp32_params", [])
            if saved_fp32 and opt is not None and hasattr(opt, "fp32_params"):
                fp32_max_diff = 0.0
                for sp, lp in zip(saved_fp32, opt.fp32_params):
                    diff = (sp.cpu().float() - lp.data.cpu().float()).abs().max().item()
                    fp32_max_diff = max(fp32_max_diff, diff)
                logger.info(
                    "CKPT VERIFY FP32 step=%d: %d params, max_diff=%.2e",
                    step,
                    len(saved_fp32),
                    fp32_max_diff,
                )

        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("CKPT VERIFY ERROR step=%d: %s", step, exc)

    def _prune_old_checkpoints(self, ckpt_dir: Path, keep: int | None = None) -> None:
        keep = self.save_total_limit if keep is None else max(0, keep)
        pattern = re.compile(r"checkpoint_(\d+)\.pt$")
        ckpts: list[tuple[int, Path]] = []
        for p in ckpt_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                ckpts.append((int(m.group(1)), p))
        ckpts.sort(key=lambda x: x[0])
        while len(ckpts) > keep:
            _, old = ckpts.pop(0)
            try:
                old.unlink()
                logger.info("Pruned old checkpoint: %s", old)
            except OSError:
                pass

    def _prune_old_ray_checkpoints(
        self, ckpt_root: Path, keep: int | None = None
    ) -> None:
        keep = self.save_total_limit if keep is None else max(0, keep)
        pattern = re.compile(r"global_step_(\d+)$")
        ckpts: list[tuple[int, Path]] = []
        for p in ckpt_root.iterdir():
            if not p.is_dir():
                continue
            m = pattern.match(p.name)
            if m:
                ckpts.append((int(m.group(1)), p))
        ckpts.sort(key=lambda x: x[0])
        while len(ckpts) > keep:
            _, old = ckpts.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            logger.info("Pruned old Ray checkpoint: %s", old)


class EvalCallback(Callback):
    """Run periodic validation using the trainer hook.

    Must run on **all ranks** because teacher forward + FSDP require
    collective operations.  Only rank 0 logs the results.
    """

    def __init__(self, interval: int) -> None:
        self.interval = max(1, int(interval))

    def on_step_end(
        self, trainer: RLTrainer, step: int, metrics: dict[str, float]
    ) -> None:
        if step % self.interval != 0:
            return
        val_metrics = trainer.run_validation()
        if trainer._rank == 0:
            parts = [f"{k}={v:.6g}" for k, v in sorted(val_metrics.items())]
            logger.info("eval step=%d %s", step + 1, " ".join(parts))
        metrics.update(val_metrics)


class WandbCallback(Callback):
    """Optional Weights & Biases logging."""

    def __init__(self, project: str, name: str = "", entity: str | None = None) -> None:
        self.project = project
        self.name = name
        self.entity = entity
        self._wandb: Any = None
        self._enabled = False

    def on_train_begin(self, trainer: RLTrainer) -> None:
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed; WandbCallback disabled.")
            return
        self._wandb = wandb
        run_name = self.name or f"atomrl-{trainer.config.algorithm.name}"
        wandb.init(project=self.project, name=run_name, entity=self.entity)
        self._enabled = True
        wandb.config.update(
            {
                "algorithm": trainer.config.algorithm.name,
                "num_training_steps": int(trainer.config.num_training_steps),
            },
            allow_val_change=True,
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step")

    # Curated "core" training-effect metrics (means only, no max/min) shown in a
    # dedicated wandb `core/` panel group.
    _CORE_MAP: typing.ClassVar = {
        "reward/mean": "core/reward_mean",
        "seq/mean_response_len": "core/response_len_mean",
        "response_length/mean": "core/response_len_mean",
        "timing/step_s": "core/step_time_s",
        "timing/gen_s": "core/gen_time_s",
        "timing/train_s": "core/train_time_s",
        "rollout_correction/kl": "core/kl",
        "rollout_corr/kl": "core/kl",
        "mismatch_kl": "core/mismatch_kl",
        # core/kl is SIGNED, so symmetric train/rollout disagreement cancels in it.
        # These two do not cancel and are what actually track the gap.
        "mismatch/abs_diff": "core/mismatch_abs_diff",
        "mismatch/k3_kl": "core/mismatch_k3_kl",
        "moe/r3_enabled": "core/r3_enabled",
        "moe/r3_route_coverage": "core/r3_route_coverage",
        "moe/r3_route_tokens": "core/r3_route_tokens",
        "entropy": "core/entropy",
        "grad_norm": "core/grad_norm",
        "loss": "core/loss",
        "mem/actor_max_reserved_gb": "core/max_reserved_mem_gb",
    }
    _VERL_ALIAS_MAP: typing.ClassVar = {
        "loss": ("actor/loss", "actor/pg_loss"),
        "lr": ("actor/lr",),
        "grad_norm": ("actor/grad_norm",),
        "entropy": ("actor/entropy",),
        "ppo_kl": ("actor/ppo_kl",),
        "mem/actor_max_allocated_gb": ("actor/perf/max_memory_allocated_gb",),
        "mem/actor_max_reserved_gb": (
            "actor/perf/max_memory_reserved_gb",
            "actor/perf/micro_batch_max_reserved_gb",
        ),
    }

    def on_step_end(
        self, trainer: RLTrainer, step: int, metrics: dict[str, float]
    ) -> None:
        if not self._enabled or self._wandb is None:
            return
        if trainer._rank != 0:
            return
        # Log verl-compatible names at the root so the AMD-BF16-VERL workspace
        # panels can be recreated verbatim. Keep the historical train/* aliases
        # for existing AtomRL dashboards.
        payload = dict(metrics)
        payload.update(
            {
                f"train/{k}": v
                for k, v in metrics.items()
                if not k.startswith(("train/", "val-", "val/"))
            }
        )
        for src, destinations in self._VERL_ALIAS_MAP.items():
            if src in metrics:
                for dst in destinations:
                    payload.setdefault(dst, metrics[src])
        for src, dst in self._CORE_MAP.items():
            if src in metrics and dst not in payload:
                payload[dst] = metrics[src]

        val_generations = getattr(trainer, "_last_val_generations", None)
        if val_generations and any(k.startswith("val-") for k in metrics):
            payload["val/generations"] = self._wandb.Table(
                columns=["response", "score", "accuracy", "response_length"],
                data=[
                    [
                        row["response"],
                        row["score"],
                        row["accuracy"],
                        row["response_length"],
                    ]
                    for row in val_generations
                ],
            )
        # 1-based step to align the wandb x-axis with verl (global_steps).
        wstep = step + 1
        payload["train/global_step"] = wstep
        self._wandb.log(payload, step=wstep, commit=True)

    def on_train_end(self, trainer: RLTrainer) -> None:
        if self._enabled and self._wandb is not None:
            self._wandb.finish()
