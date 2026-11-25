# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CLI entry point for AtomRL RL training.

Usage:
    python -m atomrl.trainer.main --config path/to/config.yaml [overrides...]

    torchrun --nproc_per_node=8 -m atomrl.trainer.main --config config.yaml
"""

from __future__ import annotations

import logging
import os
import sys

import torch


class _FlushHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit (unbuffered logging)."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _setup_logging() -> None:
    level = os.environ.get("ATOMRL_LOG_LEVEL", "INFO").upper()
    handler = _FlushHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _setup_distributed(use_ray_controller: bool) -> None:
    """Initialize torch.distributed: torchrun's group, or one of a single rank.

    A single process started without torchrun still gets a one-rank group when it
    has a GPU. Without a group FSDP2 leaves the model unsharded, so it skips the
    mixed-precision policy and computes in the fp32 of the master weights, which
    aiter's varlen flash attention rejects ("Unsupported dtype for MHA").
    """
    if use_ray_controller:
        logging.getLogger(__name__).info(
            "Skipping torch.distributed init (Ray controller mode)."
        )
        return
    if "RANK" not in os.environ and torch.cuda.is_available():
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        from datetime import timedelta

        timeout = timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT", "7200")))
        torch.distributed.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            timeout=timeout,
        )
        rank = torch.distributed.get_rank()
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)


def main() -> None:
    _setup_logging()

    from atomrl.architecture.assembly.runtime_assembler import RuntimeAssembler
    from atomrl.core.config import AtomRLConfig
    from atomrl.core.types import AlgorithmName
    from atomrl.trainer.callbacks import (
        CheckpointCallback,
        EvalCallback,
        LoggingCallback,
        WandbCallback,
    )

    config = AtomRLConfig.from_cli()
    use_ray_controller = (
        bool(config.controller.ray.enabled)
        or os.environ.get("ATOMRL_USE_RAY_CONTROLLER", "0") == "1"
    )
    algo_name = config.algorithm.name.lower()
    # Only pure RLTrainer ray-controller path replaces torch.distributed init.
    # Other trainers may enable RayCluster lifecycle but still rely on torch.distributed.
    skip_torch_distributed = (
        use_ray_controller
        and (not config.async_training.enabled)
        and (algo_name in {"grpo", "dapo", "ppo"})
    )
    _setup_distributed(use_ray_controller=skip_torch_distributed)
    logger = logging.getLogger("atomrl.main")
    logger.info(
        "AtomRL starting: algo=%s, model=%s, steps=%d, async=%s",
        config.algorithm.name,
        config.policy.model_name,
        config.num_training_steps,
        config.async_training.enabled,
    )

    algo_name = config.algorithm.name.lower()
    if config.assembly.use_new_assembler:
        assembler = RuntimeAssembler(config)
        graph = assembler.build_graph()
        trainer = graph["trainer_cls"](config)
    else:
        from atomrl.trainer.async_trainer import AsyncRLTrainer
        from atomrl.trainer.rl_trainer import RLTrainer

        if algo_name == AlgorithmName.OPD.value:
            from atomrl.trainer.opd_trainer import OPDTrainer

            trainer = OPDTrainer(config)
        elif algo_name == AlgorithmName.SPEC_DISTILL.value:
            from atomrl.trainer.spec_distill_trainer import SpecDistillTrainer

            trainer = SpecDistillTrainer(config)
        elif config.async_training.enabled:
            trainer = AsyncRLTrainer(config)
        else:
            trainer = RLTrainer(config)

    cbs: list = [LoggingCallback(interval=max(1, config.logger.log_interval))]

    if config.checkpointing.checkpoint_dir:
        cbs.append(
            CheckpointCallback(
                checkpoint_dir=config.checkpointing.checkpoint_dir,
                save_interval=config.checkpointing.save_steps,
                save_total_limit=config.checkpointing.save_total_limit,
            )
        )

    if config.eval.enabled:
        cbs.append(EvalCallback(interval=config.eval.interval))

    if config.logger.wandb_enabled:
        cbs.append(
            WandbCallback(
                project=config.logger.wandb.project,
                name=config.logger.wandb.name or None,
                entity=config.logger.wandb.entity or None,
            )
        )

    trainer.callbacks = cbs

    try:
        trainer.setup()
        _replay = os.environ.get("LUMEN_REPLAY_DUMP")
        if _replay:
            trainer.replay_compare(_replay)
            return
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception:
        logger.exception("Training failed.")
        sys.exit(1)
    finally:
        trainer.cleanup()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    logger.info("AtomRL finished.")


if __name__ == "__main__":
    main()
