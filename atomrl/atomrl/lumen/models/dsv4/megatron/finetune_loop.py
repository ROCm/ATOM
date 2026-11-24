# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Native torchrun GRPO finetune loop for DSV4 (debug-train-only, no Ray)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training import get_args, get_timers

from atomrl.lumen.models.dsv4.megatron.grpo import (
    RolloutDataIterator,
    build_bshd_rollout_batch,
    compute_grpo_advantages,
    grpo_policy_loss,
    load_fake_rollout,
)
from atomrl.lumen.models.utils import safe_add_argument

logger = logging.getLogger(__name__)

_ROLLOUT_STATE: dict[str, Any] = {}


def _is_grpo_log_rank() -> bool:
    """Primary logging rank for GRPO metrics (PP last, TP0, DP0, CP0)."""
    return (
        mpu.is_pipeline_last_stage(ignore_virtual=True)
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_data_parallel_rank() == 0
        and mpu.get_context_parallel_rank() == 0
    )


def add_dsv4_finetune_args(parser):
    group = parser.add_argument_group(title="dsv4 finetune")
    safe_add_argument(
        group,
        "--rollout-data-path",
        type=str,
        default=None,
        help="Path to fake_rollout.pt (torch.save with samples list).",
    )
    safe_add_argument(
        group,
        "--num-rollout",
        type=int,
        default=10,
        help="Number of GRPO training steps (reuse same rollout batch).",
    )
    safe_add_argument(
        group,
        "--grpo-eps-clip",
        type=float,
        default=0.2,
        help="PPO/GRPO clip epsilon (lower).",
    )
    safe_add_argument(
        group,
        "--grpo-eps-clip-high",
        type=float,
        default=0.28,
        help="PPO/GRPO clip epsilon (upper).",
    )
    safe_add_argument(
        group,
        "--data-pad-size-multiplier",
        type=int,
        default=128,
        help="Pad seq len to TP*multiplier (default 128).",
    )
    return parser


def _prepare_rollout_state(args) -> None:
    path = args.rollout_data_path or os.environ.get(
        "LUMEN_DSV4_ROLLOUT_PATH", "/root/models/fake_rollout.pt"
    )
    device = torch.device("cuda", torch.cuda.current_device())
    samples = load_fake_rollout(path)
    advantages = compute_grpo_advantages(samples)
    pad_multiplier = args.tensor_model_parallel_size * args.data_pad_size_multiplier
    rollout = build_bshd_rollout_batch(
        samples,
        advantages,
        pad_multiplier=pad_multiplier,
        device=device,
        min_seq_len=args.seq_length,
    )
    num_samples = len(samples)
    if args.global_batch_size != num_samples:
        raise ValueError(
            f"global_batch_size={args.global_batch_size} must equal rollout sample count "
            f"({num_samples}); set GBS={num_samples} in the launcher."
        )

    _ROLLOUT_STATE.clear()
    _ROLLOUT_STATE.update(
        {
            "rollout": rollout,
            "num_samples": num_samples,
            "path": path,
            "max_seq_len": rollout["max_seq_lens"][0],
        }
    )
    if torch.distributed.get_rank() == 0:
        print(
            f"[dsv4_finetune] loaded {num_samples} rollout samples from {path}, "
            f"max_seq_len={rollout['max_seq_lens'][0]}",
            flush=True,
        )


def _aggregate_train_losses(losses_reduced: list[dict[str, Any]]) -> dict[str, float]:
    """DP reduction over micro-batch loss metrics."""
    if not losses_reduced:
        return {}

    keys = losses_reduced[0]["keys"]
    values: torch.Tensor | None = None
    for log_dict in losses_reduced:
        if "keys" not in log_dict or "values" not in log_dict:
            continue
        batch_values = log_dict["values"]
        if not torch.is_tensor(batch_values):
            continue
        values = batch_values.clone() if values is None else values + batch_values

    if values is None:
        return {}

    if dist.is_initialized():
        dist.all_reduce(
            values, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group()
        )

    vals = values.detach().float().cpu().tolist()
    num_samples_or_tokens = vals[0]
    cp_size = mpu.get_context_parallel_world_size()
    return {
        key: float(val) * cp_size / num_samples_or_tokens
        for key, val in zip(keys, vals[1:], strict=False)
    }


def _log_rollout_summary(rollout_id: int, rollout: dict[str, Any]) -> None:
    """Log ``rollout {id}: {...}`` summary (debug-train-only batch)."""
    advantages = rollout.get("advantages") or []
    rollout_log_probs = rollout.get("rollout_log_probs") or []
    adv_vals = [float(a.mean()) for a in advantages if a.numel()]
    lp_vals = [float(lp.mean()) for lp in rollout_log_probs if lp.numel()]
    log_dict = {
        "rollout/num_samples": len(rollout.get("tokens") or []),
        "rollout/max_seq_len": rollout.get("max_seq_lens", [0])[0],
    }
    if adv_vals:
        log_dict["rollout/advantages"] = sum(adv_vals) / len(adv_vals)
    if lp_vals:
        log_dict["rollout/rollout_log_probs"] = sum(lp_vals) / len(lp_vals)
    logger.info("rollout %s: %s", rollout_id, log_dict)


def _log_train_step(
    rollout_id: int,
    step_id: int,
    num_steps_per_rollout: int,
    loss_dict: dict[str, float],
    grad_norm: float,
    opt_param_scheduler,
    optimizer,
) -> None:
    """Log ``step {accumulated_step_id}: {'train/...': ...}``."""
    accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
    log_dict = {f"train/{key}": val for key, val in loss_dict.items()}
    log_dict["train/grad_norm"] = float(grad_norm)
    for param_group_id, param_group in enumerate(optimizer.param_groups):
        log_dict[f"train/lr-pg_{param_group_id}"] = float(
            opt_param_scheduler.get_lr(param_group)
        )
    log_dict["train/step"] = accumulated_step_id
    logger.info("step %s: %s", accumulated_step_id, log_dict)


def _log_perf_data(
    rollout_id: int, actor_train_time: float, total_lengths: list[int]
) -> None:
    """Log ``perf {rollout_id}: {'perf/actor_train_time': ...}``."""
    log_dict: dict[str, float] = {"perf/actor_train_time": actor_train_time}
    if actor_train_time > 0:
        log_dict["perf/actor_train_tok_per_s"] = sum(total_lengths) / actor_train_time
    logger.info("perf %s: %s", rollout_id, log_dict)


def dsv4_grpo_forward_step(data_iterator, model, return_schedule_plan: bool = False):
    from pretrain_gpt import stimer

    args = get_args()
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        batch = data_iterator.get_next(
            [
                "tokens",
                "unconcat_tokens",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "advantages",
                "rollout_log_probs",
                "max_seq_lens",
                "full_loss_masks",
            ]
        )
    timers("batch-generator").stop()

    with stimer:
        if return_schedule_plan:
            raise RuntimeError("schedule_plan not supported for GRPO finetune")
        output_tensor = model(
            batch["tokens"],
            None,
            None,
            labels=None,
            loss_mask=batch["full_loss_masks"],
        )

    def loss_func(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        loss, log = grpo_policy_loss(
            logits,
            batch,
            eps_clip=args.grpo_eps_clip,
            eps_clip_high=args.grpo_eps_clip_high,
        )
        num_samples = len(batch["response_lengths"])
        num_microbatches = max(
            1,
            args.global_batch_size
            // (args.micro_batch_size * mpu.get_data_parallel_world_size()),
        )
        loss_parallel_size = mpu.get_data_parallel_world_size()
        scaled = loss * num_microbatches / args.global_batch_size * loss_parallel_size
        return (
            scaled,
            torch.tensor(1, device=logits.device),
            {
                "keys": list(log.keys()),
                "values": torch.tensor(
                    [num_samples] + [float(v) for v in log.values()],
                    device=logits.device,
                ),
            },
        )

    return output_tensor, loss_func


def _build_data_iterator(args) -> RolloutDataIterator:
    return RolloutDataIterator(_ROLLOUT_STATE["rollout"], args.micro_batch_size)


def run_dsv4_grpo_finetune(
    model_provider,
    extra_args_provider=None,
) -> None:
    from megatron.training import pretrain
    from pretrain_gpt import train_valid_test_datasets_provider

    def _finetune_train(
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
        process_non_loss_data_func,
        config,
        checkpointing_context,
        non_loss_data_func,
    ):
        args = get_args()
        _prepare_rollout_state(args)

        forward_backward_func = get_forward_backward_func()
        num_microbatches = args.global_batch_size // (
            args.micro_batch_size * mpu.get_data_parallel_world_size()
        )
        assert (
            num_microbatches
            * args.micro_batch_size
            * mpu.get_data_parallel_world_size()
            == args.global_batch_size
        )

        num_steps_per_rollout = 1
        rollout = _ROLLOUT_STATE["rollout"]
        total_lengths = rollout["total_lengths"]

        for rollout_id in range(args.num_rollout):
            iterator = _build_data_iterator(args)
            iterator.reset()

            if _is_grpo_log_rank():
                _log_rollout_summary(rollout_id, rollout)

            for model_module in model:
                model_module.zero_grad_buffer()
            optimizer.zero_grad()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            train_start = time.perf_counter()

            get_model_config(model[0])
            train_seq_len = _ROLLOUT_STATE["max_seq_len"]
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=train_seq_len,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=train_seq_len,
                forward_only=False,
            )

            update_successful, grad_norm, _ = optimizer.step()
            if update_successful:
                opt_param_scheduler.step(increment=1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            actor_train_time = time.perf_counter() - train_start

            if _is_grpo_log_rank() and update_successful:
                loss_dict = _aggregate_train_losses(losses_reduced)
                _log_train_step(
                    rollout_id,
                    step_id=0,
                    num_steps_per_rollout=num_steps_per_rollout,
                    loss_dict=loss_dict,
                    grad_norm=float(grad_norm),
                    opt_param_scheduler=opt_param_scheduler,
                    optimizer=optimizer,
                )
                _log_perf_data(rollout_id, actor_train_time, total_lengths)

        return args.num_rollout, args.num_floating_point_operations_so_far

    import megatron.training.training as training_module

    _orig = training_module.train
    training_module.train = _finetune_train

    try:
        train_valid_test_datasets_provider.is_distributed = True

        pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            dsv4_grpo_forward_step,
            extra_args_provider=extra_args_provider,
            args_defaults={"tokenizer_type": "NullTokenizer"},
        )
    finally:
        training_module.train = _orig
