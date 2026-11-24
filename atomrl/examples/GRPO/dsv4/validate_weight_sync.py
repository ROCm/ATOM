# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Isolated Megatron-to-vLLM RDMA weight synchronization validation.

This initializes the training and rollout models but performs no RL rollout loop,
backward pass, or optimizer step. It generates a small fixed validation batch
twice before synchronization, then after two identical weight synchronizations.
The repeated captures distinguish normal generation nondeterminism from changes
introduced by the synchronization path.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any


def compare_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, int | float | None]:
    """Compare generated token IDs and aligned rollout log-probabilities."""
    before_ids = before["token_ids"]
    after_ids = after["token_ids"]
    if len(before_ids) != len(after_ids):
        raise ValueError("snapshot batch sizes differ")

    exact = 0
    first_mismatch: int | None = None
    lp_abs_sum = 0.0
    lp_count = 0
    for idx, (lhs, rhs) in enumerate(zip(before_ids, after_ids)):
        if lhs != rhs:
            if first_mismatch is None:
                first_mismatch = idx
            continue
        exact += 1
        lhs_lp = before["logprobs"][idx]
        rhs_lp = after["logprobs"][idx]
        if len(lhs_lp) != len(rhs_lp):
            raise ValueError(f"logprob lengths differ for matching sample {idx}")
        lp_abs_sum += sum(abs(float(a) - float(b)) for a, b in zip(lhs_lp, rhs_lp))
        lp_count += len(lhs_lp)

    return {
        "exact_token_matches": exact,
        "total": len(before_ids),
        "first_token_mismatch": first_mismatch,
        "matching_logprob_mae": (lp_abs_sum / lp_count) if lp_count else None,
    }


def compare_cross_engine_logprobs(
    rollout_logprobs: Any,
    megatron_logprobs: Any,
    response_mask: Any,
) -> dict[str, int | float]:
    """Compare vLLM and Megatron log-probs on response tokens only."""
    if rollout_logprobs.shape != megatron_logprobs.shape:
        raise ValueError(
            "cross-engine log-prob shapes differ: "
            f"{tuple(rollout_logprobs.shape)} != {tuple(megatron_logprobs.shape)}"
        )
    if response_mask.shape != rollout_logprobs.shape:
        raise ValueError(
            "response mask shape differs from log-probs: "
            f"{tuple(response_mask.shape)} != {tuple(rollout_logprobs.shape)}"
        )
    selected_rollout = rollout_logprobs.float()[response_mask.bool()]
    selected_megatron = megatron_logprobs.float()[response_mask.bool()]
    if selected_rollout.numel() == 0:
        raise ValueError("cross-engine comparison has no response tokens")
    diff = selected_rollout - selected_megatron
    return {
        "token_count": int(diff.numel()),
        "rollout_minus_megatron_mean": float(diff.mean().item()),
        "logprob_mae": float(diff.abs().mean().item()),
        "logprob_max_abs": float(diff.abs().max().item()),
        "rollout_nll": float((-selected_rollout).mean().item()),
        "megatron_nll": float((-selected_megatron).mean().item()),
    }


def _capture(
    trainer: Any,
    prompts: list[str],
    ground_truths: list[str],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    from atomrl.rewards.math_reward import compute_math_reward

    sequences, seq_mask, prompt_lengths, rollout_lp, _ = trainer._rollout_with_ray_vllm(
        prompts,
        num_generations=1,
        sampling_params={
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": 0,
            "seed": 0,
        },
    )
    if rollout_lp is None:
        raise RuntimeError("vLLM returned no rollout log-probabilities")

    sequences = sequences.cpu()
    seq_mask = seq_mask.cpu()
    rollout_lp = rollout_lp.float().cpu()
    response_mask = (
        trainer._build_response_mask(
            sequences,
            seq_mask,
            prompt_lengths,
        )
        .bool()
        .cpu()
    )

    token_ids: list[list[int]] = []
    logprobs: list[list[float]] = []
    responses: list[str] = []
    for idx, prompt_len in enumerate(prompt_lengths):
        real_ids = sequences[idx][seq_mask[idx].bool()].tolist()
        response_ids = [int(x) for x in real_ids[int(prompt_len) :]]
        token_ids.append(response_ids)
        responses.append(
            trainer._tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        logprobs.append(
            [float(x) for x in rollout_lp[idx][response_mask[idx]].tolist()]
        )

    rewards, details = compute_math_reward(responses, ground_truths)
    return {
        "token_ids": token_ids,
        "logprobs": logprobs,
        "responses": responses,
        "scores": [float(x) for x in rewards.tolist()],
        "accuracy": [1 if item["acc"] else 0 for item in details],
        "response_lengths": [len(x) for x in token_ids],
        "_sequences": sequences,
        "_seq_mask": seq_mask,
        "_response_mask": response_mask,
        "_rollout_logprobs": rollout_lp,
    }


def _cross_engine_snapshot(
    trainer: Any, snapshot: dict[str, Any]
) -> dict[str, int | float]:
    if trainer._actor_wg is None:
        raise RuntimeError("Megatron actor worker group is unavailable")
    megatron_logprobs = trainer._compute_log_probs_with_worker_group(
        trainer._actor_wg,
        snapshot["_sequences"],
        role="actor",
        attention_mask=snapshot["_seq_mask"],
    )
    return compare_cross_engine_logprobs(
        snapshot["_rollout_logprobs"].cpu(),
        megatron_logprobs.cpu(),
        snapshot["_response_mask"].cpu(),
    )


def _stage_summary(snapshot: dict[str, Any]) -> dict[str, float | int]:
    lengths = snapshot["response_lengths"]
    scores = snapshot["scores"]
    return {
        "samples": len(lengths),
        "accuracy": sum(snapshot["accuracy"]) / max(1, len(lengths)),
        "score_mean": sum(scores) / max(1, len(scores)),
        "response_length_mean": sum(lengths) / max(1, len(lengths)),
        "response_length_max": max(lengths, default=0),
    }


def _skip_sync_requested() -> bool:
    return os.environ.get("ATOMRL_SYNC_VALIDATE_SKIP_SYNC", "0") == "1"


def main() -> None:
    from atomrl.core.config import AtomRLConfig
    from atomrl.trainer.main import _setup_logging
    from atomrl.trainer.rl_trainer import RLTrainer

    _setup_logging()
    logger = logging.getLogger("atomrl.validate_weight_sync")
    config = AtomRLConfig.from_cli()
    config.checkpointing.resume = False
    config.eval.enabled = False
    config.logger.wandb_enabled = False
    config.moe.r3.enabled = False
    config.policy.generation.vllm_cfg.calculate_log_probs = True

    sample_count = max(1, int(os.environ.get("ATOMRL_SYNC_VALIDATE_SAMPLES", "4")))
    max_tokens = max(1, int(os.environ.get("ATOMRL_SYNC_VALIDATE_MAX_TOKENS", "512")))
    temperature = float(config.policy.generation.vllm_cfg.temperature or 1.0)
    trainer = RLTrainer(config)
    try:
        trainer.setup()
        if trainer._val_dataset is None:
            raise RuntimeError("validation dataset is unavailable")
        count = min(sample_count, len(trainer._val_dataset))
        samples = [trainer._val_dataset[idx] for idx in range(count)]
        pairs = [trainer._extract_prompt_gt(sample) for sample in samples]
        prompts = [pair[0] for pair in pairs]
        ground_truths = [pair[1] for pair in pairs]

        logger.info("Capturing base model twice before weight synchronization")
        base_1 = _capture(trainer, prompts, ground_truths, max_tokens, temperature)
        base_2 = _capture(trainer, prompts, ground_truths, max_tokens, temperature)
        base_cross_engine = _cross_engine_snapshot(trainer, base_2)
        if _skip_sync_requested():
            result = {
                "base_1": _stage_summary(base_1),
                "base_2": _stage_summary(base_2),
                "base_repeat": compare_snapshots(base_1, base_2),
                "base_cross_engine": base_cross_engine,
                "response_tails": {
                    "base": [text[-240:] for text in base_2["responses"]],
                },
            }
            logger.info(
                "WEIGHT_SYNC_VALIDATION_JSON=%s",
                json.dumps(result, ensure_ascii=False),
            )
            return

        logger.info("Synchronizing unchanged trainer weights (version 1)")
        trainer.global_step = 0
        trainer._sync_weights_ipc()
        sync_1 = _capture(trainer, prompts, ground_truths, max_tokens, temperature)
        sync_cross_engine = _cross_engine_snapshot(trainer, sync_1)

        logger.info("Synchronizing the same unchanged trainer weights (version 2)")
        trainer.global_step = 1
        trainer._sync_weights_ipc()
        sync_2 = _capture(trainer, prompts, ground_truths, max_tokens, temperature)

        result = {
            "base_1": _stage_summary(base_1),
            "base_2": _stage_summary(base_2),
            "sync_1": _stage_summary(sync_1),
            "sync_2": _stage_summary(sync_2),
            "base_repeat": compare_snapshots(base_1, base_2),
            "base_to_sync": compare_snapshots(base_2, sync_1),
            "sync_repeat": compare_snapshots(sync_1, sync_2),
            "base_cross_engine": base_cross_engine,
            "sync_cross_engine": sync_cross_engine,
            "response_tails": {
                "base": [text[-240:] for text in base_2["responses"]],
                "sync": [text[-240:] for text in sync_1["responses"]],
            },
        }
        logger.info(
            "WEIGHT_SYNC_VALIDATION_JSON=%s", json.dumps(result, ensure_ascii=False)
        )
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
