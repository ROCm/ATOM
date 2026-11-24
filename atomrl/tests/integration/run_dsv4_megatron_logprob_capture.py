# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Capture fixed-token DSV4 Megatron log-probs without rollout or training.

Run this script twice with topology-specific configs, then use ``--compare`` to
compare the saved artifacts offline. The capture path initializes only the
Megatron actor worker group and calls ``compute_log_probs`` once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def build_fixed_batch(
    sequence_length: int,
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if response_length <= 0 or response_length >= sequence_length:
        raise ValueError("response_length must be in [1, sequence_length)")
    input_ids = torch.arange(
        100,
        100 + sequence_length,
        dtype=torch.long,
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    response_mask = torch.zeros(
        (1, sequence_length - 1),
        dtype=torch.bool,
    )
    response_mask[:, -response_length:] = True
    return input_ids, attention_mask, response_mask


def build_batch_from_ids(
    token_ids: list[int],
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(token_ids) < 2:
        raise ValueError("token_ids must contain at least two tokens")
    if response_length <= 0 or response_length >= len(token_ids):
        raise ValueError("response_length must be in [1, len(token_ids))")
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    response_mask = torch.zeros(
        (1, len(token_ids) - 1),
        dtype=torch.bool,
    )
    response_mask[:, -response_length:] = True
    return input_ids, attention_mask, response_mask


def compare_artifacts(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, int | float]:
    baseline_mask = torch.tensor(baseline["response_mask"], dtype=torch.bool)
    candidate_mask = torch.tensor(candidate["response_mask"], dtype=torch.bool)
    if not torch.equal(baseline_mask, candidate_mask):
        raise ValueError("response masks differ")
    baseline_lp = torch.tensor(baseline["log_probs"], dtype=torch.float32)
    candidate_lp = torch.tensor(candidate["log_probs"], dtype=torch.float32)
    if baseline_lp.shape != candidate_lp.shape:
        raise ValueError("log-prob shapes differ")
    if baseline_lp.shape != baseline_mask.shape:
        raise ValueError("log-prob and response-mask shapes differ")
    selected = baseline_mask.bool()
    if not selected.any():
        raise ValueError("comparison has no response tokens")
    diff = baseline_lp[selected] - candidate_lp[selected]
    return {
        "token_count": int(diff.numel()),
        "mean_delta": float(diff.mean().item()),
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "baseline_nll": float((-baseline_lp[selected]).mean().item()),
        "candidate_nll": float((-candidate_lp[selected]).mean().item()),
    }


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--input-ids")
    parser.add_argument("--rollout-routing")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--response-length", type=int, default=32)
    parser.add_argument("--update-policy-smoke", action="store_true")
    return parser.parse_known_args()


def _capture(args: argparse.Namespace, config_args: list[str]) -> dict[str, Any]:
    from atomrl.core.config import AtomRLConfig
    from atomrl.trainer.main import _setup_logging
    from atomrl.trainer.rl_trainer import RLTrainer

    if not args.output:
        raise ValueError("--output is required for capture mode")
    sys.argv = [sys.argv[0], *config_args]
    os.environ["ATOMRL_FORWARD_ONLY_INIT"] = "0" if args.update_policy_smoke else "1"
    os.environ["ATOMRL_SKIP_ROLLOUT_INIT"] = "1"
    _setup_logging()
    config = AtomRLConfig.from_cli()
    config.checkpointing.resume = False
    config.eval.enabled = False
    config.logger.wandb_enabled = False
    config.moe.r3.enabled = bool(args.rollout_routing)
    config.weight_sync.enabled = False
    config.controller.ray.rollout.num_workers = 0
    config.controller.ray.rollout.process_on_nodes = []

    if args.input_ids:
        token_ids = json.loads(Path(args.input_ids).read_text())
        input_ids, attention_mask, response_mask = build_batch_from_ids(
            token_ids,
            args.response_length,
        )
    else:
        input_ids, attention_mask, response_mask = build_fixed_batch(
            args.sequence_length,
            args.response_length,
        )
    trainer = RLTrainer(config)
    try:
        trainer.setup()
        if trainer._actor_wg is None:
            raise RuntimeError("Megatron actor worker group is unavailable")
        rollout_routing = None
        if args.rollout_routing:
            import numpy as np

            routes = np.load(args.rollout_routing)
            expected_shape = (input_ids.shape[1] - 1, 43, 6)
            if routes.shape != expected_shape:
                raise ValueError(
                    f"rollout routing has shape {routes.shape}, expected {expected_shape}"
                )
            rollout_routing = [routes]
        with torch.no_grad():
            log_probs = (
                trainer._compute_log_probs_with_worker_group(
                    trainer._actor_wg,
                    input_ids,
                    role="actor",
                    attention_mask=attention_mask,
                    rollout_routing=rollout_routing,
                )
                .detach()
                .float()
                .cpu()
            )
        update_metrics = None
        if args.update_policy_smoke:
            from atomrl.core.protocol import DataProto

            policy_batch = DataProto(
                tensors={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "old_log_probs": log_probs,
                    "ref_log_probs": torch.zeros_like(log_probs),
                    "rewards": torch.ones(input_ids.shape[0]),
                    "response_mask": response_mask,
                    "advantages": torch.ones(input_ids.shape[0]),
                },
                meta={
                    "algorithm": config.algorithm.name,
                    "algo_config": trainer._to_plain_dict(config.algorithm),
                    "temperature": float(
                        config.policy.generation.vllm_cfg.temperature or 1.0
                    ),
                    "max_token_len_per_gpu": int(config.policy.max_token_len_per_gpu),
                    "batch_num_tokens": int(response_mask.sum().item()),
                    "dp_size": int(
                        trainer._actor_dp_size
                        or (trainer._actor_wg.num_workers // max(1, trainer._actor_mp))
                    ),
                    "global_batch_size": int(input_ids.shape[0]),
                },
            )
            update_metrics = trainer._update_actor_with_ray(policy_batch)
        artifact = {
            "input_ids": input_ids[0].tolist(),
            "attention_mask": attention_mask[0].tolist(),
            "response_mask": response_mask[0].tolist(),
            "log_probs": log_probs[0].tolist(),
            "tensor_model_parallel_size": int(
                config.policy.training.megatron_cfg.tensor_model_parallel_size
            ),
            "pipeline_model_parallel_size": int(
                config.policy.training.megatron_cfg.pipeline_model_parallel_size
            ),
            "expert_model_parallel_size": int(
                config.policy.training.megatron_cfg.expert_model_parallel_size
            ),
            "update_metrics": update_metrics,
        }
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(artifact))
        return artifact
    finally:
        trainer.cleanup()


def main() -> None:
    args, config_args = _parse_args()
    if args.compare:
        baseline = json.loads(Path(args.compare[0]).read_text())
        candidate = json.loads(Path(args.compare[1]).read_text())
        print(
            "PP_LOGPROB_PARITY_JSON="
            + json.dumps(compare_artifacts(baseline, candidate)),
            flush=True,
        )
        return
    artifact = _capture(args, config_args)
    selected = torch.tensor(artifact["response_mask"], dtype=torch.bool)
    log_probs = torch.tensor(artifact["log_probs"], dtype=torch.float32)
    result = {
        "output": args.output,
        "response_tokens": int(selected.sum().item()),
        "nll": float((-log_probs[selected]).mean().item()),
        "finite": bool(torch.isfinite(log_probs[selected]).all().item()),
    }
    print("PP_LOGPROB_CAPTURE_JSON=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
