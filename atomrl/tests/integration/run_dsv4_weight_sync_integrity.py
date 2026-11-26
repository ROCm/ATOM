# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Run one DSV4 weight sync with no rollout, reward, or training step."""

from __future__ import annotations

import json
import math
import os
from typing import Any


def summarize_replica_reports(
    reports: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    first_bad = None
    worker_count = 0
    for replica_rank, replica in enumerate(reports):
        for worker_rank, report in enumerate(replica):
            worker_count += 1
            if not report.get("all_finite", False) and first_bad is None:
                first_bad = {
                    "replica_rank": replica_rank,
                    "worker_rank": worker_rank,
                    "report": report,
                }
    return {
        "replica_count": len(reports),
        "worker_count": worker_count,
        "all_finite": first_bad is None,
        "first_bad": first_bad,
    }


def _inspect_replicas(
    manager: Any,
    method: str = "inspect_weight_integrity",
) -> list[list[dict[str, Any]]]:
    import ray

    return ray.get([server.collective_rpc.remote(method) for server in manager.servers])


def _one_token_probe(manager: Any) -> dict[str, Any]:
    import ray

    result = ray.get(
        manager.servers[0].generate.remote(
            "Question: What is 12 times 13? Answer:",
            {
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 0,
                "seed": 1,
            },
        )
    )
    logprobs = result.get("logprobs") or []
    return {
        "token_ids": result.get("token_ids"),
        "logprobs": logprobs,
        "all_finite": bool(logprobs)
        and all(math.isfinite(value) for value in logprobs),
    }


def configure_diagnostic(config: Any) -> int:
    """Apply diagnostic-only settings and return the requested sync count."""
    repeat_raw = os.environ.get("ATOMRL_WEIGHT_SYNC_REPEAT", "1")
    try:
        repeat_count = int(repeat_raw)
    except ValueError as exc:
        raise ValueError(
            "ATOMRL_WEIGHT_SYNC_REPEAT must be a positive integer"
        ) from exc
    if repeat_count <= 0:
        raise ValueError("ATOMRL_WEIGHT_SYNC_REPEAT must be a positive integer")
    config.checkpointing.resume = (
        os.environ.get("ATOMRL_WEIGHT_SYNC_RESUME", "0") == "1"
    )
    config.eval.enabled = False
    config.logger.wandb_enabled = False
    config.moe.r3.enabled = False
    return repeat_count


def main() -> None:
    from atomrl.core.config import AtomRLConfig
    from atomrl.trainer.main import _setup_logging
    from atomrl.trainer.rl_trainer import RLTrainer

    os.environ["ATOMRL_WEIGHT_SYNC_INTEGRITY"] = "1"
    os.environ["ATOMRL_FORWARD_ONLY_INIT"] = "1"
    os.environ["ATOMRL_SKIP_DATASET_INIT"] = "1"
    _setup_logging()
    config = AtomRLConfig.from_cli()
    repeat_count = configure_diagnostic(config)
    if not config.weight_sync.enabled:
        raise ValueError("weight_sync.enabled must be true")
    if str(config.weight_sync.backend) != "rdma":
        raise ValueError("weight_sync.backend must be rdma")

    trainer = RLTrainer(config)
    try:
        trainer.setup()
        manager = trainer._ray_rollout_mgr
        if manager is None or trainer._actor_wg is None:
            raise RuntimeError("actor or vLLM replica manager is unavailable")
        before = _inspect_replicas(manager)
        scales_before = _inspect_replicas(manager, "inspect_fp8_scales")
        probe_before = _one_token_probe(manager)
        rounds = []
        for sync_index in range(repeat_count):
            trainer.global_step = int(trainer._resume_step) + sync_index
            trainer._sync_weights_rdma(manager)
            after = _inspect_replicas(manager)
            scales_after = _inspect_replicas(manager, "inspect_fp8_scales")
            probe_after = _one_token_probe(manager)
            round_result = {
                "sync_index": sync_index,
                "global_step": trainer.global_step,
                "after": summarize_replica_reports(after),
                "scales_after": scales_after,
                "probe_after": probe_after,
                "receiver_phases": getattr(
                    trainer,
                    "_last_weight_sync_integrity",
                    None,
                ),
                "metrics": trainer._last_weight_sync_metrics,
            }
            rounds.append(round_result)
            print(
                "WEIGHT_SYNC_ROUND_JSON=" + json.dumps(round_result),
                flush=True,
            )
            if not round_result["after"]["all_finite"]:
                raise RuntimeError(
                    f"vLLM model became non-finite after sync {sync_index}"
                )
            if not probe_after["all_finite"]:
                raise RuntimeError(
                    f"vLLM one-token logprob became non-finite after sync {sync_index}"
                )
        result = {
            "before": summarize_replica_reports(before),
            "scales_before": scales_before,
            "probe_before": probe_before,
            "resume_step": trainer._resume_step,
            "repeat_count": repeat_count,
            "rounds": rounds,
        }
        print(
            "WEIGHT_SYNC_INTEGRITY_JSON=" + json.dumps(result),
            flush=True,
        )
        if not result["before"]["all_finite"]:
            raise RuntimeError("vLLM model is non-finite before weight sync")
        if not probe_before["all_finite"]:
            raise RuntimeError("vLLM one-token logprob was non-finite before sync")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
