# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Math reward scored by HF Math-Verify, as NeMo-RL's math environment scores it
(``math_verify_impl: hf_math_verify``): the ground truth is parsed as
``\\boxed{gt}`` LaTeX, the response as an expression or LaTeX, and the two are
compared symbolically. 1.0 if they match, else 0.0, including on any parser
error or timeout.

Math-Verify times out through ``signal.alarm``, which only the main thread of a
process may use, and the rollout groups are scored on a worker thread. The
scoring therefore runs in a pool of spawned processes, like NeMo-RL's Ray
verify workers.
"""

from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import torch

from atomrl.rewards.math_reward import compute_math_reward

logger = logging.getLogger(__name__)

_verify_func = None
_pool: ProcessPoolExecutor | None = None
_pool_workers = 0


def _verifier():
    global _verify_func
    if _verify_func is None:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        _verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    return _verify_func


def hf_math_verify_score(response: str, ground_truth: str) -> dict:
    """Score one response; must run on a process's main thread."""
    from math_verify.errors import TimeoutException

    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            score, extracted = _verifier()(
                ["\\boxed{" + ground_truth + "}"], [response]
            )
        score = float(score)
    except (Exception, TimeoutException):  # noqa: BLE001 - the verifier raises anything
        return {"score": 0.0, "acc": False, "pred": "[INVALID]"}
    preds = extracted[1] if extracted and len(extracted) == 2 else []
    pred = str(preds[0]) if preds else "[INVALID]"
    return {"score": score, "acc": score == 1.0, "pred": pred}


def _score_chunk(pairs: list[tuple[str, str]]) -> list[dict]:
    return [hf_math_verify_score(resp, gt) for resp, gt in pairs]


def _get_pool(num_workers: int) -> ProcessPoolExecutor:
    global _pool, _pool_workers
    if _pool is None or _pool_workers != num_workers:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
        _pool = ProcessPoolExecutor(
            max_workers=num_workers, mp_context=multiprocessing.get_context("spawn")
        )
        _pool_workers = num_workers
    return _pool


def compute_math_verify_reward(
    responses: list[str], ground_truths: list[str], num_workers: int = 8
) -> tuple[torch.Tensor, list[dict]]:
    """Batch scoring across ``num_workers`` spawned processes, in order.

    Returns (rewards [B], details [list of {score, acc, pred}])."""
    pairs = list(zip(responses, ground_truths))
    if not pairs:
        return torch.zeros(0, dtype=torch.float32), []
    workers = max(1, int(num_workers))
    size = (len(pairs) + workers - 1) // workers
    chunks = [pairs[i : i + size] for i in range(0, len(pairs), size)]
    details = [d for part in _get_pool(workers).map(_score_chunk, chunks) for d in part]
    rewards = torch.tensor([d["score"] for d in details], dtype=torch.float32)
    return rewards, details


def score_math_responses(
    responses: list[str], ground_truths: list[str], reward_cfg
) -> tuple[torch.Tensor, list[dict]]:
    """Score with ``reward.scorer``: ``dapo`` (string match, +-1) or
    ``hf_math_verify`` (NeMo-RL's, 1 / 0)."""
    scorer = str(getattr(reward_cfg, "scorer", "dapo") or "dapo")
    if scorer == "dapo":
        return compute_math_reward(responses, ground_truths)
    if scorer == "hf_math_verify":
        return compute_math_verify_reward(
            responses, ground_truths, int(getattr(reward_cfg, "num_workers", 8))
        )
    raise ValueError(
        f"reward.scorer must be 'dapo' or 'hf_math_verify', not {scorer!r}"
    )
