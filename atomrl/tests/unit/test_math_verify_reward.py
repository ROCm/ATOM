# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("math_verify")

from atomrl.rewards.math_verify_reward import (
    compute_math_verify_reward,
    hf_math_verify_score,
    score_math_responses,
)


@pytest.mark.parametrize(
    ("response", "ground_truth", "score"),
    [
        ("so the answer is \\boxed{0.5}", "\\frac{1}{2}", 1.0),
        ("Thus $x = \\boxed{\\frac{3}{6}}$.", "1/2", 1.0),
        ("The total is \\boxed{12}.", "12", 1.0),
        ("The total is \\boxed{13}.", "12", 0.0),
        ("no answer here", "12", 0.0),
    ],
)
def test_symbolic_check_scores_one_or_zero(response, ground_truth, score) -> None:
    assert hf_math_verify_score(response, ground_truth)["score"] == score


def test_batch_keeps_order_across_worker_processes() -> None:
    responses = [f"\\boxed{{{i}}}" for i in range(10)]
    truths = [str(i if i % 3 else i + 1) for i in range(10)]
    rewards, details = compute_math_verify_reward(responses, truths, num_workers=3)
    assert rewards.tolist() == [0.0 if i % 3 == 0 else 1.0 for i in range(10)]
    assert [d["acc"] for d in details] == [i % 3 != 0 for i in range(10)]


def test_scores_from_a_worker_thread() -> None:
    """Math-Verify refuses to run off the main thread (signal.alarm), and the
    rollout groups are scored on one; the process pool is what makes it work."""
    out = {}

    def run():
        out["rewards"], _ = compute_math_verify_reward(
            ["\\boxed{2}", "\\boxed{3}"], ["2", "2"], num_workers=2
        )

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert out["rewards"].tolist() == [1.0, 0.0]


def test_scorer_selects_the_implementation() -> None:
    responses, truths = ["Answer: 0.5"], ["1/2"]
    dapo, _ = score_math_responses(responses, truths, SimpleNamespace(scorer="dapo"))
    assert dapo.tolist() == [-1.0]
    verify, _ = score_math_responses(
        responses, truths, SimpleNamespace(scorer="hf_math_verify", num_workers=1)
    )
    assert verify.tolist() == [1.0]
    with pytest.raises(ValueError, match="reward.scorer"):
        score_math_responses(responses, truths, SimpleNamespace(scorer="other"))
