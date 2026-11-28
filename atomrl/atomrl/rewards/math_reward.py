# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/utils/reward_score/math_dapo.py

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from verl's math reward implementation:
# https://github.com/volcengine/verl

"""DAPO-style math reward: verify answers against ground truth.

Ported from verl/utils/reward_score/math_dapo.py (Apache-2.0, Bytedance & EleutherAI).
The minerva criterion extracts the last ``Answer:`` line, with a final
``\boxed{...}`` fallback for benchmark prompts that do not prescribe the
training response format. No symbolic ``math_verify`` check is used.
"""

from __future__ import annotations

import logging
import re

import torch

from atomrl.core.protocol import DataProto

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{")

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last LaTeX ``\\boxed{...}`` expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left) : -1]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def compute_score(solution_str: str, ground_truth: str) -> dict:
    """Compute reward for a single (solution, ground_truth) pair.

    Take the last 300 chars and extract the answer after the last ``Answer:``
    line. If that marker is absent, accept the last LaTeX ``\boxed{...}``
    expression used by standard math benchmark outputs. Normalize both sides
    and compare as strings; no symbolic ``math_verify`` fallback is used.

    Returns dict with keys: score (float), acc (bool), pred (str).
    """
    solution_str = solution_str[-300:]

    match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", solution_str)
    boxed = last_boxed_only_string(solution_str)
    extracted = match[-1] if match else remove_boxed(boxed) if boxed else "[INVALID]"
    pred = normalize_final_answer(extracted)
    gt = normalize_final_answer(ground_truth)

    correct = pred == gt
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": pred}


def compute_math_reward(
    responses: list[str],
    ground_truths: list[str],
    overlong_buffer: int = 0,
    max_response_len: int = 0,
    overlong_penalty: float = -1.0,
) -> tuple[torch.Tensor, list[dict]]:
    """Batch math reward computation.

    Returns (rewards [B], details [list of dicts]).
    """
    rewards = []
    details = []
    for resp, gt in zip(responses, ground_truths):
        if max_response_len > 0 and len(resp) > max_response_len + overlong_buffer:
            result = {"score": overlong_penalty, "acc": False, "pred": "[OVERLONG]"}
        else:
            result = compute_score(resp, gt)
        rewards.append(result["score"])
        details.append(result)
    return torch.tensor(rewards, dtype=torch.float32), details


def dapo_math_reward(
    batch=None,
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs,
):
    """Reward function compatible with both AtomRL and verl interfaces.

    AtomRL calls: dapo_math_reward(batch: DataProto) → Tensor
    verl calls:    dapo_math_reward(data_source, solution_str, ground_truth, ...) → float
    """
    if data_source is not None or solution_str is not None:
        return _verl_compute_score(
            data_source, solution_str, ground_truth, extra_info, **kwargs
        )
    return _atomrl_batch_reward(batch, **kwargs)


def _verl_compute_score(
    data_source, solution_str, ground_truth, extra_info=None, **kwargs
):
    """verl-compatible per-sample scoring."""
    try:
        from math_verify import parse, verify
    except ImportError:
        return 0.0

    try:
        answer = parse(solution_str)
        expected = parse(ground_truth)
        return 1.0 if verify(answer, expected) else 0.0
    # Optional per-sample parser/scorer failures yield zero reward.
    except Exception:  # noqa: BLE001 -- boundary
        return 0.0


def _atomrl_batch_reward(batch: DataProto, **kwargs) -> torch.Tensor:
    """Reward function compatible with ``RewardWorker``'s function-based interface.

    Expects ``batch.meta`` to contain:
      - ``responses``: list[str] — decoded model outputs
      - ``ground_truths``: list[str] — reference answers

    Optional meta keys:
      - ``overlong_buffer``: int (default 0)
      - ``max_response_len``: int (default 0 = no limit)
      - ``overlong_penalty``: float (default -1.0)
    """
    responses = batch.meta.get("responses", [])
    ground_truths = batch.meta.get("ground_truths", [])
    if not responses or not ground_truths:
        b = batch.batch_size
        logger.warning(
            "dapo_math_reward: no responses/ground_truths in meta; returning zeros."
        )
        return torch.zeros(b, dtype=torch.float32)

    rewards, details = compute_math_reward(
        responses,
        ground_truths,
        overlong_buffer=int(batch.meta.get("overlong_buffer", 0)),
        max_response_len=int(batch.meta.get("max_response_len", 0)),
        overlong_penalty=float(batch.meta.get("overlong_penalty", -1.0)),
    )
    batch.meta["reward_details"] = details
    return rewards
