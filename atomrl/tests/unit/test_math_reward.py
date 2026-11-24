# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import sys
from types import ModuleType

import pytest
from atomrl.rewards.math_reward import _verl_compute_score, compute_score


def test_compute_score_accepts_last_boxed_answer_without_answer_prefix() -> None:
    result = compute_score(
        r"The absolute value is \(\frac{25}{8}\), so \(m+n=33\). \boxed{33}",
        "33",
    )

    assert result == {"score": 1.0, "acc": True, "pred": "33"}


def test_verl_score_returns_zero_without_math_verify(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "math_verify", None)

    assert _verl_compute_score("math", "answer", "expected") == 0.0


@pytest.mark.parametrize("failing_name", ["parse", "verify"])
def test_verl_score_contains_malformed_math_verify_failures(
    monkeypatch, failing_name
) -> None:
    module = ModuleType("math_verify")

    def parse(value):
        if failing_name == "parse":
            raise RuntimeError("malformed expression")
        return value

    def verify(_answer, _expected):
        if failing_name == "verify":
            raise RuntimeError("scorer rejected expression")
        return True

    module.parse = parse
    module.verify = verify
    monkeypatch.setitem(sys.modules, "math_verify", module)

    assert _verl_compute_score("math", "answer", "expected") == 0.0
