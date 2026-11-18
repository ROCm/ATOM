# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Behavior tables for explicit controls, template defaults, and model rules."""

from copy import deepcopy

import pytest

from atom.entrypoints.openai.deepseek_v41_encoder import REASONING_RULES, prepare_kwargs
from atom.entrypoints.openai.reasoning_controls import (
    CHAT_REASONING_RULES,
    ReasoningRules,
    apply_reasoning_controls,
    resolve_reasoning_controls,
)


@pytest.mark.parametrize(
    "thinking,effort,expected",
    [
        (None, None, (None, None)),
        (None, 37, (None, 37)),
        (None, "none", (False, None)),
        (None, "low", (True, "low")),
        (None, "minimal", (True, "low")),
        (None, "medium", (True, "high")),
        (None, "xhigh", (True, "high")),
        ({"type": "disabled"}, "high", (False, "high")),
        ({"type": "enabled"}, "none", (False, None)),
        ({"type": "enabled", "effort": "low"}, "none", (False, "low")),
        ({"type": "enabled", "effort": "max"}, "low", (True, "max")),
        ({"type": "enabled", "effort": "none"}, "high", (False, None)),
        ({"type": "enabled"}, 37, (True, 37)),
    ],
)
def test_explicit_control_precedence(thinking, effort, expected):
    assert (
        resolve_reasoning_controls(
            thinking=thinking, effort=effort, rules=REASONING_RULES
        )
        == expected
    )


@pytest.mark.parametrize("effort", [True, False, 0, 101, 3.5, "ultra", [], {}])
def test_invalid_effort_is_not_hidden_by_disabled_thinking(effort):
    with pytest.raises(ValueError):
        resolve_reasoning_controls(
            thinking={"type": "disabled"}, effort=effort, rules=REASONING_RULES
        )


@pytest.mark.parametrize(
    "toggle", [("thinking_mode", "chat", "thinking"), ("enable_thinking", False, True)]
)
@pytest.mark.parametrize(
    "effort,expected", [(None, False), (37, False), ("high", True), ("none", False)]
)
def test_model_rules_use_the_resolved_template_toggle(toggle, effort, expected):
    key, off, on = toggle
    defaults = {key: off, "reasoning_effort": 20}
    before = deepcopy(defaults)
    controls = resolve_reasoning_controls(effort=effort, rules=REASONING_RULES)
    actual = apply_reasoning_controls(defaults, toggle, controls, rules=REASONING_RULES)
    assert actual[key] == (on if expected else off)
    assert defaults == before
    if effort == "none":
        assert "reasoning_effort" not in actual and "thinking_effort" not in actual


def test_rules_do_not_rewrite_another_models_effort_vocabulary():
    rules = ReasoningRules(normalize_effort=lambda value: value)
    assert resolve_reasoning_controls(effort="medium", rules=rules) == (None, "medium")
    assert resolve_reasoning_controls(effort="high", rules=CHAT_REASONING_RULES) == (
        None,
        "high",
    )


def test_encoder_normalization_does_not_mutate_callers_dictionary():
    source = {
        "thinking_effort": "none",
        "reasoning_effort": 20,
        "thinking_mode": "thinking",
    }
    before = deepcopy(source)
    assert prepare_kwargs(source) == {"thinking_mode": "chat"}
    assert source == before
