# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request control resolution shared by frontend protocols, independent of models."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .reasoning import VALID_TEMPLATE_EFFORTS


def _legacy_chat_effort(effort: Any) -> str | int | None:
    if isinstance(effort, (int, float)):
        if type(effort) is not int or not 1 <= effort <= 100:
            raise ValueError("reasoning effort must be an integer in [1, 100]")
        return effort
    if isinstance(effort, str) and effort in VALID_TEMPLATE_EFFORTS:
        return effort
    return None


def _passthrough_effort(effort: Any) -> Any:
    return effort


@dataclass(frozen=True)
class ReasoningRules:
    normalize_effort: Callable[[Any], Any] = _legacy_chat_effort
    string_effort_enables: bool = False
    none_disables: bool = True
    reconcile_effort_with_toggle: bool = False


CHAT_REASONING_RULES = ReasoningRules()
# Preserve the existing non-adapted Responses path, including template extensions.
RESPONSES_REASONING_RULES = ReasoningRules(
    normalize_effort=_passthrough_effort, none_disables=False
)


def resolve_reasoning_controls(
    *, thinking=None, effort=None, rules=CHAT_REASONING_RULES
):
    """Return (enabled, effort), retaining None for an unstated toggle.

    Compatibility precedence: a top-level effort of none disables thinking;
    otherwise an explicit thinking.type wins over effort-based activation.
    thinking.effort overrides the effort value. Integer budgets never enable
    thinking on their own. Alias mappings and string activation come from rules.
    """
    enabled = None
    if isinstance(thinking, dict) and thinking.get("type") is not None:
        enabled = thinking["type"] != "disabled"
    if rules.none_disables and effort == "none":
        enabled = False
    if isinstance(thinking, dict) and thinking.get("effort") is not None:
        effort = thinking["effort"]
    effort = rules.normalize_effort(effort)
    if rules.none_disables and effort == "none":
        return False, None
    if rules.string_effort_enables and isinstance(effort, str) and enabled is None:
        enabled = True
    return enabled, effort


def apply_reasoning_controls(kwargs, toggle, controls, *, rules=CHAT_REASONING_RULES):
    """Merge resolved controls into a copy; validation never mutates defaults."""
    result = dict(kwargs)
    enabled, effort = controls
    if toggle is not None and enabled is not None:
        name, off_value, on_value = toggle
        result[name] = on_value if enabled else off_value
    if effort is not None:
        result["thinking_effort"] = effort
    if enabled is True and rules.reconcile_effort_with_toggle:
        # An explicit enable must override a disabled effort inherited from
        # template defaults, rather than being undone during encoder rendering.
        for key in ("thinking_effort", "reasoning_effort"):
            if result.get(key) == "none":
                result.pop(key)
    if enabled is False and rules.reconcile_effort_with_toggle:
        result.pop("thinking_effort", None)
        result.pop("reasoning_effort", None)
    return result
