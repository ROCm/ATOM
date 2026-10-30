# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request adaptation for the checkpoint's standalone V4.1 encoder.

Prompt construction stays in the official encoder. These hooks only translate
ATOM's common controls and attach top-level tool schemas to the system turn.
"""

from typing import Any

from .reasoning_controls import ReasoningRules


def normalize_effort(effort: Any) -> str | int | None:
    """Translate the hosted API aliases while retaining ATOM's integer budget."""
    if effort is None:
        return None
    if isinstance(effort, str):
        effort = {"minimal": "low", "medium": "high", "xhigh": "high"}.get(
            effort, effort
        )
        if effort in ("none", "low", "high", "max"):
            return effort
    elif type(effort) is int and 1 <= effort <= 100:
        return effort
    raise ValueError(
        "DeepSeek-V4.1 reasoning_effort must be none, low, high, max, "
        "minimal, medium, xhigh, or an integer in [1, 100]"
    )


REASONING_RULES = ReasoningRules(
    normalize_effort=normalize_effort,
    string_effort_enables=True,
    clear_effort_when_disabled=True,
)


def prepare_messages(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    prepared = [dict(message) for message in messages]
    if tools:
        if prepared and prepared[0].get("role") == "system":
            prepared[0]["tools"] = tools
        else:
            prepared.insert(0, {"role": "system", "tools": tools})
    return prepared


def prepare_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    # Explicit API effort is merged last under the common thinking_effort key;
    # it must also override a native reasoning_effort in template defaults.
    if "thinking_effort" in kwargs:
        kwargs["reasoning_effort"] = kwargs.pop("thinking_effort")
    effort = normalize_effort(kwargs.get("reasoning_effort"))
    if effort == "none":
        kwargs["thinking_mode"] = "chat"
        kwargs.pop("reasoning_effort", None)
    elif effort is not None:
        kwargs["reasoning_effort"] = effort
    if kwargs.get("thinking_mode", "thinking") not in ("thinking", "chat"):
        raise ValueError("DeepSeek-V4.1 thinking_mode must be 'thinking' or 'chat'")
    if kwargs.get("return_multi_modal_data"):
        raise ValueError("return_multi_modal_data is not a text chat-template control")
    return kwargs
