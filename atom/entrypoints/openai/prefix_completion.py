# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat prefix completion: continue an assistant turn the client started.

DeepSeek's beta contract. The last message is an assistant message carrying
``prefix: true``; the model's reply begins with its ``content``, and in
thinking mode an optional ``reasoning_content`` is the start of the chain of
thought. The response carries only what the model adds.

The prompt is the chat template's own render of every earlier message -- which
ends by opening an assistant turn, exactly as for an ordinary request -- with
the prefix appended. No template is asked to render a partial turn, so this
works the same for Jinja templates and model-shipped encoders.
"""

from dataclasses import dataclass

from .protocol import ChatMessage
from .reasoning import dialect_opened_by


@dataclass(frozen=True)
class AssistantPrefix:
    """The start of the assistant turn the model is asked to continue."""

    content: str
    reasoning_content: str | None = None


def take_assistant_prefix(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], AssistantPrefix | None]:
    """Split a trailing ``prefix: true`` assistant message off ``messages``.

    Returns the messages to render and the prefix, or ``messages`` unchanged
    and ``None`` when no message asks for prefix completion.
    """
    flagged = [i for i, m in enumerate(messages) if _prefix_flag(m)]
    if not flagged:
        return messages, None
    last = messages[-1]
    if flagged != [len(messages) - 1] or last.role != "assistant":
        raise ValueError(
            "'prefix' may be set only on the last message, which must be an "
            "assistant message"
        )
    if len(messages) < 2 or messages[-2].role == "assistant":
        raise ValueError(
            "an assistant prefix must follow a user, system or tool message"
        )
    extras = last.model_extra or {}
    if extras.get("tool_calls"):
        raise ValueError("an assistant prefix cannot carry tool_calls")
    if isinstance(last.content, list) and any(
        not isinstance(part, dict) or part.get("type") != "text"
        for part in last.content
    ):
        raise ValueError("an assistant prefix may contain only text")
    reasoning = extras.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("an assistant prefix's reasoning_content must be a string")
    return messages[:-1], AssistantPrefix(last.get_content_text(), reasoning)


def continue_prompt(prompt: str, prefix: AssistantPrefix) -> tuple[str, bool]:
    """Append ``prefix`` to a prompt that has just opened the assistant turn.

    Returns the prompt and whether the model's output still begins inside the
    reasoning channel. A prompt that opened the channel with only a content
    prefix has the channel closed empty first: the reply begins with that
    content, not with reasoning about it. A format that frames the answer
    in a channel of its own (Kimi-K3) has that channel opened before the
    content.
    """
    dialect = dialect_opened_by(prompt)
    if dialect is None:
        if prefix.reasoning_content is not None:
            raise ValueError(
                "reasoning_content in an assistant prefix requires thinking mode"
            )
        return prompt + prefix.content, False
    reasoning = prefix.reasoning_content or ""
    if prefix.reasoning_content is not None and not prefix.content:
        return prompt + reasoning, True
    answer = dialect.think_end_marker + dialect.answer_open_marker
    return prompt + reasoning + answer + prefix.content, False


def _prefix_flag(message: ChatMessage) -> bool:
    flag = (message.model_extra or {}).get("prefix", False)
    if not isinstance(flag, bool):
        raise ValueError("'prefix' must be a boolean")  # noqa: TRY004
    return flag
