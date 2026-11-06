# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat prefix completion: continue an assistant turn the client started.

DeepSeek's beta contract. The last message is an assistant message carrying
``prefix: true``; the model's reply begins with its ``content``, and in
thinking mode an optional ``reasoning_content`` is the start of the chain of
thought. The response carries only what the model adds.

The prompt is the chat template's own render of every earlier message -- which
ends by opening an assistant turn, exactly as for an ordinary request -- with
the prefix appended, so no template is asked to render a partial turn. For
DeepSeek-V4.1 that is the official encoder's unterminated assistant turn. A
template whose finished turns put more than the markers around the reasoning
channel (Qwen3 writes line breaks into an empty one) sees the prefix without
them.
"""

from dataclasses import dataclass, replace

from .chat_encoder_adapters import MessageEncoderAdapter
from .protocol import ChatMessage
from .reasoning import ReasoningChannel
from .reasoning_dialects import FALLBACK_DIALECT


@dataclass(frozen=True)
class AssistantPrefix:
    """The start of the assistant turn the model is asked to continue."""

    content: str
    reasoning_content: str | None = None


def take_assistant_prefix(
    messages: list[ChatMessage],
    encoder: MessageEncoderAdapter | None = None,
) -> tuple[list[ChatMessage], AssistantPrefix | None]:
    """Split a trailing ``prefix: true`` assistant message off ``messages``.

    Returns the messages to render and the prefix, or ``messages`` unchanged
    and ``None`` when no message asks for prefix completion.

    The prefix is the message as the renderer would have taken it: its text
    joined by ``get_content_text``, as every message is before a chat
    template sees it, and then through ``encoder``'s own message processing
    when it has one, as the rendered messages are. DeepSeek-V4.1's joins a
    content list's text blocks a blank line apart and refuses image
    placeholder tokens (``ValueError``, a 400, as in the history).
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
    text = last.get_content_text()
    process = getattr(encoder, "process_image_messages", None)
    if process is not None:
        message = {
            "role": "assistant",
            "content": last.content if encoder.preserve_content else text,
            "reasoning_content": reasoning,
        }
        (message,), _images = process([message])
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content")
    return messages[:-1], AssistantPrefix(text, reasoning)


def continue_prompt(
    prompt: str, prefix: AssistantPrefix, channel: ReasoningChannel
) -> tuple[str, ReasoningChannel]:
    """Append ``prefix`` to a prompt that has just opened the assistant turn.

    ``channel`` is the server's own reading of this reply: whether it begins
    inside the reasoning channel -- because the prompt opened it, or because
    the model starts there with no marker at all (DeepSeek-R1) -- and in
    which dialect. Returns the prompt and how to read its continuation.

    A content prefix in a reply that begins in reasoning has the channel
    closed empty first: the reply continues that content, not reasoning
    about it. A format that frames the answer in a channel of its own
    (Kimi-K3) has that channel opened before the content.

    In either mode, a content prefix has begun the answer, so the
    continuation is read as the answer: a think marker in it is the model's
    own words, as it is when the model closes the channel itself.
    """
    if not channel.starts_open:
        if prefix.reasoning_content is not None:
            raise ValueError(
                "reasoning_content in an assistant prefix requires thinking mode, "
                "in which the reply begins in the reasoning channel"
            )
        if not prefix.content:
            return prompt, channel  # nothing has begun: an ordinary reply
        return prompt + prefix.content, replace(channel, answer_started=True)
    reasoning = prefix.reasoning_content or ""
    if prefix.reasoning_content is not None and not prefix.content:
        return prompt + reasoning, channel
    dialect = channel.dialect or FALLBACK_DIALECT
    answer = dialect.think_end_marker + dialect.answer_open_marker
    return (
        prompt + reasoning + answer + prefix.content,
        replace(channel, starts_open=False, answer_started=True),
    )


def _prefix_flag(message: ChatMessage) -> bool:
    flag = (message.model_extra or {}).get("prefix")
    if flag is None:
        return False  # unset, however the client spells it
    if not isinstance(flag, bool):
        raise ValueError("'prefix' must be a boolean")  # noqa: TRY004
    return flag
