# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reasoning-channel dialects (model-specific data for the general engine).

The engine in ``reasoning.py`` is dialect-agnostic: it iterates ``DIALECTS`` to
detect/split the reasoning channel. All model-specific marker knowledge lives
here. Adding a model = add one ``ReasoningDialect`` entry (and its ``split`` for
whole-response separation).

Two dialects today, named by format rather than model:
  - inline ``<think>...</think>`` (DeepSeek-R1, Qwen3, Kimi-K2, MiniMax, ...).
    The opening tag may be emitted in the output or injected by the template.
  - structured channel format: one stream split into named channels (think /
    response / tools), each wrapped in framing tokens. The same concept as
    OpenAI Harmony's analysis/final/commentary channels (gpt-oss). The opening
    tag is template-injected, so the output begins *inside* the reasoning
    channel. Different channel-format models use different framing tokens; the
    entry below carries Kimi-K3's (``<|open|>think<|sep|>`` ...), and another
    such model would add its own entry with its own tokens.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

# Structured-channel format tokens (``<|open|>SECTION<|sep|>`` ... framing).
# Named by the format concept, not the model: channel formats are a cross-model
# pattern (e.g. gpt-oss/Harmony uses the same idea with different framing tokens).
# The token *values* below are Kimi-K3's — a different channel-format model would
# declare its own values. Declared locally so this module is self-contained
# (each parser owns the token strings it uses); the tool-call parser keeps its
# own copies of the subset it needs.
CHANNEL_THINK_START = "<|open|>think<|sep|>"
CHANNEL_THINK_END = "<|close|>think<|sep|>"
CHANNEL_RESPONSE_START = "<|open|>response<|sep|>"
CHANNEL_RESPONSE_END = "<|close|>response<|sep|>"
CHANNEL_MESSAGE_END = "<|close|>message<|sep|>"
CHANNEL_END_OF_MSG = "<|end_of_msg|>"
CHANNEL_TOOLS_START = "<|open|>tools<|sep|>"
CHANNEL_CALL_PREFIX = '<|open|>call tool="'

# Result of splitting a full response: (reasoning_content or None, content).
SplitResult = tuple[str | None, str]


@dataclass(frozen=True)
class ReasoningDialect:
    """How one model family delimits its reasoning channel.

    - ``prompt_open_marker``: what a rendered prompt ends with when the template
      has already opened the reasoning channel (output then begins in reasoning).
    - ``output_open_marker``: the marker the model *emits* to open reasoning
      mid-stream (``<think>``); ``None`` when the template injects it instead.
    - ``think_end_marker``: the marker that ends the reasoning channel.
    - ``split``: whole-response separator returning ``SplitResult`` or ``None``
      if this dialect does not apply to the text.
    - ``template_efforts``: reasoning-effort levels this model's chat template
      accepts (e.g. K3's ``low``/``high``/``max``); empty when the model has no
      effort control.
    """

    prompt_open_marker: str
    output_open_marker: str | None
    think_end_marker: str
    split: Callable[[str], SplitResult | None]
    template_efforts: frozenset[str] = frozenset()


# --- Structured-channel dialect ---


def _strip_channel_response_markers(text: str) -> str:
    # Preserve tool-call sections: they follow <|close|>response<|sep|> (an
    # empty response channel), so truncating at CHANNEL_RESPONSE_END would drop
    # the whole tools block. Leave it intact for parse_tool_calls to handle.
    if CHANNEL_TOOLS_START in text or CHANNEL_CALL_PREFIX in text:
        return text

    text = text.removeprefix(CHANNEL_RESPONSE_START)

    for marker in (CHANNEL_RESPONSE_END, CHANNEL_MESSAGE_END, CHANNEL_END_OF_MSG):
        if marker in text:
            text = text.partition(marker)[0]
    return text.strip()


def _split_channel(text: str) -> SplitResult | None:
    combined = CHANNEL_THINK_END + CHANNEL_RESPONSE_START
    if combined in text:
        reasoning, _, content = text.partition(combined)
        return (reasoning.strip() or None, _strip_channel_response_markers(content))
    if CHANNEL_RESPONSE_START in text:
        _, _, content = text.partition(CHANNEL_RESPONSE_START)
        return (None, _strip_channel_response_markers(content))
    if CHANNEL_THINK_END in text:
        reasoning, _, content = text.partition(CHANNEL_THINK_END)
        return (reasoning.strip() or None, _strip_channel_response_markers(content))
    return None


# --- Generic <think>...</think> dialect (K2/DeepSeek/Qwen3/MiniMax/...) ---

_THINK_CLOSED_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", flags=re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>(.*)", flags=re.DOTALL)


def _split_think_tag(text: str) -> SplitResult | None:
    # Closed block: <think>...</think> answer
    match = _THINK_CLOSED_RE.match(text)
    if match:
        return (match.group(1).strip() or None, match.group(2).strip())
    # </think> without <think> — template injected the opening tag into the prompt.
    if "</think>" in text:
        reasoning, _, content = text.partition("</think>")
        return (reasoning.strip() or None, content.strip())
    # Unclosed block (truncated response).
    match = _THINK_OPEN_RE.match(text)
    if match:
        return (match.group(1).strip() or None, "")
    return None


# "Channel" here follows the established meaning from OpenAI's Harmony format:
# one output stream carrying several named sections (think / response / tools),
# each wrapped in framing tokens, that we de-multiplex into separate fields.
# Harmony's analysis/final/commentary channels map onto K3's think/response/tools.
# We name the tokens by this cross-model concept (CHANNEL_*) rather than by the
# model. Channel-format models differ in their framing tokens, so each gets its
# own DIALECTS entry; the entry below carries Kimi-K3's token values.
#
# Detection/priority order: structured-channel dialects before inline-tag ones,
# so a specific channel marker is tried before the generic <think> tag.
# separate_reasoning() returns the first dialect whose split() matches. A dialect
# is identified by its markers/split behavior, not a label.
DIALECTS: tuple[ReasoningDialect, ...] = (
    # Structured channel format — Kimi-K3 token values (see CHANNEL_* above)
    ReasoningDialect(
        prompt_open_marker=CHANNEL_THINK_START,
        output_open_marker=None,  # template-injected; not emitted in output
        think_end_marker=CHANNEL_THINK_END,
        split=_split_channel,
        template_efforts=frozenset({"low", "high", "max"}),  # Kimi-K3
    ),
    # Generic <think>...</think> (K2/DeepSeek/Qwen3/MiniMax/...)
    ReasoningDialect(
        prompt_open_marker="<think>",
        output_open_marker="<think>",
        think_end_marker="</think>",
        split=_split_think_tag,
    ),
)
