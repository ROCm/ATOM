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
#
# **Both halves come back byte-for-byte.** What may be removed is a marker a
# dialect declares; everything else, and whitespace in particular, survives.
# This is the rule `ToolCallParser.parse` already states for the stage after
# this one, and it exists for the same reason: the streaming filter releases
# bytes as they arrive and owns nothing to tidy them with, so any trimming
# done only here is a divergence a client sees.
#
# It was not applied here, and the symptom was the one that rule cites
# verbatim -- a trailing `.strip()` cost a code-block answer its final
# newline. Measured across 12544 (dialect, shape, chunking) comparisons,
# stripping put `stream=false` and `stream=true` at 50% byte-agreement on
# content; without it they agree exactly.
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
    split: Callable[[str, bool], SplitResult | None]
    template_efforts: frozenset[str] = frozenset()
    # Every marker that ends the reasoning channel, `think_end_marker`
    # included. A channel format can leave the think channel by *opening*
    # another one, and the streaming filter knew only the explicit close: a
    # K3 answer that goes straight to `<|open|>response<|sep|>` -- which its
    # own docs call the common path -- was streamed entirely as
    # `reasoning_content` with an empty `content`, while the non-streaming
    # split read it correctly.
    extra_end_markers: tuple[str, ...] = ()

    @property
    def end_markers(self) -> tuple[str, ...]:
        return (self.think_end_marker, *self.extra_end_markers)


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
    return text


_CHANNEL_END_MARKERS = (CHANNEL_THINK_END, CHANNEL_RESPONSE_START)


def _split_channel(text: str, starts_thinking: bool = False) -> SplitResult | None:
    """One rule: the reasoning channel ends at whichever closer comes first.

    This was three branches, and the middle one -- `<|open|>response<|sep|>`
    with no `<|close|>think<|sep|>` immediately before it -- returned
    `reasoning=None` and threw away everything ahead of the marker. A single
    byte between the two markers was enough to reach it, and the chain of
    thought then appeared in neither field. Silent data loss, not a
    divergence.

    Gated on `starts_thinking`, which the two other branches were not: these
    markers only mean anything if a reasoning channel was actually opened.
    Ungated, any model's answer that *quotes* one had the text before it
    deleted -- an answer about K3's wire format lost 19 characters on
    `stream=false` and kept them on `stream=true`. That is the inference
    `parse_tool_calls` was changed to stop making, in the half that was left
    still making it; a prompt that opens some *other* dialect's channel can
    still reach this one, and the structural answer is to resolve the dialect
    at startup as the tool-call format now is.
    """
    if not starts_thinking:
        return None
    best_at, best = len(text), None
    for marker in _CHANNEL_END_MARKERS:
        at = text.find(marker)
        if 0 <= at < best_at:
            best_at, best = at, marker
    if best is None:
        return None
    reasoning = text[:best_at]
    content = text[best_at + len(best) :]
    return (reasoning or None, _strip_channel_response_markers(content))


# --- Generic <think>...</think> dialect (K2/DeepSeek/Qwen3/MiniMax/...) ---

# No `\s*` after `</think>`. The newline a model puts before its answer is
# not a marker this dialect declares, so it survives -- see `SplitResult`.
THINK_OPEN_MARKER = "<think>"
THINK_END_MARKER = "</think>"
_THINK_CLOSED_RE = re.compile(
    re.escape(THINK_OPEN_MARKER) + r"(.*?)" + re.escape(THINK_END_MARKER) + r"(.*)",
    flags=re.DOTALL,
)
_THINK_OPEN_RE = re.compile(re.escape(THINK_OPEN_MARKER) + r"(.*)", flags=re.DOTALL)


def _split_think_tag(text: str, starts_thinking: bool = False) -> SplitResult | None:
    # Ordered as the streaming filter's state machine is, because that is what
    # this has to agree with. `starts_thinking` means the prompt already
    # opened the channel, so the output *begins* in it: the first `</think>`
    # closes it and any `<think>` before that is literal text inside the
    # reasoning, not an opener. Letting the searches below run first read one
    # as an opener and dropped everything ahead of it.
    if starts_thinking:
        # `</think>` with no `<think>`. Reasoning only because the prompt says
        # the channel is open -- ungated, this guessed, and disagreed with the
        # streaming path, which cannot honour an end marker it has no opener
        # for without waiting for one, and waiting is the stall. vLLM's
        # non-streaming path still guesses here and its streaming path does
        # not; the two do not agree, and this is the half worth copying.
        if THINK_END_MARKER in text:
            reasoning, _, content = text.partition(THINK_END_MARKER)
            return (reasoning or None, content)
        # Never closed: all reasoning, no answer. That is what a reasoning
        # model stopped at `max_tokens` looks like, and `separate_reasoning`'s
        # own fallback says it -- returning `None` defers to it rather than
        # writing the same answer twice.
        return None

    # Closed block: <think>...</think> answer.
    #
    # Searched, not anchored at position 0: a block does not have to open the
    # output. Anchoring meant a model that answers, opens a `<think>` block
    # and answers again matched nothing, so the client was handed the literal
    # tags with the chain of thought inside `content`.
    #
    # What precedes the block is content because it *is* content -- text
    # outside the reasoning channel. Nothing about the split needs the
    # streaming filter to justify it; this function has the whole output.
    #
    # The *first* block only, and that one IS a parity choice rather than a
    # reading of the format: the streaming filter closes the channel on the
    # first `</think>` and never reopens it, so splitting every block here
    # would make `stream=false` disagree with `stream=true` on any output
    # with two -- swapping one divergence for another. Whether *both* should
    # reopen is a separate question, and answering it means changing the
    # filter, not this.
    match = _THINK_CLOSED_RE.search(text)
    if match:
        return (match.group(1) or None, text[: match.start()] + match.group(2))
    # Unclosed block (truncated response). Searched, and split, for the same
    # reasons as the closed one above.
    match = _THINK_OPEN_RE.search(text)
    if match:
        return (match.group(1) or None, text[: match.start()])
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
        extra_end_markers=(CHANNEL_RESPONSE_START,),
        split=_split_channel,
        template_efforts=frozenset({"low", "high", "max"}),  # Kimi-K3
    ),
    # Generic <think>...</think> (K2/DeepSeek/Qwen3/MiniMax/...)
    ReasoningDialect(
        prompt_open_marker=THINK_OPEN_MARKER,
        output_open_marker=THINK_OPEN_MARKER,
        think_end_marker=THINK_END_MARKER,
        split=_split_think_tag,
    ),
)
