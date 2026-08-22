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
    # The literals whose presence in a *chat template* says the model speaks
    # this dialect. Read once at startup, exactly as the tool-call format is:
    # a template states what it taught the model, while an output only
    # exhibits whatever it happens to contain.
    template_markers: tuple[str, ...] = ()
    # Literals this format wraps every answer in, which are neither the
    # channel's own delimiters nor any tool format's. Removed from both halves
    # of the split and dropped by the streaming filter, so the two agree
    # without either of them depending on which tool-call parser happened to
    # be resolved -- the tool parser removes the same tokens, and a K3
    # deployment whose template refused the tools probe has none.
    content_framing: tuple[str, ...] = ()
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

    def strip_framing(self, text: str) -> str:
        for marker in self.content_framing:
            text = text.replace(marker, "")
        return text

    def taught_by(self, template_source: str) -> bool:
        """Does this template teach the model this dialect?"""
        return any(m in template_source for m in self.template_markers)


# --- Structured-channel dialect ---


_CHANNEL_END_MARKERS = (CHANNEL_THINK_END, CHANNEL_RESPONSE_START)


def _split_channel(text: str, starts_thinking: bool = False) -> SplitResult | None:
    """One rule: the reasoning channel ends at whichever closer comes first.

    This was three branches, and the middle one -- `<|open|>response<|sep|>`
    with no `<|close|>think<|sep|>` immediately before it -- returned
    `reasoning=None` and threw away everything ahead of the marker. A single
    byte between the two markers was enough to reach it, and the chain of
    thought then appeared in neither field. Silent data loss, not a
    divergence.

    The content comes back with its channel framing still on it, and that is
    deliberate: removing a tool-format's literals is the tool parser's job and
    it does it on both delivery paths, from one declared list. Doing it here
    too meant the reasoning stage removed them when the response was not
    streamed and left them when it was -- the streaming filter has no such
    step -- so the two paths only agreed by way of a *third* stage, and only
    when a K3 parser had been resolved. This one also truncated at
    `<|end_of_msg|>` rather than removing it, so anything after that token was
    deleted on one path alone.

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
    return (reasoning or None, content)


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
_CHANNEL_CONTENT_FRAMING = (
    CHANNEL_RESPONSE_START,
    CHANNEL_RESPONSE_END,
    CHANNEL_MESSAGE_END,
    CHANNEL_END_OF_MSG,
)

_CHANNEL_DIALECT = ReasoningDialect(
    prompt_open_marker=CHANNEL_THINK_START,
    output_open_marker=None,  # template-injected; not emitted in output
    think_end_marker=CHANNEL_THINK_END,
    extra_end_markers=(CHANNEL_RESPONSE_START,),
    split=_split_channel,
    template_efforts=frozenset({"low", "high", "max"}),  # Kimi-K3
    template_markers=(CHANNEL_THINK_START, CHANNEL_THINK_END),
    content_framing=_CHANNEL_CONTENT_FRAMING,
)

DIALECTS: tuple[ReasoningDialect, ...] = (
    # Structured channel format — Kimi-K3 token values (see CHANNEL_* above)
    _CHANNEL_DIALECT,
    # Generic <think>...</think> (K2/DeepSeek/Qwen3/MiniMax/...)
    ReasoningDialect(
        prompt_open_marker=THINK_OPEN_MARKER,
        output_open_marker=THINK_OPEN_MARKER,
        think_end_marker=THINK_END_MARKER,
        split=_split_think_tag,
        template_markers=(THINK_OPEN_MARKER, THINK_END_MARKER),
    ),
)

# The dialect a model speaks when its template names none of them. The
# inline-`<think>` one, and not "no dialect at all": a template that renders
# the tag conditionally may not carry it in the source we can read, and the
# cost of guessing wrong here is nil -- a model that never writes `<think>`
# never matches its markers, so the split is a no-op. Guessing the other way
# would ship a chain of thought to the client as the answer.
FALLBACK_DIALECT = DIALECTS[-1]


def resolve_dialect(
    template_source: str, rendered_prompt: str = ""
) -> tuple[ReasoningDialect, bool]:
    """Which reasoning dialect this model speaks, and whether it was stated.

    Asked once, at startup -- the same question, in the same place and by the
    same means, as the tool-call format. It used to be asked twice per response
    and differently each time: the non-streaming split tried every dialect in
    order and took the first that matched, while the streaming filter used the
    *union* of every dialect's end markers and closed on whichever came first.
    So a `<think>` model answering a question about Kimi's wire format ended
    its chain of thought at the quoted `<|open|>response<|sep|>` when streamed
    and at the real `</think>` when not.

    Both kinds of evidence, and the *render* first, because the source is not
    always readable. Kimi-K3 ships neither a Jinja template nor an encoder
    under the path the loader looks in, so `chat_template_source` returns ""
    for it -- and falling back to the inline-`<think>` dialect on that made a
    K3 answer come back as `reasoning_content` with `content` empty, on both
    delivery modes, while the tool-call format resolved correctly off the
    rendered probe. Same model, two answers to "what does this model speak",
    from two different pieces of evidence. Now it is one function reading
    both.

    The second return value is whether a dialect was actually named, so the
    caller can say at startup which one it is and whether it was a fallback.
    """
    for evidence in (rendered_prompt, template_source):
        if not evidence:
            continue
        for dialect in DIALECTS:
            if dialect.taught_by(evidence):
                return dialect, True
    return FALLBACK_DIALECT, False
