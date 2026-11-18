# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reasoning-channel dialects (model-specific data for the general engine).

The engine in ``reasoning.py`` is dialect-agnostic: ``resolve_dialect`` picks
a model's entry from ``DIALECTS`` once, at startup, and the one reader reads
the reasoning channel with it. All model-specific marker knowledge lives
here. Adding a model = add one ``ReasoningDialect`` entry.

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

from dataclasses import dataclass

from . import kimi_k3_tokens as k3


@dataclass(frozen=True)
class ReasoningDialect:
    """How one model family delimits its reasoning channel.

    - ``prompt_open_marker``: what a rendered prompt ends with when the template
      has already opened the reasoning channel (output then begins in reasoning).
    - ``output_open_marker``: the marker the model *emits* to open reasoning
      mid-stream (``<think>``); ``None`` when the template injects it instead.
    - ``think_end_marker``: the marker that ends the reasoning channel.
    - ``template_efforts``: reasoning-effort levels this model's chat template
      accepts (e.g. K3's ``low``/``high``/``max``); empty when the model has no
      effort control.
    """

    prompt_open_marker: str
    output_open_marker: str | None
    think_end_marker: str
    template_efforts: frozenset[str] = frozenset()
    # The literals whose presence in a *chat template* says the model speaks
    # this dialect. Read once at startup, exactly as the tool-call format is:
    # a template states what it taught the model, while an output only
    # exhibits whatever it happens to contain.
    template_markers: tuple[str, ...] = ()
    # Literals this format wraps every answer in, which are neither the
    # channel's own delimiters nor any tool format's. Dropped by the reasoning
    # filter, so the answer does not depend on which tool-call parser happened
    # to be resolved -- the tool parser removes the same tokens, and a K3
    # deployment whose template refused the tools probe has none.
    content_framing: tuple[str, ...] = ()
    # Every marker that ends the reasoning channel besides `think_end_marker`.
    # A channel format can leave the think channel by *opening* another one:
    # a K3 answer that goes straight to `<|open|>response<|sep|>` -- which its
    # own docs call the common path -- would otherwise be read entirely as
    # `reasoning_content` with an empty `content`.
    extra_end_markers: tuple[str, ...] = ()
    # Markers that close the *answer*. A channel whose last words, whitespace
    # and framing aside, are one of these, and that ends at the end of the
    # output or at a channel other than the answer's own, was the reply,
    # written where reasoning goes; see `ReasoningFilter._close_channel`.
    answer_end_markers: tuple[str, ...] = ()
    # What opens the answer after `think_end_marker`, in a format that gives
    # the answer a channel of its own; empty when the answer follows the
    # closer directly. Kimi-K3's encoder renders every assistant turn as
    # think, then `<|open|>response<|sep|>`, and opens `response` directly
    # when thinking is off.
    answer_open_marker: str = ""

    @property
    def end_markers(self) -> tuple[str, ...]:
        return (self.think_end_marker, *self.extra_end_markers)

    def taught_by(self, template_source: str) -> bool:
        """Does this template teach the model this dialect?"""
        return any(m in template_source for m in self.template_markers)


# --- Generic <think>...</think> dialect (K2/DeepSeek/Qwen3/MiniMax/...) ---

# No `\s*` after `</think>`. The newline a model puts before its answer is
# not a marker this dialect declares, so it survives -- see
# `ReasoningChannel.split`.
THINK_OPEN_MARKER = "<think>"
THINK_END_MARKER = "</think>"

# MiniMax-M3 spells the same shape with its own tags. Its template sets
# `think_begin_token = '<mm:think>'` and ends the generation prompt with it
# when `thinking_mode == "enabled"`, leaves no prefix under "adaptive" (so the
# model writes the opener itself), and emits `</mm:think>` alone under
# "disabled". Neither literal was declared anywhere, so the whole chain of
# thought was delivered as `content` with `reasoning_content: null`, on both
# delivery paths -- and on `/v1/messages`, inside the client's text block.
MM_THINK_OPEN_MARKER = "<mm:think>"
MM_THINK_END_MARKER = "</mm:think>"


# "Channel" here follows the established meaning from OpenAI's Harmony format:
# one output stream carrying several named sections (think / response / tools),
# each wrapped in framing tokens, that we de-multiplex into separate fields.
# Harmony's analysis/final/commentary channels map onto K3's think/response/tools.
# Channel-format models differ in their framing tokens, so each gets its own
# DIALECTS entry; the entry below carries Kimi-K3's, read from
# `kimi_k3_tokens`, the one module that spells them.
#
# Detection/priority order: structured-channel dialects before inline-tag ones,
# so `resolve_dialect` tries a specific channel marker before the generic
# <think> tag. A dialect is identified by its markers, not a label.

# Read from the module both consumers read, not restated here -- a second
# hand-kept copy is how the two lists drifted apart. The tool-region brackets
# stay out: the reasoning stage cannot remove what is between them (see
# `kimi_k3_tokens`).
_CHANNEL_CONTENT_FRAMING = k3.CHANNEL_FRAMING

_CHANNEL_DIALECT = ReasoningDialect(
    prompt_open_marker=k3.THINK_START,
    # The same literal, not `None`. "The template injects it, so the model
    # never writes one" is false when the prompt leaves thinking off and the
    # model opens a channel anyway -- and `ReasoningFilter` builds
    # `_open_markers` from this field, so state 0 had nothing to leave itself
    # on and delivered the whole trace as the answer.
    output_open_marker=k3.THINK_START,
    think_end_marker=k3.THINK_END,
    # `response` is what the trained format always opens next -- Kimi-K3's
    # encoder emits a complete response section before any tools section, so
    # `tools` is reachable only from a malformed generation. Listed anyway,
    # and cheaply: without it a think block that jumps straight to
    # `<|open|>tools<|sep|>` puts the entire tool call in `reasoning_content`
    # with empty `content`, on both delivery paths, and on `/v1/messages`
    # where reasoning is dropped by default the client gets nothing at all.
    extra_end_markers=(k3.RESPONSE_START, k3.TOOLS_START),
    # The answer's closer, which the trained format writes only after
    # `<|open|>response<|sep|>`.
    answer_end_markers=(k3.RESPONSE_END,),
    answer_open_marker=k3.RESPONSE_START,
    template_efforts=frozenset({"low", "high", "max"}),  # Kimi-K3
    template_markers=(k3.THINK_START, k3.THINK_END),
    content_framing=_CHANNEL_CONTENT_FRAMING,
)

DIALECTS: tuple[ReasoningDialect, ...] = (
    # Structured channel format — Kimi-K3's token values
    _CHANNEL_DIALECT,
    # MiniMax-M3: the inline-tag shape in its own spelling. Before the generic
    # entry, so its longer markers are matched first -- though the two sets do
    # not overlap as substrings, which is checked below.
    ReasoningDialect(
        prompt_open_marker=MM_THINK_OPEN_MARKER,
        output_open_marker=MM_THINK_OPEN_MARKER,
        think_end_marker=MM_THINK_END_MARKER,
        template_markers=(MM_THINK_OPEN_MARKER, MM_THINK_END_MARKER),
    ),
    # Generic <think>...</think> (K2/DeepSeek/Qwen3/...)
    ReasoningDialect(
        prompt_open_marker=THINK_OPEN_MARKER,
        output_open_marker=THINK_OPEN_MARKER,
        think_end_marker=THINK_END_MARKER,
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
