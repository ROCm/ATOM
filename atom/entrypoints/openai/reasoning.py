# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reasoning/thinking content separation for thinking models.

A dialect-agnostic engine: it separates the reasoning channel from the final
answer for a token stream (``ReasoningFilter``), and for a complete response
(``separate_reasoning``) by running that same filter over it in one chunk,
emitting the standard ``reasoning_content`` field.
All model-specific marker knowledge lives in ``reasoning_dialects.DIALECTS``;
this module contains no per-model conditions. Add a model there, not here.
"""

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# The same reader the tool-call side uses, not a second one: this module's
# whole thesis is that asking "how much can be released without splitting a
# marker" in more than one place is how the answers drift apart, and it used
# to answer it itself, twice, with a different tie-break.
from .marker_scanner import MarkerScanner
from .reasoning_dialects import DIALECTS, FALLBACK_DIALECT, ReasoningDialect

# Markers a rendered prompt ends with when the template already opened reasoning
# (the output then begins inside the reasoning channel with no opening tag).
# Every dialect's, because this is asked of a *prompt* before the response's
# dialect matters, and a prompt carries at most one of them.
_REASONING_OPEN_MARKERS = tuple(d.prompt_open_marker for d in DIALECTS)
# Reasoning-effort levels accepted across all loaded dialects' chat templates.
# resolve_thinking() clamps a request's effort to this set before forwarding it,
# so an effort no template understands is never passed through.
VALID_TEMPLATE_EFFORTS = frozenset().union(*(d.template_efforts for d in DIALECTS))


def template_opens_reasoning_implicitly(template_source: str) -> bool:
    """Does this model begin inside the reasoning channel with no marker at all?

    Some families close a reasoning block they never open: DeepSeek-R1 emits
    `</think>` but neither its prompt nor its output carries `<think>`. Nothing
    in a single response says so -- the first token is already reasoning and
    looks like an answer -- so it has to be known before the response starts.

    The chat template says it. A template that mentions an end marker and not
    the matching opener is describing exactly that shape; one that mentions
    both (Qwen3, MiniMax-M3) describes a model that opens its own, and one
    that mentions neither (gpt-oss) has no reasoning channel to speak of.

    ``template_source`` is the template's own text, which is the only place
    this shows: an end marker is what the template does with a *reply*, so it
    never reaches a fresh prompt. Measured -- Qwen3.5's source carries both
    markers while its rendered prompt carries only the opener, and Qwen3-8B's
    rendered prompt carries neither, so asking a render would answer False for
    every model alive. Get it from `chat_encoders.chat_template_source`, which
    also handles the two shapes that answer False by accident: a ``dict`` of
    named templates, and the ``None`` of a model that ships a Python encoder.

    This is what vLLM expresses by registering `DeepSeekR1ReasoningParser` for
    R1 -- an override whose only job is to treat a stream with no start token
    as reasoning until `</think>`. Same fact, derived instead of listed.
    """
    for dialect in DIALECTS:
        if not dialect.think_end_marker:
            continue
        opener = dialect.output_open_marker or dialect.prompt_open_marker
        if dialect.think_end_marker in template_source and (
            not opener or opener not in template_source
        ):
            return True
    return False


def thinking_switched_off(
    template_kwargs: dict | None, toggle: tuple[str, Any, Any] | None
) -> bool:
    """Did the render actually carry this template's off-switch?

    The merged kwargs are the whole answer -- server defaults, the client's
    `chat_template_kwargs`, and the request's `thinking` written in by the
    toggle's name on top. Reading the request field alone missed the first
    two, so an operator's `--default-chat-template-kwargs` never reached the
    decision.

    Only the resolved toggle's own name and value count: a kwarg the template
    does not read changes nothing about what the model does, and believing it
    would stop the channel being separated while the model went on producing
    one.
    """
    if not template_kwargs or toggle is None:
        return False
    name, off_value, _on_value = toggle
    return name in template_kwargs and template_kwargs[name] == off_value


def prompt_starts_in_reasoning(prompt: str) -> bool:
    """True if the rendered ``prompt`` ends by opening a reasoning channel.

    Model-agnostic: callers pass the rendered prompt and don't need to know which
    dialect's marker applies. Used to seed the streaming filter
    (:attr:`ReasoningFilter.starts_thinking`).

    Offsets, not `prompt.rstrip()`, which copies the whole rendered prompt to
    read its last twenty bytes. That prompt is the largest string the server
    handles -- system prompt, tool schemas and the entire history -- and this
    runs once per request on the event loop: 4.8 us at 512 KB against 0.43 us
    here. Same shape as `begin_of_markup`.
    """
    end = len(prompt)
    while end > 0 and prompt[end - 1].isspace():
        end -= 1
    return any(prompt.endswith(m, 0, end) for m in _REASONING_OPEN_MARKERS)


def prompt_tokens_start_in_reasoning(
    token_ids: Sequence[int], decode: Callable[[Sequence[int]], str]
) -> bool:
    """:func:`prompt_starts_in_reasoning` for an already-tokenized prompt.

    Multimodal requests reach the engine as token ids, and decoding all of them
    would be wasteful — an image prompt runs to thousands of tokens while only
    the end is inspected. Decoding as many trailing tokens as the longest marker
    has *characters* is always enough, because a token never renders to fewer
    than one character.

    ``decode`` is injected so this module stays free of tokenizer knowledge.
    """
    if not _REASONING_OPEN_MARKERS or not len(token_ids):
        return False
    tail = max(len(m) for m in _REASONING_OPEN_MARKERS)
    return prompt_starts_in_reasoning(decode(token_ids[-tail:]))


def separate_reasoning(
    text: str,
    starts_thinking: bool = False,
    dialect: ReasoningDialect | None = None,
) -> tuple[str | None, str]:
    """Separate reasoning content from the final answer.

    The streaming filter run over the whole output as one chunk, not a second
    reader: a hand-written split per dialect had to be kept in step with the
    filter, and every rule added to one of them drifted from the other. The
    filter is chunk-invariant, so this is exactly what any chunking of the
    same output streams, and every endpoint and delivery mode agrees by
    construction.

    ``dialect`` is the one this model speaks, resolved from its chat template
    at startup (:func:`~.reasoning_dialects.resolve_dialect`); trying every
    dialect in turn let a model's answer be split by one it does not speak.
    ``starts_thinking`` says the prompt already opened the channel, which
    nothing in the output can. Callers should go through
    :class:`ReasoningChannel`, which carries the two together.

    Returns ``(reasoning_content, content)``, each byte-for-byte what the
    model wrote minus the markers the dialect declares; ``reasoning_content``
    is ``None`` when there is none.
    """
    reader = ReasoningFilter(starts_thinking=starts_thinking, dialect=dialect)
    reasoning: list[str] = []
    content: list[str] = []
    for field, segment in reader.process(text) + reader.flush():
        (reasoning if field == "reasoning_content" else content).append(segment)
    return "".join(reasoning) or None, "".join(content)


class ReasoningFilter:
    """Stateful streaming filter that separates reasoning from content.

    Chunks in, ``(field, text)`` tuples out, where field is
    ``"reasoning_content"`` or ``"content"``. Three states: before the channel
    opens, inside it, after it.

    ``starts_thinking`` handles templates that inject the opening marker into
    the prompt itself (Kimi-K3 ends the prompt with its think opener): the
    output then begins *inside* the channel with no opening tag, so nothing in
    the text says so and the filter must be told.

    ``dialect`` is the one this model speaks, and the same object
    :func:`separate_reasoning` is given. Go through :class:`ReasoningChannel`
    rather than passing it by hand; that is what keeps the two paths on one
    answer.

    Every marker question goes to :class:`MarkerScanner`, the same reader the
    tool-call side uses. It used to have its own: an `_earliest_marker` whose
    tie-break was the opposite of the scanner's (first-declared rather than
    longest, so `<think>` would be reported where `<thinking>` was meant), the
    hold-and-cut arithmetic open-coded twice, and `self.buf += text` on an
    attribute -- the quadratic append `Region` exists to avoid -- on the
    reasoning half of every long chain of thought. All three in a module whose
    first paragraph says asking this question in more than one place is the
    bug being retired.
    """

    def __init__(
        self,
        starts_thinking: bool = False,
        dialect: ReasoningDialect | None = None,
        literal_answer: bool = False,
    ) -> None:
        self.starts_thinking = starts_thinking
        self.dialect = dialect
        speaking = dialect or FALLBACK_DIALECT
        self._end_markers = frozenset(speaking.end_markers)
        self._answer_end_markers = frozenset(speaking.answer_end_markers)
        self._framing = frozenset(speaking.content_framing)
        self._open_markers = frozenset(
            (speaking.output_open_marker,) if speaking.output_open_marker else ()
        )
        self.state = 1 if starts_thinking else (2 if literal_answer else 0)
        # Nothing has been released yet, so an end marker arriving now sits at
        # offset 0 -- the only place it means "I did not think this turn"
        # (MiniMax-M3's trained form for a no-thinking turn). Anywhere else it
        # is the model's own words.
        self._at_offset_zero = not starts_thinking
        # The channel's text so far, kept only by a dialect whose answer can
        # close inside the channel, in case the channel turns out to have been
        # the reply. One string, dropped when the channel ends.
        self._channel_text = io.StringIO() if self._answer_end_markers else None
        # The channel's last words, framing aside, were the answer's closer.
        self._answer_closed = False
        self._scanner = (
            None
            if literal_answer and not starts_thinking
            else self._scanner_for(self.state)
        )

    def _scanner_for(self, state: int) -> MarkerScanner | None:
        """What this state has to watch for. ``None`` when that is nothing,
        which is the common case for the inline-`<think>` dialect after the
        channel has closed."""
        watched = set(self._framing)
        if state == 0:
            # End markers too, though the channel is not open: at offset 0 one
            # means the model declined to think, further in it is the model's
            # own words, and watching is what tells them apart. Free --
            # measured 1.00x on all four dialects, since `MarkerScanner` costs
            # by chunk and not by marker count.
            watched |= self._open_markers | self._end_markers
        elif state == 1:
            watched |= self._end_markers | self._answer_end_markers
        return MarkerScanner(tuple(sorted(watched))) if watched else None

    @property
    def _field(self) -> str:
        return "reasoning_content" if self.state == 1 else "content"

    def process(self, text: str) -> list:
        """Consume one chunk; return what it completed."""
        out: list = []
        while True:
            if self._scanner is None:
                if text:
                    out.append((self._field, text))
                return out
            scan = self._scanner.feed(text)
            text = ""
            if scan.released:
                out.append((self._field, scan.released))
                if self.state == 0:
                    self._at_offset_zero = False
                elif self.state == 1:
                    # More channel text, so an answer closer before it did
                    # not end the reply.
                    self._answer_closed = False
                    if self._channel_text is not None:
                        self._channel_text.write(scan.released)
            if scan.hit is None:
                return out
            if self.state == 1 and scan.hit in self._answer_end_markers:
                # Decided by what follows: see `_close_channel`.
                self._answer_closed = True
                text = scan.rest
                continue
            if scan.hit in self._framing and not (
                self.state == 1 and scan.hit in self._end_markers
            ):
                # Framing this format wraps every answer in. Removed here and
                # not only by the tool parser: doing it there alone made the
                # answer depend on which tool-call format was resolved, so a
                # K3 deployment whose template refused the tools probe handed
                # its channel tokens to the client on both delivery paths.
                text = scan.rest
                continue
            if self.state == 0 and scan.hit in self._end_markers:
                # Below the framing branch on purpose: K3 declares
                # `<|open|>response<|sep|>` as both, and framing wins -- it
                # says the same thing this rule would, at any offset.
                if self._at_offset_zero:
                    self.state = 2  # declined to think; the rest is the answer
                    self._scanner = self._scanner_for(self.state)
                else:
                    out.append((self._field, scan.hit))  # the model's own words
                text = scan.rest
                continue
            # A channel boundary: state 0 -> 1 on the opener, 1 -> 2 on any
            # end marker. A dialect that opens the channel from the prompt has
            # no opener to see, so it starts in state 1 and only ever takes
            # the second of these.
            if self.state == 1:
                out.extend(self._close_channel())
            self.state = 1 if self.state == 0 else 2
            self._scanner = self._scanner_for(self.state)
            text = scan.rest

    def _close_channel(self) -> list:
        """The channel has ended, at an end marker or at the end of the output.

        If its last words, framing aside, were the answer's closer, the model
        skipped the channel and wrote its reply there (Kimi-K3 does this on
        ~1M-token prompts). That text already went out as reasoning and cannot
        be recalled, so it goes out again as `content`; the non-streaming
        split is this filter, so every path returns it twice.
        """
        kept, self._channel_text = self._channel_text, None  # never reopens
        reply = kept.getvalue() if kept is not None and self._answer_closed else ""
        self._answer_closed = False
        return [("content", reply)] if reply else []

    def flush(self) -> list:
        """End of stream: whatever is held never became a marker."""
        rest = self._scanner.flush() if self._scanner is not None else ""
        self._scanner = None
        out = [(self._field, rest)] if rest else []
        if self.state == 1:
            if rest:
                self._answer_closed = False
            out.extend(self._close_channel())
        return out


@dataclass(frozen=True)
class ReasoningChannel:
    """How to read one response's reasoning channel, decided before it starts.

    Two facts, and they have to travel together: which dialect the model
    speaks, and whether its output begins inside the channel already. Passed
    separately, they were passed differently -- the streaming filter got the
    second and never the first, and answered with the union of every dialect's
    markers instead. This is one object with one accessor per delivery mode,
    so the two cannot be handed different answers.

    ``starts_open`` is not simply "this template opens the channel": a request
    that turns thinking off renders a prompt that does not, and OR-ing the
    model-level fact in regardless made an ordinary answer come back as
    ``reasoning_content`` with empty ``content``. The caller resolves it.
    """

    dialect: ReasoningDialect | None = None
    starts_open: bool = False
    literal_answer: bool = False

    def split(self, text: str) -> tuple[str | None, str]:
        """A complete output as ``(reasoning, content)``."""
        if self.literal_answer:
            reader = self.stream()
            events = [*reader.process(text), *reader.flush()]
            reasoning = "".join(v for k, v in events if k == "reasoning_content")
            content = "".join(v for k, v in events if k == "content")
            return reasoning or None, content
        return separate_reasoning(text, self.starts_open, self.dialect)

    def stream(self) -> ReasoningFilter:
        """A filter over the same output arriving in chunks."""
        return ReasoningFilter(
            starts_thinking=self.starts_open,
            dialect=self.dialect,
            literal_answer=self.literal_answer,
        )


# For a caller with no model behind it -- tests, and the completions endpoint,
# which has no chat template and therefore no reasoning channel to read.
NO_REASONING = ReasoningChannel()
