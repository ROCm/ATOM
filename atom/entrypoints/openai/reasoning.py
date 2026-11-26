# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reasoning/thinking content separation for thinking models.

A dialect-agnostic engine: one reader (``ReasoningFilter``) separates the
reasoning channel from the final answer, for a token stream and for a
complete response alike (``ReasoningChannel.split``), emitting the standard
``reasoning_content`` field.
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
from .logprobs import LogprobText, join_text
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
# The most a replayed reply is sent as in one event, so a long one reaches
# the tool parser and the client in pieces rather than one oversized frame.
_REPLAY_PIECE = 4096


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


class ReasoningFilter:
    """Stateful streaming filter that separates reasoning from content.

    Chunks in, ``(field, text)`` tuples out, where field is
    ``"reasoning_content"`` or ``"content"``. Three states: before the channel
    opens, inside it, after it.

    ``starts_thinking`` handles templates that inject the opening marker into
    the prompt itself (Kimi-K3 ends the prompt with its think opener): the
    output then begins *inside* the channel with no opening tag, so nothing in
    the text says so and the filter must be told.

    ``answer_started`` is for a prompt that has begun the answer itself (an
    assistant prefix's content): the output is read as the text after the
    channel's end marker is, its framing removed and any think markers in it
    the model's own words.

    ``dialect`` is the one this model speaks. Go through
    :class:`ReasoningChannel` rather than passing it by hand.

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
        answer_started: bool = False,
    ) -> None:
        self.starts_thinking = starts_thinking
        self.dialect = dialect
        speaking = dialect or FALLBACK_DIALECT
        self._think_end = speaking.think_end_marker
        self._end_markers = frozenset(speaking.end_markers)
        self._answer_end_markers = frozenset(speaking.answer_end_markers)
        self._framing = frozenset(speaking.content_framing)
        # End markers after which the answer has a channel of its own: the
        # think exit, and one that is itself answer framing (K3's
        # `<|open|>response<|sep|>`). A channel that ends at one of these was
        # not the reply, whatever its last words were.
        self._exits_to_answer = frozenset(
            m for m in self._end_markers if m == self._think_end or m in self._framing
        )
        self._open_markers = frozenset(
            (speaking.output_open_marker,) if speaking.output_open_marker else ()
        )
        self.state = (
            1 if starts_thinking else (2 if literal_answer or answer_started else 0)
        )
        # Nothing has been released yet, so an end marker arriving now sits at
        # offset 0 -- the only place it means "I did not think this turn"
        # (MiniMax-M3's trained form for a no-thinking turn). Anywhere else it
        # is the model's own words.
        self._at_offset_zero = self.state == 0
        # The channel's text so far, kept only by a dialect whose answer can
        # close inside the channel, in case the channel turns out to have been
        # the reply. One string, dropped when the channel ends.
        self._channel_text = io.StringIO() if self._answer_end_markers else None
        self._channel_parts = None
        # The channel's last words, whitespace and framing aside, were the
        # answer's closer.
        self._answer_closed = False
        self._scanner = (
            None
            if literal_answer and not starts_thinking
            else self._scanner_for(self.state)
        )

    @property
    def held_text(self) -> str:
        """Unresolved marker prefix that may still become visible text."""
        return self._scanner.held if self._scanner is not None else ""

    def _scanner_for(self, state: int) -> MarkerScanner | None:
        """What this state has to watch for. ``None`` when that is nothing,
        which is the common case for the inline-`<think>` dialect after the
        channel has closed."""
        watched = set(self._framing)
        if state == 0:
            # End markers too, though the channel is not open. The think exit
            # at offset 0 means the model declined to think, and further in it
            # is the model's own words; another one (K3's tools opener) means
            # the answer's channels have begun, wherever it falls.
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
            # After a hit the scanner keeps the text past it, so the next
            # pass reads on with `text` empty.
            scan = self._scanner.feed(text)
            text = ""
            if scan.released:
                out.append((self._field, scan.released))
                if self.state == 0:
                    self._at_offset_zero = False
                elif self.state == 1:
                    self._keep(scan.released)
            if scan.hit is None:
                return out
            if self.state == 1 and scan.hit in self._answer_end_markers:
                # Decided by what follows: see `_close_channel`.
                self._answer_closed = True
                continue
            if scan.hit in self._framing and not (
                self.state == 1 and scan.hit in self._end_markers
            ):
                # Framing this format wraps every answer in. Removed here and
                # not only by the tool parser: doing it there alone made the
                # answer depend on which tool-call format was resolved, so a
                # K3 deployment whose template refused the tools probe handed
                # its channel tokens to the client on both delivery paths.
                continue
            if self.state == 0 and scan.hit in self._end_markers:
                if scan.hit != self._think_end:
                    # A channel that only follows the answer's start (K3's
                    # tools): no think channel comes after it, so a think
                    # marker inside the call is the call's text. Forwarded,
                    # so the tool parser sees its opener.
                    out.append((self._field, scan.hit))
                    text = self._enter(2)
                elif self._at_offset_zero:
                    text = self._enter(2)  # declined to think; the rest is the answer
                else:
                    out.append((self._field, scan.hit))  # the model's own words
                continue
            # A channel boundary: state 0 -> 1 on the opener, 1 -> 2 on any
            # end marker. A dialect that opens the channel from the prompt has
            # no opener to see, so it starts in state 1 and only ever takes
            # the second of these.
            if self.state == 1:
                out.extend(self._close_channel(scan.hit, stopped=True))
            text = self._enter(1 if self.state == 0 else 2)

    def _enter(self, state: int) -> str:
        """Move to `state`, whose markers differ; return the text the old
        state's scanner had not read yet, for the new one."""
        rest = self._scanner.take_rest() if self._scanner is not None else ""
        self.state = state
        self._scanner = self._scanner_for(state)
        return rest

    def _keep(self, released: str) -> None:
        """Channel text, kept for the reply it may turn out to be.

        Words after the answer's closer mean it did not end the reply;
        whitespace does not. The text is looked at only once a closer has been
        seen, so the common case is one write. The one place this is decided:
        the text the scanner holds at the end of the output is a marker's
        first bytes cut off, not words, so it is never kept.
        """
        if self._channel_text is None:
            return
        if isinstance(released, LogprobText):
            if self._channel_parts is None:
                self._channel_parts = []
            self._channel_parts.append(released)
        else:
            self._channel_text.write(released)
        if self._answer_closed and not released.isspace():
            self._answer_closed = False

    def _close_channel(self, at: str | None, *, stopped: bool) -> list:
        """The channel has ended, at end marker ``at`` or, for ``None``, at
        the end of the output, which ``stopped`` says the model chose.

        If its last words, whitespace and framing aside, were the answer's
        closer, it did not end by handing over to the answer's own channel,
        and the output was not cut off, the model skipped the channel and
        wrote its reply there. That text already went out as reasoning and
        cannot be recalled, so it goes out again as `content`.
        """
        kept, self._channel_text = self._channel_text, None  # never reopens
        parts, self._channel_parts = self._channel_parts, None
        if not self._answer_closed or at in self._exits_to_answer or not stopped:
            return []
        reply = join_text(parts) if parts is not None else kept.getvalue()
        return [
            ("content", reply[i : i + _REPLAY_PIECE])
            for i in range(0, len(reply), _REPLAY_PIECE)
        ]

    def flush(self, *, stopped: bool) -> list:
        """End of stream: whatever is held never became a marker.

        ``stopped`` is :func:`~.protocol.ended_by_stop` of the finish reason. An output
        cut off at `max_tokens` right after a quoted answer closer looks like a
        finished reply, and only the finish reason tells the two apart.
        """
        rest = self._scanner.flush() if self._scanner is not None else ""
        self._scanner = None
        out = [(self._field, rest)] if rest else []
        if self.state == 1:
            out.extend(self._close_channel(None, stopped=stopped))
        return out

    def read_whole_events(self, text: str, *, stopped: bool) -> list:
        """A complete output's events, flushed: the whole output in one feed,
        which `MarkerScanner` reads in time linear in its length."""
        return self.process(text) + self.flush(stopped=stopped)


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

    ``answer_started``: the prompt has begun the answer itself -- an
    assistant prefix's content (``prefix_completion``) -- so the output reads
    as it would after the model's own end marker.
    """

    dialect: ReasoningDialect | None = None
    starts_open: bool = False
    literal_answer: bool = False
    answer_started: bool = False

    def split(self, text: str, *, stopped: bool) -> tuple[str | None, str]:
        """A complete output as ``(reasoning, content)``: the stream's own
        events, so the two delivery modes cannot differ. Each half is
        byte-for-byte what the model wrote minus the markers the dialect
        declares, except that a replayed channel is in both; ``reasoning`` is
        ``None`` when there is none. ``stopped`` is as for
        :meth:`ReasoningFilter.flush`."""
        events = self.stream().read_whole_events(text, stopped=stopped)
        join = join_text if isinstance(text, LogprobText) else "".join
        reasoning = join(v for k, v in events if k == "reasoning_content")
        content = join(v for k, v in events if k == "content")
        return reasoning or None, content

    def stream(self) -> ReasoningFilter:
        """A filter over the same output arriving in chunks."""
        return ReasoningFilter(
            starts_thinking=self.starts_open,
            dialect=self.dialect,
            literal_answer=self.literal_answer,
            answer_started=self.answer_started,
        )


# For a caller with no model behind it -- tests, and the completions endpoint,
# which has no chat template and therefore no reasoning channel to read.
NO_REASONING = ReasoningChannel()
