# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reasoning/thinking content separation for thinking models.

A dialect-agnostic engine: it separates the reasoning channel from the final
answer, both for a complete response (``separate_reasoning``) and for a token
stream (``ReasoningFilter``), emitting the standard ``reasoning_content`` field.
All model-specific marker knowledge lives in ``reasoning_dialects.DIALECTS``;
this module contains no per-model conditions. Add a model there, not here.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Called directly and not through a local wrapper: the tool-call path asks the
# same question, and a rename in between is how the two sides drifted apart in
# the first place. It also carries the first-character reject this path had
# gone without, which it pays for once per token on every stream -- including
# models whose reasoning markers never appear.
from .marker_scanner import held_suffix_len
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
    both (Qwen3) describes a model that opens its own, and one that mentions
    neither (MiniMax-M3, gpt-oss) has no reasoning channel to speak of.

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


def prompt_starts_in_reasoning(prompt: str) -> bool:
    """True if the rendered ``prompt`` ends by opening a reasoning channel.

    Model-agnostic: callers pass the rendered prompt and don't need to know which
    dialect's marker applies. Used to seed the streaming filter
    (:attr:`ReasoningFilter.starts_thinking`)."""
    p = prompt.rstrip()
    return any(p.endswith(m) for m in _REASONING_OPEN_MARKERS)


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


def _earliest_marker(buf: str, markers) -> tuple[int, str | None]:
    """Return (index, marker) of the earliest-occurring marker in ``buf``."""
    best_i, best_m = -1, None
    for m in markers:
        i = buf.find(m)
        if i != -1 and (best_i == -1 or i < best_i):
            best_i, best_m = i, m
    return best_i, best_m


def separate_reasoning(
    text: str,
    starts_thinking: bool = False,
    dialect: ReasoningDialect | None = None,
) -> tuple[str | None, str]:
    """Separate reasoning content from the final answer.

    ``dialect`` is the one this model speaks, resolved from its chat template
    at startup (:func:`~.reasoning_dialects.resolve_dialect`). One, not each in
    turn: trying them in order let any model's answer be split by a dialect it
    does not speak, so an answer *about* Kimi's wire format lost the text
    before a quoted channel token -- and the streaming filter, which used a
    different rule again, kept it. Callers should go through
    :class:`ReasoningChannel`, which carries this and ``starts_thinking``
    together so the two paths cannot be given different ones.

    ``starts_thinking`` is the same answer :class:`ReasoningFilter` takes, and
    for the same reason: an output that begins inside the reasoning channel
    carries no opening marker, so nothing in the text says so. Both paths have
    to be told, or the same response is split one way when streamed and
    another when not -- measured, a reasoning model truncated at ``max_tokens``
    returned its whole trace as ``content`` here and as ``reasoning_content``
    when streamed.

    Returns:
        Tuple of (reasoning_content, content). reasoning_content is None if no
        thinking block was found.
    """
    result = (dialect or FALLBACK_DIALECT).split(text, starts_thinking)
    if result is not None:
        return result
    if starts_thinking:
        # The prompt opened the channel and the model never closed it, which
        # is what a reasoning model stopped at `max_tokens` looks like. It
        # produced reasoning and no answer; the streaming filter says exactly
        # that from state 1, and this has to agree.
        #
        # `or None` because it has to agree on the empty case too: the filter
        # emits no segment at all for an empty output, so a bare `text` here
        # put `reasoning_content: ""` in the non-streaming body and nothing in
        # the streamed one. Every other return in this function spells absence
        # as `None`.
        return (text or None, "")
    # No reasoning markers — return content as-is (tool calls parsed separately).
    return (None, text)


@dataclass
class ReasoningFilter:
    """Stateful streaming filter that separates reasoning from content.

    Processes tokens one chunk at a time and yields (field, text) tuples where
    field is either "reasoning_content" or "content". Dialect-agnostic: reasoning
    openers/terminators come from the registry-derived marker tables.

    States:
        0 = before reasoning opens (buffering to detect)
        1 = inside reasoning (emitting as reasoning_content)
        2 = after reasoning (emitting as content)

    ``starts_thinking`` handles templates that inject the opening reasoning marker
    into the prompt itself (e.g. Kimi-K3 ends the prompt with its think opener):
    the output then begins *inside* the reasoning channel with no opening tag, so
    the filter must start in state 1.

    ``dialect`` is the one this model speaks, and it is the same object
    :func:`separate_reasoning` is given. This used to hold no dialect at all --
    it closed the channel on the *union* of every registered dialect's end
    markers -- so a `<think>` model quoting a K3 channel token ended its chain
    of thought there while the non-streaming split ended it at the real
    `</think>`. Go through :class:`ReasoningChannel` rather than passing this
    by hand; that is what keeps the two paths on one answer.
    """

    state: int = 0
    buf: str = ""
    starts_thinking: bool = False
    dialect: ReasoningDialect | None = None

    def __post_init__(self):
        if self.starts_thinking and self.state == 0:
            self.state = 1
        speaking = self.dialect or FALLBACK_DIALECT
        self._end_markers = speaking.end_markers
        self._open_markers = (
            (speaking.output_open_marker,) if speaking.output_open_marker else ()
        )

    def _close_thinking(self, idx: int, marker: str) -> list:
        """A think-end marker was found at ``idx``: emit everything before it as
        reasoning, switch to content (state 2), and process anything after."""
        results = []
        reasoning = self.buf[:idx]
        # Byte-for-byte, like the non-streaming split -- and this one could
        # not have been made to agree anyway: it saw only what happened to be
        # buffered when the marker arrived, so the same answer kept its
        # newlines at one chunk size and lost them at another.
        after = self.buf[idx + len(marker) :]
        if reasoning:
            results.append(("reasoning_content", reasoning))
        self.state = 2
        self.buf = ""
        if after:
            results.extend(self._process_content(after))
        return results

    def _drain_thinking(self) -> list:
        """State-1 helper: emit buffered reasoning up to a think-end marker; on
        match switch to content. Otherwise emit what's safe, holding back a
        partial trailing marker so it isn't split across chunks."""
        idx, marker = _earliest_marker(self.buf, self._end_markers)
        if idx != -1:
            return self._close_thinking(idx, marker)
        hold = held_suffix_len(self.buf, self._end_markers)
        emit = self.buf[: len(self.buf) - hold] if hold else self.buf
        self.buf = self.buf[len(self.buf) - hold :] if hold else ""
        return [("reasoning_content", emit)] if emit else []

    def process(self, text: str) -> list:
        """Process a chunk of text and return list of (field, text) tuples."""
        results = []

        if self.state == 0:
            self.buf += text
            # A reasoning opener emitted in the output (e.g. "<think>").
            oidx, omark = _earliest_marker(self.buf, self._open_markers)
            if oidx != -1:
                before = self.buf[:oidx]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf[oidx + len(omark) :]
                results.extend(self._drain_thinking())
            else:
                # No opener, so this is the answer: release it, holding back
                # only a suffix that could still grow into one.
                #
                # State 0 no longer honours a `</think>` it never saw opened,
                # and no longer buffers 100 characters hoping for one. Both
                # were the same guess -- that the template had injected the
                # opener -- made at run time about something the prompt
                # answers: `starts_thinking` is that answer, and a filter in
                # state 0 has been told the output does *not* begin inside the
                # reasoning channel. The guess cost an unbounded first byte
                # (its `"<" not in self.buf` gate could never be satisfied
                # again once an ordinary answer contained a '<') and it let
                # pre-`</think>` text reach the tool-call sniffer, which is
                # how reasoning that merely mentions `<tool_call>` came to be
                # emitted as a tool call. vLLM's streaming path resolves this
                # from the token vocabulary and buffers nothing; this is the
                # same position, reached from the prompt.
                hold = held_suffix_len(self.buf, self._open_markers)
                cut = len(self.buf) - hold
                if cut:
                    results.append(("content", self.buf[:cut]))
                    self.buf = self.buf[cut:]

        elif self.state == 1:
            self.buf += text
            results.extend(self._drain_thinking())

        else:  # state == 2
            results.extend(self._process_content(text))

        return results

    def _process_content(self, text: str) -> list:
        """Process content after thinking. Tool calls are handled by ToolCallStreamParser."""
        if text:
            return [("content", text)]
        return []

    def flush(self) -> list:
        """Flush any remaining buffered content."""
        results = []
        if self.buf:
            if self.state == 0:
                results.append(("content", self.buf))
            elif self.state == 1:
                results.append(("reasoning_content", self.buf))
            self.buf = ""
        return results


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

    def split(self, text: str) -> tuple[str | None, str]:
        """A complete output as ``(reasoning, content)``."""
        return separate_reasoning(text, self.starts_open, self.dialect)

    def stream(self) -> ReasoningFilter:
        """A filter over the same output arriving in chunks."""
        return ReasoningFilter(starts_thinking=self.starts_open, dialect=self.dialect)


# For a caller with no model behind it -- tests, and the completions endpoint,
# which has no chat template and therefore no reasoning channel to read.
NO_REASONING = ReasoningChannel()
