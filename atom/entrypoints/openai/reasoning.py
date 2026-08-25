# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reasoning/thinking content separation for thinking models (e.g., Kimi-K2, DeepSeek-R1).

This module provides utilities to separate <think>...</think> reasoning blocks
from the final answer, following the SGLang/vLLM reasoning_content pattern.
Also strips raw tool call tokens that the model may output.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .stream_buffer import releasable_prefix_len

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

#: Open/close marker pairs used by the thinking models ATOM serves. MiniMax-M3
#: spells them ``<mm:think>``/``</mm:think>`` and keeps them in the vocabulary
#: *without* marking them special, so they survive
#: ``decode(skip_special_tokens=True)`` and reach this filter verbatim.
THINK_MARKER_PAIRS = ((THINK_OPEN, THINK_CLOSE), ("<mm:think>", "</mm:think>"))
THINK_OPEN_MARKERS = tuple(open_marker for open_marker, _ in THINK_MARKER_PAIRS)
THINK_CLOSE_MARKERS = tuple(close_marker for _, close_marker in THINK_MARKER_PAIRS)
_THINK_MARKERS = THINK_OPEN_MARKERS + THINK_CLOSE_MARKERS


def find_first_marker(text: str, markers: Tuple[str, ...]) -> Tuple[int, Optional[str]]:
    """``(index, marker)`` of the earliest marker present, else ``(-1, None)``."""
    best, found = -1, None
    for marker in markers:
        index = text.find(marker)
        if index != -1 and (best == -1 or index < best):
            best, found = index, marker
    return best, found


# How much text to buffer in state 0 before giving up on seeing a thinking
# marker. Models whose chat template injected <think> emit only the closing
# </think>; the grace period gives them room to do so before their reasoning is
# misfiled as content. prompt_primes_thinking() is the reliable signal —
# this is the fallback for prompts ATOM cannot inspect (custom encoders).
_CONTENT_HOLD_CHARS = 100


def prompt_primes_thinking(prompt: Optional[str]) -> bool:
    """Whether a rendered prompt ends inside an open ``<think>`` block.

    MiniMax-M3 (``thinking_mode="enabled"``) and DeepSeek-R1 chat templates
    append their open thinking marker to the generation prompt, so the model
    starts generating *inside* the block and only the closing marker appears on
    the wire. A streaming filter has to start in "already thinking" state or the
    leading reasoning text is emitted as ``content``.
    """
    return bool(prompt) and prompt.rstrip().endswith(THINK_OPEN_MARKERS)


def separate_reasoning(text: str, primed: bool = False) -> Tuple[Optional[str], str]:
    """Separate reasoning content from the final answer.

    Args:
        text: Raw model output that may contain <think>...</think> blocks.
        primed: The chat template left a ``<think>`` block open at the end of
            the prompt (see :func:`prompt_primes_thinking`), so the output starts
            inside it. Reconstructing the opening marker makes a *truncated*
            response resolve to reasoning rather than being misfiled as the
            answer, and matches what the streaming filter does.

    Returns:
        Tuple of (reasoning_content, content). reasoning_content is None if
        no thinking block was found.
    """
    # Thinking block opened in the output: <open>reasoning</close>answer, or
    # unclosed when the response was truncated mid-thought.
    for open_marker, close_marker in THINK_MARKER_PAIRS:
        if text.startswith(open_marker):
            rest = text[len(open_marker) :]
            reasoning, closed, content = rest.partition(close_marker)
            if closed:
                return (reasoning.strip() or None, content.strip())
            return (rest.strip() or None, "")

    # A close marker with no open one: the chat template primed the block, so
    # everything up to it is reasoning.
    close_at, close_marker = find_first_marker(text, THINK_CLOSE_MARKERS)
    if close_marker is not None:
        reasoning = text[:close_at].strip()
        content = text[close_at + len(close_marker) :].strip()
        return (reasoning or None, content)

    # Primed but never closed: the whole (truncated) response is reasoning.
    if primed:
        return (text.strip() or None, "")

    # No thinking block — return content as-is (tool calls parsed separately)
    return (None, text)


@dataclass
class ReasoningFilter:
    """Stateful streaming filter that separates reasoning from content.

    Processes tokens one chunk at a time and yields (field, text) tuples
    where field is either "reasoning_content" or "content".

    States:
        0 = before <think> (buffering to detect)
        1 = inside <think> (emitting as reasoning_content)
        2 = after </think> (emitting as content)

    Use :meth:`for_stream` rather than the bare constructor: which state a
    stream starts in depends on whether the chat template primed ``<think>``
    and on whether the request asked for thinking at all.
    """

    state: int = 0
    buf: str = ""

    @classmethod
    def for_stream(
        cls, *, enabled: bool = True, primed: bool = False
    ) -> "ReasoningFilter":
        """Build a filter for one streamed choice.

        ``primed`` (from :func:`prompt_primes_thinking`) starts the filter inside
        the thinking block, because the model's first token already is reasoning.

        ``enabled=False`` (the request disabled thinking) starts in state 2 so
        every token is content: the prompt already closed the thinking block, so
        a stray ``</think>`` in the output must not resurrect the
        ``reasoning_content`` the client asked not to receive.
        """
        if not enabled:
            return cls(state=2)
        return cls(state=1 if primed else 0)

    def process(self, text: str) -> list:
        """Process a chunk of text and return list of (field, text) tuples.

        Args:
            text: New text chunk from the model.

        Returns:
            List of (field_name, text) tuples where field_name is
            "reasoning_content" or "content".
        """
        results = []

        if self.state == 0:
            self.buf += text
            open_at, open_marker = find_first_marker(self.buf, THINK_OPEN_MARKERS)
            close_at, close_marker = find_first_marker(self.buf, THINK_CLOSE_MARKERS)
            if open_marker is not None and (close_marker is None or open_at < close_at):
                before = self.buf[:open_at]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf[open_at + len(open_marker) :]
                results.extend(self._drain_reasoning())
            elif close_marker is not None:
                # A close marker with no open one: the chat template primed the
                # thinking block, so what arrived first is reasoning.
                reasoning = self.buf[:close_at]
                after = self.buf[close_at + len(close_marker) :].lstrip("\n")
                if reasoning:
                    results.append(("reasoning_content", reasoning))
                self.state = 2
                self.buf = ""
                if after:
                    results.extend(self._process_content(after))
            elif len(self.buf) > _CONTENT_HOLD_CHARS:
                # No think tags after the grace period — emit as content, but
                # hold back a tail that could still be a partial marker. Bare
                # '<' in the answer (code, math, HTML) must not stall the stream
                # until end of generation.
                cut = releasable_prefix_len(self.buf, _THINK_MARKERS)
                if cut > 0:
                    results.append(("content", self.buf[:cut]))
                    self.buf = self.buf[cut:]

        elif self.state == 1:
            self.buf += text
            results.extend(self._drain_reasoning())

        else:  # state == 2
            results.extend(self._process_content(text))

        return results

    def _drain_reasoning(self) -> list:
        """State 1: emit reasoning, switching to content once the block closes.

        Holds back a tail that could still be a partial close marker — without
        that, a ``</think>`` split across two chunks leaks its first half into
        ``reasoning_content`` and is then never recognised as the delimiter.
        """
        results: list = []
        close_at, close_marker = find_first_marker(self.buf, THINK_CLOSE_MARKERS)
        if close_marker is not None:
            reasoning = self.buf[:close_at]
            after = self.buf[close_at + len(close_marker) :].lstrip("\n")
            if reasoning:
                results.append(("reasoning_content", reasoning))
            self.state = 2
            self.buf = ""
            if after:
                results.extend(self._process_content(after))
            return results
        cut = releasable_prefix_len(self.buf, THINK_CLOSE_MARKERS)
        if cut > 0:
            results.append(("reasoning_content", self.buf[:cut]))
            self.buf = self.buf[cut:]
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
