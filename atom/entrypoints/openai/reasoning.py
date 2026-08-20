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

from .reasoning_dialects import DIALECTS

# Marker tables derived from the dialect registry (no model literals here).
_THINK_END_MARKERS = tuple(d.think_end_marker for d in DIALECTS)
# Markers a rendered prompt ends with when the template already opened reasoning
# (the output then begins inside the reasoning channel with no opening tag).
_REASONING_OPEN_MARKERS = tuple(d.prompt_open_marker for d in DIALECTS)
# Markers the model itself emits mid-output to open reasoning (e.g. "<think>").
_OUTPUT_OPEN_MARKERS = tuple(
    d.output_open_marker for d in DIALECTS if d.output_open_marker
)
# Reasoning-effort levels accepted across all loaded dialects' chat templates.
# resolve_thinking() clamps a request's effort to this set before forwarding it,
# so an effort no template understands is never passed through.
VALID_TEMPLATE_EFFORTS = frozenset().union(*(d.template_efforts for d in DIALECTS))


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


def _hold_back_len(buf: str, markers) -> int:
    """Length of the longest suffix of ``buf`` that is a strict prefix of any
    marker, so a marker split across chunk boundaries isn't emitted as text."""
    n = 0
    for m in markers:
        limit = min(len(buf), len(m) - 1)
        for k in range(limit, 0, -1):
            if m.startswith(buf[-k:]):
                n = max(n, k)
                break
    return n


def separate_reasoning(text: str) -> tuple[str | None, str]:
    """Separate reasoning content from the final answer.

    Tries each registered dialect in priority order; the first that applies wins.

    Returns:
        Tuple of (reasoning_content, content). reasoning_content is None if no
        thinking block was found.
    """
    for dialect in DIALECTS:
        result = dialect.split(text)
        if result is not None:
            return result
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
    """

    state: int = 0
    buf: str = ""
    starts_thinking: bool = False

    def __post_init__(self):
        if self.starts_thinking and self.state == 0:
            self.state = 1

    def _close_thinking(self, idx: int, marker: str) -> list:
        """A think-end marker was found at ``idx``: emit everything before it as
        reasoning, switch to content (state 2), and process anything after."""
        results = []
        reasoning = self.buf[:idx]
        after = self.buf[idx + len(marker) :].lstrip("\n")
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
        idx, marker = _earliest_marker(self.buf, _THINK_END_MARKERS)
        if idx != -1:
            return self._close_thinking(idx, marker)
        hold = _hold_back_len(self.buf, _THINK_END_MARKERS)
        emit = self.buf[: len(self.buf) - hold] if hold else self.buf
        self.buf = self.buf[len(self.buf) - hold :] if hold else ""
        return [("reasoning_content", emit)] if emit else []

    def process(self, text: str) -> list:
        """Process a chunk of text and return list of (field, text) tuples."""
        results = []

        if self.state == 0:
            self.buf += text
            # A reasoning opener emitted in the output (e.g. "<think>").
            oidx, omark = _earliest_marker(self.buf, _OUTPUT_OPEN_MARKERS)
            if oidx != -1:
                before = self.buf[:oidx]
                if before:
                    results.append(("content", before))
                self.state = 1
                self.buf = self.buf[oidx + len(omark) :]
                results.extend(self._drain_thinking())
            else:
                # No explicit opener, but a think-end marker means the model
                # started reasoning without one (template injected the opener).
                idx, marker = _earliest_marker(self.buf, _THINK_END_MARKERS)
                if idx != -1:
                    results.extend(self._close_thinking(idx, marker))
                elif len(self.buf) > 100:
                    # No reasoning markers after significant buffering — emit as
                    # content. The threshold gives a model whose template
                    # injected the opener time to reach its end marker.
                    #
                    # Only a partial marker at the *end* of the buffer is worth
                    # holding: it may complete on the next chunk. Anything
                    # earlier cannot become a marker any more. Testing instead
                    # for a bare "<" anywhere withheld the entire stream for as
                    # long as the answer happened to contain one — and code,
                    # markup and comparisons are full of them, so a request
                    # would return 200 and then never emit a byte while the
                    # engine decoded to max_tokens.
                    hold = _hold_back_len(
                        self.buf, _OUTPUT_OPEN_MARKERS + _THINK_END_MARKERS
                    )
                    emit = self.buf[: len(self.buf) - hold] if hold else self.buf
                    if emit:
                        results.append(("content", emit))
                        self.buf = self.buf[len(self.buf) - hold :] if hold else ""

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
