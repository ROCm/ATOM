# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parser interface and the shared buffered-marker streaming strategy.

Every wire format implements :class:`ToolCallParser`. Four of the five
(Qwen / DSML / GLM / MiniMax) stream identically — buffer from the first start
marker, parse the whole block at flush — so that strategy lives once in
:class:`BufferedMarkerParser` and each format only declares its markers and its
``parse``. Kimi is the exception: its token format is self-delimiting, so it
emits tool calls incrementally and implements ``process``/``flush`` itself.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from ..marker_scanner import MarkerScanner


def unique_tool_call_id() -> str:
    # OpenAI tool_call ids must be unique across the whole conversation, not just
    # within one response. A per-response index (call_0, call_1, ...) collides
    # across turns -> clients (e.g. qwen-code) dedupe by id and silently ignore
    # every repeat, causing an infinite tool-call retry loop. Use a random id.
    return f"call_{uuid.uuid4().hex}"


@dataclass
class ToolCall:
    """Parsed tool call in OpenAI format."""

    id: str
    type: str
    function: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "function": self.function}


class ToolCallParser(ABC):
    """One on-the-wire tool-call format.

    Class side is the stateless non-streaming path (``detect`` + ``parse``);
    instance side is the stateful streaming path (``process`` + ``flush``).
    """

    NAME: ClassVar[str]
    # Every literal that opens this format's tool-call region. Declared once,
    # here, rather than spelled out again in each parser's own logic: a reader
    # ahead of detection needs them to know how much of its buffer could still
    # be the start of one, `BufferedMarkerParser` locates the region with them,
    # and the property tests enumerate them so a newly registered format is
    # covered the moment it exists rather than when someone writes a case.
    START_MARKERS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, tools: list | None = None):
        self.tools = tools
        self.buf = ""
        # 0 = still in plain content, 1 = inside a tool-call region. Kimi adds
        # 2 = section closed; see KimiParser.
        self.state = 0
        self.current_index = 0
        self.emitted_calls = 0
        # Only `BufferedMarkerParser` reads it, but every parser is built
        # through here, so a subclass that forgot to initialise it would fail
        # on its first chunk rather than at construction.
        self._scanner_cache = None

    # -- non-streaming ------------------------------------------------------
    @classmethod
    @abstractmethod
    def detect(cls, text: str) -> bool:
        """Whether a complete model output is in this format."""

    @classmethod
    @abstractmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        """Parse a complete output into ``(leading_content, tool_calls)``."""

    # -- streaming ----------------------------------------------------------
    @abstractmethod
    def process(self, text: str) -> list:
        """Consume one chunk; return ``(event_type, data)`` tuples."""

    @abstractmethod
    def flush(self) -> list:
        """Drain whatever is buffered at end of stream."""

    def _emit_call(self, tc: ToolCall) -> list:
        """Render one parsed ToolCall as start+args stream events."""
        events = [
            (
                "tool_call_start",
                {
                    "index": self.current_index,
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function["name"], "arguments": ""},
                },
            ),
            (
                "tool_call_args",
                {
                    "index": self.current_index,
                    "function": {"arguments": tc.function["arguments"]},
                },
            ),
        ]
        self.current_index += 1
        self.emitted_calls += 1
        return events


class BufferedMarkerParser(ToolCallParser):
    """Formats that buffer from a start marker and parse the block at flush.

    The block is only parsed once complete because partial XML streams badly
    (a half-written ``<parameter=`` would emit garbage). Content before the
    first marker still streams normally.

    Subclasses declare ``START_MARKERS`` and implement ``parse``.
    """

    @classmethod
    def find_start(cls, text: str) -> int:
        """Index of the earliest start marker, or -1."""
        positions = [i for i in (text.find(m) for m in cls.START_MARKERS) if i != -1]
        return min(positions) if positions else -1

    @classmethod
    def detect(cls, text: str) -> bool:
        return cls.find_start(text) != -1

    @property
    def _scanner(self) -> MarkerScanner:
        """Reads the pre-region text; built on first use, per instance."""
        if self._scanner_cache is None:
            self._scanner_cache = MarkerScanner(self.START_MARKERS)
        return self._scanner_cache

    def process(self, text: str) -> list:
        results: list = []
        if self.state == 0:
            # Held back: only a suffix that could still grow into a start
            # marker. This used to hold from the *last* holdback character
            # anywhere in the buffer, which had no branch for that character
            # landing at index 0 -- and after one emission the buffer always
            # began with one, so the parser stopped emitting for the rest of
            # the stream and delivered the remainder in a single frame at EOS.
            # `HOLDBACK_CHARS` went with it: it was a hand-kept copy of the
            # first characters of `START_MARKERS`, which the scanner derives.
            scan = self._scanner.feed(text)
            if scan.released:
                results.append(("content", scan.released))
            if scan.hit is None:
                return results
            self.buf = scan.hit + scan.rest
            self.state = 1
            return results
        self.buf += text
        return results

    def flush(self) -> list:
        results: list = []
        if self.state == 0:
            rest = self._scanner.flush() + self.buf
            self.buf = ""
            if rest:
                results.append(("content", rest))
            return results
        # state 1: parse the complete (or trailing) tool-call block.
        region, self.buf = self.buf, ""
        _content, tool_calls = self.parse(region, self.tools)
        if not tool_calls:
            # A start marker is not a promise. An answer explaining that a
            # model "writes <tool_call> to call something" opens the region
            # and never closes it, and this used to drop everything from the
            # marker on -- no event, no error, `finish_reason` still `stop`.
            # Fifty characters of eighty-two, on the shapes measured.
            #
            # Released verbatim rather than as `parse`'s content, which strips.
            return [("content", region)] if region else []
        for tc in tool_calls:
            results.extend(self._emit_call(tc))
        if self.emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results
