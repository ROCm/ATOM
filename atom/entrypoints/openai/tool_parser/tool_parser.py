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

    def __init__(self, tools: list | None = None):
        self.tools = tools
        self.buf = ""
        # 0 = still in plain content, 1 = inside a tool-call region. Kimi adds
        # 2 = section closed; see KimiParser.
        self.state = 0
        self.current_index = 0
        self.emitted_calls = 0

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

    # Any of these opening the tool-call region; the earliest one wins.
    START_MARKERS: ClassVar[tuple[str, ...]] = ()
    # While no marker has been seen, a trailing run starting with one of these
    # may be the first bytes of a marker, so it is held back rather than emitted
    # as content (it would otherwise leak '<' into the user-visible text).
    HOLDBACK_CHARS: ClassVar[tuple[str, ...]] = ("<",)

    @classmethod
    def find_start(cls, text: str) -> int:
        """Index of the earliest start marker, or -1."""
        positions = [i for i in (text.find(m) for m in cls.START_MARKERS) if i != -1]
        return min(positions) if positions else -1

    @classmethod
    def detect(cls, text: str) -> bool:
        return cls.find_start(text) != -1

    def process(self, text: str) -> list:
        results: list = []
        self.buf += text
        if self.state == 0:
            m = self.find_start(self.buf)
            if m != -1:
                before = self.buf[:m]
                if before:
                    results.append(("content", before))
                self.buf = self.buf[m:]
                self.state = 1
            else:
                # Emit content but hold back a possible partial marker tail.
                cut = max(self.buf.rfind(c) for c in self.HOLDBACK_CHARS)
                if cut == -1:
                    if self.buf:
                        results.append(("content", self.buf))
                        self.buf = ""
                elif cut > 0:
                    results.append(("content", self.buf[:cut]))
                    self.buf = self.buf[cut:]
        return results

    def flush(self) -> list:
        results: list = []
        if self.state == 0:
            if self.buf:
                results.append(("content", self.buf))
                self.buf = ""
            return results
        # state 1: parse the complete (or trailing) tool-call block.
        _content, tool_calls = self.parse(self.buf, self.tools)
        self.buf = ""
        for tc in tool_calls:
            results.extend(self._emit_call(tc))
        if self.emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results
