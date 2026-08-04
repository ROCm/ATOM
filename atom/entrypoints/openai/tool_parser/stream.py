# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Streaming facade: sniff the format once, then delegate every chunk to it."""

from dataclasses import dataclass, field

from .registry import EMIT_CONTENT, WAIT, sniff_stream
from .tool_parser import ToolCallParser


@dataclass
class ToolCallStreamParser:
    """Stateful streaming parser; format is auto-detected from the first chunks.

    Emits structured events:
    - ("content", text) — regular content before tool calls
    - ("tool_call_start", {"index": N, "id": ..., "function": {"name": ..., "arguments": ""}})
    - ("tool_call_args", {"index": N, "function": {"arguments": chunk}})
    - ("tool_call_end", None) — all tool calls complete

    ``tools`` enables JSON-Schema type coercion of parameter values. It may be
    assigned after construction (several call sites do) and is re-read on every
    delegated call, so it takes effect as long as it is set before the stream
    ends.
    """

    tools: list | None = None
    # Pre-detection accumulator. Once a format is chosen this is handed to the
    # concrete parser and never used again.
    _buf: str = ""
    _parser: ToolCallParser | None = field(default=None, repr=False)

    @property
    def fmt(self) -> str | None:
        """Detected format name, or None while still undecided."""
        return self._parser.NAME if self._parser is not None else None

    def process(self, text: str) -> list:
        """Process a text chunk and return list of (event_type, data) tuples."""
        if self._parser is None:
            self._buf += text
            choice = sniff_stream(self._buf)
            if choice is WAIT:
                return []
            if choice is EMIT_CONTENT:
                out = [("content", self._buf)]
                self._buf = ""
                return out
            self._parser = choice(tools=self.tools)
            # Replay everything accumulated while undecided.
            text, self._buf = self._buf, ""

        self._parser.tools = self.tools
        return self._parser.process(text)

    def flush(self) -> list:
        """Flush remaining buffer content."""
        if self._parser is None:
            # Undecided at EOS: no tool markers ever appeared -> plain content.
            if self._buf:
                out = [("content", self._buf)]
                self._buf = ""
                return out
            return []

        self._parser.tools = self.tools
        return self._parser.flush()
