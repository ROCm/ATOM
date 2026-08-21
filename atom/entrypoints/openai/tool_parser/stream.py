# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Streaming facade: sniff the format once, then delegate every chunk to it."""

from dataclasses import dataclass, field

from ..marker_scanner import MarkerScanner
from .registry import WAIT, all_markers, sniff_stream
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
    # Pre-detection accumulator, reached only once a marker has actually been
    # seen. Bounded by the length of one tool call rather than by the length
    # of the response, because nothing lands here until a format's opening
    # literal has arrived in full.
    _buf: str = ""
    _parser: ToolCallParser | None = field(default=None, repr=False)
    # Live while no marker has been seen at all: reads ahead of any format,
    # releasing everything that cannot begin one. `None` once a marker has
    # arrived, or once a format has been chosen.
    _scanner: MarkerScanner | None = field(
        default_factory=lambda: MarkerScanner(all_markers()), repr=False
    )

    @property
    def fmt(self) -> str | None:
        """Detected format name, or None while still undecided."""
        return self._parser.NAME if self._parser is not None else None

    def process(self, text: str) -> list:
        """Process a text chunk and return list of (event_type, data) tuples.

        Two phases before a format is known, and the split is what keeps an
        ordinary answer streaming. Until an opening literal actually arrives,
        the scanner releases every byte that could not be part of one and
        holds only a possible partial marker -- bounded by the longest marker,
        so the wait is bounded too. From the moment one arrives in full, text
        is accumulated instead: it may belong to a tool call, and sending it
        to the client is a decision that cannot be taken back.
        """
        out: list = []
        if self._parser is None:
            if self._scanner is not None:
                scan = self._scanner.feed(text)
                if scan.released:
                    out.append(("content", scan.released))
                if scan.hit is None:
                    return out
                # A marker landed. It, and everything after it, belong to the
                # format cascade rather than to the client.
                self._scanner = None
                self._buf, text = scan.hit + scan.rest, ""
            else:
                self._buf += text

            choice = sniff_stream(self._buf)
            if choice is WAIT:
                return out
            self._parser = choice(tools=self.tools)
            # Replay everything accumulated since the marker.
            text, self._buf = self._buf, ""

        self._parser.tools = self.tools
        return out + self._parser.process(text)

    def flush(self) -> list:
        """Flush remaining buffer content."""
        if self._parser is None:
            # Undecided at EOS: whatever the scanner still held never became a
            # marker, and whatever the cascade accumulated never became a
            # format. Both are plain content.
            held = self._scanner.flush() if self._scanner is not None else ""
            rest = held + self._buf
            self._buf = ""
            return [("content", rest)] if rest else []

        self._parser.tools = self.tools
        return self._parser.flush()
