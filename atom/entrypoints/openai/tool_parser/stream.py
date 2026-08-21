# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Streaming facade: read ahead of the chosen format, then delegate to it."""

from dataclasses import dataclass, field

from ..marker_scanner import MarkerScanner
from .tool_parser import ToolCallParser


@dataclass
class ToolCallStreamParser:
    """Stateful streaming parser for one request's output.

    Emits structured events:
    - ("content", text) — regular content before tool calls
    - ("tool_call_start", {"index": N, "id": ..., "function": {"name": ..., "arguments": ""}})
    - ("tool_call_args", {"index": N, "function": {"arguments": chunk}})
    - ("tool_call_end", None) — all tool calls complete

    ``parser_cls`` is the format this model was resolved to at startup, from
    its chat template (`registry.resolve_from_prompt`). ``None`` means no
    registered format recognised it: this then emits everything as content and
    parses nothing, which the server announced at startup. It is deliberately
    not a fallback to guessing — the guess this replaces mis-read a Hermes
    `<tool_call>{...}` as GLM and delivered the whole JSON blob as the tool's
    *name*.

    ``tools`` enables JSON-Schema type coercion of parameter values. It may be
    assigned after construction (several call sites do) and is re-read on every
    delegated call, so it takes effect as long as it is set before the stream
    ends.
    """

    tools: list | None = None
    parser_cls: type[ToolCallParser] | None = None
    _parser: ToolCallParser | None = field(default=None, repr=False)
    # Reads ahead of the region: releases everything that cannot begin one of
    # `parser_cls`'s own openers. `None` once the region has started, or when
    # there is no format to look for.
    _scanner: MarkerScanner | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.parser_cls is not None:
            self._scanner = MarkerScanner(self.parser_cls.START_MARKERS)

    @property
    def fmt(self) -> str | None:
        """The format this stream is being read as, or None."""
        return self.parser_cls.NAME if self.parser_cls is not None else None

    def process(self, text: str) -> list:
        """Process a text chunk and return list of (event_type, data) tuples."""
        if self.parser_cls is None:
            return [("content", text)] if text else []

        out: list = []
        while self._parser is None:
            scan = self._scanner.feed(text)
            if scan.released:
                out.append(("content", scan.released))
            if scan.hit is None:
                return out
            if not self.parser_cls.opens_region(scan.hit):
                # Framing this format wraps *every* answer in, which the
                # non-streaming path also removes. Dropping it and carrying on
                # is what lets such a format stream at all: treating it as a
                # handover meant a Kimi-K3 answer, which opens with
                # `<|open|>response<|sep|>`, buffered its entire body to EOS
                # -- measured, 324 of 324 characters arriving in one frame.
                text = scan.rest
                continue
            # The region has opened. It and everything after it belong to the
            # format, not to the client.
            self._scanner = None
            self._parser = self.parser_cls(tools=self.tools)
            text = scan.hit + scan.rest

        self._parser.tools = self.tools
        return out + self._parser.process(text)

    def flush(self) -> list:
        """Flush whatever is still held; at EOS none of it became a region."""
        if self._parser is None:
            rest = self._scanner.flush() if self._scanner is not None else ""
            return [("content", rest)] if rest else []
        self._parser.tools = self.tools
        return self._parser.flush()
