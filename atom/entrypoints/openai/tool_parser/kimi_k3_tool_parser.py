# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 channel-format tool-call format::

    <|open|>call tool="NAME" index="i"<|sep|>
      <|open|>argument key="K" type="T"<|sep|>VALUE<|close|>argument<|sep|> ...
    <|close|>call ...

Argument VALUEs are raw-by-type (string unquoted, number/bool/object as literals)
and are coerced then assembled into one JSON object per call. Unlike the XML-ish
formats the type travels on the wire (``type="..."``) so no request ``tools``
schema is needed; ``tools`` is unused.

K3 interleaves think/response/tools sections in one channel-framed stream, so
partial-chunk parsing is unreliable: this parser buffers the whole output and
parses once at flush. Plain (non-tool) answers also carry channel framing
tokens, which are stripped from ``content``.
"""

import json
import re
from typing import ClassVar

from .tool_parser import ToolCall, ToolCallParser, unique_tool_call_id

# K3 channel tokens this parser matches on. Kept local so the parser is
# self-contained; the reasoning splitter declares its own copies.
KIMI_K3_CALL_PREFIX = '<|open|>call tool="'
KIMI_K3_TOOLS_START = "<|open|>tools<|sep|>"
KIMI_K3_RESPONSE_START = "<|open|>response<|sep|>"
KIMI_K3_RESPONSE_END = "<|close|>response<|sep|>"
KIMI_K3_END_OF_MSG = "<|end_of_msg|>"

_K3_CALL_RE = re.compile(
    r'<\|open\|>call tool="(?P<name>[^"]*)"(?:\s+index="(?P<index>\d+)")?<\|sep\|>'
    r"(?P<body>.*?)<\|close\|>call",
    re.DOTALL,
)
_K3_ARG_RE = re.compile(
    r'<\|open\|>argument key="(?P<key>[^"]*)"(?:\s+type="(?P<type>[^"]*)")?<\|sep\|>'
    r"(?P<val>.*?)<\|close\|>argument",
    re.DOTALL,
)
_K3_FRAMING_RE = re.compile(
    r"<\|(?:open|close)\|>(?:response|message|tools|think|call|argument)[^<]*?<\|sep\|>"
    r"|<\|close\|>(?:response|message|tools|think)"
    r"|<\|end_of_msg\|>|<\|sep\|>"
)


def is_kimi_k3(text: str) -> bool:
    return (
        KIMI_K3_CALL_PREFIX in text
        or KIMI_K3_TOOLS_START in text
        or KIMI_K3_RESPONSE_START in text
        or KIMI_K3_RESPONSE_END in text
        or KIMI_K3_END_OF_MSG in text
    )


def _k3_coerce(val: str, ptype: str | None):
    t = (ptype or "").lower()
    # Strings are returned verbatim: leading/trailing whitespace can be
    # semantically significant (e.g. an enum of whitespace values), so stripping
    # would corrupt valid values. Non-string types strip first, since surrounding
    # whitespace there is only formatting noise around the literal to coerce.
    if t.startswith("str"):
        return val
    v = val.strip()
    try:
        if t.startswith("int"):
            return int(v)
        if t.startswith(("num", "float", "double", "decimal")):
            f = float(v)
            return int(f) if f.is_integer() else f
        if t.startswith("bool"):
            return v.lower() == "true"
        return json.loads(v)  # object / array / unknown
    except Exception:  # noqa: BLE001 - best-effort coercion, return raw
        return v


def _strip_k3_framing(text: str) -> str:
    """Remove the channel tokens, and nothing else.

    No `.strip()`. Removing framing is this format's own normalization and
    both paths do it; trimming whitespace is position-dependent and so cannot
    agree between them -- streaming applies it to the region after a marker
    and non-streaming to the whole answer, which turned `writes <tok> to` into
    `writes to` on one path and `writes  to` on the other.
    """
    return _K3_FRAMING_RE.sub("", text)


_PEEK_NAME_RE = re.compile(r'<\|open\|>call tool="([^"]+)"')


class KimiK3Parser(ToolCallParser):
    """Kimi-K3 channel format: buffer the tools section, parse + emit at flush.

    A tools section is parsed whole because K3's arguments interleave and a
    partial one emits garbage. The *response* channel is not a tools section
    and streams as it arrives: it is plain text wrapped in framing this format
    also removes when it is not streaming.

    Buffering everything was simpler and was justified by "the outputs are
    short". Every K3 answer opens with `<|open|>response<|sep|>`, which is one
    of the markers below, so that read as a tool region and the whole body
    arrived in one frame at EOS -- measured on a 324-character answer, 324 of
    them. It is the common path for this model, not an edge case.
    """

    NAME: ClassVar[str] = "kimi_k3"
    # The same five `is_kimi_k3` decides by, named rather than spelled out:
    # a hand-written copy of this list had four of them.
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        KIMI_K3_CALL_PREFIX,
        KIMI_K3_TOOLS_START,
        KIMI_K3_RESPONSE_START,
        KIMI_K3_RESPONSE_END,
        KIMI_K3_END_OF_MSG,
    )
    # Of those five, the two that mean a tool call is coming. The other three
    # wrap every answer this model gives, so they are literals the read-ahead
    # must not split and nothing more.
    _REGION_MARKERS: ClassVar[frozenset[str]] = frozenset(
        {KIMI_K3_CALL_PREFIX, KIMI_K3_TOOLS_START}
    )

    @classmethod
    def peek_name(cls, region: str) -> str | None:
        """`<|open|>call tool="NAME"` -- the name travels in the opener."""
        m = _PEEK_NAME_RE.search(region)
        return m.group(1) if m else None

    @classmethod
    def opens_region(cls, marker: str) -> bool:
        return marker in cls._REGION_MARKERS

    @classmethod
    def detect(cls, text: str) -> bool:
        return is_kimi_k3(text)

    @classmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        """Parse the Kimi-K3 channel format; return (clean_content, tool_calls)."""
        tool_calls: list[ToolCall] = []
        for m in _K3_CALL_RE.finditer(text):
            args: dict = {}
            for a in _K3_ARG_RE.finditer(m.group("body")):
                args[a.group("key")] = _k3_coerce(a.group("val"), a.group("type"))
            tool_calls.append(
                ToolCall(
                    id=unique_tool_call_id(),
                    type="function",
                    function={
                        "name": m.group("name"),
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                )
            )
        # Truncated at the tools marker only when a section really opened
        # there. An answer that merely names the token opened none, and
        # cutting at it dropped everything after.
        #
        # "A call parsed" is the wrong test for that, and was: a call cut off
        # by `max_tokens` parses to nothing, so the half-written payload was
        # kept and shipped as content -- `_K3_FRAMING_RE` has no alternative
        # for the dangling `<|close|>argument`, so it survived too. What
        # separates the two cases is a call *prefix* after the marker, which
        # a truncated call still has and a mention of the token does not.
        ts = text.find(KIMI_K3_TOOLS_START)
        opened = ts != -1 and KIMI_K3_CALL_PREFIX in text[ts:]
        return _strip_k3_framing(text[:ts] if opened else text), tool_calls

    def process(self, text: str) -> list:
        # Buffer everything; K3's interleaved framing is parsed once at flush.
        # The name is the exception: it travels in the call opener, so it can
        # go out now rather than after the arguments -- see `announce`.
        self.buf += text
        return self.announce(self.buf)

    def flush(self) -> list:
        content, tool_calls = self.parse(self.buf, self.tools)
        self.buf = ""
        results: list = []
        if content:
            results.append(("content", content))
        for tc in tool_calls:
            results.extend(self._emit_call(tc))
        if self.emitted_calls > 0:
            results.append(("tool_call_end", None))
        return results
