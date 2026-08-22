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

from .. import kimi_k3_tokens as k3
from .tool_parser import RegionParse, ToolCall, ToolCallParser, unique_tool_call_id

# K3 channel tokens this parser matches on. Kept local so the parser is
# self-contained; the reasoning splitter declares its own copies.
KIMI_K3_CALL_PREFIX = '<|open|>call tool="'
KIMI_K3_TOOLS_START = "<|open|>tools<|sep|>"
KIMI_K3_RESPONSE_START = "<|open|>response<|sep|>"
KIMI_K3_RESPONSE_END = "<|close|>response<|sep|>"
KIMI_K3_END_OF_MSG = "<|end_of_msg|>"

# A call's body may not contain another call opener, nor an argument's value
# another argument opener -- those literals are what open one. Without the
# guard the non-greedy body ran from an opener *quoted in prose* to the real
# call's closer, so an answer explaining `<|open|>call tool="NAME"<|sep|>`
# before making a real call produced one call named `NAME` carrying the real
# call's arguments. Every registered format carries this guard now.
_NOT_NESTED_CALL = r'(?:(?!<\|open\|>call tool=").)'
_NOT_NESTED_ARG = r'(?:(?!<\|open\|>argument key=").)'
_K3_CALL_RE = re.compile(
    r'<\|open\|>call tool="(?P<name>[^"]*)"(?:\s+index="(?P<index>\d+)")?<\|sep\|>'
    r"(?P<body>" + _NOT_NESTED_CALL + r"*?)<\|close\|>call",
    re.DOTALL,
)
_K3_ARG_RE = re.compile(
    r'<\|open\|>argument key="(?P<key>[^"]*)"(?:\s+type="(?P<type>[^"]*)")?<\|sep\|>'
    r"(?P<val>" + _NOT_NESTED_ARG + r"*?)<\|close\|>argument",
    re.DOTALL,
)
# The channel framing this format wraps every answer in, tool call or not.
# Declared once and consumed once: `START_MARKERS` lists these so the
# read-ahead cannot split one, `opens_region` says no to them so the reader
# drops them, and that is the only place they are removed. There used to be a
# second removal inside `parse` -- a regex built from a hand-kept copy of this
# list -- and the two disagreed about four tokens, which reached the client
# verbatim when streamed and were deleted when not.
#
# `call` and `argument` are deliberately absent: they carry data
# (`tool="..."`, `key="..."`) so they cannot be declared as literals, and they
# only ever occur inside a tools section, where `_K3_CALL_RE` and `_K3_ARG_RE`
# account for them.

# This format's own framing: what wraps every answer, plus what brackets a
# call. Both halves come from `kimi_k3_tokens`, which the reasoning dialect
# reads too -- do not re-spell these literals here, that is how the two
# copies came to disagree.
_K3_CONTENT_FRAMING = (
    *k3.CHANNEL_FRAMING,
    k3.THINK_START,
    k3.THINK_END,
    *k3.TOOL_REGION_FRAMING,
    *k3.UNPAIRED_FRAMING,
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
    # Every token `parse` strips from content, plus the call prefix that opens
    # a region. Derived from `_K3_CONTENT_FRAMING` rather than written out
    # again: the read-ahead must not split a token the stripper removes, and a
    # hand-kept second copy of the list is how four of them came to be missing
    # -- they reached the client verbatim when streamed and were deleted when
    # not. `is_kimi_k3` keys on five of these; this is not that list.
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        KIMI_K3_CALL_PREFIX,
        *_K3_CONTENT_FRAMING,
    )
    # The two that mean a tool call is coming. Every other marker above is
    # channel framing that wraps every answer this model gives, so they are
    # literals the read-ahead must not split and nothing more.
    _REGION_MARKERS: ClassVar[frozenset[str]] = frozenset(
        {KIMI_K3_CALL_PREFIX, KIMI_K3_TOOLS_START}
    )
    # The tools channel closing after the last call. Framing would drop it
    # anyway once it is handed back, but naming it here keeps the newline
    # between the two tokens out of the answer.
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ("<|open|>tools<|sep|>",)
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ("<|close|>tools",)

    @classmethod
    def opens_region(cls, marker: str) -> bool:
        return marker in cls._REGION_MARKERS

    @classmethod
    def detect(cls, text: str) -> bool:
        return is_kimi_k3(text)

    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        """Every complete call in the section, and where the section ends.

        A call cut off by `max_tokens` has no `<|close|>call` and so parses to
        nothing, which means the region was not a call after all and its bytes
        are released unchanged -- the same answer this format now gives to an
        answer that merely quotes the opener. Both used to be decided here, by
        a second opener regex that accepted shapes `_K3_CALL_RE` rejects, and
        the two ways of getting it wrong were opposite: a quotation lost 62
        characters, and a truncated call kept its half-written payload with the
        dangling `<|close|>argument` still in it.
        """
        tool_calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for m in _K3_CALL_RE.finditer(region):
            args: dict = {}
            for a in _K3_ARG_RE.finditer(m.group("body")):
                args[a.group("key")] = _k3_coerce(a.group("val"), a.group("type"))
            spans.append(
                (cls.markup_begin(region, m.start()), cls.markup_end(region, m.end()))
            )
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
        return RegionParse(tuple(tool_calls), tuple(spans))
