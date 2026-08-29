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
from .schema import build_param_types
from .tool_parser import (
    RegionParse,
    ToolCall,
    ToolCallParser,
    continues_a_call,
    declared_tools_allow,
    unique_tool_call_id,
    usable_tool_name,
)

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
# What may follow a call's opener in a real call: an argument, or the close of
# the call the name opened. One tuple, read by `_is_truncated_call`.
_CALL_CONTINUES = ('<|open|>argument key="', "<|close|>call")
# Closed, cut off where the next call or section opens, or cut off at end of
# input. Without end-of-input a call truncated by `max_tokens` parsed to
# nothing and its raw tokens went out as the answer. The opener lookaheads are
# the fifth terminator `parse_region` names, and this format needed them most:
# the tempered dot already stops the body at a sibling, so with no terminator
# accepting that position the whole entry matched nothing and was dropped --
# after the announcement had named it.
_K3_CALL_RE = re.compile(
    r'<\|open\|>call tool="(?P<name>[^"]*)"(?:\s+index="(?P<index>\d+)")?<\|sep\|>'
    r"(?P<body>" + _NOT_NESTED_CALL + r"*?)"
    # The separator belongs to the closer, as it does in the model's encoder
    # and in vLLM's and SGLang's readers. Optional: a truncation can stop
    # between the two.
    r"(?:(?P<closed><\|close\|>call(?:<\|sep\|>)?)"
    r'|(?=<\|open\|>call tool=")'
    r"|(?=<\|open\|>tools<\|sep\|>)"
    r"|(?=<\|close\|>tools)"
    r"|\Z)",
    re.DOTALL,
)
_K3_ARG_RE = re.compile(
    r'<\|open\|>argument key="(?P<key>[^"]*)"(?:\s+type="(?P<type>[^"]*)")?<\|sep\|>'
    # Same alternation one level down: a value the model was cut off inside
    # survives, and an argument missing its closer before a sibling is not
    # deleted -- that dropped a parameter and dispatched the call without it.
    r"(?P<val>" + _NOT_NESTED_ARG + r"*?)"
    r"(?:<\|close\|>argument"
    r'|(?=<\|open\|>argument key=")'
    r"|(?=<\|open\|>tools<\|sep\|>)"
    r"|(?=<\|close\|>call)"
    r"|\Z)",
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


def _is_truncated_call(
    name: str, body: str, param_types: dict, *, at_end: bool
) -> bool:
    """Is this unclosed `<|open|>call tool=...` a cut-off call, or prose?

    The gate travels with the unclosed alternative or a sentence that merely
    *quotes* the opener parses as a call.
    """
    if not declared_tools_allow(name, param_types):
        return False
    rest = body.lstrip()
    return (not rest and at_end) or continues_a_call(
        rest, _CALL_CONTINUES, arrived=not at_end
    )


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
    CALL_OPENERS: ClassVar[tuple[str, ...]] = (k3.TOOLS_START,)
    # With the separator, because that is where the model puts it. Spelled
    # without it, `markup_end` stopped on the separator and
    # `<|sep|><|close|>tools<|sep|>` reached the client as the answer between
    # two adjacent calls. Not `CALL_FILLERS = (k3.BARE_SEPARATOR,)`, which
    # would fix the same leak: that tuple feeds the *call*-level walkers too
    # and moves every K3 call's span.
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = k3.both_spellings(k3.TOOLS_END)
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = k3.both_spellings(k3.CALL_END)

    @classmethod
    def render_call(cls, name: str, args: dict[str, str]) -> str:
        """Byte for byte what `/data/Kimi-K3/encoding_k3.py` emits.

        It used to emit a reduction: no `index`, no `type`, no separator after
        any closer, no `<|close|>tools` at all. This is the sole source of the
        K3 corpus, so every K3 property was asked about a shape the model
        never writes -- `_k3_coerce`'s `type=` branch was unreachable from it.

        `type="string"` because the signature takes `dict[str, str]`; the
        model chooses per value in `_xtml_type`.
        """
        sep = k3.BARE_SEPARATOR
        body = "".join(
            f'<|open|>argument key="{k}" type="string"{sep}{v}{k3.ARGUMENT_END}{sep}'
            for k, v in args.items()
        )
        return (
            f'{k3.TOOLS_START}<|open|>call tool="{name}" index="0"{sep}'
            f"{body}{k3.CALL_END}{sep}{k3.TOOLS_END}{sep}"
        )

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
        param_types = build_param_types(tools)
        tool_calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for m in _K3_CALL_RE.finditer(region):
            # Before the truncation gate, which only runs for an unclosed
            # call: `tool="(?P<name>[^"]*)"` matches an empty and an all-space
            # name, and a *closed* one had nothing to stop it.
            if not usable_tool_name(m.group("name")):
                continue
            if m.group("closed") is None and not _is_truncated_call(
                m.group("name"), m.group("body"), param_types, at_end=at_end
            ):
                continue
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
