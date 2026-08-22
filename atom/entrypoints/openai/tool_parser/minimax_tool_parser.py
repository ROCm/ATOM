# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 tool-call format.

Every tag is prefixed by the ns_token ``]<]minimax[>[``::

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="NAME">
    ]<]minimax[>[<pname>value]<]minimax[>[</pname>
    ...
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

Unlike DSML, parameters are named by the TAG itself (``<city>Paris</city>``),
not a ``name="..."`` attribute. Strip the ns_token first, then parse
<invoke>/<tag> pairs. Values: schema type wins, else JSON, else raw string.
"""

import json
import re
from typing import Any, ClassVar

from .schema import build_param_types, coerce_json_or_raw
from .tool_parser import RegionParse, ToolCall, ToolCallParser, unique_tool_call_id

MINIMAX_NS = "]<]minimax[>["

# The ns_token is matched optionally wherever a tag can appear, as DSML does
# with its own marker, rather than deleted from a copy of the text first.
# Deleting it made every offset in the parsed copy meaningless against the
# bytes that actually arrived, which is why `parse` could only ever report
# "content precedes the call" and never "and this follows it".
_NS = r"(?:" + re.escape(MINIMAX_NS) + r")?"

# A call's body may not contain another opener -- that literal is what opens
# one. Without the guard the non-greedy body ran from a *quoted* opener in
# prose all the way to the real call's closer, so an answer explaining
# "you write <invoke name="NAME">" before making a real call produced one call named after the
# placeholder, carrying the real call's arguments, with the sentence deleted.
# `finditer` then resumed past the real call, so the call the model actually
# made never went out. GLM was given this guard first; this is the sweep.
_NOT_NESTED = r"(?:(?!" + _NS + r"<invoke\s).)"
_INVOKE_RE = re.compile(
    _NS
    + r'<invoke\s+name="([^"]*)"\s*>('
    + _NOT_NESTED
    + r"*?)"
    + _NS
    + r"</invoke>|"
    + _NS
    + r'<invoke\s+name="([^"]*)"\s*>('
    + _NOT_NESTED
    + r"*)",
    re.DOTALL,
)
_PARAM_RE = re.compile(_NS + r"<([\w-]+)>(.*?)" + _NS + r"</\1>", re.DOTALL)
_FIRST_TAG_RE = re.compile(r"^" + _NS + r"<([\w-]+)>")


def _is_truncated_call(
    name: str, body: str, param_types: dict, *, at_end: bool
) -> bool:
    """Is this unclosed `<invoke name=...>` a cut-off call, or prose?

    See :func:`QwenXmlParser._is_truncated_call`. The sweep that added that
    test to Qwen and GLM missed this format and DSML both, and "you emit
    `<invoke name="get_weather">` and then a `<city>Paris</city>` line" was
    still dispatched as a real call with the arguments filled in.

    The follower is a tag, because this format names parameters by the tag
    itself and so has no keyword to look for. Which tag is checked too when
    the request declared a schema for this tool -- otherwise `<br>` reads as
    the start of a call body -- but a tool declared without one still has to
    be callable, so an empty schema falls back to "any tag".
    """
    types = param_types.get(name)
    if types is None:
        return False
    rest = body.lstrip()
    if not rest:
        return at_end
    tag = _FIRST_TAG_RE.match(rest)
    return bool(tag) and (not types or tag.group(1) in types)


class MiniMaxParser(ToolCallParser):
    NAME: ClassVar[str] = "minimax"
    RECOGNISES_A_CALL_IN_PROGRESS: ClassVar[bool] = True
    # `<invoke name="` too: `_INVOKE_RE` matches it anywhere, so an invoke the
    # model wrote without the ns_token was a call when parsed whole and plain
    # text when streamed -- the read-ahead never opened a region for it. DSML
    # lists the same marker-less malform for the same reason.
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        MINIMAX_NS,
        "<tool_call>",
        '<invoke name="',
    )
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ("</tool_call>",)
    CALL_FILLERS: ClassVar[tuple[str, ...]] = (MINIMAX_NS,)

    # The ns_token opens a region like the other two, rather than being
    # framing the reader drops. It prefixes every tag including ones inside a
    # call, so calling it framing would delete it from an answer that merely
    # *mentions* it while an answer that mentions the tag it prefixes keeps
    # both -- the same token surviving or not depending on what followed it.
    # `parse` had exactly that split (every occurrence deleted from a copy of
    # the text when a call was found, all of them left when one was not); a
    # region that parses to nothing is released whole, so this way both
    # answers keep their bytes.

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the MiniMax-M3 ns_token tool-call format."""
        return MINIMAX_NS in text

    @classmethod
    def parse_region(
        cls, region: str, tools: list | None, *, at_end: bool
    ) -> RegionParse:
        param_types = build_param_types(tools)
        tool_calls: list[ToolCall] = []
        begin = end = 0
        for m in _INVOKE_RE.finditer(region):
            closed = m.group(1) is not None
            name = m.group(1) if closed else m.group(3)
            body = m.group(2) if closed else (m.group(4) or "")
            if not name:
                continue
            name = name.strip()
            if not closed and not _is_truncated_call(
                name, body, param_types, at_end=at_end
            ):
                continue
            types = param_types.get(name, {})
            args: dict[str, Any] = {}
            for pm in _PARAM_RE.finditer(body):
                k = pm.group(1).strip()
                if k:
                    args[k] = coerce_json_or_raw(pm.group(2), types.get(k))
            if not tool_calls:
                begin = cls.markup_begin(region, m.start())
            tool_calls.append(
                ToolCall(
                    id=unique_tool_call_id(),
                    type="function",
                    function={
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                )
            )
            end = cls.markup_end(region, m.end())
        return RegionParse(tuple(tool_calls), begin, end)
