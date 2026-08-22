# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3 (qwen3_coder / qwen3_xml) XML tool-call format::

    <tool_call>
    <function=NAME>
    <parameter=PNAME>VALUE</parameter>
    ...
    </function>
    </tool_call>

The XML carries no value types, so parameters are coerced against the request's
``tools`` schema when supplied. Mirrors the qwen3_coder/qwen3_xml parsers in
vLLM and SGLang.
"""

import json
import re
from typing import Any, ClassVar

from .kimi_tool_parser import KIMI_SECTION_BEGIN
from .schema import build_param_types, coerce_param_value
from .tool_parser import BufferedMarkerParser, ToolCall, unique_tool_call_id

# Also read by GlmParser.detect: '<function=' is what tells Qwen's <tool_call>
# apart from GLM's identically-named tag.
QWEN_TOOL_PREFIX = "<function="

_FUNCTION_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_PARAM_OPENER = "<parameter="
_PARAM_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


# What may follow the name inside a call: another parameter, or the close of
# the very block the name opened. NOT `</tool_call>`, which closes the
# *outer* wrapper -- `<function=get_weather></tool_call>` leaves the function
# block unterminated, so `parse` reads it as prose. The peek used to accept
# it and `parse` did not, which is the whole of the mismatch this shared
# tuple exists to prevent: one spelling, both readers.
_CALL_CONTINUES = (_PARAM_OPENER, "</function>")


def _continues_a_call(rest: str) -> bool:
    """Is `rest` the start of this format's own next token?"""
    return any(tok.startswith(rest[: len(tok)]) for tok in _CALL_CONTINUES)


def _name_and_rest(fn_text: str) -> tuple[str, str] | None:
    """Split `NAME>whatever` at the tag close, or ``None`` if it has not come."""
    gt = fn_text.find(">")
    if gt == -1:
        return None
    return fn_text[:gt].strip(), fn_text[gt + 1 :].lstrip()


def _is_truncated_call(fn_text: str, param_types: dict) -> bool:
    """Is this unclosed `<function=...` a cut-off call, or prose quoting a tag?

    The unclosed branch exists for a call the model was cut off mid-way
    through. It cannot tell that from an answer explaining how to call a tool,
    and used to accept both: "the model writes <tool_call><function=get_weather>
    and then the parameters" produced `get_weather({})`, deleted the rest of
    the sentence and reported `finish_reason: tool_calls`, so an agentic
    client ran a tool nobody asked for.

    Two things separate them, and prose has to fail both. The name is one the
    request declared -- prose can name a real tool, so this alone is not
    enough. And what follows the name is this format's own next token: a
    cut-off call stops inside its own syntax, while prose continues in
    English. `peek_name` applies the same two, from the same code.
    """
    split = _name_and_rest(fn_text)
    if split is None:
        return False
    name, rest = split
    return name in param_types and (not rest or _continues_a_call(rest))


def _parse_function(
    fn_text: str, param_types: dict[str, dict[str, Any]]
) -> ToolCall | None:
    """Parse the inside of one ``<function=NAME>...`` block into a ToolCall."""
    gt = fn_text.find(">")
    if gt == -1:
        return None
    name = fn_text[:gt].strip()
    if not name:
        return None
    body = fn_text[gt + 1 :]
    types = param_types.get(name, {})
    args: dict[str, Any] = {}
    for pm in _PARAM_RE.finditer(body):
        seg = pm.group(1)
        if seg is None:
            continue
        pgt = seg.find(">")
        if pgt == -1:
            continue
        pname = seg[:pgt].strip()
        pval = seg[pgt + 1 :]
        if pname:
            args[pname] = coerce_param_value(pval, types.get(pname))
    return ToolCall(
        id=unique_tool_call_id(),
        type="function",
        function={"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    )


# The name, and enough of what follows to tell a call from prose. A name
# alone matches an answer explaining how to call a tool, and an announcement
# cannot be retracted -- "the model writes <tool_call><function=get_weather>
# and then..." announced `get_weather`. Waiting for the structure costs a few
# characters against the thousands this saves.
_PEEK_NAME_RE = re.compile(r"<function=([^>\n]+)>(.*)", re.DOTALL)


class QwenXmlParser(BufferedMarkerParser):
    NAME: ClassVar[str] = "qwen"
    START_MARKERS: ClassVar[tuple[str, ...]] = ("<tool_call>", QWEN_TOOL_PREFIX)

    @classmethod
    def peek_name(cls, region: str, tools: list | None = None) -> str | None:
        """`<function=NAME>` -- legible once the tag closes and something
        follows it that only a call would write.

        The follower must have *arrived*, unlike in `_is_truncated_call`,
        which accepts nothing-after because it runs at end of stream where
        nothing more is coming. Here more is coming, so an empty tail is
        "not yet" rather than "cut off".
        """
        m = _PEEK_NAME_RE.search(region)
        if m is None:
            return None
        rest = m.group(2).lstrip()
        return m.group(1) if rest and _continues_a_call(rest) else None

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the Qwen3 XML format (and not the Kimi token format)."""
        return QWEN_TOOL_PREFIX in text and KIMI_SECTION_BEGIN not in text

    @classmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        """Parse Qwen3 XML tool calls; return (leading_content, tool_calls)."""
        param_types = build_param_types(tools)
        # Content precedes the first tool marker.
        start = cls.find_start(text)
        content = text[:start] if start != -1 else text
        tool_calls: list[ToolCall] = []
        for fm in _FUNCTION_RE.finditer(text):
            closed = fm.group(1) is not None
            fn_text = fm.group(1) if closed else fm.group(2)
            if not fn_text:
                continue
            tc = _parse_function(fn_text, param_types)
            if tc is None:
                continue
            if not closed and not _is_truncated_call(fn_text, param_types):
                continue
            tool_calls.append(tc)
        # No call -> verbatim; see ToolCallParser.parse.
        return (content.strip() if tool_calls else text), tool_calls
