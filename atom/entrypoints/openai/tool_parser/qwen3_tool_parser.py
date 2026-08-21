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
    enough. And what follows the name is a parameter or nothing: a cut-off
    call stops inside its own syntax, while prose continues in English.
    """
    gt = fn_text.find(">")
    if gt == -1:
        return False
    if fn_text[:gt].strip() not in param_types:
        return False
    rest = fn_text[gt + 1 :].lstrip()
    return not rest or _PARAM_OPENER.startswith(rest[: len(_PARAM_OPENER)])


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


# The name *and* the token that must follow it. A name alone matches prose
# explaining how to call a tool, and an announcement cannot be retracted --
# "the model writes <tool_call><function=get_weather> and then..." announced
# `get_weather`. Waiting for the structure costs a few characters against
# the thousands this saves.
_PEEK_NAME_RE = re.compile(
    r"<function=([^>\n]+)>\s*(?:<parameter=|</function>|</tool_call>)"
)


class QwenXmlParser(BufferedMarkerParser):
    NAME: ClassVar[str] = "qwen"
    START_MARKERS: ClassVar[tuple[str, ...]] = ("<tool_call>", QWEN_TOOL_PREFIX)

    @classmethod
    def peek_name(cls, region: str) -> str | None:
        """`<function=NAME>` -- legible once the tag closes."""
        m = _PEEK_NAME_RE.search(region)
        return m.group(1) if m else None

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
