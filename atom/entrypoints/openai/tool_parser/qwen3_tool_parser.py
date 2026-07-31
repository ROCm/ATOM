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
_PARAM_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


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


class QwenXmlParser(BufferedMarkerParser):
    NAME: ClassVar[str] = "qwen"
    START_MARKERS: ClassVar[tuple[str, ...]] = ("<tool_call>", QWEN_TOOL_PREFIX)

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
            fn_text = fm.group(1) if fm.group(1) is not None else fm.group(2)
            if not fn_text:
                continue
            tc = _parse_function(fn_text, param_types)
            if tc is not None:
                tool_calls.append(tc)
        return content.strip(), tool_calls
