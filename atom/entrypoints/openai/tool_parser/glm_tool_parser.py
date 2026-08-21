# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-4.5 / 4.6 / 5.x tool-call format::

    <tool_call>NAME
    <arg_key>K1</arg_key><arg_value>V1</arg_value>
    <arg_key>K2</arg_key><arg_value>V2</arg_value>
    ...</tool_call>

The function name follows the opening tag directly (no ``<function=`` wrapper,
which is how this is told apart from the Qwen3 XML format). GLM's chat template
renders non-string argument values with ``tojson`` and string values raw, so a
value is JSON-decoded when the request schema declares a non-string type (or
when it parses as JSON) and otherwise kept as a raw string.
"""

import json
import re
from typing import Any, ClassVar

from .qwen3_tool_parser import QWEN_TOOL_PREFIX
from .schema import build_param_types, coerce_json_or_raw
from .tool_parser import BufferedMarkerParser, ToolCall, unique_tool_call_id

_TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)$", re.DOTALL)
_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>"
    r"(.*?)(?:</arg_value>|(?=<arg_key>)|(?=</tool_call>)|$)",
    re.DOTALL,
)


class GlmParser(BufferedMarkerParser):
    NAME: ClassVar[str] = "glm"
    START_MARKERS: ClassVar[tuple[str, ...]] = ("<tool_call>",)

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the GLM ``<tool_call>...<arg_key>`` format (never Qwen/DSML)."""
        if QWEN_TOOL_PREFIX in text:  # '<function=' -> Qwen, not GLM
            return False
        return "<arg_key>" in text or "<tool_call>" in text

    @classmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        """Parse GLM tool calls; return (leading_content, tool_calls)."""
        param_types = build_param_types(tools)
        start = text.find("<tool_call>")
        if start == -1:
            return text.strip(), []
        content = text[:start]
        tool_calls: list[ToolCall] = []
        for m in _TOOLCALL_RE.finditer(text):
            body = m.group(1) if m.group(1) is not None else m.group(2)
            if not body:
                continue
            ak = body.find("<arg_key>")
            name = (body if ak == -1 else body[:ak]).strip()
            if not name:
                continue
            types = param_types.get(name, {})
            args: dict[str, Any] = {}
            for pm in _ARG_RE.finditer(body):
                k = pm.group(1).strip()
                if k:
                    args[k] = coerce_json_or_raw(pm.group(2), types.get(k))
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
        return content.strip(), tool_calls
