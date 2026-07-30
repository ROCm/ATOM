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
from .tool_parser import BufferedMarkerParser, ToolCall, unique_tool_call_id

MINIMAX_NS = "]<]minimax[>["

_INVOKE_RE = re.compile(
    r'<invoke\s+name="(.*?)"\s*>(.*?)</invoke>|<invoke\s+name="(.*?)"\s*>(.*)$',
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<([\w-]+)>(.*?)</\1>", re.DOTALL)


class MiniMaxParser(BufferedMarkerParser):
    NAME: ClassVar[str] = "minimax"
    START_MARKERS: ClassVar[tuple[str, ...]] = (MINIMAX_NS, "<tool_call>")
    # The ns_token starts with ']', so a trailing ']' may be a partial marker.
    HOLDBACK_CHARS: ClassVar[tuple[str, ...]] = ("<", "]")

    @classmethod
    def detect(cls, text: str) -> bool:
        """Detect the MiniMax-M3 ns_token tool-call format."""
        return MINIMAX_NS in text

    @classmethod
    def parse(cls, text: str, tools: list | None) -> tuple[str, list[ToolCall]]:
        """Parse MiniMax-M3 tool calls; return (leading_content, tool_calls)."""
        param_types = build_param_types(tools)
        clean = text.replace(MINIMAX_NS, "")
        tc = clean.find("<tool_call>")
        content = clean[:tc] if tc > 0 else ("" if tc == 0 else clean)
        tool_calls: list[ToolCall] = []
        for m in _INVOKE_RE.finditer(clean):
            name = m.group(1) if m.group(1) is not None else m.group(3)
            body = m.group(2) if m.group(2) is not None else (m.group(4) or "")
            if not name:
                continue
            name = name.strip()
            types = param_types.get(name, {})
            args: dict[str, Any] = {}
            for pm in _PARAM_RE.finditer(body):
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
        for mk in ("<tool_call>", "</tool_call>"):
            content = content.replace(mk, "")
        return content.strip(), tool_calls
