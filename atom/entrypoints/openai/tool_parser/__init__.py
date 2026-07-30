# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tool call parsing for models that emit tool calls in their text output.

Five on-the-wire formats are auto-detected and normalized into the OpenAI
``tool_calls`` structure:

==============================  ===============================================
Module                          Format
==============================  ===============================================
`kimi_tool_parser`              Kimi-K2 ``<|tool_call_begin|>`` special tokens
`qwen3_tool_parser`             Qwen3 (qwen3_coder / qwen3_xml) ``<function=`` XML
`deepseekv4_tool_parser`        DeepSeek-V4 ``<｜DSML｜invoke>`` markup
`glm_tool_parser`               GLM-4.5/4.6/5.x ``<tool_call>``/``<arg_key>``
`minimax_tool_parser`           MiniMax-M3 ``]<]minimax[>[``-prefixed tags
==============================  ===============================================

Formats other than Kimi carry no value types on the wire, so when the request's
``tools`` schema is supplied each parameter is coerced to its declared
JSON-Schema type; otherwise it is left as a string.

Two entry points, both format-agnostic:

- :func:`parse_tool_calls` — a complete output -> ``(content, [ToolCall])``
- :class:`ToolCallStreamParser` — chunks -> ``(event_type, data)`` tuples

OpenAI format::

    {"tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "NAME", "arguments": "ARGS_JSON"}}]}

To add a format: implement :class:`~.tool_parser.ToolCallParser` (or, if it
buffers from a start marker like most do,
:class:`~.tool_parser.BufferedMarkerParser`) in its own
``<model>_tool_parser.py``, then register it in :mod:`.registry` — in both
``_DETECT_ORDER`` and ``sniff_stream``, whose ordering constraints are
documented there.
"""

from .deepseekv4_tool_parser import DsmlParser
from .glm_tool_parser import GlmParser
from .kimi_tool_parser import KimiParser
from .minimax_tool_parser import MiniMaxParser
from .qwen3_tool_parser import QwenXmlParser
from .registry import parse_tool_calls
from .stream import ToolCallStreamParser
from .tool_parser import BufferedMarkerParser, ToolCall, ToolCallParser

__all__ = [
    "BufferedMarkerParser",
    "DsmlParser",
    "GlmParser",
    "KimiParser",
    "MiniMaxParser",
    "QwenXmlParser",
    "ToolCall",
    "ToolCallParser",
    "ToolCallStreamParser",
    "parse_tool_calls",
]
