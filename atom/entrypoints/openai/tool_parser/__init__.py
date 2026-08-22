# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tool call parsing for models that emit tool calls in their text output.

Six on-the-wire formats, normalized into the OpenAI ``tool_calls`` structure.
Which one a model uses is resolved once at startup from its chat template
(:func:`~.registry.resolve_tool_call_parser`) or set explicitly with
``--tool-call-parser``; it is not inferred from the output, because inferring
it means deciding from a prefix that may not carry the discriminator yet.

==============================  ===============================================
Module                          Format
==============================  ===============================================
`kimi_tool_parser`              Kimi-K2 ``<|tool_call_begin|>`` special tokens
`kimi_k3_tool_parser`           Kimi-K3 ``<|open|>call tool="`` channel format
`qwen3_tool_parser`             Qwen3 (qwen3_coder / qwen3_xml) ``<function=`` XML
`deepseekv4_tool_parser`        DeepSeek-V4 ``<｜DSML｜invoke>`` markup
`glm_tool_parser`               GLM-4.5/4.6/5.x ``<tool_call>``/``<arg_key>``
`minimax_tool_parser`           MiniMax-M3 ``]<]minimax[>[``-prefixed tags
==============================  ===============================================

The XML-ish formats (Qwen / DSML / GLM / MiniMax) carry no value types on the
wire, so when the request's ``tools`` schema is supplied each parameter is
coerced to its declared JSON-Schema type; otherwise it is left as a string.
Kimi-K2 arguments are already JSON; Kimi-K3 carries a per-argument
``type="..."`` on the wire, so neither needs the request schema.

Two entry points, both format-agnostic:

- :func:`parse_tool_calls` — a complete output -> ``(content, [ToolCall])``
- :class:`ToolCallStreamParser` — chunks -> ``(event_type, data)`` tuples

OpenAI format::

    {"tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "NAME", "arguments": "ARGS_JSON"}}]}

To add a format: implement :class:`~.tool_parser.ToolCallParser` in its own
``<model>_tool_parser.py``, declaring its ``START_MARKERS`` and a
``parse_region``, then add it to ``_DETECT_ORDER`` in :mod:`.registry`, whose
ordering constraints are documented there. That one entry is enough: startup
resolution, the streaming read-ahead, ``--tool-call-parser``\'s accepted names
and the property tests are all derived from it. A format writes no state
machine and no whole-output parser of its own -- :mod:`.stream` is the only
reader, and it is the only reader for both delivery modes.
"""

from .deepseekv4_tool_parser import DsmlParser
from .glm_tool_parser import GlmParser
from .kimi_k3_tool_parser import KimiK3Parser
from .kimi_tool_parser import KimiParser
from .minimax_tool_parser import MiniMaxParser
from .qwen3_tool_parser import QwenXmlParser
from .registry import parse_tool_calls
from .stream import (
    flatten_tool_events,
    read_whole_events,
    ToolCallStreamParser,
    read_whole,
)
from .tool_parser import RegionParse, ToolCall, ToolCallParser

__all__ = [
    "flatten_tool_events",
    "read_whole_events",
    "DsmlParser",
    "GlmParser",
    "KimiK3Parser",
    "KimiParser",
    "MiniMaxParser",
    "QwenXmlParser",
    "RegionParse",
    "ToolCall",
    "ToolCallParser",
    "ToolCallStreamParser",
    "parse_tool_calls",
    "read_whole",
]
